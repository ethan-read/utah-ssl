"""Session-held-out linear probes for CTC-aligned articulator targets."""

from __future__ import annotations

import csv
import json
import warnings
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning

from .analysis import ctc_forced_align


DEFAULT_TAXONOMY_PATH = (
    Path(__file__).resolve().parent / "design" / "articulatory_feature_taxonomy.csv"
)
DEFAULT_ARTICULATOR_TARGETS = ("lips", "tongue_front", "tongue_body")


def load_articulatory_taxonomy(
    path: str | Path = DEFAULT_TAXONOMY_PATH,
) -> dict[int, dict[str, str]]:
    """Load the canonical taxonomy keyed by phoneme ID."""

    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    taxonomy = {int(row["phoneme_id"]): row for row in rows}
    if len(rows) != 41 or len(taxonomy) != 41:
        raise ValueError("Articulatory taxonomy must contain 41 unique phoneme IDs.")
    return taxonomy


def load_representation_arrays(
    model_dir: str | Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load hidden states, logits, and example indices from an export."""

    root = Path(model_dir)
    shard_manifest = json.loads((root / "shards.json").read_text())
    metadata = json.loads((root / "metadata.json").read_text())
    if not isinstance(shard_manifest, list) or not shard_manifest:
        raise ValueError(f"Representation export has an empty shard manifest: {root}")
    hidden_parts: list[np.ndarray] = []
    logit_parts: list[np.ndarray] = []
    example_index_parts: list[np.ndarray] = []
    shard_names: set[str] = set()
    for shard in shard_manifest:
        shard_name = str(shard["shard"])
        if shard_name in shard_names:
            raise ValueError(f"Duplicate shard in representation manifest: {shard_name}")
        shard_names.add(shard_name)
        with np.load(root / "shards" / shard_name) as arrays:
            hidden_part = np.asarray(arrays["hidden"], dtype=np.float32)
            logit_part = np.asarray(arrays["logits"], dtype=np.float32)
            example_index_part = np.asarray(
                arrays["token_example_index"], dtype=np.int64
            )
        if hidden_part.ndim != 2 or logit_part.ndim != 2:
            raise ValueError(f"Shard {shard_name} hidden/logit arrays must be matrices.")
        if example_index_part.ndim != 1:
            raise ValueError(f"Shard {shard_name} example indices must be a vector.")
        if not (
            len(hidden_part) == len(logit_part) == len(example_index_part)
        ):
            raise ValueError(f"Shard {shard_name} arrays differ in row count.")
        expected_shape = (
            int(shard["token_count"]),
            int(shard["hidden_dim"]),
            int(shard["vocab_size"]),
        )
        observed_shape = (
            len(hidden_part),
            hidden_part.shape[1],
            logit_part.shape[1],
        )
        if observed_shape != expected_shape:
            raise ValueError(
                f"Shard {shard_name} shape does not match its manifest: "
                f"observed={observed_shape}, expected={expected_shape}"
            )
        hidden_parts.append(hidden_part)
        logit_parts.append(logit_part)
        example_index_parts.append(example_index_part)
    hidden = np.concatenate(hidden_parts, axis=0)
    logits = np.concatenate(logit_parts, axis=0)
    example_indices = np.concatenate(example_index_parts, axis=0)
    if len(hidden) != len(logits) or len(hidden) != len(example_indices):
        raise ValueError("Exported hidden, logit, and example-index rows differ.")
    if not np.isfinite(hidden).all() or not np.isfinite(logits).all():
        raise ValueError("Representation export contains nonfinite hidden states or logits.")
    expected_vocab_size = int(metadata["vocab"]["num_classes"])
    expected_summary = (
        int(metadata["token_count"]),
        int(metadata["hidden_dim"]),
        expected_vocab_size,
    )
    observed_summary = (len(hidden), hidden.shape[1], logits.shape[1])
    if observed_summary != expected_summary:
        raise ValueError(
            "Combined representation arrays do not match metadata: "
            f"observed={observed_summary}, expected={expected_summary}"
        )
    expected_example_indices = np.arange(int(metadata["example_count"]), dtype=np.int64)
    observed_example_indices = np.unique(example_indices)
    if not np.array_equal(observed_example_indices, expected_example_indices):
        raise ValueError(
            "Representation example-index coverage does not match metadata."
        )
    return hidden, logits, example_indices


def _parse_id_sequence(value: Any) -> list[int]:
    text = str(value).strip()
    return [] if not text else [int(token) for token in text.split()]


def _contiguous_example_rows(example_indices: np.ndarray) -> dict[int, slice]:
    values = np.asarray(example_indices, dtype=np.int64)
    if values.ndim != 1 or not len(values):
        raise ValueError("example_indices must be a non-empty vector")
    change_points = np.flatnonzero(np.diff(values) != 0) + 1
    starts = np.concatenate(([0], change_points))
    stops = np.concatenate((change_points, [len(values)]))
    result: dict[int, slice] = {}
    for start, stop in zip(starts, stops):
        example_index = int(values[start])
        if example_index in result:
            raise ValueError(f"Example {example_index} is not stored contiguously.")
        result[example_index] = slice(int(start), int(stop))
    return result


def build_aligned_consonant_events(
    *,
    hidden: np.ndarray,
    logits: np.ndarray,
    token_example_indices: np.ndarray,
    examples: pd.DataFrame,
    taxonomy: dict[int, dict[str, str]],
    blank_index: int,
    targets: Iterable[str] = DEFAULT_ARTICULATOR_TARGETS,
    excluded_symbols: Iterable[str] = ("HH",),
) -> tuple[np.ndarray, pd.DataFrame]:
    """Mean-pool hidden states over reference-constrained CTC phone spans."""

    hidden = np.asarray(hidden, dtype=np.float32)
    logits = np.asarray(logits, dtype=np.float32)
    token_example_indices = np.asarray(token_example_indices, dtype=np.int64)
    if hidden.ndim != 2 or logits.ndim != 2:
        raise ValueError("hidden and logits must be matrices")
    if len(hidden) != len(logits) or len(hidden) != len(token_example_indices):
        raise ValueError("hidden, logits, and token_example_indices must align")
    required_columns = {
        "example_export_index",
        "example_id",
        "session_id",
        "reference_ids",
    }
    missing = required_columns - set(examples.columns)
    if missing:
        raise ValueError(f"Example table is missing columns: {sorted(missing)}")

    resolved_targets = tuple(str(target) for target in targets)
    excluded = {str(symbol) for symbol in excluded_symbols}
    row_slices = _contiguous_example_rows(token_example_indices)
    event_features: list[np.ndarray] = []
    event_rows: list[dict[str, Any]] = []
    seen_examples: set[int] = set()

    for example in examples.itertuples(index=False):
        example_index = int(example.example_export_index)
        if example_index in seen_examples:
            raise ValueError(f"Duplicate example_export_index: {example_index}")
        seen_examples.add(example_index)
        row_slice = row_slices.get(example_index)
        if row_slice is None:
            raise ValueError(f"Export arrays are missing example {example_index}")
        reference = _parse_id_sequence(example.reference_ids)
        spans = ctc_forced_align(
            logits[row_slice],
            reference,
            blank_index=int(blank_index),
        )
        for label_index, (label_id, (onset, offset)) in enumerate(
            zip(reference, spans)
        ):
            taxonomy_row = taxonomy.get(int(label_id))
            if taxonomy_row is None:
                raise KeyError(f"Taxonomy is missing phoneme ID {label_id}")
            symbol = str(taxonomy_row["symbol"])
            if (
                taxonomy_row["segment_family"] != "consonant"
                or symbol in excluded
            ):
                continue
            span_hidden = hidden[row_slice][int(onset) : int(offset)]
            if not len(span_hidden):
                raise ValueError(
                    f"Aligned span is empty for {example.example_id}:{label_index}"
                )
            event_features.append(span_hidden.mean(axis=0, dtype=np.float64).astype(np.float32))
            articulators = set(str(taxonomy_row["primary_articulators"]).split("|"))
            event_row: dict[str, Any] = {
                "event_index": len(event_rows),
                "example_export_index": example_index,
                "example_id": str(example.example_id),
                "session_id": str(example.session_id),
                "label_index": int(label_index),
                "phoneme_id": int(label_id),
                "symbol": symbol,
                "aligned_onset_token": int(onset),
                "aligned_offset_token": int(offset),
                "aligned_span_tokens": int(offset) - int(onset),
            }
            for target in resolved_targets:
                event_row[target] = int(target in articulators)
            event_rows.append(event_row)

    if not event_features:
        raise ValueError("No eligible aligned consonant events were extracted.")
    features = np.stack(event_features).astype(np.float32, copy=False)
    event_frame = pd.DataFrame(event_rows)
    if len(features) != len(event_frame):
        raise AssertionError("Event features and metadata differ in length.")
    return features, event_frame


def _binary_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, float | int]:
    labels = np.asarray(labels, dtype=np.int64)
    predictions = np.asarray(predictions, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("Binary evaluation requires both classes.")
    return {
        "n_events": int(len(labels)),
        "positive_fraction": float(labels.mean()),
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "average_precision": float(average_precision_score(labels, probabilities)),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
    }


def fit_session_heldout_articulator_probes(
    *,
    features: np.ndarray,
    events: pd.DataFrame,
    train_session_ids: Iterable[str],
    test_session_ids: Iterable[str],
    targets: Iterable[str] = DEFAULT_ARTICULATOR_TARGETS,
    permutations: int = 1000,
    seed: int = 7,
    max_iterations: int = 2000,
    tolerance: float = 1e-4,
) -> dict[str, Any]:
    """Fit train-session-only logistic probes and evaluate future sessions."""

    features = np.asarray(features, dtype=np.float32)
    if features.ndim != 2 or len(features) != len(events):
        raise ValueError("features must align one-to-one with event rows")
    train_sessions = tuple(str(value) for value in train_session_ids)
    test_sessions = tuple(str(value) for value in test_session_ids)
    if not train_sessions or not test_sessions:
        raise ValueError("Both training and test session lists are required.")
    if set(train_sessions) & set(test_sessions):
        raise ValueError("Training and test sessions must be disjoint.")
    if int(max_iterations) <= 0 or float(tolerance) <= 0:
        raise ValueError("max_iterations and tolerance must be positive.")
    observed_sessions = set(events["session_id"].astype(str))
    requested_sessions = set(train_sessions) | set(test_sessions)
    if not requested_sessions.issubset(observed_sessions):
        missing = sorted(requested_sessions - observed_sessions)
        raise ValueError(f"Event table is missing requested sessions: {missing}")
    train_mask = events["session_id"].astype(str).isin(train_sessions).to_numpy()
    test_mask = events["session_id"].astype(str).isin(test_sessions).to_numpy()
    if np.any(train_mask & test_mask):
        raise AssertionError("Training and test event masks overlap.")

    pooled_rows: list[dict[str, Any]] = []
    session_rows: list[dict[str, Any]] = []
    null_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    models: dict[str, Pipeline] = {}
    rng = np.random.default_rng(int(seed))

    for target in tuple(str(value) for value in targets):
        if target not in events:
            raise KeyError(f"Event table is missing target column {target!r}")
        labels = events[target].to_numpy(dtype=np.int64)
        train_labels = labels[train_mask]
        test_labels = labels[test_mask]
        if set(np.unique(train_labels)) != {0, 1}:
            raise ValueError(f"Training target {target!r} does not contain both classes.")
        if set(np.unique(test_labels)) != {0, 1}:
            raise ValueError(f"Test target {target!r} does not contain both classes.")

        model = Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "probe",
                    LogisticRegression(
                        C=1.0,
                        class_weight="balanced",
                        max_iter=int(max_iterations),
                        random_state=int(seed),
                        solver="liblinear",
                        tol=float(tolerance),
                    ),
                ),
            ]
        )
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always", ConvergenceWarning)
            model.fit(features[train_mask], train_labels)
        convergence_warnings = [
            warning
            for warning in caught_warnings
            if issubclass(warning.category, ConvergenceWarning)
        ]
        classifier = model.named_steps["probe"]
        iterations = int(np.max(classifier.n_iter_))
        if convergence_warnings:
            raise RuntimeError(
                f"Probe {target!r} did not converge within {max_iterations} iterations."
            )
        models[target] = model
        probabilities = model.predict_proba(features[test_mask])[:, 1]
        predictions = (probabilities >= 0.5).astype(np.int64)
        pooled_metrics = _binary_metrics(test_labels, predictions, probabilities)
        majority_value = int(np.mean(train_labels) >= 0.5)
        majority_predictions = np.full_like(test_labels, majority_value)
        pooled_metrics.update(
            {
                "target": target,
                "train_events": int(train_mask.sum()),
                "train_positive_fraction": float(train_labels.mean()),
                "majority_class": majority_value,
                "majority_accuracy": float(
                    accuracy_score(test_labels, majority_predictions)
                ),
                "majority_balanced_accuracy": float(
                    balanced_accuracy_score(test_labels, majority_predictions)
                ),
                "converged": True,
                "iterations": iterations,
                "max_iterations": int(max_iterations),
                "tolerance": float(tolerance),
            }
        )

        test_events = events.loc[test_mask].reset_index(drop=True)
        for row_index, event in test_events.iterrows():
            prediction_rows.append(
                {
                    "event_index": int(event["event_index"]),
                    "example_id": str(event["example_id"]),
                    "session_id": str(event["session_id"]),
                    "symbol": str(event["symbol"]),
                    "target": target,
                    "label": int(test_labels[row_index]),
                    "probability": float(probabilities[row_index]),
                    "prediction": int(predictions[row_index]),
                }
            )

        session_arrays: list[np.ndarray] = []
        for session_id in test_sessions:
            session_mask = test_events["session_id"].astype(str).eq(session_id).to_numpy()
            if set(np.unique(test_labels[session_mask])) != {0, 1}:
                raise ValueError(
                    f"Test session {session_id!r} lacks both classes for {target!r}."
                )
            session_metrics = _binary_metrics(
                test_labels[session_mask],
                predictions[session_mask],
                probabilities[session_mask],
            )
            session_rows.append(
                {"target": target, "session_id": session_id, **session_metrics}
            )
            session_arrays.append(np.flatnonzero(session_mask))

        null_values = np.empty(int(permutations), dtype=np.float64)
        for permutation_index in range(int(permutations)):
            shuffled = test_labels.copy()
            for indices in session_arrays:
                shuffled[indices] = rng.permutation(shuffled[indices])
            null_values[permutation_index] = balanced_accuracy_score(
                shuffled,
                predictions,
            )
            null_rows.append(
                {
                    "target": target,
                    "permutation": permutation_index,
                    "balanced_accuracy": float(null_values[permutation_index]),
                }
            )
        observed = float(pooled_metrics["balanced_accuracy"])
        pooled_metrics.update(
            {
                "within_session_shuffle_mean_balanced_accuracy": float(
                    null_values.mean()
                ),
                "within_session_shuffle_q025": float(
                    np.quantile(null_values, 0.025)
                ),
                "within_session_shuffle_q975": float(
                    np.quantile(null_values, 0.975)
                ),
                "within_session_shuffle_p_value": float(
                    (1 + np.sum(null_values >= observed)) / (len(null_values) + 1)
                ),
            }
        )
        pooled_rows.append(pooled_metrics)

    return {
        "pooled_metrics": pd.DataFrame(pooled_rows),
        "session_metrics": pd.DataFrame(session_rows),
        "null_metrics": pd.DataFrame(null_rows),
        "predictions": pd.DataFrame(prediction_rows),
        "models": models,
        "train_event_count": int(train_mask.sum()),
        "test_event_count": int(test_mask.sum()),
    }


__all__ = [
    "DEFAULT_ARTICULATOR_TARGETS",
    "DEFAULT_TAXONOMY_PATH",
    "build_aligned_consonant_events",
    "fit_session_heldout_articulator_probes",
    "load_articulatory_taxonomy",
    "load_representation_arrays",
]

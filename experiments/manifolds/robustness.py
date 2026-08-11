"""Cross-session robustness tests for phoneme-aligned 20 ms trajectories."""

from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from .analysis import ctc_forced_align
from .io import load_representation_export

from .representation_export import category_for_symbol


@dataclass(frozen=True)
class RobustnessConfig:
    before_bins: int = 15
    after_bins: int = 15
    null_centers_per_event: int = 5
    null_exclusion_bins: int = 30
    min_train_events: int = 20
    min_test_events: int = 4
    max_train_events_per_phoneme: int = 100
    max_test_events_per_phoneme: int = 100
    primary_pca_components: int = 6
    sensitivity_pca_components: tuple[int, ...] = (24,)
    repeated_split_count: int = 25
    heldout_session_fraction: float = 0.25
    max_distance_pairs: int = 20_000
    permutation_repetitions: int = 200
    bootstrap_repetitions: int = 1_000
    overlap_atol: float = 1e-5
    seed: int = 7

    def validate(self) -> None:
        if self.before_bins < 0 or self.after_bins < 0:
            raise ValueError("before_bins and after_bins must be non-negative")
        if self.null_centers_per_event < 1:
            raise ValueError("null_centers_per_event must be positive")
        minimum_null_exclusion = self.before_bins + self.after_bins
        if self.null_exclusion_bins < minimum_null_exclusion:
            raise ValueError(
                "null_exclusion_bins must be at least before_bins + after_bins "
                "so real and null paths cannot overlap"
            )
        if self.min_train_events < 2 or self.min_test_events < 2:
            raise ValueError("minimum event counts must be at least 2")
        if (
            self.max_train_events_per_phoneme < self.min_train_events
            or self.max_test_events_per_phoneme < self.min_test_events
        ):
            raise ValueError("event caps must be at least their corresponding minimums")
        if not 0.0 < self.heldout_session_fraction < 1.0:
            raise ValueError("heldout_session_fraction must be between zero and one")
        if self.primary_pca_components < 2:
            raise ValueError("primary_pca_components must be at least 2")
        if any(component < 2 for component in self.sensitivity_pca_components):
            raise ValueError("sensitivity PCA component counts must be at least 2")
        if self.repeated_split_count < 1:
            raise ValueError("repeated_split_count must be positive")
        if self.max_distance_pairs < 1:
            raise ValueError("max_distance_pairs must be positive")
        if self.permutation_repetitions < 1:
            raise ValueError("permutation_repetitions must be positive")
        if self.bootstrap_repetitions < 1:
            raise ValueError("bootstrap_repetitions must be positive")
        if self.overlap_atol < 0:
            raise ValueError("overlap_atol must be non-negative")


@dataclass(frozen=True)
class TrajectoryEvent:
    event_index: int
    label_id: int
    example_index: int
    session_id: str
    real_center_bin: int
    null_center_bins: tuple[int, ...]
    alignment_confidence: float


@dataclass
class ReconstructedEventSet:
    sequences: dict[int, np.ndarray]
    events: tuple[TrajectoryEvent, ...]
    metadata: dict[str, Any]
    diagnostics: pd.DataFrame
    symbol_by_id: dict[int, str]
    before_bins: int
    after_bins: int

    def path(self, event: TrajectoryEvent, *, null_index: int | None = None) -> np.ndarray:
        center = (
            event.real_center_bin
            if null_index is None
            else event.null_center_bins[int(null_index)]
        )
        return self.sequences[event.example_index][
            center - self.before_bins : center + self.after_bins + 1
        ]


@dataclass
class FoldDetails:
    session_ids: tuple[str, ...]
    labels: np.ndarray
    symbols: tuple[str, ...]
    time_ms: np.ndarray
    real_paths: np.ndarray
    null_paths: np.ndarray
    distance_matrix: np.ndarray
    distance_label_ids: tuple[int, ...]


@dataclass
class RobustnessResult:
    config: RobustnessConfig
    representation: str
    metadata: dict[str, Any]
    diagnostics: pd.DataFrame
    loso: pd.DataFrame
    repeated_splits: pd.DataFrame
    sensitivity: pd.DataFrame
    summary: pd.DataFrame
    reference_details: FoldDetails | None


def reconstruct_overlapping_bins(
    token_windows: np.ndarray,
    *,
    patch_size: int,
    patch_stride: int,
    overlap_atol: float = 1e-5,
) -> tuple[np.ndarray, float]:
    """Reassemble pre-patching bins and verify duplicate overlap consistency."""
    token_windows = np.asarray(token_windows, dtype=np.float32)
    if token_windows.ndim != 2 or len(token_windows) == 0:
        raise ValueError("token_windows must have shape [nonzero tokens, features]")
    if token_windows.shape[1] % int(patch_size):
        raise ValueError("window feature dimension is not divisible by patch_size")
    feature_dim = token_windows.shape[1] // int(patch_size)
    patches = token_windows.reshape(len(token_windows), int(patch_size), feature_dim)
    output_length = (len(patches) - 1) * int(patch_stride) + int(patch_size)
    total = np.zeros((output_length, feature_dim), dtype=np.float32)
    count = np.zeros((output_length, 1), dtype=np.float32)
    max_overlap_error = 0.0
    for token_index, patch in enumerate(patches):
        start = token_index * int(patch_stride)
        stop = start + int(patch_size)
        existing = count[start:stop, 0] > 0
        if existing.any():
            existing_mean = total[start:stop][existing] / count[start:stop][existing]
            error = float(np.max(np.abs(existing_mean - patch[existing])))
            max_overlap_error = max(max_overlap_error, error)
            if error > float(overlap_atol):
                raise ValueError(
                    f"overlapping saved windows disagree: max error {error:.3g} "
                    f"> tolerance {float(overlap_atol):.3g}"
                )
        total[start:stop] += patch
        count[start:stop] += 1
    if np.any(count == 0):
        raise ValueError("patch schedule left uncovered bins inside the reconstructed span")
    return total / count, max_overlap_error


def _parse_reference_ids(value: object) -> list[int]:
    if pd.isna(value) or not str(value).strip():
        return []
    return [int(token) for token in str(value).split()]


def _symbol_map(metadata: dict[str, Any]) -> dict[int, str]:
    vocab = dict(metadata.get("vocab", {}))
    if isinstance(vocab.get("id_to_symbol"), dict):
        return {int(key): str(value) for key, value in vocab["id_to_symbol"].items()}
    return {
        index: str(value)
        for index, value in enumerate(vocab.get("index_to_symbol", []))
    }


def _log_softmax(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))


def build_reconstructed_event_set(
    model_dir: str | Path,
    *,
    representation: str,
    config: RobustnessConfig,
) -> ReconstructedEventSet:
    """Load an export, reconstruct its 20 ms sequences, and create matched events."""
    config.validate()
    if representation not in {"input_windows", "adapted_input_windows"}:
        raise ValueError("representation must be input_windows or adapted_input_windows")
    payload = load_representation_export(model_dir, representation=representation)
    metadata = payload["metadata"]
    patch_size = int(metadata["patch_size_bins"])
    patch_stride = int(metadata["patch_stride_bins"])
    bin_size_ms = int(metadata.get("bin_size_ms", 20))
    blank_index = int(metadata["vocab"]["blank_index"])
    example_table = payload["examples"].set_index("example_export_index")
    sequences: dict[int, np.ndarray] = {}
    events: list[TrajectoryEvent] = []
    diagnostics: list[dict[str, Any]] = []
    rng = np.random.default_rng(config.seed)

    for example_index, example in example_table.iterrows():
        example_index = int(example_index)
        rows = np.flatnonzero(payload["example_indices"] == example_index)
        reference = _parse_reference_ids(example.reference_ids)
        diagnostic = {
            "example_index": example_index,
            "session_id": str(example.session_id),
            "status": "included",
            "reference_count": len(reference),
            "event_count": 0,
            "boundary_excluded_count": 0,
            "null_excluded_count": 0,
            "reconstructed_bins": 0,
            "unavailable_tail_bins": 0,
            "max_overlap_error": np.nan,
        }
        if not len(rows) or not reference:
            diagnostic["status"] = "missing_tokens_or_reference"
            diagnostics.append(diagnostic)
            continue
        try:
            spans = ctc_forced_align(
                payload["logits"][rows],
                reference,
                blank_index=blank_index,
            )
        except (ValueError, IndexError) as exc:
            diagnostic["status"] = f"alignment_failed:{type(exc).__name__}"
            diagnostics.append(diagnostic)
            continue
        # Reconstruction inconsistencies indicate a broken export contract and
        # are fatal; silently dropping them would bias the session analysis.
        raw_bins, overlap_error = reconstruct_overlapping_bins(
            payload["values"][rows],
            patch_size=patch_size,
            patch_stride=patch_stride,
            overlap_atol=config.overlap_atol,
        )
        sequences[example_index] = raw_bins
        diagnostic["reconstructed_bins"] = len(raw_bins)
        diagnostic["max_overlap_error"] = overlap_error
        input_length = int(getattr(example, "input_length_bins", len(raw_bins)))
        diagnostic["unavailable_tail_bins"] = max(0, input_length - len(raw_bins))
        low = int(config.before_bins)
        high = len(raw_bins) - int(config.after_bins)
        if high <= low:
            diagnostic["status"] = "sequence_too_short"
            diagnostics.append(diagnostic)
            sequences.pop(example_index, None)
            continue
        log_probs = _log_softmax(payload["logits"][rows])
        for label_id, (onset, offset) in zip(reference, spans):
            token_center = (int(onset) + int(offset) - 1) // 2
            raw_center = token_center * patch_stride + patch_size // 2
            if raw_center < low or raw_center >= high:
                diagnostic["boundary_excluded_count"] += 1
                continue
            candidates = np.arange(low, high, dtype=np.int64)
            candidates = candidates[
                np.abs(candidates - raw_center) > int(config.null_exclusion_bins)
            ]
            if not len(candidates):
                diagnostic["null_excluded_count"] += 1
                continue
            replace = len(candidates) < int(config.null_centers_per_event)
            null_centers = tuple(
                int(value)
                for value in rng.choice(
                    candidates,
                    size=int(config.null_centers_per_event),
                    replace=replace,
                )
            )
            label_frames = np.arange(int(onset), int(offset), dtype=np.int64)
            confidence = float(np.exp(log_probs[label_frames, int(label_id)].mean()))
            events.append(
                TrajectoryEvent(
                    event_index=len(events),
                    label_id=int(label_id),
                    example_index=example_index,
                    session_id=str(example.session_id),
                    real_center_bin=int(raw_center),
                    null_center_bins=null_centers,
                    alignment_confidence=confidence,
                )
            )
            diagnostic["event_count"] += 1
        if diagnostic["event_count"] == 0:
            diagnostic["status"] = "no_usable_events"
            sequences.pop(example_index, None)
        diagnostics.append(diagnostic)

    metadata = dict(metadata)
    metadata["reconstructed_bin_size_ms"] = bin_size_ms
    return ReconstructedEventSet(
        sequences=sequences,
        events=tuple(events),
        metadata=metadata,
        diagnostics=pd.DataFrame(diagnostics),
        symbol_by_id=_symbol_map(metadata),
        before_bins=int(config.before_bins),
        after_bins=int(config.after_bins),
    )


def generate_unique_session_splits(
    sessions: Sequence[str],
    *,
    count: int,
    heldout_fraction: float,
    seed: int,
) -> tuple[tuple[str, ...], ...]:
    """Generate deterministic unique held-out-session combinations."""
    sessions = tuple(sorted(str(session) for session in sessions))
    if len(sessions) < 2:
        raise ValueError("at least two sessions are required")
    heldout_count = max(1, min(len(sessions) - 1, math.ceil(len(sessions) * heldout_fraction)))
    maximum = math.comb(len(sessions), heldout_count)
    target = min(int(count), maximum)
    rng = np.random.default_rng(seed)
    splits: set[tuple[str, ...]] = set()
    attempts = 0
    while len(splits) < target and attempts < max(1_000, target * 100):
        selected = tuple(sorted(rng.choice(sessions, heldout_count, replace=False).tolist()))
        splits.add(selected)
        attempts += 1
    if len(splits) < target:
        raise RuntimeError("could not generate the requested number of unique session splits")
    return tuple(sorted(splits))


def _balanced_sample(
    events: Sequence[TrajectoryEvent],
    label_ids: Sequence[int],
    *,
    maximum: int,
    rng: np.random.Generator,
) -> list[TrajectoryEvent]:
    sampled: list[TrajectoryEvent] = []
    for label_id in label_ids:
        candidates = [event for event in events if event.label_id == int(label_id)]
        if len(candidates) > int(maximum):
            indices = np.sort(rng.choice(len(candidates), int(maximum), replace=False))
            candidates = [candidates[int(index)] for index in indices]
        sampled.extend(candidates)
    return sampled


def _equal_phoneme_grand_mean(
    event_set: ReconstructedEventSet,
    events: Sequence[TrajectoryEvent],
) -> np.ndarray:
    label_means = []
    for label_id in sorted({event.label_id for event in events}):
        label_paths = [
            event_set.path(event)
            for event in events
            if event.label_id == label_id
        ]
        label_means.append(np.mean(label_paths, axis=0))
    return np.mean(label_means, axis=0)


def _fit_projector(
    event_set: ReconstructedEventSet,
    train_events: Sequence[TrajectoryEvent],
    *,
    components: int,
    seed: int,
) -> tuple[np.ndarray, StandardScaler, PCA]:
    grand_mean = _equal_phoneme_grand_mean(event_set, train_events)
    residual_points = np.concatenate(
        [event_set.path(event) - grand_mean for event in train_events],
        axis=0,
    )
    scaler = StandardScaler()
    scaled = scaler.fit_transform(residual_points)
    component_count = min(int(components), scaled.shape[0], scaled.shape[1])
    pca = PCA(
        n_components=component_count,
        svd_solver="randomized",
        random_state=int(seed),
    ).fit(scaled)
    return grand_mean, scaler, pca


def _project_paths(
    event_set: ReconstructedEventSet,
    events: Sequence[TrajectoryEvent],
    *,
    grand_mean: np.ndarray,
    scaler: StandardScaler,
    pca: PCA,
    null_index: int | None,
) -> np.ndarray:
    paths = []
    for event in events:
        path = event_set.path(event, null_index=null_index)
        projected = pca.transform(scaler.transform(path - grand_mean))
        projected -= projected.mean(axis=0, keepdims=True)
        paths.append(projected.astype(np.float32, copy=False))
    return np.stack(paths)


def _sample_pair_indices(
    event_count: int,
    *,
    max_pairs: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Choose one event-pair sample that can be reused across real/null paths."""
    total_pairs = int(event_count) * (int(event_count) - 1) // 2
    if total_pairs <= int(max_pairs):
        return np.triu_indices(int(event_count), k=1)
    first = rng.integers(0, int(event_count), size=int(max_pairs))
    second = rng.integers(0, int(event_count) - 1, size=int(max_pairs))
    second += second >= first
    return first, second


def _separation_for_pairs(
    paths: np.ndarray,
    labels: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
) -> float:
    """Compute between-minus-within distance for a fixed event-pair sample."""
    flat = paths.reshape(len(paths), -1)
    if len(first) != len(second):
        raise ValueError("pair index arrays must have equal length")
    if len(first) == 0:
        return float("nan")
    if (
        np.any(first < 0)
        or np.any(second < 0)
        or np.any(first >= len(flat))
        or np.any(second >= len(flat))
    ):
        raise IndexError("pair indices are outside the path array")
    distances = np.sqrt(np.mean((flat[first] - flat[second]) ** 2, axis=1))
    same = labels[first] == labels[second]
    if not same.any() or same.all():
        return float("nan")
    return float(distances[~same].mean() - distances[same].mean())


def _nearest_centroid_predictions(
    train_paths: np.ndarray,
    train_labels: np.ndarray,
    test_paths: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, tuple[Any, ...]]:
    label_order = tuple(sorted(set(train_labels.tolist())))
    train_flat = train_paths.reshape(len(train_paths), -1)
    test_flat = test_paths.reshape(len(test_paths), -1)
    centroids = np.stack(
        [train_flat[train_labels == label].mean(axis=0) for label in label_order]
    )
    squared_distances = (
        (test_flat**2).sum(axis=1, keepdims=True)
        + (centroids**2).sum(axis=1)[None, :]
        - 2.0 * test_flat @ centroids.T
    ) / train_flat.shape[1]
    predictions = np.asarray(label_order, dtype=object)[squared_distances.argmin(axis=1)]
    return predictions, squared_distances, label_order


def _balanced_accuracy(labels: np.ndarray, predictions: np.ndarray) -> float:
    recalls = [
        float(np.mean(predictions[labels == label] == label))
        for label in sorted(set(labels.tolist()))
        if np.any(labels == label)
    ]
    return float(np.mean(recalls)) if recalls else float("nan")


def _permuted_balanced_accuracy(
    labels: np.ndarray,
    predictions: np.ndarray,
    session_ids: np.ndarray,
    *,
    repetitions: int,
    rng: np.random.Generator,
) -> np.ndarray:
    values = np.empty(int(repetitions), dtype=np.float64)
    unique_sessions = tuple(sorted(set(session_ids.tolist())))
    for repetition in range(int(repetitions)):
        shuffled = labels.copy()
        for session_id in unique_sessions:
            indices = np.flatnonzero(session_ids == session_id)
            shuffled[indices] = rng.permutation(shuffled[indices])
        values[repetition] = _balanced_accuracy(shuffled, predictions)
    return values


def _category_labels(
    labels: Iterable[int],
    symbol_by_id: dict[int, str],
) -> np.ndarray:
    return np.asarray(
        [category_for_symbol(symbol_by_id.get(int(label), str(label))) for label in labels],
        dtype=object,
    )


def evaluate_session_fold(
    event_set: ReconstructedEventSet,
    *,
    test_sessions: Sequence[str],
    config: RobustnessConfig,
    components: int,
    seed: int,
    confidence_quantile: float | None = None,
    return_details: bool = False,
) -> tuple[dict[str, Any] | None, FoldDetails | None]:
    """Fit on non-test sessions and evaluate real and matched-null path shape."""
    config.validate()
    rng = np.random.default_rng(seed)
    test_session_set = {str(session) for session in test_sessions}
    train_pool = [
        event for event in event_set.events if event.session_id not in test_session_set
    ]
    test_pool = [
        event for event in event_set.events if event.session_id in test_session_set
    ]
    confidence_threshold = float("nan")
    if confidence_quantile is not None:
        if not 0.0 <= float(confidence_quantile) < 1.0:
            raise ValueError("confidence_quantile must be in [0, 1)")
        if not train_pool:
            return None, None
        confidence_threshold = float(
            np.quantile(
                [event.alignment_confidence for event in train_pool],
                float(confidence_quantile),
            )
        )
        train_pool = [
            event
            for event in train_pool
            if event.alignment_confidence >= confidence_threshold
        ]
        test_pool = [
            event
            for event in test_pool
            if event.alignment_confidence >= confidence_threshold
        ]
    train_counts = Counter(event.label_id for event in train_pool)
    test_counts = Counter(event.label_id for event in test_pool)
    retained_ids = tuple(
        sorted(
            label_id
            for label_id, count in train_counts.items()
            if count >= int(config.min_train_events)
            and test_counts[label_id] >= int(config.min_test_events)
        )
    )
    if len(retained_ids) < 2:
        return None, None
    train_events = _balanced_sample(
        train_pool,
        retained_ids,
        maximum=config.max_train_events_per_phoneme,
        rng=rng,
    )
    test_events = _balanced_sample(
        test_pool,
        retained_ids,
        maximum=config.max_test_events_per_phoneme,
        rng=rng,
    )
    grand_mean, scaler, pca = _fit_projector(
        event_set,
        train_events,
        components=components,
        seed=seed,
    )
    train_paths = _project_paths(
        event_set,
        train_events,
        grand_mean=grand_mean,
        scaler=scaler,
        pca=pca,
        null_index=None,
    )
    real_paths = _project_paths(
        event_set,
        test_events,
        grand_mean=grand_mean,
        scaler=scaler,
        pca=pca,
        null_index=None,
    )
    null_path_sets = [
        _project_paths(
            event_set,
            test_events,
            grand_mean=grand_mean,
            scaler=scaler,
            pca=pca,
            null_index=null_index,
        )
        for null_index in range(int(config.null_centers_per_event))
    ]
    train_labels = np.asarray([event.label_id for event in train_events])
    test_labels = np.asarray([event.label_id for event in test_events])
    test_session_ids = np.asarray([event.session_id for event in test_events], dtype=object)
    train_categories = _category_labels(train_labels, event_set.symbol_by_id)
    test_categories = _category_labels(test_labels, event_set.symbol_by_id)

    real_predictions, real_distances, distance_label_order = _nearest_centroid_predictions(
        train_paths,
        train_labels,
        real_paths,
    )
    real_category_predictions, _, _ = _nearest_centroid_predictions(
        train_paths,
        train_categories,
        real_paths,
    )
    null_predictions = [
        _nearest_centroid_predictions(train_paths, train_labels, paths)[0]
        for paths in null_path_sets
    ]
    null_category_predictions = [
        _nearest_centroid_predictions(train_paths, train_categories, paths)[0]
        for paths in null_path_sets
    ]
    real_phoneme_accuracy = _balanced_accuracy(test_labels, real_predictions)
    real_category_accuracy = _balanced_accuracy(
        test_categories,
        real_category_predictions,
    )
    null_phoneme_accuracies = np.asarray(
        [_balanced_accuracy(test_labels, prediction) for prediction in null_predictions]
    )
    null_category_accuracies = np.asarray(
        [
            _balanced_accuracy(test_categories, prediction)
            for prediction in null_category_predictions
        ]
    )
    pair_first, pair_second = _sample_pair_indices(
        len(test_events),
        max_pairs=config.max_distance_pairs,
        rng=rng,
    )
    real_separation = _separation_for_pairs(
        real_paths,
        test_labels,
        pair_first,
        pair_second,
    )
    null_separations = np.asarray(
        [
            _separation_for_pairs(
                paths,
                test_labels,
                pair_first,
                pair_second,
            )
            for paths in null_path_sets
        ]
    )
    phoneme_permutations = _permuted_balanced_accuracy(
        test_labels,
        real_predictions,
        test_session_ids,
        repetitions=config.permutation_repetitions,
        rng=rng,
    )
    category_permutations = _permuted_balanced_accuracy(
        test_categories,
        real_category_predictions,
        test_session_ids,
        repetitions=config.permutation_repetitions,
        rng=rng,
    )
    phoneme_null_mean = float(phoneme_permutations.mean())
    category_null_mean = float(category_permutations.mean())
    row = {
        "test_sessions": "|".join(sorted(test_session_set)),
        "test_session_count": len(test_session_set),
        "events": len(test_events),
        "phonemes": len(retained_ids),
        "components": int(pca.n_components_),
        "confidence_subset": (
            "all"
            if confidence_quantile is None
            else f"top_{int(round((1.0 - confidence_quantile) * 100))}pct"
        ),
        "confidence_threshold": confidence_threshold,
        "real_separation": real_separation,
        "null_separation": float(null_separations.mean()),
        "null_separation_sd": float(null_separations.std(ddof=0)),
        "real_minus_null_separation": real_separation - float(null_separations.mean()),
        "real_phoneme_balanced_accuracy": real_phoneme_accuracy,
        "null_phoneme_balanced_accuracy": float(null_phoneme_accuracies.mean()),
        "phoneme_permutation_mean": phoneme_null_mean,
        "phoneme_chance_adjusted_accuracy": (
            (real_phoneme_accuracy - phoneme_null_mean) / (1.0 - phoneme_null_mean)
            if phoneme_null_mean < 1.0
            else float("nan")
        ),
        "real_category_balanced_accuracy": real_category_accuracy,
        "null_category_balanced_accuracy": float(null_category_accuracies.mean()),
        "category_permutation_mean": category_null_mean,
        "category_chance_adjusted_accuracy": (
            (real_category_accuracy - category_null_mean) / (1.0 - category_null_mean)
            if category_null_mean < 1.0
            else float("nan")
        ),
        "median_alignment_confidence": float(
            np.median([event.alignment_confidence for event in test_events])
        ),
    }
    details = None
    if return_details:
        labels_in_distance_order = tuple(int(label) for label in distance_label_order)
        distance_sums = np.zeros((len(retained_ids), len(labels_in_distance_order)))
        distance_counts = np.zeros(len(retained_ids), dtype=np.int64)
        retained_to_row = {label: index for index, label in enumerate(retained_ids)}
        for event_label, distances in zip(test_labels, real_distances):
            row_index = retained_to_row[int(event_label)]
            distance_sums[row_index] += distances
            distance_counts[row_index] += 1
        distance_matrix = distance_sums / np.maximum(distance_counts[:, None], 1)
        details = FoldDetails(
            session_ids=tuple(sorted(test_session_set)),
            labels=test_labels,
            symbols=tuple(event_set.symbol_by_id.get(label, str(label)) for label in test_labels),
            time_ms=(
                np.arange(-config.before_bins, config.after_bins + 1)
                * int(event_set.metadata.get("reconstructed_bin_size_ms", 20))
            ),
            real_paths=real_paths,
            null_paths=np.mean(null_path_sets, axis=0),
            distance_matrix=distance_matrix,
            distance_label_ids=retained_ids,
        )
    return row, details


def _exact_sign_test(values: np.ndarray) -> float:
    values = np.asarray(values)
    positive = int(np.sum(values > 0))
    negative = int(np.sum(values < 0))
    count = positive + negative
    if count == 0:
        return 1.0
    tail = min(positive, negative)
    probability = sum(math.comb(count, index) for index in range(tail + 1)) / (2**count)
    return float(min(1.0, 2.0 * probability))


def _bootstrap_median_interval(
    values: np.ndarray,
    *,
    repetitions: int,
    seed: int,
) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    medians = np.asarray(
        [
            np.median(rng.choice(values, size=len(values), replace=True))
            for _ in range(int(repetitions))
        ]
    )
    return float(np.quantile(medians, 0.025)), float(np.quantile(medians, 0.975))


def summarize_loso(
    loso: pd.DataFrame,
    *,
    bootstrap_repetitions: int,
    seed: int,
) -> pd.DataFrame:
    rows = []
    metrics = {
        "separation": "real_minus_null_separation",
        "phoneme_balanced_accuracy": None,
        "category_balanced_accuracy": None,
    }
    for label, column in metrics.items():
        if column is None:
            real_column = f"real_{label}"
            null_column = f"null_{label}"
            values = (loso[real_column] - loso[null_column]).to_numpy()
        else:
            values = loso[column].to_numpy()
        low, high = _bootstrap_median_interval(
            values,
            repetitions=bootstrap_repetitions,
            seed=seed + len(rows),
        )
        rows.append(
            {
                "metric": label,
                "session_count": len(values),
                "median_real_minus_null": float(np.median(values)),
                "bootstrap_ci_low": low,
                "bootstrap_ci_high": high,
                "fraction_positive": float(np.mean(values > 0)),
                "exact_sign_p": _exact_sign_test(values),
            }
        )
    return pd.DataFrame(rows)


def run_robustness_analysis(
    model_dir: str | Path,
    *,
    representation: str,
    config: RobustnessConfig | None = None,
    progress: Callable[[str], None] | None = None,
) -> RobustnessResult:
    """Run LOSO, unique repeated splits, and PCA sensitivity analyses."""
    config = config or RobustnessConfig()
    event_set = build_reconstructed_event_set(
        model_dir,
        representation=representation,
        config=config,
    )
    if progress is not None:
        progress(
            f"{representation}: reconstructed {len(event_set.events):,} events "
            f"from {len(event_set.sequences):,} examples"
        )
    sessions = tuple(sorted({event.session_id for event in event_set.events}))
    if len(sessions) < 2:
        raise RuntimeError("fewer than two sessions have usable events")
    loso_rows = []
    reference_details = None
    for index, session in enumerate(sessions):
        row, details = evaluate_session_fold(
            event_set,
            test_sessions=(session,),
            config=config,
            components=config.primary_pca_components,
            seed=config.seed + 10_000 + index,
            return_details=reference_details is None,
        )
        if row is not None:
            row["heldout_session"] = session
            row["fold"] = index
            loso_rows.append(row)
            if reference_details is None:
                reference_details = details
        if progress is not None:
            progress(f"{representation}: LOSO {index + 1}/{len(sessions)}")
    loso = pd.DataFrame(loso_rows)
    if loso.empty:
        raise RuntimeError("no leave-one-session-out folds met the event thresholds")

    repeated_rows = []
    splits = generate_unique_session_splits(
        sessions,
        count=config.repeated_split_count,
        heldout_fraction=config.heldout_session_fraction,
        seed=config.seed,
    )
    for index, split in enumerate(splits):
        row, _ = evaluate_session_fold(
            event_set,
            test_sessions=split,
            config=config,
            components=config.primary_pca_components,
            seed=config.seed + index,
        )
        if row is not None:
            row["split"] = index
            repeated_rows.append(row)
        if progress is not None:
            progress(f"{representation}: repeated split {index + 1}/{len(splits)}")
    repeated_splits = pd.DataFrame(repeated_rows)

    sensitivity_rows = []
    for component_count in config.sensitivity_pca_components:
        for index, session in enumerate(sessions):
            row, _ = evaluate_session_fold(
                event_set,
                test_sessions=(session,),
                config=config,
                components=component_count,
                seed=config.seed + 20_000 + index,
            )
            if row is not None:
                row["heldout_session"] = session
                sensitivity_rows.append(row)
            if progress is not None:
                progress(
                    f"{representation}: {component_count}-PC sensitivity "
                    f"{index + 1}/{len(sessions)}"
                )
    for index, session in enumerate(sessions):
        row, _ = evaluate_session_fold(
            event_set,
            test_sessions=(session,),
            config=config,
            components=config.primary_pca_components,
            seed=config.seed + 30_000 + index,
            confidence_quantile=0.25,
        )
        if row is not None:
            row["heldout_session"] = session
            sensitivity_rows.append(row)
        if progress is not None:
            progress(
                f"{representation}: high-confidence sensitivity "
                f"{index + 1}/{len(sessions)}"
            )
    sensitivity = pd.DataFrame(sensitivity_rows)
    summary = summarize_loso(
        loso,
        bootstrap_repetitions=config.bootstrap_repetitions,
        seed=config.seed,
    )
    summary.insert(0, "representation", representation)
    return RobustnessResult(
        config=config,
        representation=representation,
        metadata=event_set.metadata,
        diagnostics=event_set.diagnostics,
        loso=loso,
        repeated_splits=repeated_splits,
        sensitivity=sensitivity,
        summary=summary,
        reference_details=reference_details,
    )


def save_robustness_result(
    result: RobustnessResult,
    output_dir: str | Path,
) -> dict[str, str]:
    """Write result tables and compact diagnostic figures."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "diagnostics": output_dir / "reconstruction_diagnostics.csv",
        "loso": output_dir / "leave_one_session_out.csv",
        "repeated": output_dir / "repeated_session_splits.csv",
        "sensitivity": output_dir / "pca_sensitivity.csv",
        "summary": output_dir / "robustness_summary.csv",
        "metadata": output_dir / "robustness_metadata.json",
    }
    result.diagnostics.to_csv(paths["diagnostics"], index=False)
    result.loso.to_csv(paths["loso"], index=False)
    result.repeated_splits.to_csv(paths["repeated"], index=False)
    result.sensitivity.to_csv(paths["sensitivity"], index=False)
    result.summary.to_csv(paths["summary"], index=False)
    paths["metadata"].write_text(
        json.dumps(
            {
                "representation": result.representation,
                "config": asdict(result.config),
                "model_metadata": result.metadata,
            },
            indent=2,
            default=str,
        )
    )

    metric_pairs = (
        ("separation", "real_separation", "null_separation"),
        (
            "phoneme balanced accuracy",
            "real_phoneme_balanced_accuracy",
            "null_phoneme_balanced_accuracy",
        ),
        (
            "category balanced accuracy",
            "real_category_balanced_accuracy",
            "null_category_balanced_accuracy",
        ),
    )
    for name, frame in (
        ("loso_real_vs_null.png", result.loso),
        ("repeated_split_real_vs_null.png", result.repeated_splits),
    ):
        if frame.empty:
            continue
        fig, axes = plt.subplots(1, 3, figsize=(13, 4))
        for ax, (title, real_column, null_column) in zip(axes, metric_pairs):
            for _, row in frame.iterrows():
                ax.plot(
                    [0, 1],
                    [row[real_column], row[null_column]],
                    color="0.75",
                    linewidth=0.7,
                    alpha=0.6,
                )
            ax.scatter(
                np.zeros(len(frame)),
                frame[real_column],
                s=14,
                color="tab:blue",
                label="real",
            )
            ax.scatter(
                np.ones(len(frame)),
                frame[null_column],
                s=14,
                color="tab:orange",
                label="matched null",
            )
            ax.set_xticks([0, 1], ["real", "null"])
            ax.set_title(title)
        fig.suptitle(f"{result.representation}: {name.replace('_', ' ').replace('.png', '')}")
        fig.tight_layout()
        figure_path = output_dir / name
        fig.savefig(figure_path, dpi=180)
        plt.close(fig)
        paths[name] = figure_path

    details = result.reference_details
    if details is not None:
        top_labels = [
            label
            for label, _ in Counter(details.labels.tolist()).most_common(6)
        ]
        fig, axes = plt.subplots(2, len(top_labels), figsize=(3.6 * len(top_labels), 6))
        for row, (kind, paths_array) in enumerate(
            (("real", details.real_paths), ("matched null", details.null_paths))
        ):
            for ax, label_id in zip(axes[row], top_labels):
                selected = paths_array[details.labels == label_id]
                mean = selected.mean(axis=0)
                sem = selected.std(axis=0, ddof=0) / np.sqrt(max(1, len(selected)))
                ax.plot(details.time_ms, mean[:, 0], color="tab:blue")
                ax.fill_between(
                    details.time_ms,
                    mean[:, 0] - 1.96 * sem[:, 0],
                    mean[:, 0] + 1.96 * sem[:, 0],
                    color="tab:blue",
                    alpha=0.2,
                )
                ax.axvline(0, color="tab:red", linewidth=0.8)
                ax.set_title(
                    f"{kind}: {result.metadata.get('vocab', {}).get('id_to_symbol', {}).get(str(label_id), label_id)}"
                )
                ax.set_xlabel("time from aligned center (ms)")
                ax.set_ylabel("PC1")
        heldout_label = ", ".join(details.session_ids)
        fig.suptitle(
            f"{result.representation}: representative LOSO fold "
            f"(held out: {heldout_label})"
        )
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        timecourse_path = output_dir / "representative_fold_heldout_timecourses.png"
        fig.savefig(timecourse_path, dpi=180)
        plt.close(fig)
        paths["timecourses"] = timecourse_path

        fig, ax = plt.subplots(figsize=(8, 7))
        image = ax.imshow(details.distance_matrix, aspect="auto", cmap="viridis")
        symbols = [
            result.metadata.get("vocab", {}).get("id_to_symbol", {}).get(str(label), str(label))
            for label in details.distance_label_ids
        ]
        ax.set_xticks(np.arange(len(symbols)), symbols, rotation=90, fontsize=7)
        ax.set_yticks(np.arange(len(symbols)), symbols, fontsize=7)
        ax.set_xlabel("training phoneme template")
        ax.set_ylabel("held-out true phoneme")
        ax.set_title(f"Representative LOSO fold — held out: {heldout_label}")
        fig.colorbar(image, ax=ax, label="mean squared trajectory distance")
        fig.tight_layout()
        matrix_path = output_dir / "representative_fold_heldout_distance_matrix.png"
        fig.savefig(matrix_path, dpi=180)
        plt.close(fig)
        paths["distance_matrix"] = matrix_path
    return {key: str(value) for key, value in paths.items()}


__all__ = [
    "FoldDetails",
    "ReconstructedEventSet",
    "RobustnessConfig",
    "RobustnessResult",
    "TrajectoryEvent",
    "build_reconstructed_event_set",
    "evaluate_session_fold",
    "generate_unique_session_splits",
    "reconstruct_overlapping_bins",
    "run_robustness_analysis",
    "save_robustness_result",
    "summarize_loso",
]

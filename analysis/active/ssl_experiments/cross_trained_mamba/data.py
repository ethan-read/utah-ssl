"""Data helpers for cross-trained area-6v Mamba runs."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch

try:
    from masked_ssl.probe import (
        AREA6V_FEATURE_DIM,
        CanonicalProbeManifestRow,
        CanonicalSequenceDataset,
        canonical_rows_padded_time_percentile,
        collate_sequence_batch,
        compute_feature_stats,
    )
    from ssl_core.ctc import LengthAwareBatchSampler
except ModuleNotFoundError:  # pragma: no cover - repo-root unittest fallback
    from analysis.active.ssl_experiments.masked_ssl.probe import (
        AREA6V_FEATURE_DIM,
        CanonicalProbeManifestRow,
        CanonicalSequenceDataset,
        canonical_rows_padded_time_percentile,
        collate_sequence_batch,
        compute_feature_stats,
    )
    from analysis.active.ssl_experiments.ssl_core.ctc import LengthAwareBatchSampler


DATASET_SPLITS: dict[str, tuple[str, str]] = {
    "brain2text24": ("competition_train", "competition_test"),
    "brain2text25": ("train", "val"),
}


def _load_probe_manifest(manifest_path: Path) -> list[CanonicalProbeManifestRow]:
    rows: list[CanonicalProbeManifestRow] = []
    for line in manifest_path.read_text().splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        rows.append(
            CanonicalProbeManifestRow(
                example_id=str(payload["example_id"]),
                session_id=str(payload["session_id"]),
                subject_id=None if payload.get("subject_id") in {None, ""} else str(payload.get("subject_id")),
                source_split=str(payload.get("source_split", "")),
                has_labels=bool(payload.get("has_labels", False)),
                shard_relpath=str(payload["shard_relpath"]),
                example_index=int(payload["example_index"]),
                n_tx_features=int(payload.get("n_tx_features", 0) or 0),
                n_sbp_features=int(payload.get("n_sbp_features", 0) or 0),
                target_length=None if payload.get("target_length") is None else int(payload["target_length"]),
                transcript=str(payload.get("transcript", "")),
                n_time_bins=None if payload.get("n_time_bins") is None else int(payload["n_time_bins"]),
                block_num=None if payload.get("block_num") is None else int(payload["block_num"]),
                normalization_group=None
                if payload.get("normalization_group") in {None, ""}
                else str(payload.get("normalization_group")),
            )
        )
    return rows


def _load_probe_metadata(metadata_path: Path) -> dict[str, Any]:
    payload = json.loads(metadata_path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"Expected dict metadata at {metadata_path}, got {type(payload).__name__}")
    return payload


def _row_matches_feature_mode(row: CanonicalProbeManifestRow, *, feature_mode: str) -> bool:
    if feature_mode == "tx_only":
        return int(row.n_tx_features) >= AREA6V_FEATURE_DIM
    return int(row.n_tx_features) >= AREA6V_FEATURE_DIM and int(row.n_sbp_features) >= AREA6V_FEATURE_DIM


def cross_dataset_adapter_key(row: CanonicalProbeManifestRow, *, dataset: str) -> str:
    subject_id = None if row.subject_id in {None, ""} else str(row.subject_id)
    session_id = str(row.session_id)
    session_date = session_id.split(".", 1)[1] if "." in session_id else ""
    if subject_id and session_date:
        return f"{dataset}:{subject_id}:{session_date}"
    if subject_id:
        return f"{dataset}:{subject_id}"
    return f"{dataset}:{session_id}"


class CrossDatasetSequenceDataset(CanonicalSequenceDataset):
    def __init__(
        self,
        rows: tuple[CanonicalProbeManifestRow, ...] | list[CanonicalProbeManifestRow],
        *,
        cache_root: Path,
        dataset: str,
        stats: dict[str, tuple[np.ndarray, np.ndarray]] | tuple[np.ndarray, np.ndarray] | None,
        feature_mode: str,
        area6v_feature_dim: int,
    ) -> None:
        super().__init__(
            rows,
            cache_root=cache_root,
            stats=stats,
            feature_mode=feature_mode,
            boundary_key_mode="session",
            dataset=dataset,
            area6v_feature_dim=area6v_feature_dim,
        )

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = super().__getitem__(idx)
        row = self.rows[idx]
        item["boundary_key"] = cross_dataset_adapter_key(row, dataset=self.dataset)
        item["dataset"] = self.dataset
        return item


def _group_rows_by_session(
    rows: list[CanonicalProbeManifestRow],
) -> tuple[tuple[CanonicalProbeManifestRow, ...], dict[str, int], tuple[str, ...]]:
    grouped: dict[str, list[CanonicalProbeManifestRow]] = {}
    for row in rows:
        grouped.setdefault(str(row.session_id), []).append(row)
    session_ids = tuple(sorted(grouped))
    flattened = tuple(row for session_id in session_ids for row in grouped[session_id])
    counts = {session_id: len(grouped[session_id]) for session_id in session_ids}
    return flattened, counts, session_ids


def build_cross_dataset_problem(
    *,
    cache_root: str | Path,
    datasets: tuple[str, ...] | list[str],
    feature_mode: str,
) -> dict[str, Any]:
    root = Path(cache_root)
    dataset_names = tuple(str(name) for name in datasets)
    if not dataset_names:
        raise ValueError("datasets must contain at least one dataset")

    vocab: dict[str, Any] | None = None
    metadata_by_dataset: dict[str, dict[str, Any]] = {}
    rows_by_dataset: dict[str, dict[str, tuple[CanonicalProbeManifestRow, ...]]] = {}
    session_counts_by_dataset: dict[str, dict[str, dict[str, int] | tuple[str, ...]]] = {}

    for dataset in dataset_names:
        dataset_dir = root / dataset
        manifest_rows = _load_probe_manifest(dataset_dir / "manifest.jsonl")
        metadata = _load_probe_metadata(dataset_dir / "metadata.json")
        metadata_by_dataset[dataset] = metadata
        current_vocab = dict(metadata.get("phoneme_vocabulary", {}))
        if not current_vocab:
            raise ValueError(f"Dataset {dataset!r} does not define a phoneme_vocabulary in metadata.")
        if vocab is None:
            vocab = current_vocab
        elif vocab != current_vocab:
            raise ValueError(f"Dataset {dataset!r} phoneme vocabulary does not match the first dataset vocabulary.")

        train_split, val_split = DATASET_SPLITS.get(dataset, ("train", "val"))
        split_counts: Counter[str] = Counter()
        train_candidates: list[CanonicalProbeManifestRow] = []
        val_candidates: list[CanonicalProbeManifestRow] = []
        for row in manifest_rows:
            split_name = str(row.source_split).strip().lower()
            if bool(row.has_labels):
                split_counts[split_name] += 1
            if not bool(row.has_labels) or not _row_matches_feature_mode(row, feature_mode=feature_mode):
                continue
            if split_name == str(train_split).lower():
                train_candidates.append(row)
            elif split_name == str(val_split).lower():
                val_candidates.append(row)

        if not train_candidates:
            raise ValueError(
                f"No labeled {train_split!r} rows were found for dataset {dataset!r}. "
                f"Observed labeled split counts: {dict(split_counts)}"
            )
        if not val_candidates:
            raise ValueError(
                f"No labeled {val_split!r} rows were found for dataset {dataset!r}. "
                f"Observed labeled split counts: {dict(split_counts)}"
            )
        train_rows, train_examples_by_session, train_session_ids = _group_rows_by_session(train_candidates)
        val_rows, val_examples_by_session, val_session_ids = _group_rows_by_session(val_candidates)
        rows_by_dataset[dataset] = {"train": train_rows, "val": val_rows}
        session_counts_by_dataset[dataset] = {
            "train_examples_by_session": train_examples_by_session,
            "val_examples_by_session": val_examples_by_session,
            "train_session_ids": train_session_ids,
            "val_session_ids": val_session_ids,
        }

    assert vocab is not None
    pooled_train_rows = tuple(row for dataset in dataset_names for row in rows_by_dataset[dataset]["train"])
    pooled_val_rows = tuple(row for dataset in dataset_names for row in rows_by_dataset[dataset]["val"])
    return {
        "cache_root": root,
        "datasets": dataset_names,
        "feature_mode": str(feature_mode),
        "vocab": vocab,
        "metadata_by_dataset": metadata_by_dataset,
        "rows_by_dataset": rows_by_dataset,
        "session_counts_by_dataset": session_counts_by_dataset,
        "train_rows": pooled_train_rows,
        "val_rows": pooled_val_rows,
    }


def compute_dataset_train_stats(
    *,
    problem: dict[str, Any],
    area6v_feature_dim: int,
    mode: str = "global",
) -> dict[str, dict[str, tuple[np.ndarray, np.ndarray]] | tuple[np.ndarray, np.ndarray] | None]:
    stats_by_dataset: dict[str, dict[str, tuple[np.ndarray, np.ndarray]] | tuple[np.ndarray, np.ndarray] | None] = {}
    for dataset in problem["datasets"]:
        stats_by_dataset[dataset] = compute_feature_stats(
            problem["rows_by_dataset"][dataset]["train"],
            cache_root=Path(problem["cache_root"]),
            mode=str(mode),
            feature_mode=str(problem["feature_mode"]),
            area6v_feature_dim=int(area6v_feature_dim),
        )
    return stats_by_dataset


def group_rows_by_adapter_key(
    rows: tuple[CanonicalProbeManifestRow, ...] | list[CanonicalProbeManifestRow],
    *,
    dataset: str,
) -> dict[str, tuple[CanonicalProbeManifestRow, ...]]:
    grouped: dict[str, list[CanonicalProbeManifestRow]] = {}
    for row in rows:
        grouped.setdefault(cross_dataset_adapter_key(row, dataset=dataset), []).append(row)
    return {key: tuple(group_rows) for key, group_rows in grouped.items()}


def make_length_aware_batch_sampler(
    rows: tuple[CanonicalProbeManifestRow, ...] | list[CanonicalProbeManifestRow],
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> LengthAwareBatchSampler:
    p95_train_input_length = canonical_rows_padded_time_percentile(rows, percentile=95.0)
    max_examples_per_microbatch = int(batch_size)
    max_padded_time_per_microbatch = int(max_examples_per_microbatch * p95_train_input_length)
    return LengthAwareBatchSampler(
        rows,
        max_examples_per_microbatch=max_examples_per_microbatch,
        max_padded_time_per_microbatch=max_padded_time_per_microbatch,
        shuffle=bool(shuffle),
        seed=int(seed),
    )


def loader_kwargs(device: torch.device) -> dict[str, Any]:
    return {
        "num_workers": 0,
        "pin_memory": device.type == "cuda",
        "collate_fn": collate_sequence_batch,
    }

"""Canonical Utah-array datasets, manifests, sampling, and normalization helpers."""

from __future__ import annotations

import copy
import importlib
import json
import math
import time
from collections import Counter
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from utah_ssl.experiment_contract import SignalSpec
from .cache import resolve_boundary_key


AREA6V_FEATURE_DIM = 128
DEFAULT_PROBE_SUMMARY_BASENAME = "downstream_probe_summary.json"
DEFAULT_PHONEME_VOCABULARY = {
    "index_to_symbol": [
        "BLANK",
        "AA",
        "AE",
        "AH",
        "AO",
        "AW",
        "AY",
        "B",
        "CH",
        "D",
        "DH",
        "EH",
        "ER",
        "EY",
        "F",
        "G",
        "HH",
        "IH",
        "IY",
        "JH",
        "K",
        "L",
        "M",
        "N",
        "NG",
        "OW",
        "OY",
        "P",
        "R",
        "S",
        "SH",
        "T",
        "TH",
        "UH",
        "UW",
        "V",
        "W",
        "Y",
        "Z",
        "ZH",
        "SIL",
    ],
    "num_classes": 41,
    "blank_index": 0,
    "sil_index": 40,
}


def _canonical_probe_paths(
    cache_root: Path,
    *,
    dataset: str = "brain2text25",
) -> tuple[Path, Path, Path]:
    canonical_root = Path(cache_root) / str(dataset)
    return canonical_root, canonical_root / "manifest.jsonl", canonical_root / "metadata.json"


def _validate_canonical_probe_assets(
    cache_root: Path,
    *,
    dataset: str = "brain2text25",
) -> tuple[Path, Path, Path]:
    canonical_root, manifest_path, metadata_path = _canonical_probe_paths(cache_root, dataset=str(dataset))
    if not manifest_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(
            f"Canonical {dataset} cache manifest / metadata is missing from the mounted cache. "
            f"Expected {manifest_path} and {metadata_path}."
        )
    return canonical_root, manifest_path, metadata_path


def _load_canonical_inventory_from_manifest(
    *,
    data_module: Any,
    canonical_root: Path,
    cache_root: Path,
) -> list[Any]:
    manifest_path = canonical_root / "manifest.jsonl"
    grouped: dict[str, dict[str, Any]] = {}
    with manifest_path.open() as handle:
        for line in handle:
            payload = json.loads(line)
            session_id = str(payload["session_id"])
            row = grouped.setdefault(
                session_id,
                {
                    "date_key": str(payload["session_date"]) if payload.get("session_date") is not None else None,
                    "total_examples": 0,
                    "has_tx": False,
                    "has_sbp": False,
                },
            )
            row["total_examples"] += 1
            row["has_tx"] = row["has_tx"] or bool(payload.get("has_tx", False))
            row["has_sbp"] = row["has_sbp"] or bool(payload.get("has_sbp", False))

    dataset_relpath = str(canonical_root.relative_to(cache_root))
    entries = []
    for session_id in sorted(grouped):
        meta = grouped[session_id]
        entries.append(
            data_module.SessionInventoryEntry(
                session_key=session_id,
                session_base=session_id,
                date_key=meta["date_key"],
                tx_root_key="canonical_cache_root" if meta["has_tx"] else None,
                tx_relpath=dataset_relpath if meta["has_tx"] else None,
                sbp_root_key="canonical_cache_root" if meta["has_sbp"] else None,
                sbp_relpath=dataset_relpath if meta["has_sbp"] else None,
                tx_windows=int(meta["total_examples"]) if meta["has_tx"] else None,
                sbp_windows=int(meta["total_examples"]) if meta["has_sbp"] else None,
                n_channels=256 if (meta["has_tx"] and meta["has_sbp"]) else 128,
                has_tx=bool(meta["has_tx"]),
                has_sbp=bool(meta["has_sbp"]),
            )
        )
    return entries


@dataclass(frozen=True)
class CanonicalProbeManifestRow:
    example_id: str
    session_id: str
    subject_id: str | None
    source_split: str
    has_labels: bool
    shard_relpath: str
    example_index: int
    n_tx_features: int
    n_sbp_features: int
    target_length: int | None
    transcript: str
    n_time_bins: int | None = None
    block_num: int | None = None
    normalization_group: str | None = None


@dataclass(frozen=True)
class CanonicalProbePartitions:
    source_pretrain: tuple[CanonicalProbeManifestRow, ...]
    target_train_by_session: dict[str, tuple[CanonicalProbeManifestRow, ...]]
    target_val_by_session: dict[str, tuple[CanonicalProbeManifestRow, ...]]


class CanonicalShardAccessor:
    def __init__(self, cache_root: Path) -> None:
        self.cache_root = Path(cache_root)
        self._shards: dict[str, dict[str, np.ndarray | None]] = {}

    def _get_shard(
        self,
        row: CanonicalProbeManifestRow,
        *,
        modalities: tuple[str, ...] = (),
    ) -> dict[str, np.ndarray | None]:
        shard_path = self.cache_root / row.shard_relpath
        key = str(shard_path)
        cached = self._shards.get(key)
        if cached is None:
            phoneme_ids_path = shard_path / "phoneme_ids.npy"
            cached = {
                "time_offsets": np.load(shard_path / "time_offsets.npy", mmap_mode="r"),
                "tx": None,
                "sbp": None,
                "phoneme_offsets": np.load(shard_path / "phoneme_offsets.npy", mmap_mode="r"),
                "phoneme_ids": np.load(phoneme_ids_path, mmap_mode="r") if phoneme_ids_path.exists() else None,
            }
            self._shards[key] = cached
        for modality in modalities:
            if modality not in {"tx", "sbp"}:
                raise ValueError(f"Unsupported shard modality: {modality!r}")
            if cached[modality] is None:
                feature_path = shard_path / f"{modality}.npy"
                if feature_path.exists():
                    cached[modality] = np.load(feature_path, mmap_mode="r")
        return cached

    def load_features(
        self,
        row: CanonicalProbeManifestRow,
        *,
        signal_spec: SignalSpec | dict[str, Any],
    ) -> np.ndarray:
        resolved_signal_spec = SignalSpec.from_value(signal_spec)
        feature_contract = resolved_signal_spec.contract
        shard = self._get_shard(row, modalities=feature_contract.modalities)
        time_offsets = shard["time_offsets"]
        assert time_offsets is not None
        start = int(time_offsets[row.example_index])
        stop = int(time_offsets[row.example_index + 1])

        parts: list[np.ndarray] = []
        tx = shard["tx"]
        sbp = shard["sbp"]
        if feature_contract.uses_tx:
            if tx is None:
                raise ValueError(
                    f"Shard {row.shard_relpath} is missing tx.npy for "
                    f"signal mode={resolved_signal_spec.mode!r}"
                )
            tx_start, tx_stop = resolved_signal_spec.selected_columns("tx")
            tx_part = np.asarray(
                tx[start:stop, tx_start:min(tx_stop, tx.shape[1])],
                dtype=np.float32,
            )
            if (
                tx_part.shape[1] < int(resolved_signal_spec.tx_dim)
                and resolved_signal_spec.missing_channel_policy == "error"
            ):
                raise ValueError(
                    f"Row {row.example_id} provides {tx_part.shape[1]} selected TX "
                    f"channels but signal_spec requires {resolved_signal_spec.tx_dim}."
                )
            if (
                tx_part.shape[1] < int(resolved_signal_spec.tx_dim)
                and resolved_signal_spec.missing_channel_policy == "zero_pad"
            ):
                tx_part = np.pad(
                    tx_part,
                    ((0, 0), (0, int(resolved_signal_spec.tx_dim) - tx_part.shape[1])),
                )
            parts.append(tx_part)
        if feature_contract.uses_sbp:
            if sbp is None:
                raise ValueError(
                    f"Shard {row.shard_relpath} is missing sbp.npy for "
                    f"signal mode={resolved_signal_spec.mode!r}"
                )
            sbp_start, sbp_stop = resolved_signal_spec.selected_columns("sbp")
            sbp_part = np.asarray(
                sbp[start:stop, sbp_start:min(sbp_stop, sbp.shape[1])],
                dtype=np.float32,
            )
            if (
                sbp_part.shape[1] < int(resolved_signal_spec.sbp_dim)
                and resolved_signal_spec.missing_channel_policy == "error"
            ):
                raise ValueError(
                    f"Row {row.example_id} provides {sbp_part.shape[1]} selected SBP "
                    f"channels but signal_spec requires {resolved_signal_spec.sbp_dim}."
                )
            if (
                sbp_part.shape[1] < int(resolved_signal_spec.sbp_dim)
                and resolved_signal_spec.missing_channel_policy == "zero_pad"
            ):
                sbp_part = np.pad(
                    sbp_part,
                    ((0, 0), (0, int(resolved_signal_spec.sbp_dim) - sbp_part.shape[1])),
                )
            parts.append(sbp_part)
        if len(parts) == 1:
            return parts[0]
        return np.concatenate(parts, axis=1)

    def load_labels(self, row: CanonicalProbeManifestRow) -> np.ndarray | None:
        if not row.has_labels:
            return None
        shard = self._get_shard(row)
        phoneme_offsets = shard["phoneme_offsets"]
        phoneme_ids = shard["phoneme_ids"]
        assert phoneme_offsets is not None
        if phoneme_ids is None:
            return np.zeros((0,), dtype=np.int64)
        start = int(phoneme_offsets[row.example_index])
        stop = int(phoneme_offsets[row.example_index + 1])
        return np.array(phoneme_ids[start:stop], dtype=np.int64, copy=True)

    def close(self) -> None:
        self._shards.clear()

    def __del__(self) -> None:
        self.close()


def _load_probe_metadata_json(metadata_path: Path) -> dict[str, Any]:
    if not metadata_path.exists():
        raise FileNotFoundError(f"Probe metadata not found: {metadata_path}")
    payload = json.loads(metadata_path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"Expected dict probe metadata at {metadata_path}, got {type(payload).__name__}")
    return payload


def _resolve_phoneme_vocabulary(metadata: dict[str, Any]) -> dict[str, Any]:
    vocab = metadata.get("phoneme_vocabulary")
    if isinstance(vocab, dict) and "index_to_symbol" in vocab:
        return vocab
    return DEFAULT_PHONEME_VOCABULARY


def _load_canonical_probe_manifest(manifest_path: Path) -> list[CanonicalProbeManifestRow]:
    rows: list[CanonicalProbeManifestRow] = []
    with manifest_path.open() as handle:
        for line in handle:
            payload = json.loads(line)
            rows.append(
                CanonicalProbeManifestRow(
                    example_id=str(payload["example_id"]),
                    session_id=str(payload["session_id"]),
                    subject_id=(
                        str(payload["subject_id"])
                        if payload.get("subject_id") is not None
                        else None
                    ),
                    source_split=str(payload["source_split"]),
                    has_labels=bool(payload["has_labels"]),
                    shard_relpath=str(payload["shard_relpath"]),
                    example_index=int(payload["example_index"]),
                    n_tx_features=int(payload.get("n_tx_features", 0) or 0),
                    n_sbp_features=int(payload.get("n_sbp_features", 0) or 0),
                    target_length=int(payload["target_length"]) if payload.get("target_length") is not None else None,
                    transcript=str(payload.get("transcript", payload.get("transcription", ""))),
                    n_time_bins=int(payload["n_time_bins"]) if payload.get("n_time_bins") is not None else None,
                    block_num=int(payload["block_num"]) if payload.get("block_num") is not None else None,
                    normalization_group=(
                        str(payload["normalization_group"])
                        if payload.get("normalization_group") is not None
                        else None
                    ),
                )
            )
    return rows


def canonical_row_input_length(row: CanonicalProbeManifestRow) -> int:
    length = getattr(row, "n_time_bins", None)
    if length is None:
        raise ValueError(f"Canonical row is missing n_time_bins metadata: {row.example_id}")
    resolved = int(length)
    if resolved <= 0:
        raise ValueError(f"Canonical row has non-positive n_time_bins={resolved}: {row.example_id}")
    return resolved


def canonical_rows_padded_time_percentile(
    rows: Sequence[CanonicalProbeManifestRow],
    *,
    percentile: float,
) -> int:
    if not rows:
        raise ValueError("Cannot compute a padded-time percentile on an empty canonical row set.")
    if not (0.0 < float(percentile) <= 100.0):
        raise ValueError("percentile must be in (0, 100].")
    lengths = np.array([canonical_row_input_length(row) for row in rows], dtype=np.float64)
    value = np.percentile(lengths, float(percentile), method="linear")
    return max(1, int(math.ceil(float(value))))


class LengthAwareBatchSampler(torch.utils.data.Sampler[list[int]]):
    def __init__(
        self,
        rows: Sequence[CanonicalProbeManifestRow],
        *,
        max_examples_per_microbatch: int,
        max_padded_time_per_microbatch: int,
        shuffle: bool,
        seed: int,
    ) -> None:
        if int(max_examples_per_microbatch) <= 0:
            raise ValueError("max_examples_per_microbatch must be positive")
        if int(max_padded_time_per_microbatch) <= 0:
            raise ValueError("max_padded_time_per_microbatch must be positive")
        self.rows = tuple(rows)
        self.max_examples_per_microbatch = int(max_examples_per_microbatch)
        self.max_padded_time_per_microbatch = int(max_padded_time_per_microbatch)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self._iteration_count = 0

    def state_dict(self) -> dict[str, int]:
        return {"iteration_count": int(self._iteration_count)}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        iteration_count = int(state.get("iteration_count", 0))
        if iteration_count < 0:
            raise ValueError("iteration_count must be non-negative")
        self._iteration_count = iteration_count

    def _ordered_indices(self, *, iteration_count: int | None = None) -> list[int]:
        indices = list(range(len(self.rows)))
        if not self.shuffle:
            return indices
        resolved_iteration_count = (
            int(self._iteration_count)
            if iteration_count is None
            else int(iteration_count)
        )
        if resolved_iteration_count < 0:
            raise ValueError("iteration_count must be non-negative")
        generator = torch.Generator()
        generator.manual_seed(self.seed + resolved_iteration_count)
        permutation = torch.randperm(len(indices), generator=generator)
        return permutation.tolist()

    def num_batches_for_iteration(self, iteration_count: int) -> int:
        """Return the deterministic batch count for one sampler iteration."""
        ordered_indices = self._ordered_indices(iteration_count=int(iteration_count))
        return len(self._build_batches(ordered_indices))

    def _build_batches(self, ordered_indices: Sequence[int]) -> list[list[int]]:
        batches: list[list[int]] = []
        current_batch: list[int] = []
        current_max_length = 0
        for row_idx in ordered_indices:
            row_length = canonical_row_input_length(self.rows[row_idx])
            proposed_count = len(current_batch) + 1
            proposed_max_length = max(current_max_length, row_length)
            proposed_padded_time = proposed_count * proposed_max_length
            exceeds_examples = proposed_count > self.max_examples_per_microbatch
            exceeds_padded_time = proposed_padded_time > self.max_padded_time_per_microbatch
            if current_batch and (exceeds_examples or exceeds_padded_time):
                batches.append(list(current_batch))
                current_batch = []
                current_max_length = 0
            current_batch.append(int(row_idx))
            current_max_length = max(current_max_length, row_length)
        if current_batch:
            batches.append(list(current_batch))
        return batches

    def __iter__(self) -> Iterator[list[int]]:
        ordered_indices = self._ordered_indices()
        batches = self._build_batches(ordered_indices)
        if self.shuffle:
            self._iteration_count += 1
        for batch in batches:
            yield batch

    def __len__(self) -> int:
        return self.num_batches_for_iteration(self._iteration_count)


def _session_ids_from_split(split: Any) -> tuple[tuple[str, ...], tuple[str, ...]]:
    def to_session_id(session_base: str) -> str:
        return session_base.split("_", 1)[0]

    source_session_ids = tuple(to_session_id(entry.session_base) for entry in split.train)
    target_session_ids = tuple(to_session_id(entry.session_base) for entry in split.val)
    return source_session_ids, target_session_ids


def _partition_probe_records(
    rows: list[CanonicalProbeManifestRow],
    *,
    source_session_ids: tuple[str, ...],
    target_session_ids: tuple[str, ...],
    pretrain_source_splits: tuple[str, ...] = ("train",),
    probe_train_split: str = "train",
    probe_val_split: str = "val",
) -> CanonicalProbePartitions:
    source_set = set(source_session_ids)
    target_set = set(target_session_ids)

    source_pretrain = tuple(
        row for row in rows if row.session_id in source_set and row.source_split in pretrain_source_splits
    )
    target_train_by_session: dict[str, list[CanonicalProbeManifestRow]] = {sid: [] for sid in target_session_ids}
    target_val_by_session: dict[str, list[CanonicalProbeManifestRow]] = {sid: [] for sid in target_session_ids}
    for row in rows:
        if row.session_id not in target_set or not row.has_labels:
            continue
        if row.source_split == probe_train_split:
            target_train_by_session[row.session_id].append(row)
        elif row.source_split == probe_val_split:
            target_val_by_session[row.session_id].append(row)

    return CanonicalProbePartitions(
        source_pretrain=source_pretrain,
        target_train_by_session={sid: tuple(records) for sid, records in target_train_by_session.items()},
        target_val_by_session={sid: tuple(records) for sid, records in target_val_by_session.items()},
    )


def _group_rows_by_session(
    rows: Sequence[CanonicalProbeManifestRow],
) -> tuple[tuple[CanonicalProbeManifestRow, ...], dict[str, int], tuple[str, ...]]:
    grouped: dict[str, list[CanonicalProbeManifestRow]] = {}
    for row in rows:
        grouped.setdefault(str(row.session_id), []).append(row)
    session_ids = tuple(sorted(grouped))
    flattened = tuple(row for session_id in session_ids for row in grouped[session_id])
    counts = {session_id: len(grouped[session_id]) for session_id in session_ids}
    return flattened, counts, session_ids


def build_competition_split_problem(
    *,
    cache_root: Path,
    signal_spec: SignalSpec | dict[str, Any],
    dataset: str = "brain2text24",
    boundary_key_mode: str = "session",
) -> dict[str, Any]:
    canonical_root, manifest_path, metadata_path = _validate_canonical_probe_assets(
        cache_root,
        dataset=str(dataset),
    )
    manifest_rows = _load_canonical_probe_manifest(manifest_path)
    metadata = _load_probe_metadata_json(metadata_path)
    resolved_signal_spec = SignalSpec.from_value(signal_spec)

    def _row_matches_feature_mode(row: CanonicalProbeManifestRow) -> bool:
        return resolved_signal_spec.row_is_compatible(
            has_tx=row.n_tx_features > 0,
            has_sbp=row.n_sbp_features > 0,
            n_tx_features=row.n_tx_features,
            n_sbp_features=row.n_sbp_features,
        )

    split_counts: Counter[str] = Counter()
    train_candidates: list[CanonicalProbeManifestRow] = []
    val_candidates: list[CanonicalProbeManifestRow] = []
    for row in manifest_rows:
        split_name = str(row.source_split).strip().lower()
        if bool(row.has_labels):
            split_counts[split_name] += 1
        if not bool(row.has_labels) or not _row_matches_feature_mode(row):
            continue
        if split_name == "competition_train":
            train_candidates.append(row)
        elif split_name == "competition_test":
            val_candidates.append(row)

    if not train_candidates:
        raise ValueError(
            "No labeled competition_train rows were found for the competition-style stage-2 split. "
            f"Observed labeled split counts: {dict(split_counts)}"
        )
    if not val_candidates:
        raise ValueError(
            "No labeled competition_test rows were found for the competition-style stage-2 split. "
            f"Observed labeled split counts: {dict(split_counts)}"
        )

    train_rows, train_examples_by_session, train_session_ids = _group_rows_by_session(train_candidates)
    val_rows, val_examples_by_session, val_session_ids = _group_rows_by_session(val_candidates)

    return {
        "canonical_root": canonical_root,
        "manifest_path": manifest_path,
        "metadata_path": metadata_path,
        "manifest_rows": manifest_rows,
        "metadata": metadata,
        "vocab": _resolve_phoneme_vocabulary(metadata),
        "cache_root": Path(cache_root),
        "dataset": str(dataset),
        "feature_mode": resolved_signal_spec.mode,
        "signal_spec": resolved_signal_spec,
        "boundary_key_mode": str(boundary_key_mode),
        "split_policy": "competition_train_test",
        "train_split_name": "competition_train",
        "val_split_name": "competition_test",
        "train_rows": train_rows,
        "val_rows": val_rows,
        "train_examples_by_session": train_examples_by_session,
        "val_examples_by_session": val_examples_by_session,
        "train_session_ids": train_session_ids,
        "val_session_ids": val_session_ids,
    }

def build_source_split_problem(
    *,
    cache_root: Path,
    dataset: str,
    signal_spec: SignalSpec | dict[str, Any],
    boundary_key_mode: str = "session",
    train_split_name: str = "train",
    val_split_name: str = "val",
) -> dict[str, Any]:
    canonical_root, manifest_path, metadata_path = _validate_canonical_probe_assets(
        cache_root,
        dataset=str(dataset),
    )
    manifest_rows = _load_canonical_probe_manifest(manifest_path)
    metadata = _load_probe_metadata_json(metadata_path)

    resolved_signal_spec = SignalSpec.from_value(signal_spec)

    def _row_matches_feature_mode(row: CanonicalProbeManifestRow) -> bool:
        return resolved_signal_spec.row_is_compatible(
            has_tx=row.n_tx_features > 0,
            has_sbp=row.n_sbp_features > 0,
            n_tx_features=row.n_tx_features,
            n_sbp_features=row.n_sbp_features,
        )

    train_key = str(train_split_name).strip().lower()
    val_key = str(val_split_name).strip().lower()
    split_counts: Counter[str] = Counter()
    train_candidates: list[CanonicalProbeManifestRow] = []
    val_candidates: list[CanonicalProbeManifestRow] = []
    for row in manifest_rows:
        split_name = str(row.source_split).strip().lower()
        if bool(row.has_labels):
            split_counts[split_name] += 1
        if not bool(row.has_labels) or not _row_matches_feature_mode(row):
            continue
        if split_name == train_key:
            train_candidates.append(row)
        elif split_name == val_key:
            val_candidates.append(row)

    if not train_candidates:
        raise ValueError(
            f"No labeled {train_split_name!r} rows were found for the source split problem. "
            f"Observed labeled split counts: {dict(split_counts)}"
        )
    if not val_candidates:
        raise ValueError(
            f"No labeled {val_split_name!r} rows were found for the source split problem. "
            f"Observed labeled split counts: {dict(split_counts)}"
        )

    train_rows, train_examples_by_session, train_session_ids = _group_rows_by_session(train_candidates)
    val_rows, val_examples_by_session, val_session_ids = _group_rows_by_session(val_candidates)

    return {
        "canonical_root": canonical_root,
        "manifest_path": manifest_path,
        "metadata_path": metadata_path,
        "manifest_rows": manifest_rows,
        "metadata": metadata,
        "vocab": _resolve_phoneme_vocabulary(metadata),
        "cache_root": Path(cache_root),
        "dataset": str(dataset),
        "feature_mode": resolved_signal_spec.mode,
        "signal_spec": resolved_signal_spec,
        "boundary_key_mode": str(boundary_key_mode),
        "split_policy": "source_split",
        "train_split_name": str(train_split_name),
        "val_split_name": str(val_split_name),
        "train_rows": train_rows,
        "val_rows": val_rows,
        "train_examples_by_session": train_examples_by_session,
        "val_examples_by_session": val_examples_by_session,
        "train_session_ids": train_session_ids,
        "val_session_ids": val_session_ids,
    }


def compute_feature_stats(
    rows: tuple[CanonicalProbeManifestRow, ...] | list[CanonicalProbeManifestRow],
    *,
    cache_root: Path,
    mode: str,
    signal_spec: SignalSpec | dict[str, Any],
) -> dict[str, tuple[np.ndarray, np.ndarray]] | tuple[np.ndarray, np.ndarray]:
    resolved_signal_spec = SignalSpec.from_value(signal_spec)
    accessor = CanonicalShardAccessor(cache_root)
    try:
        if mode == "global":
            total_count = 0
            sum_x = None
            sum_x2 = None
            for row in rows:
                x = accessor.load_features(
                    row,
                    signal_spec=resolved_signal_spec,
                )
                x64 = x.astype(np.float64, copy=False)
                if sum_x is None:
                    sum_x = x64.sum(axis=0)
                    sum_x2 = np.square(x64).sum(axis=0)
                else:
                    sum_x += x64.sum(axis=0)
                    sum_x2 += np.square(x64).sum(axis=0)
                total_count += x.shape[0]
            if sum_x is None or sum_x2 is None or total_count == 0:
                raise ValueError("Cannot compute global feature stats on an empty record set.")
            mean = sum_x / total_count
            var = np.maximum(sum_x2 / total_count - np.square(mean), 1e-6)
            std = np.sqrt(var)
            return mean.astype(np.float32), std.astype(np.float32)

        if mode == "per_session":
            grouped: dict[str, list[CanonicalProbeManifestRow]] = {}
            for row in rows:
                grouped.setdefault(row.session_id, []).append(row)
            return {
                session_id: compute_feature_stats(
                    tuple(session_rows),
                    cache_root=cache_root,
                    mode="global",
                    signal_spec=resolved_signal_spec,
                )  # type: ignore[arg-type]
                for session_id, session_rows in grouped.items()
            }

        raise ValueError("mode must be either 'global' or 'per_session'")
    finally:
        accessor.close()


def apply_feature_stats(
    x: np.ndarray,
    *,
    row: CanonicalProbeManifestRow,
    stats: dict[str, tuple[np.ndarray, np.ndarray]] | tuple[np.ndarray, np.ndarray],
) -> np.ndarray:
    if isinstance(stats, dict):
        candidate_keys: list[str] = []
        if row.block_num is not None:
            candidate_keys.append(f"{row.session_id}::block:{int(row.block_num)}")
        if row.normalization_group is not None:
            candidate_keys.append(str(row.normalization_group))
        candidate_keys.append(row.session_id)
        mean = std = None
        for key in candidate_keys:
            pair = stats.get(key)
            if pair is not None:
                mean, std = pair
                break
        if mean is None or std is None:
            raise KeyError(
                f"No feature stats found for row {row.example_id} using keys {candidate_keys!r}."
            )
    else:
        mean, std = stats
    return ((x - mean) / std).astype(np.float32, copy=False)


class CanonicalSequenceDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        rows: tuple[CanonicalProbeManifestRow, ...] | list[CanonicalProbeManifestRow],
        *,
        cache_root: Path,
        signal_spec: SignalSpec | dict[str, Any],
        stats: dict[str, tuple[np.ndarray, np.ndarray]] | tuple[np.ndarray, np.ndarray] | None = None,
        boundary_key_mode: str = "session",
        dataset: str = "brain2text25",
        input_tail_bins: int | None = None,
        pad_feature_dim_to: int | None = None,
    ) -> None:
        self.rows = list(rows)
        self.stats = stats
        self.signal_spec = SignalSpec.from_value(signal_spec)
        self.feature_mode = self.signal_spec.mode
        self.boundary_key_mode = str(boundary_key_mode)
        self.dataset = str(dataset)
        self.input_tail_bins = int(input_tail_bins) if input_tail_bins is not None else None
        self.pad_feature_dim_to = int(pad_feature_dim_to) if pad_feature_dim_to is not None else None
        self._accessor = CanonicalShardAccessor(cache_root)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.rows[idx]
        x = self._accessor.load_features(
            row,
            signal_spec=self.signal_spec,
        )
        if self.stats is not None:
            x = apply_feature_stats(x, row=row, stats=self.stats)
        else:
            x = np.array(x, dtype=np.float32, copy=True)
        if self.pad_feature_dim_to is not None:
            target_dim = int(self.pad_feature_dim_to)
            if int(x.shape[1]) > target_dim:
                raise ValueError(
                    f"Example {row.example_id} has feature dim {int(x.shape[1])}, "
                    f"which exceeds requested padded dim {target_dim}."
                )
            if int(x.shape[1]) < target_dim:
                pad = np.zeros((int(x.shape[0]), target_dim - int(x.shape[1])), dtype=np.float32)
                x = np.concatenate([x, pad], axis=1)
        if self.input_tail_bins is not None and x.shape[0] > self.input_tail_bins:
            x = x[-self.input_tail_bins :, :]
        labels = self._accessor.load_labels(row)
        if labels is None:
            labels = np.zeros((0,), dtype=np.int64)
        return {
            "x": torch.from_numpy(x),
            "input_length": int(x.shape[0]),
            "labels": torch.from_numpy(labels),
            "label_length": int(labels.shape[0]),
            "session_id": row.session_id,
            "boundary_key": resolve_boundary_key(
                dataset=self.dataset,
                session_id=row.session_id,
                subject_id=row.subject_id,
                boundary_key_mode=self.boundary_key_mode,
            ),
            "example_id": row.example_id,
        }

    def __del__(self) -> None:
        accessor = getattr(self, "_accessor", None)
        if accessor is not None:
            accessor.close()


def collate_sequence_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    batch_size = len(batch)
    max_time = max(item["input_length"] for item in batch)
    max_label = max(item["label_length"] for item in batch)
    input_dim = int(batch[0]["x"].shape[1])

    x = torch.zeros((batch_size, max_time, input_dim), dtype=torch.float32)
    labels = torch.zeros((batch_size, max_label), dtype=torch.int64)
    input_lengths = torch.empty((batch_size,), dtype=torch.long)
    label_lengths = torch.empty((batch_size,), dtype=torch.long)
    session_ids: list[str] = []
    boundary_keys: list[str] = []
    example_ids: list[str] = []

    for idx, item in enumerate(batch):
        t = item["input_length"]
        l = item["label_length"]
        x[idx, :t] = item["x"]
        if l > 0:
            labels[idx, :l] = item["labels"]
        input_lengths[idx] = t
        label_lengths[idx] = l
        session_ids.append(item["session_id"])
        boundary_keys.append(item["boundary_key"])
        example_ids.append(item["example_id"])

    return {
        "x": x,
        "labels": labels,
        "input_lengths": input_lengths,
        "label_lengths": label_lengths,
        "session_ids": session_ids,
        "boundary_keys": boundary_keys,
        "example_ids": example_ids,
    }

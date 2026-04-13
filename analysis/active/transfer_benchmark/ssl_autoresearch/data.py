"""Dataset inventory, cache, and split helpers for the full SSL autoresearch scaffold."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from prepare import CACHE_ROOT, resolve_relative_path


@dataclass(frozen=True)
class SessionInventoryEntry:
    session_key: str
    session_base: str
    date_key: str | None
    tx_root_key: str | None
    tx_relpath: str | None
    sbp_root_key: str | None
    sbp_relpath: str | None
    tx_windows: int | None
    sbp_windows: int | None
    n_channels: int | None
    has_tx: bool
    has_sbp: bool


@dataclass(frozen=True)
class SessionSplit:
    train: tuple[SessionInventoryEntry, ...]
    val: tuple[SessionInventoryEntry, ...]


@dataclass(frozen=True)
class SourceStatus:
    name: str
    root_key: str | None
    env_var: str | None
    path: str
    exists: bool
    kind: str


@dataclass(frozen=True)
class CacheSessionEntry:
    session_id: str
    session_date: str | None
    subject_id: str
    source_splits: tuple[str, ...]
    total_examples: int
    labeled_examples: int
    has_train: bool
    has_val: bool
    has_test: bool
    has_tx: bool
    has_sbp: bool


CANONICAL_B2T25_ROOT = CACHE_ROOT / "brain2text25"


def _load_manifest(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    raw = json.loads(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"Expected dict manifest at {path}, got {type(raw).__name__}")
    out: dict[str, dict[str, Any]] = {}
    for key, value in raw.items():
        if isinstance(value, dict):
            out[key] = value
    return out


def _session_base_from_key(key: str) -> str:
    suffixes = ("_sbp_64", "_64")
    for suffix in suffixes:
        if key.endswith(suffix):
            return key[: -len(suffix)]
    return key


def _extract_date_key(session_base: str) -> str | None:
    parts = session_base.split("_", 1)[0].split(".")
    if len(parts) >= 4:
        return ".".join(parts[1:4])
    return None


def load_b2t25_cache_inventory(tx_cache_dir: Path, sbp_cache_dir: Path) -> list[SessionInventoryEntry]:
    tx_manifest = _load_manifest(tx_cache_dir / "manifest.json")
    sbp_manifest = _load_manifest(sbp_cache_dir / "manifest.json")

    all_keys = sorted(set(tx_manifest) | set(sbp_manifest))
    entries: list[SessionInventoryEntry] = []
    for key in all_keys:
        tx_meta = tx_manifest.get(key)
        sbp_key = f"{_session_base_from_key(key)}_sbp_64"
        sbp_meta = sbp_manifest.get(sbp_key)
        session_base = _session_base_from_key(key)
        tx_filename = tx_meta.get("filename") if tx_meta else None
        sbp_filename = sbp_meta.get("filename") if sbp_meta else None
        entries.append(
            SessionInventoryEntry(
                session_key=key,
                session_base=session_base,
                date_key=_extract_date_key(session_base),
                tx_root_key="ssl_tx_cache" if tx_filename else None,
                tx_relpath=tx_filename if tx_filename else None,
                sbp_root_key="ssl_sbp_cache" if sbp_filename else None,
                sbp_relpath=sbp_filename if sbp_filename else None,
                tx_windows=int(tx_meta["n_windows"]) if tx_meta and "n_windows" in tx_meta else None,
                sbp_windows=int(sbp_meta["n_windows"]) if sbp_meta and "n_windows" in sbp_meta else None,
                n_channels=int(tx_meta["n_channels"]) if tx_meta and "n_channels" in tx_meta else None,
                has_tx=tx_meta is not None,
                has_sbp=sbp_meta is not None,
            )
        )
    return entries


def filter_matched_tx_sbp(entries: list[SessionInventoryEntry]) -> list[SessionInventoryEntry]:
    return [entry for entry in entries if entry.has_tx and entry.has_sbp]


def split_latest_sessions(
    entries: list[SessionInventoryEntry],
    *,
    session_limit: int,
    val_session_count: int,
) -> SessionSplit:
    matched = sorted(entries, key=lambda entry: (entry.date_key or "", entry.session_base))
    if session_limit <= 0:
        raise ValueError("session_limit must be positive")
    if val_session_count <= 0:
        raise ValueError("val_session_count must be positive")
    selected = matched[-session_limit:]
    if len(selected) <= val_session_count:
        raise ValueError("Need more selected sessions than validation sessions.")
    train = tuple(selected[:-val_session_count])
    val = tuple(selected[-val_session_count:])
    return SessionSplit(train=train, val=val)


def discover_b2t25_sources(brain2text25_root: Path) -> list[SourceStatus]:
    candidates = [
        ("brain2text25_root", "brain2text25_root", "SSL_AUTORESEARCH_B2T25_ROOT", brain2text25_root, "directory"),
        ("hdf5_data_final", "brain2text25_hdf5", "SSL_AUTORESEARCH_B2T25_HDF5_ROOT", brain2text25_root / "hdf5_data_final", "directory"),
        ("release_2024", None, None, brain2text25_root / "2024", "directory"),
        ("model_training", None, None, brain2text25_root / "model_training", "directory"),
    ]
    return [
        SourceStatus(
            name=name,
            root_key=root_key,
            env_var=env_var,
            path=str(path.resolve()),
            exists=path.exists(),
            kind=kind,
        )
        for name, root_key, env_var, path, kind in candidates
    ]


def inventory_summary(entries: list[SessionInventoryEntry]) -> dict[str, Any]:
    matched = filter_matched_tx_sbp(entries)
    return {
        "total_entries": len(entries),
        "matched_tx_sbp_entries": len(matched),
        "date_range": [
            min((entry.date_key for entry in matched if entry.date_key), default=None),
            max((entry.date_key for entry in matched if entry.date_key), default=None),
        ],
        "sample_session_bases": [entry.session_base for entry in matched[:5]],
    }


def inventory_to_jsonable(entries: list[SessionInventoryEntry]) -> list[dict[str, Any]]:
    return [asdict(entry) for entry in entries]


def load_b2t25_canonical_inventory(cache_dataset_root: Path = CANONICAL_B2T25_ROOT) -> list[SessionInventoryEntry]:
    manifest_path = cache_dataset_root / "manifest.jsonl"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Canonical cache manifest not found: {manifest_path}")

    grouped: dict[str, dict[str, Any]] = {}
    with manifest_path.open() as handle:
        for line in handle:
            payload = json.loads(line)
            session_id = str(payload["session_id"])
            row = grouped.setdefault(
                session_id,
                {
                    "date_key": str(payload["session_date"]) if payload["session_date"] is not None else None,
                    "source_splits": set(),
                    "total_examples": 0,
                    "has_tx": False,
                    "has_sbp": False,
                },
            )
            row["source_splits"].add(str(payload["source_split"]))
            row["total_examples"] += 1
            row["has_tx"] = row["has_tx"] or bool(payload.get("has_tx", False))
            row["has_sbp"] = row["has_sbp"] or bool(payload.get("has_sbp", False))

    entries: list[SessionInventoryEntry] = []
    for session_id in sorted(grouped):
        meta = grouped[session_id]
        entries.append(
            SessionInventoryEntry(
                session_key=session_id,
                session_base=session_id,
                date_key=meta["date_key"],
                tx_root_key="canonical_cache_root" if meta["has_tx"] else None,
                tx_relpath=str(cache_dataset_root.relative_to(CACHE_ROOT)) if meta["has_tx"] else None,
                sbp_root_key="canonical_cache_root" if meta["has_sbp"] else None,
                sbp_relpath=str(cache_dataset_root.relative_to(CACHE_ROOT)) if meta["has_sbp"] else None,
                tx_windows=int(meta["total_examples"]) if meta["has_tx"] else None,
                sbp_windows=int(meta["total_examples"]) if meta["has_sbp"] else None,
                n_channels=512 if (meta["has_tx"] and meta["has_sbp"]) else 256,
                has_tx=bool(meta["has_tx"]),
                has_sbp=bool(meta["has_sbp"]),
            )
        )
    return entries


@dataclass(frozen=True)
class ProbeManifestRow:
    example_id: str
    dataset_family: str
    subject_id: str
    has_labels: bool
    session_id: str
    session_date: str | None
    source_split: str
    cache_root_key: str
    cache_dataset_relpath: str
    shard_id: str
    shard_relpath: str
    example_index: int
    source_root_key: str
    source_relpath: str
    trial_key: str
    block_num: int
    trial_num: int
    feature_modalities: str
    bin_size_ms: int
    n_time_bins: int
    n_features: int
    n_tx_features: int
    n_sbp_features: int
    target_length: int | None
    transcription: str
    sentence_label: str
    normalization_group: str


@dataclass(frozen=True)
class ProbeBenchmarkPartitions:
    source_pretrain: tuple[ProbeManifestRow, ...]
    target_train_by_session: dict[str, tuple[ProbeManifestRow, ...]]
    target_val_by_session: dict[str, tuple[ProbeManifestRow, ...]]


def load_probe_manifest(manifest_path: Path) -> list[ProbeManifestRow]:
    rows: list[ProbeManifestRow] = []
    with manifest_path.open() as handle:
        for line in handle:
            payload = json.loads(line)
            rows.append(
                ProbeManifestRow(
                    example_id=str(payload["example_id"]),
                    dataset_family=str(payload["dataset_family"]),
                    subject_id=str(payload["subject_id"]),
                    has_labels=bool(payload["has_labels"]),
                    session_id=str(payload["session_id"]),
                    session_date=str(payload["session_date"]) if payload["session_date"] is not None else None,
                    source_split=str(payload["source_split"]),
                    cache_root_key=str(payload["cache_root_key"]),
                    cache_dataset_relpath=str(payload["cache_dataset_relpath"]),
                    shard_id=str(payload["shard_id"]),
                    shard_relpath=str(payload["shard_relpath"]),
                    example_index=int(payload["example_index"]),
                    source_root_key=str(payload["source_root_key"]),
                    source_relpath=str(payload["source_relpath"]),
                    trial_key=str(payload["trial_key"]),
                    block_num=int(payload["block_num"]),
                    trial_num=int(payload["trial_num"]),
                    feature_modalities=str(payload["feature_modalities"]),
                    bin_size_ms=int(payload["bin_size_ms"]),
                    n_time_bins=int(payload["n_time_bins"]),
                    n_features=int(payload["n_tx_features"]) + int(payload["n_sbp_features"]),
                    n_tx_features=int(payload["n_tx_features"]),
                    n_sbp_features=int(payload["n_sbp_features"]),
                    target_length=int(payload["target_length"]) if payload["target_length"] is not None else None,
                    transcription=str(payload["transcript"]),
                    sentence_label=str(payload["sentence_label"]),
                    normalization_group=str(payload["normalization_group"]),
                )
            )
    return rows


def load_probe_metadata(metadata_path: Path) -> dict[str, Any]:
    if not metadata_path.exists():
        raise FileNotFoundError(f"Probe metadata not found: {metadata_path}")
    payload = json.loads(metadata_path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"Expected dict probe metadata at {metadata_path}, got {type(payload).__name__}")
    return payload


def session_ids_from_cache_split(split: SessionSplit) -> tuple[tuple[str, ...], tuple[str, ...]]:
    def to_session_id(session_base: str) -> str:
        return session_base.split("_", 1)[0]

    source_session_ids = tuple(to_session_id(entry.session_base) for entry in split.train)
    target_session_ids = tuple(to_session_id(entry.session_base) for entry in split.val)
    return source_session_ids, target_session_ids


def partition_probe_records(
    rows: list[ProbeManifestRow],
    *,
    source_session_ids: tuple[str, ...],
    target_session_ids: tuple[str, ...],
    pretrain_source_splits: tuple[str, ...] = ("train",),
    probe_train_split: str = "train",
    probe_val_split: str = "val",
) -> ProbeBenchmarkPartitions:
    source_set = set(source_session_ids)
    target_set = set(target_session_ids)

    source_pretrain = tuple(
        row
        for row in rows
        if row.session_id in source_set and row.source_split in pretrain_source_splits
    )

    target_train_by_session: dict[str, list[ProbeManifestRow]] = {sid: [] for sid in target_session_ids}
    target_val_by_session: dict[str, list[ProbeManifestRow]] = {sid: [] for sid in target_session_ids}
    for row in rows:
        if row.session_id not in target_set or not row.has_labels:
            continue
        if row.source_split == probe_train_split:
            target_train_by_session[row.session_id].append(row)
        elif row.source_split == probe_val_split:
            target_val_by_session[row.session_id].append(row)

    return ProbeBenchmarkPartitions(
        source_pretrain=source_pretrain,
        target_train_by_session={sid: tuple(records) for sid, records in target_train_by_session.items()},
        target_val_by_session={sid: tuple(records) for sid, records in target_val_by_session.items()},
    )


def probe_partition_summary(partitions: ProbeBenchmarkPartitions) -> dict[str, Any]:
    return {
        "source_pretrain_examples": len(partitions.source_pretrain),
        "target_train_examples_by_session": {
            session_id: len(rows)
            for session_id, rows in partitions.target_train_by_session.items()
        },
        "target_val_examples_by_session": {
            session_id: len(rows)
            for session_id, rows in partitions.target_val_by_session.items()
        },
    }


class CanonicalShardAccessor:
    def __init__(self) -> None:
        self._shards: dict[str, dict[str, np.ndarray | None]] = {}

    def _get_shard(self, row: ProbeManifestRow) -> dict[str, np.ndarray | None]:
        shard_path = resolve_relative_path(row.cache_root_key, row.shard_relpath)
        key = str(shard_path)
        cached = self._shards.get(key)
        if cached is None:
            tx_path = shard_path / "tx.npy"
            sbp_path = shard_path / "sbp.npy"
            phoneme_ids_path = shard_path / "phoneme_ids.npy"
            cached = {
                "time_offsets": np.load(shard_path / "time_offsets.npy", mmap_mode="r"),
                "tx": np.load(tx_path, mmap_mode="r") if tx_path.exists() else None,
                "sbp": np.load(sbp_path, mmap_mode="r") if sbp_path.exists() else None,
                "phoneme_offsets": np.load(shard_path / "phoneme_offsets.npy", mmap_mode="r"),
                "phoneme_ids": np.load(phoneme_ids_path, mmap_mode="r") if phoneme_ids_path.exists() else None,
            }
            self._shards[key] = cached
        return cached

    def load_features(self, row: ProbeManifestRow) -> np.ndarray:
        shard = self._get_shard(row)
        time_offsets = shard["time_offsets"]
        assert time_offsets is not None
        start = int(time_offsets[row.example_index])
        stop = int(time_offsets[row.example_index + 1])

        parts: list[np.ndarray] = []
        tx = shard["tx"]
        sbp = shard["sbp"]
        if tx is not None:
            parts.append(np.asarray(tx[start:stop], dtype=np.float32))
        if sbp is not None:
            parts.append(np.asarray(sbp[start:stop], dtype=np.float32))
        if not parts:
            raise ValueError(f"Shard {row.shard_id} has neither tx.npy nor sbp.npy")
        if len(parts) == 1:
            return parts[0]
        return np.concatenate(parts, axis=1)

    def load_labels(self, row: ProbeManifestRow) -> np.ndarray | None:
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
        return np.asarray(phoneme_ids[start:stop], dtype=np.int64)

    def close(self) -> None:
        self._shards.clear()

    def __del__(self) -> None:
        self.close()


def compute_feature_stats(
    rows: tuple[ProbeManifestRow, ...] | list[ProbeManifestRow],
    *,
    mode: str,
) -> dict[str, tuple[np.ndarray, np.ndarray]] | tuple[np.ndarray, np.ndarray]:
    accessor = CanonicalShardAccessor()
    try:
        if mode == "global":
            total_count = 0
            sum_x = None
            sum_x2 = None
            for row in rows:
                x = accessor.load_features(row)
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
            grouped: dict[str, list[ProbeManifestRow]] = {}
            for row in rows:
                grouped.setdefault(row.session_id, []).append(row)
            out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
            for session_id, session_rows in grouped.items():
                out[session_id] = compute_feature_stats(tuple(session_rows), mode="global")  # type: ignore[arg-type]
            return out

        raise ValueError("mode must be either 'global' or 'per_session'")
    finally:
        accessor.close()


def apply_feature_stats(
    x: np.ndarray,
    *,
    row: ProbeManifestRow,
    stats: dict[str, tuple[np.ndarray, np.ndarray]] | tuple[np.ndarray, np.ndarray],
) -> np.ndarray:
    if isinstance(stats, dict):
        mean, std = stats[row.session_id]
    else:
        mean, std = stats
    return ((x - mean) / std).astype(np.float32, copy=False)


class CanonicalSequenceDataset(Dataset):
    def __init__(
        self,
        rows: tuple[ProbeManifestRow, ...] | list[ProbeManifestRow],
        *,
        stats: dict[str, tuple[np.ndarray, np.ndarray]] | tuple[np.ndarray, np.ndarray] | None = None,
    ) -> None:
        self.rows = list(rows)
        self.stats = stats
        self._accessor = CanonicalShardAccessor()

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.rows[idx]
        x = self._accessor.load_features(row)
        if self.stats is not None:
            x = apply_feature_stats(x, row=row, stats=self.stats)
        labels = self._accessor.load_labels(row)
        if labels is None:
            labels = np.zeros((0,), dtype=np.int64)
        return {
            "x": torch.from_numpy(x),
            "input_length": int(x.shape[0]),
            "labels": torch.from_numpy(labels),
            "label_length": int(labels.shape[0]),
            "session_id": row.session_id,
            "example_id": row.example_id,
        }

    def __del__(self) -> None:
        self._accessor.close()


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
        example_ids.append(item["example_id"])

    return {
        "x": x,
        "labels": labels,
        "input_lengths": input_lengths,
        "label_lengths": label_lengths,
        "session_ids": session_ids,
        "example_ids": example_ids,
    }

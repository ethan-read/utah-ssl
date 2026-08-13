"""Canonical manifest records and memory-mapped shard access."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .experiment_contract import SignalSpec


DEFAULT_PHONEME_VOCABULARY = {
    "index_to_symbol": [
        "BLANK", "AA", "AE", "AH", "AO", "AW", "AY", "B", "CH", "D",
        "DH", "EH", "ER", "EY", "F", "G", "HH", "IH", "IY", "JH", "K",
        "L", "M", "N", "NG", "OW", "OY", "P", "R", "S", "SH", "T",
        "TH", "UH", "UW", "V", "W", "Y", "Z", "ZH", "SIL",
    ],
    "num_classes": 41,
    "blank_index": 0,
    "sil_index": 40,
}


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


def canonical_dataset_paths(
    cache_root: str | Path,
    *,
    dataset: str,
) -> tuple[Path, Path, Path]:
    dataset_root = Path(cache_root) / str(dataset)
    return dataset_root, dataset_root / "manifest.jsonl", dataset_root / "metadata.json"


def validate_canonical_dataset(
    cache_root: str | Path,
    *,
    dataset: str,
) -> tuple[Path, Path, Path]:
    dataset_root, manifest_path, metadata_path = canonical_dataset_paths(
        cache_root,
        dataset=str(dataset),
    )
    if not manifest_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(
            f"Canonical {dataset} cache manifest / metadata is missing. "
            f"Expected {manifest_path} and {metadata_path}."
        )
    return dataset_root, manifest_path, metadata_path


def load_canonical_metadata(metadata_path: str | Path) -> dict[str, Any]:
    path = Path(metadata_path)
    if not path.exists():
        raise FileNotFoundError(f"Canonical metadata not found: {path}")
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"Expected dict metadata at {path}, got {type(payload).__name__}")
    return payload


def resolve_phoneme_vocabulary(metadata: dict[str, Any]) -> dict[str, Any]:
    vocabulary = metadata.get("phoneme_vocabulary")
    if isinstance(vocabulary, dict) and "index_to_symbol" in vocabulary:
        return vocabulary
    return DEFAULT_PHONEME_VOCABULARY


def load_canonical_manifest(manifest_path: str | Path) -> list[CanonicalProbeManifestRow]:
    rows: list[CanonicalProbeManifestRow] = []
    with Path(manifest_path).open() as handle:
        for line in handle:
            if not line.strip():
                continue
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
                    target_length=(
                        int(payload["target_length"])
                        if payload.get("target_length") is not None
                        else None
                    ),
                    transcript=str(payload.get("transcript", payload.get("transcription", ""))),
                    n_time_bins=(
                        int(payload["n_time_bins"])
                        if payload.get("n_time_bins") is not None
                        else None
                    ),
                    block_num=(
                        int(payload["block_num"])
                        if payload.get("block_num") is not None
                        else None
                    ),
                    normalization_group=(
                        str(payload["normalization_group"])
                        if payload.get("normalization_group") is not None
                        else None
                    ),
                )
            )
    return rows


class CanonicalShardAccessor:
    """Lazily memory-map canonical feature and label shards."""

    def __init__(self, cache_root: str | Path) -> None:
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
                "phoneme_ids": (
                    np.load(phoneme_ids_path, mmap_mode="r")
                    if phoneme_ids_path.exists()
                    else None
                ),
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
        signal = SignalSpec.from_value(signal_spec)
        shard = self._get_shard(row, modalities=signal.modalities)
        time_offsets = shard["time_offsets"]
        assert time_offsets is not None
        start = int(time_offsets[row.example_index])
        stop = int(time_offsets[row.example_index + 1])

        parts: list[np.ndarray] = []
        for modality in signal.modalities:
            values = shard[modality]
            if values is None:
                raise ValueError(
                    f"Shard {row.shard_relpath} is missing {modality}.npy for "
                    f"signal mode={signal.mode!r}"
                )
            column_start, column_stop = signal.selected_columns(modality)
            part = np.asarray(
                values[start:stop, column_start:min(column_stop, values.shape[1])],
                dtype=np.float32,
            )
            required_dim = signal.required_dim(modality)
            if part.shape[1] < required_dim and signal.missing_channel_policy == "error":
                raise ValueError(
                    f"Row {row.example_id} provides {part.shape[1]} selected "
                    f"{modality.upper()} channels but signal_spec requires {required_dim}."
                )
            if part.shape[1] < required_dim and signal.missing_channel_policy == "zero_pad":
                part = np.pad(part, ((0, 0), (0, required_dim - part.shape[1])))
            parts.append(part)
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


__all__ = [
    "CanonicalProbeManifestRow",
    "CanonicalShardAccessor",
    "DEFAULT_PHONEME_VOCABULARY",
    "canonical_dataset_paths",
    "load_canonical_manifest",
    "load_canonical_metadata",
    "resolve_phoneme_vocabulary",
    "validate_canonical_dataset",
]

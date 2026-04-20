"""Held-out phoneme probe helpers for masked SSL notebook experiments."""

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
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .cache import resolve_boundary_key
from .model import ContrastiveSSLModel, S5ContrastiveEncoder, SessionLinearBank
from .model_mae import MAX_PATCH_COUNT as MAE_MAX_PATCH_COUNT
from .model_mae import S5ContrastiveEncoder as MAES5ContrastiveEncoder
from .training import resolve_ssl_checkpoint_path


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


@dataclass
class DownstreamProbeConfig:
    enabled: bool = True
    seed: int = 7
    comparison_mode: str = "ssl_only"
    session_limit: int = 4
    target_session_count: int = 1
    adaptation_regime: str = "A"
    probe_batch_size: int = 8
    probe_budget_seconds: int = 240
    max_probe_steps: int = 80
    progress_every_steps: int = 25
    progress_every_seconds: float = 15.0
    probe_head_learning_rate: float = 1e-3
    encoder_learning_rate: float | None = None
    weight_decay: float = 1e-2
    probe_head_type: str = "linear"
    probe_mlp_hidden_size: int | None = None
    probe_mlp_dropout: float = 0.1
    probe_lstm_hidden_size: int = 64
    probe_conv_hidden_size: int = 128
    probe_conv_kernel_size: int = 3
    checkpoint_source: str = "most_recent_valid_then_in_memory"
    explicit_checkpoint_path: str | None = None
    summary_basename: str = DEFAULT_PROBE_SUMMARY_BASENAME

    def __post_init__(self) -> None:
        if self.comparison_mode != "ssl_only":
            raise ValueError("comparison_mode must currently be 'ssl_only'")
        if self.adaptation_regime != "A":
            raise ValueError("adaptation_regime must currently be 'A'")
        if self.target_session_count <= 0:
            raise ValueError("target_session_count must be positive")
        if self.target_session_count >= self.session_limit:
            raise ValueError("target_session_count must be smaller than session_limit")
        if float(self.probe_head_learning_rate) <= 0.0:
            raise ValueError("probe_head_learning_rate must be positive")
        if self.encoder_learning_rate is not None and float(self.encoder_learning_rate) <= 0.0:
            raise ValueError("encoder_learning_rate must be positive when provided")
        if float(self.weight_decay) < 0.0:
            raise ValueError("weight_decay must be non-negative")
        if self.probe_head_type not in {"linear", "mlp2", "lstm", "conv1d"}:
            raise ValueError("probe_head_type must be one of {'linear', 'mlp2', 'lstm', 'conv1d'}")
        if self.probe_mlp_hidden_size is not None and int(self.probe_mlp_hidden_size) <= 0:
            raise ValueError("probe_mlp_hidden_size must be positive when provided")
        if not (0.0 <= float(self.probe_mlp_dropout) < 1.0):
            raise ValueError("probe_mlp_dropout must be in [0, 1)")
        if self.checkpoint_source not in {
            "most_recent_valid_then_in_memory",
            "in_memory_then_most_recent_valid",
        }:
            raise ValueError("checkpoint_source must be a supported selection policy")


class NotebookProbeEncoderAdapter(nn.Module):
    def __init__(self, encoder: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.input_dim = int(getattr(encoder, "input_dim"))
        self.hidden_size = int(getattr(encoder, "hidden_size"))
        self.token_dim = int(getattr(encoder, "token_dim"))
        self.feature_mode = str(getattr(encoder, "feature_mode", "tx_only"))
        self.source_session_keys = tuple(getattr(encoder, "source_session_keys", ()))

    def encode(
        self,
        x: torch.Tensor,
        input_lengths: torch.Tensor,
        session_ids: list[str],
        *,
        use_source_affines: bool,
        target_affines=None,
    ) -> SimpleNamespace:
        tokens, token_lengths = self.encoder.patch_batch(x, input_lengths)
        outputs = self.encoder.encode_patched(
            tokens,
            token_lengths,
            token_mask=None,
            session_keys=session_ids,
            use_source_affines=bool(use_source_affines),
            target_affines=target_affines,
        )
        return SimpleNamespace(
            hidden=outputs["hidden"],
            token_lengths=outputs["token_lengths"],
            tokens=outputs["tokens"],
        )


class LinearCTCProbe(nn.Module):
    def __init__(self, hidden_size: int, vocab_size: int):
        super().__init__()
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.linear(hidden)


class TwoLayerMLPCTCProbe(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, vocab_size: int, dropout: float = 0.1):
        super().__init__()
        self.proj_in = nn.Linear(input_size, hidden_size)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(float(dropout))
        self.proj_out = nn.Linear(hidden_size, vocab_size)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        x = self.proj_in(hidden)
        x = self.activation(x)
        x = self.dropout(x)
        return self.proj_out(x)


class TinyLSTMCTCProbe(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, vocab_size: int):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )
        self.classifier = nn.Linear(hidden_size, vocab_size)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        outputs, _ = self.lstm(hidden)
        return self.classifier(outputs)


class CausalConvCTCProbe(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, vocab_size: int, kernel_size: int):
        super().__init__()
        self.kernel_size = int(kernel_size)
        self.conv = nn.Conv1d(
            in_channels=input_size,
            out_channels=hidden_size,
            kernel_size=self.kernel_size,
        )
        self.activation = nn.GELU()
        self.classifier = nn.Linear(hidden_size, vocab_size)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        x = hidden.transpose(1, 2)
        x = F.pad(x, (self.kernel_size - 1, 0))
        x = self.conv(x)
        x = self.activation(x)
        x = x.transpose(1, 2)
        return self.classifier(x)


def _load_benchmark_modules() -> tuple[Any, Any]:
    prepare_module = importlib.import_module("prepare")
    prepare_module = importlib.reload(prepare_module)
    data_module = importlib.import_module("data")
    data_module = importlib.reload(data_module)
    return prepare_module, data_module


def _canonical_probe_paths(cache_root: Path) -> tuple[Path, Path, Path]:
    canonical_root = Path(cache_root) / "brain2text25"
    return canonical_root, canonical_root / "manifest.jsonl", canonical_root / "metadata.json"


def _validate_canonical_probe_assets(cache_root: Path) -> tuple[Path, Path, Path]:
    canonical_root, manifest_path, metadata_path = _canonical_probe_paths(cache_root)
    if not manifest_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(
            "Canonical Brain2Text25 cache manifest / metadata is missing from the mounted cache. "
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
                n_channels=512 if (meta["has_tx"] and meta["has_sbp"]) else 256,
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


@dataclass(frozen=True)
class CanonicalProbePartitions:
    source_pretrain: tuple[CanonicalProbeManifestRow, ...]
    target_train_by_session: dict[str, tuple[CanonicalProbeManifestRow, ...]]
    target_val_by_session: dict[str, tuple[CanonicalProbeManifestRow, ...]]


class CanonicalShardAccessor:
    def __init__(self, cache_root: Path) -> None:
        self.cache_root = Path(cache_root)
        self._shards: dict[str, dict[str, np.ndarray | None]] = {}

    def _get_shard(self, row: CanonicalProbeManifestRow) -> dict[str, np.ndarray | None]:
        shard_path = self.cache_root / row.shard_relpath
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

    def load_features(self, row: CanonicalProbeManifestRow, *, feature_mode: str) -> np.ndarray:
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
        if feature_mode == "tx_sbp" and sbp is not None:
            parts.append(np.asarray(sbp[start:stop], dtype=np.float32))
        if not parts:
            raise ValueError(
                f"Shard {row.shard_relpath} does not contain features for feature_mode={feature_mode!r}"
            )
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
        return np.asarray(phoneme_ids[start:stop], dtype=np.int64)

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
                )
            )
    return rows


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


def compute_feature_stats(
    rows: tuple[CanonicalProbeManifestRow, ...] | list[CanonicalProbeManifestRow],
    *,
    cache_root: Path,
    mode: str,
    feature_mode: str,
) -> dict[str, tuple[np.ndarray, np.ndarray]] | tuple[np.ndarray, np.ndarray]:
    accessor = CanonicalShardAccessor(cache_root)
    try:
        if mode == "global":
            total_count = 0
            sum_x = None
            sum_x2 = None
            for row in rows:
                x = accessor.load_features(row, feature_mode=feature_mode)
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
                    feature_mode=feature_mode,
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
        mean, std = stats[row.session_id]
    else:
        mean, std = stats
    return ((x - mean) / std).astype(np.float32, copy=False)


class CanonicalSequenceDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        rows: tuple[CanonicalProbeManifestRow, ...] | list[CanonicalProbeManifestRow],
        *,
        cache_root: Path,
        stats: dict[str, tuple[np.ndarray, np.ndarray]] | tuple[np.ndarray, np.ndarray] | None = None,
        feature_mode: str = "tx_only",
        boundary_key_mode: str = "session",
    ) -> None:
        self.rows = list(rows)
        self.stats = stats
        self.feature_mode = str(feature_mode)
        self.boundary_key_mode = str(boundary_key_mode)
        self._accessor = CanonicalShardAccessor(cache_root)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.rows[idx]
        x = self._accessor.load_features(row, feature_mode=self.feature_mode)
        if self.stats is not None:
            x = apply_feature_stats(x, row=row, stats=self.stats)
        else:
            x = np.array(x, dtype=np.float32, copy=True)
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
                dataset="brain2text25",
                session_id=row.session_id,
                subject_id=row.subject_id,
                boundary_key_mode=self.boundary_key_mode,
            ),
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


def _default_checkpoint_config(default_checkpoint_config: dict[str, Any] | None) -> dict[str, Any]:
    if default_checkpoint_config is None:
        raise ValueError("default_checkpoint_config is required for downstream probe recovery.")
    resolved = {
        "patch_size": int(default_checkpoint_config["patch_size"]),
        "patch_stride": int(default_checkpoint_config["patch_stride"]),
        "hidden_size": int(default_checkpoint_config["hidden_size"]),
        "s5_state_size": int(default_checkpoint_config["s5_state_size"]),
        "num_layers": int(default_checkpoint_config["num_layers"]),
        "dropout": float(default_checkpoint_config["dropout"]),
        "post_proj_norm": str(default_checkpoint_config.get("post_proj_norm", "rms")),
        "backbone_direction": str(default_checkpoint_config.get("backbone_direction", "causal")),
    }
    if "feature_mode" in default_checkpoint_config:
        resolved["feature_mode"] = str(default_checkpoint_config["feature_mode"])
    if "boundary_key_mode" in default_checkpoint_config:
        resolved["boundary_key_mode"] = str(default_checkpoint_config["boundary_key_mode"])
    if "input_dim" in default_checkpoint_config:
        resolved["input_dim"] = int(default_checkpoint_config["input_dim"])
    if "source_session_keys" in default_checkpoint_config:
        resolved["source_session_keys"] = [str(key) for key in default_checkpoint_config["source_session_keys"]]
    return resolved


def _resolve_candidate_checkpoint_path(
    *,
    output_root: Path,
    probe_config: DownstreamProbeConfig,
    current_checkpoint_path: Path | None,
) -> Path | None:
    explicit_checkpoint_path = probe_config.explicit_checkpoint_path
    if explicit_checkpoint_path is not None:
        candidate = Path(explicit_checkpoint_path)
        if not candidate.exists():
            raise FileNotFoundError(
                f"Explicit checkpoint path does not exist: {candidate}"
            )
        return candidate

    # In notebook workflows, the currently selected/recovered checkpoint should win
    # over older completed runs discovered under OUTPUT_ROOT.
    if current_checkpoint_path is not None:
        candidate = Path(current_checkpoint_path)
        if candidate.exists():
            return candidate

    try:
        return resolve_ssl_checkpoint_path(output_root=output_root)
    except RuntimeError:
        return None

    return None


def _recover_encoder_from_notebook_checkpoint(
    *,
    path: Path,
    input_dim: int,
) -> tuple[nn.Module, dict[str, Any]]:
    payload = torch.load(path, map_location="cpu")
    checkpoint_cfg = dict(payload.get("config", {}))
    required_keys = [
        "patch_size",
        "patch_stride",
        "hidden_size",
        "s5_state_size",
        "num_layers",
        "dropout",
    ]
    missing_keys = [key for key in required_keys if key not in checkpoint_cfg]
    if missing_keys:
        raise KeyError(
            f"Checkpoint config is missing keys needed for encoder recovery: {missing_keys}"
        )

    model_state = payload.get("model_state")
    if model_state is None:
        raise KeyError("Notebook checkpoint is missing 'model_state'.")

    encoder_state = {
        key.split("encoder.", 1)[1]: value
        for key, value in model_state.items()
        if key.startswith("encoder.")
    }
    if not encoder_state:
        raise KeyError("Notebook checkpoint does not contain encoder weights.")
    inferred_source_session_keys = tuple(str(key) for key in checkpoint_cfg.get("source_session_keys", ()))
    if not inferred_source_session_keys:
        module_keys = sorted(
            {
                key[len("source_readin.layers.") : -len(".weight")]
                for key in encoder_state.keys()
                if key.startswith("source_readin.layers.") and key.endswith(".weight")
            }
        )
        inferred_source_session_keys = tuple(
            module_key.replace("_dot_", ".").replace("_slash_", "/")
            for module_key in module_keys
        )

    is_mae_checkpoint = (
        str(checkpoint_cfg.get("objective_mode", "")) == "masked_reconstruction_mae"
        or "encoder_pos_embed" in encoder_state
    )
    if is_mae_checkpoint:
        max_patches = checkpoint_cfg.get("max_patches")
        if max_patches is None and "segment_bins" in checkpoint_cfg:
            max_patches = int(
                MAE_MAX_PATCH_COUNT(
                    int(checkpoint_cfg["segment_bins"]),
                    int(checkpoint_cfg["patch_size"]),
                    int(checkpoint_cfg["patch_stride"]),
                )
            )
        if max_patches is None and "encoder_pos_embed" in encoder_state:
            max_patches = int(encoder_state["encoder_pos_embed"].shape[1])
        if max_patches is None:
            raise KeyError(
                "MAE checkpoint is missing max_patches and could not infer it from config/state."
            )
        recovered_encoder = MAES5ContrastiveEncoder(
            input_dim=int(checkpoint_cfg.get("input_dim", input_dim)),
            hidden_size=int(checkpoint_cfg["hidden_size"]),
            s5_state_size=int(checkpoint_cfg["s5_state_size"]),
            num_layers=int(checkpoint_cfg["num_layers"]),
            dropout=float(checkpoint_cfg["dropout"]),
            patch_size=int(checkpoint_cfg["patch_size"]),
            patch_stride=int(checkpoint_cfg["patch_stride"]),
            post_proj_norm=str(checkpoint_cfg.get("post_proj_norm", "rms")),
            max_patches=int(max_patches),
            source_session_keys=inferred_source_session_keys,
            feature_mode=str(checkpoint_cfg.get("feature_mode", "tx_only")),
            backbone_direction=str(checkpoint_cfg.get("backbone_direction", "bidirectional")),
        )
    else:
        recovered_encoder = S5ContrastiveEncoder(
            input_dim=int(checkpoint_cfg.get("input_dim", input_dim)),
            hidden_size=int(checkpoint_cfg["hidden_size"]),
            s5_state_size=int(checkpoint_cfg["s5_state_size"]),
            num_layers=int(checkpoint_cfg["num_layers"]),
            dropout=float(checkpoint_cfg["dropout"]),
            patch_size=int(checkpoint_cfg["patch_size"]),
            patch_stride=int(checkpoint_cfg["patch_stride"]),
            post_proj_norm=str(checkpoint_cfg.get("post_proj_norm", "rms")),
            source_session_keys=inferred_source_session_keys,
            feature_mode=str(checkpoint_cfg.get("feature_mode", "tx_only")),
            backbone_direction=str(checkpoint_cfg.get("backbone_direction", "causal")),
        )
    recovered_encoder.load_state_dict(encoder_state)
    return recovered_encoder, checkpoint_cfg


def _resolve_downstream_probe_base_run_dir(
    *,
    output_root: Path,
    checkpoint_candidate: Path | None,
    current_run_dir: Path | None,
) -> Path:
    if checkpoint_candidate is not None:
        candidate = Path(checkpoint_candidate)
        if candidate.exists():
            if candidate.parent.name == "checkpoints":
                return candidate.parent.parent
            return candidate.parent
    if current_run_dir is not None:
        candidate = Path(current_run_dir)
        if candidate.exists():
            return candidate
    fallback = Path(output_root) / (
        f"colab_s5_downstream_probe_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    )
    fallback.mkdir(parents=True, exist_ok=True)
    return fallback


def _build_probe_state(
    *,
    base_encoder: nn.Module,
    source: str,
    checkpoint_config: dict[str, Any],
    checkpoint_path: Path | None,
    base_run_dir: Path,
) -> dict[str, Any]:
    base_encoder = base_encoder.cpu()
    return {
        "base_encoder": base_encoder,
        "encoder": NotebookProbeEncoderAdapter(base_encoder),
        "source": source,
        "checkpoint_config": dict(checkpoint_config),
        "checkpoint_path": checkpoint_path,
        "base_run_dir": Path(base_run_dir),
        "feature_mode": str(getattr(base_encoder, "feature_mode", checkpoint_config.get("feature_mode", "tx_only"))),
        "boundary_key_mode": str(checkpoint_config.get("boundary_key_mode", "session")),
        "input_dim": int(getattr(base_encoder, "input_dim")),
        "source_session_keys": list(getattr(base_encoder, "source_session_keys", checkpoint_config.get("source_session_keys", ()))),
    }


def recover_downstream_probe_state(
    *,
    probe_config: DownstreamProbeConfig,
    output_root: Path,
    input_dim: int,
    default_checkpoint_config: dict[str, Any],
    in_memory_model: ContrastiveSSLModel | None = None,
    current_checkpoint_path: Path | None = None,
    current_run_dir: Path | None = None,
) -> dict[str, Any]:
    default_cfg = _default_checkpoint_config(default_checkpoint_config)

    def load_checkpoint_state() -> dict[str, Any] | None:
        checkpoint_candidate = _resolve_candidate_checkpoint_path(
            output_root=output_root,
            probe_config=probe_config,
            current_checkpoint_path=current_checkpoint_path,
        )
        if checkpoint_candidate is None:
            return None
        base_encoder, checkpoint_cfg = _recover_encoder_from_notebook_checkpoint(
            path=checkpoint_candidate,
            input_dim=input_dim,
        )
        return _build_probe_state(
            base_encoder=base_encoder,
            source="checkpoint",
            checkpoint_config=checkpoint_cfg,
            checkpoint_path=checkpoint_candidate,
            base_run_dir=_resolve_downstream_probe_base_run_dir(
                output_root=output_root,
                checkpoint_candidate=checkpoint_candidate,
                current_run_dir=current_run_dir,
            ),
        )

    def load_in_memory_state() -> dict[str, Any] | None:
        if in_memory_model is None or not hasattr(in_memory_model, "encoder"):
            return None
        checkpoint_candidate = _resolve_candidate_checkpoint_path(
            output_root=output_root,
            probe_config=probe_config,
            current_checkpoint_path=current_checkpoint_path,
        )
        return _build_probe_state(
            base_encoder=copy.deepcopy(in_memory_model.encoder),
            source="in_memory",
            checkpoint_config=default_cfg,
            checkpoint_path=None,
            base_run_dir=_resolve_downstream_probe_base_run_dir(
                output_root=output_root,
                checkpoint_candidate=checkpoint_candidate,
                current_run_dir=current_run_dir,
            ),
        )

    if probe_config.explicit_checkpoint_path is not None:
        state_loaders = (load_checkpoint_state,)
    else:
        state_loaders = (
            (load_in_memory_state, load_checkpoint_state)
            if probe_config.checkpoint_source == "in_memory_then_most_recent_valid"
            else (load_checkpoint_state, load_in_memory_state)
        )
    for loader in state_loaders:
        state = loader()
        if state is not None:
            return state

    raise RuntimeError(
        "No in-memory encoder is available and no usable checkpoint was found. "
        "Run the training cell first, select an explicit saved step checkpoint, or make the final checkpoint available under OUTPUT_ROOT."
    )


def build_random_init_probe_state(
    *,
    reference_config: dict[str, Any],
    input_dim: int,
    seed: int,
    base_run_dir: Path,
) -> dict[str, Any]:
    checkpoint_cfg = _default_checkpoint_config(reference_config)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        base_encoder = S5ContrastiveEncoder(
            input_dim=int(reference_config.get("input_dim", input_dim)),
            hidden_size=int(checkpoint_cfg["hidden_size"]),
            s5_state_size=int(checkpoint_cfg["s5_state_size"]),
            num_layers=int(checkpoint_cfg["num_layers"]),
            dropout=float(checkpoint_cfg["dropout"]),
            patch_size=int(checkpoint_cfg["patch_size"]),
            patch_stride=int(checkpoint_cfg["patch_stride"]),
            post_proj_norm=str(checkpoint_cfg.get("post_proj_norm", "rms")),
            source_session_keys=tuple(checkpoint_cfg.get("source_session_keys", ())),
            feature_mode=str(checkpoint_cfg.get("feature_mode", "tx_only")),
        )

    return _build_probe_state(
        base_encoder=base_encoder,
        source="random_init",
        checkpoint_config=checkpoint_cfg,
        checkpoint_path=None,
        base_run_dir=Path(base_run_dir),
    )


def build_downstream_probe_problem(
    *,
    cache_root: Path,
    probe_config: DownstreamProbeConfig,
    feature_mode: str = "tx_only",
    boundary_key_mode: str = "session",
) -> dict[str, Any]:
    _, data_module = _load_benchmark_modules()
    canonical_root, manifest_path, metadata_path = _validate_canonical_probe_assets(cache_root)

    inventory = _load_canonical_inventory_from_manifest(
        data_module=data_module,
        canonical_root=canonical_root,
        cache_root=Path(cache_root),
    )
    if feature_mode == "tx_only":
        eligible_entries = [entry for entry in inventory if entry.has_tx]
    elif feature_mode == "tx_sbp":
        eligible_entries = [entry for entry in inventory if entry.has_tx and entry.has_sbp]
    else:
        raise ValueError("feature_mode must be one of {'tx_only', 'tx_sbp'}")
    split = data_module.split_latest_sessions(
        eligible_entries,
        session_limit=int(probe_config.session_limit),
        val_session_count=int(probe_config.target_session_count),
    )
    if hasattr(data_module, "session_ids_from_cache_split"):
        source_session_ids, target_session_ids = data_module.session_ids_from_cache_split(split)
    else:
        source_session_ids, target_session_ids = _session_ids_from_split(split)
    if len(target_session_ids) <= 0:
        raise ValueError("No held-out target sessions were selected for the downstream probe.")

    manifest_rows = _load_canonical_probe_manifest(manifest_path)
    metadata = _load_probe_metadata_json(metadata_path)
    partitions = _partition_probe_records(
        manifest_rows,
        source_session_ids=source_session_ids,
        target_session_ids=target_session_ids,
    )

    target_train_examples_by_session = {
        session_id: len(partitions.target_train_by_session[session_id])
        for session_id in target_session_ids
    }
    target_val_examples_by_session = {
        session_id: len(partitions.target_val_by_session[session_id])
        for session_id in target_session_ids
    }
    missing_sessions = [
        session_id
        for session_id in target_session_ids
        if target_train_examples_by_session[session_id] == 0 or target_val_examples_by_session[session_id] == 0
    ]
    if missing_sessions:
        raise ValueError(
            "At least one held-out target session does not have both train and val examples with phoneme labels. "
            f"Missing sessions: {missing_sessions}"
        )
    target_train_rows = tuple(
        row
        for session_id in target_session_ids
        for row in partitions.target_train_by_session[session_id]
    )
    target_val_rows = tuple(
        row
        for session_id in target_session_ids
        for row in partitions.target_val_by_session[session_id]
    )
    target_session_label = (
        target_session_ids[0]
        if len(target_session_ids) == 1
        else f"pooled_{len(target_session_ids)}_sessions"
    )

    return {
        "canonical_root": canonical_root,
        "manifest_path": manifest_path,
        "metadata_path": metadata_path,
        "inventory": inventory,
        "eligible_entries": eligible_entries,
        "split": split,
        "manifest_rows": manifest_rows,
        "metadata": metadata,
        "partitions": partitions,
        "source_session_ids": source_session_ids,
        "target_session_ids": target_session_ids,
        "target_session_label": target_session_label,
        "target_train_examples_by_session": target_train_examples_by_session,
        "target_val_examples_by_session": target_val_examples_by_session,
        "target_train_rows": target_train_rows,
        "target_val_rows": target_val_rows,
        "vocab": _resolve_phoneme_vocabulary(metadata),
        "cache_root": Path(cache_root),
        "feature_mode": str(feature_mode),
        "boundary_key_mode": str(boundary_key_mode),
    }


def _loader_kwargs(device: torch.device, batch_size: int, *, shuffle: bool, collate_fn: Any) -> dict[str, Any]:
    return {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": 0,
        "pin_memory": device.type == "cuda",
        "collate_fn": collate_fn,
    }


def _make_loader_generator(seed: int) -> torch.Generator:
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return generator


def _flatten_targets(labels: torch.Tensor, label_lengths: torch.Tensor) -> torch.Tensor:
    pieces = []
    for row_idx, length in enumerate(label_lengths.tolist()):
        if length > 0:
            pieces.append(labels[row_idx, :length])
    if not pieces:
        return labels.new_zeros((0,), dtype=torch.long)
    return torch.cat(pieces, dim=0)


def compute_ctc_loss_sum(
    logits: torch.Tensor,
    token_lengths: torch.Tensor,
    labels: torch.Tensor,
    label_lengths: torch.Tensor,
    *,
    blank_index: int,
) -> tuple[torch.Tensor, int]:
    target_count = int(label_lengths.sum().item())
    if target_count <= 0:
        return logits.new_zeros(()), 0
    log_probs = F.log_softmax(logits, dim=-1).transpose(0, 1)
    targets = _flatten_targets(labels, label_lengths)
    loss_sum = F.ctc_loss(
        log_probs,
        targets,
        token_lengths,
        label_lengths,
        blank=blank_index,
        reduction="sum",
        zero_infinity=True,
    )
    return loss_sum, target_count


def _ctc_greedy_decode(
    logits: torch.Tensor,
    token_lengths: torch.Tensor,
    *,
    blank_index: int,
) -> list[list[int]]:
    token_ids = logits.argmax(dim=-1)
    decoded: list[list[int]] = []
    for batch_idx, length in enumerate(token_lengths.tolist()):
        sequence: list[int] = []
        prev_token: int | None = None
        for token in token_ids[batch_idx, :length].tolist():
            if token == blank_index:
                prev_token = None
                continue
            if token != prev_token:
                sequence.append(int(token))
            prev_token = int(token)
        decoded.append(sequence)
    return decoded


def _edit_distance(reference: list[int], hypothesis: list[int]) -> int:
    if not reference:
        return len(hypothesis)
    if not hypothesis:
        return len(reference)
    previous = list(range(len(hypothesis) + 1))
    for ref_idx, ref_token in enumerate(reference, start=1):
        current = [ref_idx]
        for hyp_idx, hyp_token in enumerate(hypothesis, start=1):
            substitution_cost = 0 if ref_token == hyp_token else 1
            current.append(
                min(
                    previous[hyp_idx] + 1,
                    current[hyp_idx - 1] + 1,
                    previous[hyp_idx - 1] + substitution_cost,
                )
            )
        previous = current
    return previous[-1]


def _align_sequences(
    reference: list[int],
    hypothesis: list[int],
) -> list[tuple[str, int | None, int | None]]:
    num_ref = len(reference)
    num_hyp = len(hypothesis)
    dp = [[0] * (num_hyp + 1) for _ in range(num_ref + 1)]
    back: list[list[tuple[str, int | None, int | None] | None]] = [
        [None] * (num_hyp + 1) for _ in range(num_ref + 1)
    ]

    for ref_idx in range(1, num_ref + 1):
        dp[ref_idx][0] = ref_idx
        back[ref_idx][0] = ("deletion", int(reference[ref_idx - 1]), None)
    for hyp_idx in range(1, num_hyp + 1):
        dp[0][hyp_idx] = hyp_idx
        back[0][hyp_idx] = ("insertion", None, int(hypothesis[hyp_idx - 1]))

    for ref_idx in range(1, num_ref + 1):
        ref_token = int(reference[ref_idx - 1])
        for hyp_idx in range(1, num_hyp + 1):
            hyp_token = int(hypothesis[hyp_idx - 1])
            candidates = []
            if ref_token == hyp_token:
                candidates.append((dp[ref_idx - 1][hyp_idx - 1], 0, "correct", ref_token, hyp_token))
            else:
                candidates.append((dp[ref_idx - 1][hyp_idx - 1] + 1, 0, "substitution", ref_token, hyp_token))
            candidates.append((dp[ref_idx - 1][hyp_idx] + 1, 1, "deletion", ref_token, None))
            candidates.append((dp[ref_idx][hyp_idx - 1] + 1, 2, "insertion", None, hyp_token))
            best_cost, _, op_name, ref_value, hyp_value = min(candidates, key=lambda item: (item[0], item[1]))
            dp[ref_idx][hyp_idx] = best_cost
            back[ref_idx][hyp_idx] = (op_name, ref_value, hyp_value)

    aligned: list[tuple[str, int | None, int | None]] = []
    ref_idx = num_ref
    hyp_idx = num_hyp
    while ref_idx > 0 or hyp_idx > 0:
        step = back[ref_idx][hyp_idx]
        if step is None:
            break
        op_name, ref_value, hyp_value = step
        aligned.append((op_name, ref_value, hyp_value))
        if op_name in {"correct", "substitution"}:
            ref_idx -= 1
            hyp_idx -= 1
        elif op_name == "deletion":
            ref_idx -= 1
        elif op_name == "insertion":
            hyp_idx -= 1
        else:
            raise ValueError(f"Unexpected alignment op: {op_name}")

    aligned.reverse()
    return aligned


def _top_counter_items(counter: Counter[int], *, top_k: int = 10) -> list[list[int]]:
    return [[int(item), int(count)] for item, count in counter.most_common(top_k)]


def _top_pair_items(counter: Counter[tuple[int, int]], *, top_k: int = 10) -> list[dict[str, int]]:
    return [
        {
            "reference_id": int(reference_id),
            "predicted_id": int(predicted_id),
            "count": int(count),
        }
        for (reference_id, predicted_id), count in counter.most_common(top_k)
    ]


def _emit_progress(progress_log_path: Path | None, **payload: Any) -> None:
    if progress_log_path is None:
        return
    progress_log_path.parent.mkdir(parents=True, exist_ok=True)
    with progress_log_path.open("a") as handle:
        handle.write(json.dumps(payload) + "\n")


def evaluate_probe_session_metrics(
    *,
    encoder: nn.Module,
    probe_head: nn.Module,
    target_affines: SessionLinearBank | None,
    loader: DataLoader,
    device: torch.device,
    blank_index: int,
) -> dict[str, Any]:
    encoder.eval()
    probe_head.eval()
    if target_affines is not None:
        target_affines.eval()

    total_loss_sum = 0.0
    total_targets = 0
    total_edit_distance = 0
    total_reference_tokens = 0
    total_predicted_tokens = 0

    reference_counter: Counter[int] = Counter()
    prediction_counter: Counter[int] = Counter()
    insertion_counter: Counter[int] = Counter()
    deletion_counter: Counter[int] = Counter()
    substitution_counter: Counter[tuple[int, int]] = Counter()
    error_prediction_counter: Counter[int] = Counter()

    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            input_lengths = batch["input_lengths"].to(device)
            labels = batch["labels"].to(device)
            label_lengths = batch["label_lengths"].to(device)

            outputs = encoder.encode(
                x,
                input_lengths,
                batch["session_ids"],
                use_source_affines=False,
                target_affines=target_affines,
            )
            logits = probe_head(outputs.hidden)
            loss_sum, target_count = compute_ctc_loss_sum(
                logits,
                outputs.token_lengths,
                labels,
                label_lengths,
                blank_index=blank_index,
            )
            total_loss_sum += float(loss_sum.item())
            total_targets += target_count

            predictions = _ctc_greedy_decode(
                logits,
                outputs.token_lengths,
                blank_index=blank_index,
            )
            for row_idx, prediction in enumerate(predictions):
                reference_length = int(label_lengths[row_idx].item())
                reference = labels[row_idx, :reference_length].tolist()
                total_edit_distance += _edit_distance(reference, prediction)
                total_reference_tokens += len(reference)
                total_predicted_tokens += len(prediction)
                reference_counter.update(int(token) for token in reference)
                prediction_counter.update(int(token) for token in prediction)

                for op_name, ref_token, hyp_token in _align_sequences(reference, prediction):
                    if op_name == "substitution":
                        assert ref_token is not None and hyp_token is not None
                        substitution_counter[(int(ref_token), int(hyp_token))] += 1
                        error_prediction_counter[int(hyp_token)] += 1
                    elif op_name == "insertion":
                        assert hyp_token is not None
                        insertion_counter[int(hyp_token)] += 1
                        error_prediction_counter[int(hyp_token)] += 1
                    elif op_name == "deletion":
                        assert ref_token is not None
                        deletion_counter[int(ref_token)] += 1

    if total_targets <= 0:
        raise ValueError("Validation target count is zero; cannot compute val_bpphone.")
    if total_reference_tokens <= 0:
        raise ValueError("Validation reference token count is zero; cannot compute phoneme error rate.")

    total_insertions = int(sum(insertion_counter.values()))
    total_deletions = int(sum(deletion_counter.values()))
    total_substitutions = int(sum(substitution_counter.values()))
    total_wrong_predictions = int(sum(error_prediction_counter.values()))

    val_ctc_bpphone = total_loss_sum / total_targets / math.log(2.0)
    val_phoneme_error_rate = total_edit_distance / total_reference_tokens
    return {
        "val_ctc_bpphone": float(val_ctc_bpphone),
        "val_phoneme_error_rate": float(val_phoneme_error_rate),
        "alignment_diagnostics": {
            "total_reference_tokens": int(total_reference_tokens),
            "total_predicted_tokens": int(total_predicted_tokens),
            "total_edit_distance": int(total_edit_distance),
            "total_insertions": total_insertions,
            "total_deletions": total_deletions,
            "total_substitutions": total_substitutions,
            "total_wrong_predictions": total_wrong_predictions,
            "reference_top_ids": _top_counter_items(reference_counter),
            "prediction_top_ids": _top_counter_items(prediction_counter),
            "insertion_top_ids": _top_counter_items(insertion_counter),
            "deletion_top_ids": _top_counter_items(deletion_counter),
            "false_prediction_top_ids": _top_counter_items(error_prediction_counter),
            "substitution_top_pairs": _top_pair_items(substitution_counter),
        },
    }


def _build_probe_run_config(
    probe_config: DownstreamProbeConfig,
    probe_overrides: dict[str, Any] | None,
) -> DownstreamProbeConfig:
    return replace(probe_config, **(probe_overrides or {}))


def resolve_probe_head_type(config: DownstreamProbeConfig) -> str:
    return str(config.probe_head_type)


def probe_head_suffix(probe_head_type: str) -> str:
    return {
        "linear": "linear_probe",
        "mlp2": "mlp2_probe",
        "lstm": "lstm_probe",
        "conv1d": "conv1d_probe",
    }[probe_head_type]


def build_probe_head(
    *,
    encoder_hidden_size: int,
    probe_vocab_size: int,
    probe_config: DownstreamProbeConfig,
) -> nn.Module:
    probe_head_type = resolve_probe_head_type(probe_config)
    if probe_head_type == "linear":
        return LinearCTCProbe(encoder_hidden_size, probe_vocab_size)
    if probe_head_type == "mlp2":
        mlp_hidden = (
            int(probe_config.probe_mlp_hidden_size)
            if probe_config.probe_mlp_hidden_size is not None
            else int(encoder_hidden_size)
        )
        return TwoLayerMLPCTCProbe(
            input_size=int(encoder_hidden_size),
            hidden_size=mlp_hidden,
            vocab_size=probe_vocab_size,
            dropout=float(probe_config.probe_mlp_dropout),
        )
    if probe_head_type == "lstm":
        return TinyLSTMCTCProbe(
            input_size=encoder_hidden_size,
            hidden_size=int(probe_config.probe_lstm_hidden_size),
            vocab_size=probe_vocab_size,
        )
    return CausalConvCTCProbe(
        input_size=encoder_hidden_size,
        hidden_size=int(probe_config.probe_conv_hidden_size),
        vocab_size=probe_vocab_size,
        kernel_size=int(probe_config.probe_conv_kernel_size),
    )


def count_trainable_parameters(module: nn.Module) -> int:
    return int(sum(param.numel() for param in module.parameters() if param.requires_grad))


def train_probe_with_metrics(
    *,
    problem: dict[str, Any],
    pretrained_encoder: nn.Module,
    probe_config: DownstreamProbeConfig,
    device: torch.device,
    progress_log_path: Path | None,
    train_encoder: bool,
) -> tuple[dict[str, Any], int]:
    target_stats_mode = "global" if len(problem["target_session_ids"]) == 1 else "per_session"
    target_stats = compute_feature_stats(
        problem["target_train_rows"],
        cache_root=Path(problem["cache_root"]),
        mode=target_stats_mode,
        feature_mode=str(problem["feature_mode"]),
    )
    train_loader = DataLoader(
        CanonicalSequenceDataset(
            problem["target_train_rows"],
            cache_root=Path(problem["cache_root"]),
            stats=target_stats,
            feature_mode=str(problem["feature_mode"]),
            boundary_key_mode=str(problem.get("boundary_key_mode", "session")),
        ),
        **_loader_kwargs(
            device,
            int(probe_config.probe_batch_size),
            shuffle=True,
            collate_fn=collate_sequence_batch,
        ),
        generator=_make_loader_generator(int(probe_config.seed)),
    )
    val_loader = DataLoader(
        CanonicalSequenceDataset(
            problem["target_val_rows"],
            cache_root=Path(problem["cache_root"]),
            stats=target_stats,
            feature_mode=str(problem["feature_mode"]),
            boundary_key_mode=str(problem.get("boundary_key_mode", "session")),
        ),
        **_loader_kwargs(
            device,
            int(probe_config.probe_batch_size),
            shuffle=False,
            collate_fn=collate_sequence_batch,
        ),
        generator=_make_loader_generator(int(probe_config.seed) + 1),
    )

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(probe_config.seed))
        encoder = copy.deepcopy(pretrained_encoder).to(device)
    for parameter in encoder.parameters():
        parameter.requires_grad = bool(train_encoder)
    if train_encoder:
        encoder.train()
    else:
        encoder.eval()

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(probe_config.seed) + 2)
        probe_head = build_probe_head(
            encoder_hidden_size=encoder.hidden_size,
            probe_vocab_size=int(problem["vocab"]["num_classes"]),
            probe_config=probe_config,
        ).to(device)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(probe_config.seed) + 3)
        target_affines = SessionLinearBank(
            tuple(problem["target_session_ids"]),
            int(encoder.token_dim),
        ).to(device)
    probe_head_num_parameters = count_trainable_parameters(probe_head)
    probe_head_learning_rate = float(probe_config.probe_head_learning_rate)
    encoder_learning_rate = (
        float(probe_config.encoder_learning_rate)
        if probe_config.encoder_learning_rate is not None
        else probe_head_learning_rate
    )
    probe_head_parameters = [param for param in probe_head.parameters() if param.requires_grad]
    encoder_parameters = [param for param in encoder.parameters() if param.requires_grad]
    target_affine_parameters = [param for param in target_affines.parameters() if param.requires_grad]
    trainable_parameters = list(probe_head_parameters)
    trainable_parameters.extend(target_affine_parameters)
    if train_encoder:
        trainable_parameters.extend(encoder_parameters)

    optimizer_param_groups: list[dict[str, Any]] = [
        {
            "params": probe_head_parameters,
            "lr": probe_head_learning_rate,
        },
        {
            "params": target_affine_parameters,
            "lr": probe_head_learning_rate,
        },
    ]
    if train_encoder and encoder_parameters:
        optimizer_param_groups.append(
            {
                "params": encoder_parameters,
                "lr": encoder_learning_rate,
            }
        )

    optimizer = torch.optim.AdamW(
        optimizer_param_groups,
        lr=probe_head_learning_rate,
        weight_decay=float(probe_config.weight_decay),
    )

    start_time = time.time()
    last_report_elapsed = 0.0
    steps = 0
    while True:
        elapsed = time.time() - start_time
        if elapsed >= float(probe_config.probe_budget_seconds):
            break
        if probe_config.max_probe_steps is not None and steps >= int(probe_config.max_probe_steps):
            break

        made_progress = False
        for batch in train_loader:
            elapsed = time.time() - start_time
            if elapsed >= float(probe_config.probe_budget_seconds):
                break
            if probe_config.max_probe_steps is not None and steps >= int(probe_config.max_probe_steps):
                break

            if train_encoder:
                encoder.train()
            else:
                encoder.eval()
            probe_head.train()
            target_affines.train()

            x = batch["x"].to(device)
            input_lengths = batch["input_lengths"].to(device)
            labels = batch["labels"].to(device)
            label_lengths = batch["label_lengths"].to(device)

            outputs = encoder.encode(
                x,
                input_lengths,
                batch["session_ids"],
                use_source_affines=False,
                target_affines=target_affines,
            )
            logits = probe_head(outputs.hidden)
            loss_sum, target_count = compute_ctc_loss_sum(
                logits,
                outputs.token_lengths,
                labels,
                label_lengths,
                blank_index=int(problem["vocab"]["blank_index"]),
            )
            if target_count <= 0:
                continue

            loss = loss_sum / target_count
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_parameters, max_norm=1.0)
            optimizer.step()
            steps += 1
            made_progress = True

            elapsed = time.time() - start_time
            should_report = (
                steps == 1
                or steps % int(probe_config.progress_every_steps) == 0
                or elapsed - last_report_elapsed >= float(probe_config.progress_every_seconds)
            )
            if should_report:
                last_report_elapsed = elapsed
                _emit_progress(
                    progress_log_path,
                    event="probe_train_report",
                    stage="probe_train",
                    session_id=problem["target_session_label"],
                    step=steps,
                    elapsed_seconds=round(elapsed, 3),
                    train_ctc_bpphone=float(loss.item()) / math.log(2.0),
                    train_encoder=bool(train_encoder),
                    seed=int(probe_config.seed),
                    probe_head_type=resolve_probe_head_type(probe_config),
                    probe_head_learning_rate=probe_head_learning_rate,
                    encoder_learning_rate=encoder_learning_rate if train_encoder else None,
                    weight_decay=float(probe_config.weight_decay),
                    budget_seconds=float(probe_config.probe_budget_seconds),
                )
        if not made_progress:
            break

    metrics = evaluate_probe_session_metrics(
        encoder=encoder,
        probe_head=probe_head,
        target_affines=target_affines,
        loader=val_loader,
        device=device,
        blank_index=int(problem["vocab"]["blank_index"]),
    )
    metrics["probe_head_num_parameters"] = int(probe_head_num_parameters)
    _emit_progress(
        progress_log_path,
        event="probe_session_complete",
        stage="probe_train",
        session_id=problem["target_session_label"],
        step=steps,
        elapsed_seconds=round(time.time() - start_time, 3),
        train_encoder=bool(train_encoder),
        seed=int(probe_config.seed),
        probe_head_type=resolve_probe_head_type(probe_config),
        probe_head_learning_rate=probe_head_learning_rate,
        encoder_learning_rate=encoder_learning_rate if train_encoder else None,
        weight_decay=float(probe_config.weight_decay),
        **metrics,
    )
    return metrics, steps


def run_downstream_probe(
    *,
    probe_state: dict[str, Any],
    probe_config: DownstreamProbeConfig,
    cache_root: Path,
    device: torch.device,
    variant_prefix: str,
    artifact_prefix: str,
    train_encoder: bool,
    probe_overrides: dict[str, Any] | None = None,
    comparison_mode: str | None = None,
) -> dict[str, Any]:
    effective_probe_config = _build_probe_run_config(probe_config, probe_overrides)
    problem = build_downstream_probe_problem(
        cache_root=cache_root,
        probe_config=effective_probe_config,
        feature_mode=str(probe_state.get("feature_mode", "tx_only")),
        boundary_key_mode=str(probe_state.get("boundary_key_mode", "session")),
    )

    suffix = probe_head_suffix(resolve_probe_head_type(effective_probe_config))
    model_variant = f"{variant_prefix}_{suffix}"
    artifact_name = f"{artifact_prefix}_{suffix}"
    artifact_dir = Path(probe_state["base_run_dir"]) / artifact_name
    artifact_dir.mkdir(parents=True, exist_ok=True)

    progress_path = artifact_dir / "progress.jsonl"
    if progress_path.exists():
        progress_path.unlink()

    start_time = time.time()
    metrics, steps = train_probe_with_metrics(
        problem=problem,
        pretrained_encoder=probe_state["encoder"],
        probe_config=effective_probe_config,
        device=device,
        progress_log_path=progress_path,
        train_encoder=train_encoder,
    )
    elapsed_seconds = time.time() - start_time

    index_to_symbol = list(problem["vocab"].get("index_to_symbol", []))

    def symbol_for(token_id: int | None) -> str | None:
        if token_id is None:
            return None
        if 0 <= int(token_id) < len(index_to_symbol):
            return str(index_to_symbol[int(token_id)])
        return str(token_id)

    alignment = dict(metrics.get("alignment_diagnostics", {}))

    def format_ranked_ids(items: list[list[int]], *, denominator: int) -> list[dict[str, Any]]:
        formatted = []
        for token_id, count in items:
            formatted.append(
                {
                    "id": int(token_id),
                    "symbol": symbol_for(int(token_id)),
                    "count": int(count),
                    "rate": float(count / denominator) if denominator > 0 else None,
                }
            )
        return formatted

    def format_ranked_pairs(items: list[dict[str, int]], *, denominator: int) -> list[dict[str, Any]]:
        formatted = []
        for item in items:
            count = int(item["count"])
            reference_id = int(item["reference_id"])
            predicted_id = int(item["predicted_id"])
            formatted.append(
                {
                    "reference_id": reference_id,
                    "reference_symbol": symbol_for(reference_id),
                    "predicted_id": predicted_id,
                    "predicted_symbol": symbol_for(predicted_id),
                    "count": count,
                    "rate": float(count / denominator) if denominator > 0 else None,
                }
            )
        return formatted

    prediction_histogram = format_ranked_ids(
        list(alignment.get("prediction_top_ids", [])),
        denominator=int(alignment.get("total_predicted_tokens", 0)),
    )
    reference_histogram = format_ranked_ids(
        list(alignment.get("reference_top_ids", [])),
        denominator=int(alignment.get("total_reference_tokens", 0)),
    )
    insertion_histogram = format_ranked_ids(
        list(alignment.get("insertion_top_ids", [])),
        denominator=int(alignment.get("total_insertions", 0)),
    )
    deletion_histogram = format_ranked_ids(
        list(alignment.get("deletion_top_ids", [])),
        denominator=int(alignment.get("total_deletions", 0)),
    )
    false_prediction_histogram = format_ranked_ids(
        list(alignment.get("false_prediction_top_ids", [])),
        denominator=int(alignment.get("total_wrong_predictions", 0)),
    )
    substitution_pairs = format_ranked_pairs(
        list(alignment.get("substitution_top_pairs", [])),
        denominator=int(alignment.get("total_substitutions", 0)),
    )

    alignment_stats = {
        "total_reference_tokens": int(alignment.get("total_reference_tokens", 0)),
        "total_predicted_tokens": int(alignment.get("total_predicted_tokens", 0)),
        "total_edit_distance": int(alignment.get("total_edit_distance", 0)),
        "total_insertions": int(alignment.get("total_insertions", 0)),
        "total_deletions": int(alignment.get("total_deletions", 0)),
        "total_substitutions": int(alignment.get("total_substitutions", 0)),
        "prediction_histogram_top": prediction_histogram,
        "reference_histogram_top": reference_histogram,
        "false_prediction_histogram_top": false_prediction_histogram,
        "insertion_histogram_top": insertion_histogram,
        "deletion_histogram_top": deletion_histogram,
        "substitution_pairs_top": substitution_pairs,
        "most_common_prediction": prediction_histogram[0] if prediction_histogram else None,
        "most_common_false_prediction": false_prediction_histogram[0] if false_prediction_histogram else None,
        "top_substitution_pair": substitution_pairs[0] if substitution_pairs else None,
    }

    alignment_stats_path = artifact_dir / "val_alignment_stats.json"
    alignment_stats_path.write_text(json.dumps(alignment_stats, indent=2))

    summary_path = artifact_dir / effective_probe_config.summary_basename
    summary = {
        "model_variant": model_variant,
        "comparison_mode": comparison_mode or model_variant,
        "probe_head_type": resolve_probe_head_type(effective_probe_config),
        "seed": int(effective_probe_config.seed),
        "probe_head_num_parameters": int(metrics.get("probe_head_num_parameters", 0)),
        "probe_head_learning_rate": float(effective_probe_config.probe_head_learning_rate),
        "encoder_learning_rate": (
            float(effective_probe_config.encoder_learning_rate)
            if effective_probe_config.encoder_learning_rate is not None
            else float(effective_probe_config.probe_head_learning_rate)
            if train_encoder
            else None
        ),
        "weight_decay": float(effective_probe_config.weight_decay),
        "train_encoder": bool(train_encoder),
        "adaptation_regime": str(effective_probe_config.adaptation_regime),
        "session_limit": int(effective_probe_config.session_limit),
        "target_session_count": int(effective_probe_config.target_session_count),
        "heldout_target_session_id": (
            problem["target_session_ids"][0] if len(problem["target_session_ids"]) == 1 else None
        ),
        "source_session_ids": list(problem["source_session_ids"]),
        "target_session_ids": list(problem["target_session_ids"]),
        "pooled_target_sessions": len(problem["target_session_ids"]) > 1,
        "selected_session_bases": [
            entry.session_base for entry in problem["split"].train + problem["split"].val
        ],
        "target_train_examples_by_session": dict(problem["target_train_examples_by_session"]),
        "target_val_examples_by_session": dict(problem["target_val_examples_by_session"]),
        "target_train_examples": len(problem["target_train_rows"]),
        "target_val_examples": len(problem["target_val_rows"]),
        "val_bpphone": float(metrics["val_ctc_bpphone"]),
        "val_ctc_bpphone": float(metrics["val_ctc_bpphone"]),
        "val_phoneme_error_rate": float(metrics["val_phoneme_error_rate"]),
        "most_common_prediction": (
            alignment_stats["most_common_prediction"]["symbol"]
            if alignment_stats["most_common_prediction"] is not None
            else None
        ),
        "most_common_prediction_rate": (
            float(alignment_stats["most_common_prediction"]["rate"])
            if alignment_stats["most_common_prediction"] is not None
            else None
        ),
        "most_common_false_prediction": (
            alignment_stats["most_common_false_prediction"]["symbol"]
            if alignment_stats["most_common_false_prediction"] is not None
            else None
        ),
        "most_common_false_prediction_rate": (
            float(alignment_stats["most_common_false_prediction"]["rate"])
            if alignment_stats["most_common_false_prediction"] is not None
            else None
        ),
        "top_substitution_pair": (
            f"{alignment_stats['top_substitution_pair']['reference_symbol']}->{alignment_stats['top_substitution_pair']['predicted_symbol']}"
            if alignment_stats["top_substitution_pair"] is not None
            else None
        ),
        "top_substitution_pair_rate": (
            float(alignment_stats["top_substitution_pair"]["rate"])
            if alignment_stats["top_substitution_pair"] is not None
            else None
        ),
        "probe_steps": int(steps),
        "probe_elapsed_seconds": float(elapsed_seconds),
        "checkpoint_source_used": str(probe_state["source"]),
        "checkpoint_path": (
            str(probe_state["checkpoint_path"]) if probe_state["checkpoint_path"] is not None else None
        ),
        "feature_mode": str(problem["feature_mode"]),
        "run_dir": str(artifact_dir),
        "progress_log_path": str(progress_path),
        "alignment_stats_path": str(alignment_stats_path),
        "summary_path": str(summary_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    return summary


def run_probe_head_sweep(
    *,
    probe_state: dict[str, Any],
    probe_config: DownstreamProbeConfig,
    cache_root: Path,
    device: torch.device,
    variant_prefix: str,
    artifact_prefix: str,
    train_encoder: bool = False,
    comparison_mode_prefix: str | None = None,
    head_specs: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    specs = head_specs or [
        {"probe_head_type": "linear"},
        {"probe_head_type": "mlp2"},
    ]
    results = []
    for spec in specs:
        effective_probe_config = _build_probe_run_config(probe_config, spec)
        suffix = probe_head_suffix(resolve_probe_head_type(effective_probe_config))
        results.append(
            run_downstream_probe(
                probe_state=probe_state,
                probe_config=effective_probe_config,
                cache_root=cache_root,
                device=device,
                variant_prefix=variant_prefix,
                artifact_prefix=artifact_prefix,
                train_encoder=train_encoder,
                comparison_mode=(
                    f"{comparison_mode_prefix}_{suffix}"
                    if comparison_mode_prefix is not None
                    else None
                ),
            )
        )
    return results


def _checkpoint_probe_label(checkpoint_path: Path, *, ordinal: int) -> str:
    name = checkpoint_path.name
    stem = checkpoint_path.stem
    if name.startswith("step_"):
        parts = stem.split("_")
        if len(parts) >= 2 and parts[1].isdigit():
            return f"step{int(parts[1]):06d}"
    if name.startswith("checkpoint_final_step_"):
        parts = stem.split("_")
        if len(parts) >= 4 and parts[3].isdigit():
            return f"step{int(parts[3]):06d}"
    if name == "checkpoint_best.pt":
        return "best"
    if name == "checkpoint_final.pt":
        return "final"
    clean = "".join(char if char.isalnum() else "_" for char in stem).strip("_")
    return clean or f"checkpoint_{ordinal:02d}"


def run_checkpoint_probe_suite(
    *,
    checkpoint_paths: list[str | Path],
    probe_config: DownstreamProbeConfig,
    output_root: Path,
    input_dim: int,
    default_checkpoint_config: dict[str, Any],
    cache_root: Path,
    device: torch.device,
    head_specs: list[dict[str, Any]] | None = None,
    variant_prefix: str = "ssl_checkpoint",
    artifact_prefix: str = "ssl_checkpoint",
    train_encoder: bool = False,
) -> list[dict[str, Any]]:
    if not checkpoint_paths:
        raise ValueError("checkpoint_paths must be non-empty")

    summaries: list[dict[str, Any]] = []
    for idx, checkpoint_value in enumerate(checkpoint_paths):
        checkpoint_path = Path(checkpoint_value)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        if checkpoint_path.parent.name == "checkpoints":
            run_dir = checkpoint_path.parent.parent
        else:
            run_dir = checkpoint_path.parent

        checkpoint_probe_config = replace(
            probe_config,
            explicit_checkpoint_path=str(checkpoint_path),
        )
        checkpoint_state = recover_downstream_probe_state(
            probe_config=checkpoint_probe_config,
            output_root=output_root,
            input_dim=int(input_dim),
            default_checkpoint_config=default_checkpoint_config,
            in_memory_model=None,
            current_checkpoint_path=checkpoint_path,
            current_run_dir=run_dir,
        )

        checkpoint_label = _checkpoint_probe_label(checkpoint_path, ordinal=idx)
        checkpoint_summaries = run_probe_head_sweep(
            probe_state=checkpoint_state,
            probe_config=checkpoint_probe_config,
            cache_root=cache_root,
            device=device,
            variant_prefix=f"{variant_prefix}_{checkpoint_label}",
            artifact_prefix=f"{artifact_prefix}_{checkpoint_label}",
            train_encoder=bool(train_encoder),
            comparison_mode_prefix=checkpoint_label,
            head_specs=head_specs,
        )
        for summary in checkpoint_summaries:
            summary["evaluated_checkpoint_path"] = str(checkpoint_path)
            summary["evaluated_checkpoint_label"] = checkpoint_label
        summaries.extend(checkpoint_summaries)
    return summaries

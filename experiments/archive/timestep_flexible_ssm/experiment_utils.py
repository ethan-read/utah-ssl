"""Shared experiment helpers for mixed-bin, missing-bin, and future runs."""

from __future__ import annotations

import hashlib
import math
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from utah_ssl.datasets import (
    CanonicalSequenceDataset,
    apply_feature_stats,
)
from utah_ssl.cache import resolve_boundary_key

from .data import (
    CANONICAL_BIN_SIZE_MS,
    compute_rebinned_normalization_stats,
    rebin_features,
    rebinned_input_length,
    signal_spec_for_rows,
)
from .train import _load_or_compute_stats_for_view


def sample_feature_dim(problem: dict[str, Any], feature_mode: str) -> int:
    row = problem["train_rows"][0]
    if str(feature_mode) == "tx_only":
        return int(row.n_tx_features)
    return int(row.n_tx_features + row.n_sbp_features)


def duplicate_frames(x: np.ndarray, *, factor: int) -> np.ndarray:
    repeated = np.repeat(np.asarray(x, dtype=np.float32), int(factor), axis=0)
    return np.asarray(repeated, dtype=np.float32)


def deterministic_uniform_01(*, example_id: str, seed: int, slot: int) -> float:
    digest = hashlib.sha1(f"{seed}:{example_id}:{slot}".encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], byteorder="big", signed=False)
    return float((value % 10_000_000) / 10_000_000.0)


def bernoulli_keep_mask(*, example_id: str, seed: int, length: int, drop_probability: float) -> np.ndarray:
    keep = np.ones((int(length),), dtype=bool)
    threshold = float(drop_probability)
    for idx in range(int(length)):
        keep[idx] = deterministic_uniform_01(example_id=str(example_id), seed=int(seed), slot=idx) >= threshold
    if not bool(keep.any()):
        keep[0] = True
    return keep


def interpolate_missing_frames(x: np.ndarray, keep_mask: np.ndarray) -> np.ndarray:
    x32 = np.asarray(x, dtype=np.float32)
    keep = np.asarray(keep_mask, dtype=bool)
    if int(x32.shape[0]) == 0:
        return np.array(x32, copy=True)
    observed = np.flatnonzero(keep)
    if observed.size == 0:
        return np.array(x32, copy=True)
    if observed.size == int(x32.shape[0]):
        return np.array(x32, copy=True)
    target_positions = np.arange(int(x32.shape[0]), dtype=np.float32)
    observed_positions = observed.astype(np.float32)
    filled = np.zeros_like(x32, dtype=np.float32)
    for dim in range(int(x32.shape[1])):
        filled[:, dim] = np.interp(target_positions, observed_positions, x32[observed, dim])
    return filled


def carry_forward_missing_frames(x: np.ndarray, keep_mask: np.ndarray) -> np.ndarray:
    x32 = np.asarray(x, dtype=np.float32)
    keep = np.asarray(keep_mask, dtype=bool)
    if int(x32.shape[0]) == 0:
        return np.array(x32, copy=True)
    observed = np.flatnonzero(keep)
    if observed.size == 0 or observed.size == int(x32.shape[0]):
        return np.array(x32, copy=True)
    filled = np.zeros_like(x32, dtype=np.float32)
    last = x32[observed[0]]
    obs_set = set(int(idx) for idx in observed.tolist())
    for idx in range(int(x32.shape[0])):
        if idx in obs_set:
            last = x32[idx]
        filled[idx] = last
    return filled


def irregular_observation_view(
    x: np.ndarray,
    *,
    example_id: str,
    seed: int,
    drop_probability: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x32 = np.asarray(x, dtype=np.float32)
    keep = bernoulli_keep_mask(
        example_id=str(example_id),
        seed=int(seed),
        length=int(x32.shape[0]),
        drop_probability=float(drop_probability),
    )
    observed_idx = np.flatnonzero(keep)
    observed = x32[observed_idx]
    deltas = np.empty((int(observed.shape[0]),), dtype=np.float32)
    prev = -1
    for out_idx, idx in enumerate(observed_idx.tolist()):
        gap = int(idx) - int(prev)
        deltas[out_idx] = float(gap * CANONICAL_BIN_SIZE_MS)
        prev = int(idx)
    return observed, deltas, keep


def collate_sequence_extras(batch: list[dict[str, Any]]) -> dict[str, Any]:
    batch_size = len(batch)
    max_time = max(int(item["input_length"]) for item in batch)
    max_label = max(int(item["label_length"]) for item in batch)
    input_dim = int(batch[0]["x"].shape[1]) if max_time > 0 else int(batch[0]["x"].shape[-1])

    x = torch.zeros((batch_size, max_time, input_dim), dtype=torch.float32)
    labels = torch.zeros((batch_size, max_label), dtype=torch.int64)
    input_lengths = torch.empty((batch_size,), dtype=torch.long)
    label_lengths = torch.empty((batch_size,), dtype=torch.long)
    session_ids: list[str] = []
    boundary_keys: list[str] = []
    example_ids: list[str] = []

    has_active_bin = "active_bin_size_ms" in batch[0]
    has_time_delta = "time_deltas_ms" in batch[0]
    active_bin_sizes_ms = torch.empty((batch_size,), dtype=torch.long) if has_active_bin else None
    time_deltas_ms = torch.zeros((batch_size, max_time), dtype=torch.float32) if has_time_delta else None

    for idx, item in enumerate(batch):
        t = int(item["input_length"])
        l = int(item["label_length"])
        if t > 0:
            x[idx, :t] = item["x"]
        if l > 0:
            labels[idx, :l] = item["labels"]
        input_lengths[idx] = t
        label_lengths[idx] = l
        session_ids.append(str(item["session_id"]))
        boundary_keys.append(str(item["boundary_key"]))
        example_ids.append(str(item["example_id"]))
        if has_active_bin and active_bin_sizes_ms is not None:
            active_bin_sizes_ms[idx] = int(item["active_bin_size_ms"])
        if has_time_delta and time_deltas_ms is not None and t > 0:
            time_deltas_ms[idx, :t] = item["time_deltas_ms"]

    payload = {
        "x": x,
        "labels": labels,
        "input_lengths": input_lengths,
        "label_lengths": label_lengths,
        "session_ids": session_ids,
        "boundary_keys": boundary_keys,
        "example_ids": example_ids,
    }
    if active_bin_sizes_ms is not None:
        payload["active_bin_sizes_ms"] = active_bin_sizes_ms
    if time_deltas_ms is not None:
        payload["time_deltas_ms"] = time_deltas_ms
    return payload


class MixedBinSequenceDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        rows: tuple[Any, ...] | list[Any],
        *,
        cache_root: Path,
        stats: Any,
        feature_mode: str,
        boundary_key_mode: str,
        dataset: str,
        active_bin_size_ms: int,
        duplicate_to_canonical: bool = False,
    ) -> None:
        self.rows = list(rows)
        self.stats = stats
        self.feature_mode = str(feature_mode)
        self.signal_spec = signal_spec_for_rows(self.rows, feature_mode=self.feature_mode)
        self.boundary_key_mode = str(boundary_key_mode)
        self.dataset = str(dataset)
        self.active_bin_size_ms = int(active_bin_size_ms)
        self.duplicate_to_canonical = bool(duplicate_to_canonical)
        self._base = CanonicalSequenceDataset(
            self.rows,
            cache_root=Path(cache_root),
            signal_spec=self.signal_spec,
            stats=None,
            boundary_key_mode=self.boundary_key_mode,
            dataset=self.dataset,
        )
        self._accessor = self._base._accessor

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.rows[idx]
        x = self._accessor.load_features(row, signal_spec=self.signal_spec)
        x = rebin_features(x, bin_size_ms=self.active_bin_size_ms)
        if self.duplicate_to_canonical:
            factor = max(1, int(self.active_bin_size_ms) // int(CANONICAL_BIN_SIZE_MS))
            x = duplicate_frames(x, factor=factor)
        if self.stats is not None:
            x = apply_feature_stats(x, row=row, stats=self.stats)
        labels = self._accessor.load_labels(row)
        labels = (
            np.zeros((0,), dtype=np.int64)
            if labels is None
            else np.array(labels, dtype=np.int64, copy=True)
        )
        return {
            "x": torch.from_numpy(np.asarray(x, dtype=np.float32)),
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
            "active_bin_size_ms": int(self.active_bin_size_ms),
        }


class MissingBinSequenceDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        rows: tuple[Any, ...] | list[Any],
        *,
        cache_root: Path,
        stats: Any,
        feature_mode: str,
        boundary_key_mode: str,
        dataset: str,
        drop_probability: float,
        seed: int,
        mode: str,
    ) -> None:
        self.rows = list(rows)
        self.stats = stats
        self.feature_mode = str(feature_mode)
        self.signal_spec = signal_spec_for_rows(self.rows, feature_mode=self.feature_mode)
        self.boundary_key_mode = str(boundary_key_mode)
        self.dataset = str(dataset)
        self.drop_probability = float(drop_probability)
        self.seed = int(seed)
        self.mode = str(mode)
        self._base = CanonicalSequenceDataset(
            self.rows,
            cache_root=Path(cache_root),
            signal_spec=self.signal_spec,
            stats=None,
            boundary_key_mode=self.boundary_key_mode,
            dataset=self.dataset,
        )
        self._accessor = self._base._accessor

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.rows[idx]
        raw = np.asarray(
            self._accessor.load_features(row, signal_spec=self.signal_spec),
            dtype=np.float32,
        )
        observed, deltas_ms, keep_mask = irregular_observation_view(
            raw,
            example_id=str(row.example_id),
            seed=int(self.seed),
            drop_probability=float(self.drop_probability),
        )
        if self.mode == "s5":
            x = observed
            time_deltas_ms = deltas_ms
        elif self.mode == "gru_train":
            x = interpolate_missing_frames(raw, keep_mask)
            time_deltas_ms = np.full((int(x.shape[0]),), float(CANONICAL_BIN_SIZE_MS), dtype=np.float32)
        elif self.mode == "gru_eval":
            x = carry_forward_missing_frames(raw, keep_mask)
            time_deltas_ms = np.full((int(x.shape[0]),), float(CANONICAL_BIN_SIZE_MS), dtype=np.float32)
        else:
            raise ValueError("mode must be one of {'s5', 'gru_train', 'gru_eval'}")
        if self.stats is not None:
            x = apply_feature_stats(x, row=row, stats=self.stats)
        labels = self._accessor.load_labels(row)
        labels = (
            np.zeros((0,), dtype=np.int64)
            if labels is None
            else np.array(labels, dtype=np.int64, copy=True)
        )
        return {
            "x": torch.from_numpy(np.asarray(x, dtype=np.float32)),
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
            "time_deltas_ms": torch.from_numpy(np.asarray(time_deltas_ms, dtype=np.float32)),
            "active_bin_size_ms": int(CANONICAL_BIN_SIZE_MS),
        }


class FutureBinsDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        rows: tuple[Any, ...] | list[Any],
        *,
        cache_root: Path,
        stats: Any,
        feature_mode: str,
        boundary_key_mode: str,
        dataset: str,
    ) -> None:
        self.rows = list(rows)
        self.stats = stats
        self.feature_mode = str(feature_mode)
        self.signal_spec = signal_spec_for_rows(self.rows, feature_mode=self.feature_mode)
        self.boundary_key_mode = str(boundary_key_mode)
        self.dataset = str(dataset)
        self._base = CanonicalSequenceDataset(
            self.rows,
            cache_root=Path(cache_root),
            signal_spec=self.signal_spec,
            stats=None,
            boundary_key_mode=self.boundary_key_mode,
            dataset=self.dataset,
        )
        self._accessor = self._base._accessor

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.rows[idx]
        x = np.asarray(
            self._accessor.load_features(row, signal_spec=self.signal_spec),
            dtype=np.float32,
        )
        if self.stats is not None:
            x = apply_feature_stats(x, row=row, stats=self.stats)
        return {
            "x": torch.from_numpy(np.asarray(x, dtype=np.float32)),
            "input_length": int(x.shape[0]),
            "labels": torch.zeros((0,), dtype=torch.int64),
            "label_length": 0,
            "session_id": row.session_id,
            "boundary_key": resolve_boundary_key(
                dataset=self.dataset,
                session_id=row.session_id,
                subject_id=row.subject_id,
                boundary_key_mode=self.boundary_key_mode,
            ),
            "example_id": row.example_id,
            "active_bin_size_ms": int(CANONICAL_BIN_SIZE_MS),
        }


def build_train_val_stats(
    *,
    config: Any,
    problem: dict[str, Any],
    sample_dim: int,
    eval_bin_sizes_ms: tuple[int, ...],
) -> tuple[Any, dict[int, Any], dict[int, Any], dict[int, str | None]]:
    train_stats, train_metadata, train_stats_path = _load_or_compute_stats_for_view(
        config=config,
        problem=problem,
        rows=problem["train_rows"],
        bin_size_ms=CANONICAL_BIN_SIZE_MS,
        sample_dim=sample_dim,
    )
    val_stats_by_view: dict[int, Any] = {CANONICAL_BIN_SIZE_MS: train_stats}
    val_metadata_by_view: dict[int, Any] = {CANONICAL_BIN_SIZE_MS: train_metadata}
    val_path_by_view: dict[int, str | None] = {
        CANONICAL_BIN_SIZE_MS: None if train_stats_path is None else str(train_stats_path)
    }
    for bin_size_ms in tuple(dict.fromkeys(int(item) for item in eval_bin_sizes_ms)):
        if int(bin_size_ms) == CANONICAL_BIN_SIZE_MS:
            continue
        stats, metadata, path = _load_or_compute_stats_for_view(
            config=config,
            problem=problem,
            rows=problem["train_rows"],
            bin_size_ms=int(bin_size_ms),
            sample_dim=sample_dim,
        )
        val_stats_by_view[int(bin_size_ms)] = stats
        val_metadata_by_view[int(bin_size_ms)] = metadata
        val_path_by_view[int(bin_size_ms)] = None if path is None else str(path)
    return train_stats, val_stats_by_view, val_metadata_by_view, val_path_by_view


def future_valid_mask(
    *,
    token_lengths: torch.Tensor,
    horizon_bins: int,
    patch_size_bins: int,
    patch_stride_bins: int,
    frame_lengths: torch.Tensor,
) -> list[tuple[int, int, int]]:
    valid: list[tuple[int, int, int]] = []
    for batch_idx, token_length in enumerate(token_lengths.tolist()):
        frame_length = int(frame_lengths[batch_idx].item())
        for token_idx in range(int(token_length)):
            target_idx = token_idx * int(patch_stride_bins) + int(patch_size_bins) - 1 + int(horizon_bins)
            if target_idx < frame_length:
                valid.append((batch_idx, token_idx, target_idx))
    return valid


def sequence_mse(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(a, b, reduction="mean")


__all__ = [
    "FutureBinsDataset",
    "MixedBinSequenceDataset",
    "MissingBinSequenceDataset",
    "bernoulli_keep_mask",
    "build_train_val_stats",
    "carry_forward_missing_frames",
    "collate_sequence_extras",
    "deterministic_uniform_01",
    "duplicate_frames",
    "future_valid_mask",
    "interpolate_missing_frames",
    "irregular_observation_view",
    "sample_feature_dim",
    "sequence_mse",
]

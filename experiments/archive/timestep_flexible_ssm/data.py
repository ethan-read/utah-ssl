"""Data helpers for timestep-flexible supervised S5 decoding."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

try:
    from utah_ssl.feature_stats import apply_feature_stats
    from utah_ssl.sequence_data import (
        CanonicalSequenceDataset,
        LengthAwareBatchSampler,
        collate_sequence_batch,
    )
    from utah_ssl.session_keys import resolve_boundary_key
    from utah_ssl.experiment_contract import SignalSpec
except ModuleNotFoundError:  # pragma: no cover - repo-root unittest fallback
    from utah_ssl.feature_stats import apply_feature_stats
    from utah_ssl.sequence_data import (
        CanonicalSequenceDataset,
        LengthAwareBatchSampler,
        collate_sequence_batch,
    )
    from utah_ssl.session_keys import resolve_boundary_key
    from utah_ssl.experiment_contract import SignalSpec

from experiments.supervised_baselines.data import (
    adapter_keys_from_rows,
    build_willett_problem,
    group_rows_by_adapter_key,
    normalization_key_for_row,
    normalization_stats_missing_rows,
)


CANONICAL_BIN_SIZE_MS = 20


def signal_spec_for_rows(
    rows: tuple[Any, ...] | list[Any],
    *,
    feature_mode: str,
) -> SignalSpec:
    if not rows:
        raise ValueError("Cannot infer a signal contract from an empty row set")
    first_row = rows[0]
    return SignalSpec.from_mode(
        str(feature_mode),
        tx_dim=int(first_row.n_tx_features),
        sbp_dim=int(first_row.n_sbp_features),
    )


def signal_spec_for_cache(
    cache_root: Path,
    *,
    dataset: str,
    feature_mode: str,
) -> SignalSpec:
    manifest_path = Path(cache_root) / str(dataset) / "manifest.jsonl"
    first_row = next(
        json.loads(line)
        for line in manifest_path.read_text().splitlines()
        if line.strip()
    )
    return SignalSpec.from_mode(
        str(feature_mode),
        tx_dim=int(first_row.get("n_tx_features", 0)),
        sbp_dim=int(first_row.get("n_sbp_features", 0)),
    )


@dataclass(frozen=True)
class TimestepFlexibleInputTransformConfig:
    input_smoothing_sigma_ms: float = 40.0
    input_smoothing_kernel_size_ms: float = 2000.0
    input_smoothing_threshold: float = 0.01
    white_noise_sd: float = 1.0
    constant_offset_sd: float = 0.2


def rebin_factor_for_bin_size(bin_size_ms: int) -> int:
    resolved = int(bin_size_ms)
    if resolved <= 0:
        raise ValueError("bin_size_ms must be positive")
    if resolved % CANONICAL_BIN_SIZE_MS != 0:
        raise ValueError(
            f"bin_size_ms={resolved} must be an integer multiple of canonical {CANONICAL_BIN_SIZE_MS} ms bins."
        )
    return resolved // CANONICAL_BIN_SIZE_MS


def rebin_features(x: np.ndarray, *, bin_size_ms: int) -> np.ndarray:
    factor = rebin_factor_for_bin_size(int(bin_size_ms))
    x32 = np.asarray(x, dtype=np.float32)
    if factor == 1:
        return np.array(x32, dtype=np.float32, copy=True)
    usable = int(x32.shape[0]) // factor * factor
    if usable <= 0:
        return np.zeros((0, int(x32.shape[1])), dtype=np.float32)
    trimmed = x32[:usable]
    rebinned = trimmed.reshape(usable // factor, factor, int(x32.shape[1])).mean(axis=1)
    return np.asarray(rebinned, dtype=np.float32)


def rebinned_input_length(length: int, *, bin_size_ms: int) -> int:
    factor = rebin_factor_for_bin_size(int(bin_size_ms))
    return max(0, int(length) // factor)


def resolve_patch_bins(duration_ms: int, *, bin_size_ms: int, field_name: str) -> int:
    resolved_duration = int(duration_ms)
    resolved_bin = int(bin_size_ms)
    if resolved_duration <= 0:
        raise ValueError(f"{field_name} must be positive")
    if resolved_duration % resolved_bin != 0:
        raise ValueError(
            f"{field_name}={resolved_duration} ms must be divisible by active bin_size_ms={resolved_bin}."
        )
    return resolved_duration // resolved_bin


def _gaussian_kernel_1d(
    *,
    sigma_bins: float,
    kernel_size_bins: int,
    threshold: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    sigma = float(sigma_bins)
    if sigma <= 0.0:
        return torch.ones((1,), device=device, dtype=dtype)
    positions = torch.arange(int(kernel_size_bins), device=device, dtype=dtype) - float(
        int(kernel_size_bins) // 2
    )
    kernel = torch.exp(-0.5 * (positions / sigma).pow(2))
    kernel = kernel / kernel.sum().clamp_min(1e-8)
    keep = kernel > float(threshold)
    if not bool(keep.any().item()):
        keep[int(kernel.numel() // 2)] = True
    kept_positions = torch.nonzero(keep, as_tuple=False).squeeze(1)
    start = int(kept_positions.min().item())
    stop = int(kept_positions.max().item()) + 1
    kernel = kernel[start:stop]
    if int(kernel.numel()) % 2 == 0:
        kernel = torch.cat([kernel, kernel.new_zeros((1,))], dim=0)
    return kernel / kernel.sum().clamp_min(1e-8)


def _sequence_mask_from_lengths(lengths: torch.Tensor, max_time: int) -> torch.Tensor:
    return torch.arange(max_time, device=lengths.device).unsqueeze(0) < lengths.unsqueeze(1)


def smooth_batch(
    x: torch.Tensor,
    input_lengths: torch.Tensor,
    *,
    sigma_ms: float,
    active_bin_size_ms: int,
    kernel_size_ms: float,
    threshold: float,
) -> torch.Tensor:
    sigma_bins = float(sigma_ms) / float(int(active_bin_size_ms))
    if sigma_bins <= 0.0 or int(x.shape[1]) <= 1:
        return x
    kernel_size_bins = max(1, int(math.ceil(float(kernel_size_ms) / float(int(active_bin_size_ms)))))
    kernel = _gaussian_kernel_1d(
        sigma_bins=sigma_bins,
        kernel_size_bins=int(kernel_size_bins),
        threshold=float(threshold),
        device=x.device,
        dtype=x.dtype,
    )
    channels = int(x.shape[-1])
    weight = kernel.view(1, 1, -1).expand(channels, 1, -1)
    smoothed = torch.nn.functional.conv1d(
        x.transpose(1, 2),
        weight,
        padding=int(kernel.numel() // 2),
        groups=channels,
    ).transpose(1, 2)
    valid = _sequence_mask_from_lengths(input_lengths.to(x.device), int(x.shape[1]))
    return smoothed * valid.unsqueeze(-1).to(smoothed.dtype)


def prepare_timestep_flexible_inputs(
    x: torch.Tensor,
    input_lengths: torch.Tensor,
    *,
    config: TimestepFlexibleInputTransformConfig,
    active_bin_size_ms: int,
    is_training: bool,
) -> torch.Tensor:
    transformed = x
    if is_training and float(config.white_noise_sd) > 0.0:
        transformed = transformed + torch.randn_like(transformed) * float(config.white_noise_sd)
    if is_training and float(config.constant_offset_sd) > 0.0:
        transformed = transformed + torch.randn(
            (int(transformed.shape[0]), 1, int(transformed.shape[2])),
            device=transformed.device,
            dtype=transformed.dtype,
        ) * float(config.constant_offset_sd)
    return smooth_batch(
        transformed,
        input_lengths,
        sigma_ms=float(config.input_smoothing_sigma_ms),
        active_bin_size_ms=int(active_bin_size_ms),
        kernel_size_ms=float(config.input_smoothing_kernel_size_ms),
        threshold=float(config.input_smoothing_threshold),
    )


def build_timestep_flexible_problem(
    *,
    cache_root: str | Path,
    dataset: str,
    feature_mode: str,
    boundary_key_mode: str,
    split_policy: str = "competition_train_test",
    cv_num_folds: int = 5,
    cv_fold_index: int = 0,
) -> dict[str, Any]:
    return build_willett_problem(
        cache_root=Path(cache_root),
        dataset=str(dataset),
        feature_mode=str(feature_mode),
        boundary_key_mode=str(boundary_key_mode),
        split_policy=str(split_policy),
        cv_num_folds=int(cv_num_folds),
        cv_fold_index=int(cv_fold_index),
    )


class RebinnedSequenceDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        rows: tuple[Any, ...] | list[Any],
        *,
        cache_root: Path,
        stats: dict[str, tuple[np.ndarray, np.ndarray]] | tuple[np.ndarray, np.ndarray] | None = None,
        feature_mode: str = "tx_only",
        boundary_key_mode: str = "session",
        dataset: str = "brain2text24",
        active_bin_size_ms: int = CANONICAL_BIN_SIZE_MS,
    ) -> None:
        self.rows = list(rows)
        self.stats = stats
        self.feature_mode = str(feature_mode)
        self.boundary_key_mode = str(boundary_key_mode)
        self.dataset = str(dataset)
        self.signal_spec = (
            signal_spec_for_rows(self.rows, feature_mode=self.feature_mode)
            if self.rows
            else signal_spec_for_cache(
                cache_root,
                dataset=self.dataset,
                feature_mode=self.feature_mode,
            )
        )
        self.active_bin_size_ms = int(active_bin_size_ms)
        self._base = CanonicalSequenceDataset(
            self.rows,
            cache_root=cache_root,
            signal_spec=self.signal_spec,
            stats=None,
            boundary_key_mode=str(boundary_key_mode),
            dataset=str(dataset),
        )
        self._accessor = self._base._accessor

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.rows[idx]
        x = self._accessor.load_features(row, signal_spec=self.signal_spec)
        x = rebin_features(x, bin_size_ms=self.active_bin_size_ms)
        if self.stats is not None:
            x = apply_feature_stats(x, row=row, stats=self.stats)
        labels = self._accessor.load_labels(row)
        if labels is None:
            labels = np.zeros((0,), dtype=np.int64)
        else:
            labels = np.array(labels, dtype=np.int64, copy=True)
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


def compute_rebinned_normalization_stats(
    rows: tuple[Any, ...] | list[Any],
    *,
    cache_root: Path,
    feature_mode: str,
    mode: str,
    bin_size_ms: int,
) -> dict[str, tuple[np.ndarray, np.ndarray]] | tuple[np.ndarray, np.ndarray] | None:
    resolved_mode = str(mode)
    if resolved_mode == "none":
        return None
    signal_spec = signal_spec_for_rows(rows, feature_mode=str(feature_mode))
    accessor = CanonicalSequenceDataset(
        rows,
        cache_root=cache_root,
        signal_spec=signal_spec,
        stats=None,
    )._accessor
    try:
        if resolved_mode == "global":
            total_count = 0
            sum_x = None
            sum_x2 = None
            for row in rows:
                x = rebin_features(
                    accessor.load_features(row, signal_spec=signal_spec),
                    bin_size_ms=bin_size_ms,
                )
                if int(x.shape[0]) <= 0:
                    continue
                x64 = x.astype(np.float64, copy=False)
                if sum_x is None:
                    sum_x = x64.sum(axis=0)
                    sum_x2 = np.square(x64).sum(axis=0)
                else:
                    sum_x += x64.sum(axis=0)
                    sum_x2 += np.square(x64).sum(axis=0)
                total_count += int(x.shape[0])
            if sum_x is None or sum_x2 is None or total_count <= 0:
                raise ValueError("Cannot compute global stats for an empty rebinned dataset.")
            mean = sum_x / total_count
            var = np.maximum(sum_x2 / total_count - np.square(mean), 1e-6)
            return (
                mean.astype(np.float32, copy=False),
                np.sqrt(var).astype(np.float32, copy=False),
            )
        if resolved_mode in {"per_session", "block"}:
            grouped: dict[str, list[Any]] = {}
            for row in rows:
                key = str(row.session_id) if resolved_mode == "per_session" else normalization_key_for_row(row)
                grouped.setdefault(key, []).append(row)
            stats: dict[str, tuple[np.ndarray, np.ndarray]] = {}
            for key, group_rows in grouped.items():
                total_count = 0
                sum_x = None
                sum_x2 = None
                for row in group_rows:
                    x = rebin_features(
                        accessor.load_features(row, signal_spec=signal_spec),
                        bin_size_ms=bin_size_ms,
                    )
                    if int(x.shape[0]) <= 0:
                        continue
                    x64 = x.astype(np.float64, copy=False)
                    if sum_x is None:
                        sum_x = x64.sum(axis=0)
                        sum_x2 = np.square(x64).sum(axis=0)
                    else:
                        sum_x += x64.sum(axis=0)
                        sum_x2 += np.square(x64).sum(axis=0)
                    total_count += int(x.shape[0])
                if sum_x is None or sum_x2 is None or total_count <= 0:
                    raise ValueError(f"Cannot compute rebinned stats for empty group {key!r}.")
                mean = sum_x / total_count
                var = np.maximum(sum_x2 / total_count - np.square(mean), 1e-6)
                stats[str(key)] = (
                    mean.astype(np.float32, copy=False),
                    np.sqrt(var).astype(np.float32, copy=False),
                )
            return stats
        raise ValueError("mode must be one of {'block', 'global', 'per_session', 'none'}")
    finally:
        accessor.close()


def make_length_aware_batch_sampler(
    rows: tuple[Any, ...] | list[Any],
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
    bin_size_ms: int = CANONICAL_BIN_SIZE_MS,
) -> LengthAwareBatchSampler:
    sampler_rows = tuple(
        replace(
            row,
            n_time_bins=max(
                1,
                rebinned_input_length(
                    int(getattr(row, "n_time_bins", 0) or 0),
                    bin_size_ms=int(bin_size_ms),
                ),
            ),
        )
        for row in rows
    )
    lengths = [max(1, int(getattr(row, "n_time_bins", 0))) for row in sampler_rows]
    sorted_lengths = sorted(lengths)
    percentile_index = min(
        max(int(round(0.95 * max(len(sorted_lengths) - 1, 0))), 0),
        max(len(sorted_lengths) - 1, 0),
    )
    p95_train_input_length = int(sorted_lengths[percentile_index]) if sorted_lengths else 1
    max_examples_per_microbatch = int(batch_size)
    max_padded_time_per_microbatch = int(max_examples_per_microbatch * p95_train_input_length)
    return LengthAwareBatchSampler(
        sampler_rows,
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


__all__ = [
    "CANONICAL_BIN_SIZE_MS",
    "TimestepFlexibleInputTransformConfig",
    "RebinnedSequenceDataset",
    "adapter_keys_from_rows",
    "build_timestep_flexible_problem",
    "collate_sequence_batch",
    "compute_rebinned_normalization_stats",
    "group_rows_by_adapter_key",
    "loader_kwargs",
    "make_length_aware_batch_sampler",
    "normalization_stats_missing_rows",
    "prepare_timestep_flexible_inputs",
    "rebin_factor_for_bin_size",
    "rebin_features",
    "rebinned_input_length",
    "resolve_patch_bins",
]

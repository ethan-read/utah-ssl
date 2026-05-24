"""Native cache data helpers for Willett reconstruction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

try:
    from masked_ssl.cache import (
        resolve_boundary_key,
    )
    from masked_ssl.probe import (
        CanonicalSequenceDataset,
        LengthAwareBatchSampler,
        build_competition_split_problem,
        canonical_rows_padded_time_percentile,
        collate_sequence_batch,
        compute_feature_stats,
    )
except ModuleNotFoundError:  # pragma: no cover - repo-root unittest fallback
    from analysis.active.ssl_experiments.masked_ssl.cache import (
        resolve_boundary_key,
    )
    from analysis.active.ssl_experiments.masked_ssl.probe import (
        CanonicalSequenceDataset,
        LengthAwareBatchSampler,
        build_competition_split_problem,
        canonical_rows_padded_time_percentile,
        collate_sequence_batch,
        compute_feature_stats,
    )


@dataclass(frozen=True)
class WillettInputTransformConfig:
    input_smoothing_sigma_bins: float = 2.0
    input_smoothing_kernel_size: int = 100
    input_smoothing_threshold: float = 0.01
    white_noise_sd: float = 1.0
    constant_offset_sd: float = 0.2


def normalization_key_for_row(row: Any) -> str:
    block_num = getattr(row, "block_num", None)
    if block_num is not None:
        return f"{row.session_id}::block:{int(block_num)}"
    normalization_group = getattr(row, "normalization_group", None)
    if normalization_group is not None:
        return str(normalization_group)
    return str(row.session_id)


def normalization_stats_missing_rows(
    stats: dict[str, tuple[np.ndarray, np.ndarray]] | tuple[np.ndarray, np.ndarray] | None,
    rows: tuple[Any, ...] | list[Any],
) -> list[str]:
    if stats is None or not isinstance(stats, dict):
        return []
    missing: list[str] = []
    for row in rows:
        candidate_keys: list[str] = []
        block_num = getattr(row, "block_num", None)
        if block_num is not None:
            candidate_keys.append(f"{row.session_id}::block:{int(block_num)}")
        normalization_group = getattr(row, "normalization_group", None)
        if normalization_group is not None:
            candidate_keys.append(str(normalization_group))
        candidate_keys.append(str(row.session_id))
        if any(candidate_key in stats for candidate_key in candidate_keys):
            continue
        missing.append(str(getattr(row, "example_id", row)))
    return missing


def adapter_keys_from_rows(
    rows: tuple[Any, ...] | list[Any],
    *,
    dataset: str,
    boundary_key_mode: str,
) -> tuple[str, ...]:
    keys: list[str] = []
    seen: set[str] = set()
    for row in rows:
        key = resolve_boundary_key(
            dataset=str(dataset),
            session_id=str(row.session_id),
            subject_id=None if getattr(row, "subject_id", None) is None else str(row.subject_id),
            boundary_key_mode=str(boundary_key_mode),
        )
        if key in seen:
            continue
        seen.add(key)
        keys.append(key)
    return tuple(keys)


def group_rows_by_adapter_key(
    rows: tuple[Any, ...] | list[Any],
    *,
    dataset: str,
    boundary_key_mode: str,
) -> dict[str, tuple[Any, ...]]:
    grouped: dict[str, list[Any]] = {}
    for row in rows:
        key = resolve_boundary_key(
            dataset=str(dataset),
            session_id=str(row.session_id),
            subject_id=None if getattr(row, "subject_id", None) is None else str(row.subject_id),
            boundary_key_mode=str(boundary_key_mode),
        )
        grouped.setdefault(key, []).append(row)
    return {key: tuple(group_rows) for key, group_rows in grouped.items()}


def _willett_gaussian_kernel_1d(
    *,
    sigma_bins: float,
    kernel_size: int,
    threshold: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    sigma = float(sigma_bins)
    if sigma <= 0.0:
        return torch.ones((1,), device=device, dtype=dtype)
    positions = torch.arange(int(kernel_size), device=device, dtype=dtype) - float(int(kernel_size) // 2)
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


def smooth_batch_like_willett(
    x: torch.Tensor,
    input_lengths: torch.Tensor,
    *,
    sigma_bins: float,
    kernel_size: int,
    threshold: float,
) -> torch.Tensor:
    if float(sigma_bins) <= 0.0 or int(x.shape[1]) <= 1:
        return x
    kernel = _willett_gaussian_kernel_1d(
        sigma_bins=float(sigma_bins),
        kernel_size=int(kernel_size),
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


def prepare_willett_inputs(
    x: torch.Tensor,
    input_lengths: torch.Tensor,
    *,
    config: WillettInputTransformConfig,
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
    return smooth_batch_like_willett(
        transformed,
        input_lengths,
        sigma_bins=float(config.input_smoothing_sigma_bins),
        kernel_size=int(config.input_smoothing_kernel_size),
        threshold=float(config.input_smoothing_threshold),
    )


def build_willett_problem(
    *,
    cache_root: str | Path,
    dataset: str,
    feature_mode: str,
    boundary_key_mode: str,
) -> dict[str, Any]:
    return build_competition_split_problem(
        cache_root=Path(cache_root),
        dataset=str(dataset),
        feature_mode=str(feature_mode),
        boundary_key_mode=str(boundary_key_mode),
    )


def compute_willett_normalization_stats(
    rows: tuple[Any, ...] | list[Any],
    *,
    cache_root: Path,
    feature_mode: str,
    mode: str,
) -> dict[str, tuple[np.ndarray, np.ndarray]] | tuple[np.ndarray, np.ndarray] | None:
    resolved_mode = str(mode)
    if resolved_mode == "none":
        return None
    if resolved_mode == "global":
        return compute_feature_stats(
            rows,
            cache_root=cache_root,
            mode="global",
            feature_mode=feature_mode,
        )
    if resolved_mode == "per_session":
        return compute_feature_stats(
            rows,
            cache_root=cache_root,
            mode="per_session",
            feature_mode=feature_mode,
        )
    if resolved_mode != "block":
        raise ValueError("mode must be one of {'block', 'global', 'per_session', 'none'}")

    accessor = CanonicalSequenceDataset(rows, cache_root=cache_root, stats=None, feature_mode=feature_mode)._accessor
    try:
        grouped: dict[str, list[Any]] = {}
        for row in rows:
            grouped.setdefault(normalization_key_for_row(row), []).append(row)
        stats: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for key, group_rows in grouped.items():
            total_count = 0
            sum_x = None
            sum_x2 = None
            for row in group_rows:
                x = accessor.load_features(row, feature_mode=feature_mode)
                x64 = x.astype(np.float64, copy=False)
                if sum_x is None:
                    sum_x = x64.sum(axis=0)
                    sum_x2 = np.square(x64).sum(axis=0)
                else:
                    sum_x += x64.sum(axis=0)
                    sum_x2 += np.square(x64).sum(axis=0)
                total_count += int(x.shape[0])
            if sum_x is None or sum_x2 is None or total_count <= 0:
                raise ValueError(f"Cannot compute block stats for empty group {key!r}.")
            mean = sum_x / total_count
            var = np.maximum(sum_x2 / total_count - np.square(mean), 1e-6)
            stats[str(key)] = (
                mean.astype(np.float32, copy=False),
                np.sqrt(var).astype(np.float32, copy=False),
            )
        return stats
    finally:
        accessor.close()

def make_length_aware_batch_sampler(
    rows: tuple[Any, ...] | list[Any],
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


__all__ = [
    "CanonicalSequenceDataset",
    "WillettInputTransformConfig",
    "adapter_keys_from_rows",
    "build_willett_problem",
    "compute_willett_normalization_stats",
    "collate_sequence_batch",
    "compute_feature_stats",
    "group_rows_by_adapter_key",
    "loader_kwargs",
    "make_length_aware_batch_sampler",
    "normalization_key_for_row",
    "normalization_stats_missing_rows",
    "prepare_willett_inputs",
    "smooth_batch_like_willett",
]

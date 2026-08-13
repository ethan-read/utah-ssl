"""Shared Willett-style preprocessing for neural speech decoding.

This is the repository's adapted implementation of the smoothing and training
augmentation recipe used by its Willett-derived decoder path. It is not a
universal Utah-array preprocessing default or an official Stanford
implementation. The exact upstream code provenance remains unverified; see
``experiments/supervised_baselines/PROVENANCE.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch


class WillettInputTransform(Protocol):
    input_smoothing_sigma_bins: float
    input_smoothing_kernel_size: int
    input_smoothing_threshold: float
    white_noise_sd: float
    constant_offset_sd: float


@dataclass(frozen=True)
class WillettInputTransformConfig:
    input_smoothing_sigma_bins: float = 2.0
    input_smoothing_kernel_size: int = 100
    input_smoothing_threshold: float = 0.01
    white_noise_sd: float = 1.0
    constant_offset_sd: float = 0.2


def willett_input_transform_config_from(
    value: WillettInputTransform,
) -> WillettInputTransformConfig:
    return WillettInputTransformConfig(
        input_smoothing_sigma_bins=float(value.input_smoothing_sigma_bins),
        input_smoothing_kernel_size=int(value.input_smoothing_kernel_size),
        input_smoothing_threshold=float(value.input_smoothing_threshold),
        white_noise_sd=float(value.white_noise_sd),
        constant_offset_sd=float(value.constant_offset_sd),
    )


def willett_gaussian_kernel_1d(
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
    kernel_size = int(kernel_size)
    if kernel_size <= 0:
        raise ValueError("kernel_size must be positive")
    center = int(kernel_size // 2)
    positions = torch.arange(kernel_size, device=device, dtype=dtype) - float(center)
    kernel = torch.exp(-0.5 * (positions / sigma).pow(2))
    kernel = kernel / kernel.sum().clamp_min(1e-8)
    keep = kernel > float(threshold)
    if not bool(keep.any().item()):
        keep[center] = True
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
    kernel = willett_gaussian_kernel_1d(
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
    config: WillettInputTransform,
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


__all__ = [
    "WillettInputTransform",
    "WillettInputTransformConfig",
    "prepare_willett_inputs",
    "smooth_batch_like_willett",
    "willett_gaussian_kernel_1d",
    "willett_input_transform_config_from",
]

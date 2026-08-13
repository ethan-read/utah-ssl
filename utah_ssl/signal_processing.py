"""Model-independent signal processing for Utah-array cache construction."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def gaussian_kernel_1d(
    sigma_bins: float,
    *,
    device: torch.device,
    dtype: torch.dtype,
    radius: int | None = None,
) -> torch.Tensor:
    """Build a normalized Gaussian kernel with an optional explicit radius."""

    sigma = float(sigma_bins)
    if sigma <= 0.0:
        return torch.ones((1,), device=device, dtype=dtype)
    effective_radius = (
        int(radius) if radius is not None else max(1, int(math.ceil(4.0 * sigma)))
    )
    positions = torch.arange(
        -effective_radius,
        effective_radius + 1,
        device=device,
        dtype=dtype,
    )
    kernel = torch.exp(-0.5 * (positions / sigma).pow(2))
    return kernel / kernel.sum().clamp_min(1e-8)


def apply_gaussian_smoothing(
    x_seq: torch.Tensor,
    feature_mask: torch.Tensor,
    *,
    sigma_bins: float,
) -> torch.Tensor:
    """Smooth present features over time using reflection padding.

    This is the cache-construction smoother. Willett-style decoder smoothing
    has different kernel truncation and padding semantics and remains in
    ``utah_ssl.decoding_preprocessing``.
    """

    sigma = float(sigma_bins)
    if sigma <= 0.0 or x_seq.shape[0] <= 1:
        return x_seq
    present_idx = torch.nonzero(feature_mask.bool(), as_tuple=False).squeeze(1)
    if present_idx.numel() == 0:
        return x_seq

    max_reflect_radius = int(x_seq.shape[0] - 1)
    if max_reflect_radius <= 0:
        return x_seq
    kernel_radius = min(max(1, int(math.ceil(4.0 * sigma))), max_reflect_radius)
    kernel = gaussian_kernel_1d(
        sigma,
        device=x_seq.device,
        dtype=x_seq.dtype,
        radius=kernel_radius,
    )
    selected = x_seq[:, present_idx].transpose(0, 1).unsqueeze(0)
    padded = F.pad(selected, (kernel_radius, kernel_radius), mode="reflect")
    weight = kernel.view(1, 1, -1).expand(selected.shape[1], 1, -1)
    smoothed = F.conv1d(
        padded,
        weight,
        groups=selected.shape[1],
    ).squeeze(0).transpose(0, 1)

    out = x_seq.clone()
    out[:, present_idx] = smoothed
    return out


__all__ = ["apply_gaussian_smoothing", "gaussian_kernel_1d"]

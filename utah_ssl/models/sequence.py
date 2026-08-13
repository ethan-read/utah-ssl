"""Masking and reversal operations shared by padded sequence models."""

from __future__ import annotations

import torch


def sequence_mask(lengths: torch.Tensor, max_length: int) -> torch.Tensor:
    positions = torch.arange(int(max_length), device=lengths.device).unsqueeze(0)
    return positions < lengths.unsqueeze(1)


def apply_sequence_mask(x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    if x.shape[1] == 0:
        return x
    mask = sequence_mask(lengths.to(x.device), x.shape[1]).unsqueeze(-1)
    return x * mask.to(x.dtype)


def reverse_padded_sequence(x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """Reverse each valid sequence prefix while leaving padding aligned."""
    if x.ndim < 2:
        raise ValueError("reverse_padded_sequence expects a tensor with shape (B, T, ...).")
    positions = torch.arange(x.shape[1], device=lengths.device).unsqueeze(0)
    valid = positions < lengths.unsqueeze(1)
    reversed_positions = (lengths.unsqueeze(1) - 1 - positions).clamp_min(0)
    gather_positions = torch.where(valid, reversed_positions, positions)
    view_shape = (*gather_positions.shape, *([1] * (x.ndim - 2)))
    return x.gather(dim=1, index=gather_positions.view(view_shape).expand_as(x))


__all__ = ["apply_sequence_mask", "reverse_padded_sequence", "sequence_mask"]

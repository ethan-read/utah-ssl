"""Canonical temporal patching utilities.

Two patch policies are intentionally supported because the existing experiment
families made different choices:

``floor``
    Willett/POSSM-style patching. Uses starts ``0, stride, ...`` and stops
    before a trailing partial window unless the sequence is shorter than one
    patch.

``cover_tail``
    Legacy masked/contrastive SSL behavior. Adds a final patch at
    ``length - patch_size`` when needed so the tail of a positive-length
    sequence is represented.
"""

from __future__ import annotations

from typing import Literal

import torch

PatchPolicy = Literal["floor", "cover_tail"]


def _validate_patch_args(patch_size: int, patch_stride: int, policy: str) -> tuple[int, int, PatchPolicy]:
    resolved_size = int(patch_size)
    resolved_stride = int(patch_stride)
    if resolved_size <= 0 or resolved_stride <= 0:
        raise ValueError("patch_size and patch_stride must be positive")
    if policy not in {"floor", "cover_tail"}:
        raise ValueError("policy must be one of {'floor', 'cover_tail'}")
    return resolved_size, resolved_stride, policy  # type: ignore[return-value]


def patch_starts(
    length: int,
    *,
    patch_size: int,
    patch_stride: int,
    policy: PatchPolicy = "floor",
) -> list[int]:
    patch_size, patch_stride, policy = _validate_patch_args(patch_size, patch_stride, policy)
    resolved_length = int(length)
    if resolved_length <= 0:
        return []
    if policy == "floor":
        if resolved_length <= patch_size:
            return [0]
        total_patches = 1 + ((resolved_length - patch_size) // patch_stride)
        return [idx * patch_stride for idx in range(total_patches)]

    if patch_size == 1:
        return list(range(resolved_length))
    max_start = max(resolved_length - patch_size, 0)
    starts = list(range(0, max(resolved_length - patch_size + 1, 1), patch_stride))
    if not starts:
        starts = [0]
    if starts[-1] != max_start:
        starts.append(max_start)
    return starts


def patched_length(
    length: int,
    *,
    patch_size: int,
    patch_stride: int,
    policy: PatchPolicy = "floor",
) -> int:
    return len(
        patch_starts(
            int(length),
            patch_size=int(patch_size),
            patch_stride=int(patch_stride),
            policy=policy,
        )
    )


def patched_lengths(
    lengths: torch.Tensor,
    *,
    patch_size: int,
    patch_stride: int,
    policy: PatchPolicy = "floor",
) -> torch.Tensor:
    _validate_patch_args(patch_size, patch_stride, policy)
    values = [
        patched_length(
            int(length),
            patch_size=int(patch_size),
            patch_stride=int(patch_stride),
            policy=policy,
        )
        for length in lengths.detach().cpu().tolist()
    ]
    return torch.as_tensor(values, device=lengths.device, dtype=torch.long)


def patch_batch(
    x: torch.Tensor,
    lengths: torch.Tensor,
    *,
    patch_size: int,
    patch_stride: int,
    policy: PatchPolicy = "floor",
) -> tuple[torch.Tensor, torch.Tensor]:
    if x.ndim != 3:
        raise ValueError(f"Expected x to have shape [B, T, D], got {tuple(x.shape)}")
    patch_size, patch_stride, policy = _validate_patch_args(patch_size, patch_stride, policy)
    if int(lengths.numel()) != int(x.shape[0]):
        raise ValueError("lengths must have one entry per batch item")

    token_sequences: list[torch.Tensor] = []
    token_lengths: list[int] = []
    feature_dim = int(x.shape[-1])
    patch_dim = feature_dim * patch_size
    for sample, length_tensor in zip(x, lengths):
        length = max(0, int(length_tensor.item()))
        starts = patch_starts(
            length,
            patch_size=patch_size,
            patch_stride=patch_stride,
            policy=policy,
        )
        if not starts:
            tokens = sample.new_zeros((0, patch_dim))
        else:
            valid = sample[:length]
            patches: list[torch.Tensor] = []
            for start in starts:
                patch = valid[start : start + patch_size]
                if int(patch.shape[0]) < patch_size:
                    pad = valid.new_zeros((patch_size - int(patch.shape[0]), feature_dim))
                    patch = torch.cat([patch, pad], dim=0)
                patches.append(patch.reshape(-1))
            tokens = torch.stack(patches, dim=0)
        token_sequences.append(tokens)
        token_lengths.append(int(tokens.shape[0]))

    max_tokens = max(token_lengths, default=0)
    patched = x.new_zeros((int(x.shape[0]), max_tokens, patch_dim))
    for batch_idx, tokens in enumerate(token_sequences):
        if int(tokens.shape[0]) > 0:
            patched[batch_idx, : int(tokens.shape[0])] = tokens
    return patched, torch.as_tensor(token_lengths, device=lengths.device, dtype=torch.long)


def causal_conv_lengths(lengths: torch.Tensor, *, stride: int) -> torch.Tensor:
    resolved_stride = int(stride)
    if resolved_stride <= 0:
        raise ValueError("stride must be positive")
    lengths = lengths.to(dtype=torch.long)
    positive = lengths > 0
    safe = torch.clamp(lengths - 1, min=0)
    output = torch.div(safe, resolved_stride, rounding_mode="floor") + 1
    return torch.where(positive, output, torch.zeros_like(output))


__all__ = [
    "PatchPolicy",
    "causal_conv_lengths",
    "patch_batch",
    "patch_starts",
    "patched_length",
    "patched_lengths",
]

"""Masked time/channel reconstruction objective for generic SSM SSL."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F

from utah_ssl.patching import patched_lengths

from .model import GenericMaskedSSMModel


def _valid_token_mask(token_lengths: torch.Tensor, max_tokens: int) -> torch.Tensor:
    positions = torch.arange(max_tokens, device=token_lengths.device).unsqueeze(0)
    return positions < token_lengths.unsqueeze(1)


def build_time_channel_mask(
    token_lengths: torch.Tensor,
    *,
    token_dim: int,
    time_mask_ratio: float,
    channel_mask_ratio: float,
    chunk_size: int,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    if int(token_lengths.numel()) == 0:
        return torch.zeros((0, 0, int(token_dim)), dtype=torch.bool, device=token_lengths.device)
    max_tokens = int(token_lengths.max().item())
    token_dim = int(token_dim)
    mask = torch.zeros(
        (int(token_lengths.numel()), max_tokens, token_dim),
        dtype=torch.bool,
        device=token_lengths.device,
    )
    chunk_size = max(1, int(chunk_size))
    for batch_idx, length_tensor in enumerate(token_lengths.tolist()):
        length = int(length_tensor)
        if length <= 0:
            continue
        if float(time_mask_ratio) > 0.0:
            masked_tokens = max(1, int(math.ceil(length * float(time_mask_ratio))))
            num_chunks = max(1, int(math.ceil(masked_tokens / chunk_size)))
            max_start = max(1, length)
            starts = torch.randint(
                low=0,
                high=max_start,
                size=(num_chunks,),
                generator=generator,
                device=token_lengths.device,
            )
            for start_tensor in starts.tolist():
                start = int(start_tensor)
                stop = min(length, start + chunk_size)
                mask[batch_idx, start:stop, :] = True
        if float(channel_mask_ratio) > 0.0:
            masked_channels = max(1, int(math.ceil(token_dim * float(channel_mask_ratio))))
            channel_ids = torch.randperm(
                token_dim,
                generator=generator,
                device=token_lengths.device,
            )[:masked_channels]
            mask[batch_idx, :length, channel_ids] = True
        if not bool(mask[batch_idx, :length].any().item()):
            mask[batch_idx, 0, 0] = True
    valid = _valid_token_mask(token_lengths, max_tokens).unsqueeze(-1)
    return mask & valid


def masked_reconstruction_loss(
    model: GenericMaskedSSMModel,
    batch: dict[str, Any],
    *,
    device: torch.device,
    time_mask_ratio: float,
    channel_mask_ratio: float,
    chunk_size: int,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    x = batch["x"].to(device)
    lengths = (batch.get("lengths") if "lengths" in batch else batch["input_lengths"]).to(device)
    tokens, token_lengths = model.encoder.tokenize(x, lengths)
    corruption_mask = build_time_channel_mask(
        token_lengths,
        token_dim=int(tokens.shape[-1]),
        time_mask_ratio=float(time_mask_ratio),
        channel_mask_ratio=float(channel_mask_ratio),
        chunk_size=int(chunk_size),
        generator=generator,
    )
    outputs = model.forward_tokens(tokens, token_lengths, corruption_mask=corruption_mask)
    mask_float = corruption_mask.to(tokens.dtype)
    denom = mask_float.sum().clamp_min(1.0)
    loss = F.mse_loss(outputs["reconstruction"], tokens, reduction="none")
    masked_loss = (loss * mask_float).sum() / denom
    with torch.no_grad():
        valid_tokens = int(token_lengths.sum().item())
        masked_entries = int(corruption_mask.sum().item())
        max_entries = max(1, valid_tokens * int(tokens.shape[-1]))
        metrics = {
            "loss": float(masked_loss.detach().item()),
            "masked_entry_fraction": float(masked_entries / max_entries),
            "mean_token_length": float(token_lengths.float().mean().item()),
        }
    return masked_loss, metrics


def expected_token_lengths_for_config(config: Any, input_lengths: torch.Tensor) -> torch.Tensor:
    if str(config.input_mode) == "temporal_patch":
        return patched_lengths(
            input_lengths,
            patch_size=int(config.patch_size),
            patch_stride=int(config.patch_stride),
            policy=str(config.patch_policy),
        )
    if str(config.input_mode) == "causal_conv_stem":
        from utah_ssl.patching import causal_conv_lengths

        return causal_conv_lengths(input_lengths, stride=int(config.conv_stride))
    return input_lengths.to(dtype=torch.long)


__all__ = [
    "build_time_channel_mask",
    "expected_token_lengths_for_config",
    "masked_reconstruction_loss",
]

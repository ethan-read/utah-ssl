"""Masked-reconstruction objectives and shared metric helpers."""

from __future__ import annotations

import random
from typing import Any

import torch

from .model import MaskedSSLModel


def summarize_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    summary = {
        "loss": float(metrics["loss"].detach().cpu().item()),
        "actual_mask_ratio": float(metrics["actual_mask_ratio"]),
        "masked_token_ratio": float(metrics["masked_token_ratio"]),
        "masked_element_ratio": float(metrics["masked_element_ratio"]),
        "masked_token_count": int(metrics["masked_token_count"]),
        "masked_element_count": int(metrics["masked_element_count"]),
        "masked_token_full_patch_mse": float(metrics["masked_token_full_patch_mse"]),
        "masked_prediction_mean": float(metrics["masked_prediction_mean"]),
        "masked_prediction_std": float(metrics["masked_prediction_std"]),
        "masked_target_mean": float(metrics["masked_target_mean"]),
        "masked_target_std": float(metrics["masked_target_std"]),
    }
    if "patch_fraction_weighted_mse" in metrics:
        summary["patch_fraction_weighted_mse"] = float(metrics["patch_fraction_weighted_mse"])
    return summary


def _resolve_target_count(length: int, mask_ratio: float) -> int:
    if length <= 0:
        return 0
    return min(length, max(1, int(round(float(mask_ratio) * length))))


def _unmasked_segments(mask: torch.Tensor) -> list[tuple[int, int]]:
    segments: list[tuple[int, int]] = []
    start: int | None = None
    for idx in range(mask.shape[0] + 1):
        masked = True if idx == mask.shape[0] else bool(mask[idx].item())
        if not masked and start is None:
            start = idx
        if masked and start is not None:
            segments.append((start, idx))
            start = None
    return segments


def sample_mask_indices(
    *,
    length: int,
    mask_ratio: float,
    span_length_min: int,
    span_length_max: int,
    num_spans_mode: str,
) -> torch.Tensor:
    if num_spans_mode not in {"one", "multiple"}:
        raise ValueError("num_spans_mode must be one of {'one', 'multiple'}")
    if length <= 0:
        return torch.zeros((0,), dtype=torch.bool)

    target_count = _resolve_target_count(length, mask_ratio)
    span_min = max(1, min(int(span_length_min), length))
    span_max = max(span_min, min(int(span_length_max), length))
    mask = torch.zeros((length,), dtype=torch.bool)

    if num_spans_mode == "one":
        if target_count < span_min or target_count > span_max:
            raise ValueError(
                "num_spans_mode='one' requires the exact target mask count to fit inside the "
                f"configured span bounds. target_count={target_count}, "
                f"feasible_span_range=[{span_min}, {span_max}]. "
                "Increase span_length_max, reduce mask_ratio, or use num_spans_mode='multiple'."
            )
        span_length = target_count
        start = random.randint(0, max(length - span_length, 0)) if length > span_length else 0
        mask[start : start + span_length] = True
        return mask

    while int(mask.sum().item()) < target_count:
        segments = _unmasked_segments(mask)
        if not segments:
            break

        remaining = target_count - int(mask.sum().item())
        feasible_segments = [(start, stop) for start, stop in segments if stop - start > 0]
        if not feasible_segments:
            break
        seg_start, seg_stop = random.choice(feasible_segments)
        seg_len = seg_stop - seg_start

        span_cap = min(span_max, remaining, seg_len)
        if span_cap <= 0:
            break
        span_floor = min(span_min, span_cap)
        span_length = random.randint(span_floor, span_cap)
        start = random.randint(seg_start, seg_stop - span_length)
        mask[start : start + span_length] = True

    return mask


def _patch_feature_presence(
    model: MaskedSSLModel,
    batch: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor]:
    presence_seq = batch["feature_mask"].to(batch["x"].dtype).unsqueeze(1).expand(
        -1,
        batch["x"].shape[1],
        -1,
    )
    token_feature_mask, token_lengths = model.encoder.patch_batch(
        presence_seq,
        batch["lengths"],
    )
    return token_feature_mask > 0.5, token_lengths


def _span_start_mask(mask: torch.Tensor) -> torch.Tensor:
    span_starts = mask.clone()
    if mask.shape[1] > 1:
        span_starts[:, 1:] = span_starts[:, 1:] & ~mask[:, :-1]
    return span_starts


def build_masked_batch(
    model: MaskedSSLModel,
    batch: dict[str, Any],
    *,
    mask_unit: str,
    mask_ratio: float,
    span_length_min: int,
    span_length_max: int,
    num_spans_mode: str,
    allow_bin_fractional_overlap: bool,
) -> dict[str, Any]:
    if mask_unit not in {"patch", "bin"}:
        raise ValueError("mask_unit must be one of {'patch', 'bin'}")

    x = batch["x"]
    lengths = batch["lengths"]
    tokens, token_lengths = model.encoder.patch_batch(x, lengths)
    token_feature_mask, feature_token_lengths = _patch_feature_presence(model, batch)
    if not torch.equal(token_lengths, feature_token_lengths):
        raise ValueError("Feature-mask patching must preserve token lengths.")

    valid_token_mask = (
        torch.arange(tokens.shape[1], device=token_lengths.device).unsqueeze(0)
        < token_lengths.unsqueeze(1)
    )
    token_mask = torch.zeros_like(valid_token_mask)
    token_loss_mask = torch.zeros_like(tokens, dtype=torch.bool)
    token_loss_token_mask = torch.zeros_like(valid_token_mask)

    if mask_unit == "patch":
        raw_unit_mask = torch.zeros_like(valid_token_mask)
        for sample_idx, token_length in enumerate(token_lengths.tolist()):
            raw_unit_mask[sample_idx, :token_length] = sample_mask_indices(
                length=int(token_length),
                mask_ratio=mask_ratio,
                span_length_min=span_length_min,
                span_length_max=span_length_max,
                num_spans_mode=num_spans_mode,
            )
        token_mask = raw_unit_mask & valid_token_mask
        token_loss_token_mask = _span_start_mask(token_mask)
        token_loss_mask = token_feature_mask & token_loss_token_mask.unsqueeze(-1)
        token_overlap_fraction = token_mask.to(tokens.dtype)
        raw_unit_count = int(raw_unit_mask.sum().item())
        total_unit_count = int(token_lengths.sum().item())
        raw_bin_mask = None
    else:
        raw_bin_mask = torch.zeros(
            batch["x"].shape[:2],
            device=batch["x"].device,
            dtype=torch.bool,
        )
        for sample_idx, length in enumerate(lengths.tolist()):
            raw_bin_mask[sample_idx, :length] = sample_mask_indices(
                length=int(length),
                mask_ratio=mask_ratio,
                span_length_min=span_length_min,
                span_length_max=span_length_max,
                num_spans_mode=num_spans_mode,
            )

        raw_element_mask = raw_bin_mask.unsqueeze(-1) & batch["feature_mask"].bool().unsqueeze(1)
        token_loss_mask_float, token_lengths_from_mask = model.encoder.patch_batch(
            raw_element_mask.to(tokens.dtype),
            lengths,
        )
        if not torch.equal(token_lengths, token_lengths_from_mask):
            raise ValueError("Element-mask patching must preserve token lengths.")
        token_loss_mask = token_loss_mask_float > 0.5

        if allow_bin_fractional_overlap:
            token_mask = token_loss_mask.any(dim=-1) & valid_token_mask
        else:
            full_patch_coverage = token_loss_mask.sum(dim=-1) >= token_feature_mask.sum(dim=-1)
            token_mask = full_patch_coverage & valid_token_mask

        token_loss_token_mask = token_loss_mask.any(dim=-1) & valid_token_mask
        token_overlap_fraction = (
            token_loss_mask.to(tokens.dtype).sum(dim=-1)
            / token_feature_mask.to(tokens.dtype).sum(dim=-1).clamp_min(1.0)
        )
        raw_unit_count = int(raw_bin_mask.sum().item())
        total_unit_count = int(lengths.sum().item())

    masked_token_count = int((token_mask & valid_token_mask).sum().item())
    if masked_token_count <= 0:
        raise ValueError("Masked reconstruction requires at least one masked token.")

    masked_element_count = int(token_loss_mask.sum().item())
    if masked_element_count <= 0:
        raise ValueError("Masked reconstruction requires at least one masked element.")

    return {
        "tokens": tokens,
        "token_lengths": token_lengths,
        "token_feature_mask": token_feature_mask,
        "token_mask": token_mask,
        "token_loss_token_mask": token_loss_token_mask,
        "token_loss_mask": token_loss_mask,
        "valid_token_mask": valid_token_mask,
        "token_overlap_fraction": token_overlap_fraction,
        "raw_bin_mask": raw_bin_mask,
        "raw_unit_count": raw_unit_count,
        "total_unit_count": total_unit_count,
    }


def compute_masked_reconstruction_metrics(
    model: MaskedSSLModel,
    batch: dict[str, Any],
    *,
    mask_unit: str,
    mask_token_placement: str,
    mask_ratio: float,
    span_length_min: int,
    span_length_max: int,
    num_spans_mode: str,
    allow_bin_fractional_overlap: bool,
    device: torch.device,
) -> dict[str, Any]:
    masked_batch = build_masked_batch(
        model,
        batch,
        mask_unit=mask_unit,
        mask_ratio=mask_ratio,
        span_length_min=span_length_min,
        span_length_max=span_length_max,
        num_spans_mode=num_spans_mode,
        allow_bin_fractional_overlap=allow_bin_fractional_overlap,
    )

    tokens = masked_batch["tokens"].to(device)
    token_lengths = masked_batch["token_lengths"].to(device)
    token_mask = masked_batch["token_mask"].to(device)
    token_loss_token_mask = masked_batch["token_loss_token_mask"].to(device)
    token_loss_mask = masked_batch["token_loss_mask"].to(device)
    token_feature_mask = masked_batch["token_feature_mask"].to(device)
    token_overlap_fraction = masked_batch["token_overlap_fraction"].to(device=tokens.device, dtype=tokens.dtype)
    valid_token_mask = masked_batch["valid_token_mask"].to(device)

    outputs = model.reconstruct_from_patched_tokens(
        tokens,
        token_lengths,
        token_mask=token_mask,
        mask_token_placement=mask_token_placement,
    )
    reconstruction = outputs["reconstruction"]
    sqerr = (reconstruction - tokens).pow(2)

    element_weights = token_loss_mask.to(sqerr.dtype)
    loss_denom = element_weights.sum().clamp_min(1.0)
    loss = (sqerr * element_weights).sum() / loss_denom
    masked_predictions = reconstruction[token_loss_mask]
    masked_targets = tokens[token_loss_mask]

    token_feature_weights = token_feature_mask.to(sqerr.dtype)
    per_token_full_patch_mse = (
        (sqerr * token_feature_weights).sum(dim=-1)
        / token_feature_weights.sum(dim=-1).clamp_min(1.0)
    )
    masked_token_selector = token_mask & valid_token_mask
    scored_token_selector = token_loss_token_mask & valid_token_mask
    masked_token_full_patch_mse = float(
        per_token_full_patch_mse[scored_token_selector].mean().detach().cpu().item()
    )

    metrics = {
        "loss": loss,
        "actual_mask_ratio": float(
            masked_batch["raw_unit_count"] / max(masked_batch["total_unit_count"], 1)
        ),
        "masked_token_ratio": float(
            masked_token_selector.sum().item() / max(valid_token_mask.sum().item(), 1)
        ),
        "masked_element_ratio": float(
            token_loss_mask.sum().item() / max(token_feature_mask.sum().item(), 1)
        ),
        "masked_token_count": int(masked_token_selector.sum().item()),
        "masked_element_count": int(token_loss_mask.sum().item()),
        "masked_token_full_patch_mse": masked_token_full_patch_mse,
        "masked_prediction_mean": float(masked_predictions.mean().detach().cpu().item()),
        "masked_prediction_std": float(masked_predictions.std(unbiased=False).detach().cpu().item()),
        "masked_target_mean": float(masked_targets.mean().detach().cpu().item()),
        "masked_target_std": float(masked_targets.std(unbiased=False).detach().cpu().item()),
    }

    if mask_unit == "bin":
        weights = token_overlap_fraction * masked_token_selector.to(token_overlap_fraction.dtype)
        patch_fraction_weighted_mse = float(
            (
                per_token_full_patch_mse * weights
            ).sum().detach().cpu().item()
            / max(float(weights.sum().detach().cpu().item()), 1e-8)
        )
        metrics["patch_fraction_weighted_mse"] = patch_fraction_weighted_mse

    return metrics


def compute_objective_metrics(
    model: MaskedSSLModel,
    batch: dict[str, Any],
    *,
    mask_unit: str,
    mask_token_placement: str,
    mask_ratio: float,
    span_length_min: int,
    span_length_max: int,
    num_spans_mode: str,
    allow_bin_fractional_overlap: bool,
    device: torch.device,
) -> dict[str, Any]:
    return compute_masked_reconstruction_metrics(
        model,
        batch,
        mask_unit=mask_unit,
        mask_token_placement=mask_token_placement,
        mask_ratio=mask_ratio,
        span_length_min=span_length_min,
        span_length_max=span_length_max,
        num_spans_mode=num_spans_mode,
        allow_bin_fractional_overlap=allow_bin_fractional_overlap,
        device=device,
    )

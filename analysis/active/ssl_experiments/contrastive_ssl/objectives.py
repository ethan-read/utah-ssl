"""Contrastive SSL objectives and shared metric helpers."""

from __future__ import annotations

import random
from typing import Any

import torch
import torch.nn.functional as F

from .model import ContrastiveSSLModel


def summarize_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    summary = {
        "loss": float(metrics["loss"].detach().cpu().item()),
        "top1": float(metrics["top1"].detach().cpu().item()),
        "positive_pairs": int(metrics.get("positive_pairs", 0)),
    }
    if "per_horizon_losses" in metrics:
        summary["per_horizon_losses"] = {
            str(key): float(value) for key, value in metrics["per_horizon_losses"].items()
        }
    if "per_horizon_top1" in metrics:
        summary["per_horizon_top1"] = {
            str(key): float(value) for key, value in metrics["per_horizon_top1"].items()
        }
    if "mean_abs_view_delta" in metrics:
        summary["mean_abs_view_delta"] = float(metrics["mean_abs_view_delta"])
    return summary


def compute_future_infonce_metrics(
    model: ContrastiveSSLModel,
    batch: dict[str, Any],
    *,
    temperature: float,
    horizons: tuple[int, ...],
    device: torch.device,
) -> dict[str, Any]:
    outputs = model.encode_sequence(batch["x"].to(device), batch["lengths"].to(device))
    anchor_z = F.normalize(model.anchor_head(outputs["hidden"]), dim=-1)
    future_z = F.normalize(model.future_head(outputs["hidden"]), dim=-1)

    horizon_losses = {}
    horizon_top1 = {}
    loss_terms = []
    top1_terms = []
    positive_pairs = 0

    for horizon in horizons:
        q_list = []
        k_list = []
        for idx, length in enumerate(outputs["token_lengths"].tolist()):
            usable = int(length) - int(horizon)
            if usable <= 0:
                continue
            q_list.append(anchor_z[idx, :usable])
            k_list.append(future_z[idx, horizon:length])
        if not q_list:
            continue

        q = torch.cat(q_list, dim=0)
        k = torch.cat(k_list, dim=0)
        logits = q @ k.T / float(temperature)
        labels = torch.arange(q.shape[0], device=logits.device)
        loss = F.cross_entropy(logits, labels)
        top1 = (logits.argmax(dim=1) == labels).float().mean()

        horizon_losses[int(horizon)] = float(loss.detach().cpu().item())
        horizon_top1[int(horizon)] = float(top1.detach().cpu().item())
        loss_terms.append(loss)
        top1_terms.append(top1)
        positive_pairs += int(q.shape[0])

    if not loss_terms:
        raise ValueError("No valid future InfoNCE positive pairs remain. Reduce patching or horizon values.")

    return {
        "loss": torch.stack(loss_terms).mean(),
        "top1": torch.stack(top1_terms).mean(),
        "per_horizon_losses": horizon_losses,
        "per_horizon_top1": horizon_top1,
        "positive_pairs": positive_pairs,
    }


def augment_segment(x_seq: torch.Tensor, feature_mask: torch.Tensor, cfg: dict[str, Any]) -> torch.Tensor:
    x_aug = x_seq.clone()
    present_idx = torch.nonzero(feature_mask.bool(), as_tuple=False).squeeze(1)
    if present_idx.numel() == 0:
        return x_aug

    present = x_aug[:, present_idx]
    present = present + torch.randn_like(present) * float(cfg["noise_std"])

    scale = 1.0 + torch.randn(
        present.shape[1],
        dtype=present.dtype,
        device=present.device,
    ) * float(cfg["scale_jitter"])
    offset = torch.randn(
        present.shape[1],
        dtype=present.dtype,
        device=present.device,
    ) * float(cfg["offset_jitter"])
    present = present * scale.unsqueeze(0) + offset.unsqueeze(0)

    time_mask_frac = float(cfg["time_mask_frac"])
    if time_mask_frac > 0.0 and present.shape[0] > 1:
        width = max(1, int(round(time_mask_frac * present.shape[0])))
        width = min(width, present.shape[0])
        start = random.randrange(present.shape[0] - width + 1)
        present[start : start + width] = 0.0

    dropout_prob = float(cfg["channel_dropout_prob"])
    if dropout_prob > 0.0 and present.shape[1] > 1:
        keep = torch.rand(present.shape[1], device=present.device) > dropout_prob
        if not bool(keep.any()):
            keep[random.randrange(present.shape[1])] = True
        present = present * keep.to(present.dtype).unsqueeze(0)

    clip_value = float(cfg["clip_value"])
    x_aug[:, present_idx] = present.clamp(min=-clip_value, max=clip_value)
    return x_aug


def _sample_nonzero_view_shift_bins(max_shift_strides: int, patch_stride: int) -> int:
    if max_shift_strides <= 0:
        return 0
    shift_strides = random.randint(1, max_shift_strides)
    shift_bins = shift_strides * int(patch_stride)
    return shift_bins if random.random() < 0.5 else -shift_bins


def build_augmented_views(
    batch: dict[str, Any],
    cfg: dict[str, Any],
    *,
    patch_stride: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    max_shift_strides = max(0, int(cfg.get("view_shift_max_strides", 0)))

    view1 = []
    view2 = []
    shifted_lengths = []

    for x_seq, feature_mask, length in zip(batch["x"], batch["feature_mask"], batch["lengths"].tolist()):
        aug1 = augment_segment(x_seq, feature_mask, cfg)
        aug2 = augment_segment(x_seq, feature_mask, cfg)

        seq_len = int(length)
        max_allowed_shift_strides = min(
            max_shift_strides,
            max(0, (seq_len - 1) // int(patch_stride)),
        )
        shift_bins = _sample_nonzero_view_shift_bins(max_allowed_shift_strides, patch_stride)
        overlap_len = max(0, seq_len - abs(shift_bins))

        aligned1 = torch.zeros_like(aug1)
        aligned2 = torch.zeros_like(aug2)
        if overlap_len > 0:
            if shift_bins >= 0:
                src1_start = 0
                src2_start = shift_bins
            else:
                src1_start = -shift_bins
                src2_start = 0
            aligned1[:overlap_len] = aug1[src1_start : src1_start + overlap_len]
            aligned2[:overlap_len] = aug2[src2_start : src2_start + overlap_len]

        view1.append(aligned1)
        view2.append(aligned2)
        shifted_lengths.append(overlap_len)

    return (
        torch.stack(view1, dim=0),
        torch.stack(view2, dim=0),
        torch.tensor(shifted_lengths, dtype=batch["lengths"].dtype),
    )


def _flatten_valid_patch_embeddings(
    z: torch.Tensor,
    token_lengths: torch.Tensor,
) -> torch.Tensor:
    chunks = []
    for sample_idx, token_length in enumerate(token_lengths.tolist()):
        valid_length = int(token_length)
        if valid_length <= 0:
            continue
        chunks.append(z[sample_idx, :valid_length])

    if not chunks:
        raise ValueError("No valid patch embeddings remain for augment_infonce.")

    return torch.cat(chunks, dim=0)


def compute_augment_infonce_metrics(
    model: ContrastiveSSLModel,
    batch: dict[str, Any],
    *,
    temperature: float,
    augment_cfg: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    view1, view2, shifted_lengths = build_augmented_views(
        batch,
        augment_cfg,
        patch_stride=model.encoder.patch_stride,
    )
    lengths = shifted_lengths.to(device)
    out1 = model.encode_sequence(view1.to(device), lengths)
    out2 = model.encode_sequence(view2.to(device), lengths)
    if not torch.equal(out1["token_lengths"], out2["token_lengths"]):
        raise ValueError("Augmented views must preserve patch alignment for patch-level augment_infonce.")

    z1 = model.project_patches(out1["hidden"])
    z2 = model.project_patches(out2["hidden"])
    q = _flatten_valid_patch_embeddings(z1, out1["token_lengths"])
    k = _flatten_valid_patch_embeddings(z2, out2["token_lengths"])

    logits12 = q @ k.T / float(temperature)
    logits21 = k @ q.T / float(temperature)
    labels = torch.arange(q.shape[0], device=q.device)
    loss12 = F.cross_entropy(logits12, labels)
    loss21 = F.cross_entropy(logits21, labels)
    top1_12 = (logits12.argmax(dim=1) == labels).float().mean()
    top1_21 = (logits21.argmax(dim=1) == labels).float().mean()

    return {
        "loss": 0.5 * (loss12 + loss21),
        "top1": 0.5 * (top1_12 + top1_21),
        "positive_pairs": int(q.shape[0]),
        "mean_abs_view_delta": float((view1 - view2).abs().mean().item()),
    }


def compute_objective_metrics(
    model: ContrastiveSSLModel,
    batch: dict[str, Any],
    *,
    objective_mode: str,
    device: torch.device,
    temperature: float,
    horizons: tuple[int, ...],
    augment_cfg: dict[str, Any],
) -> dict[str, Any]:
    if objective_mode == "future_infonce":
        return compute_future_infonce_metrics(
            model,
            batch,
            temperature=temperature,
            horizons=horizons,
            device=device,
        )
    if objective_mode == "augment_infonce":
        return compute_augment_infonce_metrics(
            model,
            batch,
            temperature=temperature,
            augment_cfg=augment_cfg,
            device=device,
        )
    raise ValueError(f"Unknown objective_mode: {objective_mode}")

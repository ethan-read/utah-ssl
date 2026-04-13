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


_AUGMENT_VIEW_KEYS = (
    "noise_std",
    "scale_jitter",
    "offset_jitter",
    "time_mask_frac",
    "channel_dropout_prob",
    "clip_value",
)


def _resolve_view_cfg(cfg: dict[str, Any], view_key: str) -> dict[str, Any]:
    view_cfg = cfg.get(view_key)
    if isinstance(view_cfg, dict):
        return dict(view_cfg)
    return {key: cfg[key] for key in _AUGMENT_VIEW_KEYS if key in cfg}


def _resolve_shift_choices(cfg: dict[str, Any]) -> tuple[int, ...]:
    shift_choices = cfg.get("view_shift_choices")
    if shift_choices is not None:
        return tuple(int(shift) for shift in shift_choices)

    max_shift_strides = max(0, int(cfg.get("view_shift_max_strides", 0)))
    if max_shift_strides <= 0:
        return (0,)
    return tuple(shift for shift in range(-max_shift_strides, max_shift_strides + 1) if shift != 0)


def _sample_crop_starts(
    *,
    seq_len: int,
    crop_bins: int,
    patch_stride: int,
    shift_choices: tuple[int, ...],
) -> tuple[int, int]:
    crop_bins = max(1, min(int(crop_bins), int(seq_len)))
    max_start_stride = max(0, (int(seq_len) - crop_bins) // int(patch_stride))
    start1_stride = random.randint(0, max_start_stride) if max_start_stride > 0 else 0

    if max_start_stride <= 0:
        return start1_stride * int(patch_stride), start1_stride * int(patch_stride)

    feasible_shift_choices = [
        int(shift)
        for shift in shift_choices
        if -start1_stride <= int(shift) <= max_start_stride - start1_stride
    ]
    if not feasible_shift_choices:
        return start1_stride * int(patch_stride), start1_stride * int(patch_stride)

    shift_stride = random.choice(feasible_shift_choices)
    start2_stride = start1_stride + shift_stride
    return start1_stride * int(patch_stride), start2_stride * int(patch_stride)


def build_augmented_views(
    batch: dict[str, Any],
    cfg: dict[str, Any],
    *,
    patch_stride: int,
) -> dict[str, torch.Tensor]:
    requested_crop_bins = max(1, int(cfg.get("crop_bins", batch["x"].shape[1])))
    shift_choices = _resolve_shift_choices(cfg)
    view1_cfg = _resolve_view_cfg(cfg, "view1_cfg")
    view2_cfg = _resolve_view_cfg(cfg, "view2_cfg")

    view1 = []
    view2 = []
    crop_lengths = []
    view1_start_patches = []
    view2_start_patches = []

    for x_seq, feature_mask, length in zip(batch["x"], batch["feature_mask"], batch["lengths"].tolist()):
        aug1 = augment_segment(x_seq, feature_mask, view1_cfg)
        aug2 = augment_segment(x_seq, feature_mask, view2_cfg)

        seq_len = int(length)
        crop_bins = min(requested_crop_bins, seq_len)
        view1_start_bin, view2_start_bin = _sample_crop_starts(
            seq_len=seq_len,
            crop_bins=crop_bins,
            patch_stride=patch_stride,
            shift_choices=shift_choices,
        )
        crop1 = aug1[view1_start_bin : view1_start_bin + crop_bins]
        crop2 = aug2[view2_start_bin : view2_start_bin + crop_bins]

        crop1_padded = aug1.new_zeros((requested_crop_bins, aug1.shape[1]))
        crop2_padded = aug2.new_zeros((requested_crop_bins, aug2.shape[1]))
        crop1_padded[:crop_bins] = crop1
        crop2_padded[:crop_bins] = crop2

        view1.append(crop1_padded)
        view2.append(crop2_padded)
        crop_lengths.append(crop_bins)
        view1_start_patches.append(view1_start_bin // int(patch_stride))
        view2_start_patches.append(view2_start_bin // int(patch_stride))

    return {
        "view1": torch.stack(view1, dim=0),
        "view2": torch.stack(view2, dim=0),
        "crop_lengths": torch.tensor(crop_lengths, dtype=batch["lengths"].dtype),
        "view1_start_patches": torch.tensor(view1_start_patches, dtype=torch.long),
        "view2_start_patches": torch.tensor(view2_start_patches, dtype=torch.long),
    }


def _flatten_valid_patch_embeddings(
    z: torch.Tensor,
    token_lengths: torch.Tensor,
) -> dict[str, torch.Tensor]:
    chunks = []
    sample_indices = []
    patch_indices = []
    device = z.device
    for sample_idx, token_length in enumerate(token_lengths.tolist()):
        valid_length = int(token_length)
        if valid_length <= 0:
            continue
        chunks.append(z[sample_idx, :valid_length])
        sample_indices.append(torch.full((valid_length,), sample_idx, device=device, dtype=torch.long))
        patch_indices.append(torch.arange(valid_length, device=device, dtype=torch.long))

    if not chunks:
        raise ValueError("No valid patch embeddings remain for augment_infonce.")

    return {
        "z": torch.cat(chunks, dim=0),
        "sample_idx": torch.cat(sample_indices, dim=0),
        "patch_idx": torch.cat(patch_indices, dim=0),
    }


def _compute_local_band_infonce_direction(
    *,
    query: dict[str, torch.Tensor],
    key: dict[str, torch.Tensor],
    patch_offset_by_sample: torch.Tensor,
    positive_radius: int,
    candidate_radius: int,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    logits = query["z"] @ key["z"].T / float(temperature)
    target_patch_idx = query["patch_idx"] - patch_offset_by_sample[query["sample_idx"]]
    sample_match = query["sample_idx"].unsqueeze(1) == key["sample_idx"].unsqueeze(0)
    patch_delta = key["patch_idx"].unsqueeze(0) - target_patch_idx.unsqueeze(1)

    candidate_mask = sample_match & (patch_delta.abs() <= int(candidate_radius))
    positive_mask = sample_match & (patch_delta.abs() <= int(positive_radius))

    candidate_counts = candidate_mask.sum(dim=1)
    positive_counts = positive_mask.sum(dim=1)
    valid_queries = (positive_counts > 0) & (candidate_counts > positive_counts)
    if not bool(valid_queries.any()):
        valid_queries = positive_counts > 0
    if not bool(valid_queries.any()):
        raise ValueError("No valid local-band positives remain for augment_infonce.")

    logits = logits[valid_queries]
    candidate_mask = candidate_mask[valid_queries]
    positive_mask = positive_mask[valid_queries]

    candidate_logits = logits.masked_fill(~candidate_mask, float("-inf"))
    positive_logits = logits.masked_fill(~positive_mask, float("-inf"))
    losses = torch.logsumexp(candidate_logits, dim=1) - torch.logsumexp(positive_logits, dim=1)
    pred_idx = candidate_logits.argmax(dim=1)
    top1 = positive_mask.gather(1, pred_idx.unsqueeze(1)).to(logits.dtype).mean()
    return losses.mean(), top1, int(valid_queries.sum().item())


def compute_augment_infonce_metrics(
    model: ContrastiveSSLModel,
    batch: dict[str, Any],
    *,
    temperature: float,
    augment_cfg: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    view_bundle = build_augmented_views(
        batch,
        augment_cfg,
        patch_stride=model.encoder.patch_stride,
    )
    lengths = view_bundle["crop_lengths"].to(device)
    view1 = view_bundle["view1"].to(device)
    view2 = view_bundle["view2"].to(device)
    out1 = model.encode_sequence(view1, lengths)
    out2 = model.encode_sequence(view2, lengths)
    if not torch.equal(out1["token_lengths"], out2["token_lengths"]):
        raise ValueError("Augmented views must preserve patch alignment for patch-level augment_infonce.")

    z1 = model.project_patches(out1["hidden"])
    z2 = model.project_patches(out2["hidden"])
    query12 = _flatten_valid_patch_embeddings(z1, out1["token_lengths"])
    query21 = _flatten_valid_patch_embeddings(z2, out2["token_lengths"])

    patch_offset_12 = view_bundle["view2_start_patches"].to(device) - view_bundle["view1_start_patches"].to(device)
    patch_offset_21 = -patch_offset_12
    positive_radius = max(0, int(augment_cfg.get("positive_radius_patches", 0)))
    candidate_radius = max(
        positive_radius + 1,
        int(augment_cfg.get("local_candidate_radius_patches", positive_radius + 1)),
    )

    loss12, top1_12, pairs12 = _compute_local_band_infonce_direction(
        query=query12,
        key=query21,
        patch_offset_by_sample=patch_offset_12,
        positive_radius=positive_radius,
        candidate_radius=candidate_radius,
        temperature=temperature,
    )
    loss21, top1_21, pairs21 = _compute_local_band_infonce_direction(
        query=query21,
        key=query12,
        patch_offset_by_sample=patch_offset_21,
        positive_radius=positive_radius,
        candidate_radius=candidate_radius,
        temperature=temperature,
    )

    return {
        "loss": 0.5 * (loss12 + loss21),
        "top1": 0.5 * (top1_12 + top1_21),
        "positive_pairs": int(round(0.5 * (pairs12 + pairs21))),
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

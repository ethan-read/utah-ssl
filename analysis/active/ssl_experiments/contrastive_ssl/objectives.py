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

    scale = 1.0 + torch.randn(present.shape[1], dtype=present.dtype) * float(cfg["scale_jitter"])
    offset = torch.randn(present.shape[1], dtype=present.dtype) * float(cfg["offset_jitter"])
    present = present * scale.unsqueeze(0) + offset.unsqueeze(0)

    time_mask_frac = float(cfg["time_mask_frac"])
    if time_mask_frac > 0.0 and present.shape[0] > 1:
        width = max(1, int(round(time_mask_frac * present.shape[0])))
        width = min(width, present.shape[0])
        start = random.randrange(present.shape[0] - width + 1)
        present[start : start + width] = 0.0

    dropout_prob = float(cfg["channel_dropout_prob"])
    if dropout_prob > 0.0 and present.shape[1] > 1:
        keep = torch.rand(present.shape[1]) > dropout_prob
        if not bool(keep.any()):
            keep[random.randrange(present.shape[1])] = True
        present = present * keep.to(present.dtype).unsqueeze(0)

    clip_value = float(cfg["clip_value"])
    x_aug[:, present_idx] = present.clamp(min=-clip_value, max=clip_value)
    return x_aug


def build_augmented_views(batch: dict[str, Any], cfg: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    view1 = []
    view2 = []
    for x_seq, feature_mask in zip(batch["x"], batch["feature_mask"]):
        view1.append(augment_segment(x_seq, feature_mask, cfg))
        view2.append(augment_segment(x_seq, feature_mask, cfg))
    return torch.stack(view1, dim=0), torch.stack(view2, dim=0)


def compute_augment_infonce_metrics(
    model: ContrastiveSSLModel,
    batch: dict[str, Any],
    *,
    temperature: float,
    augment_cfg: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    view1, view2 = build_augmented_views(batch, augment_cfg)
    lengths = batch["lengths"].to(device)
    out1 = model.encode_pooled(view1.to(device), lengths)
    out2 = model.encode_pooled(view2.to(device), lengths)
    z1 = out1["z"]
    z2 = out2["z"]

    logits12 = z1 @ z2.T / float(temperature)
    logits21 = z2 @ z1.T / float(temperature)
    labels = torch.arange(z1.shape[0], device=z1.device)
    loss12 = F.cross_entropy(logits12, labels)
    loss21 = F.cross_entropy(logits21, labels)
    top1_12 = (logits12.argmax(dim=1) == labels).float().mean()
    top1_21 = (logits21.argmax(dim=1) == labels).float().mean()

    return {
        "loss": 0.5 * (loss12 + loss21),
        "top1": 0.5 * (top1_12 + top1_21),
        "positive_pairs": int(z1.shape[0]),
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

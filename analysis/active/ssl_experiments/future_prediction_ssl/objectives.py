"""Future-prediction targets and losses."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from .model import FuturePredictionModel


def _session_stats_tensors(
    *,
    boundary_keys: list[str] | tuple[str, ...],
    session_feature_stats: dict[str, tuple[torch.Tensor, torch.Tensor]] | None,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    if session_feature_stats is None:
        raise ValueError("session_feature_stats are required for denormalization-aware future losses.")
    means: list[torch.Tensor] = []
    stds: list[torch.Tensor] = []
    for key in boundary_keys:
        if key not in session_feature_stats:
            raise KeyError(f"Missing session feature stats for {key!r}")
        mean_t, std_t = session_feature_stats[key]
        means.append(mean_t.to(device=device, dtype=dtype))
        stds.append(std_t.to(device=device, dtype=dtype))
    return torch.stack(means, dim=0), torch.stack(stds, dim=0)


def aggregate_time_bins(
    x: torch.Tensor,
    input_lengths: torch.Tensor,
    *,
    stride: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean-pool adjacent time bins to create a coarser causal sequence."""

    factor = int(stride)
    if factor <= 1:
        return x, input_lengths.to(device=x.device, dtype=torch.long)
    batch_size, max_time, input_dim = x.shape
    pooled_time = max_time // factor
    if pooled_time <= 0:
        raise ValueError("temporal aggregation produced zero timesteps; increase segment length or lower stride.")
    trimmed = x[:, : pooled_time * factor, :]
    pooled = trimmed.view(batch_size, pooled_time, factor, input_dim).mean(dim=2)
    pooled_lengths = torch.div(
        input_lengths.to(device=x.device, dtype=torch.long),
        factor,
        rounding_mode="floor",
    )
    return pooled, pooled_lengths


def build_future_prediction_targets(
    x: torch.Tensor,
    input_lengths: torch.Tensor,
    *,
    future_bins: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return direct multi-horizon targets and a validity mask.

    For each timestep ``t``, horizon slot ``h`` predicts ``x[t + h + 1]``.
    """

    batch_size, max_time, input_dim = x.shape
    horizon = int(future_bins)
    targets = x.new_zeros((batch_size, max_time, horizon, input_dim))
    valid = torch.zeros((batch_size, max_time, horizon), dtype=torch.bool, device=x.device)
    positions = torch.arange(max_time, device=x.device).unsqueeze(0)

    for horizon_idx in range(horizon):
        shift = horizon_idx + 1
        if shift < max_time:
            targets[:, : max_time - shift, horizon_idx, :] = x[:, shift:, :]
        valid[:, :, horizon_idx] = positions + shift < input_lengths.unsqueeze(1)
    return targets, valid


def future_prediction_loss(
    model: FuturePredictionModel,
    batch: dict[str, Any],
    *,
    device: torch.device,
    delta: float,
    temporal_bin_stride: int = 1,
    variance_match_weight: float = 0.0,
    tx_dim: int,
    sbp_dim: int,
    feature_mode: str,
    use_normalization: bool,
    tx_loss_type: str = "huber",
    sbp_loss_type: str = "huber",
    session_feature_stats: dict[str, tuple[torch.Tensor, torch.Tensor]] | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    x = batch["x"].to(device)
    input_lengths = (batch.get("lengths") if "lengths" in batch else batch["input_lengths"]).to(device)
    x, input_lengths = aggregate_time_bins(
        x,
        input_lengths,
        stride=int(temporal_bin_stride),
    )
    outputs = model(x, input_lengths)
    targets, valid_mask = build_future_prediction_targets(
        x,
        input_lengths,
        future_bins=int(model.future_bins),
    )
    forecast = outputs["forecast"]
    prediction_for_metrics = forecast.clone()
    valid_mask_f = valid_mask.unsqueeze(-1).to(forecast.dtype)
    losses = torch.zeros_like(forecast)

    tx_slice = slice(0, int(tx_dim))
    tx_pred = forecast[..., tx_slice]
    tx_target = targets[..., tx_slice]
    tx_target_for_loss = tx_target
    if int(tx_dim) > 0:
        if str(tx_loss_type) == "poisson_nll":
            tx_rate_pred = F.softplus(tx_pred)
            if bool(use_normalization):
                boundary_keys = batch.get("boundary_keys") or batch.get("session_keys")
                if boundary_keys is None:
                    raise KeyError("boundary_keys are required for Poisson TX loss with normalized inputs.")
                mean_t, std_t = _session_stats_tensors(
                    boundary_keys=tuple(str(item) for item in boundary_keys),
                    session_feature_stats=session_feature_stats,
                    device=device,
                    dtype=forecast.dtype,
                )
                mean_t = mean_t[:, None, None, tx_slice]
                std_t = std_t[:, None, None, tx_slice].clamp_min(1e-6)
                tx_target_for_loss = tx_target * std_t + mean_t
                prediction_for_metrics[..., tx_slice] = (tx_rate_pred - mean_t) / std_t
            else:
                prediction_for_metrics[..., tx_slice] = tx_rate_pred
            losses[..., tx_slice] = F.poisson_nll_loss(
                tx_rate_pred,
                tx_target_for_loss.clamp_min(0.0),
                log_input=False,
                full=False,
                reduction="none",
            )
        else:
            losses[..., tx_slice] = F.huber_loss(tx_pred, tx_target, delta=float(delta), reduction="none")

    if str(feature_mode) == "tx_sbp" and int(sbp_dim) > 0:
        sbp_slice = slice(int(tx_dim), int(tx_dim) + int(sbp_dim))
        sbp_pred = forecast[..., sbp_slice]
        sbp_target = targets[..., sbp_slice]
        if str(sbp_loss_type) != "huber":
            raise ValueError(f"Unsupported sbp_loss_type: {sbp_loss_type!r}")
        losses[..., sbp_slice] = F.huber_loss(sbp_pred, sbp_target, delta=float(delta), reduction="none")

    denom = valid_mask_f.expand_as(losses).sum().clamp_min(1.0)
    base_loss = (losses * valid_mask_f).sum() / denom
    pred_masked = prediction_for_metrics[valid_mask_f.expand_as(prediction_for_metrics).bool()]
    target_masked = targets[valid_mask_f.expand_as(targets).bool()]
    variance_match_penalty = base_loss.new_zeros(())
    if (
        float(variance_match_weight) > 0.0
        and pred_masked.numel() > 1
        and target_masked.numel() > 1
    ):
        pred_std_t = pred_masked.float().std(unbiased=False)
        target_std_t = target_masked.float().std(unbiased=False)
        variance_match_penalty = torch.square((pred_std_t - target_std_t) / target_std_t.clamp_min(1e-6))
    loss = base_loss + float(variance_match_weight) * variance_match_penalty
    with torch.no_grad():
        tx_loss_value = 0.0
        if int(tx_dim) > 0:
            tx_valid_mask = valid_mask_f.expand_as(losses[..., tx_slice]).bool()
            tx_loss_value = float(losses[..., tx_slice][tx_valid_mask].float().mean().item()) if tx_valid_mask.any() else 0.0
        sbp_loss_value = 0.0
        if str(feature_mode) == "tx_sbp" and int(sbp_dim) > 0:
            sbp_slice = slice(int(tx_dim), int(tx_dim) + int(sbp_dim))
            sbp_valid_mask = valid_mask_f.expand_as(losses[..., sbp_slice]).bool()
            sbp_loss_value = (
                float(losses[..., sbp_slice][sbp_valid_mask].float().mean().item()) if sbp_valid_mask.any() else 0.0
            )
        abs_error = (prediction_for_metrics - targets).abs()
        zero_abs_error = targets.abs()
        pred_std_value = float(pred_masked.float().std(unbiased=False).item()) if pred_masked.numel() > 0 else 0.0
        target_std_value = float(target_masked.float().std(unbiased=False).item()) if target_masked.numel() > 0 else 0.0
        metrics: dict[str, float] = {
            "loss": float(loss.detach().item()),
            "base_loss": float(base_loss.detach().item()),
            "tx_loss": tx_loss_value,
            "sbp_loss": sbp_loss_value,
            "variance_match_penalty": float(variance_match_penalty.detach().item()),
            "mean_token_length": float(outputs["token_lengths"].float().mean().item()),
            "valid_step_fraction": float(valid_mask.float().mean().item()),
            "pred_abs_mean": float(pred_masked.abs().float().mean().item()) if pred_masked.numel() > 0 else 0.0,
            "pred_std": pred_std_value,
            "target_abs_mean": float(target_masked.abs().float().mean().item()) if target_masked.numel() > 0 else 0.0,
            "target_std": target_std_value,
            "zero_baseline_mae": float(
                zero_abs_error[valid_mask_f.expand_as(zero_abs_error).bool()].float().mean().item()
            )
            if bool(valid_mask.any().item())
            else 0.0,
            "pred_to_target_std_ratio": float(
                pred_std_value / max(target_std_value, 1e-8)
            )
            if pred_masked.numel() > 0 and target_masked.numel() > 0
            else 0.0,
        }
        for horizon_idx in range(int(model.future_bins)):
            horizon_mask = valid_mask[:, :, horizon_idx].unsqueeze(-1)
            if bool(horizon_mask.any().item()):
                horizon_error = abs_error[:, :, horizon_idx, :][horizon_mask.expand_as(abs_error[:, :, horizon_idx, :])]
                metrics[f"h{horizon_idx + 1}_mae"] = float(horizon_error.float().mean().item())
                zero_horizon_error = zero_abs_error[:, :, horizon_idx, :][
                    horizon_mask.expand_as(zero_abs_error[:, :, horizon_idx, :])
                ]
                pred_horizon = prediction_for_metrics[:, :, horizon_idx, :][
                    horizon_mask.expand_as(outputs["forecast"][:, :, horizon_idx, :])
                ]
                target_horizon = targets[:, :, horizon_idx, :][
                    horizon_mask.expand_as(targets[:, :, horizon_idx, :])
                ]
                metrics[f"h{horizon_idx + 1}_zero_mae"] = float(zero_horizon_error.float().mean().item())
                metrics[f"h{horizon_idx + 1}_pred_std"] = float(pred_horizon.float().std(unbiased=False).item())
                metrics[f"h{horizon_idx + 1}_target_std"] = float(target_horizon.float().std(unbiased=False).item())
    return loss, metrics


__all__ = [
    "aggregate_time_bins",
    "build_future_prediction_targets",
    "future_prediction_loss",
]

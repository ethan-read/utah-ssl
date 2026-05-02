"""Stage-1 objective plug-ins for POSSM reconstruction training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class Stage1Batch:
    x_input: torch.Tensor
    x_target: torch.Tensor
    lengths: torch.Tensor
    feature_mask: torch.Tensor
    loss_mask: torch.Tensor | None = None
    mask_metadata: dict[str, Any] | None = None
    session_ids: list[str] | tuple[str, ...] | None = None


class Stage1Objective:
    name: str = "stage1_objective"

    def prepare_batch(
        self,
        raw_batch: dict[str, Any],
        *,
        device: torch.device,
        config: dict[str, Any],
    ) -> Stage1Batch:
        raise NotImplementedError

    def compute_loss(
        self,
        model_outputs: dict[str, torch.Tensor],
        stage1_batch: Stage1Batch,
    ) -> dict[str, Any]:
        raise NotImplementedError


class PlainReconstructionObjective(Stage1Objective):
    name = "plain_mse"

    def prepare_batch(
        self,
        raw_batch: dict[str, Any],
        *,
        device: torch.device,
        config: dict[str, Any],
    ) -> Stage1Batch:
        del config
        x = raw_batch["x"].to(device)
        lengths = raw_batch["lengths"].to(device)
        feature_mask = raw_batch["feature_mask"].to(device)
        return Stage1Batch(
            x_input=x,
            x_target=x,
            lengths=lengths,
            feature_mask=feature_mask,
            loss_mask=None,
            mask_metadata={
                "mask_type": "none",
                "masked_fraction": 0.0,
                "mask_shape": [int(x.shape[0]), int(x.shape[1]), int(x.shape[2])],
            },
            session_ids=raw_batch.get("session_keys"),
        )

    def compute_loss(
        self,
        model_outputs: dict[str, torch.Tensor],
        stage1_batch: Stage1Batch,
    ) -> dict[str, Any]:
        reconstruction = model_outputs["reconstruction"]
        valid_time = (
            torch.arange(stage1_batch.x_target.shape[1], device=stage1_batch.lengths.device).unsqueeze(0)
            < stage1_batch.lengths.unsqueeze(1)
        )
        valid_features = stage1_batch.feature_mask.bool().unsqueeze(1)
        valid = valid_time.unsqueeze(-1) & valid_features
        diff_sq = (reconstruction - stage1_batch.x_target).pow(2)
        loss = diff_sq.masked_select(valid).mean()
        return {
            "loss": loss,
            "mse": float(loss.detach().item()),
            "num_valid_elements": int(valid.sum().item()),
            "reconstruction": reconstruction.detach(),
            "objective_type": self.name,
            "masked_fraction": 0.0,
        }


class MaskedReconstructionObjective(Stage1Objective):
    name = "masked_mse"

    def __init__(
        self,
        *,
        masking_type: str,
        mask_prob: float,
        mask_span_bins: int,
        mask_replace_mode: str,
        seed: int,
    ) -> None:
        if masking_type not in {"none", "random", "span", "channel"}:
            raise ValueError("masking_type must be one of {'none', 'random', 'span', 'channel'}")
        if not (0.0 <= float(mask_prob) <= 1.0):
            raise ValueError("mask_prob must be in [0, 1]")
        if int(mask_span_bins) <= 0:
            raise ValueError("mask_span_bins must be positive")
        if mask_replace_mode not in {"zero", "mean", "gaussian_noise"}:
            raise ValueError("mask_replace_mode must be one of {'zero', 'mean', 'gaussian_noise'}")
        self.masking_type = str(masking_type)
        self.mask_prob = float(mask_prob)
        self.mask_span_bins = int(mask_span_bins)
        self.mask_replace_mode = str(mask_replace_mode)
        self.generator = torch.Generator(device="cpu")
        self.generator.manual_seed(int(seed))

    def _valid_candidate_mask(
        self,
        *,
        x: torch.Tensor,
        lengths: torch.Tensor,
        feature_mask: torch.Tensor,
    ) -> torch.Tensor:
        _, time_bins, _ = x.shape
        valid_time = (
            torch.arange(time_bins, device=lengths.device).unsqueeze(0)
            < lengths.unsqueeze(1)
        )
        valid_features = feature_mask.bool().unsqueeze(1)
        return valid_time.unsqueeze(-1) & valid_features

    def _ensure_nonempty_mask(self, proposed_mask: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        if bool(proposed_mask.any().item()) or not bool(valid_mask.any().item()):
            return proposed_mask
        valid_indices = torch.nonzero(valid_mask, as_tuple=False)
        chosen_idx = int(
            torch.randint(
                low=0,
                high=valid_indices.shape[0],
                size=(1,),
                generator=self.generator,
            ).item()
        )
        batch_idx, time_idx, feature_idx = valid_indices[chosen_idx].tolist()
        proposed_mask[batch_idx, time_idx, feature_idx] = True
        return proposed_mask

    def _make_mask(
        self,
        *,
        x: torch.Tensor,
        lengths: torch.Tensor,
        feature_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, time_bins, input_dim = x.shape
        valid = self._valid_candidate_mask(x=x, lengths=lengths, feature_mask=feature_mask)
        if self.masking_type == "none" or self.mask_prob <= 0.0:
            return torch.zeros_like(valid)

        if self.masking_type == "random":
            random_mask = torch.rand(
                (batch_size, time_bins, input_dim),
                generator=self.generator,
                dtype=torch.float32,
            ).to(device=x.device)
            return self._ensure_nonempty_mask(valid & (random_mask < self.mask_prob), valid)

        if self.masking_type == "channel":
            channel_draw = torch.rand(
                (batch_size, input_dim),
                generator=self.generator,
                dtype=torch.float32,
            ).to(device=x.device)
            channel_mask = channel_draw < self.mask_prob
            return self._ensure_nonempty_mask(valid & channel_mask.unsqueeze(1), valid)

        # span masking
        span_mask = torch.zeros((batch_size, time_bins), dtype=torch.bool, device=x.device)
        span_len = int(self.mask_span_bins)
        for batch_idx in range(batch_size):
            sequence_len = int(lengths[batch_idx].item())
            if sequence_len <= 0:
                continue
            n_spans = max(1, int(round((sequence_len * self.mask_prob) / max(1, span_len))))
            max_start = max(0, sequence_len - span_len)
            starts = torch.randint(
                low=0,
                high=max_start + 1,
                size=(n_spans,),
                generator=self.generator,
            ).to(device=x.device)
            for start in starts.tolist():
                stop = min(sequence_len, int(start) + span_len)
                span_mask[batch_idx, int(start):stop] = True
        return self._ensure_nonempty_mask(valid & span_mask.unsqueeze(-1), valid)

    def _apply_mask(
        self,
        x: torch.Tensor,
        loss_mask: torch.Tensor,
        feature_mask: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        x_input = x.clone()
        if not loss_mask.any():
            return x_input
        if self.mask_replace_mode == "zero":
            x_input[loss_mask] = 0.0
            return x_input
        if self.mask_replace_mode == "mean":
            valid = self._valid_candidate_mask(x=x, lengths=lengths, feature_mask=feature_mask).to(dtype=x.dtype)
            per_example_counts = valid.sum(dim=1).clamp_min(1.0)
            per_example_means = (x * valid).sum(dim=1) / per_example_counts
            x_input[loss_mask] = per_example_means.unsqueeze(1).expand_as(x)[loss_mask]
            return x_input

        noise = torch.randn(x.shape, generator=self.generator, dtype=x.dtype).to(device=x.device)
        x_input[loss_mask] = noise[loss_mask]
        return x_input

    def prepare_batch(
        self,
        raw_batch: dict[str, Any],
        *,
        device: torch.device,
        config: dict[str, Any],
    ) -> Stage1Batch:
        del config
        x_target = raw_batch["x"].to(device)
        lengths = raw_batch["lengths"].to(device)
        feature_mask = raw_batch["feature_mask"].to(device)
        loss_mask = self._make_mask(x=x_target, lengths=lengths, feature_mask=feature_mask)
        x_input = self._apply_mask(x_target, loss_mask, feature_mask, lengths)
        valid = self._valid_candidate_mask(x=x_target, lengths=lengths, feature_mask=feature_mask)
        valid_count = int(valid.sum().item())
        masked_fraction = 0.0 if valid_count <= 0 else float(loss_mask.sum().item() / valid_count)
        return Stage1Batch(
            x_input=x_input,
            x_target=x_target,
            lengths=lengths,
            feature_mask=feature_mask,
            loss_mask=loss_mask,
            mask_metadata={
                "mask_type": self.masking_type,
                "masked_fraction": masked_fraction,
                "mask_shape": [int(loss_mask.shape[0]), int(loss_mask.shape[1]), int(loss_mask.shape[2])],
            },
            session_ids=raw_batch.get("session_keys"),
        )

    def compute_loss(
        self,
        model_outputs: dict[str, torch.Tensor],
        stage1_batch: Stage1Batch,
    ) -> dict[str, Any]:
        reconstruction = model_outputs["reconstruction"]
        valid = self._valid_candidate_mask(
            x=stage1_batch.x_target,
            lengths=stage1_batch.lengths,
            feature_mask=stage1_batch.feature_mask,
        )
        if stage1_batch.loss_mask is None:
            raise ValueError("MaskedReconstructionObjective requires stage1_batch.loss_mask.")
        valid = valid & stage1_batch.loss_mask.bool()
        if not bool(valid.any().item()):
            raise ValueError("MaskedReconstructionObjective received an empty effective loss mask.")
        diff_sq = (reconstruction - stage1_batch.x_target).pow(2)
        loss = diff_sq.masked_select(valid).mean()
        masked_fraction = (
            float(stage1_batch.mask_metadata["masked_fraction"])
            if stage1_batch.mask_metadata is not None and "masked_fraction" in stage1_batch.mask_metadata
            else 0.0
        )
        return {
            "loss": loss,
            "mse": float(loss.detach().item()),
            "num_valid_elements": int(valid.sum().item()),
            "reconstruction": reconstruction.detach(),
            "objective_type": self.name,
            "masked_fraction": masked_fraction,
        }


def build_stage1_objective(*, config: dict[str, Any], seed: int) -> Stage1Objective:
    objective_type = str(config.get("stage1_objective_type", "plain_mse"))
    if objective_type == "plain_mse":
        return PlainReconstructionObjective()
    if objective_type == "masked_mse":
        return MaskedReconstructionObjective(
            masking_type=str(config.get("masking_type", "none")),
            mask_prob=float(config.get("mask_prob", 0.0)),
            mask_span_bins=int(config.get("mask_span_bins", 8)),
            mask_replace_mode=str(config.get("mask_replace_mode", "zero")),
            seed=int(seed),
        )
    raise ValueError("stage1_objective_type must be one of {'plain_mse', 'masked_mse'}")

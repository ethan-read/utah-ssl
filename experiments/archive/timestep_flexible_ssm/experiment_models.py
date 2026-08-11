"""Experiment-specific models for irregular and future prediction runs."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from experiments.supervised_baselines.model import WillettPhonemeModel

from .model import TimestepFlexibleS5Model
from .data import resolve_patch_bins
from .model import patch_resample_batch


class IrregularTimestepS5Model(TimestepFlexibleS5Model):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.time_delta_projection = nn.Sequential(
            nn.Linear(1, self.input_projection_size),
            nn.Tanh(),
        )

    def forward(
        self,
        x: torch.Tensor,
        input_lengths: torch.Tensor,
        *,
        active_bin_size_ms: int,
        session_ids: list[str] | tuple[str, ...] | None = None,
        time_deltas_ms: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | float | int]:
        adapted_input = self.session_input_adapter(
            x,
            session_ids,
            session_adapter_enabled=self.session_adapter_enabled,
        )
        if time_deltas_ms is None:
            time_deltas_ms = torch.full(
                (int(x.shape[0]), int(x.shape[1])),
                float(active_bin_size_ms),
                device=x.device,
                dtype=x.dtype,
            )
        delta_scale = (time_deltas_ms.to(device=x.device, dtype=x.dtype) / float(self.train_bin_size_ms)).unsqueeze(-1)
        adapted_input = adapted_input + self.time_delta_projection(delta_scale)
        active_patch_size_bins = resolve_patch_bins(
            int(self.patch_size_ms),
            bin_size_ms=int(active_bin_size_ms),
            field_name="patch_size_ms",
        )
        active_patch_stride_bins = resolve_patch_bins(
            int(self.patch_stride_ms),
            bin_size_ms=int(active_bin_size_ms),
            field_name="patch_stride_ms",
        )
        patched_inputs, token_lengths = patch_resample_batch(
            adapted_input,
            input_lengths,
            active_patch_size_bins=int(active_patch_size_bins),
            active_patch_stride_bins=int(active_patch_stride_bins),
            reference_patch_size_bins=int(self.reference_patch_size_bins),
        )
        projected_inputs = self.s5_hidden_norm(self.s5_input_projection(self.s5_input_norm(patched_inputs)))
        hidden = self.s5(projected_inputs, token_lengths)
        logits = self.classifier(hidden)
        return {
            "adapted_input": adapted_input,
            "patched_inputs": patched_inputs,
            "projected_inputs": projected_inputs,
            "hidden": hidden,
            "decoder_hidden": hidden,
            "token_lengths": token_lengths,
            "logits": logits,
            "dt_scale": 1.0,
            "active_patch_size_bins": int(active_patch_size_bins),
            "active_patch_stride_bins": int(active_patch_stride_bins),
        }


@dataclass(frozen=True)
class FutureModelOutputs:
    hidden: torch.Tensor
    token_lengths: torch.Tensor
    patch_size_bins: int
    patch_stride_bins: int


class FuturePredictionHead(nn.Module):
    def __init__(self, *, hidden_dim: int, input_dim: int, horizons_ms: tuple[int, ...], projection_dim: int = 128):
        super().__init__()
        self.horizons_ms = tuple(int(item) for item in horizons_ms)
        self.query_heads = nn.ModuleDict(
            {
                str(horizon_ms): nn.Sequential(
                    nn.Linear(int(hidden_dim), int(projection_dim)),
                    nn.GELU(),
                    nn.Linear(int(projection_dim), int(projection_dim)),
                )
                for horizon_ms in self.horizons_ms
            }
        )
        self.target_heads = nn.ModuleDict(
            {
                str(horizon_ms): nn.Sequential(
                    nn.Linear(int(input_dim), int(projection_dim)),
                    nn.GELU(),
                    nn.Linear(int(projection_dim), int(projection_dim)),
                )
                for horizon_ms in self.horizons_ms
            }
        )

    def project_query(self, hidden: torch.Tensor, *, horizon_ms: int) -> torch.Tensor:
        return F.normalize(self.query_heads[str(int(horizon_ms))](hidden), dim=-1)

    def project_target(self, target: torch.Tensor, *, horizon_ms: int) -> torch.Tensor:
        return F.normalize(self.target_heads[str(int(horizon_ms))](target), dim=-1)


class FutureS5Model(nn.Module):
    def __init__(self, encoder: TimestepFlexibleS5Model, *, horizons_ms: tuple[int, ...], input_dim: int, projection_dim: int = 128) -> None:
        super().__init__()
        self.encoder = encoder
        self.head = FuturePredictionHead(
            hidden_dim=int(encoder.s5_hidden_size),
            input_dim=int(input_dim),
            horizons_ms=tuple(int(item) for item in horizons_ms),
            projection_dim=int(projection_dim),
        )

    def encode(
        self,
        x: torch.Tensor,
        input_lengths: torch.Tensor,
        *,
        session_ids: list[str] | tuple[str, ...] | None,
    ) -> FutureModelOutputs:
        outputs = self.encoder(
            x,
            input_lengths,
            active_bin_size_ms=20,
            session_ids=session_ids,
        )
        return FutureModelOutputs(
            hidden=outputs["hidden"],
            token_lengths=outputs["token_lengths"],
            patch_size_bins=int(outputs["active_patch_size_bins"]),
            patch_stride_bins=int(outputs["active_patch_stride_bins"]),
        )


class FutureGRUModel(nn.Module):
    def __init__(self, encoder: WillettPhonemeModel, *, horizons_ms: tuple[int, ...], input_dim: int, projection_dim: int = 128) -> None:
        super().__init__()
        self.encoder = encoder
        self.head = FuturePredictionHead(
            hidden_dim=int(encoder.gru_hidden_size),
            input_dim=int(input_dim),
            horizons_ms=tuple(int(item) for item in horizons_ms),
            projection_dim=int(projection_dim),
        )

    def encode(
        self,
        x: torch.Tensor,
        input_lengths: torch.Tensor,
        *,
        session_ids: list[str] | tuple[str, ...] | None,
    ) -> FutureModelOutputs:
        outputs = self.encoder(x, input_lengths, session_ids=session_ids)
        return FutureModelOutputs(
            hidden=outputs["hidden"],
            token_lengths=outputs["token_lengths"],
            patch_size_bins=int(self.encoder.patch_size),
            patch_stride_bins=int(self.encoder.patch_stride),
        )


__all__ = [
    "FutureGRUModel",
    "FutureModelOutputs",
    "FuturePredictionHead",
    "FutureS5Model",
    "IrregularTimestepS5Model",
]

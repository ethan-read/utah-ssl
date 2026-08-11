"""Core models for future-prediction SSL."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

_EXPERIMENTS_ROOT = Path(__file__).resolve().parent.parent
if str(_EXPERIMENTS_ROOT) not in sys.path:
    sys.path.insert(0, str(_EXPERIMENTS_ROOT))

try:
    from experiments.archive.generic_ssm_ssl.model import GenericSSMEncoder, make_encoder_from_config
except ModuleNotFoundError:  # pragma: no cover - repo-root unittest fallback
    from experiments.archive.generic_ssm_ssl.model import GenericSSMEncoder, make_encoder_from_config


class FuturePredictionModel(nn.Module):
    """Forecast the next fixed number of feature bins from causal encoder states."""

    def __init__(
        self,
        *,
        encoder: GenericSSMEncoder,
        input_dim: int,
        future_bins: int,
    ) -> None:
        super().__init__()
        if str(encoder.input_mode) != "raw_bin":
            raise ValueError("FuturePredictionModel currently supports only raw-bin encoders.")
        self.encoder = encoder
        self.input_dim = int(input_dim)
        self.future_bins = int(future_bins)
        self.forecast_head = nn.Linear(int(encoder.hidden_size), int(self.future_bins * self.input_dim))

    def forward(self, x: torch.Tensor, input_lengths: torch.Tensor) -> dict[str, torch.Tensor]:
        outputs = self.encoder.encode(x, input_lengths)
        forecast = self.forecast_head(outputs.hidden).view(
            int(outputs.hidden.shape[0]),
            int(outputs.hidden.shape[1]),
            self.future_bins,
            self.input_dim,
        )
        return {
            "tokens": outputs.tokens,
            "hidden": outputs.hidden,
            "forecast": forecast,
            "token_lengths": outputs.token_lengths,
        }


class FuturePredictionCTCProbeModel(nn.Module):
    """CTC probe over either encoder hidden states or forecasted future bins."""

    def __init__(
        self,
        *,
        future_model: FuturePredictionModel,
        vocab_size: int,
        feature_source: str,
        forecast_horizon_index: int = 0,
    ) -> None:
        super().__init__()
        self.future_model = future_model
        self.feature_source = str(feature_source)
        self.forecast_horizon_index = int(forecast_horizon_index)
        if self.feature_source == "encoder_hidden":
            classifier_input_dim = int(future_model.encoder.hidden_size)
        elif self.feature_source == "forecast_bin":
            classifier_input_dim = int(future_model.input_dim)
        else:
            raise ValueError("feature_source must be one of {'encoder_hidden', 'forecast_bin'}")
        self.classifier = nn.Linear(classifier_input_dim, int(vocab_size))

    def forward(
        self,
        x: torch.Tensor,
        input_lengths: torch.Tensor,
        *,
        session_ids: list[str] | tuple[str, ...] | None = None,
    ) -> dict[str, torch.Tensor]:
        del session_ids
        outputs = self.future_model(x, input_lengths)
        if self.feature_source == "encoder_hidden":
            features = outputs["hidden"]
        else:
            features = outputs["forecast"][:, :, self.forecast_horizon_index, :]
        logits = self.classifier(features)
        return {
            "tokens": outputs["tokens"],
            "hidden": outputs["hidden"],
            "forecast": outputs["forecast"],
            "probe_features": features,
            "logits": logits,
            "token_lengths": outputs["token_lengths"],
        }


def make_future_prediction_model(config: Any, *, input_dim: int | None = None) -> FuturePredictionModel:
    resolved_input_dim = int(input_dim if input_dim is not None else config.input_dim)
    encoder = make_encoder_from_config(config, input_dim=resolved_input_dim)
    return FuturePredictionModel(
        encoder=encoder,
        input_dim=resolved_input_dim,
        future_bins=int(config.future_bins),
    )


__all__ = [
    "FuturePredictionCTCProbeModel",
    "FuturePredictionModel",
    "make_future_prediction_model",
]

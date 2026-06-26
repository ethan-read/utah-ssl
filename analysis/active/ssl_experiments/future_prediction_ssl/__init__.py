"""Future-prediction SSL experiments for Brain2Text24."""

from .config import FuturePredictionSSLConfig
from .model import FuturePredictionModel, make_future_prediction_model
from .objectives import aggregate_time_bins, build_future_prediction_targets, future_prediction_loss
from .training import (
    load_encoder_checkpoint,
    run_frozen_linear_ctc_probe,
    run_future_prediction_pretraining,
    run_future_prediction_ssl,
)

__all__ = [
    "FuturePredictionModel",
    "FuturePredictionSSLConfig",
    "aggregate_time_bins",
    "build_future_prediction_targets",
    "future_prediction_loss",
    "load_encoder_checkpoint",
    "make_future_prediction_model",
    "run_frozen_linear_ctc_probe",
    "run_future_prediction_pretraining",
    "run_future_prediction_ssl",
]

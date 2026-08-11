"""Configuration for future-prediction SSL experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class FuturePredictionSSLConfig:
    seed: int = 7
    backbone_type: str = "mamba"
    dataset: str = "brain2text24"
    pretrain_datasets: tuple[str, ...] = ("brain2text24",)
    feature_mode: str = "tx_sbp"
    boundary_key_mode: str = "session"
    cache_root: str | Path = "/Users/home/thesis/data/cache_v1"
    cache_mode: str = "drive_direct"
    local_cache_base: str | Path = "/content/utah_ssl_cache"
    use_normalization: bool = True
    precomputed_session_stats_path: str | Path | None = None
    tx_dim: int = 128
    sbp_dim: int = 128
    segment_bins: int = 256
    temporal_bin_stride: int = 1
    batch_size: int = 16
    ssl_steps: int = 1000
    learning_rate: float = 3e-4
    weight_decay: float = 1e-2
    max_grad_norm: float = 1.0
    hidden_size: int = 256
    state_size: int = 64
    num_layers: int = 4
    dropout: float = 0.1
    direction: str = "causal"
    ffn_multiplier: float = 2.0
    input_mode: str = "raw_bin"
    patch_size: int = 1
    patch_stride: int = 1
    patch_policy: str = "floor"
    conv_kernel_size: int = 1
    conv_stride: int = 1
    future_bins: int = 3
    forecast_loss_delta: float = 1.0
    variance_match_weight: float = 0.05
    tx_loss_type: str = "huber"
    sbp_loss_type: str = "huber"
    val_every_steps: int = 100
    val_batches: int = 4
    progress_every_steps: int = 25
    progress_every_seconds: float = 30.0
    checkpoint_every_steps: int | None = None
    resume: bool = False
    resume_checkpoint_path: str | Path | None = None
    run_frozen_probe: bool = True
    probe_feature_source: str = "encoder_hidden"
    probe_forecast_horizon_index: int = 0
    probe_steps: int = 2000
    probe_batch_size: int = 8
    probe_learning_rate: float = 1e-3
    probe_weight_decay: float = 0.0
    output_root: str | Path = "experiments/archive/mamba_future_prediction_runs"
    run_name: str | None = None

    def __post_init__(self) -> None:
        if self.backbone_type not in {"s5", "mamba"}:
            raise ValueError("backbone_type must be one of {'s5', 'mamba'}")
        if self.feature_mode not in {"tx_only", "tx_sbp"}:
            raise ValueError("feature_mode must be one of {'tx_only', 'tx_sbp'}")
        if self.boundary_key_mode not in {"session", "subject_if_available"}:
            raise ValueError("boundary_key_mode must be one of {'session', 'subject_if_available'}")
        if self.cache_mode not in {"copy_to_local", "drive_direct"}:
            raise ValueError("cache_mode must be one of {'copy_to_local', 'drive_direct'}")
        if self.direction != "causal":
            raise ValueError("Future-prediction SSL currently requires direction='causal'.")
        if self.input_mode != "raw_bin":
            raise ValueError("Future-prediction SSL currently requires input_mode='raw_bin'.")
        if self.patch_policy not in {"floor", "cover_tail"}:
            raise ValueError("patch_policy must be one of {'floor', 'cover_tail'}")
        if int(self.tx_dim) <= 0:
            raise ValueError("tx_dim must be positive")
        if self.feature_mode == "tx_sbp" and int(self.sbp_dim) <= 0:
            raise ValueError("sbp_dim must be positive when feature_mode='tx_sbp'")
        if self.feature_mode == "tx_only" and int(self.sbp_dim) < 0:
            raise ValueError("sbp_dim must be non-negative when feature_mode='tx_only'")
        if int(self.segment_bins) <= int(self.future_bins):
            raise ValueError("segment_bins must exceed future_bins so valid targets exist.")
        if int(self.temporal_bin_stride) <= 0:
            raise ValueError("temporal_bin_stride must be positive")
        if int(self.segment_bins) < int(self.temporal_bin_stride):
            raise ValueError("segment_bins must be at least temporal_bin_stride")
        if int(self.batch_size) <= 0 or int(self.probe_batch_size) <= 0:
            raise ValueError("batch_size and probe_batch_size must be positive")
        if int(self.ssl_steps) <= 0 or int(self.probe_steps) <= 0:
            raise ValueError("ssl_steps and probe_steps must be positive")
        if float(self.learning_rate) <= 0.0 or float(self.probe_learning_rate) <= 0.0:
            raise ValueError("learning rates must be positive")
        if float(self.weight_decay) < 0.0 or float(self.probe_weight_decay) < 0.0:
            raise ValueError("weight decays must be non-negative")
        if float(self.max_grad_norm) <= 0.0:
            raise ValueError("max_grad_norm must be positive")
        if int(self.hidden_size) <= 0 or int(self.state_size) <= 0 or int(self.num_layers) <= 0:
            raise ValueError("hidden_size, state_size, and num_layers must be positive")
        if int(self.patch_size) <= 0 or int(self.patch_stride) <= 0:
            raise ValueError("patch_size and patch_stride must be positive")
        if int(self.conv_kernel_size) <= 0 or int(self.conv_stride) <= 0:
            raise ValueError("conv_kernel_size and conv_stride must be positive")
        if float(self.dropout) < 0.0:
            raise ValueError("dropout must be non-negative")
        if int(self.future_bins) <= 0:
            raise ValueError("future_bins must be positive")
        if float(self.forecast_loss_delta) <= 0.0:
            raise ValueError("forecast_loss_delta must be positive")
        if float(self.variance_match_weight) < 0.0:
            raise ValueError("variance_match_weight must be non-negative")
        if self.tx_loss_type not in {"huber", "poisson_nll"}:
            raise ValueError("tx_loss_type must be one of {'huber', 'poisson_nll'}")
        if self.sbp_loss_type not in {"huber"}:
            raise ValueError("sbp_loss_type must be 'huber'")
        if self.probe_feature_source not in {"encoder_hidden", "forecast_bin"}:
            raise ValueError("probe_feature_source must be one of {'encoder_hidden', 'forecast_bin'}")
        if int(self.probe_forecast_horizon_index) < 0:
            raise ValueError("probe_forecast_horizon_index must be non-negative")
        if int(self.probe_forecast_horizon_index) >= int(self.future_bins):
            raise ValueError("probe_forecast_horizon_index must be less than future_bins")
        if self.resume_checkpoint_path is not None and not str(self.resume_checkpoint_path).strip():
            self.resume_checkpoint_path = None
        normalized_datasets = tuple(str(item).strip() for item in self.pretrain_datasets if str(item).strip())
        if not normalized_datasets:
            raise ValueError("pretrain_datasets must contain at least one dataset name.")
        self.pretrain_datasets = normalized_datasets

    @property
    def input_dim(self) -> int:
        return int(self.tx_dim if self.feature_mode == "tx_only" else self.tx_dim + self.sbp_dim)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for key in (
            "cache_root",
            "local_cache_base",
            "precomputed_session_stats_path",
            "output_root",
            "resume_checkpoint_path",
        ):
            if payload[key] is not None:
                payload[key] = str(payload[key])
        payload["pretrain_datasets"] = list(self.pretrain_datasets)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "FuturePredictionSSLConfig":
        values = dict(payload)
        if "pretrain_datasets" in values and values["pretrain_datasets"] is not None:
            values["pretrain_datasets"] = tuple(str(item) for item in values["pretrain_datasets"])
        return cls(**values)


__all__ = ["FuturePredictionSSLConfig"]

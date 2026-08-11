"""Configuration for generic S5/Mamba SSL experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from utah_ssl.bit_cache_contract import BIT_STAGE1_DATASET_SPLITS, BIT_STAGE1_TX_DIM
from utah_ssl.experiment_contract import DatasetPlan, SignalSpec


@dataclass
class GenericSSMSSLConfig:
    seed: int = 7
    backbone_type: str = "s5"
    input_mode: str = "temporal_patch"
    objective: str = "masked_time_channel_reconstruction"
    dataset: str = "brain2text24"
    dataset_plan: DatasetPlan | dict[str, tuple[str, ...]] = field(
        default_factory=lambda: DatasetPlan.from_mapping(BIT_STAGE1_DATASET_SPLITS)
    )
    signal_spec: SignalSpec | dict[str, Any] = field(
        default_factory=lambda: SignalSpec.tx_only(
            tx_dim=BIT_STAGE1_TX_DIM,
            missing_channel_policy="zero_pad",
        )
    )
    boundary_key_mode: str = "session"
    cache_root: str | Path = "/Users/home/thesis/data/cache_v1"
    cache_mode: str = "drive_direct"
    local_cache_base: str | Path = "/content/utah_ssl_cache"
    use_normalization: bool = True
    precomputed_session_stats_path: str | Path | None = None
    precomputed_split_stats_path: str | Path | None = None
    normalization_mode: str = "global"
    segment_bins: int = 256
    batch_size: int = 16
    ssl_steps: int = 1000
    ctc_steps: int = 1000
    run_downstream_ctc: bool = True
    learning_rate: float = 3e-4
    ctc_learning_rate: float = 1e-3
    weight_decay: float = 1e-2
    max_grad_norm: float = 1.0
    hidden_size: int = 256
    state_size: int = 64
    num_layers: int = 4
    dropout: float = 0.1
    direction: str = "causal"
    ffn_multiplier: float = 2.0
    patch_size: int = 14
    patch_stride: int = 4
    patch_policy: str = "floor"
    conv_kernel_size: int = 14
    conv_stride: int = 4
    mask_time_ratio: float = 0.25
    mask_channel_ratio: float = 0.10
    mask_chunk_size: int = 4
    ctc_input_smoothing_sigma_bins: float = 2.0
    ctc_input_smoothing_kernel_size: int = 100
    ctc_input_smoothing_threshold: float = 0.01
    ctc_white_noise_sd: float = 1.0
    ctc_constant_offset_sd: float = 0.2
    val_every_steps: int = 100
    val_batches: int = 4
    progress_every_steps: int = 25
    progress_every_seconds: float = 30.0
    checkpoint_every_steps: int | None = None
    output_root: str | Path = "experiments/bit_style_runs"
    run_name: str | None = None

    def __post_init__(self) -> None:
        if self.backbone_type not in {"s5", "mamba"}:
            raise ValueError("backbone_type must be one of {'s5', 'mamba'}")
        if self.input_mode not in {"raw_bin", "temporal_patch", "causal_conv_stem"}:
            raise ValueError("input_mode must be one of {'raw_bin', 'temporal_patch', 'causal_conv_stem'}")
        if self.objective != "masked_time_channel_reconstruction":
            raise ValueError("objective must currently be 'masked_time_channel_reconstruction'")
        self.dataset_plan = DatasetPlan.from_value(self.dataset_plan)
        self.signal_spec = SignalSpec.from_value(self.signal_spec)
        if self.boundary_key_mode not in {"session", "subject_if_available"}:
            raise ValueError("boundary_key_mode must be one of {'session', 'subject_if_available'}")
        if self.cache_mode not in {"copy_to_local", "drive_direct"}:
            raise ValueError("cache_mode must be one of {'copy_to_local', 'drive_direct'}")
        if self.normalization_mode not in {"global", "per_session", "block", "none"}:
            raise ValueError("normalization_mode must be one of {'global', 'per_session', 'block', 'none'}")
        if int(self.batch_size) <= 0:
            raise ValueError("batch_size must be positive")
        if int(self.ssl_steps) <= 0 or int(self.ctc_steps) <= 0:
            raise ValueError("ssl_steps and ctc_steps must be positive")
        if float(self.learning_rate) <= 0.0 or float(self.ctc_learning_rate) <= 0.0:
            raise ValueError("learning rates must be positive")
        if float(self.weight_decay) < 0.0 or float(self.max_grad_norm) <= 0.0:
            raise ValueError("weight_decay must be non-negative and max_grad_norm positive")
        if int(self.hidden_size) <= 0 or int(self.state_size) <= 0 or int(self.num_layers) <= 0:
            raise ValueError("hidden_size, state_size, and num_layers must be positive")
        if self.direction not in {"causal", "bidirectional"}:
            raise ValueError("direction must be one of {'causal', 'bidirectional'}")
        if self.backbone_type == "mamba" and self.direction != "causal":
            raise ValueError("Mamba experiments currently require direction='causal'")
        if int(self.patch_size) <= 0 or int(self.patch_stride) <= 0:
            raise ValueError("patch_size and patch_stride must be positive")
        if self.patch_policy not in {"floor", "cover_tail"}:
            raise ValueError("patch_policy must be one of {'floor', 'cover_tail'}")
        if int(self.conv_kernel_size) <= 0 or int(self.conv_stride) <= 0:
            raise ValueError("conv_kernel_size and conv_stride must be positive")
        if not (0.0 <= float(self.mask_time_ratio) <= 1.0):
            raise ValueError("mask_time_ratio must be in [0, 1]")
        if not (0.0 <= float(self.mask_channel_ratio) <= 1.0):
            raise ValueError("mask_channel_ratio must be in [0, 1]")
        if int(self.mask_chunk_size) <= 0:
            raise ValueError("mask_chunk_size must be positive")
        if float(self.ctc_input_smoothing_sigma_bins) < 0.0:
            raise ValueError("ctc_input_smoothing_sigma_bins must be non-negative")
        if int(self.ctc_input_smoothing_kernel_size) <= 0:
            raise ValueError("ctc_input_smoothing_kernel_size must be positive")
        if float(self.ctc_input_smoothing_threshold) < 0.0:
            raise ValueError("ctc_input_smoothing_threshold must be non-negative")
        if float(self.ctc_white_noise_sd) < 0.0:
            raise ValueError("ctc_white_noise_sd must be non-negative")
        if float(self.ctc_constant_offset_sd) < 0.0:
            raise ValueError("ctc_constant_offset_sd must be non-negative")

    @property
    def input_dim(self) -> int:
        return int(self.signal_spec.full_dim)

    @property
    def feature_mode(self) -> str:
        return self.signal_spec.mode

    @property
    def tx_dim(self) -> int:
        return int(self.signal_spec.tx_dim)

    @property
    def sbp_dim(self) -> int:
        return int(self.signal_spec.sbp_dim)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["dataset_plan"] = self.dataset_plan.to_dict()
        payload["signal_spec"] = self.signal_spec.to_dict()
        for key in (
            "cache_root",
            "local_cache_base",
            "precomputed_session_stats_path",
            "precomputed_split_stats_path",
            "output_root",
        ):
            if payload[key] is not None:
                payload[key] = str(payload[key])
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "GenericSSMSSLConfig":
        values = dict(payload)
        if "dataset_plan" in values:
            values["dataset_plan"] = DatasetPlan.from_value(values["dataset_plan"])
        if "signal_spec" in values:
            values["signal_spec"] = SignalSpec.from_value(values["signal_spec"])
        return cls(**values)


__all__ = ["GenericSSMSSLConfig"]

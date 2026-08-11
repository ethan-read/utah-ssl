"""Configuration for cross-trained area-6v Mamba decoding runs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class CrossTrainedMambaConfig:
    seed: int = 7
    datasets: tuple[str, ...] = ("brain2text24", "brain2text25")
    feature_mode: str = "tx_sbp"
    area6v_feature_dim: int = 128
    batch_size: int = 64
    max_steps: int = 60000
    learning_rate: float = 1e-3
    min_learning_rate: float = 1e-5
    warmup_steps: int = 1000
    weight_decay: float = 1e-5
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 10.0
    val_every_steps: int = 100
    checkpoint_every_steps: int = 500
    checkpoint_keep_last: int | None = 2
    progress_every_steps: int = 25
    progress_every_seconds: float = 30.0
    hidden_size: int = 512
    state_size: int = 64
    stage1_num_layers: int = 2
    stage2_num_layers: int = 2
    stage3_num_layers: int = 1
    dropout: float = 0.1
    ffn_multiplier: float = 2.0
    adapter_mode: str = "affine"
    session_adapter_enabled: bool = True
    feedback_detach: bool = False
    intermediate_ctc_weight: float = 0.3
    input_smoothing_sigma_bins: float = 2.0
    input_smoothing_kernel_size: int = 100
    input_smoothing_threshold: float = 0.01
    white_noise_sd: float = 1.0
    constant_offset_sd: float = 0.2
    cache_root: str | Path = "/Users/home/thesis/data/cache_v1"
    output_root: str | Path = "experiments/archive/cross_trained_mamba_runs"
    run_name: str | None = None
    resume_checkpoint_path: str | Path | None = None
    resume_latest: bool = False

    def __post_init__(self) -> None:
        self.datasets = tuple(str(name).strip() for name in self.datasets if str(name).strip())
        if not self.datasets:
            raise ValueError("datasets must contain at least one dataset.")
        if self.feature_mode not in {"tx_only", "tx_sbp"}:
            raise ValueError("feature_mode must be one of {'tx_only', 'tx_sbp'}")
        if int(self.area6v_feature_dim) <= 0:
            raise ValueError("area6v_feature_dim must be positive")
        if int(self.batch_size) <= 0 or int(self.max_steps) <= 0:
            raise ValueError("batch_size and max_steps must be positive")
        if float(self.learning_rate) <= 0.0 or float(self.min_learning_rate) < 0.0:
            raise ValueError("learning rates must be non-negative and learning_rate must be positive")
        if int(self.warmup_steps) < 0:
            raise ValueError("warmup_steps must be non-negative")
        if float(self.weight_decay) < 0.0 or float(self.adam_epsilon) <= 0.0:
            raise ValueError("weight_decay must be non-negative and adam_epsilon positive")
        if float(self.max_grad_norm) <= 0.0:
            raise ValueError("max_grad_norm must be positive")
        if int(self.hidden_size) <= 0 or int(self.state_size) <= 0:
            raise ValueError("hidden_size and state_size must be positive")
        if min(int(self.stage1_num_layers), int(self.stage2_num_layers), int(self.stage3_num_layers)) <= 0:
            raise ValueError("All stage layer counts must be positive")
        if float(self.dropout) < 0.0 or float(self.ffn_multiplier) <= 0.0:
            raise ValueError("dropout must be non-negative and ffn_multiplier positive")
        if self.adapter_mode not in {"affine", "stanford_input_net"}:
            raise ValueError("adapter_mode must be one of {'affine', 'stanford_input_net'}")
        if float(self.intermediate_ctc_weight) < 0.0:
            raise ValueError("intermediate_ctc_weight must be non-negative")
        if self.resume_checkpoint_path is not None and not str(self.resume_checkpoint_path).strip():
            self.resume_checkpoint_path = None

    @property
    def input_dim(self) -> int:
        if self.feature_mode == "tx_only":
            return int(self.area6v_feature_dim)
        return int(self.area6v_feature_dim) * 2

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["datasets"] = list(self.datasets)
        for key in ("cache_root", "output_root", "resume_checkpoint_path"):
            if payload[key] is not None:
                payload[key] = str(payload[key])
        return payload


"""Configuration contract for Willett-style supervised decoders."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from utah_ssl.feature_contract import SUPPORTED_FEATURE_MODES


@dataclass
class WillettReconstructionConfig:
    seed: int = 7
    dataset: str = "brain2text24"
    feature_mode: str = "tx_only"
    boundary_key_mode: str = "session"
    split_policy: str = "competition_train_test"
    cv_num_folds: int = 5
    cv_fold_index: int = 0
    normalization_mode: str = "global"
    batch_size: int = 64
    max_steps: int = 120000
    learning_rate: float = 1e-2
    min_learning_rate: float = 1e-4
    warmup_steps: int = 1000
    weight_decay: float = 1e-5
    adam_epsilon: float = 1e-1
    max_grad_norm: float = 10.0
    val_every_steps: int = 100
    checkpoint_every_steps: int = 500
    checkpoint_keep_last: int | None = 2
    progress_every_steps: int = 25
    input_projection_size: int = 256
    input_projection_dropout: float = 0.2
    decoder_backbone_type: str = "gru"
    gru_hidden_size: int = 512
    gru_num_layers: int = 5
    gru_dropout: float = 0.4
    s5_hidden_size: int = 512
    s5_state_size: int = 128
    s5_num_layers: int = 5
    s5_dropout: float = 0.2
    s5_direction: str = "causal"
    s5_ffn_multiplier: float = 2.0
    s4d_hidden_size: int = 512
    s4d_state_size: int = 128
    s4d_num_layers: int = 5
    s4d_dropout: float = 0.2
    s4d_direction: str = "causal"
    s4d_ffn_multiplier: float = 2.0
    patch_size: int = 14
    patch_stride: int = 4
    session_adapter_enabled: bool = True
    input_feature_source: str = "raw"
    predicted_export_root: str | Path | None = None
    input_smoothing_sigma_bins: float = 2.0
    input_smoothing_kernel_size: int = 100
    input_smoothing_threshold: float = 0.01
    white_noise_sd: float = 1.0
    constant_offset_sd: float = 0.2
    precomputed_split_stats_path: str | Path | None = None
    output_root: str | Path = "experiments/supervised_baselines_runs"
    run_name: str | None = None
    cache_root: str | Path = "/Users/home/thesis/data/cache_v1"
    resume_checkpoint_path: str | Path | None = None
    resume_latest: bool = False

    def __post_init__(self) -> None:
        if self.feature_mode not in SUPPORTED_FEATURE_MODES:
            raise ValueError(f"feature_mode must be one of {SUPPORTED_FEATURE_MODES}")
        if self.boundary_key_mode not in {"session", "subject_if_available"}:
            raise ValueError("boundary_key_mode must be one of {'session', 'subject_if_available'}")
        if self.split_policy not in {"competition_train_test", "competition_train_kfold", "source_train_val"}:
            raise ValueError(
                "split_policy must be one of "
                "{'competition_train_test', 'competition_train_kfold', 'source_train_val'}"
            )
        if int(self.cv_num_folds) < 2:
            raise ValueError("cv_num_folds must be at least 2")
        if int(self.cv_fold_index) < 0 or int(self.cv_fold_index) >= int(self.cv_num_folds):
            raise ValueError("cv_fold_index must satisfy 0 <= cv_fold_index < cv_num_folds")
        if self.normalization_mode not in {"block", "global", "per_session", "none"}:
            raise ValueError("normalization_mode must be one of {'block', 'global', 'per_session', 'none'}")
        if int(self.batch_size) <= 0 or int(self.max_steps) <= 0:
            raise ValueError("batch_size and max_steps must be positive")
        if float(self.learning_rate) <= 0.0 or float(self.min_learning_rate) < 0.0:
            raise ValueError("learning rates must be non-negative and max lr must be positive")
        if int(self.warmup_steps) < 0:
            raise ValueError("warmup_steps must be non-negative")
        if int(self.patch_size) <= 0 or int(self.patch_stride) <= 0:
            raise ValueError("patch_size and patch_stride must be positive")
        if int(self.input_projection_size) <= 0:
            raise ValueError("input_projection_size must be positive")
        if self.decoder_backbone_type not in {"gru", "s5", "s4d"}:
            raise ValueError("decoder_backbone_type must be one of {'gru', 's5', 's4d'}")
        if self.input_feature_source not in {"raw", "raw_plus_predicted_tx"}:
            raise ValueError("input_feature_source must be one of {'raw', 'raw_plus_predicted_tx'}")
        if self.input_feature_source == "raw_plus_predicted_tx":
            if self.predicted_export_root is None:
                raise ValueError("predicted_export_root is required when input_feature_source='raw_plus_predicted_tx'")
            if self.feature_mode != "tx_only":
                raise ValueError("raw_plus_predicted_tx currently requires feature_mode='tx_only'")
        if int(self.gru_hidden_size) <= 0 or int(self.gru_num_layers) <= 0:
            raise ValueError("GRU sizes must be positive")
        if int(self.s5_hidden_size) <= 0 or int(self.s5_state_size) <= 0 or int(self.s5_num_layers) <= 0:
            raise ValueError("S5 sizes must be positive")
        if self.s5_direction not in {"causal", "bidirectional"}:
            raise ValueError("s5_direction must be one of {'causal', 'bidirectional'}")
        if float(self.s5_ffn_multiplier) <= 0.0:
            raise ValueError("s5_ffn_multiplier must be positive")
        if int(self.s4d_hidden_size) <= 0 or int(self.s4d_state_size) <= 0 or int(self.s4d_num_layers) <= 0:
            raise ValueError("S4D sizes must be positive")
        if self.s4d_direction not in {"causal", "bidirectional"}:
            raise ValueError("s4d_direction must be one of {'causal', 'bidirectional'}")
        if float(self.s4d_ffn_multiplier) <= 0.0:
            raise ValueError("s4d_ffn_multiplier must be positive")


__all__ = ["WillettReconstructionConfig"]

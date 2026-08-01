"""Stage-2 POSSM phoneme fine-tuning helpers."""

from __future__ import annotations

import copy
import json
import math
import time
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from ssl_core.experiment_contract import SignalSpec

from ssl_core.cache import (
    load_cache_smoothing_provenance,
    resolve_boundary_key,
)
from masked_ssl.probe import (
    CanonicalSequenceDataset,
    LengthAwareBatchSampler,
    build_competition_split_problem,
    build_source_split_problem,
    canonical_rows_padded_time_percentile,
    collate_sequence_batch,
    compute_ctc_loss_sum,
    compute_feature_stats,
)
from ssl_core.stats import (
    load_precomputed_split_feature_stats,
    resolve_precomputed_split_stats_path,
)

from .model import POSSMEncoder, POSSMPhonemeModel, build_temporal_backbone
from .precision import (
    PhaseTimer,
    PrecisionRuntime,
    autocast_context,
    build_adamw,
    resolve_precision_runtime,
    validate_precision,
)
from .training import (
    find_latest_possm_step_checkpoint,
    prune_possm_resumable_checkpoints,
    resolve_latest_possm_checkpoint_path,
)


@dataclass
class POSSMFinetuneConfig:
    seed: int = 7
    mode: str = "finetune_full"
    init_source: str = "stage1"
    dataset: str = "brain2text24"
    split_policy: str = "competition_train_test"
    signal_spec: SignalSpec | dict[str, Any] | None = None
    data_mode: str | None = None
    boundary_key_mode: str | None = None
    precision: str = "float32"
    batch_size: int = 8
    num_steps: int = 5000
    learning_rate: float = 1e-3
    encoder_learning_rate: float = 3e-4
    weight_decay: float = 1e-2
    max_grad_norm: float = 1.0
    val_every_steps: int = 100
    checkpoint_every_steps: int = 200
    checkpoint_keep_last: int | None = 2
    progress_every_steps: int = 25
    benchmark_warmup_steps: int = 0
    benchmark_measure_steps: int = 0
    session_adapter_enabled: bool = True
    input_smoothing_sigma_bins: float = 0.0
    input_smoothing_kernel_size: int = 100
    input_smoothing_threshold: float = 0.01
    white_noise_sd: float = 0.0
    constant_offset_sd: float = 0.0
    gru_hidden_size: int = 768
    gru_num_layers: int = 5
    gru_dropout: float = 0.4
    decoder_backbone_type: str = "gru"
    s5_hidden_size: int = 768
    s5_state_size: int = 128
    s5_num_layers: int = 5
    s5_dropout: float = 0.2
    s5_direction: str = "causal"
    s5_ffn_multiplier: float = 2.0
    emission_mode: str = "post_decoder_conv"
    pre_decoder_patch_size: int = 14
    pre_decoder_patch_stride: int = 4
    conv_hidden_size: int | None = None
    conv_kernel_size: int = 14
    conv_stride: int = 4
    conv_dropout: float = 0.1
    precomputed_split_stats_path: str | Path | None = None

    def __post_init__(self) -> None:
        self.conv_kernel_size = int(self.conv_kernel_size)
        self.conv_stride = int(self.conv_stride)
        if self.mode not in {"probe_frozen", "finetune_full"}:
            raise ValueError("mode must be one of {'probe_frozen', 'finetune_full'}")
        if self.init_source not in {"stage1", "random"}:
            raise ValueError("init_source must be one of {'stage1', 'random'}")
        if self.split_policy not in {"competition_train_test", "source_train_val"}:
            raise ValueError(
                "split_policy must be one of {'competition_train_test', 'source_train_val'}"
            )
        if self.signal_spec is not None:
            self.signal_spec = SignalSpec.from_value(self.signal_spec)
        if self.data_mode is not None and self.data_mode not in {"raw", "normalized"}:
            raise ValueError("data_mode must be one of {'raw', 'normalized'} when provided")
        if self.boundary_key_mode is not None and self.boundary_key_mode not in {
            "session",
            "subject_if_available",
        }:
            raise ValueError(
                "boundary_key_mode must be one of {'session', 'subject_if_available'} when provided"
            )
        self.precision = validate_precision(self.precision)
        if int(self.batch_size) <= 0 or int(self.num_steps) <= 0:
            raise ValueError("batch_size and num_steps must be positive")
        if float(self.learning_rate) <= 0.0 or float(self.encoder_learning_rate) <= 0.0:
            raise ValueError("learning rates must be positive")
        if float(self.weight_decay) < 0.0:
            raise ValueError("weight_decay must be non-negative")
        if float(self.max_grad_norm) <= 0.0:
            raise ValueError("max_grad_norm must be positive")
        if int(self.val_every_steps) <= 0:
            raise ValueError("val_every_steps must be positive")
        if int(self.checkpoint_every_steps) <= 0:
            raise ValueError("checkpoint_every_steps must be positive")
        if self.checkpoint_keep_last is not None and int(self.checkpoint_keep_last) < 0:
            raise ValueError("checkpoint_keep_last must be non-negative when provided")
        if int(self.progress_every_steps) <= 0:
            raise ValueError("progress_every_steps must be positive")
        if int(self.benchmark_warmup_steps) < 0 or int(self.benchmark_measure_steps) < 0:
            raise ValueError("benchmark warm-up and measurement steps must be non-negative")
        if int(self.benchmark_warmup_steps) > 0 and int(self.benchmark_measure_steps) <= 0:
            raise ValueError("benchmark_measure_steps must be positive when warm-up is requested")
        if float(self.input_smoothing_sigma_bins) < 0.0:
            raise ValueError("input_smoothing_sigma_bins must be non-negative")
        if int(self.input_smoothing_kernel_size) <= 0:
            raise ValueError("input_smoothing_kernel_size must be positive")
        if not (0.0 <= float(self.input_smoothing_threshold) < 1.0):
            raise ValueError("input_smoothing_threshold must be in [0, 1)")
        if float(self.white_noise_sd) < 0.0 or float(self.constant_offset_sd) < 0.0:
            raise ValueError("input augmentation standard deviations must be non-negative")
        if self.decoder_backbone_type not in {"gru", "s5"}:
            raise ValueError("decoder_backbone_type must be one of {'gru', 's5'}")
        if self.emission_mode not in {"post_decoder_conv", "pre_decoder_patch"}:
            raise ValueError("emission_mode must be one of {'post_decoder_conv', 'pre_decoder_patch'}")
        if int(self.pre_decoder_patch_size) <= 0 or int(self.pre_decoder_patch_stride) <= 0:
            raise ValueError("pre-decoder patch size and stride must be positive")
        if int(self.gru_hidden_size) <= 0 or int(self.gru_num_layers) <= 0:
            raise ValueError("GRU sizes must be positive")
        if not (0.0 <= float(self.gru_dropout) < 1.0):
            raise ValueError("gru_dropout must be in [0, 1)")
        if int(self.s5_hidden_size) <= 0 or int(self.s5_state_size) <= 0 or int(self.s5_num_layers) <= 0:
            raise ValueError("S5 sizes must be positive")
        if not (0.0 <= float(self.s5_dropout) < 1.0):
            raise ValueError("s5_dropout must be in [0, 1)")
        if self.s5_direction not in {"causal", "bidirectional"}:
            raise ValueError("s5_direction must be one of {'causal', 'bidirectional'}")
        if float(self.s5_ffn_multiplier) <= 0.0:
            raise ValueError("s5_ffn_multiplier must be positive")
        if int(self.conv_kernel_size) <= 0 or int(self.conv_stride) <= 0:
            raise ValueError("Conv kernel and stride must be positive")
        if not (0.0 <= float(self.conv_dropout) < 1.0):
            raise ValueError("conv_dropout must be in [0, 1)")

    @property
    def feature_mode(self) -> str | None:
        return None if self.signal_spec is None else self.signal_spec.mode

def _seed_all(seed: int) -> None:
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _capture_torch_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {"cpu": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_torch_rng_state(state: dict[str, Any] | None) -> None:
    if not state:
        return
    if state.get("cpu") is not None:
        torch.set_rng_state(state["cpu"])
    if torch.cuda.is_available() and state.get("cuda") is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


def _loader_kwargs(device: torch.device) -> dict[str, Any]:
    return {
        "num_workers": 0,
        "pin_memory": device.type == "cuda",
        "collate_fn": collate_sequence_batch,
    }


def _count_trainable_parameters(module: torch.nn.Module) -> int:
    return int(sum(param.numel() for param in module.parameters() if param.requires_grad))


def _count_trainable_sequence_encoder_parameters(model: POSSMPhonemeModel) -> int:
    total = _count_trainable_parameters(model.base_encoder)
    if model.pre_decoder_backbone is not None:
        total += _count_trainable_parameters(model.pre_decoder_backbone)
    return int(total)


def _emit_progress(progress_log_path: Path | None, **payload: Any) -> None:
    if progress_log_path is None:
        return
    progress_log_path.parent.mkdir(parents=True, exist_ok=True)
    with progress_log_path.open("a") as handle:
        handle.write(json.dumps(payload) + "\n")


def _set_train_mode(
    model: POSSMPhonemeModel,
    *,
    train_encoder: bool,
) -> None:
    if train_encoder:
        model.train()
        return
    model.eval()
    if model.session_adapter_enabled:
        model.session_input_adapter.train()
    model.sequence_decoder.train()
    model.conv.train()
    model.conv_dropout.train()
    model.classifier.train()


def _stage2_decoder_train_modules(
    model: POSSMPhonemeModel,
    *,
    session_adapter_enabled: bool,
) -> tuple[torch.nn.Module, ...]:
    modules: list[torch.nn.Module] = []
    if bool(session_adapter_enabled):
        modules.append(model.session_input_adapter)
    modules.extend((model.sequence_decoder, model.conv, model.classifier))
    return tuple(modules)


def _willett_gaussian_kernel_1d(
    *,
    sigma_bins: float,
    kernel_size: int,
    threshold: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    sigma = float(sigma_bins)
    if sigma <= 0.0:
        return torch.ones((1,), device=device, dtype=dtype)
    kernel_size = int(kernel_size)
    if kernel_size <= 0:
        raise ValueError("kernel_size must be positive")
    center = int(kernel_size // 2)
    positions = torch.arange(kernel_size, device=device, dtype=dtype) - float(center)
    kernel = torch.exp(-0.5 * (positions / sigma).pow(2))
    kernel = kernel / kernel.sum().clamp_min(1e-8)
    keep = kernel > float(threshold)
    if not bool(keep.any().item()):
        keep[center] = True
    kept_positions = torch.nonzero(keep, as_tuple=False).squeeze(1)
    start = int(kept_positions.min().item())
    stop = int(kept_positions.max().item()) + 1
    kernel = kernel[start:stop]
    if kernel.numel() % 2 == 0:
        # Keep SAME-length convolution simple and centered if non-default settings create an even kernel.
        kernel = torch.cat([kernel, kernel.new_zeros((1,))], dim=0)
    return kernel / kernel.sum().clamp_min(1e-8)


def _sequence_mask_from_lengths(
    lengths: torch.Tensor,
    max_time: int,
) -> torch.Tensor:
    return torch.arange(max_time, device=lengths.device).unsqueeze(0) < lengths.unsqueeze(1)


def _smooth_batch_like_willett(
    x: torch.Tensor,
    input_lengths: torch.Tensor,
    *,
    sigma_bins: float,
    kernel_size: int,
    threshold: float,
) -> torch.Tensor:
    if float(sigma_bins) <= 0.0 or int(x.shape[1]) <= 1:
        return x
    kernel = _willett_gaussian_kernel_1d(
        sigma_bins=float(sigma_bins),
        kernel_size=int(kernel_size),
        threshold=float(threshold),
        device=x.device,
        dtype=x.dtype,
    )
    channels = int(x.shape[-1])
    weight = kernel.view(1, 1, -1).expand(channels, 1, -1)
    smoothed = torch.nn.functional.conv1d(
        x.transpose(1, 2),
        weight,
        padding=int(kernel.numel() // 2),
        groups=channels,
    ).transpose(1, 2)
    valid = _sequence_mask_from_lengths(input_lengths.to(x.device), int(x.shape[1]))
    return smoothed * valid.unsqueeze(-1).to(smoothed.dtype)


def _prepare_stage2_inputs(
    x: torch.Tensor,
    input_lengths: torch.Tensor,
    *,
    config: POSSMFinetuneConfig,
    is_training: bool,
) -> torch.Tensor:
    transformed = x
    if is_training and float(config.white_noise_sd) > 0.0:
        transformed = transformed + torch.randn(
            transformed.shape,
            device=transformed.device,
            dtype=transformed.dtype,
        ) * float(config.white_noise_sd)
    if is_training and float(config.constant_offset_sd) > 0.0:
        transformed = transformed + torch.randn(
            (int(transformed.shape[0]), 1, int(transformed.shape[2])),
            device=transformed.device,
            dtype=transformed.dtype,
        ) * float(config.constant_offset_sd)
    return _smooth_batch_like_willett(
        transformed,
        input_lengths,
        sigma_bins=float(config.input_smoothing_sigma_bins),
        kernel_size=int(config.input_smoothing_kernel_size),
        threshold=float(config.input_smoothing_threshold),
    )


def _empty_microbatch_range() -> dict[str, int | None]:
    return {"min": None, "max": None}


def _update_microbatch_range(range_payload: dict[str, int | None], value: int) -> None:
    resolved = int(value)
    current_min = range_payload.get("min")
    current_max = range_payload.get("max")
    range_payload["min"] = resolved if current_min is None else min(int(current_min), resolved)
    range_payload["max"] = resolved if current_max is None else max(int(current_max), resolved)


def _ctc_greedy_decode(
    logits: torch.Tensor,
    token_lengths: torch.Tensor,
    *,
    blank_index: int,
) -> list[list[int]]:
    token_ids = logits.argmax(dim=-1)
    decoded: list[list[int]] = []
    for batch_idx, length in enumerate(token_lengths.tolist()):
        sequence: list[int] = []
        prev_token: int | None = None
        for token in token_ids[batch_idx, :length].tolist():
            if token == blank_index:
                prev_token = None
                continue
            if token != prev_token:
                sequence.append(int(token))
            prev_token = int(token)
        decoded.append(sequence)
    return decoded


def _edit_distance(reference: list[int], hypothesis: list[int]) -> int:
    if not reference:
        return len(hypothesis)
    if not hypothesis:
        return len(reference)
    previous = list(range(len(hypothesis) + 1))
    for ref_idx, ref_token in enumerate(reference, start=1):
        current = [ref_idx]
        for hyp_idx, hyp_token in enumerate(hypothesis, start=1):
            substitution_cost = 0 if ref_token == hyp_token else 1
            current.append(
                min(
                    previous[hyp_idx] + 1,
                    current[hyp_idx - 1] + 1,
                    previous[hyp_idx - 1] + substitution_cost,
                )
            )
        previous = current
    return previous[-1]


def _top_counter_items(counter: Counter[int], *, top_k: int = 10) -> list[list[int]]:
    return [[int(item), int(count)] for item, count in counter.most_common(top_k)]


def evaluate_possm_phoneme_metrics(
    *,
    model: POSSMPhonemeModel,
    loader: DataLoader,
    device: torch.device,
    blank_index: int,
    input_transform_config: POSSMFinetuneConfig | None = None,
    precision_runtime: PrecisionRuntime | None = None,
) -> dict[str, Any]:
    runtime = precision_runtime or resolve_precision_runtime(
        "float32" if input_transform_config is None else input_transform_config.precision,
        device=device,
    )
    non_blocking = torch.device(device).type == "cuda"
    model.eval()
    total_loss_sum = 0.0
    total_targets = 0
    total_edit_distance = 0
    total_reference_tokens = 0
    total_predicted_tokens = 0
    total_frames = 0
    total_blank_frames = 0
    reference_counter: Counter[int] = Counter()
    prediction_counter: Counter[int] = Counter()
    with torch.no_grad():
        for batch in loader:
            input_lengths_cpu = batch["input_lengths"]
            label_lengths_cpu = batch["label_lengths"]
            label_length_values = [int(length) for length in label_lengths_cpu.tolist()]
            target_count_cpu = int(sum(label_length_values))
            x = batch["x"].to(
                device=device,
                dtype=torch.float32,
                non_blocking=non_blocking,
            )
            input_lengths = input_lengths_cpu.to(device, non_blocking=non_blocking)
            labels = batch["labels"].to(device, non_blocking=non_blocking)
            label_lengths = label_lengths_cpu.to(device, non_blocking=non_blocking)
            with autocast_context(runtime):
                if input_transform_config is not None:
                    x = _prepare_stage2_inputs(
                        x,
                        input_lengths,
                        config=input_transform_config,
                        is_training=False,
                    )
                outputs = model(x, input_lengths, session_ids=batch["boundary_keys"])
            logits_fp32 = outputs["logits"].float()
            loss_sum, target_count = compute_ctc_loss_sum(
                logits_fp32,
                outputs["token_lengths"],
                labels,
                label_lengths,
                blank_index=blank_index,
                target_count=target_count_cpu,
                label_length_values=label_length_values,
            )
            total_loss_sum += float(loss_sum.item())
            total_targets += int(target_count)
            predictions = _ctc_greedy_decode(
                logits_fp32,
                outputs["token_lengths"],
                blank_index=blank_index,
            )
            frame_ids = logits_fp32.argmax(dim=-1)
            for row_idx, prediction in enumerate(predictions):
                reference_length = int(label_lengths_cpu[row_idx].item())
                reference = labels[row_idx, :reference_length].tolist()
                token_length = int(outputs["token_lengths"][row_idx].item())
                total_edit_distance += _edit_distance(reference, prediction)
                total_reference_tokens += len(reference)
                total_predicted_tokens += len(prediction)
                total_frames += token_length
                total_blank_frames += int(
                    (frame_ids[row_idx, :token_length] == int(blank_index)).sum().item()
                )
                reference_counter.update(int(token) for token in reference)
                prediction_counter.update(int(token) for token in prediction)
    if total_targets <= 0:
        raise ValueError("Validation target count is zero; cannot compute val_ctc_bpphone.")
    if total_reference_tokens <= 0:
        raise ValueError("Validation reference token count is zero; cannot compute PER.")
    return {
        "val_ctc_bpphone": float(total_loss_sum / total_targets / math.log(2.0)),
        "val_phoneme_error_rate": float(total_edit_distance / total_reference_tokens),
        "collapse_diagnostics": {
            "total_reference_tokens": int(total_reference_tokens),
            "total_predicted_tokens": int(total_predicted_tokens),
            "predicted_to_reference_token_ratio": float(total_predicted_tokens / total_reference_tokens),
            "blank_frame_rate": (
                float(total_blank_frames / total_frames)
                if total_frames > 0
                else float("nan")
            ),
            "reference_top_ids": _top_counter_items(reference_counter),
            "prediction_top_ids": _top_counter_items(prediction_counter),
        },
    }


def _load_stage1_checkpoint(
    checkpoint_path: Path,
    *,
    map_location: str | torch.device,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], Path]:
    payload = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    checkpoint_cfg = dict(payload.get("config", {}))
    if str(payload.get("model_family", checkpoint_cfg.get("model_family", ""))) != "possm":
        raise ValueError("Checkpoint is not a POSSM checkpoint.")
    if str(payload.get("stage", checkpoint_cfg.get("stage", ""))) != "stage1_reconstruction":
        raise ValueError("Checkpoint is not a POSSM Stage-1 reconstruction checkpoint.")
    required_config_fields = {
        "data_mode",
        "boundary_key_mode",
        "dataset_plan",
        "input_dim",
        "model_dim",
        "latent_count",
        "value_encoder_type",
        "value_mlp_hidden_size",
        "ffn_hidden_size",
        "dropout",
        "use_token_norm",
        "signal_spec",
        "temporal_backbone_type",
        "temporal_gru_hidden_size",
        "temporal_gru_num_layers",
        "temporal_gru_dropout",
        "temporal_gru_bidirectional",
        "temporal_backbone_kwargs",
    }
    missing_fields = sorted(required_config_fields - set(checkpoint_cfg))
    if missing_fields:
        raise ValueError(
            "POSSM Stage-1 checkpoint config is incomplete; missing required fields: "
            + ", ".join(missing_fields)
        )
    model_state = payload.get("model_state")
    if model_state is None:
        raise KeyError("Stage-1 POSSM checkpoint is missing 'model_state'.")
    if not isinstance(model_state, dict):
        raise TypeError("Stage-1 POSSM checkpoint 'model_state' must be a state dict.")
    run_dir = checkpoint_path.parent.parent if checkpoint_path.parent.name == "checkpoints" else checkpoint_path.parent
    return payload, checkpoint_cfg, model_state, run_dir


def _build_stage1_encoder_from_checkpoint_state(
    *,
    checkpoint_cfg: dict[str, Any],
    model_state: dict[str, Any],
) -> POSSMEncoder:
    encoder = _build_stage1_encoder_from_checkpoint_config(checkpoint_cfg=checkpoint_cfg)
    encoder_state = {
        key.split("encoder.", 1)[1]: value
        for key, value in model_state.items()
        if key.startswith("encoder.")
    }
    if not encoder_state:
        raise KeyError("Stage-1 POSSM checkpoint does not contain encoder weights.")
    encoder.load_state_dict(encoder_state)
    return encoder


def _build_stage1_encoder_from_checkpoint_config(
    *,
    checkpoint_cfg: dict[str, Any],
) -> POSSMEncoder:
    return POSSMEncoder(
        input_dim=int(checkpoint_cfg["input_dim"]),
        model_dim=int(checkpoint_cfg["model_dim"]),
        latent_count=int(checkpoint_cfg["latent_count"]),
        value_encoder_type=str(checkpoint_cfg["value_encoder_type"]),
        value_mlp_hidden_size=(
            None
            if checkpoint_cfg.get("value_mlp_hidden_size") is None
            else int(checkpoint_cfg["value_mlp_hidden_size"])
        ),
        ffn_hidden_size=int(checkpoint_cfg["ffn_hidden_size"]),
        dropout=float(checkpoint_cfg["dropout"]),
        use_token_norm=bool(checkpoint_cfg["use_token_norm"]),
        signal_spec=SignalSpec.from_value(checkpoint_cfg["signal_spec"]),
    )


def _build_stage1_temporal_backbone_from_checkpoint_config(
    *,
    checkpoint_cfg: dict[str, Any],
    input_size: int,
) -> torch.nn.Module:
    return build_temporal_backbone(
        backbone_type=str(checkpoint_cfg["temporal_backbone_type"]),
        input_size=int(input_size),
        gru_hidden_size=(
            None
            if checkpoint_cfg.get("temporal_gru_hidden_size") is None
            else int(checkpoint_cfg["temporal_gru_hidden_size"])
        ),
        gru_num_layers=int(checkpoint_cfg["temporal_gru_num_layers"]),
        gru_dropout=float(checkpoint_cfg["temporal_gru_dropout"]),
        gru_bidirectional=bool(checkpoint_cfg["temporal_gru_bidirectional"]),
        backbone_kwargs=dict(checkpoint_cfg["temporal_backbone_kwargs"]),
    )


def recover_possm_stage1_encoder(
    *,
    checkpoint_path: Path,
    map_location: str | torch.device = "cpu",
) -> tuple[POSSMEncoder, dict[str, Any], Path]:
    _, checkpoint_cfg, model_state, run_dir = _load_stage1_checkpoint(
        checkpoint_path,
        map_location=map_location,
    )
    encoder = _build_stage1_encoder_from_checkpoint_state(
        checkpoint_cfg=checkpoint_cfg,
        model_state=model_state,
    )
    return encoder, checkpoint_cfg, run_dir


def recover_possm_stage1_sequence_components(
    *,
    checkpoint_path: Path,
    map_location: str | torch.device = "cpu",
) -> tuple[POSSMEncoder, torch.nn.Module, dict[str, Any], Path]:
    _, checkpoint_cfg, model_state, run_dir = _load_stage1_checkpoint(
        checkpoint_path,
        map_location=map_location,
    )
    encoder = _build_stage1_encoder_from_checkpoint_state(
        checkpoint_cfg=checkpoint_cfg,
        model_state=model_state,
    )
    temporal_backbone = _build_stage1_temporal_backbone_from_checkpoint_config(
        checkpoint_cfg=checkpoint_cfg,
        input_size=int(encoder.hidden_size),
    )
    temporal_state = {
        key.split("temporal_backbone.", 1)[1]: value
        for key, value in model_state.items()
        if key.startswith("temporal_backbone.")
    }
    if not temporal_state:
        temporal_backbone_type = str(checkpoint_cfg["temporal_backbone_type"])
        if temporal_backbone_type != "identity":
            raise KeyError(
                "Stage-1 POSSM checkpoint declares a temporal backbone but contains no "
                "'temporal_backbone.*' weights."
            )
        return encoder, temporal_backbone, checkpoint_cfg, run_dir
    temporal_backbone.load_state_dict(temporal_state)
    return encoder, temporal_backbone, checkpoint_cfg, run_dir


def initialize_possm_stage2_sequence_components(
    *,
    checkpoint_path: Path,
    init_source: str,
    map_location: str | torch.device = "cpu",
) -> tuple[POSSMEncoder, torch.nn.Module, dict[str, Any], Path]:
    resolved_init_source = str(init_source)
    if resolved_init_source not in {"stage1", "random"}:
        raise ValueError("init_source must be one of {'stage1', 'random'}")
    if resolved_init_source == "stage1":
        return recover_possm_stage1_sequence_components(
            checkpoint_path=checkpoint_path,
            map_location=map_location,
        )
    _, checkpoint_cfg, _, run_dir = _load_stage1_checkpoint(
        checkpoint_path,
        map_location=map_location,
    )
    encoder = _build_stage1_encoder_from_checkpoint_config(checkpoint_cfg=checkpoint_cfg)
    temporal_backbone = _build_stage1_temporal_backbone_from_checkpoint_config(
        checkpoint_cfg=checkpoint_cfg,
        input_size=int(encoder.hidden_size),
    )
    return encoder, temporal_backbone, checkpoint_cfg, run_dir


def _build_problem(
    *,
    cache_root: Path,
    config: POSSMFinetuneConfig,
    signal_spec: SignalSpec,
    boundary_key_mode: str,
) -> dict[str, Any]:
    if config.split_policy == "source_train_val":
        return build_source_split_problem(
            cache_root=cache_root,
            dataset=str(config.dataset),
            signal_spec=signal_spec,
            boundary_key_mode=str(boundary_key_mode),
            train_split_name="train",
            val_split_name="val",
        )
    return build_competition_split_problem(
        cache_root=cache_root,
        dataset=str(config.dataset),
        signal_spec=signal_spec,
        boundary_key_mode=str(boundary_key_mode),
    )


def _session_adapter_keys_for_rows(
    rows: list[Any] | tuple[Any, ...],
    *,
    dataset: str,
    boundary_key_mode: str,
) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            resolve_boundary_key(
                dataset=str(dataset),
                session_id=str(row.session_id),
                subject_id=row.subject_id,
                boundary_key_mode=str(boundary_key_mode),
            )
            for row in rows
        )
    )


def _checkpoint_payload(
    *,
    model: POSSMPhonemeModel,
    optimizer: torch.optim.Optimizer | None,
    resolved_config: POSSMFinetuneConfig,
    resolved_checkpoint_path: Path,
    checkpoint_cfg: dict[str, Any],
    problem: dict[str, Any],
    train_rows: tuple[Any, ...] | list[Any],
    val_rows: tuple[Any, ...] | list[Any],
    session_adapter_keys: tuple[str, ...],
    steps: int,
    metrics: dict[str, Any],
    checkpoint_kind: str,
    elapsed_seconds: float,
    train_iteration_index: int,
    train_batches_consumed: int,
    precision_runtime: PrecisionRuntime | None = None,
    optimizer_fused: bool = False,
    batching_diagnostics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    resolved_batching_diagnostics = dict(batching_diagnostics or {})
    payload: dict[str, Any] = {
        "model_family": "possm",
        "stage": "stage2_phoneme_finetune",
        "stage1_checkpoint_path": str(resolved_checkpoint_path),
        "stage1_checkpoint_config": dict(checkpoint_cfg),
        "config": asdict(resolved_config),
        "precision": (
            precision_runtime.metadata()
            if precision_runtime is not None
            else {
                "requested_precision": str(resolved_config.precision),
                "resolved_precision": str(resolved_config.precision),
                "amp_enabled": str(resolved_config.precision) == "amp_fp16",
                "grad_scaler_enabled": False,
            }
        ),
        "grad_scaler_state": (
            precision_runtime.scaler.state_dict()
            if precision_runtime is not None and precision_runtime.scaler.is_enabled()
            else None
        ),
        "optimizer_fused": bool(optimizer_fused),
        "feature_mode": str(problem["feature_mode"]),
        "data_mode": str(metrics.get("data_mode", resolved_config.data_mode)),
        "dataset": str(problem["dataset"]),
        "cache_root": str(problem["cache_root"]),
        "cache_smoothing_provenance": problem.get("cache_smoothing_provenance"),
        "split_policy": str(problem.get("split_policy", "competition_train_test")),
        "train_split_name": str(problem.get("train_split_name", "competition_train")),
        "val_split_name": str(problem.get("val_split_name", "competition_test")),
        "train_examples": int(len(train_rows)),
        "val_examples": int(len(val_rows)),
        "train_examples_by_session": {
            str(session_id): int(count)
            for session_id, count in dict(problem.get("train_examples_by_session", {})).items()
        },
        "val_examples_by_session": {
            str(session_id): int(count)
            for session_id, count in dict(problem.get("val_examples_by_session", {})).items()
        },
        "train_session_ids": [str(session_id) for session_id in tuple(problem.get("train_session_ids", ()))],
        "val_session_ids": [str(session_id) for session_id in tuple(problem.get("val_session_ids", ()))],
        "session_adapter_enabled": bool(model.session_adapter_enabled),
        "session_adapter_keys": list(session_adapter_keys),
        "encoder_state": model.base_encoder.state_dict(),
        "model_state": model.state_dict(),
        "vocab": problem["vocab"],
        "steps": int(steps),
        "elapsed_seconds": float(elapsed_seconds),
        "metrics": dict(metrics),
        "checkpoint_kind": str(checkpoint_kind),
        "rng_state": _capture_torch_rng_state(),
        "train_batch_position": {
            "iteration_index": int(train_iteration_index),
            "batches_consumed": int(train_batches_consumed),
        },
        "dynamic_batching_enabled": bool(resolved_batching_diagnostics.get("dynamic_batching_enabled", False)),
        "p95_train_input_length": resolved_batching_diagnostics.get("p95_train_input_length"),
        "max_padded_time_per_microbatch": resolved_batching_diagnostics.get("max_padded_time_per_microbatch"),
        "train_microbatch_examples_range": resolved_batching_diagnostics.get("train_microbatch_examples_range"),
        "train_microbatch_max_input_length_range": resolved_batching_diagnostics.get(
            "train_microbatch_max_input_length_range"
        ),
        "batching_diagnostics": resolved_batching_diagnostics,
    }
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
    return payload


def _find_latest_step_checkpoint(checkpoints_dir: Path) -> Path | None:
    return find_latest_possm_step_checkpoint(checkpoints_dir)


def _checkpoint_mtime(path: Path) -> int:
    return int(path.stat().st_mtime_ns) if path.exists() else -1


def _stage2_run_dir_for_checkpoint(path: str | Path) -> Path:
    resolved_path = Path(path)
    return resolved_path.parent.parent if resolved_path.parent.name == "checkpoints" else resolved_path.parent


def find_latest_possm_stage2_run_dir(output_root: str | Path) -> Path:
    """Find the Stage-2 run directory containing the newest checkpoint file."""
    root = Path(output_root)
    if not root.exists():
        raise FileNotFoundError(f"Stage-2 output root does not exist: {root}")
    candidates: list[Path] = []
    for pattern in ("*/checkpoints/step_*.pt", "*/checkpoint_final.pt", "*/checkpoint_best.pt"):
        candidates.extend(path for path in root.glob(pattern) if path.is_file())
    if not candidates:
        raise FileNotFoundError(f"No POSSM Stage-2 checkpoints found under {root}")
    latest_checkpoint = max(candidates, key=_checkpoint_mtime)
    return _stage2_run_dir_for_checkpoint(latest_checkpoint)


def recover_possm_stage2_summary(
    output_root: str | Path,
    *,
    run_dir: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
) -> dict[str, Any]:
    """Recover lightweight metadata for a POSSM Stage-2 run.

    ``resume_checkpoint_path`` is intentionally separate from
    ``checkpoint_final_path``. Interrupted runs often have no final checkpoint,
    but they can still resume from the latest step checkpoint.
    """
    if run_dir is not None and checkpoint_path is not None:
        raise ValueError("Specify at most one of run_dir or checkpoint_path.")

    root = Path(output_root)
    if checkpoint_path is not None:
        resolved_checkpoint_path = Path(checkpoint_path)
        resolved_run_dir = _stage2_run_dir_for_checkpoint(resolved_checkpoint_path)
    else:
        resolved_run_dir = Path(run_dir) if run_dir is not None else find_latest_possm_stage2_run_dir(root)
        try:
            resolved_checkpoint_path = resolve_latest_possm_checkpoint_path(run_dir=resolved_run_dir)
        except RuntimeError as exc:
            raise FileNotFoundError(f"No POSSM Stage-2 checkpoints found for run dir: {resolved_run_dir}") from exc

    if not resolved_checkpoint_path.exists():
        raise FileNotFoundError(f"Stage-2 checkpoint does not exist: {resolved_checkpoint_path}")
    if not resolved_run_dir.exists():
        raise FileNotFoundError(f"Stage-2 run dir does not exist: {resolved_run_dir}")

    payload = torch.load(resolved_checkpoint_path, map_location="cpu", weights_only=False)
    if str(payload.get("stage", "")) != "stage2_phoneme_finetune":
        raise ValueError(f"Checkpoint is not a POSSM Stage-2 checkpoint: {resolved_checkpoint_path}")
    config = dict(payload.get("config", {}))
    metrics = dict(payload.get("metrics", {}))
    checkpoints_dir = resolved_run_dir / "checkpoints"
    best_path = resolved_run_dir / "checkpoint_best.pt"
    final_path = resolved_run_dir / "checkpoint_final.pt"
    progress_path = resolved_run_dir / "progress.jsonl"

    return {
        "run_name": resolved_run_dir.name,
        "run_dir": str(resolved_run_dir),
        "progress_log_path": str(progress_path),
        "checkpoints_dir": str(checkpoints_dir),
        "resume_checkpoint_path": str(resolved_checkpoint_path),
        "checkpoint_best_path": str(best_path) if best_path.exists() else None,
        "checkpoint_final_path": str(final_path) if final_path.exists() else None,
        "stage1_checkpoint_path": payload.get("stage1_checkpoint_path"),
        "cache_root": payload.get("cache_root"),
        "config": config,
        "steps": int(payload.get("steps", 0) or 0),
        "metrics": metrics,
        "decoder_backbone_type": config.get("decoder_backbone_type"),
        "s5_hidden_size": config.get("s5_hidden_size"),
        "s5_state_size": config.get("s5_state_size"),
        "s5_num_layers": config.get("s5_num_layers"),
        "s5_dropout": config.get("s5_dropout"),
        "s5_direction": config.get("s5_direction"),
        "s5_ffn_multiplier": config.get("s5_ffn_multiplier"),
        "emission_mode": config.get("emission_mode", "post_decoder_conv"),
        "pre_decoder_patch_size": config.get("pre_decoder_patch_size"),
        "pre_decoder_patch_stride": config.get("pre_decoder_patch_stride"),
    }


def _validate_stage2_resume_payload(
    payload: dict[str, Any],
    *,
    resolved_config: POSSMFinetuneConfig,
    resolved_checkpoint_path: Path,
) -> None:
    if str(payload.get("stage", "")) != "stage2_phoneme_finetune":
        raise ValueError("Resume checkpoint is not a POSSM stage-2 phoneme fine-tuning checkpoint.")
    payload_stage1 = str(payload.get("stage1_checkpoint_path", ""))
    if payload_stage1 and payload_stage1 != str(resolved_checkpoint_path):
        raise ValueError(
            "Resume checkpoint stage-1 path does not match requested stage-1 checkpoint. "
            f"resume={payload_stage1} requested={resolved_checkpoint_path}"
        )
    raw_payload_config = payload.get("config")
    if not isinstance(raw_payload_config, dict):
        raise TypeError("Resume checkpoint is missing a valid stage-2 config payload.")
    payload_config = asdict(POSSMFinetuneConfig(**raw_payload_config))
    expected_config = asdict(resolved_config)
    if payload_config != expected_config:
        raise ValueError(
            "Resume checkpoint config does not match the requested stage-2 config. "
            "Use a new run directory or the exact same sweep settings to resume."
        )


def _timestamp_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def run_possm_phoneme_finetuning(
    *,
    checkpoint_path: str | Path,
    cache_root: str | Path,
    output_root: str | Path | None = None,
    config: POSSMFinetuneConfig | None = None,
    device: torch.device | None = None,
    run_name: str | None = None,
    resume_from_latest: bool = False,
) -> dict[str, Any]:
    resolved_config = config or POSSMFinetuneConfig()
    _seed_all(int(resolved_config.seed))
    resolved_checkpoint_path = Path(checkpoint_path)
    resolved_cache_root = Path(cache_root)
    resolved_device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base_encoder, pre_decoder_backbone, checkpoint_cfg, stage1_run_dir = initialize_possm_stage2_sequence_components(
        checkpoint_path=resolved_checkpoint_path,
        init_source=str(resolved_config.init_source),
        map_location="cpu",
    )
    if "signal_spec" not in checkpoint_cfg:
        raise ValueError(
            "Stage-1 checkpoint does not contain signal_spec; rerun Stage 1 with "
            "the explicit signal contract before fine-tuning"
        )
    checkpoint_signal_spec = SignalSpec.from_value(checkpoint_cfg["signal_spec"])
    effective_signal_spec = (
        SignalSpec.from_value(resolved_config.signal_spec)
        if resolved_config.signal_spec is not None
        else checkpoint_signal_spec
    )
    if effective_signal_spec != checkpoint_signal_spec:
        raise ValueError(
            "POSSM Stage 2 signal_spec does not match the Stage-1 checkpoint. "
            f"checkpoint={checkpoint_signal_spec.to_dict()}, "
            f"requested={effective_signal_spec.to_dict()}"
        )
    effective_feature_mode = effective_signal_spec.mode
    effective_data_mode = (
        str(resolved_config.data_mode)
        if resolved_config.data_mode is not None
        else str(checkpoint_cfg["data_mode"])
    )
    effective_boundary_key_mode = (
        str(resolved_config.boundary_key_mode)
        if resolved_config.boundary_key_mode is not None
        else str(checkpoint_cfg["boundary_key_mode"])
    )
    effective_config = POSSMFinetuneConfig(
        **{
            **asdict(resolved_config),
            "signal_spec": effective_signal_spec,
            "data_mode": effective_data_mode,
            "boundary_key_mode": effective_boundary_key_mode,
        }
    )
    precision_runtime = resolve_precision_runtime(
        effective_config.precision,
        device=resolved_device,
    )

    problem = _build_problem(
        cache_root=resolved_cache_root,
        config=effective_config,
        signal_spec=effective_signal_spec,
        boundary_key_mode=effective_boundary_key_mode,
    )
    cache_smoothing_provenance = load_cache_smoothing_provenance(
        Path(problem["cache_root"]),
        dataset=str(problem["dataset"]),
    )
    if float(effective_config.input_smoothing_sigma_bins) > 0.0 and cache_smoothing_provenance:
        raise ValueError(
            "POSSM stage-2 online smoothing was requested, but the selected cache root "
            "already declares pre-smoothed features. Use the raw cache root for "
            "Willett-style normalize -> augment -> smooth stage-2 training."
        )
    problem["cache_smoothing_provenance"] = cache_smoothing_provenance

    if effective_data_mode == "normalized":
        resolved_split_stats_path = resolve_precomputed_split_stats_path(
            cache_root=problem["cache_root"],
            dataset=str(problem["dataset"]),
            train_split_name=str(problem.get("train_split_name", "competition_train")),
            signal_spec=effective_signal_spec,
            preferred_path=effective_config.precomputed_split_stats_path,
        )
        (mean_t, std_t), target_stats_metadata, loaded_stats_path = load_precomputed_split_feature_stats(
            stats_path=resolved_split_stats_path,
            cache_root=problem["cache_root"],
            dataset=str(problem["dataset"]),
            signal_spec=effective_signal_spec,
            boundary_key_mode=str(problem.get("boundary_key_mode", effective_boundary_key_mode)),
            train_split_name=str(problem.get("train_split_name", "competition_train")),
            val_split_name=str(problem.get("val_split_name", "competition_test")),
            split_policy=str(problem.get("split_policy", resolved_config.split_policy)),
        )
        target_stats = (
            mean_t.numpy().astype(np.float32, copy=False),
            std_t.numpy().astype(np.float32, copy=False),
        )
        print(f"loaded precomputed POSSM stage-2 split stats: {loaded_stats_path}")
    else:
        target_stats = None
        target_stats_metadata = None
        loaded_stats_path = None

    p95_train_input_length = canonical_rows_padded_time_percentile(
        problem["train_rows"],
        percentile=95.0,
    )
    max_examples_per_microbatch = int(effective_config.batch_size)
    max_padded_time_per_microbatch = int(max_examples_per_microbatch * p95_train_input_length)

    train_dataset = CanonicalSequenceDataset(
        problem["train_rows"],
        cache_root=Path(problem["cache_root"]),
        stats=target_stats,
        signal_spec=problem["signal_spec"],
        boundary_key_mode=str(problem.get("boundary_key_mode", "session")),
        dataset=str(problem.get("dataset", effective_config.dataset)),
    )
    val_dataset = CanonicalSequenceDataset(
        problem["val_rows"],
        cache_root=Path(problem["cache_root"]),
        stats=target_stats,
        signal_spec=problem["signal_spec"],
        boundary_key_mode=str(problem.get("boundary_key_mode", "session")),
        dataset=str(problem.get("dataset", effective_config.dataset)),
    )
    train_batch_sampler = LengthAwareBatchSampler(
        problem["train_rows"],
        max_examples_per_microbatch=max_examples_per_microbatch,
        max_padded_time_per_microbatch=max_padded_time_per_microbatch,
        shuffle=True,
        seed=int(effective_config.seed),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_batch_sampler,
        **_loader_kwargs(resolved_device),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_sampler=LengthAwareBatchSampler(
            problem["val_rows"],
            max_examples_per_microbatch=max_examples_per_microbatch,
            max_padded_time_per_microbatch=max_padded_time_per_microbatch,
            shuffle=False,
            seed=int(effective_config.seed) + 1,
        ),
        **_loader_kwargs(resolved_device),
    )
    batching_diagnostics: dict[str, Any] = {
        "dynamic_batching_enabled": True,
        "p95_train_input_length": int(p95_train_input_length),
        "max_padded_time_per_microbatch": int(max_padded_time_per_microbatch),
        "max_examples_per_microbatch": int(max_examples_per_microbatch),
        "train_microbatch_examples_range": _empty_microbatch_range(),
        "train_microbatch_max_input_length_range": _empty_microbatch_range(),
    }

    vocab = dict(problem["vocab"])
    session_adapter_keys = _session_adapter_keys_for_rows(
        [*problem["train_rows"], *problem["val_rows"]],
        dataset=str(problem.get("dataset", effective_config.dataset)),
        boundary_key_mode=str(problem.get("boundary_key_mode", effective_boundary_key_mode)),
    )
    model = POSSMPhonemeModel(
        base_encoder=copy.deepcopy(base_encoder),
        pre_decoder_backbone=copy.deepcopy(pre_decoder_backbone),
        vocab_size=int(vocab["num_classes"]),
        gru_hidden_size=int(effective_config.gru_hidden_size),
        gru_num_layers=int(effective_config.gru_num_layers),
        gru_dropout=float(effective_config.gru_dropout),
        decoder_backbone_type=str(effective_config.decoder_backbone_type),
        s5_hidden_size=int(effective_config.s5_hidden_size),
        s5_state_size=int(effective_config.s5_state_size),
        s5_num_layers=int(effective_config.s5_num_layers),
        s5_dropout=float(effective_config.s5_dropout),
        s5_direction=str(effective_config.s5_direction),
        s5_ffn_multiplier=float(effective_config.s5_ffn_multiplier),
        emission_mode=str(effective_config.emission_mode),
        pre_decoder_patch_size=int(effective_config.pre_decoder_patch_size),
        pre_decoder_patch_stride=int(effective_config.pre_decoder_patch_stride),
        conv_hidden_size=effective_config.conv_hidden_size,
        conv_kernel_size=int(effective_config.conv_kernel_size),
        conv_stride=int(effective_config.conv_stride),
        conv_dropout=float(effective_config.conv_dropout),
        session_adapter_keys=session_adapter_keys,
        session_adapter_enabled=bool(effective_config.session_adapter_enabled),
    )

    train_encoder = str(effective_config.mode) == "finetune_full"
    for parameter in model.base_encoder.parameters():
        parameter.requires_grad = bool(train_encoder)
    if model.pre_decoder_backbone is not None:
        for parameter in model.pre_decoder_backbone.parameters():
            parameter.requires_grad = bool(train_encoder)
    for parameter in model.session_input_adapter.parameters():
        parameter.requires_grad = bool(effective_config.session_adapter_enabled)
    for parameter in model.sequence_decoder.parameters():
        parameter.requires_grad = True
    for parameter in model.conv.parameters():
        parameter.requires_grad = True
    for parameter in model.classifier.parameters():
        parameter.requires_grad = True
    model.to(resolved_device)

    trainable_groups: list[dict[str, Any]] = [
        {
            "params": [
                param
                for module in _stage2_decoder_train_modules(
                    model,
                    session_adapter_enabled=bool(effective_config.session_adapter_enabled),
                )
                for param in module.parameters()
                if param.requires_grad
            ],
            "lr": float(effective_config.learning_rate),
        }
    ]
    if train_encoder:
        encoder_params = [param for param in model.base_encoder.parameters() if param.requires_grad]
        if encoder_params:
            trainable_groups.append(
                {"params": encoder_params, "lr": float(effective_config.encoder_learning_rate)}
            )
        temporal_backbone_params = (
            []
            if model.pre_decoder_backbone is None
            else [param for param in model.pre_decoder_backbone.parameters() if param.requires_grad]
        )
        if temporal_backbone_params:
            trainable_groups.append(
                {"params": temporal_backbone_params, "lr": float(effective_config.encoder_learning_rate)}
            )
    optimizer, optimizer_fused = build_adamw(
        trainable_groups,
        learning_rate=float(effective_config.learning_rate),
        weight_decay=float(effective_config.weight_decay),
        device=resolved_device,
    )
    clip_params = [param for group in trainable_groups for param in group["params"] if param.requires_grad]

    if output_root is None:
        base_output_root = stage1_run_dir / "phoneme_finetune"
    else:
        base_output_root = Path(output_root)
    base_output_root.mkdir(parents=True, exist_ok=True)

    resolved_run_name = (
        str(run_name)
        if run_name is not None
        else f"possm_stage2_{effective_config.mode}_{effective_feature_mode}_{_timestamp_utc()}"
    )
    if run_name is None and str(effective_config.init_source) == "random":
        resolved_run_name = f"{resolved_run_name}_randominit"
    run_dir = base_output_root / resolved_run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    progress_log_path = run_dir / "progress.jsonl"
    summary_path = run_dir / "summary.json"
    checkpoints_dir = run_dir / "checkpoints"
    checkpoint_best_path = run_dir / "checkpoint_best.pt"
    checkpoint_final_path = run_dir / "checkpoint_final.pt"

    resume_elapsed_seconds = 0.0
    start_time = time.time()
    last_report_elapsed = 0.0
    steps = 0
    last_eval_step = 0
    best_metrics: dict[str, Any] | None = None
    best_payload: dict[str, Any] | None = None
    best_step = 0
    resumed_from_checkpoint: str | None = None
    train_iteration_index = 0
    train_batches_consumed = 0
    resume_rng_state: dict[str, Any] | None = None

    if resume_from_latest:
        latest_checkpoint_path = _find_latest_step_checkpoint(checkpoints_dir)
        if latest_checkpoint_path is not None:
            payload = torch.load(latest_checkpoint_path, map_location="cpu", weights_only=False)
            _validate_stage2_resume_payload(
                payload,
                resolved_config=effective_config,
                resolved_checkpoint_path=resolved_checkpoint_path,
            )
            model_state = payload.get("model_state")
            if not isinstance(model_state, dict):
                raise TypeError(f"Resume checkpoint missing valid model_state: {latest_checkpoint_path}")
            model.load_state_dict(model_state)
            optimizer_state = payload.get("optimizer_state")
            if isinstance(optimizer_state, dict):
                optimizer.load_state_dict(optimizer_state)
            scaler_state = payload.get("grad_scaler_state")
            if precision_runtime.scaler.is_enabled() and isinstance(scaler_state, dict):
                precision_runtime.scaler.load_state_dict(scaler_state)
            steps = int(payload.get("steps", 0))
            last_eval_step = int(steps)
            raw_batch_position = payload.get("train_batch_position")
            if isinstance(raw_batch_position, dict):
                train_iteration_index = int(
                    raw_batch_position.get("iteration_index", 0)
                )
                train_batches_consumed = int(
                    raw_batch_position.get("batches_consumed", 0)
                )
                resume_rng_state = payload.get("rng_state")
            else:
                print(
                    "warning: Stage-2 checkpoint predates exact batch/RNG recovery; "
                    "resuming from the beginning of the deterministic data order."
                )
            resume_elapsed_seconds = float(payload.get("elapsed_seconds", 0.0) or 0.0)
            resumed_from_checkpoint = str(latest_checkpoint_path)
            checkpoint_metrics = payload.get("metrics")
            if isinstance(checkpoint_metrics, dict) and checkpoint_metrics.get("val_ctc_bpphone") is not None:
                best_metrics = dict(checkpoint_metrics)
                best_payload = payload
                best_step = int(steps)
            if checkpoint_best_path.exists():
                best_payload_disk = torch.load(checkpoint_best_path, map_location="cpu", weights_only=False)
                _validate_stage2_resume_payload(
                    best_payload_disk,
                    resolved_config=effective_config,
                    resolved_checkpoint_path=resolved_checkpoint_path,
                )
                disk_metrics = best_payload_disk.get("metrics")
                if isinstance(disk_metrics, dict) and disk_metrics.get("val_ctc_bpphone") is not None:
                    best_payload = best_payload_disk
                    best_metrics = dict(disk_metrics)
                    best_step = int(best_payload_disk.get("steps", best_step))
            _emit_progress(
                progress_log_path,
                event="phoneme_resume",
                stage="possm_phoneme_finetune",
                step=int(steps),
                elapsed_seconds=round(resume_elapsed_seconds, 3),
                resumed_from_checkpoint=resumed_from_checkpoint,
                mode=str(effective_config.mode),
                init_source=str(effective_config.init_source),
                data_mode=effective_data_mode,
                feature_mode=effective_feature_mode,
            )
            start_time = time.time() - resume_elapsed_seconds

    benchmark_warmup_steps = int(effective_config.benchmark_warmup_steps)
    benchmark_measure_steps = int(effective_config.benchmark_measure_steps)
    # Benchmark a fresh process only. A resumed process has no persisted
    # per-step timing samples, so restarting or partially continuing the
    # absolute 1..N window would produce a mislabeled benchmark.
    benchmark_active = benchmark_measure_steps > 0 and steps == 0
    benchmark_warmup_end_step = steps + benchmark_warmup_steps
    benchmark_end_step = benchmark_warmup_end_step + benchmark_measure_steps
    if benchmark_measure_steps > 0 and not benchmark_active:
        print(
            "Skipping Stage-2 benchmark collection on resume; "
            "run a fresh job for a complete warm-up and measurement window."
        )

    latest_eval_metrics: dict[str, Any] | None = None

    def maybe_evaluate(*, force: bool = False) -> dict[str, Any] | None:
        nonlocal last_eval_step, best_metrics, best_payload, best_step, latest_eval_metrics
        if steps <= 0:
            return None
        if not force and benchmark_active and steps <= benchmark_end_step:
            return None
        should_run = force or steps == 1 or steps % int(effective_config.val_every_steps) == 0
        if not should_run or steps == last_eval_step:
            return None
        metrics = evaluate_possm_phoneme_metrics(
            model=model,
            loader=val_loader,
            device=resolved_device,
            blank_index=int(problem["vocab"]["blank_index"]),
            input_transform_config=effective_config,
            precision_runtime=precision_runtime,
        )
        metrics["model_num_parameters"] = _count_trainable_parameters(model)
        metrics["encoder_num_parameters"] = _count_trainable_sequence_encoder_parameters(model)
        collapse = dict(metrics.get("collapse_diagnostics") or {})
        last_eval_step = steps
        latest_eval_metrics = dict(metrics)
        _emit_progress(
            progress_log_path,
            event="phoneme_val_report",
            stage="possm_phoneme_finetune",
            step=int(steps),
            elapsed_seconds=round(time.time() - start_time, 3),
            mode=str(effective_config.mode),
            init_source=str(effective_config.init_source),
            data_mode=effective_data_mode,
            feature_mode=effective_feature_mode,
            blank_frame_rate=collapse.get("blank_frame_rate"),
            predicted_to_reference_token_ratio=collapse.get("predicted_to_reference_token_ratio"),
            optimizer_fused=optimizer_fused,
            **precision_runtime.metadata(),
            **metrics,
        )
        if best_metrics is None or float(metrics["val_ctc_bpphone"]) < float(best_metrics["val_ctc_bpphone"]):
            best_metrics = dict(metrics)
            best_payload = _checkpoint_payload(
                model=model,
                optimizer=optimizer,
                resolved_config=effective_config,
                resolved_checkpoint_path=resolved_checkpoint_path,
                checkpoint_cfg=checkpoint_cfg,
                problem=problem,
                train_rows=problem["train_rows"],
                val_rows=problem["val_rows"],
                session_adapter_keys=session_adapter_keys,
                steps=steps,
                metrics=metrics,
                checkpoint_kind="best",
                elapsed_seconds=round(time.time() - start_time, 3),
                train_iteration_index=train_iteration_index,
                train_batches_consumed=train_batches_consumed,
                precision_runtime=precision_runtime,
                optimizer_fused=optimizer_fused,
                batching_diagnostics=batching_diagnostics,
            )
            best_step = int(steps)
            torch.save(best_payload, checkpoint_best_path)
        return metrics

    def maybe_save_resumable_checkpoint() -> None:
        if benchmark_active and steps <= benchmark_end_step:
            return
        if checkpoints_dir is not None and steps % int(effective_config.checkpoint_every_steps) == 0:
            checkpoints_dir.mkdir(parents=True, exist_ok=True)
            step_metrics = dict(latest_eval_metrics) if latest_eval_metrics is not None else {}
            payload = _checkpoint_payload(
                model=model,
                optimizer=optimizer,
                resolved_config=effective_config,
                resolved_checkpoint_path=resolved_checkpoint_path,
                checkpoint_cfg=checkpoint_cfg,
                problem=problem,
                train_rows=problem["train_rows"],
                val_rows=problem["val_rows"],
                session_adapter_keys=session_adapter_keys,
                steps=steps,
                metrics=step_metrics,
                checkpoint_kind="step",
                elapsed_seconds=round(time.time() - start_time, 3),
                train_iteration_index=train_iteration_index,
                train_batches_consumed=train_batches_consumed,
                precision_runtime=precision_runtime,
                optimizer_fused=optimizer_fused,
                batching_diagnostics=batching_diagnostics,
            )
            step_checkpoint_path = checkpoints_dir / f"step_{int(steps):06d}.pt"
            torch.save(payload, step_checkpoint_path)
            for deleted_checkpoint in prune_possm_resumable_checkpoints(
                checkpoints_dir,
                keep_last=effective_config.checkpoint_keep_last,
            ):
                print("pruned_step_checkpoint:", deleted_checkpoint)

    accumulated_examples = 0
    accumulated_target_count = 0
    accumulated_loss_sum: torch.Tensor | None = None
    accumulation_microbatches = 0
    accumulated_sample_seconds = 0.0
    accumulated_model_seconds = 0.0
    has_pending_gradients = False
    optimizer_step_timer: PhaseTimer | None = None
    optimizer_step_wall_start: float | None = None
    benchmark_step_seconds: list[float] = []
    benchmark_examples = 0

    def flush_pending_gradients(*, force_report: bool = False) -> None:
        nonlocal steps
        nonlocal last_report_elapsed
        nonlocal accumulated_examples
        nonlocal accumulated_target_count
        nonlocal accumulated_loss_sum
        nonlocal accumulation_microbatches
        nonlocal accumulated_sample_seconds
        nonlocal accumulated_model_seconds
        nonlocal has_pending_gradients
        nonlocal optimizer_step_timer
        nonlocal optimizer_step_wall_start
        nonlocal benchmark_examples
        if not has_pending_gradients:
            return
        optimizer_token = optimizer_step_timer.start() if optimizer_step_timer is not None else None
        precision_runtime.scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(clip_params, max_norm=float(effective_config.max_grad_norm))
        precision_runtime.scaler.step(optimizer)
        precision_runtime.scaler.update()
        optimizer.zero_grad(set_to_none=True)
        if optimizer_step_timer is not None:
            optimizer_step_timer.stop("optimizer", optimizer_token)
            phase_seconds = optimizer_step_timer.finish()
        else:
            phase_seconds = {}
        steps += 1
        elapsed = time.time() - start_time
        should_report = force_report or steps == 1 or steps % int(effective_config.progress_every_steps) == 0
        train_ctc_bpphone = float("nan")
        if (
            should_report
            and accumulated_target_count > 0
            and accumulated_loss_sum is not None
        ):
            train_ctc_bpphone = float(
                accumulated_loss_sum.item()
                / accumulated_target_count
                / math.log(2.0)
            )
        measured_model_seconds = sum(phase_seconds.values())
        if phase_seconds:
            accumulated_model_seconds = measured_model_seconds
        complete_step_seconds = (
            time.perf_counter() - optimizer_step_wall_start
            if optimizer_step_wall_start is not None
            else accumulated_sample_seconds + accumulated_model_seconds
        )
        examples_per_second = (
            float(accumulated_examples) / complete_step_seconds
            if complete_step_seconds > 0.0
            else None
        )
        if (
            benchmark_active
            and benchmark_warmup_end_step < steps <= benchmark_end_step
        ):
            benchmark_step_seconds.append(float(complete_step_seconds))
            benchmark_examples += int(accumulated_examples)
        if should_report:
            last_report_elapsed = elapsed
            _emit_progress(
                progress_log_path,
                event="phoneme_train_report",
                stage="possm_phoneme_finetune",
                step=int(steps),
                elapsed_seconds=round(elapsed, 3),
                train_ctc_bpphone=train_ctc_bpphone,
                microbatch_examples=int(accumulated_examples),
                accumulation_microbatches=int(accumulation_microbatches),
                optimizer_target_examples=int(effective_config.batch_size),
                effective_examples_per_step=int(accumulated_examples),
                sample_seconds=round(accumulated_sample_seconds, 4),
                model_seconds=round(accumulated_model_seconds, 4),
                transfer_preprocessing_seconds=round(phase_seconds.get("transfer_preprocessing", 0.0), 4),
                forward_loss_seconds=round(phase_seconds.get("forward_loss", 0.0), 4),
                backward_seconds=round(phase_seconds.get("backward", 0.0), 4),
                optimizer_seconds=round(phase_seconds.get("optimizer", 0.0), 4),
                complete_optimizer_step_seconds=round(complete_step_seconds, 4),
                examples_per_second=(
                    None if examples_per_second is None else round(examples_per_second, 3)
                ),
                peak_cuda_memory_bytes=(
                    int(torch.cuda.max_memory_allocated(resolved_device))
                    if resolved_device.type == "cuda"
                    else None
                ),
                optimizer_fused=optimizer_fused,
                **precision_runtime.metadata(),
                mode=str(effective_config.mode),
                init_source=str(effective_config.init_source),
                data_mode=effective_data_mode,
                feature_mode=effective_feature_mode,
                dynamic_batching_enabled=True,
                max_padded_time_per_microbatch=int(max_padded_time_per_microbatch),
            )
        if benchmark_active and steps == benchmark_end_step:
            measured_seconds = float(sum(benchmark_step_seconds))
            _emit_progress(
                progress_log_path,
                event="phoneme_precision_benchmark",
                stage="possm_phoneme_finetune",
                step=int(steps),
                warmup_steps=benchmark_warmup_steps,
                measured_steps=len(benchmark_step_seconds),
                measured_examples=int(benchmark_examples),
                measured_seconds=round(measured_seconds, 4),
                examples_per_second=(
                    None
                    if measured_seconds <= 0.0
                    else round(float(benchmark_examples) / measured_seconds, 3)
                ),
                optimizer_fused=optimizer_fused,
                **precision_runtime.metadata(),
            )
        accumulated_examples = 0
        accumulated_target_count = 0
        accumulated_loss_sum = None
        accumulation_microbatches = 0
        accumulated_sample_seconds = 0.0
        accumulated_model_seconds = 0.0
        has_pending_gradients = False
        optimizer_step_timer = None
        optimizer_step_wall_start = None

    resume_iteration_batches = train_batch_sampler.num_batches_for_iteration(
        train_iteration_index
    )
    if train_batches_consumed > resume_iteration_batches:
        raise ValueError(
            "Stage-2 checkpoint batch position exceeds the reconstructed "
            f"iteration length: iteration={train_iteration_index}, "
            f"batches_consumed={train_batches_consumed}, "
            f"iteration_batches={resume_iteration_batches}"
        )
    if train_batches_consumed == resume_iteration_batches:
        # The checkpoint was written after the final batch but before the
        # iterator observed StopIteration. Resume directly at the next
        # deterministic permutation.
        if resume_rng_state is not None:
            _restore_torch_rng_state(resume_rng_state)
        resume_rng_state = None
        train_iteration_index += 1
        train_batches_consumed = 0

    while True:
        elapsed = time.time() - start_time
        if steps >= int(effective_config.num_steps):
            break
        made_progress = False
        train_batch_sampler.load_state_dict(
            {"iteration_count": int(train_iteration_index)}
        )
        train_loader_iter = iter(train_loader)
        if train_batches_consumed:
            for _ in range(int(train_batches_consumed)):
                try:
                    next(train_loader_iter)
                except StopIteration as exc:
                    raise ValueError(
                        "Stage-2 checkpoint batch position exceeds the reconstructed "
                        f"iteration length: iteration={train_iteration_index}, "
                        f"batches_consumed={train_batches_consumed}"
                    ) from exc
        if resume_rng_state is not None:
            # Reconstructing and advancing the DataLoader can consume torch RNG.
            # Restore only after reaching the saved cursor.
            _restore_torch_rng_state(resume_rng_state)
            resume_rng_state = None
        while True:
            sample_start = time.perf_counter()
            try:
                batch = next(train_loader_iter)
            except StopIteration:
                break
            train_batches_consumed += 1
            accumulated_sample_seconds += time.perf_counter() - sample_start
            elapsed = time.time() - start_time
            if steps >= int(effective_config.num_steps):
                break
            if train_encoder:
                _set_train_mode(model, train_encoder=True)
            else:
                _set_train_mode(model, train_encoder=False)
            if optimizer_step_timer is None:
                next_step = steps + 1
                profile_step = (
                    next_step == 1
                    or next_step % int(effective_config.progress_every_steps) == 0
                    or (benchmark_active and next_step <= benchmark_end_step)
                )
                optimizer_step_timer = PhaseTimer(resolved_device, enabled=profile_step)
                optimizer_step_wall_start = sample_start if profile_step else None
                if profile_step and resolved_device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(resolved_device)
            transfer_token = (
                optimizer_step_timer.start() if optimizer_step_timer is not None else None
            )
            non_blocking = resolved_device.type == "cuda"
            input_lengths_cpu = batch["input_lengths"]
            label_lengths_cpu = batch["label_lengths"]
            label_length_values = [int(length) for length in label_lengths_cpu.tolist()]
            target_count_cpu = int(sum(label_length_values))
            microbatch_max_input_length = int(input_lengths_cpu.max().item())
            x = batch["x"].to(
                device=resolved_device,
                dtype=torch.float32,
                non_blocking=non_blocking,
            )
            input_lengths = input_lengths_cpu.to(
                resolved_device,
                non_blocking=non_blocking,
            )
            labels = batch["labels"].to(resolved_device, non_blocking=non_blocking)
            label_lengths = label_lengths_cpu.to(
                resolved_device,
                non_blocking=non_blocking,
            )
            with autocast_context(precision_runtime):
                x = _prepare_stage2_inputs(
                    x,
                    input_lengths,
                    config=effective_config,
                    is_training=True,
                )
            if optimizer_step_timer is not None:
                optimizer_step_timer.stop("transfer_preprocessing", transfer_token)
            microbatch_examples = int(x.shape[0])
            _update_microbatch_range(
                batching_diagnostics["train_microbatch_examples_range"],
                microbatch_examples,
            )
            _update_microbatch_range(
                batching_diagnostics["train_microbatch_max_input_length_range"],
                microbatch_max_input_length,
            )

            forward_token = (
                optimizer_step_timer.start() if optimizer_step_timer is not None else None
            )
            with autocast_context(precision_runtime):
                outputs = model(x, input_lengths, session_ids=batch["boundary_keys"])
            logits_fp32 = outputs["logits"].float()
            loss_sum, target_count = compute_ctc_loss_sum(
                logits_fp32,
                outputs["token_lengths"],
                labels,
                label_lengths,
                blank_index=int(problem["vocab"]["blank_index"]),
                target_count=target_count_cpu,
                label_length_values=label_length_values,
            )
            if optimizer_step_timer is not None:
                optimizer_step_timer.stop("forward_loss", forward_token)
            if target_count <= 0:
                continue
            loss = loss_sum / target_count
            if not has_pending_gradients:
                optimizer.zero_grad(set_to_none=True)
            scaled_loss = loss * (float(microbatch_examples) / float(effective_config.batch_size))
            backward_token = (
                optimizer_step_timer.start() if optimizer_step_timer is not None else None
            )
            precision_runtime.scaler.scale(scaled_loss).backward()
            if optimizer_step_timer is not None:
                optimizer_step_timer.stop("backward", backward_token)
            accumulated_examples += microbatch_examples
            accumulated_target_count += int(target_count)
            detached_loss_sum = loss_sum.detach()
            accumulated_loss_sum = (
                detached_loss_sum
                if accumulated_loss_sum is None
                else accumulated_loss_sum + detached_loss_sum
            )
            accumulation_microbatches += 1
            has_pending_gradients = True
            made_progress = True

            if accumulated_examples >= int(effective_config.batch_size):
                flush_pending_gradients()
                maybe_evaluate()
                maybe_save_resumable_checkpoint()
                if steps >= int(effective_config.num_steps):
                    break
        if not made_progress:
            break
        if steps >= int(effective_config.num_steps):
            break
        flush_pending_gradients()
        maybe_evaluate()
        maybe_save_resumable_checkpoint()
        train_iteration_index += 1
        train_batches_consumed = 0

    final_metrics = maybe_evaluate(force=True)
    if final_metrics is None:
        final_metrics = evaluate_possm_phoneme_metrics(
            model=model,
            loader=val_loader,
            device=resolved_device,
            blank_index=int(problem["vocab"]["blank_index"]),
            input_transform_config=effective_config,
            precision_runtime=precision_runtime,
        )
        final_metrics["model_num_parameters"] = _count_trainable_parameters(model)
        final_metrics["encoder_num_parameters"] = _count_trainable_sequence_encoder_parameters(model)
    assert best_payload is not None
    assert best_metrics is not None
    torch.save(best_payload, checkpoint_best_path)
    torch.save(
        _checkpoint_payload(
            model=model,
            optimizer=optimizer,
            resolved_config=effective_config,
            resolved_checkpoint_path=resolved_checkpoint_path,
            checkpoint_cfg=checkpoint_cfg,
            problem=problem,
            train_rows=problem["train_rows"],
            val_rows=problem["val_rows"],
            session_adapter_keys=session_adapter_keys,
            steps=steps,
            metrics=final_metrics,
            checkpoint_kind="final",
            elapsed_seconds=round(time.time() - start_time, 3),
            train_iteration_index=train_iteration_index,
            train_batches_consumed=train_batches_consumed,
            precision_runtime=precision_runtime,
            optimizer_fused=optimizer_fused,
            batching_diagnostics=batching_diagnostics,
        ),
        checkpoint_final_path,
    )

    summary = {
        "run_name": resolved_run_name,
        "run_dir": str(run_dir),
        "progress_log_path": str(progress_log_path),
        "checkpoint_best_path": str(checkpoint_best_path),
        "checkpoint_final_path": str(checkpoint_final_path),
        "checkpoints_dir": str(checkpoints_dir),
        "resumed_from_checkpoint": resumed_from_checkpoint,
        "stage1_checkpoint_path": str(resolved_checkpoint_path),
        "stage1_run_dir": str(stage1_run_dir),
        "mode": str(effective_config.mode),
        **precision_runtime.metadata(),
        "optimizer_fused": bool(optimizer_fused),
        "init_source": str(effective_config.init_source),
        "feature_mode": effective_feature_mode,
        "data_mode": effective_data_mode,
        "boundary_key_mode": effective_boundary_key_mode,
        "dataset": str(effective_config.dataset),
        "cache_root": str(problem["cache_root"]),
        "cache_smoothing_provenance": problem.get("cache_smoothing_provenance"),
        "precomputed_split_stats_path": (
            str(loaded_stats_path) if loaded_stats_path is not None else None
        ),
        "precomputed_split_stats_metadata": target_stats_metadata,
        "split_policy": str(problem.get("split_policy", "competition_train_test")),
        "dynamic_batching_enabled": bool(batching_diagnostics["dynamic_batching_enabled"]),
        "p95_train_input_length": int(batching_diagnostics["p95_train_input_length"]),
        "max_padded_time_per_microbatch": int(batching_diagnostics["max_padded_time_per_microbatch"]),
        "session_adapter_enabled": bool(effective_config.session_adapter_enabled),
        "session_adapter_keys": list(session_adapter_keys),
        "decoder_backbone_type": str(effective_config.decoder_backbone_type),
        "emission_mode": str(effective_config.emission_mode),
        "pre_decoder_patch_size": int(effective_config.pre_decoder_patch_size),
        "pre_decoder_patch_stride": int(effective_config.pre_decoder_patch_stride),
        "s5_hidden_size": int(effective_config.s5_hidden_size),
        "s5_state_size": int(effective_config.s5_state_size),
        "s5_num_layers": int(effective_config.s5_num_layers),
        "s5_dropout": float(effective_config.s5_dropout),
        "s5_direction": str(effective_config.s5_direction),
        "s5_ffn_multiplier": float(effective_config.s5_ffn_multiplier),
        "train_split_name": str(problem.get("train_split_name", "competition_train")),
        "val_split_name": str(problem.get("val_split_name", "competition_test")),
        "train_session_ids": [str(session_id) for session_id in tuple(problem.get("train_session_ids", ()))],
        "val_session_ids": [str(session_id) for session_id in tuple(problem.get("val_session_ids", ()))],
        "train_examples": int(len(problem["train_rows"])),
        "val_examples": int(len(problem["val_rows"])),
        "train_examples_by_session": {
            str(session_id): int(count)
            for session_id, count in dict(problem.get("train_examples_by_session", {})).items()
        },
        "val_examples_by_session": {
            str(session_id): int(count)
            for session_id, count in dict(problem.get("val_examples_by_session", {})).items()
        },
        "train_microbatch_examples_range": dict(batching_diagnostics["train_microbatch_examples_range"]),
        "train_microbatch_max_input_length_range": dict(
            batching_diagnostics["train_microbatch_max_input_length_range"]
        ),
        "steps": int(steps),
        "metrics": {
            "val_ctc_bpphone": float(final_metrics["val_ctc_bpphone"]),
            "val_phoneme_error_rate": float(final_metrics["val_phoneme_error_rate"]),
            "best_val_ctc_bpphone": float(best_metrics["val_ctc_bpphone"]),
            "best_val_phoneme_error_rate": float(best_metrics["val_phoneme_error_rate"]),
            "best_step": int(best_step),
            "model_num_parameters": int(final_metrics["model_num_parameters"]),
            "encoder_num_parameters": int(final_metrics["encoder_num_parameters"]),
            "collapse_diagnostics": final_metrics.get("collapse_diagnostics"),
            "best_collapse_diagnostics": best_metrics.get("collapse_diagnostics"),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    return summary

"""Train a Willett-style phoneme decoder on the canonical Utah cache."""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

try:
    from ssl_core.ctc import CanonicalSequenceDataset, compute_ctc_loss_sum
    from ssl_core.stats import (
        load_precomputed_split_feature_stats,
        resolve_precomputed_split_stats_path,
    )
except ModuleNotFoundError:  # pragma: no cover - repo-root unittest fallback
    from analysis.active.ssl_experiments.ssl_core.ctc import (
        CanonicalSequenceDataset,
        compute_ctc_loss_sum,
    )
    from analysis.active.ssl_experiments.ssl_core.stats import (
        load_precomputed_split_feature_stats,
        resolve_precomputed_split_stats_path,
    )

from .data import (
    ConcatenatedPredictedTxSequenceDataset,
    FuturePredictionExportAccessor,
    WillettInputTransformConfig,
    adapter_keys_from_rows,
    build_willett_problem,
    compute_predicted_tx_normalization_stats,
    compute_willett_normalization_stats,
    group_rows_by_adapter_key,
    loader_kwargs,
    make_length_aware_batch_sampler,
    normalization_stats_missing_rows,
    prepare_willett_inputs,
)
from .model import WillettPhonemeModel
from .reporting import evaluate_willett_phoneme_metrics


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
    output_root: str | Path = "analysis/active/ssl_experiments/willett_reconstruction_runs"
    run_name: str | None = None
    cache_root: str | Path = "/Users/home/thesis/data/cache_v1"
    resume_checkpoint_path: str | Path | None = None
    resume_latest: bool = False

    def __post_init__(self) -> None:
        if self.feature_mode not in {"tx_only", "tx_sbp"}:
            raise ValueError("feature_mode must be one of {'tx_only', 'tx_sbp'}")
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


def _timestamp_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _seed_all(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _detect_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _emit_progress(progress_log_path: Path | None, **payload: Any) -> None:
    if progress_log_path is None:
        return
    progress_log_path.parent.mkdir(parents=True, exist_ok=True)
    with progress_log_path.open("a") as handle:
        handle.write(json.dumps(payload) + "\n")


def _count_trainable_parameters(module: torch.nn.Module) -> int:
    return int(sum(param.numel() for param in module.parameters() if param.requires_grad))


def _build_input_transform_config(config: WillettReconstructionConfig) -> WillettInputTransformConfig:
    return WillettInputTransformConfig(
        input_smoothing_sigma_bins=float(config.input_smoothing_sigma_bins),
        input_smoothing_kernel_size=int(config.input_smoothing_kernel_size),
        input_smoothing_threshold=float(config.input_smoothing_threshold),
        white_noise_sd=float(config.white_noise_sd),
        constant_offset_sd=float(config.constant_offset_sd),
    )


def _make_lr_lambda(
    *,
    warmup_steps: int,
    max_steps: int,
    min_learning_rate: float,
    learning_rate: float,
) -> Any:
    def _schedule(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(max(warmup_steps, 1))
        if max_steps <= warmup_steps:
            return max(float(min_learning_rate / learning_rate), 0.0)
        progress = float(step - warmup_steps) / float(max(max_steps - warmup_steps, 1))
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        min_ratio = float(min_learning_rate / learning_rate)
        return float(min_ratio + (1.0 - min_ratio) * cosine)

    return _schedule


def _resolve_run_dir(config: WillettReconstructionConfig) -> Path:
    output_root = Path(config.output_root)
    resolved_run_name = (
        str(config.run_name)
        if config.run_name is not None
        else f"willett_reconstruction_{config.feature_mode}_{_timestamp_utc()}"
    )
    return output_root / resolved_run_name


def _resolve_resume_checkpoint(run_dir: Path, config: WillettReconstructionConfig) -> Path | None:
    if config.resume_checkpoint_path is not None:
        path = Path(config.resume_checkpoint_path)
        return path if path.exists() else None
    if not bool(config.resume_latest):
        return None
    checkpoints_dir = run_dir / "checkpoints"
    candidates = sorted(checkpoints_dir.glob("step_*.pt"))
    if candidates:
        return candidates[-1]
    fallback = run_dir / "checkpoint_final.pt"
    return fallback if fallback.exists() else None


def _save_checkpoint(
    *,
    checkpoint_path: Path,
    config: WillettReconstructionConfig,
    model: WillettPhonemeModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    step: int,
    metrics: dict[str, Any] | None,
    best_step: int,
    best_progress_payload: dict[str, Any] | None,
    problem: dict[str, Any],
) -> None:
    config_payload = json.loads(json.dumps(asdict(config), default=str))
    payload = {
        "model_family": "willett_reconstruction",
        "config": config_payload,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "step": int(step),
        "cache_root": str(problem["cache_root"]),
        "vocab": dict(problem["vocab"]),
        "metrics": dict(metrics or {}),
        "best_step": int(best_step),
        "best_progress_payload": dict(best_progress_payload) if best_progress_payload is not None else None,
        "train_split_name": str(problem["train_split_name"]),
        "val_split_name": str(problem["val_split_name"]),
        "train_session_ids": list(problem["train_session_ids"]),
        "val_session_ids": list(problem["val_session_ids"]),
    }
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, checkpoint_path)


def _prune_step_checkpoints(checkpoints_dir: Path, keep_last: int | None) -> None:
    if keep_last is None:
        return
    candidates = sorted(checkpoints_dir.glob("step_*.pt"))
    for stale in candidates[:-int(keep_last)]:
        stale.unlink(missing_ok=True)


def run_willett_reconstruction(config: WillettReconstructionConfig) -> dict[str, Any]:
    _seed_all(int(config.seed))
    device = _detect_device()
    problem = build_willett_problem(
        cache_root=Path(config.cache_root),
        dataset=str(config.dataset),
        feature_mode=str(config.feature_mode),
        boundary_key_mode=str(config.boundary_key_mode),
        split_policy=str(config.split_policy),
        cv_num_folds=int(config.cv_num_folds),
        cv_fold_index=int(config.cv_fold_index),
    )
    run_dir = _resolve_run_dir(config)
    run_dir.mkdir(parents=True, exist_ok=True)
    progress_log_path = run_dir / "progress.jsonl"
    summary_path = run_dir / "summary.json"
    checkpoints_dir = run_dir / "checkpoints"
    checkpoint_best_path = run_dir / "checkpoint_best.pt"
    checkpoint_final_path = run_dir / "checkpoint_final.pt"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    base_sample_dim = int(
        problem["train_rows"][0].n_tx_features
        if config.feature_mode == "tx_only"
        else problem["train_rows"][0].n_tx_features + problem["train_rows"][0].n_sbp_features
    )
    export_accessor = (
        FuturePredictionExportAccessor(config.predicted_export_root)
        if str(config.input_feature_source) == "raw_plus_predicted_tx"
        else None
    )
    sample_dim = int(base_sample_dim * 2 if export_accessor is not None else base_sample_dim)
    loaded_stats_path = None
    stats_metadata = None
    if str(config.normalization_mode) == "global" and (
        str(config.split_policy) == "competition_train_test"
        or config.precomputed_split_stats_path is not None
    ):
        resolved_stats_path = resolve_precomputed_split_stats_path(
            cache_root=Path(config.cache_root),
            dataset=str(config.dataset),
            train_split_name=str(problem["train_split_name"]),
            signal_spec=problem["signal_spec"],
            preferred_path=config.precomputed_split_stats_path,
        )
        (mean_t, std_t), stats_metadata, loaded_stats_path = load_precomputed_split_feature_stats(
            stats_path=resolved_stats_path,
            cache_root=Path(problem["cache_root"]),
            dataset=str(problem["dataset"]),
            signal_spec=problem["signal_spec"],
            boundary_key_mode=str(problem["boundary_key_mode"]),
            split_policy=str(problem["split_policy"]),
            train_split_name=str(problem["train_split_name"]),
            val_split_name=str(problem["val_split_name"]),
        )
        train_stats = (
            mean_t.numpy().astype(np.float32, copy=False),
            std_t.numpy().astype(np.float32, copy=False),
        )
        print(f"loaded precomputed Willett global split stats: {loaded_stats_path}")
    else:
        train_stats = compute_willett_normalization_stats(
            problem["train_rows"],
            cache_root=Path(problem["cache_root"]),
            mode=str(config.normalization_mode),
            feature_mode=str(problem["feature_mode"]),
        )
        if str(config.normalization_mode) == "global":
            print(
                "computed Willett global split stats from current train rows: "
                f"split_policy={problem['split_policy']} train_split={problem['train_split_name']}"
            )
    val_stats = train_stats
    missing_val_examples = normalization_stats_missing_rows(val_stats, problem["val_rows"])
    if missing_val_examples:
        preview = ", ".join(missing_val_examples[:5])
        raise ValueError(
            "Train-derived normalization stats do not cover the validation rows for "
            f"normalization_mode={config.normalization_mode!r}. "
            "Use train-split global stats for the Stanford/POSSM-style setup, or choose "
            "a normalization scheme whose keys are shared between train and validation. "
            f"First missing examples: {preview}"
        )

    predicted_train_stats = (
        compute_predicted_tx_normalization_stats(
            problem["train_rows"],
            export_accessor=export_accessor,
            mode=str(config.normalization_mode),
        )
        if export_accessor is not None
        else None
    )
    predicted_val_stats = predicted_train_stats
    if export_accessor is None:
        val_dataset = CanonicalSequenceDataset(
            problem["val_rows"],
            cache_root=Path(problem["cache_root"]),
            signal_spec=problem["signal_spec"],
            stats=val_stats,
            boundary_key_mode=str(problem["boundary_key_mode"]),
            dataset=str(problem["dataset"]),
        )
    else:
        val_dataset = ConcatenatedPredictedTxSequenceDataset(
            problem["val_rows"],
            cache_root=Path(problem["cache_root"]),
            raw_stats=val_stats,
            predicted_stats=predicted_val_stats,
            export_accessor=export_accessor,
            boundary_key_mode=str(problem["boundary_key_mode"]),
            dataset=str(problem["dataset"]),
        )
    val_sampler_rows = problem["val_rows"]
    if export_accessor is not None:
        val_sampler_rows = tuple(
            replace(
                row,
                n_time_bins=min(
                    int(getattr(row, "n_time_bins", 0) or 0),
                    int(export_accessor.duplicated_predicted_tx_for_row(row).shape[0]),
                ),
            )
            for row in problem["val_rows"]
        )
    val_loader = DataLoader(
        val_dataset,
        batch_sampler=make_length_aware_batch_sampler(
            val_sampler_rows,
            batch_size=int(config.batch_size),
            shuffle=False,
            seed=int(config.seed) + 1,
        ),
        **loader_kwargs(device),
    )
    train_adapter_keys = adapter_keys_from_rows(
        problem["train_rows"],
        dataset=str(problem["dataset"]),
        boundary_key_mode=str(problem["boundary_key_mode"]),
    )
    val_adapter_keys = adapter_keys_from_rows(
        problem["val_rows"],
        dataset=str(problem["dataset"]),
        boundary_key_mode=str(problem["boundary_key_mode"]),
    )
    train_rows_by_adapter_key = group_rows_by_adapter_key(
        problem["train_rows"],
        dataset=str(problem["dataset"]),
        boundary_key_mode=str(problem["boundary_key_mode"]),
    )
    session_adapter_keys = tuple(dict.fromkeys(train_adapter_keys + val_adapter_keys))
    train_loaders_by_adapter_key = {}
    for adapter_idx, (adapter_key, adapter_rows) in enumerate(train_rows_by_adapter_key.items(), start=1):
        sampler_rows = adapter_rows
        if export_accessor is not None:
            sampler_rows = tuple(
                replace(
                    row,
                    n_time_bins=min(
                        int(getattr(row, "n_time_bins", 0) or 0),
                        int(export_accessor.duplicated_predicted_tx_for_row(row).shape[0]),
                    ),
                )
                for row in adapter_rows
            )
        dataset_obj = (
            CanonicalSequenceDataset(
                adapter_rows,
                cache_root=Path(problem["cache_root"]),
                signal_spec=problem["signal_spec"],
                stats=train_stats,
                boundary_key_mode=str(problem["boundary_key_mode"]),
                dataset=str(problem["dataset"]),
            )
            if export_accessor is None
            else ConcatenatedPredictedTxSequenceDataset(
                adapter_rows,
                cache_root=Path(problem["cache_root"]),
                raw_stats=train_stats,
                predicted_stats=predicted_train_stats,
                export_accessor=export_accessor,
                boundary_key_mode=str(problem["boundary_key_mode"]),
                dataset=str(problem["dataset"]),
            )
        )
        train_loaders_by_adapter_key[adapter_key] = DataLoader(
            dataset_obj,
            batch_sampler=make_length_aware_batch_sampler(
                sampler_rows,
                batch_size=int(config.batch_size),
                shuffle=True,
                seed=int(config.seed) + adapter_idx,
            ),
            **loader_kwargs(device),
        )
    model = WillettPhonemeModel(
        input_dim=sample_dim,
        vocab_size=int(problem["vocab"]["num_classes"]),
        patch_size=int(config.patch_size),
        patch_stride=int(config.patch_stride),
        input_projection_size=int(config.input_projection_size),
        input_projection_dropout=float(config.input_projection_dropout),
        decoder_backbone_type=str(config.decoder_backbone_type),
        gru_hidden_size=int(config.gru_hidden_size),
        gru_num_layers=int(config.gru_num_layers),
        gru_dropout=float(config.gru_dropout),
        s5_hidden_size=int(config.s5_hidden_size),
        s5_state_size=int(config.s5_state_size),
        s5_num_layers=int(config.s5_num_layers),
        s5_dropout=float(config.s5_dropout),
        s5_direction=str(config.s5_direction),
        s5_ffn_multiplier=float(config.s5_ffn_multiplier),
        s4d_hidden_size=int(config.s4d_hidden_size),
        s4d_state_size=int(config.s4d_state_size),
        s4d_num_layers=int(config.s4d_num_layers),
        s4d_dropout=float(config.s4d_dropout),
        s4d_direction=str(config.s4d_direction),
        s4d_ffn_multiplier=float(config.s4d_ffn_multiplier),
        session_adapter_keys=session_adapter_keys,
        session_adapter_enabled=bool(config.session_adapter_enabled),
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config.learning_rate),
        weight_decay=float(config.weight_decay),
        eps=float(config.adam_epsilon),
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=_make_lr_lambda(
            warmup_steps=int(config.warmup_steps),
            max_steps=int(config.max_steps),
            min_learning_rate=float(config.min_learning_rate),
            learning_rate=float(config.learning_rate),
        ),
    )
    input_transform_config = _build_input_transform_config(config)

    start_time = time.time()
    step = 0
    best_metrics: dict[str, Any] | None = None
    best_payload: dict[str, Any] | None = None
    best_step = 0
    train_rng = random.Random(int(config.seed))
    train_iterators_by_adapter_key = {
        adapter_key: iter(loader)
        for adapter_key, loader in train_loaders_by_adapter_key.items()
    }

    resume_checkpoint = _resolve_resume_checkpoint(run_dir, config)
    if resume_checkpoint is not None:
        payload = torch.load(resume_checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(payload["model_state"])
        optimizer.load_state_dict(payload["optimizer_state"])
        scheduler.load_state_dict(payload["scheduler_state"])
        step = int(payload.get("step", 0))
        best_metrics = dict(payload.get("metrics", {})) if payload.get("metrics") else None
        best_step = int(payload.get("best_step", best_step))
        restored_best_payload = payload.get("best_progress_payload")
        if isinstance(restored_best_payload, dict):
            best_payload = dict(restored_best_payload)
        print(f"resumed Willett reconstruction from {resume_checkpoint}")

    while step < int(config.max_steps):
        accumulated_examples = 0
        accumulated_loss_sum = 0.0
        accumulated_target_count = 0
        accumulation_microbatches = 0
        current_adapter_key = train_rng.choice(train_adapter_keys)
        optimizer.zero_grad(set_to_none=True)
        model.train()

        while accumulated_examples < int(config.batch_size):
            try:
                batch = next(train_iterators_by_adapter_key[current_adapter_key])
            except StopIteration:
                train_iterators_by_adapter_key[current_adapter_key] = iter(
                    train_loaders_by_adapter_key[current_adapter_key]
                )
                batch = next(train_iterators_by_adapter_key[current_adapter_key])
            x = batch["x"].to(device)
            input_lengths = batch["input_lengths"].to(device)
            labels = batch["labels"].to(device)
            label_lengths = batch["label_lengths"].to(device)
            x = prepare_willett_inputs(
                x,
                input_lengths,
                config=input_transform_config,
                is_training=True,
            )
            outputs = model(x, input_lengths, session_ids=batch["boundary_keys"])
            loss_sum, target_count = compute_ctc_loss_sum(
                outputs["logits"],
                outputs["token_lengths"],
                labels,
                label_lengths,
                blank_index=int(problem["vocab"]["blank_index"]),
            )
            if target_count <= 0:
                continue
            microbatch_examples = int(x.shape[0])
            loss = loss_sum / target_count
            scaled_loss = loss * (float(microbatch_examples) / float(config.batch_size))
            scaled_loss.backward()
            accumulated_examples += microbatch_examples
            accumulated_target_count += int(target_count)
            accumulated_loss_sum += float(loss_sum.item())
            accumulation_microbatches += 1

        torch.nn.utils.clip_grad_norm_(model.parameters(), float(config.max_grad_norm))
        optimizer.step()
        scheduler.step()
        step += 1
        train_ctc_bpphone = float(accumulated_loss_sum / accumulated_target_count / math.log(2.0))

        if step % int(config.progress_every_steps) == 0 or step == 1:
            train_payload = {
                "event": "willett_train_report",
                "step": int(step),
                "train_ctc_bpphone": float(train_ctc_bpphone),
                "optimizer_target_examples": int(config.batch_size),
                "accumulated_examples": int(accumulated_examples),
                "accumulation_microbatches": int(accumulation_microbatches),
                "train_boundary_key": str(current_adapter_key),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "elapsed_seconds": float(time.time() - start_time),
            }
            _emit_progress(progress_log_path, **train_payload)

        if step % int(config.val_every_steps) == 0 or step == int(config.max_steps):
            metrics = evaluate_willett_phoneme_metrics(
                model=model,
                loader=val_loader,
                device=device,
                blank_index=int(problem["vocab"]["blank_index"]),
                input_transform_config=input_transform_config,
            )
            best_val_ctc = min(
                float(metrics["val_ctc_bpphone"]),
                float(best_metrics.get("best_val_ctc_bpphone", float("inf"))) if best_metrics else float(metrics["val_ctc_bpphone"]),
            )
            best_val_per = min(
                float(metrics["val_phoneme_error_rate"]),
                float(best_metrics.get("best_val_phoneme_error_rate", float("inf"))) if best_metrics else float(metrics["val_phoneme_error_rate"]),
            )
            metrics["best_val_ctc_bpphone"] = float(best_val_ctc)
            metrics["best_val_phoneme_error_rate"] = float(best_val_per)
            val_payload = {
                "event": "willett_val_report",
                "step": int(step),
                **metrics,
                "elapsed_seconds": float(time.time() - start_time),
            }
            _emit_progress(progress_log_path, **val_payload)
            if best_metrics is None or float(metrics["val_phoneme_error_rate"]) < float(best_metrics["val_phoneme_error_rate"]):
                best_metrics = dict(metrics)
                best_payload = dict(val_payload)
                best_step = int(step)
                _save_checkpoint(
                    checkpoint_path=checkpoint_best_path,
                    config=config,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    step=step,
                    metrics=metrics,
                    best_step=best_step,
                    best_progress_payload=best_payload,
                    problem=problem,
                )

        if step % int(config.checkpoint_every_steps) == 0 or step == int(config.max_steps):
            step_checkpoint_path = checkpoints_dir / f"step_{step:06d}.pt"
            _save_checkpoint(
                checkpoint_path=step_checkpoint_path,
                config=config,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                step=step,
                metrics=best_metrics,
                best_step=best_step,
                best_progress_payload=best_payload,
                problem=problem,
            )
            _prune_step_checkpoints(checkpoints_dir, config.checkpoint_keep_last)

    final_metrics = evaluate_willett_phoneme_metrics(
        model=model,
        loader=val_loader,
        device=device,
        blank_index=int(problem["vocab"]["blank_index"]),
        input_transform_config=input_transform_config,
    )
    final_metrics["best_val_ctc_bpphone"] = (
        float(best_metrics["val_ctc_bpphone"]) if best_metrics is not None else float(final_metrics["val_ctc_bpphone"])
    )
    final_metrics["best_val_phoneme_error_rate"] = (
        float(best_metrics["val_phoneme_error_rate"]) if best_metrics is not None else float(final_metrics["val_phoneme_error_rate"])
    )
    _save_checkpoint(
        checkpoint_path=checkpoint_final_path,
        config=config,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        step=step,
        metrics=final_metrics,
        best_step=best_step,
        best_progress_payload=best_payload,
        problem=problem,
    )
    summary = {
        "run_name": run_dir.name,
        "run_dir": str(run_dir),
        "cache_root": str(problem["cache_root"]),
        "dataset": str(problem["dataset"]),
        "feature_mode": str(problem["feature_mode"]),
        "boundary_key_mode": str(problem["boundary_key_mode"]),
        "split_policy": str(problem["split_policy"]),
        "cv_num_folds": problem.get("cv_num_folds"),
        "cv_fold_index": problem.get("cv_fold_index"),
        "normalization_mode": str(config.normalization_mode),
        "input_feature_source": str(config.input_feature_source),
        "predicted_export_root": None if config.predicted_export_root is None else str(config.predicted_export_root),
        "train_examples": int(len(problem["train_rows"])),
        "val_examples": int(len(problem["val_rows"])),
        "train_session_ids": list(problem["train_session_ids"]),
        "val_session_ids": list(problem["val_session_ids"]),
        "train_adapter_keys": list(train_adapter_keys),
        "val_adapter_keys": list(val_adapter_keys),
        "model_adapter_keys": list(session_adapter_keys),
        "train_sampling_mode": "uniform_single_boundary_key_per_step",
        "train_split_name": str(problem["train_split_name"]),
        "val_split_name": str(problem["val_split_name"]),
        "steps": int(step),
        "best_step": int(best_step),
        "config": asdict(config),
        "metrics": final_metrics,
        "best_metrics": best_metrics,
        "best_progress_payload": best_payload,
        "progress_log_path": str(progress_log_path),
        "summary_path": str(summary_path),
        "checkpoint_best_path": str(checkpoint_best_path) if checkpoint_best_path.exists() else None,
        "checkpoint_final_path": str(checkpoint_final_path),
        "precomputed_split_stats_path": str(loaded_stats_path) if loaded_stats_path is not None else None,
        "precomputed_split_stats_metadata": stats_metadata,
        "trainable_parameters": int(_count_trainable_parameters(model)),
        "device": str(device),
    }
    summary_json = json.loads(json.dumps(summary, default=str))
    summary_path.write_text(json.dumps(summary_json, indent=2))
    return summary_json


def _parse_args() -> WillettReconstructionConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=str, required=True)
    parser.add_argument("--output-root", type=str, required=True)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--dataset", type=str, default="brain2text24")
    parser.add_argument("--feature-mode", choices=("tx_only", "tx_sbp"), default="tx_only")
    parser.add_argument("--boundary-key-mode", choices=("session", "subject_if_available"), default="session")
    parser.add_argument(
        "--split-policy",
        choices=("competition_train_test", "competition_train_kfold", "source_train_val"),
        default="competition_train_test",
    )
    parser.add_argument("--cv-num-folds", type=int, default=5)
    parser.add_argument("--cv-fold-index", type=int, default=0)
    parser.add_argument("--normalization-mode", choices=("block", "global", "per_session", "none"), default="global")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-steps", type=int, default=120000)
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--min-learning-rate", type=float, default=1e-4)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--adam-epsilon", type=float, default=1e-1)
    parser.add_argument("--max-grad-norm", type=float, default=10.0)
    parser.add_argument("--val-every-steps", type=int, default=100)
    parser.add_argument("--checkpoint-every-steps", type=int, default=500)
    parser.add_argument("--checkpoint-keep-last", type=int, default=2)
    parser.add_argument("--progress-every-steps", type=int, default=25)
    parser.add_argument("--input-projection-size", type=int, default=256)
    parser.add_argument("--input-projection-dropout", type=float, default=0.2)
    parser.add_argument("--decoder-backbone-type", choices=("gru", "s5", "s4d"), default="gru")
    parser.add_argument("--gru-hidden-size", type=int, default=512)
    parser.add_argument("--gru-num-layers", type=int, default=5)
    parser.add_argument("--gru-dropout", type=float, default=0.4)
    parser.add_argument("--s5-hidden-size", type=int, default=512)
    parser.add_argument("--s5-state-size", type=int, default=128)
    parser.add_argument("--s5-num-layers", type=int, default=5)
    parser.add_argument("--s5-dropout", type=float, default=0.2)
    parser.add_argument("--s5-direction", choices=("causal", "bidirectional"), default="causal")
    parser.add_argument("--s5-ffn-multiplier", type=float, default=2.0)
    parser.add_argument("--s4d-hidden-size", type=int, default=512)
    parser.add_argument("--s4d-state-size", type=int, default=128)
    parser.add_argument("--s4d-num-layers", type=int, default=5)
    parser.add_argument("--s4d-dropout", type=float, default=0.2)
    parser.add_argument("--s4d-direction", choices=("causal", "bidirectional"), default="causal")
    parser.add_argument("--s4d-ffn-multiplier", type=float, default=2.0)
    parser.add_argument("--patch-size", type=int, default=14)
    parser.add_argument("--patch-stride", type=int, default=4)
    parser.add_argument(
        "--input-feature-source",
        choices=("raw", "raw_plus_predicted_tx"),
        default="raw",
    )
    parser.add_argument("--predicted-export-root", type=str, default=None)
    parser.add_argument("--input-smoothing-sigma-bins", type=float, default=2.0)
    parser.add_argument("--input-smoothing-kernel-size", type=int, default=100)
    parser.add_argument("--input-smoothing-threshold", type=float, default=0.01)
    parser.add_argument("--white-noise-sd", type=float, default=1.0)
    parser.add_argument("--constant-offset-sd", type=float, default=0.2)
    parser.add_argument("--precomputed-split-stats-path", type=str, default=None)
    parser.add_argument("--resume-checkpoint-path", type=str, default=None)
    parser.add_argument("--resume-latest", action="store_true")
    parser.add_argument("--disable-session-adapter", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    return WillettReconstructionConfig(
        seed=int(args.seed),
        dataset=str(args.dataset),
        feature_mode=str(args.feature_mode),
        boundary_key_mode=str(args.boundary_key_mode),
        split_policy=str(args.split_policy),
        cv_num_folds=int(args.cv_num_folds),
        cv_fold_index=int(args.cv_fold_index),
        normalization_mode=str(args.normalization_mode),
        batch_size=int(args.batch_size),
        max_steps=int(args.max_steps),
        learning_rate=float(args.learning_rate),
        min_learning_rate=float(args.min_learning_rate),
        warmup_steps=int(args.warmup_steps),
        weight_decay=float(args.weight_decay),
        adam_epsilon=float(args.adam_epsilon),
        max_grad_norm=float(args.max_grad_norm),
        val_every_steps=int(args.val_every_steps),
        checkpoint_every_steps=int(args.checkpoint_every_steps),
        checkpoint_keep_last=int(args.checkpoint_keep_last),
        progress_every_steps=int(args.progress_every_steps),
        input_projection_size=int(args.input_projection_size),
        input_projection_dropout=float(args.input_projection_dropout),
        decoder_backbone_type=str(args.decoder_backbone_type),
        gru_hidden_size=int(args.gru_hidden_size),
        gru_num_layers=int(args.gru_num_layers),
        gru_dropout=float(args.gru_dropout),
        s5_hidden_size=int(args.s5_hidden_size),
        s5_state_size=int(args.s5_state_size),
        s5_num_layers=int(args.s5_num_layers),
        s5_dropout=float(args.s5_dropout),
        s5_direction=str(args.s5_direction),
        s5_ffn_multiplier=float(args.s5_ffn_multiplier),
        s4d_hidden_size=int(args.s4d_hidden_size),
        s4d_state_size=int(args.s4d_state_size),
        s4d_num_layers=int(args.s4d_num_layers),
        s4d_dropout=float(args.s4d_dropout),
        s4d_direction=str(args.s4d_direction),
        s4d_ffn_multiplier=float(args.s4d_ffn_multiplier),
        patch_size=int(args.patch_size),
        patch_stride=int(args.patch_stride),
        session_adapter_enabled=not bool(args.disable_session_adapter),
        input_feature_source=str(args.input_feature_source),
        predicted_export_root=args.predicted_export_root,
        input_smoothing_sigma_bins=float(args.input_smoothing_sigma_bins),
        input_smoothing_kernel_size=int(args.input_smoothing_kernel_size),
        input_smoothing_threshold=float(args.input_smoothing_threshold),
        white_noise_sd=float(args.white_noise_sd),
        constant_offset_sd=float(args.constant_offset_sd),
        precomputed_split_stats_path=args.precomputed_split_stats_path,
        output_root=args.output_root,
        run_name=args.run_name,
        cache_root=args.cache_root,
        resume_checkpoint_path=args.resume_checkpoint_path,
        resume_latest=bool(args.resume_latest),
    )


def main() -> int:
    config = _parse_args()
    summary = run_willett_reconstruction(config)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

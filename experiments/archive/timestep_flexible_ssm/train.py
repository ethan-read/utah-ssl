"""Train a timestep-flexible supervised S5 decoder on Brain2Text24."""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

try:
    from utah_ssl.stats import (
        load_precomputed_split_feature_stats,
        resolve_precomputed_split_stats_path,
    )
    from utah_ssl.ctc import compute_ctc_loss_sum
except ModuleNotFoundError:  # pragma: no cover - repo-root unittest fallback
    from utah_ssl.stats import (
        load_precomputed_split_feature_stats,
        resolve_precomputed_split_stats_path,
    )
    from utah_ssl.ctc import compute_ctc_loss_sum

from .data import (
    CANONICAL_BIN_SIZE_MS,
    RebinnedSequenceDataset,
    TimestepFlexibleInputTransformConfig,
    adapter_keys_from_rows,
    build_timestep_flexible_problem,
    compute_rebinned_normalization_stats,
    group_rows_by_adapter_key,
    loader_kwargs,
    make_length_aware_batch_sampler,
    normalization_stats_missing_rows,
    prepare_timestep_flexible_inputs,
)
from .model import TimestepFlexibleS5Model
from .reporting import evaluate_timestep_flexible_phoneme_metrics


@dataclass
class TimestepFlexibleSSMConfig:
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
    input_projection_size: int = 256
    input_projection_dropout: float = 0.2
    s5_hidden_size: int = 512
    s5_state_size: int = 128
    s5_num_layers: int = 5
    s5_dropout: float = 0.2
    s5_direction: str = "causal"
    s5_ffn_multiplier: float = 2.0
    patch_size_ms: int = 280
    patch_stride_ms: int = 80
    train_bin_size_ms: int = 20
    eval_bin_sizes_ms: tuple[int, ...] = (20, 40)
    session_adapter_enabled: bool = True
    input_smoothing_sigma_ms: float = 40.0
    input_smoothing_kernel_size_ms: float = 2000.0
    input_smoothing_threshold: float = 0.01
    white_noise_sd: float = 1.0
    constant_offset_sd: float = 0.2
    precomputed_split_stats_path: str | Path | None = None
    output_root: str | Path = "experiments/archive/timestep_flexible_ssm_runs"
    run_name: str | None = None
    cache_root: str | Path = "/Users/home/thesis/data/cache_v1"
    resume_checkpoint_path: str | Path | None = None
    resume_latest: bool = False

    def __post_init__(self) -> None:
        if self.feature_mode not in {"tx_only", "tx_sbp"}:
            raise ValueError("feature_mode must be one of {'tx_only', 'tx_sbp'}")
        if self.boundary_key_mode not in {"session", "subject_if_available"}:
            raise ValueError("boundary_key_mode must be one of {'session', 'subject_if_available'}")
        if self.split_policy not in {"competition_train_test", "competition_train_kfold"}:
            raise ValueError("split_policy must be one of {'competition_train_test', 'competition_train_kfold'}")
        if self.normalization_mode not in {"block", "global", "per_session", "none"}:
            raise ValueError("normalization_mode must be one of {'block', 'global', 'per_session', 'none'}")
        if int(self.batch_size) <= 0 or int(self.max_steps) <= 0:
            raise ValueError("batch_size and max_steps must be positive")
        if float(self.learning_rate) <= 0.0 or float(self.min_learning_rate) < 0.0:
            raise ValueError("learning rates must be non-negative and max lr must be positive")
        if self.s5_direction not in {"causal", "bidirectional"}:
            raise ValueError("s5_direction must be one of {'causal', 'bidirectional'}")
        if int(self.train_bin_size_ms) <= 0 or int(self.train_bin_size_ms) % CANONICAL_BIN_SIZE_MS != 0:
            raise ValueError("train_bin_size_ms must be a positive multiple of 20")
        if int(self.patch_size_ms) <= 0 or int(self.patch_stride_ms) <= 0:
            raise ValueError("patch_size_ms and patch_stride_ms must be positive")
        if float(self.input_smoothing_sigma_ms) < 0.0 or float(self.input_smoothing_kernel_size_ms) <= 0.0:
            raise ValueError("Smoothing sigma must be non-negative and smoothing kernel size must be positive")
        eval_bins = tuple(int(item) for item in self.eval_bin_sizes_ms)
        if not eval_bins:
            raise ValueError("eval_bin_sizes_ms must be non-empty")
        if int(self.train_bin_size_ms) not in eval_bins:
            eval_bins = (int(self.train_bin_size_ms), *eval_bins)
        for bin_size in eval_bins:
            if bin_size <= 0 or bin_size % CANONICAL_BIN_SIZE_MS != 0:
                raise ValueError("All eval_bin_sizes_ms values must be positive multiples of 20")
            if int(self.patch_size_ms) % int(bin_size) != 0:
                raise ValueError("patch_size_ms must be divisible by every eval bin size")
            if int(self.patch_stride_ms) % int(bin_size) != 0:
                raise ValueError("patch_stride_ms must be divisible by every eval bin size")
        self.eval_bin_sizes_ms = tuple(dict.fromkeys(eval_bins))


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


def _resolve_run_dir(config: TimestepFlexibleSSMConfig) -> Path:
    output_root = Path(config.output_root)
    resolved_run_name = (
        str(config.run_name)
        if config.run_name is not None
        else f"timestep_flexible_s5_{config.feature_mode}_{_timestamp_utc()}"
    )
    return output_root / resolved_run_name


def _resolve_resume_checkpoint(run_dir: Path, config: TimestepFlexibleSSMConfig) -> Path | None:
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
    config: TimestepFlexibleSSMConfig,
    model: TimestepFlexibleS5Model,
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
        "model_family": "timestep_flexible_ssm",
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


def _count_trainable_parameters(module: torch.nn.Module) -> int:
    return int(sum(param.numel() for param in module.parameters() if param.requires_grad))


def _build_input_transform_config(config: TimestepFlexibleSSMConfig) -> TimestepFlexibleInputTransformConfig:
    return TimestepFlexibleInputTransformConfig(
        input_smoothing_sigma_ms=float(config.input_smoothing_sigma_ms),
        input_smoothing_kernel_size_ms=float(config.input_smoothing_kernel_size_ms),
        input_smoothing_threshold=float(config.input_smoothing_threshold),
        white_noise_sd=float(config.white_noise_sd),
        constant_offset_sd=float(config.constant_offset_sd),
    )


def _flatten_view_metrics(metrics_by_view: dict[int, dict[str, Any]]) -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    for bin_size_ms, metrics in sorted(metrics_by_view.items()):
        flattened[f"val_{int(bin_size_ms)}ms_ctc_bpphone"] = float(metrics["val_ctc_bpphone"])
        flattened[f"val_{int(bin_size_ms)}ms_phoneme_error_rate"] = float(metrics["val_phoneme_error_rate"])
    return flattened


def _selection_view_metrics(
    metrics_by_view: dict[int, dict[str, Any]],
    *,
    train_bin_size_ms: int,
) -> dict[str, Any]:
    selected = metrics_by_view.get(int(train_bin_size_ms))
    if selected is not None:
        return selected
    return metrics_by_view[sorted(metrics_by_view)[0]]


def _load_or_compute_stats_for_view(
    *,
    config: TimestepFlexibleSSMConfig,
    problem: dict[str, Any],
    rows: tuple[Any, ...] | list[Any],
    bin_size_ms: int,
    sample_dim: int,
) -> tuple[
    dict[str, tuple[np.ndarray, np.ndarray]] | tuple[np.ndarray, np.ndarray] | None,
    dict[str, Any] | None,
    Path | None,
]:
    if str(config.normalization_mode) != "global":
        return (
            compute_rebinned_normalization_stats(
                rows,
                cache_root=Path(problem["cache_root"]),
                feature_mode=str(problem["feature_mode"]),
                mode=str(config.normalization_mode),
                bin_size_ms=int(bin_size_ms),
            ),
            None,
            None,
        )
    if (
        int(bin_size_ms) == CANONICAL_BIN_SIZE_MS
        and (
            str(config.split_policy) == "competition_train_test"
            or config.precomputed_split_stats_path is not None
        )
    ):
        resolved_stats_path = resolve_precomputed_split_stats_path(
            cache_root=Path(config.cache_root),
            dataset=str(config.dataset),
            train_split_name=str(problem["train_split_name"]),
            signal_spec=problem["signal_spec"],
            preferred_path=config.precomputed_split_stats_path,
        )
        if not resolved_stats_path.exists() and config.precomputed_split_stats_path is None:
            return (
                compute_rebinned_normalization_stats(
                    rows,
                    cache_root=Path(problem["cache_root"]),
                    feature_mode=str(problem["feature_mode"]),
                    mode="global",
                    bin_size_ms=int(bin_size_ms),
                ),
                None,
                None,
            )
        (mean_t, std_t), stats_metadata, loaded_stats_path = load_precomputed_split_feature_stats(
            stats_path=resolved_stats_path,
            cache_root=Path(problem["cache_root"]),
            dataset=str(problem["dataset"]),
            signal_spec=problem["signal_spec"],
            boundary_key_mode=str(problem["boundary_key_mode"]),
            train_split_name=str(problem["train_split_name"]),
            val_split_name=str(problem["val_split_name"]),
            split_policy=str(problem["split_policy"]),
        )
        return (
            (
                mean_t.numpy().astype(np.float32, copy=False),
                std_t.numpy().astype(np.float32, copy=False),
            ),
            stats_metadata,
            loaded_stats_path,
        )
    return (
        compute_rebinned_normalization_stats(
            rows,
            cache_root=Path(problem["cache_root"]),
            feature_mode=str(problem["feature_mode"]),
            mode="global",
            bin_size_ms=int(bin_size_ms),
        ),
        None,
        None,
    )


def run_timestep_flexible_reconstruction(config: TimestepFlexibleSSMConfig) -> dict[str, Any]:
    _seed_all(int(config.seed))
    device = _detect_device()
    problem = build_timestep_flexible_problem(
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

    sample_dim = int(
        problem["train_rows"][0].n_tx_features
        if config.feature_mode == "tx_only"
        else problem["train_rows"][0].n_tx_features + problem["train_rows"][0].n_sbp_features
    )
    input_transform_config = _build_input_transform_config(config)

    train_stats, train_stats_metadata, train_stats_path = _load_or_compute_stats_for_view(
        config=config,
        problem=problem,
        rows=problem["train_rows"],
        bin_size_ms=int(config.train_bin_size_ms),
        sample_dim=sample_dim,
    )
    val_stats_by_view: dict[int, Any] = {int(config.train_bin_size_ms): train_stats}
    val_stats_metadata_by_view: dict[int, Any] = {
        int(config.train_bin_size_ms): train_stats_metadata,
    }
    val_stats_path_by_view: dict[int, str | None] = {
        int(config.train_bin_size_ms): None if train_stats_path is None else str(train_stats_path),
    }
    for bin_size_ms in config.eval_bin_sizes_ms:
        resolved_bin = int(bin_size_ms)
        if resolved_bin == int(config.train_bin_size_ms):
            continue
        stats, metadata, stats_path = _load_or_compute_stats_for_view(
            config=config,
            problem=problem,
            rows=problem["train_rows"],
            bin_size_ms=resolved_bin,
            sample_dim=sample_dim,
        )
        val_stats_by_view[resolved_bin] = stats
        val_stats_metadata_by_view[resolved_bin] = metadata
        val_stats_path_by_view[resolved_bin] = None if stats_path is None else str(stats_path)

    for bin_size_ms in config.eval_bin_sizes_ms:
        missing = normalization_stats_missing_rows(val_stats_by_view[int(bin_size_ms)], problem["val_rows"])
        if missing:
            preview = ", ".join(missing[:5])
            raise ValueError(
                f"Normalization stats for view {int(bin_size_ms)} ms do not cover validation rows. "
                f"First missing examples: {preview}"
            )

    val_loaders_by_view: dict[int, DataLoader] = {}
    for view_idx, bin_size_ms in enumerate(config.eval_bin_sizes_ms, start=1):
        dataset = RebinnedSequenceDataset(
            problem["val_rows"],
            cache_root=Path(problem["cache_root"]),
            stats=val_stats_by_view[int(bin_size_ms)],
            feature_mode=str(problem["feature_mode"]),
            boundary_key_mode=str(problem["boundary_key_mode"]),
            dataset=str(problem["dataset"]),
            active_bin_size_ms=int(bin_size_ms),
        )
        val_loaders_by_view[int(bin_size_ms)] = DataLoader(
            dataset,
            batch_sampler=make_length_aware_batch_sampler(
                problem["val_rows"],
                batch_size=int(config.batch_size),
                shuffle=False,
                seed=int(config.seed) + view_idx,
                bin_size_ms=int(bin_size_ms),
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
    train_loaders_by_adapter_key = {
        adapter_key: DataLoader(
            RebinnedSequenceDataset(
                adapter_rows,
                cache_root=Path(problem["cache_root"]),
                stats=train_stats,
                feature_mode=str(problem["feature_mode"]),
                boundary_key_mode=str(problem["boundary_key_mode"]),
                dataset=str(problem["dataset"]),
                active_bin_size_ms=int(config.train_bin_size_ms),
            ),
            batch_sampler=make_length_aware_batch_sampler(
                adapter_rows,
                batch_size=int(config.batch_size),
                shuffle=True,
                seed=int(config.seed) + adapter_idx,
                bin_size_ms=int(config.train_bin_size_ms),
            ),
            **loader_kwargs(device),
        )
        for adapter_idx, (adapter_key, adapter_rows) in enumerate(train_rows_by_adapter_key.items(), start=1)
    }
    model = TimestepFlexibleS5Model(
        input_dim=sample_dim,
        vocab_size=int(problem["vocab"]["num_classes"]),
        train_bin_size_ms=int(config.train_bin_size_ms),
        patch_size_ms=int(config.patch_size_ms),
        patch_stride_ms=int(config.patch_stride_ms),
        input_projection_size=int(config.input_projection_size),
        input_projection_dropout=float(config.input_projection_dropout),
        s5_hidden_size=int(config.s5_hidden_size),
        s5_state_size=int(config.s5_state_size),
        s5_num_layers=int(config.s5_num_layers),
        s5_dropout=float(config.s5_dropout),
        s5_direction=str(config.s5_direction),
        s5_ffn_multiplier=float(config.s5_ffn_multiplier),
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
        print(f"resumed timestep-flexible reconstruction from {resume_checkpoint}")

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
            x = prepare_timestep_flexible_inputs(
                x,
                input_lengths,
                config=input_transform_config,
                active_bin_size_ms=int(config.train_bin_size_ms),
                is_training=True,
            )
            outputs = model(
                x,
                input_lengths,
                active_bin_size_ms=int(config.train_bin_size_ms),
                session_ids=batch["boundary_keys"],
            )
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
                "event": "timestep_flexible_train_report",
                "step": int(step),
                "train_ctc_bpphone": float(train_ctc_bpphone),
                "train_bin_size_ms": int(config.train_bin_size_ms),
                "optimizer_target_examples": int(config.batch_size),
                "accumulated_examples": int(accumulated_examples),
                "accumulation_microbatches": int(accumulation_microbatches),
                "train_boundary_key": str(current_adapter_key),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "elapsed_seconds": float(time.time() - start_time),
            }
            _emit_progress(progress_log_path, **train_payload)

        if step % int(config.val_every_steps) == 0 or step == int(config.max_steps):
            metrics_by_view: dict[int, dict[str, Any]] = {}
            for bin_size_ms in config.eval_bin_sizes_ms:
                metrics_by_view[int(bin_size_ms)] = evaluate_timestep_flexible_phoneme_metrics(
                    model=model,
                    loader=val_loaders_by_view[int(bin_size_ms)],
                    device=device,
                    blank_index=int(problem["vocab"]["blank_index"]),
                    active_bin_size_ms=int(bin_size_ms),
                    input_transform_config=input_transform_config,
                )
            selection_metrics = _selection_view_metrics(
                metrics_by_view,
                train_bin_size_ms=int(config.train_bin_size_ms),
            )
            best_val_per = min(
                float(selection_metrics["val_phoneme_error_rate"]),
                float(best_metrics.get("best_val_phoneme_error_rate", float("inf")))
                if best_metrics is not None
                else float(selection_metrics["val_phoneme_error_rate"]),
            )
            flattened = _flatten_view_metrics(metrics_by_view)
            metrics_payload = {
                "selection_bin_size_ms": int(config.train_bin_size_ms),
                "best_val_phoneme_error_rate": float(best_val_per),
                "metrics_by_bin_ms": {str(key): value for key, value in metrics_by_view.items()},
                **flattened,
            }
            val_payload = {
                "event": "timestep_flexible_val_report",
                "step": int(step),
                **metrics_payload,
                "elapsed_seconds": float(time.time() - start_time),
            }
            _emit_progress(progress_log_path, **val_payload)
            if best_metrics is None or float(selection_metrics["val_phoneme_error_rate"]) < float(
                best_metrics["best_val_phoneme_error_rate"]
            ):
                best_metrics = dict(metrics_payload)
                best_payload = dict(val_payload)
                best_step = int(step)
                _save_checkpoint(
                    checkpoint_path=checkpoint_best_path,
                    config=config,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    step=step,
                    metrics=metrics_payload,
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

    final_metrics_by_view: dict[int, dict[str, Any]] = {}
    for bin_size_ms in config.eval_bin_sizes_ms:
        final_metrics_by_view[int(bin_size_ms)] = evaluate_timestep_flexible_phoneme_metrics(
            model=model,
            loader=val_loaders_by_view[int(bin_size_ms)],
            device=device,
            blank_index=int(problem["vocab"]["blank_index"]),
            active_bin_size_ms=int(bin_size_ms),
            input_transform_config=input_transform_config,
        )
    final_selection_metrics = _selection_view_metrics(
        final_metrics_by_view,
        train_bin_size_ms=int(config.train_bin_size_ms),
    )
    final_metrics = {
        "selection_bin_size_ms": int(config.train_bin_size_ms),
        "best_val_phoneme_error_rate": (
            float(best_metrics["best_val_phoneme_error_rate"])
            if best_metrics is not None
            else float(final_selection_metrics["val_phoneme_error_rate"])
        ),
        "metrics_by_bin_ms": {str(key): value for key, value in final_metrics_by_view.items()},
        **_flatten_view_metrics(final_metrics_by_view),
    }
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
        "train_bin_size_ms": int(config.train_bin_size_ms),
        "eval_bin_sizes_ms": list(config.eval_bin_sizes_ms),
        "patch_size_ms": int(config.patch_size_ms),
        "patch_stride_ms": int(config.patch_stride_ms),
        "config": asdict(config),
        "metrics": final_metrics,
        "best_metrics": best_metrics,
        "best_progress_payload": best_payload,
        "progress_log_path": str(progress_log_path),
        "summary_path": str(summary_path),
        "checkpoint_best_path": str(checkpoint_best_path) if checkpoint_best_path.exists() else None,
        "checkpoint_final_path": str(checkpoint_final_path),
        "train_precomputed_split_stats_path": None if train_stats_path is None else str(train_stats_path),
        "train_precomputed_split_stats_metadata": train_stats_metadata,
        "val_stats_metadata_by_bin_ms": {str(key): value for key, value in val_stats_metadata_by_view.items()},
        "val_stats_path_by_bin_ms": {str(key): value for key, value in val_stats_path_by_view.items()},
        "trainable_parameters": int(_count_trainable_parameters(model)),
        "device": str(device),
    }
    summary_json = json.loads(json.dumps(summary, default=str))
    summary_path.write_text(json.dumps(summary_json, indent=2))
    return summary_json


def _parse_eval_bin_sizes_ms(value: str) -> tuple[int, ...]:
    items = [item.strip() for item in str(value).split(",") if item.strip()]
    if not items:
        raise ValueError("eval_bin_sizes_ms must contain at least one integer")
    return tuple(int(item) for item in items)


def _parse_args() -> TimestepFlexibleSSMConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=str, required=True)
    parser.add_argument("--output-root", type=str, required=True)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--dataset", type=str, default="brain2text24")
    parser.add_argument("--feature-mode", choices=("tx_only", "tx_sbp"), default="tx_only")
    parser.add_argument("--boundary-key-mode", choices=("session", "subject_if_available"), default="session")
    parser.add_argument(
        "--split-policy",
        choices=("competition_train_test", "competition_train_kfold"),
        default="competition_train_test",
    )
    parser.add_argument("--cv-num-folds", type=int, default=5)
    parser.add_argument("--cv-fold-index", type=int, default=0)
    parser.add_argument("--normalization-mode", choices=("block", "global", "per_session", "none"), default="global")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-steps", type=int, default=120000)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--min-learning-rate", type=float, default=1e-5)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--adam-epsilon", type=float, default=1e-8)
    parser.add_argument("--max-grad-norm", type=float, default=10.0)
    parser.add_argument("--val-every-steps", type=int, default=100)
    parser.add_argument("--checkpoint-every-steps", type=int, default=500)
    parser.add_argument("--checkpoint-keep-last", type=int, default=2)
    parser.add_argument("--progress-every-steps", type=int, default=25)
    parser.add_argument("--input-projection-size", type=int, default=256)
    parser.add_argument("--input-projection-dropout", type=float, default=0.2)
    parser.add_argument("--s5-hidden-size", type=int, default=512)
    parser.add_argument("--s5-state-size", type=int, default=128)
    parser.add_argument("--s5-num-layers", type=int, default=5)
    parser.add_argument("--s5-dropout", type=float, default=0.2)
    parser.add_argument("--s5-direction", choices=("causal", "bidirectional"), default="causal")
    parser.add_argument("--s5-ffn-multiplier", type=float, default=2.0)
    parser.add_argument("--patch-size-ms", type=int, default=280)
    parser.add_argument("--patch-stride-ms", type=int, default=80)
    parser.add_argument("--train-bin-size-ms", type=int, default=20)
    parser.add_argument("--eval-bin-sizes-ms", type=str, default="20,40")
    parser.add_argument("--input-smoothing-sigma-ms", type=float, default=40.0)
    parser.add_argument("--input-smoothing-kernel-size-ms", type=float, default=2000.0)
    parser.add_argument("--input-smoothing-threshold", type=float, default=0.01)
    parser.add_argument("--white-noise-sd", type=float, default=1.0)
    parser.add_argument("--constant-offset-sd", type=float, default=0.2)
    parser.add_argument("--precomputed-split-stats-path", type=str, default=None)
    parser.add_argument("--resume-checkpoint-path", type=str, default=None)
    parser.add_argument("--resume-latest", action="store_true")
    parser.add_argument("--disable-session-adapter", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    return TimestepFlexibleSSMConfig(
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
        s5_hidden_size=int(args.s5_hidden_size),
        s5_state_size=int(args.s5_state_size),
        s5_num_layers=int(args.s5_num_layers),
        s5_dropout=float(args.s5_dropout),
        s5_direction=str(args.s5_direction),
        s5_ffn_multiplier=float(args.s5_ffn_multiplier),
        patch_size_ms=int(args.patch_size_ms),
        patch_stride_ms=int(args.patch_stride_ms),
        train_bin_size_ms=int(args.train_bin_size_ms),
        eval_bin_sizes_ms=_parse_eval_bin_sizes_ms(args.eval_bin_sizes_ms),
        session_adapter_enabled=not bool(args.disable_session_adapter),
        input_smoothing_sigma_ms=float(args.input_smoothing_sigma_ms),
        input_smoothing_kernel_size_ms=float(args.input_smoothing_kernel_size_ms),
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
    summary = run_timestep_flexible_reconstruction(config)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

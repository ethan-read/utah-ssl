"""Supervised mixed-bin and missing-bin experiment runners."""

from __future__ import annotations

import json
import math
import random
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from analysis.active.ssl_experiments.ssl_core.ctc import compute_ctc_loss_sum
from analysis.active.ssl_experiments.willett_reconstruction.data import (
    WillettInputTransformConfig,
    adapter_keys_from_rows,
    group_rows_by_adapter_key,
    make_length_aware_batch_sampler as willett_make_batch_sampler,
)
from analysis.active.ssl_experiments.willett_reconstruction.model import WillettPhonemeModel
from analysis.active.ssl_experiments.willett_reconstruction.reporting import (
    evaluate_willett_phoneme_metrics,
)

from .data import (
    CANONICAL_BIN_SIZE_MS,
    TimestepFlexibleInputTransformConfig,
    build_timestep_flexible_problem,
    loader_kwargs,
    make_length_aware_batch_sampler,
    normalization_stats_missing_rows,
    prepare_timestep_flexible_inputs,
    rebinned_input_length,
)
from .experiment_models import IrregularTimestepS5Model
from .experiment_utils import (
    MixedBinSequenceDataset,
    MissingBinSequenceDataset,
    build_train_val_stats,
    collate_sequence_extras,
    sample_feature_dim,
)
from .model import TimestepFlexibleS5Model
from .reporting import evaluate_timestep_flexible_phoneme_metrics
from .train import _build_input_transform_config, _detect_device, _make_lr_lambda, _seed_all


@dataclass
class SupervisedExperimentConfig:
    seed: int = 7
    dataset: str = "brain2text24"
    feature_mode: str = "tx_only"
    boundary_key_mode: str = "session"
    split_policy: str = "competition_train_test"
    cv_num_folds: int = 5
    cv_fold_index: int = 0
    normalization_mode: str = "global"
    batch_size: int = 64
    max_steps: int = 12000
    learning_rate: float = 1e-3
    min_learning_rate: float = 1e-5
    warmup_steps: int = 1000
    weight_decay: float = 1e-5
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 10.0
    val_every_steps: int = 100
    checkpoint_every_steps: int = 500
    progress_every_steps: int = 25
    patch_size_ms: int = 280
    patch_stride_ms: int = 80
    input_projection_size: int = 256
    input_projection_dropout: float = 0.2
    s5_hidden_size: int = 512
    s5_state_size: int = 128
    s5_num_layers: int = 5
    s5_dropout: float = 0.2
    s5_direction: str = "causal"
    s5_ffn_multiplier: float = 2.0
    gru_hidden_size: int = 512
    gru_num_layers: int = 5
    gru_dropout: float = 0.4
    session_adapter_enabled: bool = True
    input_smoothing_sigma_ms: float = 40.0
    input_smoothing_kernel_size_ms: float = 2000.0
    input_smoothing_threshold: float = 0.01
    white_noise_sd: float = 1.0
    constant_offset_sd: float = 0.2
    mixed_bin_sizes_ms: tuple[int, int] = (20, 40)
    missing_drop_probability: float = 0.25
    train_mask_seed: int = 101
    val_mask_seed: int = 202
    precomputed_split_stats_path: str | Path | None = None
    output_root: str | Path = "analysis/active/ssl_experiments/timestep_flexible_ssm_runs"
    run_name: str = "supervised_experiment"
    cache_root: str | Path = "/Users/home/thesis/data/cache_v1"


def _emit_progress(progress_log_path: Path, **payload: Any) -> None:
    progress_log_path.parent.mkdir(parents=True, exist_ok=True)
    with progress_log_path.open("a") as handle:
        handle.write(json.dumps(payload) + "\n")


def _save_model_checkpoint(
    *,
    path: Path,
    config: SupervisedExperimentConfig,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    step: int,
    metrics: dict[str, Any],
    problem: dict[str, Any],
) -> None:
    payload = {
        "config": json.loads(json.dumps(asdict(config), default=str)),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "step": int(step),
        "metrics": json.loads(json.dumps(metrics, default=str)),
        "problem": {
            "dataset": str(problem["dataset"]),
            "feature_mode": str(problem["feature_mode"]),
            "boundary_key_mode": str(problem["boundary_key_mode"]),
            "train_split_name": str(problem["train_split_name"]),
            "val_split_name": str(problem["val_split_name"]),
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def _summarize_and_write(
    *,
    run_dir: Path,
    config: SupervisedExperimentConfig,
    problem: dict[str, Any],
    metrics: dict[str, Any],
    best_metrics: dict[str, Any],
    trainable_parameters: int,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    summary_path = run_dir / "summary.json"
    summary = {
        "run_name": str(config.run_name),
        "run_dir": str(run_dir),
        "summary_path": str(summary_path),
        "config": json.loads(json.dumps(asdict(config), default=str)),
        "dataset": str(problem["dataset"]),
        "feature_mode": str(problem["feature_mode"]),
        "train_split_name": str(problem["train_split_name"]),
        "val_split_name": str(problem["val_split_name"]),
        "train_examples": int(len(problem["train_rows"])),
        "val_examples": int(len(problem["val_rows"])),
        "metrics": metrics,
        "best_metrics": best_metrics,
        "trainable_parameters": int(trainable_parameters),
    }
    if extra:
        summary.update(json.loads(json.dumps(extra, default=str)))
    summary_path.write_text(json.dumps(summary, indent=2))
    return summary


def _build_problem_and_stats(config: SupervisedExperimentConfig, *, eval_bin_sizes_ms: tuple[int, ...]) -> tuple[dict[str, Any], int, Any, dict[int, Any]]:
    problem = build_timestep_flexible_problem(
        cache_root=Path(config.cache_root),
        dataset=str(config.dataset),
        feature_mode=str(config.feature_mode),
        boundary_key_mode=str(config.boundary_key_mode),
        split_policy=str(config.split_policy),
        cv_num_folds=int(config.cv_num_folds),
        cv_fold_index=int(config.cv_fold_index),
    )
    sample_dim = sample_feature_dim(problem, str(config.feature_mode))
    train_stats, val_stats_by_view, _, _ = build_train_val_stats(
        config=config,
        problem=problem,
        sample_dim=int(sample_dim),
        eval_bin_sizes_ms=tuple(int(item) for item in eval_bin_sizes_ms),
    )
    for bin_size_ms, stats in val_stats_by_view.items():
        missing = normalization_stats_missing_rows(stats, problem["val_rows"])
        if missing:
            raise ValueError(f"Missing normalization coverage for {bin_size_ms} ms validation rows: {missing[:5]}")
    return problem, int(sample_dim), train_stats, val_stats_by_view


def run_mixed_bin_s5(config: SupervisedExperimentConfig) -> dict[str, Any]:
    _seed_all(int(config.seed))
    device = _detect_device()
    problem, sample_dim, _, val_stats_by_view = _build_problem_and_stats(
        config,
        eval_bin_sizes_ms=tuple(dict.fromkeys(int(item) for item in config.mixed_bin_sizes_ms)),
    )
    run_dir = Path(config.output_root) / str(config.run_name)
    progress_path = run_dir / "progress.jsonl"
    checkpoint_best_path = run_dir / "checkpoint_best.pt"
    checkpoint_final_path = run_dir / "checkpoint_final.pt"
    input_transform_config = _build_input_transform_config(config)

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
    rows_by_adapter = group_rows_by_adapter_key(
        problem["train_rows"],
        dataset=str(problem["dataset"]),
        boundary_key_mode=str(problem["boundary_key_mode"]),
    )
    session_adapter_keys = tuple(dict.fromkeys(train_adapter_keys + val_adapter_keys))

    train_loaders: dict[tuple[str, int], DataLoader] = {}
    train_iterators: dict[tuple[str, int], Any] = {}
    for adapter_idx, (adapter_key, adapter_rows) in enumerate(rows_by_adapter.items(), start=1):
        for bin_size_ms in tuple(int(item) for item in config.mixed_bin_sizes_ms):
            dataset = MixedBinSequenceDataset(
                adapter_rows,
                cache_root=Path(problem["cache_root"]),
                stats=val_stats_by_view[int(bin_size_ms)],
                feature_mode=str(problem["feature_mode"]),
                boundary_key_mode=str(problem["boundary_key_mode"]),
                dataset=str(problem["dataset"]),
                active_bin_size_ms=int(bin_size_ms),
            )
            loader = DataLoader(
                dataset,
                batch_sampler=make_length_aware_batch_sampler(
                    adapter_rows,
                    batch_size=int(config.batch_size),
                    shuffle=True,
                    seed=int(config.seed) + adapter_idx + int(bin_size_ms),
                    bin_size_ms=int(bin_size_ms),
                ),
                collate_fn=collate_sequence_extras,
                pin_memory=device.type == "cuda",
                num_workers=0,
            )
            train_loaders[(str(adapter_key), int(bin_size_ms))] = loader
            train_iterators[(str(adapter_key), int(bin_size_ms))] = iter(loader)

    val_loaders = {
        int(bin_size_ms): DataLoader(
            MixedBinSequenceDataset(
                problem["val_rows"],
                cache_root=Path(problem["cache_root"]),
                stats=val_stats_by_view[int(bin_size_ms)],
                feature_mode=str(problem["feature_mode"]),
                boundary_key_mode=str(problem["boundary_key_mode"]),
                dataset=str(problem["dataset"]),
                active_bin_size_ms=int(bin_size_ms),
            ),
            batch_sampler=make_length_aware_batch_sampler(
                problem["val_rows"],
                batch_size=int(config.batch_size),
                shuffle=False,
                seed=int(config.seed) + 300 + int(bin_size_ms),
                bin_size_ms=int(bin_size_ms),
            ),
            collate_fn=collate_sequence_extras,
            pin_memory=device.type == "cuda",
            num_workers=0,
        )
        for bin_size_ms in tuple(int(item) for item in config.mixed_bin_sizes_ms)
    }

    model = TimestepFlexibleS5Model(
        input_dim=int(sample_dim),
        vocab_size=int(problem["vocab"]["num_classes"]),
        train_bin_size_ms=20,
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

    best_metrics: dict[str, Any] | None = None
    best_step = 0
    rng = random.Random(int(config.seed))
    start_time = time.time()

    mixed_bin_schedule = tuple(int(item) for item in config.mixed_bin_sizes_ms)
    for step in range(1, int(config.max_steps) + 1):
        optimizer.zero_grad(set_to_none=True)
        model.train()
        current_adapter_key = str(rng.choice(train_adapter_keys))
        current_bin_size_ms = int(mixed_bin_schedule[(step - 1) % len(mixed_bin_schedule)])
        loader_key = (current_adapter_key, current_bin_size_ms)
        try:
            batch = next(train_iterators[loader_key])
        except StopIteration:
            train_iterators[loader_key] = iter(train_loaders[loader_key])
            batch = next(train_iterators[loader_key])
        x = batch["x"].to(device)
        input_lengths = batch["input_lengths"].to(device)
        labels = batch["labels"].to(device)
        label_lengths = batch["label_lengths"].to(device)
        x = prepare_timestep_flexible_inputs(
            x,
            input_lengths,
            config=input_transform_config,
            active_bin_size_ms=int(current_bin_size_ms),
            is_training=True,
        )
        outputs = model(
            x,
            input_lengths,
            active_bin_size_ms=int(current_bin_size_ms),
            session_ids=batch["boundary_keys"],
        )
        loss_sum, target_count = compute_ctc_loss_sum(
            outputs["logits"],
            outputs["token_lengths"],
            labels,
            label_lengths,
            blank_index=int(problem["vocab"]["blank_index"]),
        )
        loss = loss_sum / max(int(target_count), 1)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(config.max_grad_norm))
        optimizer.step()
        scheduler.step()

        if step % int(config.progress_every_steps) == 0 or step == 1:
            _emit_progress(
                progress_path,
                event="mixed_s5_train",
                step=int(step),
                train_bin_size_ms=int(current_bin_size_ms),
                train_ctc_bpphone=float(loss_sum.item() / max(int(target_count), 1) / math.log(2.0)),
                learning_rate=float(optimizer.param_groups[0]["lr"]),
                elapsed_seconds=float(time.time() - start_time),
            )
        if step % int(config.val_every_steps) == 0 or step == int(config.max_steps):
            metrics_by_view = {
                int(bin_size_ms): evaluate_timestep_flexible_phoneme_metrics(
                    model=model,
                    loader=val_loaders[int(bin_size_ms)],
                    device=device,
                    blank_index=int(problem["vocab"]["blank_index"]),
                    active_bin_size_ms=int(bin_size_ms),
                    input_transform_config=input_transform_config,
                )
                for bin_size_ms in tuple(int(item) for item in config.mixed_bin_sizes_ms)
            }
            payload = {
                "metrics_by_bin_ms": {str(key): value for key, value in metrics_by_view.items()},
                **{
                    f"val_{int(key)}ms_ctc_bpphone": float(value["val_ctc_bpphone"])
                    for key, value in metrics_by_view.items()
                },
                **{
                    f"val_{int(key)}ms_phoneme_error_rate": float(value["val_phoneme_error_rate"])
                    for key, value in metrics_by_view.items()
                },
            }
            _emit_progress(progress_path, event="mixed_s5_val", step=int(step), **payload)
            if best_metrics is None or float(metrics_by_view[20]["val_phoneme_error_rate"]) < float(
                best_metrics["metrics_by_bin_ms"]["20"]["val_phoneme_error_rate"]
            ):
                best_metrics = payload
                best_step = int(step)
                _save_model_checkpoint(
                    path=checkpoint_best_path,
                    config=config,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    step=step,
                    metrics=payload,
                    problem=problem,
                )
    final_metrics = {
        int(bin_size_ms): evaluate_timestep_flexible_phoneme_metrics(
            model=model,
            loader=val_loaders[int(bin_size_ms)],
            device=device,
            blank_index=int(problem["vocab"]["blank_index"]),
            active_bin_size_ms=int(bin_size_ms),
            input_transform_config=input_transform_config,
        )
        for bin_size_ms in tuple(int(item) for item in config.mixed_bin_sizes_ms)
    }
    final_payload = {
        "metrics_by_bin_ms": {str(key): value for key, value in final_metrics.items()},
        **{f"val_{int(key)}ms_ctc_bpphone": float(value["val_ctc_bpphone"]) for key, value in final_metrics.items()},
        **{
            f"val_{int(key)}ms_phoneme_error_rate": float(value["val_phoneme_error_rate"])
            for key, value in final_metrics.items()
        },
    }
    _save_model_checkpoint(
        path=checkpoint_final_path,
        config=config,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        step=int(config.max_steps),
        metrics=final_payload,
        problem=problem,
    )
    return _summarize_and_write(
        run_dir=run_dir,
        config=config,
        problem=problem,
        metrics=final_payload,
        best_metrics=best_metrics or final_payload,
        trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        extra={
            "experiment_family": "mixed_bin_supervised",
            "model_family": "s5",
            "best_step": int(best_step),
            "mixed_bin_sizes_ms": list(int(item) for item in config.mixed_bin_sizes_ms),
            "progress_log_path": str(progress_path),
            "checkpoint_best_path": str(checkpoint_best_path),
            "checkpoint_final_path": str(checkpoint_final_path),
        },
    )


def _build_gru_model(config: SupervisedExperimentConfig, *, sample_dim: int, session_adapter_keys: tuple[str, ...], vocab_size: int) -> WillettPhonemeModel:
    return WillettPhonemeModel(
        input_dim=int(sample_dim),
        vocab_size=int(vocab_size),
        patch_size=int(config.patch_size_ms // CANONICAL_BIN_SIZE_MS),
        patch_stride=int(config.patch_stride_ms // CANONICAL_BIN_SIZE_MS),
        input_projection_size=int(config.input_projection_size),
        input_projection_dropout=float(config.input_projection_dropout),
        decoder_backbone_type="gru",
        gru_hidden_size=int(config.gru_hidden_size),
        gru_num_layers=int(config.gru_num_layers),
        gru_dropout=float(config.gru_dropout),
        session_adapter_keys=session_adapter_keys,
        session_adapter_enabled=bool(config.session_adapter_enabled),
    )


def _build_willett_input_transform_config(config: SupervisedExperimentConfig) -> WillettInputTransformConfig:
    return WillettInputTransformConfig(
        input_smoothing_sigma_bins=float(config.input_smoothing_sigma_ms) / float(CANONICAL_BIN_SIZE_MS),
        input_smoothing_kernel_size=max(1, int(round(float(config.input_smoothing_kernel_size_ms) / float(CANONICAL_BIN_SIZE_MS)))),
        input_smoothing_threshold=float(config.input_smoothing_threshold),
        white_noise_sd=float(config.white_noise_sd),
        constant_offset_sd=float(config.constant_offset_sd),
    )


def run_mixed_bin_gru(config: SupervisedExperimentConfig) -> dict[str, Any]:
    _seed_all(int(config.seed))
    device = _detect_device()
    problem, sample_dim, _, val_stats_by_view = _build_problem_and_stats(
        config,
        eval_bin_sizes_ms=tuple(dict.fromkeys(int(item) for item in config.mixed_bin_sizes_ms)),
    )
    run_dir = Path(config.output_root) / str(config.run_name)
    progress_path = run_dir / "progress.jsonl"
    checkpoint_best_path = run_dir / "checkpoint_best.pt"
    checkpoint_final_path = run_dir / "checkpoint_final.pt"
    input_transform_config = _build_willett_input_transform_config(config)
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
    rows_by_adapter = group_rows_by_adapter_key(
        problem["train_rows"],
        dataset=str(problem["dataset"]),
        boundary_key_mode=str(problem["boundary_key_mode"]),
    )
    session_adapter_keys = tuple(dict.fromkeys(train_adapter_keys + val_adapter_keys))
    train_loaders: dict[tuple[str, int], DataLoader] = {}
    train_iterators: dict[tuple[str, int], Any] = {}
    for adapter_idx, (adapter_key, adapter_rows) in enumerate(rows_by_adapter.items(), start=1):
        for bin_size_ms in tuple(int(item) for item in config.mixed_bin_sizes_ms):
            dataset = MixedBinSequenceDataset(
                adapter_rows,
                cache_root=Path(problem["cache_root"]),
                stats=val_stats_by_view[int(bin_size_ms)],
                feature_mode=str(problem["feature_mode"]),
                boundary_key_mode=str(problem["boundary_key_mode"]),
                dataset=str(problem["dataset"]),
                active_bin_size_ms=int(bin_size_ms),
                duplicate_to_canonical=int(bin_size_ms) > CANONICAL_BIN_SIZE_MS,
            )
            loader = DataLoader(
                dataset,
                batch_sampler=willett_make_batch_sampler(
                    [
                        replace(
                            row,
                            n_time_bins=max(
                                1,
                                rebinned_input_length(int(getattr(row, "n_time_bins", 0) or 0), bin_size_ms=int(bin_size_ms))
                                * max(1, int(bin_size_ms) // CANONICAL_BIN_SIZE_MS),
                            ),
                        )
                        for row in adapter_rows
                    ],
                    batch_size=int(config.batch_size),
                    shuffle=True,
                    seed=int(config.seed) + adapter_idx + int(bin_size_ms),
                ),
                collate_fn=collate_sequence_extras,
                pin_memory=device.type == "cuda",
                num_workers=0,
            )
            train_loaders[(str(adapter_key), int(bin_size_ms))] = loader
            train_iterators[(str(adapter_key), int(bin_size_ms))] = iter(loader)
    val_loaders = {
        int(bin_size_ms): DataLoader(
            MixedBinSequenceDataset(
                problem["val_rows"],
                cache_root=Path(problem["cache_root"]),
                stats=val_stats_by_view[int(bin_size_ms)],
                feature_mode=str(problem["feature_mode"]),
                boundary_key_mode=str(problem["boundary_key_mode"]),
                dataset=str(problem["dataset"]),
                active_bin_size_ms=int(bin_size_ms),
                duplicate_to_canonical=int(bin_size_ms) > CANONICAL_BIN_SIZE_MS,
            ),
            batch_sampler=willett_make_batch_sampler(
                [
                    replace(
                        row,
                        n_time_bins=max(
                            1,
                            rebinned_input_length(int(getattr(row, "n_time_bins", 0) or 0), bin_size_ms=int(bin_size_ms))
                            * max(1, int(bin_size_ms) // CANONICAL_BIN_SIZE_MS),
                        ),
                    )
                    for row in problem["val_rows"]
                ],
                batch_size=int(config.batch_size),
                shuffle=False,
                seed=int(config.seed) + 401 + int(bin_size_ms),
            ),
            collate_fn=collate_sequence_extras,
            pin_memory=device.type == "cuda",
            num_workers=0,
        )
        for bin_size_ms in tuple(int(item) for item in config.mixed_bin_sizes_ms)
    }
    model = _build_gru_model(
        config,
        sample_dim=int(sample_dim),
        session_adapter_keys=session_adapter_keys,
        vocab_size=int(problem["vocab"]["num_classes"]),
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
    best_metrics: dict[str, Any] | None = None
    best_step = 0
    rng = random.Random(int(config.seed))
    start_time = time.time()
    mixed_bin_schedule = tuple(int(item) for item in config.mixed_bin_sizes_ms)
    for step in range(1, int(config.max_steps) + 1):
        optimizer.zero_grad(set_to_none=True)
        model.train()
        current_adapter_key = str(rng.choice(train_adapter_keys))
        current_bin_size_ms = int(mixed_bin_schedule[(step - 1) % len(mixed_bin_schedule)])
        loader_key = (current_adapter_key, current_bin_size_ms)
        try:
            batch = next(train_iterators[loader_key])
        except StopIteration:
            train_iterators[loader_key] = iter(train_loaders[loader_key])
            batch = next(train_iterators[loader_key])
        x = batch["x"].to(device)
        input_lengths = batch["input_lengths"].to(device)
        labels = batch["labels"].to(device)
        label_lengths = batch["label_lengths"].to(device)
        x = prepare_timestep_flexible_inputs(
            x,
            input_lengths,
            config=TimestepFlexibleInputTransformConfig(
                input_smoothing_sigma_ms=float(config.input_smoothing_sigma_ms),
                input_smoothing_kernel_size_ms=float(config.input_smoothing_kernel_size_ms),
                input_smoothing_threshold=float(config.input_smoothing_threshold),
                white_noise_sd=float(config.white_noise_sd),
                constant_offset_sd=float(config.constant_offset_sd),
            ),
            active_bin_size_ms=20,
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
        loss = loss_sum / max(int(target_count), 1)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(config.max_grad_norm))
        optimizer.step()
        scheduler.step()
        if step % int(config.progress_every_steps) == 0 or step == 1:
            _emit_progress(
                progress_path,
                event="mixed_gru_train",
                step=int(step),
                train_bin_size_ms=int(current_bin_size_ms),
                train_ctc_bpphone=float(loss_sum.item() / max(int(target_count), 1) / math.log(2.0)),
                learning_rate=float(optimizer.param_groups[0]["lr"]),
                elapsed_seconds=float(time.time() - start_time),
            )
        if step % int(config.val_every_steps) == 0 or step == int(config.max_steps):
            metrics_by_view = {
                int(bin_size_ms): evaluate_willett_phoneme_metrics(
                    model=model,
                    loader=val_loaders[int(bin_size_ms)],
                    device=device,
                    blank_index=int(problem["vocab"]["blank_index"]),
                    input_transform_config=input_transform_config,
                )
                for bin_size_ms in tuple(int(item) for item in config.mixed_bin_sizes_ms)
            }
            payload = {
                "metrics_by_bin_ms": {str(key): value for key, value in metrics_by_view.items()},
                **{
                    f"val_{int(key)}ms_ctc_bpphone": float(value["val_ctc_bpphone"])
                    for key, value in metrics_by_view.items()
                },
                **{
                    f"val_{int(key)}ms_phoneme_error_rate": float(value["val_phoneme_error_rate"])
                    for key, value in metrics_by_view.items()
                },
            }
            _emit_progress(progress_path, event="mixed_gru_val", step=int(step), **payload)
            if best_metrics is None or float(metrics_by_view[20]["val_phoneme_error_rate"]) < float(
                best_metrics["metrics_by_bin_ms"]["20"]["val_phoneme_error_rate"]
            ):
                best_metrics = payload
                best_step = int(step)
                _save_model_checkpoint(
                    path=checkpoint_best_path,
                    config=config,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    step=step,
                    metrics=payload,
                    problem=problem,
                )
    final_metrics = {
        int(bin_size_ms): evaluate_willett_phoneme_metrics(
            model=model,
            loader=val_loaders[int(bin_size_ms)],
            device=device,
            blank_index=int(problem["vocab"]["blank_index"]),
            input_transform_config=input_transform_config,
        )
        for bin_size_ms in tuple(int(item) for item in config.mixed_bin_sizes_ms)
    }
    final_payload = {
        "metrics_by_bin_ms": {str(key): value for key, value in final_metrics.items()},
        **{f"val_{int(key)}ms_ctc_bpphone": float(value["val_ctc_bpphone"]) for key, value in final_metrics.items()},
        **{
            f"val_{int(key)}ms_phoneme_error_rate": float(value["val_phoneme_error_rate"])
            for key, value in final_metrics.items()
        },
    }
    _save_model_checkpoint(
        path=checkpoint_final_path,
        config=config,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        step=int(config.max_steps),
        metrics=final_payload,
        problem=problem,
    )
    return _summarize_and_write(
        run_dir=run_dir,
        config=config,
        problem=problem,
        metrics=final_payload,
        best_metrics=best_metrics or final_payload,
        trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        extra={
            "experiment_family": "mixed_bin_supervised",
            "model_family": "gru",
            "best_step": int(best_step),
            "mixed_bin_sizes_ms": list(int(item) for item in config.mixed_bin_sizes_ms),
            "progress_log_path": str(progress_path),
            "checkpoint_best_path": str(checkpoint_best_path),
            "checkpoint_final_path": str(checkpoint_final_path),
        },
    )


def _evaluate_irregular_s5(
    *,
    model: IrregularTimestepS5Model,
    loader: DataLoader,
    device: torch.device,
    blank_index: int,
    input_transform_config: TimestepFlexibleInputTransformConfig,
) -> dict[str, Any]:
    model.eval()
    total_loss_sum = 0.0
    total_targets = 0
    total_ref = 0
    total_errors = 0
    from analysis.active.ssl_experiments.ssl_core.ctc import ctc_greedy_decode, edit_counts
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            input_lengths = batch["input_lengths"].to(device)
            labels = batch["labels"].to(device)
            label_lengths = batch["label_lengths"].to(device)
            x = prepare_timestep_flexible_inputs(
                x,
                input_lengths,
                config=input_transform_config,
                active_bin_size_ms=20,
                is_training=False,
            )
            outputs = model(
                x,
                input_lengths,
                active_bin_size_ms=20,
                session_ids=batch["boundary_keys"],
                time_deltas_ms=batch["time_deltas_ms"].to(device),
            )
            loss_sum, target_count = compute_ctc_loss_sum(
                outputs["logits"],
                outputs["token_lengths"],
                labels,
                label_lengths,
                blank_index=int(blank_index),
            )
            total_loss_sum += float(loss_sum.item())
            total_targets += int(target_count)
            predictions = ctc_greedy_decode(outputs["logits"], outputs["token_lengths"], blank_index=int(blank_index))
            for row_idx, prediction in enumerate(predictions):
                reference = labels[row_idx, : int(label_lengths[row_idx].item())].tolist()
                i, d, s = edit_counts(reference, prediction)
                total_errors += int(i + d + s)
                total_ref += len(reference)
    return {
        "val_ctc_bpphone": float(total_loss_sum / max(total_targets, 1) / math.log(2.0)),
        "val_phoneme_error_rate": float(total_errors / max(total_ref, 1)),
    }


def run_missing_bin_s5(config: SupervisedExperimentConfig) -> dict[str, Any]:
    _seed_all(int(config.seed))
    device = _detect_device()
    problem, sample_dim, train_stats, val_stats_by_view = _build_problem_and_stats(config, eval_bin_sizes_ms=(20,))
    run_dir = Path(config.output_root) / str(config.run_name)
    progress_path = run_dir / "progress.jsonl"
    checkpoint_best_path = run_dir / "checkpoint_best.pt"
    checkpoint_final_path = run_dir / "checkpoint_final.pt"
    input_transform_config = _build_input_transform_config(config)
    train_adapter_keys = adapter_keys_from_rows(problem["train_rows"], dataset=str(problem["dataset"]), boundary_key_mode=str(problem["boundary_key_mode"]))
    val_adapter_keys = adapter_keys_from_rows(problem["val_rows"], dataset=str(problem["dataset"]), boundary_key_mode=str(problem["boundary_key_mode"]))
    rows_by_adapter = group_rows_by_adapter_key(problem["train_rows"], dataset=str(problem["dataset"]), boundary_key_mode=str(problem["boundary_key_mode"]))
    session_adapter_keys = tuple(dict.fromkeys(train_adapter_keys + val_adapter_keys))
    train_loaders = {}
    train_iterators = {}
    for adapter_idx, (adapter_key, adapter_rows) in enumerate(rows_by_adapter.items(), start=1):
        loader = DataLoader(
            MissingBinSequenceDataset(
                adapter_rows,
                cache_root=Path(problem["cache_root"]),
                stats=train_stats,
                feature_mode=str(problem["feature_mode"]),
                boundary_key_mode=str(problem["boundary_key_mode"]),
                dataset=str(problem["dataset"]),
                drop_probability=float(config.missing_drop_probability),
                seed=int(config.train_mask_seed),
                mode="s5",
            ),
            batch_sampler=make_length_aware_batch_sampler(
                [
                    replace(row, n_time_bins=max(1, int(math.ceil(int(getattr(row, "n_time_bins", 0) or 0) * (1.0 - float(config.missing_drop_probability))))))
                    for row in adapter_rows
                ],
                batch_size=int(config.batch_size),
                shuffle=True,
                seed=int(config.seed) + adapter_idx,
                bin_size_ms=20,
            ),
            collate_fn=collate_sequence_extras,
            pin_memory=device.type == "cuda",
            num_workers=0,
        )
        train_loaders[str(adapter_key)] = loader
        train_iterators[str(adapter_key)] = iter(loader)
    val_loader = DataLoader(
        MissingBinSequenceDataset(
            problem["val_rows"],
            cache_root=Path(problem["cache_root"]),
            stats=val_stats_by_view[20],
            feature_mode=str(problem["feature_mode"]),
            boundary_key_mode=str(problem["boundary_key_mode"]),
            dataset=str(problem["dataset"]),
            drop_probability=float(config.missing_drop_probability),
            seed=int(config.val_mask_seed),
            mode="s5",
        ),
        batch_sampler=make_length_aware_batch_sampler(
            [
                replace(row, n_time_bins=max(1, int(math.ceil(int(getattr(row, "n_time_bins", 0) or 0) * (1.0 - float(config.missing_drop_probability))))))
                for row in problem["val_rows"]
            ],
            batch_size=int(config.batch_size),
            shuffle=False,
            seed=int(config.seed) + 401,
            bin_size_ms=20,
        ),
        collate_fn=collate_sequence_extras,
        pin_memory=device.type == "cuda",
        num_workers=0,
    )
    model = IrregularTimestepS5Model(
        input_dim=int(sample_dim),
        vocab_size=int(problem["vocab"]["num_classes"]),
        train_bin_size_ms=20,
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
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config.learning_rate), weight_decay=float(config.weight_decay), eps=float(config.adam_epsilon))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=_make_lr_lambda(warmup_steps=int(config.warmup_steps), max_steps=int(config.max_steps), min_learning_rate=float(config.min_learning_rate), learning_rate=float(config.learning_rate)),
    )
    best_metrics = None
    best_step = 0
    rng = random.Random(int(config.seed))
    start_time = time.time()
    for step in range(1, int(config.max_steps) + 1):
        optimizer.zero_grad(set_to_none=True)
        model.train()
        adapter_key = str(rng.choice(train_adapter_keys))
        try:
            batch = next(train_iterators[adapter_key])
        except StopIteration:
            train_iterators[adapter_key] = iter(train_loaders[adapter_key])
            batch = next(train_iterators[adapter_key])
        x = batch["x"].to(device)
        input_lengths = batch["input_lengths"].to(device)
        labels = batch["labels"].to(device)
        label_lengths = batch["label_lengths"].to(device)
        x = prepare_timestep_flexible_inputs(x, input_lengths, config=input_transform_config, active_bin_size_ms=20, is_training=True)
        outputs = model(x, input_lengths, active_bin_size_ms=20, session_ids=batch["boundary_keys"], time_deltas_ms=batch["time_deltas_ms"].to(device))
        loss_sum, target_count = compute_ctc_loss_sum(outputs["logits"], outputs["token_lengths"], labels, label_lengths, blank_index=int(problem["vocab"]["blank_index"]))
        loss = loss_sum / max(int(target_count), 1)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(config.max_grad_norm))
        optimizer.step()
        scheduler.step()
        if step % int(config.progress_every_steps) == 0 or step == 1:
            _emit_progress(progress_path, event="missing_s5_train", step=int(step), train_ctc_bpphone=float(loss_sum.item() / max(int(target_count), 1) / math.log(2.0)), learning_rate=float(optimizer.param_groups[0]["lr"]), elapsed_seconds=float(time.time() - start_time))
        if step % int(config.val_every_steps) == 0 or step == int(config.max_steps):
            metrics = _evaluate_irregular_s5(model=model, loader=val_loader, device=device, blank_index=int(problem["vocab"]["blank_index"]), input_transform_config=input_transform_config)
            _emit_progress(progress_path, event="missing_s5_val", step=int(step), **metrics)
            if best_metrics is None or float(metrics["val_phoneme_error_rate"]) < float(best_metrics["val_phoneme_error_rate"]):
                best_metrics = dict(metrics)
                best_step = int(step)
                _save_model_checkpoint(path=checkpoint_best_path, config=config, model=model, optimizer=optimizer, scheduler=scheduler, step=step, metrics=metrics, problem=problem)
    final_metrics = _evaluate_irregular_s5(model=model, loader=val_loader, device=device, blank_index=int(problem["vocab"]["blank_index"]), input_transform_config=input_transform_config)
    _save_model_checkpoint(path=checkpoint_final_path, config=config, model=model, optimizer=optimizer, scheduler=scheduler, step=int(config.max_steps), metrics=final_metrics, problem=problem)
    return _summarize_and_write(run_dir=run_dir, config=config, problem=problem, metrics=final_metrics, best_metrics=best_metrics or final_metrics, trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad), extra={"experiment_family": "missing_bin_supervised", "model_family": "s5", "best_step": int(best_step), "missing_drop_probability": float(config.missing_drop_probability), "train_mask_seed": int(config.train_mask_seed), "val_mask_seed": int(config.val_mask_seed), "progress_log_path": str(progress_path), "checkpoint_best_path": str(checkpoint_best_path), "checkpoint_final_path": str(checkpoint_final_path)})


def run_missing_bin_gru(config: SupervisedExperimentConfig) -> dict[str, Any]:
    _seed_all(int(config.seed))
    device = _detect_device()
    problem, sample_dim, train_stats, val_stats_by_view = _build_problem_and_stats(config, eval_bin_sizes_ms=(20,))
    run_dir = Path(config.output_root) / str(config.run_name)
    progress_path = run_dir / "progress.jsonl"
    checkpoint_best_path = run_dir / "checkpoint_best.pt"
    checkpoint_final_path = run_dir / "checkpoint_final.pt"
    input_transform_config = _build_willett_input_transform_config(config)
    train_adapter_keys = adapter_keys_from_rows(problem["train_rows"], dataset=str(problem["dataset"]), boundary_key_mode=str(problem["boundary_key_mode"]))
    val_adapter_keys = adapter_keys_from_rows(problem["val_rows"], dataset=str(problem["dataset"]), boundary_key_mode=str(problem["boundary_key_mode"]))
    rows_by_adapter = group_rows_by_adapter_key(problem["train_rows"], dataset=str(problem["dataset"]), boundary_key_mode=str(problem["boundary_key_mode"]))
    session_adapter_keys = tuple(dict.fromkeys(train_adapter_keys + val_adapter_keys))
    train_loaders = {}
    train_iterators = {}
    for adapter_idx, (adapter_key, adapter_rows) in enumerate(rows_by_adapter.items(), start=1):
        loader = DataLoader(
            MissingBinSequenceDataset(
                adapter_rows,
                cache_root=Path(problem["cache_root"]),
                stats=train_stats,
                feature_mode=str(problem["feature_mode"]),
                boundary_key_mode=str(problem["boundary_key_mode"]),
                dataset=str(problem["dataset"]),
                drop_probability=float(config.missing_drop_probability),
                seed=int(config.train_mask_seed),
                mode="gru_train",
            ),
            batch_sampler=willett_make_batch_sampler(adapter_rows, batch_size=int(config.batch_size), shuffle=True, seed=int(config.seed) + adapter_idx),
            collate_fn=collate_sequence_extras,
            pin_memory=device.type == "cuda",
            num_workers=0,
        )
        train_loaders[str(adapter_key)] = loader
        train_iterators[str(adapter_key)] = iter(loader)
    val_loader = DataLoader(
        MissingBinSequenceDataset(
            problem["val_rows"],
            cache_root=Path(problem["cache_root"]),
            stats=val_stats_by_view[20],
            feature_mode=str(problem["feature_mode"]),
            boundary_key_mode=str(problem["boundary_key_mode"]),
            dataset=str(problem["dataset"]),
            drop_probability=float(config.missing_drop_probability),
            seed=int(config.val_mask_seed),
            mode="gru_eval",
        ),
        batch_sampler=willett_make_batch_sampler(problem["val_rows"], batch_size=int(config.batch_size), shuffle=False, seed=int(config.seed) + 402),
        collate_fn=collate_sequence_extras,
        pin_memory=device.type == "cuda",
        num_workers=0,
    )
    model = _build_gru_model(config, sample_dim=int(sample_dim), session_adapter_keys=session_adapter_keys, vocab_size=int(problem["vocab"]["num_classes"])).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config.learning_rate), weight_decay=float(config.weight_decay), eps=float(config.adam_epsilon))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=_make_lr_lambda(warmup_steps=int(config.warmup_steps), max_steps=int(config.max_steps), min_learning_rate=float(config.min_learning_rate), learning_rate=float(config.learning_rate)),
    )
    best_metrics = None
    best_step = 0
    rng = random.Random(int(config.seed))
    start_time = time.time()
    for step in range(1, int(config.max_steps) + 1):
        optimizer.zero_grad(set_to_none=True)
        model.train()
        adapter_key = str(rng.choice(train_adapter_keys))
        try:
            batch = next(train_iterators[adapter_key])
        except StopIteration:
            train_iterators[adapter_key] = iter(train_loaders[adapter_key])
            batch = next(train_iterators[adapter_key])
        x = batch["x"].to(device)
        input_lengths = batch["input_lengths"].to(device)
        labels = batch["labels"].to(device)
        label_lengths = batch["label_lengths"].to(device)
        x = prepare_timestep_flexible_inputs(
            x,
            input_lengths,
            config=TimestepFlexibleInputTransformConfig(
                input_smoothing_sigma_ms=float(config.input_smoothing_sigma_ms),
                input_smoothing_kernel_size_ms=float(config.input_smoothing_kernel_size_ms),
                input_smoothing_threshold=float(config.input_smoothing_threshold),
                white_noise_sd=float(config.white_noise_sd),
                constant_offset_sd=float(config.constant_offset_sd),
            ),
            active_bin_size_ms=20,
            is_training=True,
        )
        outputs = model(x, input_lengths, session_ids=batch["boundary_keys"])
        loss_sum, target_count = compute_ctc_loss_sum(outputs["logits"], outputs["token_lengths"], labels, label_lengths, blank_index=int(problem["vocab"]["blank_index"]))
        loss = loss_sum / max(int(target_count), 1)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(config.max_grad_norm))
        optimizer.step()
        scheduler.step()
        if step % int(config.progress_every_steps) == 0 or step == 1:
            _emit_progress(progress_path, event="missing_gru_train", step=int(step), train_ctc_bpphone=float(loss_sum.item() / max(int(target_count), 1) / math.log(2.0)), learning_rate=float(optimizer.param_groups[0]["lr"]), elapsed_seconds=float(time.time() - start_time))
        if step % int(config.val_every_steps) == 0 or step == int(config.max_steps):
            metrics = evaluate_willett_phoneme_metrics(model=model, loader=val_loader, device=device, blank_index=int(problem["vocab"]["blank_index"]), input_transform_config=input_transform_config)
            _emit_progress(progress_path, event="missing_gru_val", step=int(step), **metrics)
            if best_metrics is None or float(metrics["val_phoneme_error_rate"]) < float(best_metrics["val_phoneme_error_rate"]):
                best_metrics = dict(metrics)
                best_step = int(step)
                _save_model_checkpoint(path=checkpoint_best_path, config=config, model=model, optimizer=optimizer, scheduler=scheduler, step=step, metrics=metrics, problem=problem)
    final_metrics = evaluate_willett_phoneme_metrics(model=model, loader=val_loader, device=device, blank_index=int(problem["vocab"]["blank_index"]), input_transform_config=input_transform_config)
    _save_model_checkpoint(path=checkpoint_final_path, config=config, model=model, optimizer=optimizer, scheduler=scheduler, step=int(config.max_steps), metrics=final_metrics, problem=problem)
    return _summarize_and_write(run_dir=run_dir, config=config, problem=problem, metrics=final_metrics, best_metrics=best_metrics or final_metrics, trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad), extra={"experiment_family": "missing_bin_supervised", "model_family": "gru", "best_step": int(best_step), "missing_drop_probability": float(config.missing_drop_probability), "train_mask_seed": int(config.train_mask_seed), "val_mask_seed": int(config.val_mask_seed), "progress_log_path": str(progress_path), "checkpoint_best_path": str(checkpoint_best_path), "checkpoint_final_path": str(checkpoint_final_path)})


__all__ = [
    "SupervisedExperimentConfig",
    "run_missing_bin_gru",
    "run_missing_bin_s5",
    "run_mixed_bin_gru",
    "run_mixed_bin_s5",
]

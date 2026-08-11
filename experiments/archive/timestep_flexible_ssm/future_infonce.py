"""Future-bin InfoNCE training for S5 and GRU encoders."""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from experiments.supervised_baselines.data import adapter_keys_from_rows
from experiments.supervised_baselines.model import WillettPhonemeModel

from .data import (
    CANONICAL_BIN_SIZE_MS,
    TimestepFlexibleInputTransformConfig,
    build_timestep_flexible_problem,
    loader_kwargs,
    make_length_aware_batch_sampler,
    prepare_timestep_flexible_inputs,
)
from .experiment_models import FutureGRUModel, FutureS5Model
from .experiment_utils import FutureBinsDataset, build_train_val_stats, future_valid_mask, sample_feature_dim
from .model import TimestepFlexibleS5Model
from .train import _detect_device, _make_lr_lambda, _seed_all


@dataclass
class FutureInfoNCEConfig:
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
    horizons_ms: tuple[int, ...] = (20, 40, 60, 80, 100)
    projection_dim: int = 128
    temperature: float = 0.1
    precomputed_split_stats_path: str | Path | None = None
    output_root: str | Path = "experiments/archive/timestep_flexible_ssm_runs"
    run_name: str = "future_infonce"
    cache_root: str | Path = "/Users/home/thesis/data/cache_v1"
    model_family: str = "s5"


def _emit_progress(progress_log_path: Path, **payload: Any) -> None:
    progress_log_path.parent.mkdir(parents=True, exist_ok=True)
    with progress_log_path.open("a") as handle:
        handle.write(json.dumps(payload) + "\n")


def _future_loss_and_metrics(
    *,
    model: FutureS5Model | FutureGRUModel,
    x: torch.Tensor,
    input_lengths: torch.Tensor,
    session_ids: list[str] | tuple[str, ...],
    temperature: float,
    horizons_ms: tuple[int, ...],
) -> tuple[torch.Tensor, dict[str, float]]:
    encoded = model.encode(x, input_lengths, session_ids=session_ids)
    losses: list[torch.Tensor] = []
    metrics: dict[str, float] = {}
    for horizon_ms in tuple(int(item) for item in horizons_ms):
        horizon_bins = int(horizon_ms // CANONICAL_BIN_SIZE_MS)
        valid = future_valid_mask(
            token_lengths=encoded.token_lengths,
            horizon_bins=int(horizon_bins),
            patch_size_bins=int(encoded.patch_size_bins),
            patch_stride_bins=int(encoded.patch_stride_bins),
            frame_lengths=input_lengths,
        )
        if not valid:
            continue
        hidden = torch.stack([encoded.hidden[batch_idx, token_idx] for batch_idx, token_idx, _ in valid], dim=0)
        targets = torch.stack([x[batch_idx, frame_idx] for batch_idx, _, frame_idx in valid], dim=0)
        query = model.head.project_query(hidden, horizon_ms=int(horizon_ms))
        target = model.head.project_target(targets, horizon_ms=int(horizon_ms))
        logits = torch.matmul(query, target.transpose(0, 1)) / float(temperature)
        labels = torch.arange(int(logits.shape[0]), device=logits.device, dtype=torch.long)
        loss = F.cross_entropy(logits, labels)
        losses.append(loss)
        metrics[f"h{int(horizon_ms)}_infonce_loss"] = float(loss.item())
    if not losses:
        raise ValueError("No valid future-prediction pairs were constructed for the current batch.")
    total_loss = torch.stack(losses).mean()
    metrics["mean_infonce_loss"] = float(total_loss.item())
    return total_loss, metrics


def run_future_infonce(config: FutureInfoNCEConfig) -> dict[str, Any]:
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
    sample_dim = sample_feature_dim(problem, str(config.feature_mode))
    train_stats, val_stats_by_view, _, _ = build_train_val_stats(
        config=config,
        problem=problem,
        sample_dim=int(sample_dim),
        eval_bin_sizes_ms=(20,),
    )
    run_dir = Path(config.output_root) / str(config.run_name)
    run_dir.mkdir(parents=True, exist_ok=True)
    progress_path = run_dir / "progress.jsonl"
    summary_path = run_dir / "summary.json"
    input_transform_config = TimestepFlexibleInputTransformConfig(
        input_smoothing_sigma_ms=float(config.input_smoothing_sigma_ms),
        input_smoothing_kernel_size_ms=float(config.input_smoothing_kernel_size_ms),
        input_smoothing_threshold=float(config.input_smoothing_threshold),
        white_noise_sd=float(config.white_noise_sd),
        constant_offset_sd=float(config.constant_offset_sd),
    )
    train_loader = DataLoader(
        FutureBinsDataset(
            problem["train_rows"],
            cache_root=Path(problem["cache_root"]),
            stats=train_stats,
            feature_mode=str(problem["feature_mode"]),
            boundary_key_mode=str(problem["boundary_key_mode"]),
            dataset=str(problem["dataset"]),
        ),
        batch_sampler=make_length_aware_batch_sampler(
            problem["train_rows"],
            batch_size=int(config.batch_size),
            shuffle=True,
            seed=int(config.seed),
            bin_size_ms=20,
        ),
        **loader_kwargs(device),
    )
    val_loader = DataLoader(
        FutureBinsDataset(
            problem["val_rows"],
            cache_root=Path(problem["cache_root"]),
            stats=val_stats_by_view[20],
            feature_mode=str(problem["feature_mode"]),
            boundary_key_mode=str(problem["boundary_key_mode"]),
            dataset=str(problem["dataset"]),
        ),
        batch_sampler=make_length_aware_batch_sampler(
            problem["val_rows"],
            batch_size=int(config.batch_size),
            shuffle=False,
            seed=int(config.seed) + 1,
            bin_size_ms=20,
        ),
        **loader_kwargs(device),
    )
    session_adapter_keys = tuple(
        dict.fromkeys(
            adapter_keys_from_rows(problem["train_rows"], dataset=str(problem["dataset"]), boundary_key_mode=str(problem["boundary_key_mode"]))
            + adapter_keys_from_rows(problem["val_rows"], dataset=str(problem["dataset"]), boundary_key_mode=str(problem["boundary_key_mode"]))
        )
    )
    if str(config.model_family) == "s5":
        encoder = TimestepFlexibleS5Model(
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
        )
        model: FutureS5Model | FutureGRUModel = FutureS5Model(
            encoder,
            horizons_ms=tuple(int(item) for item in config.horizons_ms),
            input_dim=int(sample_dim),
            projection_dim=int(config.projection_dim),
        )
    else:
        encoder = WillettPhonemeModel(
            input_dim=int(sample_dim),
            vocab_size=int(problem["vocab"]["num_classes"]),
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
        model = FutureGRUModel(
            encoder,
            horizons_ms=tuple(int(item) for item in config.horizons_ms),
            input_dim=int(sample_dim),
            projection_dim=int(config.projection_dim),
        )
    model = model.to(device)
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
    train_iter = iter(train_loader)
    best_metrics: dict[str, Any] | None = None
    best_step = 0
    start_time = time.time()
    for step in range(1, int(config.max_steps) + 1):
        model.train()
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)
        optimizer.zero_grad(set_to_none=True)
        x = batch["x"].to(device)
        input_lengths = batch["input_lengths"].to(device)
        x = prepare_timestep_flexible_inputs(
            x,
            input_lengths,
            config=input_transform_config,
            active_bin_size_ms=20,
            is_training=True,
        )
        loss, metrics = _future_loss_and_metrics(
            model=model,
            x=x,
            input_lengths=input_lengths,
            session_ids=batch["boundary_keys"],
            temperature=float(config.temperature),
            horizons_ms=tuple(int(item) for item in config.horizons_ms),
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(config.max_grad_norm))
        optimizer.step()
        scheduler.step()
        if step % int(config.progress_every_steps) == 0 or step == 1:
            _emit_progress(progress_path, event="future_infonce_train", step=int(step), **metrics, elapsed_seconds=float(time.time() - start_time))
        if step % int(config.val_every_steps) == 0 or step == int(config.max_steps):
            model.eval()
            val_losses: dict[str, list[float]] = {}
            with torch.no_grad():
                for batch in val_loader:
                    x = batch["x"].to(device)
                    input_lengths = batch["input_lengths"].to(device)
                    x = prepare_timestep_flexible_inputs(
                        x,
                        input_lengths,
                        config=input_transform_config,
                        active_bin_size_ms=20,
                        is_training=False,
                    )
                    _, metrics = _future_loss_and_metrics(
                        model=model,
                        x=x,
                        input_lengths=input_lengths,
                        session_ids=batch["boundary_keys"],
                        temperature=float(config.temperature),
                        horizons_ms=tuple(int(item) for item in config.horizons_ms),
                    )
                    for key, value in metrics.items():
                        val_losses.setdefault(str(key), []).append(float(value))
            averaged = {key: float(sum(values) / len(values)) for key, values in val_losses.items()}
            _emit_progress(progress_path, event="future_infonce_val", step=int(step), **averaged, elapsed_seconds=float(time.time() - start_time))
            if best_metrics is None or float(averaged["mean_infonce_loss"]) < float(best_metrics["mean_infonce_loss"]):
                best_metrics = dict(averaged)
                best_step = int(step)
    final_metrics = best_metrics or {}
    summary = {
        "run_name": str(config.run_name),
        "run_dir": str(run_dir),
        "model_family": str(config.model_family),
        "experiment_family": "future_infonce",
        "dataset": str(problem["dataset"]),
        "feature_mode": str(problem["feature_mode"]),
        "train_split_name": str(problem["train_split_name"]),
        "val_split_name": str(problem["val_split_name"]),
        "config": json.loads(json.dumps(asdict(config), default=str)),
        "metrics": final_metrics,
        "best_metrics": best_metrics,
        "best_step": int(best_step),
        "progress_log_path": str(progress_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    summary["summary_path"] = str(summary_path)
    return summary


__all__ = ["FutureInfoNCEConfig", "run_future_infonce"]

"""Train a Willett-style phoneme decoder on the canonical Utah cache."""

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
    from masked_ssl.probe import CanonicalSequenceDataset, compute_ctc_loss_sum
except ModuleNotFoundError:  # pragma: no cover - repo-root unittest fallback
    from analysis.active.ssl_experiments.masked_ssl.probe import (
        CanonicalSequenceDataset,
        compute_ctc_loss_sum,
    )

from .data import (
    WillettInputTransformConfig,
    build_willett_problem,
    compute_willett_normalization_stats,
    load_precomputed_split_feature_stats,
    loader_kwargs,
    make_length_aware_batch_sampler,
    prepare_willett_inputs,
    resolve_precomputed_split_stats_path,
)
from .model import WillettPhonemeModel
from .reporting import evaluate_willett_phoneme_metrics


@dataclass
class WillettReconstructionConfig:
    seed: int = 7
    dataset: str = "brain2text24"
    feature_mode: str = "tx_only"
    boundary_key_mode: str = "session"
    normalization_mode: str = "block"
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
    gru_hidden_size: int = 512
    gru_num_layers: int = 5
    gru_dropout: float = 0.4
    patch_size: int = 14
    patch_stride: int = 4
    session_adapter_enabled: bool = True
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
        if int(self.gru_hidden_size) <= 0 or int(self.gru_num_layers) <= 0:
            raise ValueError("GRU sizes must be positive")


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
    )
    run_dir = _resolve_run_dir(config)
    run_dir.mkdir(parents=True, exist_ok=True)
    progress_log_path = run_dir / "progress.jsonl"
    summary_path = run_dir / "summary.json"
    checkpoints_dir = run_dir / "checkpoints"
    checkpoint_best_path = run_dir / "checkpoint_best.pt"
    checkpoint_final_path = run_dir / "checkpoint_final.pt"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    sample_dim = int(problem["train_rows"][0].n_tx_features if config.feature_mode == "tx_only" else problem["train_rows"][0].n_tx_features + problem["train_rows"][0].n_sbp_features)
    loaded_stats_path = None
    stats_metadata = None
    if str(config.normalization_mode) == "global":
        resolved_stats_path = resolve_precomputed_split_stats_path(
            cache_root=Path(config.cache_root),
            dataset=str(config.dataset),
            train_split_name=str(problem["train_split_name"]),
            feature_mode=str(config.feature_mode),
            preferred_path=config.precomputed_split_stats_path,
        )
        if resolved_stats_path is not None:
            train_stats, stats_metadata, loaded_stats_path = load_precomputed_split_feature_stats(
                stats_path=resolved_stats_path,
                expected_dim=sample_dim,
            )
            print(f"loaded precomputed Willett global split stats: {loaded_stats_path}")
        else:
            train_stats = compute_willett_normalization_stats(
                problem["train_rows"],
                cache_root=Path(problem["cache_root"]),
                mode="global",
                feature_mode=str(problem["feature_mode"]),
            )
        val_stats = train_stats
    else:
        train_stats = compute_willett_normalization_stats(
            problem["train_rows"],
            cache_root=Path(problem["cache_root"]),
            mode=str(config.normalization_mode),
            feature_mode=str(problem["feature_mode"]),
        )
        val_stats = compute_willett_normalization_stats(
            problem["val_rows"],
            cache_root=Path(problem["cache_root"]),
            mode=str(config.normalization_mode),
            feature_mode=str(problem["feature_mode"]),
        )

    train_dataset = CanonicalSequenceDataset(
        problem["train_rows"],
        cache_root=Path(problem["cache_root"]),
        stats=train_stats,
        feature_mode=str(problem["feature_mode"]),
        boundary_key_mode=str(problem["boundary_key_mode"]),
        dataset=str(problem["dataset"]),
    )
    val_dataset = CanonicalSequenceDataset(
        problem["val_rows"],
        cache_root=Path(problem["cache_root"]),
        stats=val_stats,
        feature_mode=str(problem["feature_mode"]),
        boundary_key_mode=str(problem["boundary_key_mode"]),
        dataset=str(problem["dataset"]),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=make_length_aware_batch_sampler(
            problem["train_rows"],
            batch_size=int(config.batch_size),
            shuffle=True,
            seed=int(config.seed),
        ),
        **loader_kwargs(device),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_sampler=make_length_aware_batch_sampler(
            problem["val_rows"],
            batch_size=int(config.batch_size),
            shuffle=False,
            seed=int(config.seed) + 1,
        ),
        **loader_kwargs(device),
    )
    session_adapter_keys = tuple(problem["train_session_ids"])
    model = WillettPhonemeModel(
        input_dim=sample_dim,
        vocab_size=int(problem["vocab"]["num_classes"]),
        patch_size=int(config.patch_size),
        patch_stride=int(config.patch_stride),
        input_projection_size=int(config.input_projection_size),
        input_projection_dropout=float(config.input_projection_dropout),
        gru_hidden_size=int(config.gru_hidden_size),
        gru_num_layers=int(config.gru_num_layers),
        gru_dropout=float(config.gru_dropout),
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
    train_iterator = iter(train_loader)

    resume_checkpoint = _resolve_resume_checkpoint(run_dir, config)
    if resume_checkpoint is not None:
        payload = torch.load(resume_checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(payload["model_state"])
        optimizer.load_state_dict(payload["optimizer_state"])
        scheduler.load_state_dict(payload["scheduler_state"])
        step = int(payload.get("step", 0))
        best_metrics = dict(payload.get("metrics", {})) if payload.get("metrics") else None
        print(f"resumed Willett reconstruction from {resume_checkpoint}")

    while step < int(config.max_steps):
        accumulated_examples = 0
        accumulated_loss_sum = 0.0
        accumulated_target_count = 0
        accumulation_microbatches = 0
        optimizer.zero_grad(set_to_none=True)
        model.train()

        while accumulated_examples < int(config.batch_size):
            try:
                batch = next(train_iterator)
            except StopIteration:
                train_iterator = iter(train_loader)
                batch = next(train_iterator)
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
        problem=problem,
    )
    summary = {
        "run_name": run_dir.name,
        "run_dir": str(run_dir),
        "cache_root": str(problem["cache_root"]),
        "dataset": str(problem["dataset"]),
        "feature_mode": str(problem["feature_mode"]),
        "boundary_key_mode": str(problem["boundary_key_mode"]),
        "normalization_mode": str(config.normalization_mode),
        "train_examples": int(len(problem["train_rows"])),
        "val_examples": int(len(problem["val_rows"])),
        "train_session_ids": list(problem["train_session_ids"]),
        "val_session_ids": list(problem["val_session_ids"]),
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
    parser.add_argument("--normalization-mode", choices=("block", "global", "per_session", "none"), default="block")
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
    parser.add_argument("--gru-hidden-size", type=int, default=512)
    parser.add_argument("--gru-num-layers", type=int, default=5)
    parser.add_argument("--gru-dropout", type=float, default=0.4)
    parser.add_argument("--patch-size", type=int, default=14)
    parser.add_argument("--patch-stride", type=int, default=4)
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
        gru_hidden_size=int(args.gru_hidden_size),
        gru_num_layers=int(args.gru_num_layers),
        gru_dropout=float(args.gru_dropout),
        patch_size=int(args.patch_size),
        patch_stride=int(args.patch_stride),
        session_adapter_enabled=not bool(args.disable_session_adapter),
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

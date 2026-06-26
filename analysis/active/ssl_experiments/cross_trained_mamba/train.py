"""Training entrypoint for cross-trained area-6v Mamba runs."""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

try:
    from ssl_core.ctc import compute_ctc_loss_sum, ctc_greedy_decode, edit_counts
    from ssl_core.reporting import ProgressPrinter, append_jsonl
    from willett_reconstruction.data import WillettInputTransformConfig, prepare_willett_inputs
except ModuleNotFoundError:  # pragma: no cover - repo-root unittest fallback
    from analysis.active.ssl_experiments.ssl_core.ctc import compute_ctc_loss_sum, ctc_greedy_decode, edit_counts
    from analysis.active.ssl_experiments.ssl_core.reporting import ProgressPrinter, append_jsonl
    from analysis.active.ssl_experiments.willett_reconstruction.data import (
        WillettInputTransformConfig,
        prepare_willett_inputs,
    )

from .config import CrossTrainedMambaConfig
from .data import (
    CrossDatasetSequenceDataset,
    build_cross_dataset_problem,
    compute_dataset_train_stats,
    cross_dataset_adapter_key,
    group_rows_by_adapter_key,
    loader_kwargs,
    make_length_aware_batch_sampler,
)
from .model import CrossTrainedMambaPhonemeModel


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
    # PyTorch CTC loss is not implemented on MPS, and this trainer is CTC-only.
    # Prefer CPU for local smoke runs unless CUDA is available.
    return torch.device("cpu")


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


def _resolve_run_dir(config: CrossTrainedMambaConfig) -> Path:
    output_root = Path(config.output_root)
    datasets_part = "".join(str(name).replace("brain2text", "b2t") for name in config.datasets)
    resolved_run_name = (
        str(config.run_name)
        if config.run_name is not None
        else (
            f"cross_mamba_{datasets_part}_{config.feature_mode}_area6v_native20ms_"
            f"h{int(config.hidden_size)}_hctc_fb_{config.adapter_mode}_seed{int(config.seed)}_{int(config.max_steps)}"
        )
    )
    return output_root / resolved_run_name


def _resolve_resume_checkpoint(run_dir: Path, config: CrossTrainedMambaConfig) -> Path | None:
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


def _build_input_transform_config(config: CrossTrainedMambaConfig) -> WillettInputTransformConfig:
    return WillettInputTransformConfig(
        input_smoothing_sigma_bins=float(config.input_smoothing_sigma_bins),
        input_smoothing_kernel_size=int(config.input_smoothing_kernel_size),
        input_smoothing_threshold=float(config.input_smoothing_threshold),
        white_noise_sd=float(config.white_noise_sd),
        constant_offset_sd=float(config.constant_offset_sd),
    )


def _save_checkpoint(
    *,
    checkpoint_path: Path,
    config: CrossTrainedMambaConfig,
    model: CrossTrainedMambaPhonemeModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    step: int,
    metrics: dict[str, Any] | None,
    best_progress_payload: dict[str, Any] | None,
    problem: dict[str, Any],
) -> None:
    payload = {
        "model_family": "cross_trained_mamba",
        "config": config.to_dict(),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "step": int(step),
        "metrics": dict(metrics or {}),
        "best_progress_payload": dict(best_progress_payload) if best_progress_payload is not None else None,
        "datasets": list(problem["datasets"]),
        "vocab": dict(problem["vocab"]),
    }
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, checkpoint_path)


def _prune_step_checkpoints(checkpoints_dir: Path, keep_last: int | None) -> None:
    if keep_last is None:
        return
    candidates = sorted(checkpoints_dir.glob("step_*.pt"))
    for stale in candidates[:-int(keep_last)]:
        stale.unlink(missing_ok=True)


def _build_train_loaders(
    *,
    problem: dict[str, Any],
    config: CrossTrainedMambaConfig,
    stats_by_dataset: dict[str, Any],
    device: torch.device,
) -> tuple[dict[str, dict[str, DataLoader]], dict[str, dict[str, Any]]]:
    train_loaders_by_dataset: dict[str, dict[str, DataLoader]] = {}
    grouped_rows_by_dataset: dict[str, dict[str, Any]] = {}
    for dataset_idx, dataset in enumerate(problem["datasets"], start=1):
        grouped_rows = group_rows_by_adapter_key(problem["rows_by_dataset"][dataset]["train"], dataset=dataset)
        grouped_rows_by_dataset[dataset] = grouped_rows
        dataset_loaders: dict[str, DataLoader] = {}
        for adapter_idx, (adapter_key, adapter_rows) in enumerate(sorted(grouped_rows.items()), start=1):
            dataset_obj = CrossDatasetSequenceDataset(
                adapter_rows,
                cache_root=Path(problem["cache_root"]),
                dataset=dataset,
                stats=stats_by_dataset[dataset],
                feature_mode=str(config.feature_mode),
                area6v_feature_dim=int(config.area6v_feature_dim),
            )
            dataset_loaders[adapter_key] = DataLoader(
                dataset_obj,
                batch_sampler=make_length_aware_batch_sampler(
                    adapter_rows,
                    batch_size=int(config.batch_size),
                    shuffle=True,
                    seed=int(config.seed) + dataset_idx * 100 + adapter_idx,
                ),
                **loader_kwargs(device),
            )
        train_loaders_by_dataset[dataset] = dataset_loaders
    return train_loaders_by_dataset, grouped_rows_by_dataset


def _build_val_loaders(
    *,
    problem: dict[str, Any],
    config: CrossTrainedMambaConfig,
    stats_by_dataset: dict[str, Any],
    device: torch.device,
) -> dict[str, DataLoader]:
    val_loaders: dict[str, DataLoader] = {}
    for dataset in problem["datasets"]:
        dataset_obj = CrossDatasetSequenceDataset(
            problem["rows_by_dataset"][dataset]["val"],
            cache_root=Path(problem["cache_root"]),
            dataset=dataset,
            stats=stats_by_dataset[dataset],
            feature_mode=str(config.feature_mode),
            area6v_feature_dim=int(config.area6v_feature_dim),
        )
        val_loaders[dataset] = DataLoader(
            dataset_obj,
            batch_sampler=make_length_aware_batch_sampler(
                problem["rows_by_dataset"][dataset]["val"],
                batch_size=int(config.batch_size),
                shuffle=False,
                seed=int(config.seed) + 1000,
            ),
            **loader_kwargs(device),
        )
    return val_loaders


def _evaluate_loader(
    *,
    model: CrossTrainedMambaPhonemeModel,
    loader: DataLoader,
    device: torch.device,
    blank_index: int,
    input_transform_config: WillettInputTransformConfig,
) -> dict[str, Any]:
    model.eval()
    totals = {
        "l1_loss_sum": 0.0,
        "l2_loss_sum": 0.0,
        "l3_loss_sum": 0.0,
        "target_count": 0,
        "reference_tokens": 0,
        "predicted_tokens": 0,
        "insertions": 0,
        "deletions": 0,
        "substitutions": 0,
    }
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            input_lengths = batch["input_lengths"].to(device)
            labels = batch["labels"].to(device)
            label_lengths = batch["label_lengths"].to(device)
            x = prepare_willett_inputs(
                x,
                input_lengths,
                config=input_transform_config,
                is_training=False,
            )
            outputs = model(x, input_lengths, session_ids=batch["boundary_keys"])
            _, loss_parts, target_count = compute_hierarchical_ctc_losses(
                outputs,
                labels,
                label_lengths,
                blank_index=blank_index,
                intermediate_ctc_weight=0.3,
            )
            totals["l1_loss_sum"] += float(loss_parts["l1_loss_sum"])
            totals["l2_loss_sum"] += float(loss_parts["l2_loss_sum"])
            totals["l3_loss_sum"] += float(loss_parts["l3_loss_sum"])
            totals["target_count"] += int(target_count)
            predictions = ctc_greedy_decode(outputs["l3"], outputs["token_lengths"], blank_index=blank_index)
            for row_idx, prediction in enumerate(predictions):
                reference_length = int(label_lengths[row_idx].item())
                reference = labels[row_idx, :reference_length].tolist()
                insertions, deletions, substitutions = edit_counts(reference, prediction)
                totals["insertions"] += int(insertions)
                totals["deletions"] += int(deletions)
                totals["substitutions"] += int(substitutions)
                totals["reference_tokens"] += len(reference)
                totals["predicted_tokens"] += len(prediction)
    if int(totals["target_count"]) <= 0 or int(totals["reference_tokens"]) <= 0:
        raise ValueError("Validation target/reference counts are zero; cannot compute metrics.")
    total_errors = int(totals["insertions"]) + int(totals["deletions"]) + int(totals["substitutions"])
    denom_bits = float(totals["target_count"]) * math.log(2.0)
    return {
        "val_ctc_l1_bpphone": float(totals["l1_loss_sum"] / denom_bits),
        "val_ctc_l2_bpphone": float(totals["l2_loss_sum"] / denom_bits),
        "val_ctc_l3_bpphone": float(totals["l3_loss_sum"] / denom_bits),
        "val_ctc_total_bpphone": float((totals["l1_loss_sum"] + totals["l3_loss_sum"] + totals["l2_loss_sum"] * 0.3) / denom_bits),
        "val_final_phoneme_error_rate": float(total_errors / int(totals["reference_tokens"])),
        "edit_diagnostics": {
            "insertions": int(totals["insertions"]),
            "deletions": int(totals["deletions"]),
            "substitutions": int(totals["substitutions"]),
            "total_reference_tokens": int(totals["reference_tokens"]),
            "total_predicted_tokens": int(totals["predicted_tokens"]),
        },
    }


def _aggregate_dataset_metrics(dataset_metrics: dict[str, dict[str, Any]]) -> dict[str, Any]:
    if not dataset_metrics:
        return {}
    numeric_keys = ("val_ctc_l1_bpphone", "val_ctc_l2_bpphone", "val_ctc_l3_bpphone", "val_ctc_total_bpphone", "val_final_phoneme_error_rate")
    aggregate = {}
    for key in numeric_keys:
        aggregate[key] = float(sum(float(metrics[key]) for metrics in dataset_metrics.values()) / len(dataset_metrics))
    return aggregate


def compute_hierarchical_ctc_losses(
    outputs: dict[str, torch.Tensor],
    labels: torch.Tensor,
    label_lengths: torch.Tensor,
    *,
    blank_index: int,
    intermediate_ctc_weight: float,
) -> tuple[torch.Tensor, dict[str, float], int]:
    l1_sum, target_count = compute_ctc_loss_sum(
        outputs["l1"],
        outputs["token_lengths"],
        labels,
        label_lengths,
        blank_index=blank_index,
    )
    l2_sum, _ = compute_ctc_loss_sum(
        outputs["l2"],
        outputs["token_lengths"],
        labels,
        label_lengths,
        blank_index=blank_index,
    )
    l3_sum, _ = compute_ctc_loss_sum(
        outputs["l3"],
        outputs["token_lengths"],
        labels,
        label_lengths,
        blank_index=blank_index,
    )
    total = l1_sum + float(intermediate_ctc_weight) * l2_sum + l3_sum
    return total, {
        "l1_loss_sum": float(l1_sum.item()),
        "l2_loss_sum": float(l2_sum.item()),
        "l3_loss_sum": float(l3_sum.item()),
    }, int(target_count)


def _parse_args() -> CrossTrainedMambaConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=str, default="/Users/home/thesis/data/cache_v1")
    parser.add_argument("--output-root", type=str, default="analysis/active/ssl_experiments/cross_trained_mamba_runs")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--dataset", action="append", dest="datasets", default=None)
    parser.add_argument("--feature-mode", choices=("tx_only", "tx_sbp"), default="tx_sbp")
    parser.add_argument("--area6v-feature-dim", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-steps", type=int, default=60000)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--min-learning-rate", type=float, default=1e-5)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--adam-epsilon", type=float, default=1e-8)
    parser.add_argument("--max-grad-norm", type=float, default=10.0)
    parser.add_argument("--val-every-steps", type=int, default=100)
    parser.add_argument("--checkpoint-every-steps", type=int, default=500)
    parser.add_argument("--progress-every-steps", type=int, default=25)
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--state-size", type=int, default=64)
    parser.add_argument("--stage1-num-layers", type=int, default=2)
    parser.add_argument("--stage2-num-layers", type=int, default=2)
    parser.add_argument("--stage3-num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--adapter-mode", choices=("affine", "stanford_input_net"), default="affine")
    parser.add_argument("--feedback-detach", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--resume-latest", action="store_true")
    parser.add_argument("--resume-checkpoint-path", type=str, default=None)
    args = parser.parse_args()
    return CrossTrainedMambaConfig(
        seed=int(args.seed),
        datasets=tuple(args.datasets) if args.datasets else ("brain2text24", "brain2text25"),
        feature_mode=str(args.feature_mode),
        area6v_feature_dim=int(args.area6v_feature_dim),
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
        progress_every_steps=int(args.progress_every_steps),
        hidden_size=int(args.hidden_size),
        state_size=int(args.state_size),
        stage1_num_layers=int(args.stage1_num_layers),
        stage2_num_layers=int(args.stage2_num_layers),
        stage3_num_layers=int(args.stage3_num_layers),
        dropout=float(args.dropout),
        adapter_mode=str(args.adapter_mode),
        feedback_detach=bool(args.feedback_detach),
        cache_root=str(args.cache_root),
        output_root=str(args.output_root),
        run_name=args.run_name,
        resume_latest=bool(args.resume_latest),
        resume_checkpoint_path=args.resume_checkpoint_path,
    )


def run_cross_trained_mamba(config: CrossTrainedMambaConfig) -> dict[str, Any]:
    return run_cross_trained_mamba_with_callbacks(config)


def run_cross_trained_mamba_with_callbacks(
    config: CrossTrainedMambaConfig,
    *,
    commit_callback: Any | None = None,
) -> dict[str, Any]:
    _seed_all(int(config.seed))
    device = _detect_device()
    problem = build_cross_dataset_problem(
        cache_root=Path(config.cache_root),
        datasets=tuple(config.datasets),
        feature_mode=str(config.feature_mode),
    )
    stats_by_dataset = compute_dataset_train_stats(
        problem=problem,
        area6v_feature_dim=int(config.area6v_feature_dim),
        mode="global",
    )
    run_dir = _resolve_run_dir(config)
    run_dir.mkdir(parents=True, exist_ok=True)
    progress_log_path = run_dir / "progress.jsonl"
    summary_path = run_dir / "summary.json"
    checkpoints_dir = run_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_best_path = run_dir / "checkpoint_best.pt"
    checkpoint_final_path = run_dir / "checkpoint_final.pt"

    train_loaders_by_dataset, grouped_rows_by_dataset = _build_train_loaders(
        problem=problem,
        config=config,
        stats_by_dataset=stats_by_dataset,
        device=device,
    )
    val_loaders = _build_val_loaders(
        problem=problem,
        config=config,
        stats_by_dataset=stats_by_dataset,
        device=device,
    )
    train_adapter_keys_by_dataset = {dataset: tuple(sorted(grouped_rows_by_dataset[dataset])) for dataset in problem["datasets"]}
    all_adapter_keys = tuple(
        key for dataset in problem["datasets"] for key in train_adapter_keys_by_dataset[dataset]
    ) + tuple(
        cross_dataset_adapter_key(row, dataset=dataset)
        for dataset in problem["datasets"]
        for row in problem["rows_by_dataset"][dataset]["val"]
    )

    model = CrossTrainedMambaPhonemeModel(
        input_dim=int(config.input_dim),
        vocab_size=int(problem["vocab"]["num_classes"]),
        hidden_size=int(config.hidden_size),
        state_size=int(config.state_size),
        stage1_num_layers=int(config.stage1_num_layers),
        stage2_num_layers=int(config.stage2_num_layers),
        stage3_num_layers=int(config.stage3_num_layers),
        dropout=float(config.dropout),
        ffn_multiplier=float(config.ffn_multiplier),
        adapter_mode=str(config.adapter_mode),
        session_adapter_keys=tuple(dict.fromkeys(all_adapter_keys)),
        session_adapter_enabled=bool(config.session_adapter_enabled),
        feedback_detach=bool(config.feedback_detach),
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
    progress_printer = ProgressPrinter(
        every_steps=int(config.progress_every_steps),
        every_seconds=float(config.progress_every_seconds),
    )

    start_time = time.time()
    step = 0
    best_progress_payload: dict[str, Any] | None = None
    best_b2t24_per = float("inf")
    train_rng = random.Random(int(config.seed))
    train_iterators_by_dataset = {
        dataset: {adapter_key: iter(loader) for adapter_key, loader in dataset_loaders.items()}
        for dataset, dataset_loaders in train_loaders_by_dataset.items()
    }

    resume_checkpoint = _resolve_resume_checkpoint(run_dir, config)
    if resume_checkpoint is not None:
        payload = torch.load(resume_checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(payload["model_state"])
        optimizer.load_state_dict(payload["optimizer_state"])
        scheduler.load_state_dict(payload["scheduler_state"])
        step = int(payload.get("step", 0))
        restored_best_payload = payload.get("best_progress_payload")
        if isinstance(restored_best_payload, dict):
            best_progress_payload = dict(restored_best_payload)
            best_b2t24_per = float(best_progress_payload.get("brain2text24_val_final_phoneme_error_rate", float("inf")))

    while step < int(config.max_steps):
        current_dataset = train_rng.choice(list(problem["datasets"]))
        current_adapter_key = train_rng.choice(list(train_adapter_keys_by_dataset[current_dataset]))
        accumulated_examples = 0
        accumulation_microbatches = 0
        l1_loss_sum = 0.0
        l2_loss_sum = 0.0
        l3_loss_sum = 0.0
        total_targets = 0
        optimizer.zero_grad(set_to_none=True)
        model.train()

        while accumulated_examples < int(config.batch_size):
            try:
                batch = next(train_iterators_by_dataset[current_dataset][current_adapter_key])
            except StopIteration:
                train_iterators_by_dataset[current_dataset][current_adapter_key] = iter(
                    train_loaders_by_dataset[current_dataset][current_adapter_key]
                )
                batch = next(train_iterators_by_dataset[current_dataset][current_adapter_key])
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
            total_loss_sum, loss_parts, target_count = compute_hierarchical_ctc_losses(
                outputs,
                labels,
                label_lengths,
                blank_index=int(problem["vocab"]["blank_index"]),
                intermediate_ctc_weight=float(config.intermediate_ctc_weight),
            )
            if int(target_count) <= 0:
                continue
            microbatch_examples = int(x.shape[0])
            loss = total_loss_sum / float(target_count)
            scaled_loss = loss * (float(microbatch_examples) / float(config.batch_size))
            scaled_loss.backward()
            accumulated_examples += microbatch_examples
            accumulation_microbatches += 1
            total_targets += int(target_count)
            l1_loss_sum += float(loss_parts["l1_loss_sum"])
            l2_loss_sum += float(loss_parts["l2_loss_sum"])
            l3_loss_sum += float(loss_parts["l3_loss_sum"])

        torch.nn.utils.clip_grad_norm_(model.parameters(), float(config.max_grad_norm))
        optimizer.step()
        scheduler.step()
        step += 1

        train_payload = {
            "event": "cross_mamba_train_report",
            "step": int(step),
            "dataset": str(current_dataset),
            "adapter_key": str(current_adapter_key),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "elapsed_seconds": float(time.time() - start_time),
            "train_ctc_l1_bpphone": float(l1_loss_sum / total_targets / math.log(2.0)),
            "train_ctc_l2_bpphone": float(l2_loss_sum / total_targets / math.log(2.0)),
            "train_ctc_l3_bpphone": float(l3_loss_sum / total_targets / math.log(2.0)),
            "train_ctc_total_bpphone": float((l1_loss_sum + float(config.intermediate_ctc_weight) * l2_loss_sum + l3_loss_sum) / total_targets / math.log(2.0)),
            "accumulated_examples": int(accumulated_examples),
            "accumulation_microbatches": int(accumulation_microbatches),
        }
        if progress_printer.should_print(step, final_step=int(config.max_steps)):
            progress_printer.print(
                prefix="cross-mamba-train",
                step=step,
                total_steps=int(config.max_steps),
                metrics={
                    "dataset": current_dataset,
                    "train_ctc_total_bpphone": float(train_payload["train_ctc_total_bpphone"]),
                    "lr": float(train_payload["learning_rate"]),
                },
            )
            append_jsonl(progress_log_path, train_payload)

        if step % int(config.val_every_steps) == 0 or step == int(config.max_steps):
            dataset_metrics = {
                dataset: _evaluate_loader(
                    model=model,
                    loader=val_loaders[dataset],
                    device=device,
                    blank_index=int(problem["vocab"]["blank_index"]),
                    input_transform_config=input_transform_config,
                )
                for dataset in problem["datasets"]
            }
            aggregate_metrics = _aggregate_dataset_metrics(dataset_metrics)
            b2t24_key = "brain2text24" if "brain2text24" in dataset_metrics else problem["datasets"][0]
            b2t24_per = float(dataset_metrics[b2t24_key]["val_final_phoneme_error_rate"])
            metrics_payload = {
                "event": "cross_mamba_val_report",
                "step": int(step),
                "elapsed_seconds": float(time.time() - start_time),
                "by_dataset": dataset_metrics,
                "aggregate": aggregate_metrics,
                "brain2text24_val_final_phoneme_error_rate": float(b2t24_per),
                "best_brain2text24_val_final_phoneme_error_rate": float(min(best_b2t24_per, b2t24_per)),
            }
            append_jsonl(progress_log_path, metrics_payload)
            progress_printer.print(
                prefix="cross-mamba-val",
                step=step,
                total_steps=int(config.max_steps),
                metrics={
                    "b2t24_per": b2t24_per,
                    "b2t25_per": float(dataset_metrics.get("brain2text25", dataset_metrics[b2t24_key])["val_final_phoneme_error_rate"]),
                    "agg_ctc": float(aggregate_metrics["val_ctc_total_bpphone"]),
                },
            )
            if b2t24_per < best_b2t24_per:
                best_b2t24_per = b2t24_per
                best_progress_payload = dict(metrics_payload)
                _save_checkpoint(
                    checkpoint_path=checkpoint_best_path,
                    config=config,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    step=step,
                    metrics=metrics_payload,
                    best_progress_payload=best_progress_payload,
                    problem=problem,
                )
                if commit_callback is not None:
                    commit_callback()

        if step % int(config.checkpoint_every_steps) == 0 or step == int(config.max_steps):
            checkpoint_path = checkpoints_dir / f"step_{int(step):06d}.pt"
            _save_checkpoint(
                checkpoint_path=checkpoint_path,
                config=config,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                step=step,
                metrics=train_payload,
                best_progress_payload=best_progress_payload,
                problem=problem,
            )
            _prune_step_checkpoints(checkpoints_dir, config.checkpoint_keep_last)
            if commit_callback is not None:
                commit_callback()

    final_metrics = dict(best_progress_payload or {})
    _save_checkpoint(
        checkpoint_path=checkpoint_final_path,
        config=config,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        step=step,
        metrics=final_metrics,
        best_progress_payload=best_progress_payload,
        problem=problem,
    )
    summary = {
        "run_name": str(run_dir.name),
        "datasets": list(problem["datasets"]),
        "feature_mode": str(config.feature_mode),
        "area6v_feature_dim": int(config.area6v_feature_dim),
        "steps": int(step),
        "metrics": final_metrics,
        "progress_log_path": str(progress_log_path),
        "checkpoint_best_path": str(checkpoint_best_path) if checkpoint_best_path.exists() else None,
        "checkpoint_final_path": str(checkpoint_final_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2, default=str))
    if commit_callback is not None:
        commit_callback()
    return json.loads(json.dumps(summary, default=str))


def main() -> None:
    run_cross_trained_mamba(_parse_args())


if __name__ == "__main__":
    main()

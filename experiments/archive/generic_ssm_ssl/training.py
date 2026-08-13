"""Training entrypoints for generic SSM SSL experiments."""

from __future__ import annotations

import json
import math
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

try:
    from utah_ssl.cache import CacheAccessConfig, prepare_cache_context
    from utah_ssl.sampling import build_segment_sampler
    from utah_ssl.ctc import (
        compute_ctc_loss_sum,
        ctc_bits_per_target,
        ctc_greedy_decode,
        edit_counts,
    )
    from utah_ssl.dataset_splits import build_competition_split_problem
    from utah_ssl.sequence_data import (
        CanonicalSequenceDataset,
        LengthAwareBatchSampler,
        canonical_rows_padded_time_percentile,
        collate_sequence_batch,
    )
    from utah_ssl.reporting import ProgressPrinter, append_jsonl, write_metrics_csv
    from utah_ssl.stats import (
        load_precomputed_split_feature_stats,
        resolve_precomputed_split_stats_path,
    )
except ModuleNotFoundError:  # pragma: no cover - repo-root unittest fallback
    from utah_ssl.cache import (
        CacheAccessConfig,
        prepare_cache_context,
    )
    from utah_ssl.sampling import build_segment_sampler
    from utah_ssl.ctc import (
        compute_ctc_loss_sum,
        ctc_bits_per_target,
        ctc_greedy_decode,
        edit_counts,
    )
    from utah_ssl.dataset_splits import build_competition_split_problem
    from utah_ssl.sequence_data import (
        CanonicalSequenceDataset,
        LengthAwareBatchSampler,
        canonical_rows_padded_time_percentile,
        collate_sequence_batch,
    )
    from utah_ssl.reporting import (
        ProgressPrinter,
        append_jsonl,
        write_metrics_csv,
    )
    from utah_ssl.stats import (
        load_precomputed_split_feature_stats,
        resolve_precomputed_split_stats_path,
    )

try:
    from experiments.supervised_baselines.data import (
        WillettInputTransformConfig,
        compute_willett_normalization_stats,
        prepare_willett_inputs,
    )
except ModuleNotFoundError:  # pragma: no cover - repo-root unittest fallback
    from experiments.supervised_baselines.data import (
        WillettInputTransformConfig,
        compute_willett_normalization_stats,
        prepare_willett_inputs,
    )

from .config import GenericSSMSSLConfig
from .model import GenericMaskedSSMModel, GenericSSMCTCModel, make_encoder_from_config
from .objectives import masked_reconstruction_loss


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


def _resolve_run_dir(config: GenericSSMSSLConfig) -> Path:
    name = (
        str(config.run_name)
        if config.run_name is not None
        else f"ssm_ssl_{config.backbone_type}_{config.input_mode}_{_timestamp_utc()}"
    )
    return Path(config.output_root) / name


def _feature_dim_from_rows(rows: tuple[Any, ...], *, feature_mode: str) -> int:
    first = rows[0]
    tx_dim = int(first.n_tx_features)
    sbp_dim = int(first.n_sbp_features)
    return tx_dim if feature_mode == "tx_only" else tx_dim + sbp_dim


def _optimizer(parameters: Any, *, learning_rate: float, weight_decay: float) -> torch.optim.Optimizer:
    return torch.optim.AdamW(parameters, lr=float(learning_rate), weight_decay=float(weight_decay))


def _ctc_input_transform_config(config: GenericSSMSSLConfig) -> WillettInputTransformConfig:
    return WillettInputTransformConfig(
        input_smoothing_sigma_bins=float(config.ctc_input_smoothing_sigma_bins),
        input_smoothing_kernel_size=int(config.ctc_input_smoothing_kernel_size),
        input_smoothing_threshold=float(config.ctc_input_smoothing_threshold),
        white_noise_sd=float(config.ctc_white_noise_sd),
        constant_offset_sd=float(config.ctc_constant_offset_sd),
    )


def _save_ssl_checkpoint(
    path: Path,
    *,
    model: GenericMaskedSSMModel,
    optimizer: torch.optim.Optimizer,
    config: GenericSSMSSLConfig,
    step: int,
    metrics: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_family": "ssm_ssl",
            "stage": "ssl_pretraining",
            "config": config.to_dict(),
            "step": int(step),
            "metrics": dict(metrics),
            "model_state": model.state_dict(),
            "encoder_state": model.encoder.state_dict(),
            "optimizer_state": optimizer.state_dict(),
        },
        path,
    )


def load_encoder_checkpoint(encoder: torch.nn.Module, checkpoint_path: str | Path) -> dict[str, Any]:
    payload = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=False)
    state = payload.get("encoder_state")
    if state is None:
        model_state = payload.get("model_state")
        if not isinstance(model_state, dict):
            raise KeyError("Checkpoint does not contain encoder_state or model_state")
        prefix = "encoder."
        state = {
            key[len(prefix) :]: value
            for key, value in model_state.items()
            if str(key).startswith(prefix)
        }
    encoder.load_state_dict(state)
    return payload


def run_ssl_pretraining(
    config: GenericSSMSSLConfig,
    *,
    run_dir: str | Path | None = None,
) -> dict[str, Any]:
    _seed_all(int(config.seed))
    device = _detect_device()
    resolved_run_dir = Path(run_dir) if run_dir is not None else _resolve_run_dir(config)
    resolved_run_dir.mkdir(parents=True, exist_ok=True)
    progress_log_path = resolved_run_dir / "progress.jsonl"
    metrics_rows: list[dict[str, Any]] = []
    cache_config = CacheAccessConfig(
        dataset_plan=config.dataset_plan,
        signal_spec=config.signal_spec,
        mode=str(config.cache_mode),
        local_cache_base=str(config.local_cache_base),
        seed=int(config.seed),
        segment_bins=int(config.segment_bins),
        use_normalization=bool(config.use_normalization),
        boundary_key_mode=str(config.boundary_key_mode),
        precomputed_session_stats_path=config.precomputed_session_stats_path,
    )
    cache_context = prepare_cache_context(
        cache_candidates=[Path(config.cache_root)],
        config=cache_config,
    )
    train_sampler = build_segment_sampler(
        cache_context,
        "train",
        int(config.batch_size),
        seed=int(config.seed),
        segment_bins=int(config.segment_bins),
        dataset_weight_alpha=0.25,
        examples_per_shard=cache_config.examples_per_shard,
    )
    val_sampler = (
        build_segment_sampler(
            cache_context,
            "val",
            int(config.batch_size),
            seed=int(config.seed) + 1,
            segment_bins=int(config.segment_bins),
            dataset_weight_alpha=0.25,
            examples_per_shard=cache_config.examples_per_shard,
        )
        if cache_context.has_val_datasets
        else None
    )
    model = GenericMaskedSSMModel(
        make_encoder_from_config(config, input_dim=int(cache_context.full_dim))
    ).to(device)
    optimizer = _optimizer(
        model.parameters(),
        learning_rate=float(config.learning_rate),
        weight_decay=float(config.weight_decay),
    )
    generator = None
    if device.type in {"cpu", "cuda"}:
        generator = torch.Generator(device=device)
        generator.manual_seed(int(config.seed) + 13)
    printer = ProgressPrinter(
        every_steps=int(config.progress_every_steps),
        every_seconds=float(config.progress_every_seconds),
    )
    start = time.time()
    best_val_loss = float("inf")
    best_checkpoint_path = resolved_run_dir / "checkpoint_best.pt"
    final_checkpoint_path = resolved_run_dir / "checkpoint_final.pt"

    for step in range(1, int(config.ssl_steps) + 1):
        model.train()
        batch = train_sampler.sample_batch()
        optimizer.zero_grad(set_to_none=True)
        loss, train_metrics = masked_reconstruction_loss(
            model,
            batch,
            device=device,
            time_mask_ratio=float(config.mask_time_ratio),
            channel_mask_ratio=float(config.mask_channel_ratio),
            chunk_size=int(config.mask_chunk_size),
            generator=generator,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(config.max_grad_norm))
        optimizer.step()

        row: dict[str, Any] = {
            "event": "ssl_train",
            "step": int(step),
            "elapsed_seconds": float(time.time() - start),
            **train_metrics,
        }
        metrics_rows.append(dict(row))
        append_jsonl(progress_log_path, row)
        if printer.should_print(step, final_step=int(config.ssl_steps)):
            printer.print(prefix="ssl", step=step, total_steps=int(config.ssl_steps), metrics=train_metrics)

        should_validate = (
            val_sampler is not None
            and (step % int(config.val_every_steps) == 0 or step == int(config.ssl_steps))
        )
        if should_validate:
            model.eval()
            val_losses: list[float] = []
            with torch.no_grad():
                for _ in range(int(config.val_batches)):
                    val_batch = val_sampler.sample_batch()
                    val_loss, _ = masked_reconstruction_loss(
                        model,
                        val_batch,
                        device=device,
                        time_mask_ratio=float(config.mask_time_ratio),
                        channel_mask_ratio=float(config.mask_channel_ratio),
                        chunk_size=int(config.mask_chunk_size),
                        generator=generator,
                    )
                    val_losses.append(float(val_loss.item()))
            val_metrics = {
                "event": "ssl_val",
                "step": int(step),
                "val_loss": float(np.mean(val_losses)),
                "elapsed_seconds": float(time.time() - start),
            }
            metrics_rows.append(dict(val_metrics))
            append_jsonl(progress_log_path, val_metrics)
            if float(val_metrics["val_loss"]) < best_val_loss:
                best_val_loss = float(val_metrics["val_loss"])
                _save_ssl_checkpoint(
                    best_checkpoint_path,
                    model=model,
                    optimizer=optimizer,
                    config=config,
                    step=step,
                    metrics=val_metrics,
                )

        if config.checkpoint_every_steps is not None and step % int(config.checkpoint_every_steps) == 0:
            _save_ssl_checkpoint(
                resolved_run_dir / "checkpoints" / f"step_{step:06d}.pt",
                model=model,
                optimizer=optimizer,
                config=config,
                step=step,
                metrics=row,
            )

    final_metrics = {
        "event": "ssl_final",
        "step": int(config.ssl_steps),
        "best_val_loss": None if math.isinf(best_val_loss) else float(best_val_loss),
        "elapsed_seconds": float(time.time() - start),
    }
    _save_ssl_checkpoint(
        final_checkpoint_path,
        model=model,
        optimizer=optimizer,
        config=config,
        step=int(config.ssl_steps),
        metrics=final_metrics,
    )
    if not best_checkpoint_path.exists():
        _save_ssl_checkpoint(
            best_checkpoint_path,
            model=model,
            optimizer=optimizer,
            config=config,
            step=int(config.ssl_steps),
            metrics=final_metrics,
        )
    metrics_rows.append(dict(final_metrics))
    append_jsonl(progress_log_path, final_metrics)
    write_metrics_csv(resolved_run_dir / "metrics.csv", metrics_rows)
    return {
        "stage": "ssl_pretraining",
        "run_dir": str(resolved_run_dir),
        "progress_log_path": str(progress_log_path),
        "metrics_csv_path": str(resolved_run_dir / "metrics.csv"),
        "best_checkpoint_path": str(best_checkpoint_path),
        "final_checkpoint_path": str(final_checkpoint_path),
        "metrics": final_metrics,
    }


def _make_ctc_loader(
    rows: tuple[Any, ...],
    *,
    cache_root: Path,
    stats: Any,
    signal_spec: Any,
    boundary_key_mode: str,
    dataset: str,
    model_input_dim: int,
    batch_size: int,
    shuffle: bool,
    seed: int,
    device: torch.device,
) -> DataLoader:
    p95 = canonical_rows_padded_time_percentile(rows, percentile=95.0)
    sampler = LengthAwareBatchSampler(
        rows,
        max_examples_per_microbatch=int(batch_size),
        max_padded_time_per_microbatch=int(batch_size) * int(p95),
        shuffle=bool(shuffle),
        seed=int(seed),
    )
    return DataLoader(
        CanonicalSequenceDataset(
            rows,
            cache_root=cache_root,
            signal_spec=signal_spec,
            stats=stats,
            boundary_key_mode=str(boundary_key_mode),
            dataset=str(dataset),
            pad_feature_dim_to=int(model_input_dim),
        ),
        batch_sampler=sampler,
        num_workers=0,
        pin_memory=device.type == "cuda",
        collate_fn=collate_sequence_batch,
    )


def _load_or_compute_ctc_stats(config: GenericSSMSSLConfig, problem: dict[str, Any], sample_dim: int) -> Any:
    if str(config.normalization_mode) == "none":
        return None
    if str(config.normalization_mode) == "global":
        try:
            stats_path = resolve_precomputed_split_stats_path(
                cache_root=Path(problem["cache_root"]),
                dataset=str(problem["dataset"]),
                train_split_name=str(problem["train_split_name"]),
                signal_spec=problem["signal_spec"],
                preferred_path=config.precomputed_split_stats_path,
            )
            (mean_t, std_t), _, loaded_path = load_precomputed_split_feature_stats(
                stats_path=stats_path,
                cache_root=Path(problem["cache_root"]),
                dataset=str(problem["dataset"]),
                signal_spec=problem["signal_spec"],
                boundary_key_mode=str(problem["boundary_key_mode"]),
                train_split_name=str(problem["train_split_name"]),
                val_split_name=str(problem["val_split_name"]),
                split_policy=str(problem["split_policy"]),
            )
            print(f"loaded precomputed generic SSM CTC split stats: {loaded_path}", flush=True)
            return (
                mean_t.numpy().astype(np.float32, copy=False),
                std_t.numpy().astype(np.float32, copy=False),
            )
        except (FileNotFoundError, ValueError) as exc:
            print(f"precomputed split stats unavailable; computing from train rows: {exc}", flush=True)
    return compute_willett_normalization_stats(
        problem["train_rows"],
        cache_root=Path(problem["cache_root"]),
        mode=str(config.normalization_mode),
        feature_mode=str(problem["feature_mode"]),
    )


def _evaluate_ctc(
    *,
    model: GenericSSMCTCModel,
    loader: DataLoader,
    device: torch.device,
    blank_index: int,
    input_transform_config: WillettInputTransformConfig,
) -> dict[str, Any]:
    model.eval()
    total_loss_sum = 0.0
    total_targets = 0
    total_reference_tokens = 0
    total_predicted_tokens = 0
    total_insertions = 0
    total_deletions = 0
    total_substitutions = 0
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
            loss_sum, target_count = compute_ctc_loss_sum(
                outputs["logits"],
                outputs["token_lengths"],
                labels,
                label_lengths,
                blank_index=int(blank_index),
            )
            total_loss_sum += float(loss_sum.item())
            total_targets += int(target_count)
            predictions = ctc_greedy_decode(
                outputs["logits"],
                outputs["token_lengths"],
                blank_index=int(blank_index),
            )
            for row_idx, prediction in enumerate(predictions):
                reference = labels[row_idx, : int(label_lengths[row_idx].item())].tolist()
                insertions, deletions, substitutions = edit_counts(reference, prediction)
                total_insertions += int(insertions)
                total_deletions += int(deletions)
                total_substitutions += int(substitutions)
                total_reference_tokens += len(reference)
                total_predicted_tokens += len(prediction)
    if total_targets <= 0 or total_reference_tokens <= 0:
        raise ValueError("Validation data must contain CTC targets.")
    total_errors = total_insertions + total_deletions + total_substitutions
    return {
        "val_ctc_bpphone": ctc_bits_per_target(total_loss_sum, total_targets),
        "val_phoneme_error_rate": float(total_errors / total_reference_tokens),
        "total_reference_tokens": int(total_reference_tokens),
        "total_predicted_tokens": int(total_predicted_tokens),
        "insertions": int(total_insertions),
        "deletions": int(total_deletions),
        "substitutions": int(total_substitutions),
    }


def run_ctc_finetuning(
    config: GenericSSMSSLConfig,
    *,
    run_dir: str | Path | None = None,
    encoder_checkpoint_path: str | Path | None = None,
    label: str = "pretrained",
) -> dict[str, Any]:
    _seed_all(int(config.seed) + (0 if label == "pretrained" else 1009))
    device = _detect_device()
    resolved_run_dir = Path(run_dir) if run_dir is not None else _resolve_run_dir(config)
    ctc_dir = resolved_run_dir / f"ctc_{label}"
    ctc_dir.mkdir(parents=True, exist_ok=True)
    progress_log_path = ctc_dir / "progress.jsonl"
    metrics_rows: list[dict[str, Any]] = []
    problem = build_competition_split_problem(
        cache_root=Path(config.cache_root),
        signal_spec=config.signal_spec,
        dataset=str(config.dataset),
        boundary_key_mode=str(config.boundary_key_mode),
    )
    raw_sample_dim = int(config.signal_spec.full_dim)
    sample_dim = max(int(raw_sample_dim), int(config.input_dim))
    stats = _load_or_compute_ctc_stats(config, problem, raw_sample_dim)
    train_loader = _make_ctc_loader(
        problem["train_rows"],
        cache_root=Path(problem["cache_root"]),
        stats=stats,
        signal_spec=problem["signal_spec"],
        boundary_key_mode=str(problem["boundary_key_mode"]),
        dataset=str(problem["dataset"]),
        model_input_dim=int(sample_dim),
        batch_size=int(config.batch_size),
        shuffle=True,
        seed=int(config.seed),
        device=device,
    )
    val_loader = _make_ctc_loader(
        problem["val_rows"],
        cache_root=Path(problem["cache_root"]),
        stats=stats,
        signal_spec=problem["signal_spec"],
        boundary_key_mode=str(problem["boundary_key_mode"]),
        dataset=str(problem["dataset"]),
        model_input_dim=int(sample_dim),
        batch_size=int(config.batch_size),
        shuffle=False,
        seed=int(config.seed) + 1,
        device=device,
    )
    model = GenericSSMCTCModel(
        encoder=make_encoder_from_config(config, input_dim=sample_dim),
        vocab_size=int(problem["vocab"]["num_classes"]),
    ).to(device)
    if encoder_checkpoint_path is not None:
        load_encoder_checkpoint(model.encoder, encoder_checkpoint_path)
        model.to(device)
    optimizer = _optimizer(
        model.parameters(),
        learning_rate=float(config.ctc_learning_rate),
        weight_decay=float(config.weight_decay),
    )
    input_transform_config = _ctc_input_transform_config(config)
    train_iter = iter(train_loader)
    printer = ProgressPrinter(
        every_steps=int(config.progress_every_steps),
        every_seconds=float(config.progress_every_seconds),
    )
    start = time.time()
    best_metrics: dict[str, Any] | None = None
    best_step: int | None = None
    best_checkpoint_path = ctc_dir / "checkpoint_best.pt"
    final_checkpoint_path = ctc_dir / "checkpoint_final.pt"

    for step in range(1, int(config.ctc_steps) + 1):
        model.train()
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)
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
        optimizer.zero_grad(set_to_none=True)
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
        loss = loss_sum / int(target_count)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(config.max_grad_norm))
        optimizer.step()
        train_metrics = {
            "event": f"ctc_train_{label}",
            "step": int(step),
            "train_ctc_bpphone": ctc_bits_per_target(loss_sum, target_count),
            "elapsed_seconds": float(time.time() - start),
        }
        metrics_rows.append(dict(train_metrics))
        append_jsonl(progress_log_path, train_metrics)
        if printer.should_print(step, final_step=int(config.ctc_steps)):
            printer.print(
                prefix=f"ctc:{label}",
                step=step,
                total_steps=int(config.ctc_steps),
                metrics={"train_ctc_bpphone": float(train_metrics["train_ctc_bpphone"])},
            )
        if step % int(config.val_every_steps) == 0 or step == int(config.ctc_steps):
            val_metrics = _evaluate_ctc(
                model=model,
                loader=val_loader,
                device=device,
                blank_index=int(problem["vocab"]["blank_index"]),
                input_transform_config=input_transform_config,
            )
            val_row = {
                "event": f"ctc_val_{label}",
                "step": int(step),
                "elapsed_seconds": float(time.time() - start),
                **val_metrics,
            }
            metrics_rows.append(dict(val_row))
            append_jsonl(progress_log_path, val_row)
            if best_metrics is None or float(val_metrics["val_phoneme_error_rate"]) < float(best_metrics["val_phoneme_error_rate"]):
                best_metrics = dict(val_metrics)
                best_step = int(step)
                torch.save(
                    {
                        "model_family": "ssm_ssl",
                        "stage": f"ctc_{label}",
                        "config": config.to_dict(),
                        "step": int(step),
                        "metrics": val_metrics,
                        "model_state": model.state_dict(),
                        "encoder_checkpoint_path": None if encoder_checkpoint_path is None else str(encoder_checkpoint_path),
                    },
                    best_checkpoint_path,
                )

    final_metrics = _evaluate_ctc(
        model=model,
        loader=val_loader,
        device=device,
        blank_index=int(problem["vocab"]["blank_index"]),
        input_transform_config=input_transform_config,
    )
    torch.save(
        {
            "model_family": "ssm_ssl",
            "stage": f"ctc_{label}",
            "config": config.to_dict(),
            "step": int(config.ctc_steps),
            "metrics": final_metrics,
            "model_state": model.state_dict(),
            "encoder_checkpoint_path": None if encoder_checkpoint_path is None else str(encoder_checkpoint_path),
        },
        final_checkpoint_path,
    )
    write_metrics_csv(ctc_dir / "metrics.csv", metrics_rows)
    return {
        "stage": f"ctc_{label}",
        "run_dir": str(ctc_dir),
        "progress_log_path": str(progress_log_path),
        "metrics_csv_path": str(ctc_dir / "metrics.csv"),
        "best_checkpoint_path": str(best_checkpoint_path),
        "final_checkpoint_path": str(final_checkpoint_path),
        "metrics": final_metrics,
        "best_metrics": best_metrics,
        "best_step": best_step,
    }


def run_generic_ssm_ssl(config: GenericSSMSSLConfig) -> dict[str, Any]:
    run_dir = _resolve_run_dir(config)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(config.to_dict(), indent=2, sort_keys=True))
    summary_log_path = run_dir / "summary.jsonl"
    ssl_summary = run_ssl_pretraining(config, run_dir=run_dir)
    append_jsonl(summary_log_path, ssl_summary)
    ctc_summaries: list[dict[str, Any]] = []
    if bool(config.run_downstream_ctc):
        pretrained = run_ctc_finetuning(
            config,
            run_dir=run_dir,
            encoder_checkpoint_path=ssl_summary["best_checkpoint_path"],
            label="pretrained",
        )
        random_init = run_ctc_finetuning(
            config,
            run_dir=run_dir,
            encoder_checkpoint_path=None,
            label="random_init",
        )
        ctc_summaries = [pretrained, random_init]
        for item in ctc_summaries:
            append_jsonl(summary_log_path, item)
    summary = {
        "run_name": run_dir.name,
        "run_dir": str(run_dir),
        "config": config.to_dict(),
        "ssl": ssl_summary,
        "ctc": ctc_summaries,
        "summary_log_path": str(summary_log_path),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, default=str))
    return summary


__all__ = [
    "load_encoder_checkpoint",
    "run_ctc_finetuning",
    "run_generic_ssm_ssl",
    "run_ssl_pretraining",
]

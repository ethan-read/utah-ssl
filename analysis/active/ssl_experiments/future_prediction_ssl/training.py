"""Training and frozen-probe entrypoints for future-prediction SSL."""

from __future__ import annotations

import json
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

_EXPERIMENTS_ROOT = Path(__file__).resolve().parent.parent
if str(_EXPERIMENTS_ROOT) not in sys.path:
    sys.path.insert(0, str(_EXPERIMENTS_ROOT))

try:
    from ssl_core.cache import CacheAccessConfig, CacheContext, build_segment_sampler, prepare_cache_context
    from ssl_core.ctc import (
        CanonicalSequenceDataset,
        LengthAwareBatchSampler,
        build_competition_split_problem,
        canonical_rows_padded_time_percentile,
        collate_sequence_batch,
        compute_ctc_loss_sum,
        ctc_bits_per_target,
        ctc_greedy_decode,
        edit_counts,
    )
    from ssl_core.reporting import ProgressPrinter, append_jsonl, write_metrics_csv
    from ssm_ssl.training import load_encoder_checkpoint
except ModuleNotFoundError:  # pragma: no cover - repo-root unittest fallback
    from analysis.active.ssl_experiments.ssl_core.cache import (
        CacheAccessConfig,
        CacheContext,
        build_segment_sampler,
        prepare_cache_context,
    )
    from analysis.active.ssl_experiments.ssl_core.ctc import (
        CanonicalSequenceDataset,
        LengthAwareBatchSampler,
        build_competition_split_problem,
        canonical_rows_padded_time_percentile,
        collate_sequence_batch,
        compute_ctc_loss_sum,
        ctc_bits_per_target,
        ctc_greedy_decode,
        edit_counts,
    )
    from analysis.active.ssl_experiments.ssl_core.reporting import (
        ProgressPrinter,
        append_jsonl,
        write_metrics_csv,
    )
    from analysis.active.ssl_experiments.ssm_ssl.training import load_encoder_checkpoint

from .config import FuturePredictionSSLConfig
from .model import FuturePredictionCTCProbeModel, FuturePredictionModel, make_future_prediction_model
from .objectives import future_prediction_loss


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


def _resolve_run_dir(config: FuturePredictionSSLConfig) -> Path:
    name = (
        str(config.run_name)
        if config.run_name is not None
        else f"future_prediction_{config.backbone_type}_h{int(config.future_bins)}_{_timestamp_utc()}"
    )
    return Path(config.output_root) / name


def _optimizer(parameters: Any, *, learning_rate: float, weight_decay: float) -> torch.optim.Optimizer:
    return torch.optim.AdamW(parameters, lr=float(learning_rate), weight_decay=float(weight_decay))


def _future_loss_kwargs(
    config: FuturePredictionSSLConfig,
    *,
    cache_context: CacheContext,
) -> dict[str, Any]:
    return {
        "delta": float(config.forecast_loss_delta),
        "temporal_bin_stride": int(config.temporal_bin_stride),
        "variance_match_weight": float(config.variance_match_weight),
        "tx_dim": int(config.tx_dim),
        "sbp_dim": int(config.sbp_dim),
        "feature_mode": str(config.feature_mode),
        "use_normalization": bool(config.use_normalization),
        "tx_loss_type": str(config.tx_loss_type),
        "sbp_loss_type": str(config.sbp_loss_type),
        "session_feature_stats": cache_context.session_feature_stats,
    }


def _probe_stage_name(config: FuturePredictionSSLConfig, *, label: str) -> str:
    if str(config.probe_feature_source) == "encoder_hidden":
        return f"probe_{label}"
    if str(config.probe_feature_source) == "forecast_bin":
        return f"probe_{label}_forecast_tplus{int(config.probe_forecast_horizon_index) + 1}"
    raise ValueError(f"Unsupported probe_feature_source: {config.probe_feature_source!r}")


def _restrict_cache_context(cache_context: CacheContext, *, datasets: tuple[str, ...]) -> None:
    available = set(str(name) for name in cache_context.rows_by_dataset)
    missing = [name for name in datasets if name not in available]
    if missing:
        raise KeyError(f"Requested pretrain datasets not found in cache context: {missing}")
    cache_context.pretrain_datasets = tuple(datasets)
    cache_context.has_val_datasets = any(
        len(cache_context.split_rows_by_dataset["val"].get(dataset, ())) > 0
        for dataset in cache_context.pretrain_datasets
    )


def _prepare_future_cache_context(config: FuturePredictionSSLConfig) -> CacheContext:
    cache_root = Path(config.cache_root)
    available_datasets = sorted(
        path.name for path in cache_root.iterdir() if path.is_dir() and (path / "metadata.json").exists()
    )
    requested = tuple(str(name) for name in config.pretrain_datasets)
    missing = [name for name in requested if name not in available_datasets]
    if missing:
        raise FileNotFoundError(
            f"Requested pretrain dataset(s) not found under {cache_root}: {missing}. "
            f"Available datasets: {available_datasets}"
        )
    excluded_datasets = tuple(name for name in available_datasets if name not in set(requested))
    cache_config = CacheAccessConfig(
        mode=str(config.cache_mode),
        local_cache_base=str(config.local_cache_base),
        excluded_datasets=excluded_datasets,
        seed=int(config.seed),
        segment_bins=int(config.segment_bins),
        use_normalization=bool(config.use_normalization),
        tx_dim=int(config.tx_dim),
        sbp_dim=int(config.sbp_dim),
        feature_mode=str(config.feature_mode),
        boundary_key_mode=str(config.boundary_key_mode),
        precomputed_session_stats_path=config.precomputed_session_stats_path,
    )
    cache_context = prepare_cache_context(
        cache_candidates=[cache_root],
        config=cache_config,
    )
    _restrict_cache_context(cache_context, datasets=tuple(config.pretrain_datasets))
    return cache_context


def _feature_dim_from_rows(rows: tuple[Any, ...] | list[Any], *, feature_mode: str) -> int:
    first = rows[0]
    tx_dim = int(first.n_tx_features)
    sbp_dim = int(first.n_sbp_features)
    return tx_dim if feature_mode == "tx_only" else tx_dim + sbp_dim


def _stats_to_numpy(
    stats: dict[str, tuple[torch.Tensor, torch.Tensor]] | None,
) -> dict[str, tuple[np.ndarray, np.ndarray]] | None:
    if not stats:
        return None
    converted: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for key, (mean_t, std_t) in stats.items():
        pair = (
            mean_t.detach().cpu().numpy().astype(np.float32, copy=False),
            std_t.detach().cpu().numpy().astype(np.float32, copy=False),
        )
        key_str = str(key)
        converted[key_str] = pair
        if ":" in key_str:
            converted.setdefault(key_str.split(":", 1)[1], pair)
    return converted


def _save_future_checkpoint(
    path: Path,
    *,
    model: FuturePredictionModel,
    optimizer: torch.optim.Optimizer,
    config: FuturePredictionSSLConfig,
    step: int,
    metrics: dict[str, Any],
    checkpoint_kind: str = "step",
    best_val_loss: float | None = None,
    train_sampler: Any | None = None,
    val_sampler: Any | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_family": "future_prediction_ssl",
            "stage": "ssl_pretraining",
            "config": config.to_dict(),
            "step": int(step),
            "metrics": dict(metrics),
            "checkpoint_kind": str(checkpoint_kind),
            "best_val_loss": None if best_val_loss is None else float(best_val_loss),
            "model_state": model.state_dict(),
            "encoder_state": model.encoder.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "rng_state": _capture_rng_state(),
            "sampler_state": {
                "train": _capture_sampler_state(train_sampler),
                "val": _capture_sampler_state(val_sampler),
            },
        },
        path,
    )


def _capture_rng_state() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        payload["torch_cuda"] = torch.cuda.get_rng_state_all()
    return payload


def _restore_rng_state(payload: dict[str, Any] | None) -> None:
    if not payload:
        return
    if payload.get("python") is not None:
        random.setstate(payload["python"])
    if payload.get("numpy") is not None:
        np.random.set_state(payload["numpy"])
    if payload.get("torch") is not None:
        torch.set_rng_state(payload["torch"])
    if torch.cuda.is_available() and payload.get("torch_cuda") is not None:
        torch.cuda.set_rng_state_all(payload["torch_cuda"])


def _capture_sampler_state(sampler: Any | None) -> dict[str, Any] | None:
    if sampler is None:
        return None
    py_rng = getattr(sampler, "py_rng", None)
    np_rng = getattr(sampler, "np_rng", None)
    if py_rng is None or np_rng is None:
        return None
    return {
        "python": py_rng.getstate(),
        "numpy_bit_generator": np_rng.bit_generator.state,
    }


def _restore_sampler_state(sampler: Any | None, payload: dict[str, Any] | None) -> None:
    if sampler is None or not payload:
        return
    py_rng = getattr(sampler, "py_rng", None)
    np_rng = getattr(sampler, "np_rng", None)
    if py_rng is not None and payload.get("python") is not None:
        py_rng.setstate(payload["python"])
    if np_rng is not None and payload.get("numpy_bit_generator") is not None:
        np_rng.bit_generator.state = payload["numpy_bit_generator"]


def _load_checkpoint_payload(path: str | Path) -> dict[str, Any]:
    # Resume payloads intentionally include Python, NumPy, and sampler states.
    return torch.load(Path(path), map_location="cpu", weights_only=False)


def _load_future_model_checkpoint(model: FuturePredictionModel, checkpoint_path: str | Path) -> None:
    payload = _load_checkpoint_payload(checkpoint_path)
    model.load_state_dict(payload["model_state"])


def _checkpoint_step(path: Path) -> int:
    try:
        payload = _load_checkpoint_payload(path)
    except Exception:
        return -1
    return int(payload.get("step", -1))


def _latest_stage_checkpoint(
    *,
    stage_dir: Path,
    final_checkpoint_path: Path,
    best_checkpoint_path: Path,
) -> Path | None:
    candidates: list[Path] = []
    checkpoints_dir = stage_dir / "checkpoints"
    if checkpoints_dir.exists():
        candidates.extend(sorted(checkpoints_dir.glob("step_*.pt")))
    for direct in (final_checkpoint_path, best_checkpoint_path):
        if direct.exists():
            candidates.append(direct)
    if not candidates:
        return None
    return max(candidates, key=_checkpoint_step)


def _resume_compatible_config(current: FuturePredictionSSLConfig, payload: dict[str, Any]) -> None:
    recovered = FuturePredictionSSLConfig.from_dict(dict(payload.get("config", {})))
    keys = (
        "backbone_type",
        "feature_mode",
        "boundary_key_mode",
        "tx_dim",
        "sbp_dim",
        "segment_bins",
        "future_bins",
        "hidden_size",
        "state_size",
        "num_layers",
        "input_mode",
        "direction",
        "pretrain_datasets",
    )
    mismatches = []
    for key in keys:
        if getattr(current, key) != getattr(recovered, key):
            mismatches.append(f"{key}: current={getattr(current, key)!r} checkpoint={getattr(recovered, key)!r}")
    if mismatches:
        raise ValueError("Resume checkpoint is incompatible with the active config.\n" + "\n".join(mismatches))


def _parse_existing_progress(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def _make_probe_loader(
    rows: tuple[Any, ...] | list[Any],
    *,
    cache_root: Path,
    stats: dict[str, tuple[np.ndarray, np.ndarray]] | None,
    feature_mode: str,
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
            stats=stats,
            feature_mode=str(feature_mode),
            boundary_key_mode=str(boundary_key_mode),
            dataset=str(dataset),
            pad_feature_dim_to=int(model_input_dim),
        ),
        batch_sampler=sampler,
        num_workers=0,
        pin_memory=device.type == "cuda",
        collate_fn=collate_sequence_batch,
    )


def _freeze_module(module: torch.nn.Module) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(False)


def _evaluate_frozen_probe(
    *,
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    blank_index: int,
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


def run_future_prediction_pretraining(
    config: FuturePredictionSSLConfig,
    *,
    run_dir: str | Path | None = None,
) -> dict[str, Any]:
    _seed_all(int(config.seed))
    device = _detect_device()
    resolved_run_dir = Path(run_dir) if run_dir is not None else _resolve_run_dir(config)
    resolved_run_dir.mkdir(parents=True, exist_ok=True)
    progress_log_path = resolved_run_dir / "progress.jsonl"
    metrics_rows: list[dict[str, Any]] = []
    if bool(config.resume) and config.resume_checkpoint_path is not None:
        explicit_resume_path = Path(config.resume_checkpoint_path)
        if not explicit_resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint does not exist: {explicit_resume_path}")
    cache_context = _prepare_future_cache_context(config)
    train_sampler = build_segment_sampler(
        cache_context,
        "train",
        int(config.batch_size),
        seed=int(config.seed),
        segment_bins=int(config.segment_bins),
        dataset_weight_alpha=0.25,
        examples_per_shard=cache_context.config.examples_per_shard,
    )
    val_sampler = (
        build_segment_sampler(
            cache_context,
            "val",
            int(config.batch_size),
            seed=int(config.seed) + 1,
            segment_bins=int(config.segment_bins),
            dataset_weight_alpha=0.25,
            examples_per_shard=cache_context.config.examples_per_shard,
        )
        if cache_context.has_val_datasets
        else None
    )
    model = make_future_prediction_model(config, input_dim=int(cache_context.full_dim)).to(device)
    optimizer = _optimizer(
        model.parameters(),
        learning_rate=float(config.learning_rate),
        weight_decay=float(config.weight_decay),
    )
    printer = ProgressPrinter(
        every_steps=int(config.progress_every_steps),
        every_seconds=float(config.progress_every_seconds),
    )
    start = time.time()
    best_val_loss = float("inf")
    best_checkpoint_path = resolved_run_dir / "checkpoint_best.pt"
    final_checkpoint_path = resolved_run_dir / "checkpoint_final.pt"
    start_step = 1
    if bool(config.resume):
        checkpoint_path = (
            Path(config.resume_checkpoint_path)
            if config.resume_checkpoint_path is not None
            else _latest_stage_checkpoint(
                stage_dir=resolved_run_dir,
                final_checkpoint_path=final_checkpoint_path,
                best_checkpoint_path=best_checkpoint_path,
            )
        )
        if checkpoint_path is not None and checkpoint_path.exists():
            payload = _load_checkpoint_payload(checkpoint_path)
            _resume_compatible_config(config, payload)
            model.load_state_dict(payload["model_state"])
            optimizer.load_state_dict(payload["optimizer_state"])
            recovered_best_val_loss = payload.get("best_val_loss")
            best_val_loss = float("inf") if recovered_best_val_loss is None else float(recovered_best_val_loss)
            _restore_rng_state(payload.get("rng_state"))
            sampler_state = payload.get("sampler_state", {})
            _restore_sampler_state(train_sampler, sampler_state.get("train"))
            _restore_sampler_state(val_sampler, sampler_state.get("val"))
            start_step = int(payload.get("step", 0)) + 1
            append_jsonl(
                progress_log_path,
                {
                    "event": "future_resume",
                    "step": int(start_step),
                    "resume_checkpoint_path": str(checkpoint_path),
                    "elapsed_seconds": 0.0,
                },
            )
    existing_progress = _parse_existing_progress(progress_log_path)
    if existing_progress:
        metrics_rows.extend(existing_progress)
    if start_step > int(config.ssl_steps):
        final_payload = _load_checkpoint_payload(final_checkpoint_path if final_checkpoint_path.exists() else best_checkpoint_path)
        write_metrics_csv(resolved_run_dir / "metrics.csv", metrics_rows)
        return {
            "stage": "ssl_pretraining",
            "run_dir": str(resolved_run_dir),
            "progress_log_path": str(progress_log_path),
            "metrics_csv_path": str(resolved_run_dir / "metrics.csv"),
            "best_checkpoint_path": str(best_checkpoint_path),
            "final_checkpoint_path": str(final_checkpoint_path),
            "metrics": final_payload.get("metrics", {}),
            "pretrain_datasets": list(config.pretrain_datasets),
            "resumed": True,
            "resume_completed": True,
        }

    for step in range(start_step, int(config.ssl_steps) + 1):
        model.train()
        batch = train_sampler.sample_batch()
        optimizer.zero_grad(set_to_none=True)
        loss, train_metrics = future_prediction_loss(
            model,
            batch,
            device=device,
            **_future_loss_kwargs(config, cache_context=cache_context),
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(config.max_grad_norm))
        optimizer.step()

        row: dict[str, Any] = {
            "event": "future_train",
            "step": int(step),
            "elapsed_seconds": float(time.time() - start),
            **train_metrics,
        }
        metrics_rows.append(dict(row))
        append_jsonl(progress_log_path, row)
        if printer.should_print(step, final_step=int(config.ssl_steps)):
            printer.print(
                prefix="future",
                step=step,
                total_steps=int(config.ssl_steps),
                metrics={
                    "loss": float(train_metrics["loss"]),
                    "h1_mae": float(train_metrics.get("h1_mae", 0.0)),
                    "h3_mae": float(train_metrics.get(f"h{int(config.future_bins)}_mae", 0.0)),
                },
            )
        should_validate = val_sampler is not None and (
            step % int(config.val_every_steps) == 0 or step == int(config.ssl_steps)
        )
        if should_validate:
            model.eval()
            val_rows: list[dict[str, Any]] = []
            with torch.no_grad():
                for _ in range(int(config.val_batches)):
                    val_batch = val_sampler.sample_batch()
                    _, val_metrics = future_prediction_loss(
                        model,
                        val_batch,
                        device=device,
                        **_future_loss_kwargs(config, cache_context=cache_context),
                    )
                    val_rows.append(val_metrics)
            mean_val_metrics = {
                key: float(np.mean([item[key] for item in val_rows]))
                for key in val_rows[0]
            }
            val_row = {
                "event": "future_val",
                "step": int(step),
                "elapsed_seconds": float(time.time() - start),
                **{f"val_{key}": value for key, value in mean_val_metrics.items()},
            }
            metrics_rows.append(dict(val_row))
            append_jsonl(progress_log_path, val_row)
            if float(mean_val_metrics["loss"]) < best_val_loss:
                best_val_loss = float(mean_val_metrics["loss"])
                _save_future_checkpoint(
                    best_checkpoint_path,
                    model=model,
                    optimizer=optimizer,
                    config=config,
                    step=int(step),
                    metrics=val_row,
                    checkpoint_kind="best",
                    best_val_loss=best_val_loss,
                    train_sampler=train_sampler,
                    val_sampler=val_sampler,
                )
        if config.checkpoint_every_steps is not None and step % int(config.checkpoint_every_steps) == 0:
            _save_future_checkpoint(
                resolved_run_dir / "checkpoints" / f"step_{int(step):06d}.pt",
                model=model,
                optimizer=optimizer,
                config=config,
                step=int(step),
                metrics=row,
                checkpoint_kind="step",
                best_val_loss=best_val_loss,
                train_sampler=train_sampler,
                val_sampler=val_sampler,
            )

    final_batch = train_sampler.sample_batch()
    _, final_train_metrics = future_prediction_loss(
        model,
        final_batch,
        device=device,
        **_future_loss_kwargs(config, cache_context=cache_context),
    )
    _save_future_checkpoint(
        final_checkpoint_path,
        model=model,
        optimizer=optimizer,
        config=config,
        step=int(config.ssl_steps),
        metrics=final_train_metrics,
        checkpoint_kind="final",
        best_val_loss=best_val_loss,
        train_sampler=train_sampler,
        val_sampler=val_sampler,
    )
    write_metrics_csv(resolved_run_dir / "metrics.csv", metrics_rows)
    return {
        "stage": "ssl_pretraining",
        "run_dir": str(resolved_run_dir),
        "progress_log_path": str(progress_log_path),
        "metrics_csv_path": str(resolved_run_dir / "metrics.csv"),
        "best_checkpoint_path": str(best_checkpoint_path),
        "final_checkpoint_path": str(final_checkpoint_path),
        "metrics": final_train_metrics,
        "pretrain_datasets": list(config.pretrain_datasets),
        "resumed": start_step > 1,
        "resume_completed": False,
    }


def _save_probe_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    config: FuturePredictionSSLConfig,
    step: int,
    metrics: dict[str, Any],
    label: str,
    checkpoint_kind: str,
    encoder_checkpoint_path: str | Path | None,
    train_iter_step: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_family": "future_prediction_ssl",
            "stage": f"probe_{label}",
            "config": config.to_dict(),
            "step": int(step),
            "metrics": dict(metrics),
            "checkpoint_kind": str(checkpoint_kind),
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "encoder_checkpoint_path": None if encoder_checkpoint_path is None else str(encoder_checkpoint_path),
            "rng_state": _capture_rng_state(),
            "train_iter_step": int(train_iter_step),
        },
        path,
    )


def run_frozen_linear_ctc_probe(
    config: FuturePredictionSSLConfig,
    *,
    run_dir: str | Path | None = None,
    encoder_checkpoint_path: str | Path | None,
    label: str,
) -> dict[str, Any]:
    _seed_all(int(config.seed) + (0 if label == "pretrained" else 1009))
    device = _detect_device()
    resolved_run_dir = Path(run_dir) if run_dir is not None else _resolve_run_dir(config)
    probe_stage_name = _probe_stage_name(config, label=label)
    probe_dir = resolved_run_dir / probe_stage_name
    probe_dir.mkdir(parents=True, exist_ok=True)
    progress_log_path = probe_dir / "progress.jsonl"
    metrics_rows: list[dict[str, Any]] = []

    cache_context = _prepare_future_cache_context(config)
    problem = build_competition_split_problem(
        cache_root=Path(config.cache_root),
        dataset=str(config.dataset),
        feature_mode=str(config.feature_mode),
        boundary_key_mode=str(config.boundary_key_mode),
    )
    raw_sample_dim = _feature_dim_from_rows(problem["train_rows"], feature_mode=str(config.feature_mode))
    sample_dim = max(int(raw_sample_dim), int(config.input_dim))
    stats = _stats_to_numpy(cache_context.session_feature_stats)
    train_loader = _make_probe_loader(
        problem["train_rows"],
        cache_root=Path(problem["cache_root"]),
        stats=stats,
        feature_mode=str(problem["feature_mode"]),
        boundary_key_mode=str(problem["boundary_key_mode"]),
        dataset=str(problem["dataset"]),
        model_input_dim=int(sample_dim),
        batch_size=int(config.probe_batch_size),
        shuffle=True,
        seed=int(config.seed),
        device=device,
    )
    val_loader = _make_probe_loader(
        problem["val_rows"],
        cache_root=Path(problem["cache_root"]),
        stats=stats,
        feature_mode=str(problem["feature_mode"]),
        boundary_key_mode=str(problem["boundary_key_mode"]),
        dataset=str(problem["dataset"]),
        model_input_dim=int(sample_dim),
        batch_size=int(config.probe_batch_size),
        shuffle=False,
        seed=int(config.seed) + 1,
        device=device,
    )

    future_model = make_future_prediction_model(config, input_dim=sample_dim)
    if encoder_checkpoint_path is not None:
        if str(config.probe_feature_source) == "forecast_bin":
            _load_future_model_checkpoint(future_model, encoder_checkpoint_path)
        else:
            load_encoder_checkpoint(future_model.encoder, encoder_checkpoint_path)
    model = FuturePredictionCTCProbeModel(
        future_model=future_model,
        vocab_size=int(problem["vocab"]["num_classes"]),
        feature_source=str(config.probe_feature_source),
        forecast_horizon_index=int(config.probe_forecast_horizon_index),
    ).to(device)
    if str(config.probe_feature_source) == "forecast_bin":
        _freeze_module(model.future_model)
    else:
        _freeze_module(model.future_model.encoder)
    optimizer = _optimizer(
        model.classifier.parameters(),
        learning_rate=float(config.probe_learning_rate),
        weight_decay=float(config.probe_weight_decay),
    )
    train_iter = iter(train_loader)
    printer = ProgressPrinter(
        every_steps=int(config.progress_every_steps),
        every_seconds=float(config.progress_every_seconds),
    )
    start = time.time()
    best_metrics: dict[str, Any] | None = None
    best_step: int | None = None
    best_checkpoint_path = probe_dir / "checkpoint_best.pt"
    final_checkpoint_path = probe_dir / "checkpoint_final.pt"
    start_step = 1
    train_iter_step = 0
    if bool(config.resume):
        checkpoint_path = _latest_stage_checkpoint(
            stage_dir=probe_dir,
            final_checkpoint_path=final_checkpoint_path,
            best_checkpoint_path=best_checkpoint_path,
        )
        if checkpoint_path is not None and checkpoint_path.exists():
            payload = _load_checkpoint_payload(checkpoint_path)
            model.load_state_dict(payload["model_state"])
            optimizer.load_state_dict(payload["optimizer_state"])
            _restore_rng_state(payload.get("rng_state"))
            best_metrics = dict(payload.get("metrics", {})) if payload.get("checkpoint_kind") == "best" else None
            best_step = int(payload.get("step")) if payload.get("checkpoint_kind") == "best" else None
            start_step = int(payload.get("step", 0)) + 1
            train_iter_step = int(payload.get("train_iter_step", 0))
            append_jsonl(
                progress_log_path,
                {
                    "event": f"probe_resume_{label}",
                    "step": int(start_step),
                    "resume_checkpoint_path": str(checkpoint_path),
                    "elapsed_seconds": 0.0,
                },
            )
    existing_progress = _parse_existing_progress(progress_log_path)
    if existing_progress:
        metrics_rows.extend(existing_progress)
    for _ in range(train_iter_step):
        try:
            next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            next(train_iter)
    if start_step > int(config.probe_steps):
        final_payload = _load_checkpoint_payload(final_checkpoint_path if final_checkpoint_path.exists() else best_checkpoint_path)
        write_metrics_csv(probe_dir / "metrics.csv", metrics_rows)
        return {
            "stage": probe_stage_name,
            "run_dir": str(probe_dir),
            "progress_log_path": str(progress_log_path),
            "metrics_csv_path": str(probe_dir / "metrics.csv"),
            "best_checkpoint_path": str(best_checkpoint_path),
            "final_checkpoint_path": str(final_checkpoint_path),
            "metrics": final_payload.get("metrics", {}),
            "best_metrics": best_metrics,
            "best_step": best_step,
            "resumed": True,
            "resume_completed": True,
        }

    for step in range(start_step, int(config.probe_steps) + 1):
        model.train()
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)
        train_iter_step += 1
        x = batch["x"].to(device)
        input_lengths = batch["input_lengths"].to(device)
        labels = batch["labels"].to(device)
        label_lengths = batch["label_lengths"].to(device)
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
        torch.nn.utils.clip_grad_norm_(model.classifier.parameters(), float(config.max_grad_norm))
        optimizer.step()
        train_metrics = {
            "event": f"probe_train_{label}",
            "step": int(step),
            "train_ctc_bpphone": ctc_bits_per_target(loss_sum, target_count),
            "elapsed_seconds": float(time.time() - start),
        }
        metrics_rows.append(dict(train_metrics))
        append_jsonl(progress_log_path, train_metrics)
        if printer.should_print(step, final_step=int(config.probe_steps)):
            printer.print(
                prefix=f"probe:{label}",
                step=step,
                total_steps=int(config.probe_steps),
                metrics={"train_ctc_bpphone": float(train_metrics["train_ctc_bpphone"])},
            )
        if step % int(config.val_every_steps) == 0 or step == int(config.probe_steps):
            val_metrics = _evaluate_frozen_probe(
                model=model,
                loader=val_loader,
                device=device,
                blank_index=int(problem["vocab"]["blank_index"]),
            )
            val_row = {
                "event": f"probe_val_{label}",
                "step": int(step),
                "elapsed_seconds": float(time.time() - start),
                **val_metrics,
            }
            metrics_rows.append(dict(val_row))
            append_jsonl(progress_log_path, val_row)
            if best_metrics is None or float(val_metrics["val_phoneme_error_rate"]) < float(best_metrics["val_phoneme_error_rate"]):
                best_metrics = dict(val_metrics)
                best_step = int(step)
                _save_probe_checkpoint(
                    best_checkpoint_path,
                    model=model,
                    optimizer=optimizer,
                    config=config,
                    step=int(step),
                    metrics=val_metrics,
                    label=label,
                    checkpoint_kind="best",
                    encoder_checkpoint_path=encoder_checkpoint_path,
                    train_iter_step=train_iter_step,
                )
        if config.checkpoint_every_steps is not None and step % int(config.checkpoint_every_steps) == 0:
            _save_probe_checkpoint(
                probe_dir / "checkpoints" / f"step_{int(step):06d}.pt",
                model=model,
                optimizer=optimizer,
                config=config,
                step=int(step),
                metrics=train_metrics,
                label=label,
                checkpoint_kind="step",
                encoder_checkpoint_path=encoder_checkpoint_path,
                train_iter_step=train_iter_step,
            )

    final_metrics = _evaluate_frozen_probe(
        model=model,
        loader=val_loader,
        device=device,
        blank_index=int(problem["vocab"]["blank_index"]),
    )
    _save_probe_checkpoint(
        final_checkpoint_path,
        model=model,
        optimizer=optimizer,
        config=config,
        step=int(config.probe_steps),
        metrics=final_metrics,
        label=label,
        checkpoint_kind="final",
        encoder_checkpoint_path=encoder_checkpoint_path,
        train_iter_step=train_iter_step,
    )
    write_metrics_csv(probe_dir / "metrics.csv", metrics_rows)
    return {
        "stage": probe_stage_name,
        "run_dir": str(probe_dir),
        "progress_log_path": str(progress_log_path),
        "metrics_csv_path": str(probe_dir / "metrics.csv"),
        "best_checkpoint_path": str(best_checkpoint_path),
        "final_checkpoint_path": str(final_checkpoint_path),
        "metrics": final_metrics,
        "best_metrics": best_metrics,
        "best_step": best_step,
        "resumed": start_step > 1,
        "resume_completed": False,
    }


def run_future_prediction_ssl(config: FuturePredictionSSLConfig) -> dict[str, Any]:
    run_dir = _resolve_run_dir(config)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(config.to_dict(), indent=2, sort_keys=True))
    summary_log_path = run_dir / "summary.jsonl"
    ssl_summary = run_future_prediction_pretraining(config, run_dir=run_dir)
    append_jsonl(summary_log_path, ssl_summary)
    probe_summaries: list[dict[str, Any]] = []
    if bool(config.run_frozen_probe):
        pretrained_encoder_checkpoint_path = (
            ssl_summary["best_checkpoint_path"]
            if Path(ssl_summary["best_checkpoint_path"]).exists()
            else ssl_summary["final_checkpoint_path"]
        )
        pretrained = run_frozen_linear_ctc_probe(
            config,
            run_dir=run_dir,
            encoder_checkpoint_path=pretrained_encoder_checkpoint_path,
            label="pretrained",
        )
        random_init = run_frozen_linear_ctc_probe(
            config,
            run_dir=run_dir,
            encoder_checkpoint_path=None,
            label="random_init",
        )
        probe_summaries = [pretrained, random_init]
        for item in probe_summaries:
            append_jsonl(summary_log_path, item)
    summary = {
        "run_name": run_dir.name,
        "run_dir": str(run_dir),
        "config": config.to_dict(),
        "ssl": ssl_summary,
        "probe": probe_summaries,
        "summary_log_path": str(summary_log_path),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, default=str))
    return summary


__all__ = [
    "FuturePredictionSSLConfig",
    "load_encoder_checkpoint",
    "run_frozen_linear_ctc_probe",
    "run_future_prediction_pretraining",
    "run_future_prediction_ssl",
]

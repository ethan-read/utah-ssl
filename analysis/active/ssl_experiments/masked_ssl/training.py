"""Masked-reconstruction S5 training helpers for Colab experiments."""

from __future__ import annotations

import json
import random
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .cache import CacheContext, build_segment_sampler
from .cache import resolve_boundary_key
from .model import MaskedSSLModel, sync_device
from .objectives import compute_objective_metrics, summarize_metrics


def _seed_training_run(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


@dataclass
class SSLTrainingConfig:
    seed: int = 7
    objective_mode: str = "masked_reconstruction"
    feature_mode: str = "tx_only"
    boundary_key_mode: str = "session"
    segment_bins: int = 80
    patch_size: int = 4
    patch_stride: int = 2
    hidden_size: int = 256
    s5_state_size: int = 128
    num_layers: int = 2
    dropout: float = 0.1
    batch_size: int = 32
    num_steps: int = 600
    learning_rate: float = 3e-4
    weight_decay: float = 1e-2
    val_every: int = 50
    val_batches: int = 10
    checkpoint_every_steps: int | None = None
    dataset_weight_alpha: float = 0.25
    examples_per_shard: int = 8
    log_every: int = 10
    post_proj_norm: str = "rms"
    reconstruction_head_mode: str = "no_output_norm"
    reconstruction_head_type: str = "mlp"
    backbone_direction: str = "bidirectional"
    mask_unit: str = "patch"
    mask_token_placement: str = "before_projection"
    mask_ratio: float = 0.15
    span_length_min: int = 1
    span_length_max: int = 1
    num_spans_mode: str = "multiple"
    reconstruct_target: str = "raw_patch"
    loss_mode: str = "masked_only"
    allow_bin_fractional_overlap: bool = True

    def __post_init__(self) -> None:
        if self.objective_mode != "masked_reconstruction":
            raise ValueError("objective_mode must be 'masked_reconstruction'")
        if self.feature_mode not in {"tx_only", "tx_sbp"}:
            raise ValueError("feature_mode must be one of {'tx_only', 'tx_sbp'}")
        if self.boundary_key_mode not in {"session", "subject_if_available"}:
            raise ValueError(
                "boundary_key_mode must be one of {'session', 'subject_if_available'}"
            )
        if self.patch_stride > self.patch_size:
            raise ValueError("patch_stride must be <= patch_size")
        if self.checkpoint_every_steps is not None and int(self.checkpoint_every_steps) <= 0:
            raise ValueError("checkpoint_every_steps must be positive when provided")
        if self.mask_unit not in {"patch", "bin"}:
            raise ValueError("mask_unit must be one of {'patch', 'bin'}")
        if self.mask_token_placement not in {"before_projection", "after_projection", "skip"}:
            raise ValueError(
                "mask_token_placement must be one of {'before_projection', 'after_projection', 'skip'}"
            )
        if not (0.0 < float(self.mask_ratio) <= 1.0):
            raise ValueError("mask_ratio must be in the interval (0, 1].")
        if int(self.span_length_min) <= 0 or int(self.span_length_max) <= 0:
            raise ValueError("span lengths must be positive.")
        if int(self.span_length_min) > int(self.span_length_max):
            raise ValueError("span_length_min must be <= span_length_max.")
        if self.num_spans_mode not in {"one", "multiple"}:
            raise ValueError("num_spans_mode must be one of {'one', 'multiple'}")
        if self.reconstruct_target != "raw_patch":
            raise ValueError("reconstruct_target must be 'raw_patch'")
        if self.loss_mode != "masked_only":
            raise ValueError("loss_mode must be 'masked_only'")
        if self.reconstruction_head_mode not in {"with_output_norm", "no_output_norm"}:
            raise ValueError(
                "reconstruction_head_mode must be one of {'with_output_norm', 'no_output_norm'}"
            )
        if self.reconstruction_head_type not in {"linear", "mlp"}:
            raise ValueError("reconstruction_head_type must be one of {'linear', 'mlp'}")
        if self.backbone_direction not in {"causal", "bidirectional"}:
            raise ValueError("backbone_direction must be one of {'causal', 'bidirectional'}")

    def checkpoint_config(self) -> dict[str, Any]:
        return {
            "feature_mode": str(self.feature_mode),
            "boundary_key_mode": str(self.boundary_key_mode),
            "patch_size": int(self.patch_size),
            "patch_stride": int(self.patch_stride),
            "hidden_size": int(self.hidden_size),
            "s5_state_size": int(self.s5_state_size),
            "num_layers": int(self.num_layers),
            "dropout": float(self.dropout),
            "post_proj_norm": str(self.post_proj_norm),
            "reconstruction_head_mode": str(self.reconstruction_head_mode),
            "reconstruction_head_type": str(self.reconstruction_head_type),
            "backbone_direction": str(self.backbone_direction),
        }


def _build_ssl_optimizer(
    model: MaskedSSLModel,
    *,
    learning_rate: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    decay_params: list[torch.nn.Parameter] = []
    no_decay_params: list[torch.nn.Parameter] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        lowered_name = name.lower()
        is_boundary_affine = (
            lowered_name.startswith("encoder.source_readin.") or lowered_name.startswith("source_readout.")
        )
        # Standard no-decay treatment (bias/norm/1D params) plus boundary affines.
        if (
            parameter.ndim <= 1
            or name.endswith(".bias")
            or "norm" in lowered_name
            or is_boundary_affine
        ):
            no_decay_params.append(parameter)
        else:
            decay_params.append(parameter)

    param_groups: list[dict[str, Any]] = []
    if decay_params:
        param_groups.append(
            {
                "params": decay_params,
                "weight_decay": float(weight_decay),
            }
        )
    if no_decay_params:
        param_groups.append(
            {
                "params": no_decay_params,
                "weight_decay": 0.0,
            }
        )
    return torch.optim.AdamW(param_groups, lr=float(learning_rate))


def _source_session_keys_from_cache_context(cache_context: CacheContext) -> tuple[str, ...]:
    session_keys = {
        resolve_boundary_key(
            dataset=dataset,
            session_id=row.session_id,
            subject_id=row.subject_id,
            boundary_key_mode=cache_context.boundary_key_mode,
        )
        for dataset, rows in cache_context.rows_by_dataset.items()
        for row in rows
    }
    return tuple(sorted(session_keys))


def evaluate_model(
    model: MaskedSSLModel,
    sampler: Any,
    *,
    num_batches: int,
    device: torch.device,
    mask_unit: str,
    mask_token_placement: str,
    mask_ratio: float,
    span_length_min: int,
    span_length_max: int,
    num_spans_mode: str,
    allow_bin_fractional_overlap: bool,
) -> dict[str, Any] | None:
    if sampler is None:
        return None

    was_training = model.training
    model.eval()

    scalar_history: dict[str, list[float]] = defaultdict(list)
    with torch.no_grad():
        for _ in range(int(num_batches)):
            batch = sampler.sample_batch()
            metrics = compute_objective_metrics(
                model,
                batch,
                mask_unit=mask_unit,
                mask_token_placement=mask_token_placement,
                mask_ratio=mask_ratio,
                span_length_min=span_length_min,
                span_length_max=span_length_max,
                num_spans_mode=num_spans_mode,
                allow_bin_fractional_overlap=allow_bin_fractional_overlap,
                device=device,
            )
            summary = summarize_metrics(metrics)
            for key, value in summary.items():
                scalar_history[key].append(float(value))

    if was_training:
        model.train()

    return {
        key: float(np.mean(values))
        for key, values in scalar_history.items()
    }


def _serialize_ssl_training_config(
    config: SSLTrainingConfig,
    *,
    cache_context: CacheContext,
    output_root: Path,
) -> dict[str, Any]:
    return {
        "seed": int(config.seed),
        "objective_mode": str(config.objective_mode),
        "feature_mode": str(config.feature_mode),
        "boundary_key_mode": str(config.boundary_key_mode),
        "input_dim": int(cache_context.full_dim),
        "source_session_keys": list(_source_session_keys_from_cache_context(cache_context)),
        "segment_bins": int(config.segment_bins),
        "patch_size": int(config.patch_size),
        "patch_stride": int(config.patch_stride),
        "hidden_size": int(config.hidden_size),
        "s5_state_size": int(config.s5_state_size),
        "num_layers": int(config.num_layers),
        "dropout": float(config.dropout),
        "batch_size": int(config.batch_size),
        "num_steps": int(config.num_steps),
        "learning_rate": float(config.learning_rate),
        "weight_decay": float(config.weight_decay),
        "val_every": int(config.val_every),
        "val_batches": int(config.val_batches),
        "checkpoint_every_steps": (
            None if config.checkpoint_every_steps is None else int(config.checkpoint_every_steps)
        ),
        "dataset_weight_alpha": float(config.dataset_weight_alpha),
        "examples_per_shard": int(config.examples_per_shard),
        "log_every": int(config.log_every),
        "post_proj_norm": str(config.post_proj_norm),
        "reconstruction_head_mode": str(config.reconstruction_head_mode),
        "reconstruction_head_type": str(config.reconstruction_head_type),
        "backbone_direction": str(config.backbone_direction),
        "mask_unit": str(config.mask_unit),
        "mask_token_placement": str(config.mask_token_placement),
        "mask_ratio": float(config.mask_ratio),
        "span_length_min": int(config.span_length_min),
        "span_length_max": int(config.span_length_max),
        "num_spans_mode": str(config.num_spans_mode),
        "reconstruct_target": str(config.reconstruct_target),
        "loss_mode": str(config.loss_mode),
        "allow_bin_fractional_overlap": bool(config.allow_bin_fractional_overlap),
        "normalize_impl_version": cache_context.normalize_impl_version,
        "normalize_context_bins": int(cache_context.normalize_context_bins),
        "has_val_datasets": bool(cache_context.has_val_datasets),
        "session_split_summary": cache_context.session_split_summary,
        "cache_root": str(cache_context.cache_root),
        "output_root": str(output_root),
    }


def _build_checkpoint_payload(
    *,
    model: MaskedSSLModel,
    optimizer: torch.optim.Optimizer | None,
    config_payload: dict[str, Any],
    step: int,
    best_score: float,
    best_step: int | None,
    checkpoint_kind: str,
    train_history: list[dict[str, Any]] | None = None,
    val_history: list[dict[str, Any]] | None = None,
    dataset_counter: Counter[str] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model_state": model.state_dict(),
        "config": config_payload,
        "step": int(step),
        "best_score": float(best_score),
        "best_step": None if best_step is None else int(best_step),
        "checkpoint_kind": str(checkpoint_kind),
    }
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
    if train_history is not None:
        payload["train_history"] = train_history
    if val_history is not None:
        payload["val_history"] = val_history
    if dataset_counter is not None:
        payload["dataset_counts"] = dict(dataset_counter)
    return payload


def _checkpoint_timestamp_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _step_checkpoint_filename(step: int, *, timestamp_utc: str | None = None) -> str:
    ts = timestamp_utc or _checkpoint_timestamp_utc()
    return f"step_{int(step):06d}_{ts}.pt"


def _final_checkpoint_filename(step: int, *, timestamp_utc: str | None = None) -> str:
    ts = timestamp_utc or _checkpoint_timestamp_utc()
    return f"checkpoint_final_step_{int(step):06d}_{ts}.pt"


def _parse_step_from_checkpoint_name(name: str) -> int | None:
    stem = Path(name).stem
    if stem.startswith("step_"):
        parts = stem.split("_")
        if len(parts) >= 2:
            try:
                return int(parts[1])
            except ValueError:
                return None
    if stem.startswith("checkpoint_final_step_"):
        parts = stem.split("_")
        if len(parts) >= 4:
            try:
                return int(parts[3])
            except ValueError:
                return None
    return None


def list_ssl_checkpoints(run_dir: str | Path) -> list[dict[str, Any]]:
    run_dir = Path(run_dir)
    checkpoint_paths: list[Path] = []
    step_dir = run_dir / "checkpoints"
    if step_dir.exists():
        checkpoint_paths.extend(sorted(step_dir.glob("step_*.pt")))
        checkpoint_paths.extend(sorted(step_dir.glob("checkpoint_final_step_*.pt")))
    final_path = run_dir / "checkpoint_final.pt"
    if final_path.exists():
        checkpoint_paths.append(final_path)
    best_path = run_dir / "checkpoint_best.pt"
    if best_path.exists():
        checkpoint_paths.append(best_path)

    rows: list[dict[str, Any]] = []
    for path in checkpoint_paths:
        row: dict[str, Any] = {
            "path": str(path),
            "name": path.name,
            "kind": (
                "best"
                if path.name == "checkpoint_best.pt"
                else
                "final"
                if path.name == "checkpoint_final.pt" or path.name.startswith("checkpoint_final_step_")
                else "step"
            ),
            "mtime_seconds": float(path.stat().st_mtime),
        }
        parsed_step = _parse_step_from_checkpoint_name(path.name)
        if parsed_step is not None:
            row["step"] = parsed_step
        try:
            payload = torch.load(path, map_location="cpu")
            row["step"] = int(payload.get("step", row.get("step", -1)))
            if payload.get("best_score") is not None:
                row["best_score"] = float(payload["best_score"])
            if payload.get("best_step") is not None:
                row["best_step"] = int(payload["best_step"])
            if payload.get("checkpoint_kind") is not None:
                row["checkpoint_kind"] = str(payload["checkpoint_kind"])
        except Exception as exc:  # pragma: no cover - diagnostic helper
            row["load_error"] = str(exc)
        rows.append(row)

    return sorted(rows, key=lambda row: (int(row.get("step", -1)), row["kind"] != "final"))


def _ssl_run_dir_from_checkpoint_path(path: str | Path) -> Path:
    checkpoint_path = Path(path)
    if checkpoint_path.parent.name == "checkpoints":
        return checkpoint_path.parent.parent
    return checkpoint_path.parent


def resolve_ssl_checkpoint_path(
    *,
    output_root: str | Path,
    explicit_checkpoint_path: str | Path | None = None,
    run_dir: str | Path | None = None,
) -> Path:
    if explicit_checkpoint_path is not None:
        candidate = Path(explicit_checkpoint_path)
        if not candidate.exists():
            raise FileNotFoundError(f"Explicit SSL checkpoint path does not exist: {candidate}")
        return candidate

    resolved_run_dir: Path | None = None
    if run_dir is not None:
        resolved_run_dir = Path(run_dir)
        if not resolved_run_dir.exists():
            raise FileNotFoundError(f"Requested SSL run directory does not exist: {resolved_run_dir}")
    else:
        run_candidates = sorted(
            [path for path in Path(output_root).glob("colab_s5_*") if path.is_dir()],
            key=lambda path: path.stat().st_mtime,
        )
        if run_candidates:
            resolved_run_dir = run_candidates[-1]

    if resolved_run_dir is None:
        raise RuntimeError(
            f"Could not find any SSL run directories under {output_root}. "
            "Provide an explicit checkpoint path or a run directory."
        )

    available_checkpoints = [
        row for row in list_ssl_checkpoints(resolved_run_dir) if not row.get("load_error")
    ]
    if not available_checkpoints:
        raise RuntimeError(
            f"No readable SSL checkpoints were found in {resolved_run_dir}. "
            "If training stopped before the first save, there may be nothing to recover yet."
        )

    selected = sorted(
        available_checkpoints,
        key=lambda row: (int(row.get("step", -1)), row.get("kind") == "final"),
    )[-1]
    return Path(selected["path"])


def recover_ssl_run_state_from_checkpoint(
    *,
    checkpoint_path: str | Path,
    cache_context: CacheContext,
    device: torch.device,
    fallback_config: SSLTrainingConfig | None = None,
) -> dict[str, Any]:
    resolved_checkpoint_path = Path(checkpoint_path)
    if not resolved_checkpoint_path.exists():
        raise FileNotFoundError(f"SSL checkpoint does not exist: {resolved_checkpoint_path}")

    payload = torch.load(resolved_checkpoint_path, map_location="cpu")
    run_dir = _ssl_run_dir_from_checkpoint_path(resolved_checkpoint_path)

    recovered_config = dict(payload.get("config", {}))
    config_path = run_dir / "config.json"
    if not recovered_config and config_path.exists():
        recovered_config = json.loads(config_path.read_text())

    had_reconstruction_head_mode = "reconstruction_head_mode" in recovered_config
    had_reconstruction_head_type = "reconstruction_head_type" in recovered_config
    had_backbone_direction = "backbone_direction" in recovered_config
    fallback_payload = asdict(fallback_config) if fallback_config is not None else {}
    for key, value in fallback_payload.items():
        recovered_config.setdefault(key, value)
    recovered_config.setdefault("boundary_key_mode", "session")
    if not had_reconstruction_head_mode:
        recovered_config["reconstruction_head_mode"] = "with_output_norm"
    if not had_reconstruction_head_type:
        recovered_config["reconstruction_head_type"] = "linear"
    if not had_backbone_direction:
        recovered_config["backbone_direction"] = "causal"

    required_keys = [
        "feature_mode",
        "segment_bins",
        "input_dim",
        "patch_size",
        "patch_stride",
        "hidden_size",
        "s5_state_size",
        "num_layers",
        "dropout",
        "batch_size",
        "seed",
        "dataset_weight_alpha",
        "examples_per_shard",
        "learning_rate",
        "weight_decay",
        "mask_unit",
        "mask_token_placement",
        "mask_ratio",
        "span_length_min",
        "span_length_max",
        "num_spans_mode",
        "allow_bin_fractional_overlap",
    ]
    missing_keys = [key for key in required_keys if key not in recovered_config]
    if missing_keys:
        raise KeyError(
            f"Recovered SSL config is missing keys needed for analysis/probe recovery: {missing_keys}"
        )
    if str(recovered_config["feature_mode"]) != str(cache_context.feature_mode):
        raise ValueError(
            "Recovered checkpoint feature_mode does not match the active cache context. "
            f"checkpoint={recovered_config['feature_mode']!r} cache={cache_context.feature_mode!r}"
        )
    cache_boundary_key_mode = str(getattr(cache_context, "boundary_key_mode", "session"))
    if str(recovered_config.get("boundary_key_mode", "session")) != cache_boundary_key_mode:
        raise ValueError(
            "Recovered checkpoint boundary_key_mode does not match the active cache context. "
            f"checkpoint={recovered_config.get('boundary_key_mode', 'session')!r} "
            f"cache={cache_boundary_key_mode!r}"
        )
    if int(recovered_config["input_dim"]) != int(cache_context.full_dim):
        raise ValueError(
            "Recovered checkpoint input_dim does not match the active cache context. "
            f"checkpoint={int(recovered_config['input_dim'])} cache={int(cache_context.full_dim)}"
        )

    model = MaskedSSLModel(
        input_dim=int(recovered_config.get("input_dim", cache_context.full_dim)),
        hidden_size=int(recovered_config["hidden_size"]),
        s5_state_size=int(recovered_config["s5_state_size"]),
        num_layers=int(recovered_config["num_layers"]),
        dropout=float(recovered_config["dropout"]),
        patch_size=int(recovered_config["patch_size"]),
        patch_stride=int(recovered_config["patch_stride"]),
        post_proj_norm=str(recovered_config.get("post_proj_norm", "rms")),
        source_session_keys=tuple(recovered_config.get("source_session_keys", ())),
        feature_mode=str(recovered_config.get("feature_mode", "tx_only")),
        reconstruction_head_mode=str(
            recovered_config.get("reconstruction_head_mode", "with_output_norm")
        ),
        reconstruction_head_type=str(recovered_config.get("reconstruction_head_type", "linear")),
        backbone_direction=str(recovered_config.get("backbone_direction", "causal")),
    ).to(device)
    model_state = payload.get("model_state")
    if model_state is None:
        raise KeyError("Recovered SSL checkpoint is missing 'model_state'.")
    model.load_state_dict(model_state)
    model.eval()

    optimizer = _build_ssl_optimizer(
        model,
        learning_rate=float(recovered_config["learning_rate"]),
        weight_decay=float(recovered_config["weight_decay"]),
    )
    if payload.get("optimizer_state") is not None:
        try:
            optimizer.load_state_dict(payload["optimizer_state"])
        except ValueError:
            # Legacy checkpoints may have a single AdamW group; keep the recovered
            # model weights and continue with a fresh optimizer grouping.
            pass

    train_sampler = build_segment_sampler(
        cache_context,
        "train",
        batch_size=int(recovered_config["batch_size"]),
        seed=int(recovered_config["seed"]),
        segment_bins=int(recovered_config["segment_bins"]),
        dataset_weight_alpha=float(recovered_config["dataset_weight_alpha"]),
        examples_per_shard=int(recovered_config["examples_per_shard"]),
    )
    try:
        val_sampler = build_segment_sampler(
            cache_context,
            "val",
            batch_size=int(recovered_config["batch_size"]),
            seed=int(recovered_config["seed"]) + 101,
            segment_bins=int(recovered_config["segment_bins"]),
            dataset_weight_alpha=float(recovered_config["dataset_weight_alpha"]),
            examples_per_shard=int(recovered_config["examples_per_shard"]),
        )
    except RuntimeError:
        val_sampler = None

    progress_path = run_dir / "progress.jsonl"
    checkpoint_step = int(payload.get("step", -1))
    train_history = list(payload.get("train_history", []))
    val_history = list(payload.get("val_history", []))
    if progress_path.exists() and (not train_history or not val_history):
        records = [json.loads(line) for line in progress_path.read_text().splitlines() if line.strip()]
        train_history = [
            record
            for record in records
            if record.get("event") == "train" and int(record.get("step", -1)) <= checkpoint_step
        ]
        val_history = [
            record
            for record in records
            if record.get("event") == "val" and int(record.get("step", -1)) <= checkpoint_step
        ]

    metric_plot_path = run_dir / "masked_metric_curve.png"
    return {
        "model": model,
        "optimizer": optimizer,
        "train_sampler": train_sampler,
        "val_sampler": val_sampler,
        "run_name": run_dir.name,
        "run_dir": run_dir,
        "progress_path": progress_path,
        "checkpoint_path": resolved_checkpoint_path,
        "best_checkpoint_path": run_dir / "checkpoint_best.pt",
        "checkpoints_dir": run_dir / "checkpoints",
        "plot_loss_path": run_dir / "loss_curve.png",
        "plot_metric_path": metric_plot_path,
        "plot_top1_path": metric_plot_path,
        "config": recovered_config,
        "best_score": payload.get("best_score"),
        "best_step": payload.get("best_step"),
        "train_history": train_history,
        "val_history": val_history,
        "dataset_counts": dict(payload.get("dataset_counts", {})),
        "checkpoint_step": checkpoint_step,
        "checkpoint_kind": payload.get("checkpoint_kind"),
    }


def run_ssl_training(
    *,
    cache_context: CacheContext,
    config: SSLTrainingConfig,
    output_root: Path,
    device: torch.device,
) -> dict[str, Any]:
    _seed_training_run(int(config.seed))
    if str(config.feature_mode) != str(cache_context.feature_mode):
        raise ValueError(
            "SSLTrainingConfig.feature_mode must match CacheAccessConfig.feature_mode. "
            f"config={config.feature_mode!r} cache={cache_context.feature_mode!r}"
        )
    if str(config.boundary_key_mode) != str(cache_context.boundary_key_mode):
        raise ValueError(
            "SSLTrainingConfig.boundary_key_mode must match CacheAccessConfig.boundary_key_mode. "
            f"config={config.boundary_key_mode!r} cache={cache_context.boundary_key_mode!r}"
        )

    train_sampler = build_segment_sampler(
        cache_context,
        "train",
        batch_size=int(config.batch_size),
        seed=int(config.seed),
        segment_bins=int(config.segment_bins),
        dataset_weight_alpha=float(config.dataset_weight_alpha),
        examples_per_shard=int(config.examples_per_shard),
    )
    try:
        val_sampler = build_segment_sampler(
            cache_context,
            "val",
            batch_size=int(config.batch_size),
            seed=int(config.seed) + 101,
            segment_bins=int(config.segment_bins),
            dataset_weight_alpha=float(config.dataset_weight_alpha),
            examples_per_shard=int(config.examples_per_shard),
        )
    except RuntimeError:
        val_sampler = None

    model = MaskedSSLModel(
        input_dim=cache_context.full_dim,
        hidden_size=int(config.hidden_size),
        s5_state_size=int(config.s5_state_size),
        num_layers=int(config.num_layers),
        dropout=float(config.dropout),
        patch_size=int(config.patch_size),
        patch_stride=int(config.patch_stride),
        post_proj_norm=str(config.post_proj_norm),
        source_session_keys=_source_session_keys_from_cache_context(cache_context),
        feature_mode=str(config.feature_mode),
        reconstruction_head_mode=str(config.reconstruction_head_mode),
        reconstruction_head_type=str(config.reconstruction_head_type),
        backbone_direction=str(config.backbone_direction),
    ).to(device)
    optimizer = _build_ssl_optimizer(
        model,
        learning_rate=float(config.learning_rate),
        weight_decay=float(config.weight_decay),
    )

    run_name = (
        f"colab_s5_{config.objective_mode}_seg{int(config.segment_bins)}_"
        f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    )
    run_dir = Path(output_root) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    progress_path = run_dir / "progress.jsonl"
    checkpoint_path = run_dir / "checkpoint_final.pt"
    best_checkpoint_path = run_dir / "checkpoint_best.pt"
    checkpoints_dir = run_dir / "checkpoints"
    plot_loss_path = run_dir / "loss_curve.png"
    plot_metric_path = run_dir / "masked_metric_curve.png"
    if config.checkpoint_every_steps is not None:
        checkpoints_dir.mkdir(parents=True, exist_ok=True)

    config_payload = _serialize_ssl_training_config(
        config,
        cache_context=cache_context,
        output_root=Path(output_root),
    )
    (run_dir / "config.json").write_text(json.dumps(config_payload, indent=2))

    best_score = float("inf")
    best_step: int | None = None
    best_state = None
    train_history: list[dict[str, Any]] = []
    val_history: list[dict[str, Any]] = []
    dataset_counter = Counter()
    start_time = time.time()

    for step in range(1, int(config.num_steps) + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        sample_start = time.time()
        batch = train_sampler.sample_batch()
        sample_seconds = time.time() - sample_start
        dataset_mix = dict(Counter(batch["datasets"]))
        dataset_counter.update(batch["datasets"])

        sync_device(device)
        model_start = time.time()
        metrics = compute_objective_metrics(
            model,
            batch,
            mask_unit=str(config.mask_unit),
            mask_token_placement=str(config.mask_token_placement),
            mask_ratio=float(config.mask_ratio),
            span_length_min=int(config.span_length_min),
            span_length_max=int(config.span_length_max),
            num_spans_mode=str(config.num_spans_mode),
            allow_bin_fractional_overlap=bool(config.allow_bin_fractional_overlap),
            device=device,
        )
        loss = metrics["loss"]
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        sync_device(device)
        model_seconds = time.time() - model_start

        summary = summarize_metrics(metrics)
        shard_summary = cache_context.shard_store.summary()
        train_record = {
            "event": "train",
            "step": step,
            "elapsed_seconds": float(time.time() - start_time),
            "sample_seconds": float(sample_seconds),
            "model_seconds": float(model_seconds),
            "grad_norm": float(grad_norm),
            "cached_shards": int(shard_summary["cached_shards"]),
            "cached_gb": float(shard_summary["cached_gb"]),
            "dataset_mix": dataset_mix,
            **summary,
        }
        train_history.append(train_record)
        with progress_path.open("a") as handle:
            handle.write(json.dumps(train_record) + "\n")

        if val_sampler is None and summary["loss"] < best_score:
            best_score = summary["loss"]
            best_step = step
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }

        if step == 1 or step % int(config.log_every) == 0:
            print(
                f"step={step:03d} train_loss={summary['loss']:.4f} "
                f"mask_ratio={summary['actual_mask_ratio']:.3f} "
                f"masked_token_ratio={summary['masked_token_ratio']:.3f} "
                f"grad_norm={float(grad_norm):.4f} sample_s={sample_seconds:.2f} model_s={model_seconds:.2f}"
            )

        if val_sampler is not None and (step % int(config.val_every) == 0 or step == int(config.num_steps)):
            val_result = evaluate_model(
                model,
                val_sampler,
                num_batches=int(config.val_batches),
                device=device,
                mask_unit=str(config.mask_unit),
                mask_token_placement=str(config.mask_token_placement),
                mask_ratio=float(config.mask_ratio),
                span_length_min=int(config.span_length_min),
                span_length_max=int(config.span_length_max),
                num_spans_mode=str(config.num_spans_mode),
                allow_bin_fractional_overlap=bool(config.allow_bin_fractional_overlap),
            )
            assert val_result is not None
            val_record = {
                "event": "val",
                "step": step,
                "elapsed_seconds": float(time.time() - start_time),
                **val_result,
            }
            val_history.append(val_record)
            with progress_path.open("a") as handle:
                handle.write(json.dumps(val_record) + "\n")
            print(
                f"step={step:03d} val_loss={val_result['loss']:.4f} "
                f"val_mask_ratio={val_result['actual_mask_ratio']:.3f} "
                f"val_masked_token_ratio={val_result['masked_token_ratio']:.3f}"
            )
            if val_result["loss"] < best_score:
                best_score = val_result["loss"]
                best_step = step
                best_state = {
                    key: value.detach().cpu().clone() for key, value in model.state_dict().items()
                }

        if config.checkpoint_every_steps is not None and step % int(config.checkpoint_every_steps) == 0:
            step_checkpoint_path = checkpoints_dir / _step_checkpoint_filename(step)
            torch.save(
                _build_checkpoint_payload(
                    model=model,
                    optimizer=optimizer,
                    config_payload=config_payload,
                    step=step,
                    best_score=best_score,
                    best_step=best_step,
                    checkpoint_kind="step",
                    dataset_counter=dataset_counter,
                ),
                step_checkpoint_path,
            )
            print("saved_step_checkpoint:", step_checkpoint_path)

    if best_state is None:
        best_state = {
            key: value.detach().cpu().clone() for key, value in model.state_dict().items()
        }

    final_payload = _build_checkpoint_payload(
        model=model,
        optimizer=optimizer,
        config_payload=config_payload,
        step=int(config.num_steps),
        best_score=best_score,
        best_step=best_step,
        checkpoint_kind="final",
        train_history=train_history,
        val_history=val_history,
        dataset_counter=dataset_counter,
    )
    torch.save(final_payload, checkpoint_path)
    timestamped_final_checkpoint_path = checkpoints_dir / _final_checkpoint_filename(int(config.num_steps))
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    torch.save(final_payload, timestamped_final_checkpoint_path)

    best_payload = dict(final_payload)
    best_payload["model_state"] = best_state
    best_payload["optimizer_state"] = None
    best_payload["step"] = int(best_step if best_step is not None else config.num_steps)
    best_payload["checkpoint_kind"] = "best"
    torch.save(best_payload, best_checkpoint_path)

    print("run_dir:", run_dir)
    print("progress_path:", progress_path)
    print("checkpoint_path:", checkpoint_path)
    print("best_checkpoint_path:", best_checkpoint_path)
    if config.checkpoint_every_steps is not None:
        print("checkpoints_dir:", checkpoints_dir)
    print("timestamped_final_checkpoint_path:", timestamped_final_checkpoint_path)
    print("best_score:", best_score)
    print("best_step:", best_step)
    print("dataset_counts:", dict(dataset_counter))

    return {
        "model": model,
        "optimizer": optimizer,
        "train_sampler": train_sampler,
        "val_sampler": val_sampler,
        "run_name": run_name,
        "run_dir": run_dir,
        "progress_path": progress_path,
        "checkpoint_path": checkpoint_path,
        "best_checkpoint_path": best_checkpoint_path,
        "checkpoints_dir": checkpoints_dir,
        "plot_loss_path": plot_loss_path,
        "plot_metric_path": plot_metric_path,
        "plot_top1_path": plot_metric_path,
        "config": config_payload,
        "best_score": best_score,
        "best_step": best_step,
        "train_history": train_history,
        "val_history": val_history,
        "dataset_counts": dict(dataset_counter),
    }


def resume_ssl_training(
    *,
    run_state: dict[str, Any],
    additional_steps: int,
    cache_context: CacheContext,
    device: torch.device,
) -> dict[str, Any]:
    """Continue an in-memory SSL run and update final/best checkpoints.

    The final checkpoint keeps the final-step model and optimizer state so it is
    suitable for continued training. Best weights are written separately to
    ``checkpoint_best.pt`` when this resume pass finds a new best score.
    """
    model = run_state["model"]
    optimizer = run_state["optimizer"]
    train_sampler = run_state["train_sampler"]
    val_sampler = run_state["val_sampler"]

    run_dir = Path(run_state["run_dir"])
    progress_path = Path(run_state["progress_path"])
    checkpoint_path = run_dir / "checkpoint_final.pt"
    best_checkpoint_path = Path(run_state.get("best_checkpoint_path", run_dir / "checkpoint_best.pt"))
    checkpoints_dir = Path(run_state["checkpoints_dir"])
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    config_payload = dict(run_state["config"])
    train_history = list(run_state.get("train_history", []))
    val_history = list(run_state.get("val_history", []))

    actionable_steps = [
        int(record["step"]) for record in train_history if int(record.get("step", 0)) > 0
    ]
    start_step = max(actionable_steps) if actionable_steps else int(run_state.get("checkpoint_step", 0) or 0)

    additional_steps = int(additional_steps)
    if additional_steps <= 0:
        raise ValueError(f"additional_steps must be positive, got {additional_steps}")

    target_step = start_step + additional_steps
    config_payload["num_steps"] = int(target_step)

    dataset_counter = Counter(run_state.get("dataset_counts", {}))
    best_score = float(run_state["best_score"]) if run_state.get("best_score") is not None else float("inf")
    best_step = int(run_state["best_step"]) if run_state.get("best_step") is not None else None
    best_state = None

    log_every = int(config_payload.get("log_every", 10))
    val_every = int(config_payload.get("val_every", 50))
    val_batches = int(config_payload.get("val_batches", 10))

    print("Continuing in-memory masked SSL training")
    print(" - run_dir:", run_dir)
    print(" - start_step:", start_step)
    print(" - target_step:", target_step)
    print(" - mask_ratio:", float(config_payload["mask_ratio"]))

    resume_start_time = time.time()

    for step in range(start_step + 1, target_step + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        sample_start = time.time()
        batch = train_sampler.sample_batch()
        sample_seconds = time.time() - sample_start
        dataset_mix = dict(Counter(batch["datasets"]))
        dataset_counter.update(batch["datasets"])

        sync_device(device)
        model_start = time.time()
        metrics = compute_objective_metrics(
            model,
            batch,
            mask_unit=str(config_payload["mask_unit"]),
            mask_token_placement=str(config_payload["mask_token_placement"]),
            mask_ratio=float(config_payload["mask_ratio"]),
            span_length_min=int(config_payload["span_length_min"]),
            span_length_max=int(config_payload["span_length_max"]),
            num_spans_mode=str(config_payload["num_spans_mode"]),
            allow_bin_fractional_overlap=bool(config_payload["allow_bin_fractional_overlap"]),
            device=device,
        )
        loss = metrics["loss"]
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        sync_device(device)
        model_seconds = time.time() - model_start

        summary = summarize_metrics(metrics)
        shard_summary = cache_context.shard_store.summary()
        train_record = {
            "event": "train",
            "step": step,
            "elapsed_seconds": float(time.time() - resume_start_time),
            "sample_seconds": float(sample_seconds),
            "model_seconds": float(model_seconds),
            "grad_norm": float(grad_norm),
            "cached_shards": int(shard_summary["cached_shards"]),
            "cached_gb": float(shard_summary["cached_gb"]),
            "dataset_mix": dataset_mix,
            **summary,
        }
        train_history.append(train_record)
        with progress_path.open("a") as handle:
            handle.write(json.dumps(train_record) + "\n")

        if val_sampler is None and summary["loss"] < best_score:
            best_score = float(summary["loss"])
            best_step = int(step)
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

        if step == start_step + 1 or step % log_every == 0:
            print(
                f"step={step:04d} train_loss={summary['loss']:.4f} "
                f"mask_ratio={summary['actual_mask_ratio']:.3f} "
                f"masked_token_ratio={summary['masked_token_ratio']:.3f} "
                f"grad_norm={float(grad_norm):.4f} sample_s={sample_seconds:.2f} model_s={model_seconds:.2f}"
            )

        if val_sampler is not None and (step % val_every == 0 or step == target_step):
            val_summary = evaluate_model(
                model,
                val_sampler,
                num_batches=val_batches,
                device=device,
                mask_unit=str(config_payload["mask_unit"]),
                mask_token_placement=str(config_payload["mask_token_placement"]),
                mask_ratio=float(config_payload["mask_ratio"]),
                span_length_min=int(config_payload["span_length_min"]),
                span_length_max=int(config_payload["span_length_max"]),
                num_spans_mode=str(config_payload["num_spans_mode"]),
                allow_bin_fractional_overlap=bool(config_payload["allow_bin_fractional_overlap"]),
            )
            assert val_summary is not None
            val_record = {
                "event": "val",
                "step": step,
                "elapsed_seconds": float(time.time() - resume_start_time),
                **val_summary,
            }
            val_history.append(val_record)
            with progress_path.open("a") as handle:
                handle.write(json.dumps(val_record) + "\n")

            print(
                f"step={step:04d} val_loss={val_summary['loss']:.4f} "
                f"val_mask_ratio={val_summary['actual_mask_ratio']:.3f} "
                f"val_masked_token_ratio={val_summary['masked_token_ratio']:.3f}"
            )

            if val_summary["loss"] < best_score:
                best_score = float(val_summary["loss"])
                best_step = int(step)
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

        checkpoint_every_steps = config_payload.get("checkpoint_every_steps")
        if checkpoint_every_steps is not None and step % int(checkpoint_every_steps) == 0:
            checkpoint_payload = _build_checkpoint_payload(
                model=model,
                optimizer=optimizer,
                config_payload=config_payload,
                step=step,
                best_score=best_score,
                best_step=best_step,
                checkpoint_kind="step",
                train_history=train_history,
                val_history=val_history,
                dataset_counter=dataset_counter,
            )
            step_checkpoint = checkpoints_dir / _step_checkpoint_filename(step)
            torch.save(checkpoint_payload, step_checkpoint)
            print("saved_step_checkpoint:", step_checkpoint)

    final_payload = _build_checkpoint_payload(
        model=model,
        optimizer=optimizer,
        config_payload=config_payload,
        step=target_step,
        best_score=best_score,
        best_step=best_step,
        checkpoint_kind="final",
        train_history=train_history,
        val_history=val_history,
        dataset_counter=dataset_counter,
    )
    final_checkpoint = run_dir / _final_checkpoint_filename(target_step)
    torch.save(final_payload, checkpoint_path)
    torch.save(final_payload, final_checkpoint)

    if best_state is not None:
        best_payload = dict(final_payload)
        best_payload["model_state"] = best_state
        best_payload["optimizer_state"] = None
        best_payload["step"] = int(best_step if best_step is not None else target_step)
        best_payload["checkpoint_kind"] = "best"
        torch.save(best_payload, best_checkpoint_path)
    elif best_checkpoint_path.exists():
        print("No new best checkpoint this resume; keeping existing best checkpoint:", best_checkpoint_path)
    else:
        print("No best checkpoint was written during this resume.")

    run_state.update(
        {
            "model": model,
            "optimizer": optimizer,
            "train_sampler": train_sampler,
            "val_sampler": val_sampler,
            "best_score": best_score,
            "best_step": best_step,
            "train_history": train_history,
            "val_history": val_history,
            "dataset_counts": dict(dataset_counter),
            "checkpoint_step": target_step,
            "checkpoint_path": checkpoint_path,
            "best_checkpoint_path": best_checkpoint_path,
        }
    )

    print("Resume complete")
    print(" - final_checkpoint:", final_checkpoint)
    print(" - checkpoint_path:", checkpoint_path)
    print(" - best_checkpoint:", best_checkpoint_path)
    print(" - best_score:", best_score, "best_step:", best_step)
    return run_state


def plot_ssl_training_history(run_state: dict[str, Any]) -> dict[str, Any]:
    import matplotlib.pyplot as plt

    progress_path = Path(run_state["progress_path"])
    plot_loss_path = Path(run_state["plot_loss_path"])
    plot_metric_path = Path(run_state.get("plot_metric_path", run_state.get("plot_top1_path")))
    objective_mode = str(run_state["config"]["objective_mode"])

    train_records = list(run_state.get("train_history", []))
    val_records = list(run_state.get("val_history", []))
    if train_records or val_records:
        records = sorted(
            [*train_records, *val_records],
            key=lambda record: (int(record.get("step", -1)), str(record.get("event", ""))),
        )
    else:
        records = [json.loads(line) for line in progress_path.read_text().splitlines() if line.strip()]
        train_records = [record for record in records if record.get("event") == "train"]
        val_records = [record for record in records if record.get("event") == "val"]

    train_steps = np.array([record["step"] for record in train_records], dtype=np.int64)
    train_losses = np.array([record["loss"] for record in train_records], dtype=np.float32)

    plt.figure(figsize=(8, 4))
    plt.plot(train_steps, train_losses, label="train loss", alpha=0.8)
    if val_records:
        val_steps = np.array([record["step"] for record in val_records], dtype=np.int64)
        val_losses = np.array([record["loss"] for record in val_records], dtype=np.float32)
        plt.plot(val_steps, val_losses, marker="o", label="val loss")
    plt.xlabel("step")
    plt.ylabel("masked reconstruction loss")
    plt.title(f"{objective_mode} loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_loss_path, dpi=160)
    plt.show()

    aux_key = (
        "patch_fraction_weighted_mse"
        if any("patch_fraction_weighted_mse" in record for record in train_records + val_records)
        else "masked_token_full_patch_mse"
    )
    train_aux = np.array([record[aux_key] for record in train_records], dtype=np.float32)
    plt.figure(figsize=(8, 4))
    plt.plot(train_steps, train_aux, label=f"train {aux_key}", alpha=0.8)
    if val_records:
        val_aux = np.array([record[aux_key] for record in val_records], dtype=np.float32)
        plt.plot(val_steps, val_aux, marker="o", label=f"val {aux_key}")
    plt.xlabel("step")
    plt.ylabel(aux_key)
    plt.title(f"{objective_mode} diagnostics")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_metric_path, dpi=160)
    plt.show()

    print("train_start_loss:", float(train_losses[0]))
    print("train_final_loss:", float(train_losses[-1]))
    print("train_best_loss:", float(train_losses.min()), "at step", int(train_steps[train_losses.argmin()]))
    if val_records:
        print("val_best_loss:", float(val_losses.min()), "at step", int(val_steps[val_losses.argmin()]))
    latest_train = train_records[-1] if train_records else None
    latest_val = val_records[-1] if val_records else None
    if latest_train is not None:
        print(
            "latest_train_masked_stats:",
            {
                "prediction_mean": float(latest_train["masked_prediction_mean"]),
                "prediction_std": float(latest_train["masked_prediction_std"]),
                "target_mean": float(latest_train["masked_target_mean"]),
                "target_std": float(latest_train["masked_target_std"]),
            },
        )
    if latest_val is not None:
        print(
            "latest_val_masked_stats:",
            {
                "prediction_mean": float(latest_val["masked_prediction_mean"]),
                "prediction_std": float(latest_val["masked_prediction_std"]),
                "target_mean": float(latest_val["masked_target_mean"]),
                "target_std": float(latest_val["masked_target_std"]),
            },
        )
    print("aux_metric_key:", aux_key)
    print("plot_loss_path:", plot_loss_path)
    print("plot_metric_path:", plot_metric_path)

    return {
        "records": records,
        "train_records": train_records,
        "val_records": val_records,
        "plot_loss_path": plot_loss_path,
        "plot_metric_path": plot_metric_path,
    }

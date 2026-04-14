"""Contrastive SSL training helpers for Colab experiments."""

from __future__ import annotations

import json
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .cache import CacheContext, build_segment_sampler
from .model import ContrastiveSSLModel, sync_device
from .objectives import compute_objective_metrics, summarize_metrics


@dataclass
class SSLTrainingConfig:
    seed: int = 7
    objective_mode: str = "future_infonce"
    segment_bins: int = 64
    future_horizons: tuple[int, ...] = (1, 3)
    patch_size: int = 1
    patch_stride: int = 1
    hidden_size: int = 128
    s5_state_size: int = 64
    num_layers: int = 1
    dropout: float = 0.1
    batch_size: int = 32
    num_steps: int = 500
    learning_rate: float = 3e-4
    weight_decay: float = 1e-2
    temperature: float = 0.1
    val_every: int = 50
    val_batches: int = 10
    checkpoint_every_steps: int | None = None
    dataset_weight_alpha: float = 0.25
    examples_per_shard: int = 8
    log_every: int = 10
    post_proj_norm: str = "rms"
    backbone_direction: str = "bidirectional"
    augment_cfg: dict[str, float] = field(
        default_factory=lambda: {
            "noise_std": 0.01,
            "scale_jitter": 0.05,
            "offset_jitter": 0.05,
            "time_mask_frac": 0.10,
            "channel_dropout_prob": 0.05,
            "clip_value": 20.0,
        }
    )

    def __post_init__(self) -> None:
        if self.objective_mode not in {"future_infonce", "augment_infonce"}:
            raise ValueError("objective_mode must be 'future_infonce' or 'augment_infonce'")
        if self.patch_stride > self.patch_size:
            raise ValueError("patch_stride must be <= patch_size")
        if self.checkpoint_every_steps is not None and int(self.checkpoint_every_steps) <= 0:
            raise ValueError("checkpoint_every_steps must be positive when provided")
        if self.backbone_direction not in {"causal", "bidirectional"}:
            raise ValueError("backbone_direction must be one of {'causal', 'bidirectional'}")
        if self.objective_mode == "future_infonce" and self.backbone_direction != "causal":
            raise ValueError(
                "future_infonce requires backbone_direction='causal' to prevent "
                "future-context leakage in anchor representations."
            )

    def checkpoint_config(self) -> dict[str, Any]:
        return {
            "patch_size": int(self.patch_size),
            "patch_stride": int(self.patch_stride),
            "hidden_size": int(self.hidden_size),
            "s5_state_size": int(self.s5_state_size),
            "num_layers": int(self.num_layers),
            "dropout": float(self.dropout),
            "post_proj_norm": str(self.post_proj_norm),
            "backbone_direction": str(self.backbone_direction),
        }


def evaluate_model(
    model: ContrastiveSSLModel,
    sampler: Any,
    *,
    objective_mode: str,
    num_batches: int,
    device: torch.device,
    temperature: float,
    horizons: tuple[int, ...],
    augment_cfg: dict[str, Any],
) -> dict[str, Any] | None:
    if sampler is None:
        return None

    was_training = model.training
    model.eval()

    losses = []
    top1_values = []
    positive_pairs = []
    per_horizon_losses = defaultdict(list)
    per_horizon_top1 = defaultdict(list)
    view_deltas = []

    with torch.no_grad():
        for _ in range(int(num_batches)):
            batch = sampler.sample_batch()
            metrics = compute_objective_metrics(
                model,
                batch,
                objective_mode=objective_mode,
                device=device,
                temperature=temperature,
                horizons=horizons,
                augment_cfg=augment_cfg,
            )
            summary = summarize_metrics(metrics)
            losses.append(summary["loss"])
            top1_values.append(summary["top1"])
            positive_pairs.append(summary["positive_pairs"])
            for horizon, value in summary.get("per_horizon_losses", {}).items():
                per_horizon_losses[horizon].append(float(value))
            for horizon, value in summary.get("per_horizon_top1", {}).items():
                per_horizon_top1[horizon].append(float(value))
            if "mean_abs_view_delta" in summary:
                view_deltas.append(float(summary["mean_abs_view_delta"]))

    if was_training:
        model.train()

    result = {
        "loss": float(np.mean(losses)),
        "top1": float(np.mean(top1_values)),
        "positive_pairs": int(np.sum(positive_pairs)),
    }
    if per_horizon_losses:
        result["per_horizon_losses"] = {
            key: float(np.mean(values)) for key, values in per_horizon_losses.items()
        }
    if per_horizon_top1:
        result["per_horizon_top1"] = {
            key: float(np.mean(values)) for key, values in per_horizon_top1.items()
        }
    if view_deltas:
        result["mean_abs_view_delta"] = float(np.mean(view_deltas))
    return result


def _serialize_ssl_training_config(
    config: SSLTrainingConfig,
    *,
    cache_context: CacheContext,
    output_root: Path,
) -> dict[str, Any]:
    payload = {
        "seed": int(config.seed),
        "objective_mode": str(config.objective_mode),
        "segment_bins": int(config.segment_bins),
        "future_horizons": list(config.future_horizons),
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
        "temperature": float(config.temperature),
        "val_every": int(config.val_every),
        "val_batches": int(config.val_batches),
        "checkpoint_every_steps": (
            None if config.checkpoint_every_steps is None else int(config.checkpoint_every_steps)
        ),
        "dataset_weight_alpha": float(config.dataset_weight_alpha),
        "examples_per_shard": int(config.examples_per_shard),
        "augment_cfg": dict(config.augment_cfg),
        "normalize_impl_version": cache_context.normalize_impl_version,
        "normalize_context_bins": int(cache_context.normalize_context_bins),
        "post_proj_norm": str(config.post_proj_norm),
        "backbone_direction": str(config.backbone_direction),
        "has_val_datasets": bool(cache_context.has_val_datasets),
        "session_split_summary": cache_context.session_split_summary,
        "cache_root": str(cache_context.cache_root),
        "output_root": str(output_root),
    }
    return payload


def _build_checkpoint_payload(
    *,
    model: ContrastiveSSLModel,
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

    rows: list[dict[str, Any]] = []
    for path in checkpoint_paths:
        row: dict[str, Any] = {
            "path": str(path),
            "name": path.name,
            "kind": "final" if path.name == "checkpoint_final.pt" or path.name.startswith("checkpoint_final_step_") else "step",
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

    had_backbone_direction = "backbone_direction" in recovered_config
    fallback_payload = asdict(fallback_config) if fallback_config is not None else {}
    for key, value in fallback_payload.items():
        recovered_config.setdefault(key, value)
    if not had_backbone_direction:
        recovered_config["backbone_direction"] = "causal"

    required_keys = [
        "segment_bins",
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
    ]
    missing_keys = [key for key in required_keys if key not in recovered_config]
    if missing_keys:
        raise KeyError(
            f"Recovered SSL config is missing keys needed for analysis/probe recovery: {missing_keys}"
        )

    model = ContrastiveSSLModel(
        input_dim=cache_context.full_dim,
        hidden_size=int(recovered_config["hidden_size"]),
        s5_state_size=int(recovered_config["s5_state_size"]),
        num_layers=int(recovered_config["num_layers"]),
        dropout=float(recovered_config["dropout"]),
        patch_size=int(recovered_config["patch_size"]),
        patch_stride=int(recovered_config["patch_stride"]),
        post_proj_norm=str(recovered_config.get("post_proj_norm", "rms")),
        backbone_direction=str(recovered_config.get("backbone_direction", "causal")),
    ).to(device)
    model_state = payload.get("model_state")
    if model_state is None:
        raise KeyError("Recovered SSL checkpoint is missing 'model_state'.")
    model.load_state_dict(model_state)
    model.eval()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(recovered_config["learning_rate"]),
        weight_decay=float(recovered_config["weight_decay"]),
    )
    if payload.get("optimizer_state") is not None:
        optimizer.load_state_dict(payload["optimizer_state"])

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

    return {
        "model": model,
        "optimizer": optimizer,
        "train_sampler": train_sampler,
        "val_sampler": val_sampler,
        "run_name": run_dir.name,
        "run_dir": run_dir,
        "progress_path": progress_path,
        "checkpoint_path": resolved_checkpoint_path,
        "checkpoints_dir": run_dir / "checkpoints",
        "plot_loss_path": run_dir / "loss_curve.png",
        "plot_top1_path": run_dir / "retrieval_curve.png",
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

    model = ContrastiveSSLModel(
        input_dim=cache_context.full_dim,
        hidden_size=int(config.hidden_size),
        s5_state_size=int(config.s5_state_size),
        num_layers=int(config.num_layers),
        dropout=float(config.dropout),
        patch_size=int(config.patch_size),
        patch_stride=int(config.patch_stride),
        post_proj_norm=str(config.post_proj_norm),
        backbone_direction=str(config.backbone_direction),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.learning_rate),
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
    checkpoints_dir = run_dir / "checkpoints"
    plot_loss_path = run_dir / "loss_curve.png"
    plot_top1_path = run_dir / "retrieval_curve.png"
    if config.checkpoint_every_steps is not None:
        checkpoints_dir.mkdir(parents=True, exist_ok=True)

    config_payload = _serialize_ssl_training_config(config, cache_context=cache_context, output_root=Path(output_root))
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
            objective_mode=str(config.objective_mode),
            device=device,
            temperature=float(config.temperature),
            horizons=tuple(config.future_horizons),
            augment_cfg=dict(config.augment_cfg),
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
                f"step={step:03d} train_loss={summary['loss']:.4f} train_top1={summary['top1']:.4f} "
                f"grad_norm={float(grad_norm):.4f} sample_s={sample_seconds:.2f} model_s={model_seconds:.2f}"
            )

        if val_sampler is not None and (step % int(config.val_every) == 0 or step == int(config.num_steps)):
            val_result = evaluate_model(
                model,
                val_sampler,
                objective_mode=str(config.objective_mode),
                num_batches=int(config.val_batches),
                device=device,
                temperature=float(config.temperature),
                horizons=tuple(config.future_horizons),
                augment_cfg=dict(config.augment_cfg),
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
                f"step={step:03d} val_loss={val_result['loss']:.4f} val_top1={val_result['top1']:.4f} "
                f"positive_pairs={val_result['positive_pairs']}"
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

    if best_state is not None:
        model.load_state_dict(best_state)
    else:
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

    print("run_dir:", run_dir)
    print("progress_path:", progress_path)
    print("checkpoint_path:", checkpoint_path)
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
        "checkpoints_dir": checkpoints_dir,
        "plot_loss_path": plot_loss_path,
        "plot_top1_path": plot_top1_path,
        "config": config_payload,
        "best_score": best_score,
        "best_step": best_step,
        "train_history": train_history,
        "val_history": val_history,
        "dataset_counts": dict(dataset_counter),
    }


def plot_ssl_training_history(run_state: dict[str, Any]) -> dict[str, Any]:
    import matplotlib.pyplot as plt

    progress_path = Path(run_state["progress_path"])
    plot_loss_path = Path(run_state["plot_loss_path"])
    plot_top1_path = Path(run_state["plot_top1_path"])
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
    train_top1 = np.array([record["top1"] for record in train_records], dtype=np.float32)

    plt.figure(figsize=(8, 4))
    plt.plot(train_steps, train_losses, label="train loss", alpha=0.8)
    if val_records:
        val_steps = np.array([record["step"] for record in val_records], dtype=np.int64)
        val_losses = np.array([record["loss"] for record in val_records], dtype=np.float32)
        plt.plot(val_steps, val_losses, marker="o", label="val loss")
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.title(f"{objective_mode} loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_loss_path, dpi=160)
    plt.show()

    plt.figure(figsize=(8, 4))
    plt.plot(train_steps, train_top1, label="train top1", alpha=0.8)
    if val_records:
        val_steps = np.array([record["step"] for record in val_records], dtype=np.int64)
        val_top1 = np.array([record["top1"] for record in val_records], dtype=np.float32)
        plt.plot(val_steps, val_top1, marker="o", label="val top1")
    plt.xlabel("step")
    plt.ylabel("retrieval top1")
    plt.title(f"{objective_mode} retrieval accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_top1_path, dpi=160)
    plt.show()

    print("train_start_loss:", float(train_losses[0]))
    print("train_final_loss:", float(train_losses[-1]))
    print("train_best_loss:", float(train_losses.min()), "at step", int(train_steps[train_losses.argmin()]))
    print("train_final_top1:", float(train_top1[-1]))
    if val_records:
        print("val_best_loss:", float(val_losses.min()), "at step", int(val_steps[val_losses.argmin()]))
        print("val_best_top1:", float(val_top1.max()), "at step", int(val_steps[val_top1.argmax()]))
    print("plot_loss_path:", plot_loss_path)
    print("plot_top1_path:", plot_top1_path)

    return {
        "records": records,
        "train_records": train_records,
        "val_records": val_records,
        "plot_loss_path": plot_loss_path,
        "plot_top1_path": plot_top1_path,
    }

"""Stage-1 POSSM-style same-bin reconstruction training helpers."""

from __future__ import annotations

import json
import math
import random
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from masked_ssl.cache import (
    CacheContext,
    build_segment_sampler,
    get_sampling_plan,
    load_cache_smoothing_provenance,
    resolve_boundary_key,
    stack_segment_batch,
)

from .model import POSSMReconstructionModel, list_registered_temporal_backbones
from .stage1_objectives import build_stage1_objective


@dataclass
class POSSMTrainingConfig:
    seed: int = 7
    model_family: str = "possm"
    stage: str = "stage1_reconstruction"
    data_mode: str = "normalized"
    feature_mode: str = "tx_sbp"
    boundary_key_mode: str = "session"
    segment_bins: int = 80
    model_dim: int = 64
    latent_count: int = 4
    value_encoder_type: str = "linear"
    value_mlp_hidden_size: int | None = None
    ffn_hidden_size: int = 256
    dropout: float = 0.1
    temporal_backbone_type: str = "gru"
    temporal_gru_hidden_size: int | None = None
    temporal_gru_num_layers: int = 1
    temporal_gru_dropout: float = 0.0
    temporal_gru_bidirectional: bool = False
    temporal_backbone_kwargs: dict[str, Any] = field(default_factory=dict)
    stage1_objective_type: str = "plain_mse"
    masking_type: str = "none"
    mask_prob: float = 0.0
    mask_span_bins: int = 8
    mask_replace_mode: str = "zero"
    batch_size: int = 8
    num_steps: int = 600
    learning_rate: float = 3e-4
    weight_decay: float = 1e-2
    val_every: int = 50
    val_batches: int = 10
    checkpoint_every_steps: int | None = None
    dataset_weight_alpha: float = 0.25
    examples_per_shard: int = 8
    log_every: int = 10
    reconstruction_head_type: str = "linear"
    reconstruction_mlp_hidden_size: int | None = None

    def __post_init__(self) -> None:
        if self.model_family != "possm":
            raise ValueError("model_family must be 'possm'")
        if self.stage != "stage1_reconstruction":
            raise ValueError("stage must be 'stage1_reconstruction'")
        if self.data_mode not in {"normalized", "raw"}:
            raise ValueError("data_mode must be one of {'normalized', 'raw'}")
        if self.feature_mode not in {"tx_only", "tx_sbp"}:
            raise ValueError("feature_mode must be one of {'tx_only', 'tx_sbp'}")
        if self.boundary_key_mode not in {"session", "subject_if_available"}:
            raise ValueError(
                "boundary_key_mode must be one of {'session', 'subject_if_available'}"
            )
        if self.value_encoder_type not in {"linear", "mlp"}:
            raise ValueError("value_encoder_type must be one of {'linear', 'mlp'}")
        available_backbones = set(list_registered_temporal_backbones())
        if self.temporal_backbone_type not in available_backbones:
            raise ValueError(
                f"temporal_backbone_type must be one of {sorted(available_backbones)}"
            )
        if not isinstance(self.temporal_backbone_kwargs, dict):
            raise ValueError("temporal_backbone_kwargs must be a dict")
        if self.temporal_backbone_type in {"identity", "gru"} and self.temporal_backbone_kwargs:
            raise ValueError(
                "temporal_backbone_kwargs is only for custom registered backbones; "
                "built-in backbones use dedicated config fields."
            )
        if self.stage1_objective_type not in {"plain_mse", "masked_mse"}:
            raise ValueError("stage1_objective_type must be one of {'plain_mse', 'masked_mse'}")
        if self.masking_type not in {"none", "random", "span", "channel"}:
            raise ValueError("masking_type must be one of {'none', 'random', 'span', 'channel'}")
        if not (0.0 <= float(self.mask_prob) <= 1.0):
            raise ValueError("mask_prob must be in [0, 1]")
        if int(self.mask_span_bins) <= 0:
            raise ValueError("mask_span_bins must be positive")
        if self.mask_replace_mode not in {"zero", "mean", "gaussian_noise"}:
            raise ValueError(
                "mask_replace_mode must be one of {'zero', 'mean', 'gaussian_noise'}"
            )
        if self.stage1_objective_type == "masked_mse":
            if self.masking_type == "none":
                raise ValueError("masked_mse requires a masking_type other than 'none'")
            if float(self.mask_prob) <= 0.0:
                raise ValueError("masked_mse requires mask_prob > 0")
        if self.temporal_gru_hidden_size is not None and int(self.temporal_gru_hidden_size) <= 0:
            raise ValueError("temporal_gru_hidden_size must be positive when provided")
        if int(self.temporal_gru_num_layers) <= 0:
            raise ValueError("temporal_gru_num_layers must be positive")
        if not (0.0 <= float(self.temporal_gru_dropout) < 1.0):
            raise ValueError("temporal_gru_dropout must be in [0, 1)")
        if self.reconstruction_head_type not in {"linear", "mlp"}:
            raise ValueError("reconstruction_head_type must be one of {'linear', 'mlp'}")
        if int(self.segment_bins) <= 0:
            raise ValueError("segment_bins must be positive")
        if int(self.model_dim) <= 0 or int(self.latent_count) <= 0:
            raise ValueError("model_dim and latent_count must be positive")
        if int(self.ffn_hidden_size) <= 0:
            raise ValueError("ffn_hidden_size must be positive")
        if not (0.0 <= float(self.dropout) < 1.0):
            raise ValueError("dropout must be in [0, 1)")
        if int(self.batch_size) <= 0 or int(self.num_steps) <= 0:
            raise ValueError("batch_size and num_steps must be positive")
        if float(self.learning_rate) <= 0.0:
            raise ValueError("learning_rate must be positive")
        if float(self.weight_decay) < 0.0:
            raise ValueError("weight_decay must be non-negative")
        if int(self.val_every) <= 0 or int(self.val_batches) <= 0:
            raise ValueError("val_every and val_batches must be positive")
        if self.checkpoint_every_steps is not None and int(self.checkpoint_every_steps) <= 0:
            raise ValueError("checkpoint_every_steps must be positive when provided")
        if float(self.dataset_weight_alpha) < 0.0:
            raise ValueError("dataset_weight_alpha must be non-negative")
        if int(self.examples_per_shard) <= 0 or int(self.log_every) <= 0:
            raise ValueError("examples_per_shard and log_every must be positive")


class RawSegmentBatchSampler:
    def __init__(
        self,
        cache_context: CacheContext,
        split_name: str,
        segment_bins: int,
        batch_size: int,
        seed: int,
        dataset_weight_alpha: float,
        examples_per_shard: int,
    ) -> None:
        self.cache_context = cache_context
        self.split_name = str(split_name)
        self.segment_bins = int(segment_bins)
        self.batch_size = int(batch_size)
        self.examples_per_shard = max(1, int(examples_per_shard))
        self.seed = int(seed)
        self.plan = get_sampling_plan(cache_context, self.split_name, self.segment_bins, dataset_weight_alpha)
        self.py_rng = random.Random(self.seed)
        self.np_rng = np.random.default_rng(self.seed)

    def _sample_raw_segment(self, example: Any) -> dict[str, Any]:
        boundary_key = resolve_boundary_key(
            dataset=example.dataset,
            session_id=example.session_id,
            subject_id=example.subject_id,
            boundary_key_mode=self.cache_context.boundary_key_mode,
        )
        shard = self.cache_context.shard_store.get(example.shard_relpath)
        time_offsets = shard["time_offsets"]
        assert isinstance(time_offsets, np.ndarray)
        start = int(time_offsets[example.example_index])
        stop = int(time_offsets[example.example_index + 1])
        length = stop - start
        total_needed = int(self.segment_bins)
        max_start = length - total_needed
        if max_start < 0:
            raise ValueError(
                f"Example {example.dataset}:{example.session_id} length={length} cannot support segment_bins={total_needed}"
            )

        offset = self.py_rng.randrange(max_start + 1)
        src_start = start + offset
        src_stop = src_start + total_needed

        x_seq = np.zeros((total_needed, self.cache_context.full_dim), dtype=np.float32)
        feature_mask = np.zeros((self.cache_context.full_dim,), dtype=np.float32)

        tx = shard["tx"]
        if isinstance(tx, np.ndarray):
            tx_window = np.asarray(tx[src_start:src_stop], dtype=np.float32)
            tx_dim = min(tx_window.shape[1], self.cache_context.tx_dim)
            x_seq[:, :tx_dim] = tx_window[:, :tx_dim]
            feature_mask[:tx_dim] = 1.0

        sbp = shard["sbp"]
        if self.cache_context.feature_mode == "tx_sbp" and isinstance(sbp, np.ndarray):
            sbp_window = np.asarray(sbp[src_start:src_stop], dtype=np.float32)
            sbp_dim = min(sbp_window.shape[1], self.cache_context.sbp_dim)
            x_seq[:, self.cache_context.tx_dim : self.cache_context.tx_dim + sbp_dim] = sbp_window[:, :sbp_dim]
            feature_mask[self.cache_context.tx_dim : self.cache_context.tx_dim + sbp_dim] = 1.0

        x_seq_t = torch.from_numpy(x_seq)
        feature_mask_t = torch.from_numpy(feature_mask)

        return {
            "x": x_seq_t,
            "feature_mask": feature_mask_t,
            "length": int(total_needed),
            "dataset": example.dataset,
            "session_id": example.session_id,
            "boundary_key": boundary_key,
            "shard_relpath": example.shard_relpath,
            "has_tx": example.has_tx,
            "has_sbp": example.has_sbp,
            "orig_len": length,
        }

    def sample_batch(self, batch_size: int | None = None) -> dict[str, Any]:
        batch_size = self.batch_size if batch_size is None else int(batch_size)
        requested_dataset_idx = self.np_rng.choice(
            len(self.plan.dataset_names),
            size=batch_size,
            p=self.plan.dataset_probs,
        )
        dataset_counts = Counter(self.plan.dataset_names[int(idx)] for idx in requested_dataset_idx)

        samples = []
        for dataset, n_examples in dataset_counts.items():
            shard_keys = self.plan.shard_keys_by_dataset[dataset]
            shard_probs = self.plan.shard_probs_by_dataset[dataset]
            n_shards = max(1, math.ceil(n_examples / self.examples_per_shard))
            replace_shards = n_shards > len(shard_keys)
            sampled_shard_idx = self.np_rng.choice(
                len(shard_keys),
                size=n_shards,
                replace=replace_shards,
                p=shard_probs,
            )

            remaining = int(n_examples)
            for shard_choice_idx, shard_idx in enumerate(np.atleast_1d(sampled_shard_idx)):
                take = min(self.examples_per_shard, remaining)
                if shard_choice_idx == n_shards - 1:
                    take = remaining
                shard_key = shard_keys[int(shard_idx)]
                shard_rows = self.plan.shard_rows_by_dataset[dataset][shard_key]
                row_probs = self.plan.row_probs_within_shard_by_dataset[dataset][shard_key]
                row_choices = self.np_rng.choice(len(shard_rows), size=take, replace=True, p=row_probs)
                for row_idx in np.atleast_1d(row_choices):
                    example = shard_rows[int(row_idx)]
                    samples.append(self._sample_raw_segment(example))
                remaining -= take
                if remaining <= 0:
                    break

        order = self.np_rng.permutation(len(samples))
        return stack_segment_batch([samples[int(idx)] for idx in order])


def build_possm_segment_sampler(
    cache_context: CacheContext,
    split_name: str,
    batch_size: int,
    *,
    seed: int,
    segment_bins: int,
    dataset_weight_alpha: float,
    examples_per_shard: int,
    data_mode: str,
) -> Any:
    if str(data_mode) == "normalized":
        return build_segment_sampler(
            cache_context,
            split_name,
            batch_size,
            seed=int(seed),
            segment_bins=int(segment_bins),
            dataset_weight_alpha=float(dataset_weight_alpha),
            examples_per_shard=int(examples_per_shard),
        )
    if split_name == "val" and not cache_context.has_val_datasets:
        raise RuntimeError("No validation datasets are eligible for session-disjoint validation.")
    return RawSegmentBatchSampler(
        cache_context=cache_context,
        split_name=str(split_name),
        segment_bins=int(segment_bins),
        batch_size=int(batch_size),
        seed=int(seed),
        dataset_weight_alpha=float(dataset_weight_alpha),
        examples_per_shard=int(examples_per_shard),
    )


def _seed_training_run(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _build_optimizer(
    model: POSSMReconstructionModel,
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
        if parameter.ndim <= 1 or name.endswith(".bias") or "norm" in lowered_name:
            no_decay_params.append(parameter)
        else:
            decay_params.append(parameter)
    param_groups: list[dict[str, Any]] = []
    if decay_params:
        param_groups.append({"params": decay_params, "weight_decay": float(weight_decay)})
    if no_decay_params:
        param_groups.append({"params": no_decay_params, "weight_decay": 0.0})
    return torch.optim.AdamW(param_groups, lr=float(learning_rate))


def _timestamp_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _step_checkpoint_filename(step: int, *, timestamp_utc: str | None = None) -> str:
    ts = timestamp_utc or _timestamp_utc()
    return f"step_{int(step):06d}_{ts}.pt"


def _final_checkpoint_filename(step: int, *, timestamp_utc: str | None = None) -> str:
    ts = timestamp_utc or _timestamp_utc()
    return f"checkpoint_final_step_{int(step):06d}_{ts}.pt"


def _parse_step_from_checkpoint_name(name: str) -> int | None:
    stem = Path(name).stem
    if stem.startswith("step_"):
        parts = stem.split("_")
        if len(parts) >= 2 and parts[1].isdigit():
            return int(parts[1])
    if stem.startswith("checkpoint_final_step_"):
        parts = stem.split("_")
        if len(parts) >= 4 and parts[3].isdigit():
            return int(parts[3])
    return None


def list_possm_checkpoints(run_dir: str | Path) -> list[dict[str, Any]]:
    base = Path(run_dir)
    checkpoint_paths: list[Path] = []
    step_dir = base / "checkpoints"
    if step_dir.exists():
        checkpoint_paths.extend(sorted(step_dir.glob("step_*.pt")))
        checkpoint_paths.extend(sorted(step_dir.glob("checkpoint_final_step_*.pt")))
    final_path = base / "checkpoint_final.pt"
    if final_path.exists():
        checkpoint_paths.append(final_path)
    best_path = base / "checkpoint_best.pt"
    if best_path.exists():
        checkpoint_paths.append(best_path)

    rows: list[dict[str, Any]] = []
    for path in checkpoint_paths:
        kind = (
            "best"
            if path.name == "checkpoint_best.pt"
            else "final"
            if path.name == "checkpoint_final.pt"
            else "step"
        )
        rows.append(
            {
                "path": str(path),
                "name": path.name,
                "kind": kind,
                "step": _parse_step_from_checkpoint_name(path.name),
                "size_bytes": int(path.stat().st_size),
                "mtime_ns": int(path.stat().st_mtime_ns),
            }
        )
    rows.sort(key=lambda row: (row["kind"] != "best", row["step"] or -1, row["mtime_ns"]))
    return rows


def resolve_possm_checkpoint_path(
    *,
    output_root: str | Path | None = None,
    run_dir: str | Path | None = None,
    explicit_checkpoint_path: str | Path | None = None,
) -> Path:
    if explicit_checkpoint_path is not None:
        candidate = Path(explicit_checkpoint_path)
        if not candidate.exists():
            raise FileNotFoundError(f"Explicit POSSM checkpoint path does not exist: {candidate}")
        return candidate
    if run_dir is not None:
        candidate = Path(run_dir) / "checkpoint_best.pt"
        if candidate.exists():
            return candidate
        candidate = Path(run_dir) / "checkpoint_final.pt"
        if candidate.exists():
            return candidate
        checkpoints = list_possm_checkpoints(Path(run_dir))
        if checkpoints:
            return Path(checkpoints[-1]["path"])
        raise RuntimeError(f"No POSSM checkpoints found under run_dir={run_dir}")
    if output_root is None:
        raise ValueError("One of output_root, run_dir, or explicit_checkpoint_path must be provided")
    root = Path(output_root)
    candidates = sorted((path for path in root.glob("*") if path.is_dir()), key=lambda p: p.stat().st_mtime_ns)
    for candidate in reversed(candidates):
        best = candidate / "checkpoint_best.pt"
        if best.exists():
            return best
        final = candidate / "checkpoint_final.pt"
        if final.exists():
            return final
    raise RuntimeError(f"No POSSM checkpoints found under output_root={root}")


def _serialize_config(
    config: POSSMTrainingConfig,
    *,
    cache_context: CacheContext,
) -> dict[str, Any]:
    cache_root = getattr(cache_context, "cache_root", None)
    pretrain_datasets = tuple(getattr(cache_context, "pretrain_datasets", ()) or ())
    smoothing_provenance = (
        None
        if cache_root is None
        else load_cache_smoothing_provenance(
            cache_root,
            dataset=(pretrain_datasets[0] if pretrain_datasets else None),
        )
    )
    return {
        **asdict(config),
        "input_dim": int(cache_context.full_dim),
        "cache_use_normalization": bool(cache_context.use_normalization),
        "cache_source_signature": (
            None
            if getattr(cache_context, "source_cache_signature", None) is None
            else str(cache_context.source_cache_signature)
        ),
        "cache_smoothing_provenance": smoothing_provenance,
        "cache_gaussian_smoothing_sigma_bins": (
            None
            if not isinstance(smoothing_provenance, dict)
            else smoothing_provenance.get("sigma_bins")
        ),
    }


def _build_checkpoint_payload(
    *,
    model: POSSMReconstructionModel,
    optimizer: torch.optim.Optimizer,
    config_payload: dict[str, Any],
    step: int,
    best_score: float | None,
    best_step: int | None,
    checkpoint_kind: str,
    dataset_counter: Counter[str],
    train_history: list[dict[str, Any]] | None = None,
    val_history: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "model_family": "possm",
        "stage": "stage1_reconstruction",
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "config": config_payload,
        "step": int(step),
        "best_score": None if best_score is None else float(best_score),
        "best_step": None if best_step is None else int(best_step),
        "checkpoint_kind": str(checkpoint_kind),
        "dataset_counts": dict(dataset_counter),
        "train_history": list(train_history or ()),
        "val_history": list(val_history or ()),
    }


def compute_reconstruction_metrics(
    model: POSSMReconstructionModel,
    raw_batch: dict[str, Any],
    objective: Any,
    config: dict[str, Any],
    *,
    device: torch.device,
) -> dict[str, Any]:
    resolved_config = dict(config)
    stage1_batch = objective.prepare_batch(raw_batch, device=device, config=resolved_config)
    outputs = model(stage1_batch.x_input, stage1_batch.lengths, session_ids=stage1_batch.session_ids)
    metrics = objective.compute_loss(outputs, stage1_batch)
    if stage1_batch.mask_metadata is not None:
        metrics = {**metrics, "mask_metadata": dict(stage1_batch.mask_metadata)}
    return metrics


def evaluate_model(
    model: POSSMReconstructionModel,
    sampler: Any,
    objective: Any,
    config: dict[str, Any],
    *,
    num_batches: int,
    device: torch.device,
) -> dict[str, Any] | None:
    if sampler is None:
        return None
    was_training = model.training
    model.eval()
    losses: list[float] = []
    with torch.no_grad():
        for _ in range(int(num_batches)):
            batch = sampler.sample_batch()
            metrics = compute_reconstruction_metrics(
                model,
                batch,
                objective,
                config,
                device=device,
            )
            losses.append(float(metrics["mse"]))
    if was_training:
        model.train()
    return {"loss": float(np.mean(losses)), "mse": float(np.mean(losses))}


def _emit_progress(progress_path: Path, payload: dict[str, Any]) -> None:
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    with progress_path.open("a") as handle:
        handle.write(json.dumps(payload) + "\n")


def _build_model_from_config(config: dict[str, Any]) -> POSSMReconstructionModel:
    return POSSMReconstructionModel(
        input_dim=int(config["input_dim"]),
        model_dim=int(config["model_dim"]),
        latent_count=int(config["latent_count"]),
        value_encoder_type=str(config["value_encoder_type"]),
        value_mlp_hidden_size=(
            None
            if config.get("value_mlp_hidden_size") is None
            else int(config["value_mlp_hidden_size"])
        ),
        ffn_hidden_size=int(config["ffn_hidden_size"]),
        dropout=float(config["dropout"]),
        temporal_backbone_type=str(config.get("temporal_backbone_type", "gru")),
        temporal_gru_hidden_size=(
            None
            if config.get("temporal_gru_hidden_size") is None
            else int(config["temporal_gru_hidden_size"])
        ),
        temporal_gru_num_layers=int(config.get("temporal_gru_num_layers", 1)),
        temporal_gru_dropout=float(config.get("temporal_gru_dropout", 0.0)),
        temporal_gru_bidirectional=bool(config.get("temporal_gru_bidirectional", False)),
        temporal_backbone_kwargs=dict(config.get("temporal_backbone_kwargs", {})),
        reconstruction_head_type=str(config["reconstruction_head_type"]),
        reconstruction_mlp_hidden_size=(
            None
            if config.get("reconstruction_mlp_hidden_size") is None
            else int(config["reconstruction_mlp_hidden_size"])
        ),
        feature_mode=str(config.get("feature_mode", "tx_sbp")),
    )


def _initial_run_state(
    *,
    model: POSSMReconstructionModel,
    optimizer: torch.optim.Optimizer,
    objective: Any,
    train_sampler: Any,
    val_sampler: Any,
    run_name: str,
    run_dir: Path,
    progress_path: Path,
    checkpoint_path: Path,
    best_checkpoint_path: Path,
    checkpoints_dir: Path,
    config_payload: dict[str, Any],
) -> dict[str, Any]:
    return {
        "model": model,
        "optimizer": optimizer,
        "objective": objective,
        "train_sampler": train_sampler,
        "val_sampler": val_sampler,
        "run_name": run_name,
        "run_dir": run_dir,
        "progress_path": progress_path,
        "checkpoint_path": checkpoint_path,
        "best_checkpoint_path": best_checkpoint_path,
        "checkpoints_dir": checkpoints_dir,
        "config": config_payload,
        "best_score": None,
        "best_step": None,
        "train_history": [],
        "val_history": [],
        "dataset_counts": {},
        "checkpoint_step": 0,
        "checkpoint_kind": None,
    }


def _train_loop(
    *,
    run_state: dict[str, Any],
    target_step: int,
    device: torch.device,
) -> dict[str, Any]:
    model: POSSMReconstructionModel = run_state["model"]
    optimizer: torch.optim.Optimizer = run_state["optimizer"]
    objective = run_state["objective"]
    train_sampler = run_state["train_sampler"]
    val_sampler = run_state["val_sampler"]
    config_payload = dict(run_state["config"])
    progress_path = Path(run_state["progress_path"])
    checkpoint_path = Path(run_state["checkpoint_path"])
    best_checkpoint_path = Path(run_state["best_checkpoint_path"])
    checkpoints_dir = Path(run_state["checkpoints_dir"])
    run_dir = Path(run_state["run_dir"])

    train_history = list(run_state.get("train_history", []))
    val_history = list(run_state.get("val_history", []))
    dataset_counter = Counter(run_state.get("dataset_counts", {}))
    best_score = (
        float(run_state["best_score"])
        if run_state.get("best_score") is not None
        else float("inf")
    )
    best_step = int(run_state["best_step"]) if run_state.get("best_step") is not None else None
    best_state = None
    start_step = int(run_state.get("checkpoint_step", 0) or 0)

    log_every = int(config_payload["log_every"])
    val_every = int(config_payload["val_every"])
    val_batches = int(config_payload["val_batches"])
    checkpoint_every_steps = config_payload.get("checkpoint_every_steps")

    loop_start = time.time()
    for step in range(start_step + 1, int(target_step) + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        sample_start = time.time()
        batch = train_sampler.sample_batch()
        sample_seconds = time.time() - sample_start
        dataset_counter.update(batch["datasets"])
        metrics = compute_reconstruction_metrics(
            model,
            batch,
            objective,
            config_payload,
            device=device,
        )
        loss = metrics["loss"]
        loss.backward()
        optimizer.step()

        train_record = {
            "event": "train",
            "step": int(step),
            "elapsed_seconds": round(time.time() - loop_start, 3),
            "loss": float(metrics["mse"]),
            "sample_seconds": round(sample_seconds, 4),
            "objective_type": str(metrics.get("objective_type", config_payload.get("stage1_objective_type", "plain_mse"))),
            "masked_fraction": float(metrics.get("masked_fraction", 0.0)),
            "dataset_mix": dict(Counter(batch["datasets"])),
        }
        train_history.append(train_record)
        if step == 1 or step % log_every == 0:
            _emit_progress(progress_path, train_record)
            print(f"step={step:04d} train_loss={metrics['mse']:.6f}")

        if val_sampler is not None and (step == 1 or step % val_every == 0):
            val_result = evaluate_model(
                model,
                val_sampler,
                objective,
                config_payload,
                num_batches=val_batches,
                device=device,
            )
            assert val_result is not None
            val_record = {
                "event": "val",
                "step": int(step),
                "elapsed_seconds": round(time.time() - loop_start, 3),
                "loss": float(val_result["loss"]),
            }
            val_history.append(val_record)
            _emit_progress(progress_path, val_record)
            print(f"step={step:04d} val_loss={val_result['loss']:.6f}")
            if float(val_result["loss"]) < float(best_score):
                best_score = float(val_result["loss"])
                best_step = int(step)
                best_state = {
                    key: value.detach().cpu().clone() for key, value in model.state_dict().items()
                }

        if checkpoint_every_steps is not None and step % int(checkpoint_every_steps) == 0:
            checkpoints_dir.mkdir(parents=True, exist_ok=True)
            step_payload = _build_checkpoint_payload(
                model=model,
                optimizer=optimizer,
                config_payload=config_payload,
                step=step,
                best_score=best_score,
                best_step=best_step,
                checkpoint_kind="step",
                dataset_counter=dataset_counter,
                train_history=train_history,
                val_history=val_history,
            )
            step_checkpoint = checkpoints_dir / _step_checkpoint_filename(step)
            torch.save(step_payload, step_checkpoint)
            print("saved_step_checkpoint:", step_checkpoint)

    if best_state is None:
        best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    final_payload = _build_checkpoint_payload(
        model=model,
        optimizer=optimizer,
        config_payload=config_payload,
        step=int(target_step),
        best_score=best_score,
        best_step=best_step,
        checkpoint_kind="final",
        dataset_counter=dataset_counter,
        train_history=train_history,
        val_history=val_history,
    )
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    torch.save(final_payload, checkpoint_path)
    torch.save(final_payload, checkpoints_dir / _final_checkpoint_filename(int(target_step)))

    best_payload = dict(final_payload)
    best_payload["model_state"] = best_state
    best_payload["optimizer_state"] = None
    best_payload["step"] = int(best_step if best_step is not None else target_step)
    best_payload["checkpoint_kind"] = "best"
    torch.save(best_payload, best_checkpoint_path)

    run_state.update(
        {
            "best_score": None if best_step is None else float(best_score),
            "best_step": best_step,
            "train_history": train_history,
            "val_history": val_history,
            "dataset_counts": dict(dataset_counter),
            "checkpoint_step": int(target_step),
            "checkpoint_kind": "final",
        }
    )
    print("run_dir:", run_dir)
    print("checkpoint_path:", checkpoint_path)
    print("best_checkpoint_path:", best_checkpoint_path)
    return run_state


def run_possm_training(
    *,
    cache_context: CacheContext,
    config: POSSMTrainingConfig,
    output_root: Path,
    device: torch.device,
) -> dict[str, Any]:
    _seed_training_run(int(config.seed))
    if str(config.feature_mode) != str(cache_context.feature_mode):
        raise ValueError(
            "POSSMTrainingConfig.feature_mode must match CacheAccessConfig.feature_mode. "
            f"config={config.feature_mode!r} cache={cache_context.feature_mode!r}"
        )
    if str(config.boundary_key_mode) != str(cache_context.boundary_key_mode):
        raise ValueError(
            "POSSMTrainingConfig.boundary_key_mode must match CacheAccessConfig.boundary_key_mode. "
            f"config={config.boundary_key_mode!r} cache={cache_context.boundary_key_mode!r}"
        )

    train_sampler = build_possm_segment_sampler(
        cache_context,
        "train",
        int(config.batch_size),
        seed=int(config.seed),
        segment_bins=int(config.segment_bins),
        dataset_weight_alpha=float(config.dataset_weight_alpha),
        examples_per_shard=int(config.examples_per_shard),
        data_mode=str(config.data_mode),
    )
    try:
        val_sampler = build_possm_segment_sampler(
            cache_context,
            "val",
            int(config.batch_size),
            seed=int(config.seed) + 101,
            segment_bins=int(config.segment_bins),
            dataset_weight_alpha=float(config.dataset_weight_alpha),
            examples_per_shard=int(config.examples_per_shard),
            data_mode=str(config.data_mode),
        )
    except RuntimeError:
        val_sampler = None

    model = POSSMReconstructionModel(
        input_dim=int(cache_context.full_dim),
        model_dim=int(config.model_dim),
        latent_count=int(config.latent_count),
        value_encoder_type=str(config.value_encoder_type),
        value_mlp_hidden_size=config.value_mlp_hidden_size,
        ffn_hidden_size=int(config.ffn_hidden_size),
        dropout=float(config.dropout),
        temporal_backbone_type=str(config.temporal_backbone_type),
        temporal_gru_hidden_size=config.temporal_gru_hidden_size,
        temporal_gru_num_layers=int(config.temporal_gru_num_layers),
        temporal_gru_dropout=float(config.temporal_gru_dropout),
        temporal_gru_bidirectional=bool(config.temporal_gru_bidirectional),
        temporal_backbone_kwargs=dict(config.temporal_backbone_kwargs),
        reconstruction_head_type=str(config.reconstruction_head_type),
        reconstruction_mlp_hidden_size=config.reconstruction_mlp_hidden_size,
        feature_mode=str(config.feature_mode),
    ).to(device)
    optimizer = _build_optimizer(
        model,
        learning_rate=float(config.learning_rate),
        weight_decay=float(config.weight_decay),
    )

    config_payload = _serialize_config(config, cache_context=cache_context)
    objective = build_stage1_objective(config=config_payload, seed=int(config.seed))
    run_name = f"possm_stage1_{config.feature_mode}_{config.data_mode}_{_timestamp_utc()}"
    run_dir = Path(output_root) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    progress_path = run_dir / "progress.jsonl"
    checkpoint_path = run_dir / "checkpoint_final.pt"
    best_checkpoint_path = run_dir / "checkpoint_best.pt"
    checkpoints_dir = run_dir / "checkpoints"

    run_state = _initial_run_state(
        model=model,
        optimizer=optimizer,
        objective=objective,
        train_sampler=train_sampler,
        val_sampler=val_sampler,
        run_name=run_name,
        run_dir=run_dir,
        progress_path=progress_path,
        checkpoint_path=checkpoint_path,
        best_checkpoint_path=best_checkpoint_path,
        checkpoints_dir=checkpoints_dir,
        config_payload=config_payload,
    )
    return _train_loop(run_state=run_state, target_step=int(config.num_steps), device=device)


def recover_possm_run_state_from_checkpoint(
    *,
    cache_context: CacheContext,
    checkpoint_path: str | Path,
    device: torch.device,
) -> dict[str, Any]:
    resolved_checkpoint_path = Path(checkpoint_path)
    payload = torch.load(resolved_checkpoint_path, map_location="cpu")
    recovered_config = dict(payload.get("config", {}))
    recovered_config.setdefault("stage1_objective_type", "plain_mse")
    recovered_config.setdefault("masking_type", "none")
    recovered_config.setdefault("mask_prob", 0.0)
    recovered_config.setdefault("mask_span_bins", 8)
    recovered_config.setdefault("mask_replace_mode", "zero")
    recovered_config.setdefault("temporal_backbone_kwargs", {})
    if str(payload.get("model_family", recovered_config.get("model_family", ""))) != "possm":
        raise ValueError("Recovered checkpoint is not a POSSM checkpoint.")
    if str(recovered_config.get("stage", payload.get("stage", ""))) != "stage1_reconstruction":
        raise ValueError("Recovered checkpoint is not a POSSM stage-1 reconstruction checkpoint.")
    if str(recovered_config["feature_mode"]) != str(cache_context.feature_mode):
        raise ValueError(
            "Recovered checkpoint feature_mode does not match the active cache context. "
            f"checkpoint={recovered_config['feature_mode']!r} cache={cache_context.feature_mode!r}"
        )
    if str(recovered_config.get("boundary_key_mode", "session")) != str(cache_context.boundary_key_mode):
        raise ValueError(
            "Recovered checkpoint boundary_key_mode does not match the active cache context. "
            f"checkpoint={recovered_config.get('boundary_key_mode', 'session')!r} "
            f"cache={cache_context.boundary_key_mode!r}"
        )
    if int(recovered_config["input_dim"]) != int(cache_context.full_dim):
        raise ValueError(
            "Recovered checkpoint input_dim does not match the active cache context. "
            f"checkpoint={int(recovered_config['input_dim'])} cache={int(cache_context.full_dim)}"
        )
    recovered_use_normalization = bool(recovered_config.get("cache_use_normalization", True))
    recovered_data_mode = str(recovered_config.get("data_mode", "normalized"))
    if recovered_data_mode == "normalized" and recovered_use_normalization != bool(cache_context.use_normalization):
        raise ValueError(
            "Recovered checkpoint cache normalization setting does not match the active cache context. "
            f"checkpoint={recovered_use_normalization!r} cache={bool(cache_context.use_normalization)!r}"
        )

    model = _build_model_from_config(recovered_config).to(device)
    model_state = payload.get("model_state")
    if model_state is None:
        raise KeyError("Recovered POSSM checkpoint is missing 'model_state'.")
    model.load_state_dict(model_state)
    model.eval()

    optimizer = _build_optimizer(
        model,
        learning_rate=float(recovered_config["learning_rate"]),
        weight_decay=float(recovered_config["weight_decay"]),
    )
    if payload.get("optimizer_state") is not None:
        optimizer.load_state_dict(payload["optimizer_state"])
    objective = build_stage1_objective(
        config=recovered_config,
        seed=int(recovered_config.get("seed", 7)),
    )

    train_sampler = build_possm_segment_sampler(
        cache_context,
        "train",
        int(recovered_config["batch_size"]),
        seed=int(recovered_config["seed"]),
        segment_bins=int(recovered_config["segment_bins"]),
        dataset_weight_alpha=float(recovered_config["dataset_weight_alpha"]),
        examples_per_shard=int(recovered_config["examples_per_shard"]),
        data_mode=str(recovered_config.get("data_mode", "normalized")),
    )
    try:
        val_sampler = build_possm_segment_sampler(
            cache_context,
            "val",
            int(recovered_config["batch_size"]),
            seed=int(recovered_config["seed"]) + 101,
            segment_bins=int(recovered_config["segment_bins"]),
            dataset_weight_alpha=float(recovered_config["dataset_weight_alpha"]),
            examples_per_shard=int(recovered_config["examples_per_shard"]),
            data_mode=str(recovered_config.get("data_mode", "normalized")),
        )
    except RuntimeError:
        val_sampler = None

    run_dir = resolved_checkpoint_path.parent.parent if resolved_checkpoint_path.parent.name == "checkpoints" else resolved_checkpoint_path.parent
    progress_path = run_dir / "progress.jsonl"
    checkpoint_step = int(payload.get("step", 0))
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
        "objective": objective,
        "train_sampler": train_sampler,
        "val_sampler": val_sampler,
        "run_name": run_dir.name,
        "run_dir": run_dir,
        "progress_path": progress_path,
        "checkpoint_path": resolved_checkpoint_path,
        "best_checkpoint_path": run_dir / "checkpoint_best.pt",
        "checkpoints_dir": run_dir / "checkpoints",
        "config": recovered_config,
        "best_score": payload.get("best_score"),
        "best_step": payload.get("best_step"),
        "train_history": train_history,
        "val_history": val_history,
        "dataset_counts": dict(payload.get("dataset_counts", {})),
        "checkpoint_step": checkpoint_step,
        "checkpoint_kind": payload.get("checkpoint_kind"),
    }


def resume_possm_training(
    *,
    run_state: dict[str, Any],
    additional_steps: int,
    cache_context: CacheContext,
    device: torch.device,
) -> dict[str, Any]:
    del cache_context
    additional_steps = int(additional_steps)
    if additional_steps <= 0:
        raise ValueError("additional_steps must be positive")
    start_step = int(run_state.get("checkpoint_step", 0) or 0)
    target_step = start_step + additional_steps
    run_state["config"] = {**dict(run_state["config"]), "num_steps": int(target_step)}
    return _train_loop(run_state=run_state, target_step=target_step, device=device)

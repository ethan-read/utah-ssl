"""Stage-2 POSSM phoneme fine-tuning helpers."""

from __future__ import annotations

import copy
import json
import math
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from masked_ssl.probe import (
    CanonicalSequenceDataset,
    DownstreamProbeConfig,
    _make_loader_generator,
    build_downstream_probe_problem,
    collate_sequence_batch,
    compute_ctc_loss_sum,
    compute_feature_stats,
)

from .model import POSSMEncoder, POSSMPhonemeModel, build_temporal_backbone


@dataclass
class POSSMFinetuneConfig:
    seed: int = 7
    mode: str = "finetune_full"
    dataset: str = "brain2text24"
    feature_mode: str | None = None
    data_mode: str | None = None
    boundary_key_mode: str | None = None
    session_limit: int = 8
    target_session_count: int = 4
    batch_size: int = 8
    num_steps: int = 400
    budget_seconds: int = 240
    learning_rate: float = 1e-3
    encoder_learning_rate: float = 3e-4
    weight_decay: float = 1e-2
    max_grad_norm: float = 1.0
    checkpoint_every_steps: int = 200
    progress_every_steps: int = 25
    progress_every_seconds: float = 15.0
    gru_hidden_size: int = 768
    gru_num_layers: int = 5
    gru_dropout: float = 0.4
    conv_hidden_size: int | None = None
    conv_kernel_size: int = 14
    conv_stride: int = 4
    conv_dropout: float = 0.1

    def __post_init__(self) -> None:
        if self.mode not in {"probe_frozen", "finetune_full"}:
            raise ValueError("mode must be one of {'probe_frozen', 'finetune_full'}")
        if self.feature_mode is not None and self.feature_mode not in {"tx_only", "tx_sbp"}:
            raise ValueError("feature_mode must be one of {'tx_only', 'tx_sbp'} when provided")
        if self.data_mode is not None and self.data_mode not in {"raw", "normalized"}:
            raise ValueError("data_mode must be one of {'raw', 'normalized'} when provided")
        if self.boundary_key_mode is not None and self.boundary_key_mode not in {
            "session",
            "subject_if_available",
        }:
            raise ValueError(
                "boundary_key_mode must be one of {'session', 'subject_if_available'} when provided"
            )
        if int(self.session_limit) <= 0 or int(self.target_session_count) <= 0:
            raise ValueError("session counts must be positive")
        if int(self.target_session_count) >= int(self.session_limit):
            raise ValueError("target_session_count must be smaller than session_limit")
        if int(self.batch_size) <= 0 or int(self.num_steps) <= 0:
            raise ValueError("batch_size and num_steps must be positive")
        if float(self.budget_seconds) <= 0.0:
            raise ValueError("budget_seconds must be positive")
        if float(self.learning_rate) <= 0.0 or float(self.encoder_learning_rate) <= 0.0:
            raise ValueError("learning rates must be positive")
        if float(self.weight_decay) < 0.0:
            raise ValueError("weight_decay must be non-negative")
        if float(self.max_grad_norm) <= 0.0:
            raise ValueError("max_grad_norm must be positive")
        if int(self.checkpoint_every_steps) <= 0:
            raise ValueError("checkpoint_every_steps must be positive")
        if int(self.progress_every_steps) <= 0 or float(self.progress_every_seconds) <= 0.0:
            raise ValueError("progress reporting cadence must be positive")
        if int(self.gru_hidden_size) <= 0 or int(self.gru_num_layers) <= 0:
            raise ValueError("GRU sizes must be positive")
        if not (0.0 <= float(self.gru_dropout) < 1.0):
            raise ValueError("gru_dropout must be in [0, 1)")
        if int(self.conv_kernel_size) <= 0 or int(self.conv_stride) <= 0:
            raise ValueError("Conv kernel and stride must be positive")
        if not (0.0 <= float(self.conv_dropout) < 1.0):
            raise ValueError("conv_dropout must be in [0, 1)")


def _seed_all(seed: int) -> None:
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _loader_kwargs(device: torch.device, batch_size: int, *, shuffle: bool) -> dict[str, Any]:
    return {
        "batch_size": int(batch_size),
        "shuffle": shuffle,
        "num_workers": 0,
        "pin_memory": device.type == "cuda",
        "collate_fn": collate_sequence_batch,
    }


def _count_trainable_parameters(module: torch.nn.Module) -> int:
    return int(sum(param.numel() for param in module.parameters() if param.requires_grad))


def _count_trainable_sequence_encoder_parameters(model: POSSMPhonemeModel) -> int:
    total = _count_trainable_parameters(model.base_encoder)
    if model.pre_decoder_backbone is not None:
        total += _count_trainable_parameters(model.pre_decoder_backbone)
    return int(total)


def _emit_progress(progress_log_path: Path | None, **payload: Any) -> None:
    if progress_log_path is None:
        return
    progress_log_path.parent.mkdir(parents=True, exist_ok=True)
    with progress_log_path.open("a") as handle:
        handle.write(json.dumps(payload) + "\n")


def _set_train_mode(
    model: POSSMPhonemeModel,
    *,
    train_encoder: bool,
) -> None:
    if train_encoder:
        model.train()
        return
    model.eval()
    model.gru.train()
    model.conv.train()
    model.conv_dropout.train()
    model.classifier.train()


def _ctc_greedy_decode(
    logits: torch.Tensor,
    token_lengths: torch.Tensor,
    *,
    blank_index: int,
) -> list[list[int]]:
    token_ids = logits.argmax(dim=-1)
    decoded: list[list[int]] = []
    for batch_idx, length in enumerate(token_lengths.tolist()):
        sequence: list[int] = []
        prev_token: int | None = None
        for token in token_ids[batch_idx, :length].tolist():
            if token == blank_index:
                prev_token = None
                continue
            if token != prev_token:
                sequence.append(int(token))
            prev_token = int(token)
        decoded.append(sequence)
    return decoded


def _edit_distance(reference: list[int], hypothesis: list[int]) -> int:
    if not reference:
        return len(hypothesis)
    if not hypothesis:
        return len(reference)
    previous = list(range(len(hypothesis) + 1))
    for ref_idx, ref_token in enumerate(reference, start=1):
        current = [ref_idx]
        for hyp_idx, hyp_token in enumerate(hypothesis, start=1):
            substitution_cost = 0 if ref_token == hyp_token else 1
            current.append(
                min(
                    previous[hyp_idx] + 1,
                    current[hyp_idx - 1] + 1,
                    previous[hyp_idx - 1] + substitution_cost,
                )
            )
        previous = current
    return previous[-1]


def evaluate_possm_phoneme_metrics(
    *,
    model: POSSMPhonemeModel,
    loader: DataLoader,
    device: torch.device,
    blank_index: int,
) -> dict[str, Any]:
    model.eval()
    total_loss_sum = 0.0
    total_targets = 0
    total_edit_distance = 0
    total_reference_tokens = 0
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
                blank_index=blank_index,
            )
            total_loss_sum += float(loss_sum.item())
            total_targets += int(target_count)
            predictions = _ctc_greedy_decode(
                outputs["logits"],
                outputs["token_lengths"],
                blank_index=blank_index,
            )
            for row_idx, prediction in enumerate(predictions):
                reference_length = int(label_lengths[row_idx].item())
                reference = labels[row_idx, :reference_length].tolist()
                total_edit_distance += _edit_distance(reference, prediction)
                total_reference_tokens += len(reference)
    if total_targets <= 0:
        raise ValueError("Validation target count is zero; cannot compute val_ctc_bpphone.")
    if total_reference_tokens <= 0:
        raise ValueError("Validation reference token count is zero; cannot compute PER.")
    return {
        "val_ctc_bpphone": float(total_loss_sum / total_targets / math.log(2.0)),
        "val_phoneme_error_rate": float(total_edit_distance / total_reference_tokens),
    }


def _load_stage1_checkpoint(
    checkpoint_path: Path,
    *,
    map_location: str | torch.device,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], Path]:
    payload = torch.load(checkpoint_path, map_location=map_location)
    checkpoint_cfg = dict(payload.get("config", {}))
    if str(payload.get("model_family", checkpoint_cfg.get("model_family", ""))) != "possm":
        raise ValueError("Checkpoint is not a POSSM checkpoint.")
    model_state = payload.get("model_state")
    if model_state is None:
        raise KeyError("Stage-1 POSSM checkpoint is missing 'model_state'.")
    if not isinstance(model_state, dict):
        raise TypeError("Stage-1 POSSM checkpoint 'model_state' must be a state dict.")
    run_dir = checkpoint_path.parent.parent if checkpoint_path.parent.name == "checkpoints" else checkpoint_path.parent
    return payload, checkpoint_cfg, model_state, run_dir


def _build_stage1_encoder_from_checkpoint_state(
    *,
    checkpoint_cfg: dict[str, Any],
    model_state: dict[str, Any],
) -> POSSMEncoder:
    encoder_state = {
        key.split("encoder.", 1)[1]: value
        for key, value in model_state.items()
        if key.startswith("encoder.")
    }
    if not encoder_state:
        raise KeyError("Stage-1 POSSM checkpoint does not contain encoder weights.")
    encoder = POSSMEncoder(
        input_dim=int(checkpoint_cfg["input_dim"]),
        model_dim=int(checkpoint_cfg["model_dim"]),
        latent_count=int(checkpoint_cfg["latent_count"]),
        value_encoder_type=str(checkpoint_cfg["value_encoder_type"]),
        value_mlp_hidden_size=(
            None
            if checkpoint_cfg.get("value_mlp_hidden_size") is None
            else int(checkpoint_cfg["value_mlp_hidden_size"])
        ),
        ffn_hidden_size=int(checkpoint_cfg["ffn_hidden_size"]),
        dropout=float(checkpoint_cfg["dropout"]),
        feature_mode=str(checkpoint_cfg.get("feature_mode", "tx_sbp")),
    )
    encoder.load_state_dict(encoder_state)
    return encoder


def recover_possm_stage1_encoder(
    *,
    checkpoint_path: Path,
    map_location: str | torch.device = "cpu",
) -> tuple[POSSMEncoder, dict[str, Any], Path]:
    _, checkpoint_cfg, model_state, run_dir = _load_stage1_checkpoint(
        checkpoint_path,
        map_location=map_location,
    )
    encoder = _build_stage1_encoder_from_checkpoint_state(
        checkpoint_cfg=checkpoint_cfg,
        model_state=model_state,
    )
    return encoder, checkpoint_cfg, run_dir


def recover_possm_stage1_sequence_components(
    *,
    checkpoint_path: Path,
    map_location: str | torch.device = "cpu",
) -> tuple[POSSMEncoder, torch.nn.Module, dict[str, Any], Path]:
    _, checkpoint_cfg, model_state, run_dir = _load_stage1_checkpoint(
        checkpoint_path,
        map_location=map_location,
    )
    encoder = _build_stage1_encoder_from_checkpoint_state(
        checkpoint_cfg=checkpoint_cfg,
        model_state=model_state,
    )
    temporal_state = {
        key.split("temporal_backbone.", 1)[1]: value
        for key, value in model_state.items()
        if key.startswith("temporal_backbone.")
    }
    if not temporal_state:
        if "temporal_backbone_type" in checkpoint_cfg:
            raise KeyError(
                "Stage-1 POSSM checkpoint declares a temporal backbone but contains no "
                "'temporal_backbone.*' weights."
            )
        temporal_backbone = build_temporal_backbone(
            backbone_type="identity",
            input_size=int(encoder.hidden_size),
        )
        return encoder, temporal_backbone, checkpoint_cfg, run_dir
    temporal_backbone = build_temporal_backbone(
        backbone_type=str(checkpoint_cfg.get("temporal_backbone_type", "gru")),
        input_size=int(encoder.hidden_size),
        gru_hidden_size=(
            None
            if checkpoint_cfg.get("temporal_gru_hidden_size") is None
            else int(checkpoint_cfg["temporal_gru_hidden_size"])
        ),
        gru_num_layers=int(checkpoint_cfg.get("temporal_gru_num_layers", 1)),
        gru_dropout=float(checkpoint_cfg.get("temporal_gru_dropout", 0.0)),
        gru_bidirectional=bool(checkpoint_cfg.get("temporal_gru_bidirectional", False)),
        backbone_kwargs=dict(checkpoint_cfg.get("temporal_backbone_kwargs", {})),
    )
    temporal_backbone.load_state_dict(temporal_state)
    return encoder, temporal_backbone, checkpoint_cfg, run_dir


def _build_problem(
    *,
    cache_root: Path,
    config: POSSMFinetuneConfig,
    feature_mode: str,
    boundary_key_mode: str,
) -> dict[str, Any]:
    probe_config = DownstreamProbeConfig(
        enabled=True,
        seed=int(config.seed),
        session_limit=int(config.session_limit),
        target_session_count=int(config.target_session_count),
        probe_batch_size=int(config.batch_size),
        probe_budget_seconds=int(config.budget_seconds),
        max_probe_steps=int(config.num_steps),
        probe_head_learning_rate=float(config.learning_rate),
        encoder_learning_rate=float(config.encoder_learning_rate),
        weight_decay=float(config.weight_decay),
        probe_head_type="linear",
    )
    return build_downstream_probe_problem(
        cache_root=cache_root,
        probe_config=probe_config,
        feature_mode=str(feature_mode),
        boundary_key_mode=str(boundary_key_mode),
        dataset=str(config.dataset),
    )


def _checkpoint_payload(
    *,
    model: POSSMPhonemeModel,
    resolved_config: POSSMFinetuneConfig,
    resolved_checkpoint_path: Path,
    checkpoint_cfg: dict[str, Any],
    problem: dict[str, Any],
    steps: int,
    metrics: dict[str, Any],
    checkpoint_kind: str,
) -> dict[str, Any]:
    return {
        "model_family": "possm",
        "stage": "stage2_phoneme_finetune",
        "stage1_checkpoint_path": str(resolved_checkpoint_path),
        "stage1_checkpoint_config": dict(checkpoint_cfg),
        "config": asdict(resolved_config),
        "feature_mode": str(problem["feature_mode"]),
        "data_mode": str(metrics.get("data_mode", resolved_config.data_mode)),
        "dataset": str(problem["dataset"]),
        "encoder_state": model.base_encoder.state_dict(),
        "model_state": model.state_dict(),
        "vocab": problem["vocab"],
        "steps": int(steps),
        "metrics": dict(metrics),
        "checkpoint_kind": str(checkpoint_kind),
    }


def _timestamp_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def run_possm_phoneme_finetuning(
    *,
    checkpoint_path: str | Path,
    cache_root: str | Path,
    output_root: str | Path | None = None,
    config: POSSMFinetuneConfig | None = None,
    device: torch.device | None = None,
) -> dict[str, Any]:
    resolved_config = config or POSSMFinetuneConfig()
    _seed_all(int(resolved_config.seed))
    resolved_checkpoint_path = Path(checkpoint_path)
    resolved_cache_root = Path(cache_root)
    resolved_device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base_encoder, pre_decoder_backbone, checkpoint_cfg, stage1_run_dir = recover_possm_stage1_sequence_components(
        checkpoint_path=resolved_checkpoint_path,
        map_location="cpu",
    )
    effective_feature_mode = (
        str(resolved_config.feature_mode)
        if resolved_config.feature_mode is not None
        else str(checkpoint_cfg.get("feature_mode", "tx_sbp"))
    )
    if effective_feature_mode != str(checkpoint_cfg.get("feature_mode", effective_feature_mode)):
        raise ValueError(
            "POSSM stage-2 does not currently support changing feature_mode from the stage-1 checkpoint."
        )
    effective_data_mode = (
        str(resolved_config.data_mode)
        if resolved_config.data_mode is not None
        else str(checkpoint_cfg.get("data_mode", "normalized"))
    )
    effective_boundary_key_mode = (
        str(resolved_config.boundary_key_mode)
        if resolved_config.boundary_key_mode is not None
        else str(checkpoint_cfg.get("boundary_key_mode", "session"))
    )
    effective_config = POSSMFinetuneConfig(
        **{
            **asdict(resolved_config),
            "feature_mode": effective_feature_mode,
            "data_mode": effective_data_mode,
            "boundary_key_mode": effective_boundary_key_mode,
        }
    )

    problem = _build_problem(
        cache_root=resolved_cache_root,
        config=effective_config,
        feature_mode=effective_feature_mode,
        boundary_key_mode=effective_boundary_key_mode,
    )

    if effective_data_mode == "normalized":
        target_stats_mode = "global" if len(problem["target_session_ids"]) == 1 else "per_session"
        target_stats = compute_feature_stats(
            problem["target_train_rows"],
            cache_root=Path(problem["cache_root"]),
            mode=target_stats_mode,
            feature_mode=str(problem["feature_mode"]),
        )
    else:
        target_stats = None

    train_loader = DataLoader(
        CanonicalSequenceDataset(
            problem["target_train_rows"],
            cache_root=Path(problem["cache_root"]),
            stats=target_stats,
            feature_mode=str(problem["feature_mode"]),
            boundary_key_mode=str(problem.get("boundary_key_mode", "session")),
            dataset=str(problem.get("dataset", effective_config.dataset)),
        ),
        **_loader_kwargs(resolved_device, int(effective_config.batch_size), shuffle=True),
        generator=_make_loader_generator(int(effective_config.seed)),
    )
    val_loader = DataLoader(
        CanonicalSequenceDataset(
            problem["target_val_rows"],
            cache_root=Path(problem["cache_root"]),
            stats=target_stats,
            feature_mode=str(problem["feature_mode"]),
            boundary_key_mode=str(problem.get("boundary_key_mode", "session")),
            dataset=str(problem.get("dataset", effective_config.dataset)),
        ),
        **_loader_kwargs(resolved_device, int(effective_config.batch_size), shuffle=False),
        generator=_make_loader_generator(int(effective_config.seed) + 1),
    )

    vocab = dict(problem["vocab"])
    model = POSSMPhonemeModel(
        base_encoder=copy.deepcopy(base_encoder),
        pre_decoder_backbone=copy.deepcopy(pre_decoder_backbone),
        vocab_size=int(vocab["num_classes"]),
        gru_hidden_size=int(effective_config.gru_hidden_size),
        gru_num_layers=int(effective_config.gru_num_layers),
        gru_dropout=float(effective_config.gru_dropout),
        conv_hidden_size=effective_config.conv_hidden_size,
        conv_kernel_size=int(effective_config.conv_kernel_size),
        conv_stride=int(effective_config.conv_stride),
        conv_dropout=float(effective_config.conv_dropout),
    )

    train_encoder = str(effective_config.mode) == "finetune_full"
    for parameter in model.base_encoder.parameters():
        parameter.requires_grad = bool(train_encoder)
    if model.pre_decoder_backbone is not None:
        for parameter in model.pre_decoder_backbone.parameters():
            parameter.requires_grad = bool(train_encoder)
    for parameter in model.gru.parameters():
        parameter.requires_grad = True
    for parameter in model.conv.parameters():
        parameter.requires_grad = True
    for parameter in model.classifier.parameters():
        parameter.requires_grad = True
    model.to(resolved_device)

    trainable_groups: list[dict[str, Any]] = [
        {
            "params": [
                param
                for module in (model.gru, model.conv, model.classifier)
                for param in module.parameters()
                if param.requires_grad
            ],
            "lr": float(effective_config.learning_rate),
        }
    ]
    if train_encoder:
        encoder_params = [param for param in model.base_encoder.parameters() if param.requires_grad]
        if encoder_params:
            trainable_groups.append(
                {"params": encoder_params, "lr": float(effective_config.encoder_learning_rate)}
            )
        temporal_backbone_params = (
            []
            if model.pre_decoder_backbone is None
            else [param for param in model.pre_decoder_backbone.parameters() if param.requires_grad]
        )
        if temporal_backbone_params:
            trainable_groups.append(
                {"params": temporal_backbone_params, "lr": float(effective_config.encoder_learning_rate)}
            )
    optimizer = torch.optim.AdamW(
        trainable_groups,
        lr=float(effective_config.learning_rate),
        weight_decay=float(effective_config.weight_decay),
    )

    if output_root is None:
        base_output_root = stage1_run_dir / "phoneme_finetune"
    else:
        base_output_root = Path(output_root)
    base_output_root.mkdir(parents=True, exist_ok=True)

    run_name = f"possm_stage2_{effective_config.mode}_{effective_feature_mode}_{_timestamp_utc()}"
    run_dir = base_output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    progress_log_path = run_dir / "progress.jsonl"
    summary_path = run_dir / "summary.json"
    checkpoints_dir = run_dir / "checkpoints"
    checkpoint_best_path = run_dir / "checkpoint_best.pt"
    checkpoint_final_path = run_dir / "checkpoint_final.pt"

    start_time = time.time()
    last_report_elapsed = 0.0
    steps = 0
    last_eval_step = 0
    best_metrics: dict[str, Any] | None = None
    best_payload: dict[str, Any] | None = None
    best_step = 0

    def maybe_evaluate_and_checkpoint(*, force: bool = False) -> dict[str, Any] | None:
        nonlocal last_eval_step, best_metrics, best_payload, best_step
        if steps <= 0:
            return None
        should_run = force or steps % int(effective_config.checkpoint_every_steps) == 0
        if not should_run or steps == last_eval_step:
            return None
        metrics = evaluate_possm_phoneme_metrics(
            model=model,
            loader=val_loader,
            device=resolved_device,
            blank_index=int(problem["vocab"]["blank_index"]),
        )
        metrics["model_num_parameters"] = _count_trainable_parameters(model)
        metrics["encoder_num_parameters"] = _count_trainable_sequence_encoder_parameters(model)
        last_eval_step = steps
        _emit_progress(
            progress_log_path,
            event="phoneme_val_report",
            stage="possm_phoneme_finetune",
            step=int(steps),
            elapsed_seconds=round(time.time() - start_time, 3),
            mode=str(effective_config.mode),
            data_mode=effective_data_mode,
            feature_mode=effective_feature_mode,
            **metrics,
        )
        payload = _checkpoint_payload(
            model=model,
            resolved_config=effective_config,
            resolved_checkpoint_path=resolved_checkpoint_path,
            checkpoint_cfg=checkpoint_cfg,
            problem=problem,
            steps=steps,
            metrics=metrics,
            checkpoint_kind="step" if not force else "final_eval",
        )
        if checkpoints_dir is not None and steps % int(effective_config.checkpoint_every_steps) == 0:
            checkpoints_dir.mkdir(parents=True, exist_ok=True)
            torch.save(payload, checkpoints_dir / f"step_{int(steps):06d}.pt")
        if best_metrics is None or float(metrics["val_ctc_bpphone"]) < float(best_metrics["val_ctc_bpphone"]):
            best_metrics = dict(metrics)
            best_payload = payload
            best_step = int(steps)
        return metrics

    while True:
        elapsed = time.time() - start_time
        if elapsed >= float(effective_config.budget_seconds) or steps >= int(effective_config.num_steps):
            break
        made_progress = False
        for batch in train_loader:
            elapsed = time.time() - start_time
            if elapsed >= float(effective_config.budget_seconds) or steps >= int(effective_config.num_steps):
                break
            if train_encoder:
                _set_train_mode(model, train_encoder=True)
            else:
                _set_train_mode(model, train_encoder=False)
            x = batch["x"].to(resolved_device)
            input_lengths = batch["input_lengths"].to(resolved_device)
            labels = batch["labels"].to(resolved_device)
            label_lengths = batch["label_lengths"].to(resolved_device)

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
            loss = loss_sum / target_count
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            clip_params = [param for group in trainable_groups for param in group["params"] if param.requires_grad]
            torch.nn.utils.clip_grad_norm_(clip_params, max_norm=float(effective_config.max_grad_norm))
            optimizer.step()
            steps += 1
            made_progress = True

            elapsed = time.time() - start_time
            should_report = (
                steps == 1
                or steps % int(effective_config.progress_every_steps) == 0
                or elapsed - last_report_elapsed >= float(effective_config.progress_every_seconds)
            )
            if should_report:
                last_report_elapsed = elapsed
                _emit_progress(
                    progress_log_path,
                    event="phoneme_train_report",
                    stage="possm_phoneme_finetune",
                    step=int(steps),
                    elapsed_seconds=round(elapsed, 3),
                    train_ctc_bpphone=float(loss.item()) / math.log(2.0),
                    mode=str(effective_config.mode),
                    data_mode=effective_data_mode,
                    feature_mode=effective_feature_mode,
                )
            maybe_evaluate_and_checkpoint()
        if not made_progress:
            break

    final_metrics = maybe_evaluate_and_checkpoint(force=True)
    if final_metrics is None:
        final_metrics = evaluate_possm_phoneme_metrics(
            model=model,
            loader=val_loader,
            device=resolved_device,
            blank_index=int(problem["vocab"]["blank_index"]),
        )
        final_metrics["model_num_parameters"] = _count_trainable_parameters(model)
        final_metrics["encoder_num_parameters"] = _count_trainable_sequence_encoder_parameters(model)
    assert best_payload is not None
    assert best_metrics is not None
    torch.save(best_payload, checkpoint_best_path)
    torch.save(
        _checkpoint_payload(
            model=model,
            resolved_config=effective_config,
            resolved_checkpoint_path=resolved_checkpoint_path,
            checkpoint_cfg=checkpoint_cfg,
            problem=problem,
            steps=steps,
            metrics=final_metrics,
            checkpoint_kind="final",
        ),
        checkpoint_final_path,
    )

    summary = {
        "run_name": run_name,
        "run_dir": str(run_dir),
        "progress_log_path": str(progress_log_path),
        "checkpoint_best_path": str(checkpoint_best_path),
        "checkpoint_final_path": str(checkpoint_final_path),
        "checkpoints_dir": str(checkpoints_dir),
        "stage1_checkpoint_path": str(resolved_checkpoint_path),
        "stage1_run_dir": str(stage1_run_dir),
        "mode": str(effective_config.mode),
        "feature_mode": effective_feature_mode,
        "data_mode": effective_data_mode,
        "boundary_key_mode": effective_boundary_key_mode,
        "dataset": str(effective_config.dataset),
        "steps": int(steps),
        "metrics": {
            "val_ctc_bpphone": float(final_metrics["val_ctc_bpphone"]),
            "val_phoneme_error_rate": float(final_metrics["val_phoneme_error_rate"]),
            "best_val_ctc_bpphone": float(best_metrics["val_ctc_bpphone"]),
            "best_val_phoneme_error_rate": float(best_metrics["val_phoneme_error_rate"]),
            "best_step": int(best_step),
            "model_num_parameters": int(final_metrics["model_num_parameters"]),
            "encoder_num_parameters": int(final_metrics["encoder_num_parameters"]),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    return summary

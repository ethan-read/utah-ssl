"""Stage-2 phoneme fine-tuning helpers for masked-SSL checkpoints."""

from __future__ import annotations

import argparse
import copy
import json
import math
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .model import S5MaskedEncoder
from .probe import (
    CanonicalSequenceDataset,
    DownstreamProbeConfig,
    LinearCTCProbe,
    _load_probe_metadata_json,
    _make_loader_generator,
    _validate_canonical_probe_assets,
    build_downstream_probe_problem,
    collate_sequence_batch,
    compute_ctc_loss_sum,
    compute_feature_stats,
    evaluate_probe_session_metrics,
)
from .training import _ssl_run_dir_from_checkpoint_path, resolve_ssl_checkpoint_path


def _seed_all(seed: int) -> None:
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


@dataclass
class PhonemeFinetuneConfig:
    seed: int = 7
    mode: str = "finetune_full"
    feature_mode: str = "tx_sbp"
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

    def __post_init__(self) -> None:
        if self.mode not in {"probe_frozen", "finetune_full"}:
            raise ValueError("mode must be one of {'probe_frozen', 'finetune_full'}")
        if self.feature_mode not in {"tx_only", "tx_sbp"}:
            raise ValueError("feature_mode must be one of {'tx_only', 'tx_sbp'}")
        if self.boundary_key_mode is not None and self.boundary_key_mode not in {
            "session",
            "subject_if_available",
        }:
            raise ValueError(
                "boundary_key_mode must be one of {'session', 'subject_if_available'} when provided"
            )
        if int(self.session_limit) <= 0:
            raise ValueError("session_limit must be positive")
        if int(self.target_session_count) <= 0:
            raise ValueError("target_session_count must be positive")
        if int(self.target_session_count) >= int(self.session_limit):
            raise ValueError("target_session_count must be smaller than session_limit")
        if int(self.batch_size) <= 0:
            raise ValueError("batch_size must be positive")
        if int(self.num_steps) <= 0:
            raise ValueError("num_steps must be positive")
        if float(self.budget_seconds) <= 0.0:
            raise ValueError("budget_seconds must be positive")
        if float(self.learning_rate) <= 0.0:
            raise ValueError("learning_rate must be positive")
        if float(self.encoder_learning_rate) <= 0.0:
            raise ValueError("encoder_learning_rate must be positive")
        if float(self.weight_decay) < 0.0:
            raise ValueError("weight_decay must be non-negative")
        if float(self.max_grad_norm) <= 0.0:
            raise ValueError("max_grad_norm must be positive")
        if int(self.checkpoint_every_steps) <= 0:
            raise ValueError("checkpoint_every_steps must be positive")
        if int(self.progress_every_steps) <= 0:
            raise ValueError("progress_every_steps must be positive")
        if float(self.progress_every_seconds) <= 0.0:
            raise ValueError("progress_every_seconds must be positive")


class RawFeatureAdapter(nn.Module):
    """Optional raw-feature adapter for stage-1 tx-only -> stage-2 tx+sbp handoff."""

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.linear = nn.Linear(self.input_dim, self.output_dim, bias=False)
        with torch.no_grad():
            self.linear.weight.zero_()
            diag = min(self.input_dim, self.output_dim)
            self.linear.weight[:diag, :diag] = torch.eye(diag)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class IdentityFeatureAdapter(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.input_dim = int(dim)
        self.output_dim = int(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class AdaptedPhonemeEncoder(nn.Module):
    """Notebook-style encoder interface with an optional raw-feature adapter."""

    def __init__(
        self,
        *,
        base_encoder: S5MaskedEncoder,
        input_adapter: nn.Module,
        external_feature_mode: str,
    ):
        super().__init__()
        self.base_encoder = base_encoder
        self.input_adapter = input_adapter
        self.input_dim = int(getattr(input_adapter, "input_dim", base_encoder.input_dim))
        self.hidden_size = int(base_encoder.hidden_size)
        self.token_dim = int(base_encoder.token_dim)
        self.feature_mode = str(external_feature_mode)
        self.source_session_keys = tuple(base_encoder.source_session_keys)

    def encode(
        self,
        x: torch.Tensor,
        input_lengths: torch.Tensor,
        session_ids: list[str],
        *,
        use_source_affines: bool = True,
        target_affines: Any = None,
    ) -> Any:
        adapted_x = self.input_adapter(x)
        tokens, token_lengths = self.base_encoder.patch_batch(adapted_x, input_lengths)
        outputs = self.base_encoder.encode_patched(
            tokens,
            token_lengths,
            token_mask=None,
            session_keys=session_ids,
            use_source_affines=bool(use_source_affines),
            target_affines=target_affines,
        )
        return type(
            "EncoderOutputs",
            (),
            {
                "hidden": outputs["hidden"],
                "token_lengths": outputs["token_lengths"],
                "tokens": outputs["tokens"],
            },
        )()


def _loader_kwargs(device: torch.device, batch_size: int, *, shuffle: bool) -> dict[str, Any]:
    return {
        "batch_size": int(batch_size),
        "shuffle": shuffle,
        "num_workers": 0,
        "pin_memory": device.type == "cuda",
        "collate_fn": collate_sequence_batch,
    }


def _emit_progress(progress_log_path: Path | None, **payload: Any) -> None:
    if progress_log_path is None:
        return
    progress_log_path.parent.mkdir(parents=True, exist_ok=True)
    with progress_log_path.open("a") as handle:
        handle.write(json.dumps(payload) + "\n")


def _count_trainable_parameters(module: nn.Module) -> int:
    return int(sum(param.numel() for param in module.parameters() if param.requires_grad))


def _checkpoint_payload(
    *,
    encoder: AdaptedPhonemeEncoder,
    phoneme_head: LinearCTCProbe,
    resolved_config: PhonemeFinetuneConfig,
    resolved_checkpoint_path: Path,
    checkpoint_cfg: dict[str, Any],
    problem: dict[str, Any],
    external_input_dim: int,
    steps: int,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    return {
        "stage": "phoneme_finetune",
        "ssl_checkpoint_path": str(resolved_checkpoint_path),
        "ssl_checkpoint_config": dict(checkpoint_cfg),
        "config": asdict(resolved_config),
        "feature_mode": str(resolved_config.feature_mode),
        "external_input_dim": int(external_input_dim),
        "encoder_input_dim": int(encoder.base_encoder.input_dim),
        "input_adapter_state": encoder.input_adapter.state_dict(),
        "encoder_state": encoder.base_encoder.state_dict(),
        "phoneme_head_state": phoneme_head.state_dict(),
        "vocab": problem["vocab"],
        "steps": int(steps),
        "metrics": dict(metrics),
    }


def _recover_stage1_encoder(
    *,
    checkpoint_path: Path,
    map_location: str | torch.device = "cpu",
) -> tuple[S5MaskedEncoder, dict[str, Any], Path]:
    payload = torch.load(checkpoint_path, map_location=map_location)
    checkpoint_cfg = dict(payload.get("config", {}))
    required_keys = [
        "input_dim",
        "patch_size",
        "patch_stride",
        "hidden_size",
        "s5_state_size",
        "num_layers",
        "dropout",
    ]
    missing = [key for key in required_keys if key not in checkpoint_cfg]
    if missing:
        raise KeyError(
            f"Checkpoint config is missing keys needed for phoneme fine-tuning: {missing}"
        )

    base_encoder = S5MaskedEncoder(
        input_dim=int(checkpoint_cfg["input_dim"]),
        hidden_size=int(checkpoint_cfg["hidden_size"]),
        s5_state_size=int(checkpoint_cfg["s5_state_size"]),
        num_layers=int(checkpoint_cfg["num_layers"]),
        dropout=float(checkpoint_cfg["dropout"]),
        patch_size=int(checkpoint_cfg["patch_size"]),
        patch_stride=int(checkpoint_cfg["patch_stride"]),
        post_proj_norm=str(checkpoint_cfg.get("post_proj_norm", "rms")),
        source_session_keys=tuple(checkpoint_cfg.get("source_session_keys", ())),
        feature_mode=str(checkpoint_cfg.get("feature_mode", "tx_only")),
    )
    model_state = payload.get("model_state")
    if model_state is None:
        raise KeyError("Stage-1 checkpoint is missing 'model_state'.")
    encoder_state = {
        key.split("encoder.", 1)[1]: value
        for key, value in model_state.items()
        if key.startswith("encoder.")
    }
    if not encoder_state:
        raise KeyError("Stage-1 checkpoint does not contain encoder weights.")
    base_encoder.load_state_dict(encoder_state)
    return base_encoder, checkpoint_cfg, _ssl_run_dir_from_checkpoint_path(checkpoint_path)


def _build_input_adapter(
    *,
    external_input_dim: int,
    encoder_input_dim: int,
) -> nn.Module:
    if int(external_input_dim) == int(encoder_input_dim):
        return IdentityFeatureAdapter(int(external_input_dim))
    return RawFeatureAdapter(int(external_input_dim), int(encoder_input_dim))


def _build_problem(
    *,
    cache_root: Path,
    config: PhonemeFinetuneConfig,
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
        feature_mode=str(config.feature_mode),
        boundary_key_mode=(
            str(config.boundary_key_mode) if config.boundary_key_mode is not None else "session"
        ),
    )


def _train_one_stage(
    *,
    encoder: AdaptedPhonemeEncoder,
    phoneme_head: LinearCTCProbe,
    problem: dict[str, Any],
    config: PhonemeFinetuneConfig,
    device: torch.device,
    progress_log_path: Path | None,
    checkpoint_dir: Path | None,
    checkpoint_payload_fn: Any,
) -> tuple[dict[str, Any], int, dict[str, Any], dict[str, Any], int]:
    target_stats_mode = "global" if len(problem["target_session_ids"]) == 1 else "per_session"
    target_stats = compute_feature_stats(
        problem["target_train_rows"],
        cache_root=Path(problem["cache_root"]),
        mode=target_stats_mode,
        feature_mode=str(problem["feature_mode"]),
    )
    train_loader = DataLoader(
        CanonicalSequenceDataset(
            problem["target_train_rows"],
            cache_root=Path(problem["cache_root"]),
            stats=target_stats,
            feature_mode=str(problem["feature_mode"]),
            boundary_key_mode=str(problem.get("boundary_key_mode", "session")),
        ),
        **_loader_kwargs(device, int(config.batch_size), shuffle=True),
        generator=_make_loader_generator(int(config.seed)),
    )
    val_loader = DataLoader(
        CanonicalSequenceDataset(
            problem["target_val_rows"],
            cache_root=Path(problem["cache_root"]),
            stats=target_stats,
            feature_mode=str(problem["feature_mode"]),
            boundary_key_mode=str(problem.get("boundary_key_mode", "session")),
        ),
        **_loader_kwargs(device, int(config.batch_size), shuffle=False),
        generator=_make_loader_generator(int(config.seed) + 1),
    )

    train_encoder = str(config.mode) == "finetune_full"
    for parameter in encoder.base_encoder.parameters():
        parameter.requires_grad = bool(train_encoder)
    for parameter in encoder.input_adapter.parameters():
        parameter.requires_grad = bool(train_encoder)
    for parameter in phoneme_head.parameters():
        parameter.requires_grad = True

    encoder.to(device)
    phoneme_head.to(device)

    trainable_groups: list[dict[str, Any]] = [
        {
            "params": [param for param in phoneme_head.parameters() if param.requires_grad],
            "lr": float(config.learning_rate),
        }
    ]
    if train_encoder:
        encoder_group_params = [param for param in encoder.base_encoder.parameters() if param.requires_grad]
        adapter_params = [param for param in encoder.input_adapter.parameters() if param.requires_grad]
        if adapter_params:
            trainable_groups.append(
                {
                    "params": adapter_params,
                    "lr": float(config.encoder_learning_rate),
                }
            )
        if encoder_group_params:
            trainable_groups.append(
                {
                    "params": encoder_group_params,
                    "lr": float(config.encoder_learning_rate),
                }
            )

    optimizer = torch.optim.AdamW(
        trainable_groups,
        lr=float(config.learning_rate),
        weight_decay=float(config.weight_decay),
    )

    start_time = time.time()
    last_report_elapsed = 0.0
    steps = 0
    last_eval_step = 0
    best_metrics: dict[str, Any] | None = None
    best_payload: dict[str, Any] | None = None
    best_step = 0

    def maybe_evaluate_and_checkpoint(*, force: bool = False) -> None:
        nonlocal last_eval_step, best_metrics, best_payload, best_step
        if steps <= 0:
            return
        should_run = force or steps % int(config.checkpoint_every_steps) == 0
        if not should_run or steps == last_eval_step:
            return

        metrics = evaluate_probe_session_metrics(
            encoder=encoder,
            probe_head=phoneme_head,
            target_affines=None,
            loader=val_loader,
            device=device,
            blank_index=int(problem["vocab"]["blank_index"]),
        )
        metrics["phoneme_head_num_parameters"] = _count_trainable_parameters(phoneme_head)
        metrics["input_adapter_num_parameters"] = _count_trainable_parameters(encoder.input_adapter)
        metrics["encoder_num_parameters"] = _count_trainable_parameters(encoder.base_encoder)
        last_eval_step = steps

        _emit_progress(
            progress_log_path,
            event="phoneme_val_report",
            stage="phoneme_finetune",
            step=int(steps),
            elapsed_seconds=round(time.time() - start_time, 3),
            mode=str(config.mode),
            seed=int(config.seed),
            feature_mode=str(config.feature_mode),
            **metrics,
        )

        payload = checkpoint_payload_fn(steps=steps, metrics=metrics)
        if checkpoint_dir is not None and steps % int(config.checkpoint_every_steps) == 0:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            torch.save(payload, checkpoint_dir / f"step_{int(steps):06d}.pt")

        if best_metrics is None or float(metrics["val_ctc_bpphone"]) < float(best_metrics["val_ctc_bpphone"]):
            best_metrics = dict(metrics)
            best_payload = payload
            best_step = int(steps)

    while True:
        elapsed = time.time() - start_time
        if elapsed >= float(config.budget_seconds):
            break
        if steps >= int(config.num_steps):
            break

        made_progress = False
        for batch in train_loader:
            elapsed = time.time() - start_time
            if elapsed >= float(config.budget_seconds) or steps >= int(config.num_steps):
                break

            if train_encoder:
                encoder.train()
            else:
                encoder.eval()
            phoneme_head.train()

            x = batch["x"].to(device)
            input_lengths = batch["input_lengths"].to(device)
            labels = batch["labels"].to(device)
            label_lengths = batch["label_lengths"].to(device)

            outputs = encoder.encode(
                x,
                input_lengths,
                batch["boundary_keys"],
            )
            logits = phoneme_head(outputs.hidden)
            loss_sum, target_count = compute_ctc_loss_sum(
                logits,
                outputs.token_lengths,
                labels,
                label_lengths,
                blank_index=int(problem["vocab"]["blank_index"]),
            )
            if target_count <= 0:
                continue

            loss = loss_sum / target_count
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            clip_params = [param for group in trainable_groups for param in group["params"]]
            torch.nn.utils.clip_grad_norm_(clip_params, max_norm=float(config.max_grad_norm))
            optimizer.step()
            steps += 1
            made_progress = True

            elapsed = time.time() - start_time
            should_report = (
                steps == 1
                or steps % int(config.progress_every_steps) == 0
                or elapsed - last_report_elapsed >= float(config.progress_every_seconds)
            )
            if should_report:
                last_report_elapsed = elapsed
                _emit_progress(
                    progress_log_path,
                    event="phoneme_train_report",
                    stage="phoneme_finetune",
                    step=int(steps),
                    elapsed_seconds=round(elapsed, 3),
                    train_ctc_bpphone=float(loss.item()) / math.log(2.0),
                    mode=str(config.mode),
                    seed=int(config.seed),
                    feature_mode=str(config.feature_mode),
                )
            maybe_evaluate_and_checkpoint()

        if not made_progress:
            break

    maybe_evaluate_and_checkpoint(force=True)
    assert best_metrics is not None
    assert best_payload is not None

    final_metrics = evaluate_probe_session_metrics(
        encoder=encoder,
        probe_head=phoneme_head,
        target_affines=None,
        loader=val_loader,
        device=device,
        blank_index=int(problem["vocab"]["blank_index"]),
    )
    final_metrics["phoneme_head_num_parameters"] = _count_trainable_parameters(phoneme_head)
    final_metrics["input_adapter_num_parameters"] = _count_trainable_parameters(encoder.input_adapter)
    final_metrics["encoder_num_parameters"] = _count_trainable_parameters(encoder.base_encoder)
    _emit_progress(
        progress_log_path,
        event="phoneme_session_complete",
        stage="phoneme_finetune",
        step=int(steps),
        elapsed_seconds=round(time.time() - start_time, 3),
        mode=str(config.mode),
        seed=int(config.seed),
        feature_mode=str(config.feature_mode),
        **final_metrics,
    )
    return final_metrics, steps, best_metrics, best_payload, best_step


def _timestamp_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def run_phoneme_finetuning(
    *,
    checkpoint_path: str | Path,
    cache_root: str | Path,
    output_root: str | Path | None = None,
    config: PhonemeFinetuneConfig | None = None,
    device: torch.device | None = None,
) -> dict[str, Any]:
    resolved_config = config or PhonemeFinetuneConfig()
    _seed_all(int(resolved_config.seed))

    resolved_checkpoint_path = Path(checkpoint_path)
    resolved_cache_root = Path(cache_root)
    resolved_device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base_encoder, checkpoint_cfg, stage1_run_dir = _recover_stage1_encoder(
        checkpoint_path=resolved_checkpoint_path,
        map_location="cpu",
    )
    effective_boundary_key_mode = (
        str(resolved_config.boundary_key_mode)
        if resolved_config.boundary_key_mode is not None
        else str(checkpoint_cfg.get("boundary_key_mode", "session"))
    )
    effective_config = PhonemeFinetuneConfig(
        **{
            **asdict(resolved_config),
            "boundary_key_mode": effective_boundary_key_mode,
        }
    )
    problem = _build_problem(
        cache_root=resolved_cache_root,
        config=effective_config,
    )

    metadata = _load_probe_metadata_json(problem["metadata_path"])
    external_input_dim = int(metadata["n_tx_features"])
    if str(resolved_config.feature_mode) == "tx_sbp":
        external_input_dim += int(metadata["n_sbp_features"])

    input_adapter = _build_input_adapter(
        external_input_dim=external_input_dim,
        encoder_input_dim=int(base_encoder.input_dim),
    )
    encoder = AdaptedPhonemeEncoder(
        base_encoder=copy.deepcopy(base_encoder),
        input_adapter=input_adapter,
        external_feature_mode=str(resolved_config.feature_mode),
    )
    phoneme_head = LinearCTCProbe(
        hidden_size=int(encoder.hidden_size),
        vocab_size=int(problem["vocab"]["num_classes"]),
    )

    if output_root is None:
        base_output_root = stage1_run_dir / "phoneme_finetune"
    else:
        base_output_root = Path(output_root)
    base_output_root.mkdir(parents=True, exist_ok=True)

    run_name = (
        f"stage2_phoneme_{resolved_config.mode}_{resolved_config.feature_mode}_{_timestamp_utc()}"
    )
    run_dir = base_output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    progress_log_path = run_dir / "progress.jsonl"
    summary_path = run_dir / "summary.json"
    checkpoints_dir = run_dir / "checkpoints"
    checkpoint_best_path = run_dir / "checkpoint_best.pt"
    checkpoint_final_path = run_dir / "checkpoint_final.pt"

    checkpoint_payload_fn = lambda *, steps, metrics: _checkpoint_payload(
        encoder=encoder,
        phoneme_head=phoneme_head,
        resolved_config=effective_config,
        resolved_checkpoint_path=resolved_checkpoint_path,
        checkpoint_cfg=checkpoint_cfg,
        problem=problem,
        external_input_dim=external_input_dim,
        steps=steps,
        metrics=metrics,
    )

    metrics, steps, best_metrics, best_payload, best_step = _train_one_stage(
        encoder=encoder,
        phoneme_head=phoneme_head,
        problem=problem,
        config=effective_config,
        device=resolved_device,
        progress_log_path=progress_log_path,
        checkpoint_dir=checkpoints_dir,
        checkpoint_payload_fn=checkpoint_payload_fn,
    )

    torch.save(best_payload, checkpoint_best_path)
    torch.save(checkpoint_payload_fn(steps=steps, metrics=metrics), checkpoint_final_path)

    summary = {
        "run_name": run_name,
        "run_dir": str(run_dir),
        "progress_log_path": str(progress_log_path),
        "checkpoint_best_path": str(checkpoint_best_path),
        "checkpoint_final_path": str(checkpoint_final_path),
        "checkpoints_dir": str(checkpoints_dir),
        "ssl_checkpoint_path": str(resolved_checkpoint_path),
        "ssl_run_dir": str(stage1_run_dir),
        "mode": str(resolved_config.mode),
        "feature_mode": str(resolved_config.feature_mode),
        "boundary_key_mode": str(effective_config.boundary_key_mode),
        "checkpoint_every_steps": int(effective_config.checkpoint_every_steps),
        "external_input_dim": int(external_input_dim),
        "encoder_input_dim": int(base_encoder.input_dim),
        "adapter_type": type(encoder.input_adapter).__name__,
        "selected_session_bases": [
            entry.session_base for entry in problem["split"].train + problem["split"].val
        ],
        "source_session_ids": list(problem["source_session_ids"]),
        "target_session_ids": list(problem["target_session_ids"]),
        "target_train_examples": len(problem["target_train_rows"]),
        "target_val_examples": len(problem["target_val_rows"]),
        "target_train_examples_by_session": dict(problem["target_train_examples_by_session"]),
        "target_val_examples_by_session": dict(problem["target_val_examples_by_session"]),
        "steps": int(steps),
        "metrics": {
            "val_ctc_bpphone": float(metrics["val_ctc_bpphone"]),
            "val_phoneme_error_rate": float(metrics["val_phoneme_error_rate"]),
            "best_val_ctc_bpphone": float(best_metrics["val_ctc_bpphone"]),
            "best_val_phoneme_error_rate": float(best_metrics["val_phoneme_error_rate"]),
            "best_step": int(best_step),
            "phoneme_head_num_parameters": int(metrics["phoneme_head_num_parameters"]),
            "input_adapter_num_parameters": int(metrics["input_adapter_num_parameters"]),
            "encoder_num_parameters": int(metrics["encoder_num_parameters"]),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage-2 phoneme fine-tuning for masked-SSL checkpoints")
    parser.add_argument("--checkpoint-path", default=None)
    parser.add_argument("--ssl-output-root", default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--mode", choices=["probe_frozen", "finetune_full"], default="finetune_full")
    parser.add_argument("--feature-mode", choices=["tx_only", "tx_sbp"], default="tx_sbp")
    parser.add_argument(
        "--boundary-key-mode",
        choices=["session", "subject_if_available"],
        default=None,
    )
    parser.add_argument("--session-limit", type=int, default=8)
    parser.add_argument("--target-session-count", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-steps", type=int, default=400)
    parser.add_argument("--budget-seconds", type=int, default=240)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--encoder-learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--checkpoint-every-steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.checkpoint_path is None and args.ssl_output_root is None:
        raise SystemExit(
            "Provide either --checkpoint-path or --ssl-output-root for stage-1 checkpoint discovery."
        )
    checkpoint_path = resolve_ssl_checkpoint_path(
        output_root=Path(args.ssl_output_root) if args.ssl_output_root is not None else Path.cwd(),
        explicit_checkpoint_path=args.checkpoint_path,
    )
    config = PhonemeFinetuneConfig(
        seed=int(args.seed),
        mode=str(args.mode),
        feature_mode=str(args.feature_mode),
        boundary_key_mode=args.boundary_key_mode,
        session_limit=int(args.session_limit),
        target_session_count=int(args.target_session_count),
        batch_size=int(args.batch_size),
        num_steps=int(args.num_steps),
        budget_seconds=int(args.budget_seconds),
        learning_rate=float(args.learning_rate),
        encoder_learning_rate=float(args.encoder_learning_rate),
        weight_decay=float(args.weight_decay),
        checkpoint_every_steps=int(args.checkpoint_every_steps),
    )
    summary = run_phoneme_finetuning(
        checkpoint_path=checkpoint_path,
        cache_root=Path(args.cache_root),
        output_root=Path(args.output_root) if args.output_root is not None else None,
        config=config,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

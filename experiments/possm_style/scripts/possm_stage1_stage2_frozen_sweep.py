"""Run POSSM Stage-1 variants and frozen-encoder Stage-2 evaluations.

This Colab-oriented script trains a small Stage-1 architecture/objective sweep,
then evaluates each resulting encoder with Stage-2 phoneme decoding in
``probe_frozen`` mode on the released Willett ``competition_train ->
competition_test`` split. It is designed to be safe to rerun after disconnects.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import traceback
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from experiments.possm_style import (
    CacheAccessConfig,
    POSSMFinetuneConfig,
    POSSMTrainingConfig,
    SUPPORTED_FEATURE_MODES,
    SignalSpec,
    possm_single_dataset_plan,
    prepare_cache_context,
    recover_possm_run_state_from_checkpoint,
    resume_possm_training,
    run_possm_phoneme_finetuning,
    run_possm_training,
)
from experiments.possm_style.training import _parse_step_from_checkpoint_name, _serialize_config
from experiments.possm_style.scripts.possm_stage2_hyperparam_sweep import (
    DEFAULT_DRIVE_ROOT,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_STAGE2_CACHE_ROOT,
    flatten_summary_row,
    install_stdout_progress_hook,
    load_json,
    make_logger,
    write_json,
    write_text_atomic,
)


DEFAULT_STAGE1_CACHE_ROOT = DEFAULT_DRIVE_ROOT / "utah_ssl" / "data" / "cache_v1_smoothed_sigma2p0"
DEFAULT_STAGE1_LOCAL_CACHE_BASE = "/content/utah_ssl_cache"


def timestamp_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def make_stage1_base_config(args: argparse.Namespace) -> POSSMTrainingConfig:
    return POSSMTrainingConfig(
        signal_spec=SignalSpec.from_mode(
            args.feature_mode,
            tx_dim=int(args.tx_dim),
            sbp_dim=int(args.sbp_dim),
        ),
        seed=int(args.seed),
        data_mode=str(args.stage1_data_mode),
        boundary_key_mode=str(args.boundary_key_mode),
        segment_bins=int(args.segment_bins),
        model_dim=int(args.model_dim),
        latent_count=int(args.latent_count),
        value_encoder_type=str(args.value_encoder_type),
        value_mlp_hidden_size=args.value_mlp_hidden_size,
        ffn_hidden_size=int(args.ffn_hidden_size),
        dropout=float(args.stage1_dropout),
        use_token_norm=bool(args.use_token_norm),
        batch_size=int(args.stage1_batch_size),
        num_steps=int(args.stage1_steps),
        learning_rate=float(args.stage1_learning_rate),
        weight_decay=float(args.stage1_weight_decay),
        val_every=int(args.stage1_val_every),
        val_batches=int(args.stage1_val_batches),
        checkpoint_every_steps=int(args.stage1_checkpoint_every_steps),
        dataset_weight_alpha=float(args.dataset_weight_alpha),
        examples_per_shard=int(args.examples_per_shard),
        log_every=int(args.stage1_log_every),
    )


def make_stage2_config(args: argparse.Namespace) -> POSSMFinetuneConfig:
    return POSSMFinetuneConfig(
        seed=int(args.seed),
        mode="probe_frozen",
        dataset=str(args.dataset),
        data_mode=str(args.stage2_data_mode),
        boundary_key_mode=str(args.boundary_key_mode),
        batch_size=int(args.stage2_batch_size),
        num_steps=int(args.stage2_steps),
        learning_rate=float(args.stage2_learning_rate),
        encoder_learning_rate=float(args.stage2_encoder_learning_rate),
        weight_decay=float(args.stage2_weight_decay),
        max_grad_norm=float(args.stage2_max_grad_norm),
        checkpoint_every_steps=int(args.stage2_checkpoint_every_steps),
        progress_every_steps=int(args.stage2_progress_every_steps),
        session_adapter_enabled=bool(args.session_adapter_enabled),
        input_smoothing_sigma_bins=float(args.input_smoothing_sigma_bins),
        input_smoothing_kernel_size=int(args.input_smoothing_kernel_size),
        input_smoothing_threshold=float(args.input_smoothing_threshold),
        white_noise_sd=float(args.white_noise_sd),
        constant_offset_sd=float(args.constant_offset_sd),
        gru_hidden_size=int(args.gru_hidden_size),
        gru_num_layers=int(args.gru_num_layers),
        gru_dropout=float(args.gru_dropout),
        conv_kernel_size=int(args.conv_kernel_size),
        conv_stride=int(args.conv_stride),
        conv_dropout=float(args.conv_dropout),
    )


def make_stage1_variants(base: POSSMTrainingConfig) -> dict[str, POSSMTrainingConfig]:
    payload = asdict(base)

    def build(**overrides: Any) -> POSSMTrainingConfig:
        config_payload = dict(payload)
        config_payload.update(overrides)
        return POSSMTrainingConfig(**config_payload)

    return {
        "identity_plain_linear": build(
            temporal_backbone_type="identity",
            stage1_objective_type="plain_mse",
            masking_type="none",
            mask_prob=0.0,
            reconstruction_head_type="linear",
            reconstruction_mlp_hidden_size=None,
        ),
        "identity_channel_mask_linear": build(
            temporal_backbone_type="identity",
            stage1_objective_type="masked_mse",
            masking_type="channel",
            mask_prob=0.15,
            mask_replace_mode="zero",
            reconstruction_head_type="linear",
            reconstruction_mlp_hidden_size=None,
        ),
        "identity_span_mask_linear": build(
            temporal_backbone_type="identity",
            stage1_objective_type="masked_mse",
            masking_type="span",
            mask_prob=0.15,
            mask_span_bins=8,
            mask_replace_mode="zero",
            reconstruction_head_type="linear",
            reconstruction_mlp_hidden_size=None,
        ),
        "gru1_plain_linear": build(
            temporal_backbone_type="gru",
            temporal_gru_num_layers=1,
            temporal_gru_dropout=0.0,
            temporal_gru_bidirectional=False,
            stage1_objective_type="plain_mse",
            masking_type="none",
            mask_prob=0.0,
            reconstruction_head_type="linear",
            reconstruction_mlp_hidden_size=None,
        ),
        "identity_plain_mlp_head": build(
            temporal_backbone_type="identity",
            stage1_objective_type="plain_mse",
            masking_type="none",
            mask_prob=0.0,
            reconstruction_head_type="mlp",
            reconstruction_mlp_hidden_size=512,
        ),
    }


def stage1_variant_dirs(output_root: Path, sweep_name: str, variant_name: str) -> tuple[Path, Path]:
    variant_root = Path(output_root) / "stage1_sweeps" / sweep_name / variant_name
    return variant_root, variant_root / "run"


def stage2_variant_root(output_root: Path, sweep_name: str, variant_name: str) -> Path:
    return Path(output_root) / "phoneme_finetune" / sweep_name / variant_name


def stage2_run_dir(output_root: Path, sweep_name: str, variant_name: str) -> Path:
    return stage2_variant_root(output_root, sweep_name, variant_name) / "run"


def read_torch_payload(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise TypeError(f"Expected checkpoint payload dict at {path}")
    return payload


def validate_stage1_payload(
    checkpoint_path: Path,
    *,
    expected_config: dict[str, Any],
    requested_steps: int,
) -> dict[str, Any]:
    payload = read_torch_payload(checkpoint_path)
    checkpoint_config = dict(payload.get("config", {}))
    if str(payload.get("model_family", checkpoint_config.get("model_family", ""))) != "possm":
        raise ValueError(f"Checkpoint is not a POSSM checkpoint: {checkpoint_path}")
    if str(checkpoint_config.get("stage", payload.get("stage", ""))) != "stage1_reconstruction":
        raise ValueError(f"Checkpoint is not a POSSM Stage-1 checkpoint: {checkpoint_path}")

    expected_without_steps = {**dict(expected_config), "num_steps": checkpoint_config.get("num_steps")}
    if checkpoint_config != expected_without_steps:
        raise ValueError(
            "Stage-1 checkpoint config does not match requested variant config: "
            f"{checkpoint_path}"
        )
    if int(payload.get("step", 0)) > int(requested_steps):
        raise ValueError(
            f"Stage-1 checkpoint step exceeds requested steps: {checkpoint_path} "
            f"step={payload.get('step')} requested={requested_steps}"
        )
    return payload


def latest_stage1_resume_checkpoint(run_dir: Path) -> Path | None:
    candidates: list[Path] = []
    final_path = run_dir / "checkpoint_final.pt"
    if final_path.exists():
        candidates.append(final_path)
    checkpoints_dir = run_dir / "checkpoints"
    if checkpoints_dir.exists():
        candidates.extend(checkpoints_dir.glob("step_*.pt"))
        candidates.extend(checkpoints_dir.glob("checkpoint_final_step_*.pt"))
    if not candidates:
        return None

    def sort_key(path: Path) -> tuple[int, int]:
        step = _parse_step_from_checkpoint_name(path.name)
        if path.name == "checkpoint_final.pt":
            try:
                step = int(read_torch_payload(path).get("step", step or 0))
            except Exception:
                step = step or 0
        return int(step or 0), int(path.stat().st_mtime_ns)

    return max(candidates, key=sort_key)


def completed_stage1_result(
    run_dir: Path,
    *,
    expected_config: dict[str, Any],
    requested_steps: int,
) -> dict[str, Any] | None:
    final_path = run_dir / "checkpoint_final.pt"
    best_path = run_dir / "checkpoint_best.pt"
    if not final_path.exists() or not best_path.exists():
        return None
    final_payload = validate_stage1_payload(
        final_path,
        expected_config=expected_config,
        requested_steps=requested_steps,
    )
    best_payload = validate_stage1_payload(
        best_path,
        expected_config=expected_config,
        requested_steps=requested_steps,
    )
    if int(final_payload.get("step", 0)) < int(requested_steps):
        return None
    return {
        "run_dir": str(run_dir),
        "checkpoint_final_path": str(final_path),
        "checkpoint_best_path": str(best_path),
        "steps": int(final_payload.get("step", 0)),
        "best_step": best_payload.get("step"),
        "best_score": best_payload.get("best_score"),
    }


def validate_stage2_final(
    run_dir: Path,
    *,
    stage1_checkpoint_path: Path,
    stage2_config: POSSMFinetuneConfig,
) -> dict[str, Any] | None:
    summary_path = run_dir / "summary.json"
    final_path = run_dir / "checkpoint_final.pt"
    if not summary_path.exists() or not final_path.exists():
        return None
    summary = json.loads(summary_path.read_text())
    payload = read_torch_payload(final_path)
    if str(payload.get("stage", "")) != "stage2_phoneme_finetune":
        raise ValueError(f"Checkpoint is not a Stage-2 phoneme checkpoint: {final_path}")
    if str(payload.get("stage1_checkpoint_path", "")) != str(stage1_checkpoint_path):
        raise ValueError(f"Stage-2 checkpoint uses a different Stage-1 checkpoint: {final_path}")
    if dict(payload.get("config", {})) != asdict(stage2_config):
        raise ValueError(f"Stage-2 checkpoint config does not match requested config: {final_path}")
    return summary


def make_stage1_cache_context(args: argparse.Namespace):
    config = CacheAccessConfig(
        dataset_plan=possm_single_dataset_plan(str(args.dataset)),
        signal_spec=SignalSpec.from_mode(
            args.feature_mode,
            tx_dim=int(args.tx_dim),
            sbp_dim=int(args.sbp_dim),
        ),
        mode=str(args.stage1_cache_mode),
        local_cache_base=str(args.stage1_local_cache_base),
        force_recopy_local_cache=bool(args.force_recopy_local_cache),
        seed=int(args.seed),
        segment_bins=int(args.segment_bins),
        use_normalization=str(args.stage1_data_mode) == "normalized",
        examples_per_shard=int(args.examples_per_shard),
        boundary_key_mode=str(args.boundary_key_mode),
        gaussian_smoothing_sigma_bins=0.0,
        precomputed_session_stats_path=args.stage1_session_stats_path,
    )
    return prepare_cache_context(cache_candidates=[Path(args.stage1_cache_root)], config=config)


def write_rows_json(path: Path, rows: list[dict[str, Any]]) -> None:
    write_text_atomic(path, json.dumps(rows, indent=2))


def write_combined_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    tmp_path = path.with_name(f".{path.name}.tmp")
    with tmp_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    tmp_path.replace(path)


def ensure_fresh_run_dir(run_dir: Path, *, stage: str, variant: str) -> None:
    if not run_dir.exists():
        return
    if any(run_dir.iterdir()):
        raise ValueError(
            f"{stage} run directory already has files for variant {variant!r}: {run_dir}. "
            "Use --resume to validate/reuse it, or choose a new --sweep-name."
        )


def mark_status(state: dict[str, Any], stage: str, variant: str, status: str) -> None:
    status_key = f"{stage}_status_by_variant"
    state.setdefault(status_key, {})
    state[status_key][variant] = status
    state["updated_utc"] = datetime.now(timezone.utc).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-name", default=None)
    parser.add_argument("--stage1-cache-root", type=Path, default=DEFAULT_STAGE1_CACHE_ROOT)
    parser.add_argument("--stage2-cache-root", type=Path, default=DEFAULT_STAGE2_CACHE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--stage1-steps", type=int, default=4000)
    parser.add_argument("--stage2-steps", type=int, default=3000)
    parser.add_argument("--variant", action="append", default=None)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--dry-run", action="store_true")

    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--dataset", default="brain2text24")
    parser.add_argument(
        "--feature-mode",
        choices=SUPPORTED_FEATURE_MODES,
        required=True,
    )
    parser.add_argument("--boundary-key-mode", choices=("session", "subject_if_available"), default="session")
    parser.add_argument("--segment-bins", type=int, default=40)
    parser.add_argument("--model-dim", type=int, default=64)
    parser.add_argument("--latent-count", type=int, default=4)
    parser.add_argument("--value-encoder-type", choices=("linear", "mlp"), default="linear")
    parser.add_argument("--value-mlp-hidden-size", type=int, default=None)
    parser.add_argument("--ffn-hidden-size", type=int, default=512)
    parser.add_argument("--use-token-norm", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--tx-dim", type=int, default=128)
    parser.add_argument("--sbp-dim", type=int, default=128)
    parser.add_argument("--stage1-data-mode", choices=("raw", "normalized"), default="normalized")
    parser.add_argument("--stage1-cache-mode", choices=("copy_to_local", "drive_direct"), default="drive_direct")
    parser.add_argument("--stage1-local-cache-base", default=DEFAULT_STAGE1_LOCAL_CACHE_BASE)
    parser.add_argument("--force-recopy-local-cache", action="store_true")
    parser.add_argument("--stage1-session-stats-path", type=Path, default=None)
    parser.add_argument("--stage1-dropout", type=float, default=0.15)
    parser.add_argument("--stage1-batch-size", type=int, default=32)
    parser.add_argument("--stage1-learning-rate", type=float, default=3e-4)
    parser.add_argument("--stage1-weight-decay", type=float, default=1e-3)
    parser.add_argument("--stage1-val-every", type=int, default=50)
    parser.add_argument("--stage1-val-batches", type=int, default=2)
    parser.add_argument("--stage1-checkpoint-every-steps", type=int, default=500)
    parser.add_argument("--dataset-weight-alpha", type=float, default=0.25)
    parser.add_argument("--examples-per-shard", type=int, default=8)
    parser.add_argument("--stage1-log-every", type=int, default=20)

    parser.add_argument("--stage2-data-mode", choices=("raw", "normalized"), default="normalized")
    parser.add_argument("--stage2-batch-size", type=int, default=32)
    parser.add_argument("--stage2-learning-rate", type=float, default=2e-4)
    parser.add_argument("--stage2-encoder-learning-rate", type=float, default=3e-5)
    parser.add_argument("--stage2-weight-decay", type=float, default=1e-2)
    parser.add_argument("--stage2-max-grad-norm", type=float, default=1.0)
    parser.add_argument("--stage2-checkpoint-every-steps", type=int, default=100)
    parser.add_argument("--stage2-progress-every-steps", type=int, default=20)
    parser.add_argument("--session-adapter-enabled", action="store_true")
    parser.add_argument("--input-smoothing-sigma-bins", type=float, default=2.0)
    parser.add_argument("--input-smoothing-kernel-size", type=int, default=100)
    parser.add_argument("--input-smoothing-threshold", type=float, default=0.01)
    parser.add_argument("--white-noise-sd", type=float, default=0.1)
    parser.add_argument("--constant-offset-sd", type=float, default=0.05)
    parser.add_argument("--gru-hidden-size", type=int, default=768)
    parser.add_argument("--gru-num-layers", type=int, default=5)
    parser.add_argument("--gru-dropout", type=float, default=0.2)
    parser.add_argument("--conv-kernel-size", type=int, default=14)
    parser.add_argument("--conv-stride", type=int, default=4)
    parser.add_argument("--conv-dropout", type=float, default=0.1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sweep_name = args.sweep_name or f"stage1_stage2_frozen_sweep_{timestamp_utc()}"
    output_root = Path(args.output_root)
    sweep_dir = output_root / "stage1_sweeps" / sweep_name
    sweep_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = sweep_dir / "sweep_manifest.json"
    state_path = sweep_dir / "sweep_state.json"
    stage1_results_path = sweep_dir / "stage1_results.json"
    stage2_results_path = sweep_dir / "stage2_frozen_results.json"
    combined_csv_path = sweep_dir / "combined_results.csv"
    log = make_logger(sweep_dir / "sweep.log")

    stage1_base_config = make_stage1_base_config(args)
    stage1_variants = make_stage1_variants(stage1_base_config)
    selected_names = list(args.variant) if args.variant else list(stage1_variants)
    missing = [name for name in selected_names if name not in stage1_variants]
    if missing:
        raise ValueError(f"Unknown variants {missing}. Available: {sorted(stage1_variants)}")
    selected_stage1_configs = {name: stage1_variants[name] for name in selected_names}
    stage2_config = make_stage2_config(args)

    paths_by_variant = {
        name: {
            "stage1_variant_root": str(stage1_variant_dirs(output_root, sweep_name, name)[0]),
            "stage1_run_dir": str(stage1_variant_dirs(output_root, sweep_name, name)[1]),
            "stage2_variant_root": str(stage2_variant_root(output_root, sweep_name, name)),
            "stage2_run_dir": str(stage2_run_dir(output_root, sweep_name, name)),
        }
        for name in selected_names
    }

    if args.dry_run:
        print("device:", device)
        print("selected variants:", ", ".join(selected_names))
        print(
            json.dumps(
                {
                    "stage1_cache_root": str(args.stage1_cache_root),
                    "stage2_cache_root": str(args.stage2_cache_root),
                    "stage1_configs": {
                        name: asdict(config) for name, config in selected_stage1_configs.items()
                    },
                    "stage2_config": asdict(stage2_config),
                    "paths_by_variant": paths_by_variant,
                },
                indent=2,
            )
        )
        return

    if not Path(args.stage1_cache_root).exists():
        raise FileNotFoundError(f"Stage-1 cache root does not exist: {args.stage1_cache_root}")
    if not Path(args.stage2_cache_root).exists():
        raise FileNotFoundError(f"Stage-2 cache root does not exist: {args.stage2_cache_root}")

    cache_context = make_stage1_cache_context(args)
    expected_stage1_configs = {
        name: _serialize_config(config, cache_context=cache_context)
        for name, config in selected_stage1_configs.items()
    }

    manifest = load_json(manifest_path) if args.resume and manifest_path.exists() else {}
    manifest.update(
        {
            "sweep_name": sweep_name,
            "created_utc": str(manifest.get("created_utc", datetime.now(timezone.utc).isoformat())),
            "updated_utc": datetime.now(timezone.utc).isoformat(),
            "device": str(device),
            "stage1_cache_root": str(args.stage1_cache_root),
            "stage2_cache_root": str(args.stage2_cache_root),
            "output_root": str(output_root),
            "selected_variants": selected_names,
            "stage1_configs": {
                name: asdict(config) for name, config in selected_stage1_configs.items()
            },
            "stage2_config": asdict(stage2_config),
            "paths_by_variant": paths_by_variant,
        }
    )
    write_json(manifest_path, manifest)

    state = load_json(state_path) if args.resume and state_path.exists() else {}
    state.update(
        {
            "sweep_name": sweep_name,
            "created_utc": str(state.get("created_utc", datetime.now(timezone.utc).isoformat())),
            "updated_utc": datetime.now(timezone.utc).isoformat(),
            "stage1_status_by_variant": dict(state.get("stage1_status_by_variant") or {}),
            "stage2_status_by_variant": dict(state.get("stage2_status_by_variant") or {}),
            "last_error_by_variant": dict(state.get("last_error_by_variant") or {}),
        }
    )
    for variant_name in selected_names:
        state["stage1_status_by_variant"].setdefault(variant_name, "pending")
        state["stage2_status_by_variant"].setdefault(variant_name, "pending")
    write_json(state_path, state)

    log(f"sweep dir: {sweep_dir}")
    log(f"device: {device}")
    log(f"stage1 cache root: {args.stage1_cache_root}")
    log(f"stage2 cache root: {args.stage2_cache_root}")
    log(f"variants: {', '.join(selected_names)}")
    log(f"resume enabled: {bool(args.resume)}")
    install_stdout_progress_hook()

    stage1_results_by_variant: dict[str, dict[str, Any]] = {}
    stage2_results_by_variant: dict[str, dict[str, Any]] = {}
    combined_rows_by_variant: dict[str, dict[str, Any]] = {}

    for variant_name in selected_names:
        stage1_config = selected_stage1_configs[variant_name]
        expected_stage1_config = expected_stage1_configs[variant_name]
        _, stage1_run_dir = stage1_variant_dirs(output_root, sweep_name, variant_name)
        stage1_checkpoint_best = stage1_run_dir / "checkpoint_best.pt"

        try:
            mark_status(state, "stage1", variant_name, "running")
            write_json(state_path, state)
            completed_stage1 = (
                completed_stage1_result(
                    stage1_run_dir,
                    expected_config=expected_stage1_config,
                    requested_steps=int(args.stage1_steps),
                )
                if args.resume
                else None
            )
            if completed_stage1 is not None:
                stage1_result = completed_stage1
                log(f"skipping completed Stage-1 variant: {variant_name}")
            else:
                if not args.resume:
                    ensure_fresh_run_dir(stage1_run_dir, stage="Stage-1", variant=variant_name)
                resume_checkpoint = (
                    latest_stage1_resume_checkpoint(stage1_run_dir)
                    if args.resume
                    else None
                )
                if resume_checkpoint is not None:
                    payload = validate_stage1_payload(
                        resume_checkpoint,
                        expected_config=expected_stage1_config,
                        requested_steps=int(args.stage1_steps),
                    )
                    start_step = int(payload.get("step", 0))
                    if start_step > 0:
                        log(
                            f"resuming Stage-1 variant: {variant_name} "
                            f"from step {start_step} checkpoint={resume_checkpoint}"
                        )
                        run_state = recover_possm_run_state_from_checkpoint(
                            cache_context=cache_context,
                            checkpoint_path=resume_checkpoint,
                            device=device,
                        )
                        run_state = resume_possm_training(
                            run_state=run_state,
                            additional_steps=int(args.stage1_steps) - start_step,
                            cache_context=cache_context,
                            device=device,
                        )
                    else:
                        raise ValueError(f"Resume checkpoint has no positive step: {resume_checkpoint}")
                else:
                    log(f"starting Stage-1 variant: {variant_name}")
                    run_state = run_possm_training(
                        cache_context=cache_context,
                        config=stage1_config,
                        output_root=stage1_run_dir.parent,
                        device=device,
                        run_name="run",
                    )
                stage1_result = {
                    "run_dir": str(run_state["run_dir"]),
                    "checkpoint_final_path": str(run_state["checkpoint_path"]),
                    "checkpoint_best_path": str(run_state["best_checkpoint_path"]),
                    "steps": int(run_state.get("checkpoint_step", args.stage1_steps)),
                    "best_step": run_state.get("best_step"),
                    "best_score": run_state.get("best_score"),
                }
            validate_stage1_payload(
                stage1_checkpoint_best,
                expected_config=expected_stage1_config,
                requested_steps=int(args.stage1_steps),
            )
            stage1_results_by_variant[variant_name] = {"variant": variant_name, **stage1_result}
            mark_status(state, "stage1", variant_name, "completed")
            state["last_error_by_variant"].pop(f"stage1:{variant_name}", None)
            write_json(state_path, state)
            write_rows_json(stage1_results_path, list(stage1_results_by_variant.values()))

            mark_status(state, "stage2", variant_name, "running")
            write_json(state_path, state)
            stage2_dir = stage2_run_dir(output_root, sweep_name, variant_name)
            completed_stage2 = (
                validate_stage2_final(
                    stage2_dir,
                    stage1_checkpoint_path=stage1_checkpoint_best,
                    stage2_config=stage2_config,
                )
                if args.resume
                else None
            )
            if completed_stage2 is not None:
                summary = completed_stage2
                log(f"skipping completed frozen Stage-2 variant: {variant_name}")
            else:
                if not args.resume:
                    ensure_fresh_run_dir(stage2_dir, stage="Stage-2", variant=variant_name)
                log(f"starting frozen Stage-2 variant: {variant_name}")
                t0 = time.time()
                summary = run_possm_phoneme_finetuning(
                    checkpoint_path=stage1_checkpoint_best,
                    cache_root=Path(args.stage2_cache_root),
                    output_root=stage2_variant_root(output_root, sweep_name, variant_name),
                    config=stage2_config,
                    device=device,
                    run_name="run",
                    resume_from_latest=bool(args.resume),
                )
                summary["elapsed_seconds"] = round(time.time() - t0, 3)
            validate_stage2_final(
                stage2_dir,
                stage1_checkpoint_path=stage1_checkpoint_best,
                stage2_config=stage2_config,
            )
            stage2_row = flatten_summary_row(variant_name=variant_name, summary=summary)
            stage2_row["elapsed_seconds"] = summary.get("elapsed_seconds")
            stage2_row["stage1_checkpoint_path"] = str(stage1_checkpoint_best)
            stage2_results_by_variant[variant_name] = stage2_row
            mark_status(state, "stage2", variant_name, "completed")
            state["last_error_by_variant"].pop(f"stage2:{variant_name}", None)
            write_json(state_path, state)
            write_rows_json(stage2_results_path, list(stage2_results_by_variant.values()))

            combined_rows_by_variant[variant_name] = {
                "variant": variant_name,
                "stage1_best_score": stage1_results_by_variant[variant_name].get("best_score"),
                "stage1_best_step": stage1_results_by_variant[variant_name].get("best_step"),
                "stage1_checkpoint_best_path": str(stage1_checkpoint_best),
                "stage2_best_val_ctc_bpphone": stage2_row.get("best_val_ctc_bpphone"),
                "stage2_best_val_phoneme_error_rate": stage2_row.get("best_val_phoneme_error_rate"),
                "stage2_best_step": stage2_row.get("best_step"),
                "stage2_val_ctc_bpphone": stage2_row.get("val_ctc_bpphone"),
                "stage2_val_phoneme_error_rate": stage2_row.get("val_phoneme_error_rate"),
                "stage2_predicted_to_reference_token_ratio": stage2_row.get(
                    "predicted_to_reference_token_ratio"
                ),
                "stage2_blank_frame_rate": stage2_row.get("blank_frame_rate"),
            }
            write_combined_csv(combined_csv_path, list(combined_rows_by_variant.values()))

        except Exception as exc:
            stage_key = (
                "stage1"
                if state.get("stage1_status_by_variant", {}).get(variant_name) == "running"
                else "stage2"
            )
            mark_status(state, stage_key, variant_name, "failed")
            state["last_error_by_variant"][f"{stage_key}:{variant_name}"] = {
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "failed_utc": datetime.now(timezone.utc).isoformat(),
            }
            write_json(state_path, state)
            log(f"{stage_key} variant failed: {variant_name} error={exc}")
            if not args.continue_on_error:
                raise

    write_rows_json(stage1_results_path, list(stage1_results_by_variant.values()))
    write_rows_json(stage2_results_path, list(stage2_results_by_variant.values()))
    write_combined_csv(combined_csv_path, list(combined_rows_by_variant.values()))
    manifest["updated_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(manifest_path, manifest)
    log(f"wrote: {manifest_path}")
    log(f"wrote: {state_path}")
    log(f"wrote: {stage1_results_path}")
    log(f"wrote: {stage2_results_path}")
    log(f"wrote: {combined_csv_path}")


if __name__ == "__main__":
    main()

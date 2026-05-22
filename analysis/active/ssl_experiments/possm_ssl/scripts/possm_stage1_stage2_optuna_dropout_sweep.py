"""Run a joint Stage-1 + Stage-2 Optuna sweep over GRU-line dropout settings.

This script targets the ``gru1_plain_linear`` POSSM recipe:

- Stage 1: POSSM reconstruction with a 1-layer GRU temporal backbone, plain MSE,
  no masking, linear reconstruction head
- Stage 2: POSSM phoneme fine-tuning from the Stage-1 best checkpoint

The search space covers:

- Stage-1 encoder dropout (``POSSMTrainingConfig.dropout``)
- Stage-1 temporal GRU dropout (``POSSMTrainingConfig.temporal_gru_dropout``)
- Stage-2 decoder GRU dropout (``POSSMFinetuneConfig.gru_dropout``)

Like the existing Colab sweep utilities, it prints live progress to stdout and
also writes a sweep log and trial artifacts to Drive-friendly paths while
evaluating Stage 2 on the released Willett ``competition_train ->
competition_test`` split.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import time
import traceback
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

EXPERIMENTS_DIR = Path(__file__).resolve().parents[2]
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))

import torch

try:
    import optuna
except ImportError:  # pragma: no cover - runtime only
    optuna = None

from possm_ssl import (
    CacheAccessConfig,
    POSSMFinetuneConfig,
    POSSMTrainingConfig,
    prepare_cache_context,
    run_possm_phoneme_finetuning,
    run_possm_training,
)
from possm_ssl.scripts.possm_stage1_stage2_frozen_sweep import (
    DEFAULT_STAGE1_CACHE_ROOT,
    DEFAULT_STAGE1_LOCAL_CACHE_BASE,
)
from possm_ssl.scripts.possm_stage2_hyperparam_sweep import (
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_STAGE2_CACHE_ROOT,
    flatten_summary_row,
    install_stdout_progress_hook,
    load_json,
    make_logger,
    write_json,
    write_text_atomic,
)


def timestamp_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def base_stage1_config(args: argparse.Namespace) -> POSSMTrainingConfig:
    return POSSMTrainingConfig(
        seed=int(args.seed),
        data_mode=str(args.stage1_data_mode),
        feature_mode=str(args.feature_mode),
        boundary_key_mode=str(args.boundary_key_mode),
        segment_bins=int(args.segment_bins),
        model_dim=int(args.model_dim),
        latent_count=int(args.latent_count),
        value_encoder_type=str(args.value_encoder_type),
        value_mlp_hidden_size=args.value_mlp_hidden_size,
        ffn_hidden_size=int(args.ffn_hidden_size),
        dropout=float(args.stage1_dropout_default),
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
        temporal_backbone_type="gru",
        temporal_gru_hidden_size=args.stage1_temporal_gru_hidden_size,
        temporal_gru_num_layers=int(args.stage1_temporal_gru_num_layers),
        temporal_gru_dropout=float(args.stage1_temporal_gru_dropout_default),
        temporal_gru_bidirectional=bool(args.stage1_temporal_gru_bidirectional),
        temporal_backbone_kwargs={},
        stage1_objective_type="plain_mse",
        masking_type="none",
        mask_prob=0.0,
        mask_span_bins=8,
        mask_replace_mode="zero",
        reconstruction_head_type="linear",
        reconstruction_mlp_hidden_size=None,
    )


def base_stage2_config(args: argparse.Namespace) -> POSSMFinetuneConfig:
    return POSSMFinetuneConfig(
        seed=int(args.seed),
        mode=str(args.mode),
        dataset=str(args.dataset),
        feature_mode=str(args.feature_mode),
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
        gru_hidden_size=int(args.stage2_gru_hidden_size),
        gru_num_layers=int(args.stage2_gru_num_layers),
        gru_dropout=float(args.stage2_gru_dropout_default),
        temporal_patch_kernel_size=int(args.temporal_patch_kernel_size),
        temporal_patch_stride=int(args.temporal_patch_stride),
        conv_dropout=float(args.conv_dropout),
    )


def make_stage1_config(base: POSSMTrainingConfig, **overrides: Any) -> POSSMTrainingConfig:
    payload = asdict(base)
    payload.update(overrides)
    return POSSMTrainingConfig(**payload)


def make_stage2_config(base: POSSMFinetuneConfig, **overrides: Any) -> POSSMFinetuneConfig:
    payload = asdict(base)
    payload.update(overrides)
    return POSSMFinetuneConfig(**payload)


def make_stage1_cache_context(args: argparse.Namespace):
    config = CacheAccessConfig(
        mode=str(args.stage1_cache_mode),
        local_cache_base=str(args.stage1_local_cache_base),
        force_recopy_local_cache=bool(args.force_recopy_local_cache),
        excluded_datasets=tuple(args.excluded_dataset),
        seed=int(args.seed),
        segment_bins=int(args.segment_bins),
        use_normalization=str(args.stage1_data_mode) == "normalized",
        examples_per_shard=int(args.examples_per_shard),
        tx_dim=int(args.tx_dim),
        sbp_dim=int(args.sbp_dim),
        feature_mode=str(args.feature_mode),
        boundary_key_mode=str(args.boundary_key_mode),
        gaussian_smoothing_sigma_bins=0.0,
        precomputed_session_stats_path=args.stage1_session_stats_path,
    )
    return prepare_cache_context(cache_candidates=[Path(args.stage1_cache_root)], config=config)


def write_results_csv(csv_path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    tmp_path = csv_path.with_name(f".{csv_path.name}.tmp")
    with tmp_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    tmp_path.replace(csv_path)


def delete_trial_checkpoints(*, trial_dir: Path, log: Any) -> None:
    checkpoint_targets = [
        trial_dir / "stage1" / "run" / "checkpoint_best.pt",
        trial_dir / "stage1" / "run" / "checkpoint_final.pt",
        trial_dir / "stage1" / "run" / "checkpoints",
        trial_dir / "stage2" / "run" / "checkpoint_best.pt",
        trial_dir / "stage2" / "run" / "checkpoint_final.pt",
        trial_dir / "stage2" / "run" / "checkpoints",
    ]
    for path in checkpoint_targets:
        if not path.exists():
            continue
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
        log(f"deleted checkpoint artifact: {path}")


def cleanup_nonbest_trial_checkpoints(
    *,
    sweep_dir: Path,
    keep_trial_number: int | None,
    log: Any,
) -> None:
    trials_root = sweep_dir / "trials"
    if not trials_root.exists():
        return
    for trial_dir in sorted(trials_root.glob("trial_*")):
        try:
            trial_number = int(trial_dir.name.split("_", 1)[1])
        except (IndexError, ValueError):
            continue
        if keep_trial_number is not None and trial_number == int(keep_trial_number):
            continue
        delete_trial_checkpoints(trial_dir=trial_dir, log=log)


def best_result_payload(study: "optuna.Study", rows_by_trial: dict[int, dict[str, Any]]) -> dict[str, Any]:
    try:
        best_trial = study.best_trial
    except ValueError:
        return {}
    payload = {
        "trial_number": int(best_trial.number),
        "objective_name": "best_val_ctc_bpphone",
        "objective_value": best_trial.value,
        "params": dict(best_trial.params),
    }
    if int(best_trial.number) in rows_by_trial:
        payload["row"] = rows_by_trial[int(best_trial.number)]
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage1-cache-root", type=Path, default=DEFAULT_STAGE1_CACHE_ROOT)
    parser.add_argument("--stage2-cache-root", type=Path, default=DEFAULT_STAGE2_CACHE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--sweep-name", default=None)
    parser.add_argument("--study-name", default=None)
    parser.add_argument("--n-trials", type=int, default=12)
    parser.add_argument("--sampler-seed", type=int, default=7)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument(
        "--delete-unsuccessful-checkpoints",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep only the current best trial's checkpoint files; preserve logs/summaries for the rest.",
    )
    parser.add_argument("--dry-run", action="store_true")

    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--dataset", default="brain2text24")
    parser.add_argument("--feature-mode", choices=("tx_only", "tx_sbp"), default="tx_only")
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
    parser.add_argument("--excluded-dataset", action="append", default=["brain2text25"])

    parser.add_argument("--stage1-steps", type=int, default=4000)
    parser.add_argument("--stage1-data-mode", choices=("raw", "normalized"), default="normalized")
    parser.add_argument("--stage1-cache-mode", choices=("copy_to_local", "drive_direct"), default="drive_direct")
    parser.add_argument("--stage1-local-cache-base", default=DEFAULT_STAGE1_LOCAL_CACHE_BASE)
    parser.add_argument("--force-recopy-local-cache", action="store_true")
    parser.add_argument("--stage1-session-stats-path", type=Path, default=None)
    parser.add_argument("--stage1-dropout-default", type=float, default=0.15)
    parser.add_argument("--stage1-batch-size", type=int, default=32)
    parser.add_argument("--stage1-learning-rate", type=float, default=3e-4)
    parser.add_argument("--stage1-weight-decay", type=float, default=1e-3)
    parser.add_argument("--stage1-val-every", type=int, default=50)
    parser.add_argument("--stage1-val-batches", type=int, default=2)
    parser.add_argument("--stage1-checkpoint-every-steps", type=int, default=500)
    parser.add_argument("--dataset-weight-alpha", type=float, default=0.25)
    parser.add_argument("--examples-per-shard", type=int, default=8)
    parser.add_argument("--stage1-log-every", type=int, default=20)
    parser.add_argument("--stage1-temporal-gru-hidden-size", type=int, default=None)
    parser.add_argument("--stage1-temporal-gru-num-layers", type=int, default=1)
    parser.add_argument("--stage1-temporal-gru-dropout-default", type=float, default=0.0)
    parser.add_argument("--stage1-temporal-gru-bidirectional", action="store_true")

    parser.add_argument("--mode", choices=("probe_frozen", "finetune_full"), default="finetune_full")
    parser.add_argument("--stage2-steps", type=int, default=1500)
    parser.add_argument("--stage2-data-mode", choices=("raw", "normalized"), default="normalized")
    parser.add_argument("--stage2-batch-size", type=int, default=32)
    parser.add_argument("--stage2-learning-rate", type=float, default=2e-4)
    parser.add_argument("--stage2-encoder-learning-rate", type=float, default=3e-5)
    parser.add_argument("--stage2-weight-decay", type=float, default=1e-3)
    parser.add_argument("--stage2-max-grad-norm", type=float, default=1.0)
    parser.add_argument("--stage2-checkpoint-every-steps", type=int, default=100)
    parser.add_argument("--stage2-progress-every-steps", type=int, default=20)
    parser.add_argument("--session-adapter-enabled", action="store_true")
    parser.add_argument("--input-smoothing-sigma-bins", type=float, default=2.0)
    parser.add_argument("--input-smoothing-kernel-size", type=int, default=100)
    parser.add_argument("--input-smoothing-threshold", type=float, default=0.01)
    parser.add_argument("--white-noise-sd", type=float, default=0.1)
    parser.add_argument("--constant-offset-sd", type=float, default=0.05)
    parser.add_argument("--stage2-gru-hidden-size", type=int, default=768)
    parser.add_argument("--stage2-gru-num-layers", type=int, default=5)
    parser.add_argument("--stage2-gru-dropout-default", type=float, default=0.2)
    parser.add_argument("--temporal-patch-kernel-size", type=int, default=14)
    parser.add_argument("--temporal-patch-stride", type=int, default=4)
    parser.add_argument("--conv-dropout", type=float, default=0.1)

    parser.add_argument("--stage1-encoder-dropout-low", type=float, default=0.05)
    parser.add_argument("--stage1-encoder-dropout-high", type=float, default=0.30)
    parser.add_argument("--stage1-encoder-dropout-step", type=float, default=0.05)
    parser.add_argument("--stage1-gru-dropout-low", type=float, default=0.0)
    parser.add_argument("--stage1-gru-dropout-high", type=float, default=0.30)
    parser.add_argument("--stage1-gru-dropout-step", type=float, default=0.05)
    parser.add_argument("--stage2-gru-dropout-low", type=float, default=0.0)
    parser.add_argument("--stage2-gru-dropout-high", type=float, default=0.40)
    parser.add_argument("--stage2-gru-dropout-step", type=float, default=0.05)
    return parser.parse_args()


def main() -> None:
    if optuna is None:
        raise ImportError(
            "optuna is not installed. In Colab, run `pip install optuna` before launching this sweep."
        )

    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if int(args.n_trials) <= 0:
        raise ValueError("--n-trials must be positive")

    stage1_base = base_stage1_config(args)
    stage2_base = base_stage2_config(args)

    if args.dry_run:
        print("device:", device)
        print("stage1 base config:")
        print(json.dumps(asdict(stage1_base), indent=2))
        print("stage2 base config:")
        print(json.dumps(asdict(stage2_base), indent=2))
        print(
            json.dumps(
                {
                    "n_trials": int(args.n_trials),
                    "delete_unsuccessful_checkpoints": bool(args.delete_unsuccessful_checkpoints),
                    "search_space": {
                        "stage1_encoder_dropout": {
                            "low": float(args.stage1_encoder_dropout_low),
                            "high": float(args.stage1_encoder_dropout_high),
                            "step": float(args.stage1_encoder_dropout_step),
                        },
                        "stage1_gru_dropout": {
                            "low": float(args.stage1_gru_dropout_low),
                            "high": float(args.stage1_gru_dropout_high),
                            "step": float(args.stage1_gru_dropout_step),
                        },
                        "stage2_gru_dropout": {
                            "low": float(args.stage2_gru_dropout_low),
                            "high": float(args.stage2_gru_dropout_high),
                            "step": float(args.stage2_gru_dropout_step),
                        },
                    },
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

    sweep_name = args.sweep_name or f"stage1_stage2_optuna_dropout_sweep_{timestamp_utc()}"
    study_name = args.study_name or sweep_name
    sweep_dir = Path(args.output_root) / "stage1_stage2_optuna_sweeps" / sweep_name
    sweep_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = sweep_dir / "sweep_manifest.json"
    state_path = sweep_dir / "sweep_state.json"
    csv_path = sweep_dir / "trial_results.csv"
    results_json_path = sweep_dir / "trial_results.json"
    best_json_path = sweep_dir / "best_result.json"
    study_db_path = sweep_dir / "optuna_study.sqlite3"
    sweep_log_path = sweep_dir / "sweep.log"
    log = make_logger(sweep_log_path)

    if args.resume and manifest_path.exists():
        manifest = load_json(manifest_path)
    else:
        manifest = {}
    manifest.update(
        {
            "sweep_name": sweep_name,
            "study_name": study_name,
            "created_utc": str(manifest.get("created_utc", datetime.now(timezone.utc).isoformat())),
            "updated_utc": datetime.now(timezone.utc).isoformat(),
            "device": str(device),
            "stage1_cache_root": str(args.stage1_cache_root),
            "stage2_cache_root": str(args.stage2_cache_root),
            "output_root": str(args.output_root),
            "study_db_path": str(study_db_path),
            "stage1_base_config": asdict(stage1_base),
            "stage2_base_config": asdict(stage2_base),
            "n_trials_per_invocation": int(args.n_trials),
            "search_space": {
                "stage1_encoder_dropout": {
                    "low": float(args.stage1_encoder_dropout_low),
                    "high": float(args.stage1_encoder_dropout_high),
                    "step": float(args.stage1_encoder_dropout_step),
                },
                "stage1_gru_dropout": {
                    "low": float(args.stage1_gru_dropout_low),
                    "high": float(args.stage1_gru_dropout_high),
                    "step": float(args.stage1_gru_dropout_step),
                },
                "stage2_gru_dropout": {
                    "low": float(args.stage2_gru_dropout_low),
                    "high": float(args.stage2_gru_dropout_high),
                    "step": float(args.stage2_gru_dropout_step),
                },
            },
        }
    )
    write_json(manifest_path, manifest)

    if args.resume and state_path.exists():
        sweep_state = load_json(state_path)
    else:
        sweep_state = {}
    sweep_state.update(
        {
            "sweep_name": sweep_name,
            "study_name": study_name,
            "created_utc": str(sweep_state.get("created_utc", datetime.now(timezone.utc).isoformat())),
            "updated_utc": datetime.now(timezone.utc).isoformat(),
            "last_error": sweep_state.get("last_error"),
        }
    )
    write_json(state_path, sweep_state)

    log(f"sweep dir: {sweep_dir}")
    log(f"study db: {study_db_path}")
    log(f"device: {device}")
    log(f"stage1 cache root: {args.stage1_cache_root}")
    log(f"stage2 cache root: {args.stage2_cache_root}")
    log(f"stage1 steps: {stage1_base.num_steps}")
    log(f"stage2 steps: {stage2_base.num_steps}")
    log(f"delete unsuccessful checkpoints: {bool(args.delete_unsuccessful_checkpoints)}")
    log(
        "search space: "
        f"stage1_encoder_dropout=[{float(args.stage1_encoder_dropout_low)}, {float(args.stage1_encoder_dropout_high)}] "
        f"step={float(args.stage1_encoder_dropout_step)}; "
        f"stage1_gru_dropout=[{float(args.stage1_gru_dropout_low)}, {float(args.stage1_gru_dropout_high)}] "
        f"step={float(args.stage1_gru_dropout_step)}; "
        f"stage2_gru_dropout=[{float(args.stage2_gru_dropout_low)}, {float(args.stage2_gru_dropout_high)}] "
        f"step={float(args.stage2_gru_dropout_step)}"
    )
    if int(args.stage1_temporal_gru_num_layers) == 1:
        log(
            "warning: stage1_temporal_gru_num_layers=1, so Stage-1 GRU dropout is structurally inactive in "
            "PyTorch GRU and will not change the run unless you set --stage1-temporal-gru-num-layers > 1."
        )

    install_stdout_progress_hook()

    sampler = optuna.samplers.TPESampler(seed=int(args.sampler_seed))
    study = optuna.create_study(
        study_name=study_name,
        direction="minimize",
        sampler=sampler,
        storage=f"sqlite:///{study_db_path}",
        load_if_exists=bool(args.resume),
    )

    rows_by_trial: dict[int, dict[str, Any]] = {}
    for trial in study.trials:
        attrs_row = trial.user_attrs.get("row")
        if isinstance(attrs_row, dict):
            rows_by_trial[int(trial.number)] = dict(attrs_row)

    def persist_results() -> None:
        ordered_rows = [rows_by_trial[number] for number in sorted(rows_by_trial)]
        write_results_csv(csv_path, ordered_rows)
        write_text_atomic(results_json_path, json.dumps(ordered_rows, indent=2))
        write_text_atomic(best_json_path, json.dumps(best_result_payload(study, rows_by_trial), indent=2))
        try:
            best_trial = study.best_trial
        except ValueError:
            best_trial = None
        manifest["updated_utc"] = datetime.now(timezone.utc).isoformat()
        manifest["completed_trial_count"] = int(
            sum(1 for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE)
        )
        manifest["failed_trial_count"] = int(
            sum(1 for trial in study.trials if trial.state == optuna.trial.TrialState.FAIL)
        )
        manifest["best_trial_number"] = None if best_trial is None else int(best_trial.number)
        manifest["best_value"] = None if best_trial is None else best_trial.value
        write_json(manifest_path, manifest)
        sweep_state["updated_utc"] = datetime.now(timezone.utc).isoformat()
        sweep_state["completed_trial_count"] = manifest["completed_trial_count"]
        sweep_state["failed_trial_count"] = manifest["failed_trial_count"]
        sweep_state["best_trial_number"] = manifest["best_trial_number"]
        write_json(state_path, sweep_state)
        if bool(args.delete_unsuccessful_checkpoints):
            cleanup_nonbest_trial_checkpoints(
                sweep_dir=sweep_dir,
                keep_trial_number=manifest["best_trial_number"],
                log=log,
            )

    def objective(trial: "optuna.Trial") -> float:
        stage1_encoder_dropout = trial.suggest_float(
            "stage1_encoder_dropout",
            float(args.stage1_encoder_dropout_low),
            float(args.stage1_encoder_dropout_high),
            step=float(args.stage1_encoder_dropout_step),
        )
        stage1_gru_dropout = trial.suggest_float(
            "stage1_gru_dropout",
            float(args.stage1_gru_dropout_low),
            float(args.stage1_gru_dropout_high),
            step=float(args.stage1_gru_dropout_step),
        )
        stage2_gru_dropout = trial.suggest_float(
            "stage2_gru_dropout",
            float(args.stage2_gru_dropout_low),
            float(args.stage2_gru_dropout_high),
            step=float(args.stage2_gru_dropout_step),
        )

        stage1_config = make_stage1_config(
            stage1_base,
            dropout=float(stage1_encoder_dropout),
            temporal_gru_dropout=float(stage1_gru_dropout),
        )
        stage2_config = make_stage2_config(
            stage2_base,
            gru_dropout=float(stage2_gru_dropout),
        )

        trial_dir = sweep_dir / "trials" / f"trial_{int(trial.number):04d}"
        stage1_output_root = trial_dir / "stage1"
        stage2_output_root = trial_dir / "stage2"
        stage1_output_root.mkdir(parents=True, exist_ok=True)
        stage2_output_root.mkdir(parents=True, exist_ok=True)

        log(
            f"starting trial {int(trial.number)}: "
            f"stage1_encoder_dropout={float(stage1_encoder_dropout):.3f}, "
            f"stage1_gru_dropout={float(stage1_gru_dropout):.3f}, "
            f"stage2_gru_dropout={float(stage2_gru_dropout):.3f}"
        )
        print(
            json.dumps(
                {
                    "trial_number": int(trial.number),
                    "stage1_config": asdict(stage1_config),
                    "stage2_config": asdict(stage2_config),
                },
                indent=2,
            ),
            flush=True,
        )

        t0 = time.time()
        try:
            stage1_run_state = run_possm_training(
                cache_context=cache_context,
                config=stage1_config,
                output_root=stage1_output_root,
                device=device,
                run_name="run",
            )
            stage1_best_checkpoint = Path(stage1_run_state["best_checkpoint_path"])
            stage2_summary = run_possm_phoneme_finetuning(
                checkpoint_path=stage1_best_checkpoint,
                cache_root=Path(args.stage2_cache_root),
                output_root=stage2_output_root,
                config=stage2_config,
                device=device,
                run_name="run",
                resume_from_latest=False,
            )
        except Exception as exc:
            sweep_state["last_error"] = {
                "trial_number": int(trial.number),
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "failed_utc": datetime.now(timezone.utc).isoformat(),
            }
            write_json(state_path, sweep_state)
            log(f"trial failed: {int(trial.number)} error={exc}")
            if bool(args.delete_unsuccessful_checkpoints):
                delete_trial_checkpoints(trial_dir=trial_dir, log=log)
            raise

        elapsed_seconds = round(time.time() - t0, 3)
        stage2_row = flatten_summary_row(variant_name=f"trial_{int(trial.number):04d}", summary=stage2_summary)
        row = {
            "trial_number": int(trial.number),
            "stage1_encoder_dropout": float(stage1_encoder_dropout),
            "stage1_gru_dropout": float(stage1_gru_dropout),
            "stage2_gru_dropout": float(stage2_gru_dropout),
            "elapsed_seconds": elapsed_seconds,
            "stage1_run_dir": str(stage1_run_state["run_dir"]),
            "stage1_best_checkpoint_path": str(stage1_best_checkpoint),
            "stage1_best_score": stage1_run_state.get("best_score"),
            "stage1_best_step": stage1_run_state.get("best_step"),
            **stage2_row,
        }
        rows_by_trial[int(trial.number)] = dict(row)
        trial.set_user_attr("row", dict(row))
        trial.set_user_attr("stage1_config", asdict(stage1_config))
        trial.set_user_attr("stage2_config", asdict(stage2_config))
        trial.set_user_attr("stage2_summary", stage2_summary)

        objective_value = float(stage2_summary["metrics"]["best_val_ctc_bpphone"])
        log(
            f"trial completed: {int(trial.number)} "
            f"best_val_ctc_bpphone={objective_value:.4f} "
            f"best_val_per={float(stage2_summary['metrics']['best_val_phoneme_error_rate']):.4f}"
        )
        print("trial result:", json.dumps(row, indent=2), flush=True)
        persist_results()
        return objective_value

    log(f"existing trials in study: {len(study.trials)}")
    study.optimize(
        objective,
        n_trials=int(args.n_trials),
        catch=(Exception,) if args.continue_on_error else (),
        show_progress_bar=False,
    )
    persist_results()

    try:
        best_trial = study.best_trial
    except ValueError:
        best_trial = None
    if best_trial is not None:
        log(
            f"best trial: {int(best_trial.number)} "
            f"best_val_ctc_bpphone={float(best_trial.value):.4f} "
            f"params={json.dumps(best_trial.params, sort_keys=True)}"
        )
        best_row = rows_by_trial.get(int(best_trial.number))
        if best_row is not None:
            print("best trial row:", json.dumps(best_row, indent=2), flush=True)

    log(f"wrote: {manifest_path}")
    log(f"wrote: {results_json_path}")
    log(f"wrote: {best_json_path}")
    log(f"wrote: {csv_path}")
    log(f"wrote: {state_path}")
    log(f"wrote: {study_db_path}")
    log(f"wrote: {sweep_log_path}")


if __name__ == "__main__":
    main()

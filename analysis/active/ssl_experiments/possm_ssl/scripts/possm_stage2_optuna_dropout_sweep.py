"""Run an Optuna sweep over POSSM Stage-2 dropout hyperparameters.

This script is meant to be launched from a Colab cell after the repository is
mounted on Drive and added to ``PYTHONPATH``. It reuses the existing Stage-2
progress hook so train / validation updates are printed live in the notebook
output while also being written to a sweep log on disk, using the released
Willett ``competition_train -> competition_test`` split.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
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
except ImportError:  # pragma: no cover - exercised in Colab/runtime, not unit tests
    optuna = None

from possm_ssl import POSSMFinetuneConfig, resolve_possm_checkpoint_path, run_possm_phoneme_finetuning
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


def base_config(args: argparse.Namespace) -> POSSMFinetuneConfig:
    return POSSMFinetuneConfig(
        seed=int(args.seed),
        mode=str(args.mode),
        dataset=str(args.dataset),
        feature_mode=str(args.feature_mode),
        data_mode=str(args.data_mode),
        boundary_key_mode=str(args.boundary_key_mode),
        batch_size=int(args.batch_size),
        num_steps=int(args.num_steps),
        learning_rate=float(args.learning_rate),
        encoder_learning_rate=float(args.encoder_learning_rate),
        weight_decay=float(args.weight_decay),
        max_grad_norm=float(args.max_grad_norm),
        checkpoint_every_steps=int(args.checkpoint_every_steps),
        progress_every_steps=int(args.progress_every_steps),
        session_adapter_enabled=bool(args.session_adapter_enabled),
        input_smoothing_sigma_bins=float(args.input_smoothing_sigma_bins),
        input_smoothing_kernel_size=int(args.input_smoothing_kernel_size),
        input_smoothing_threshold=float(args.input_smoothing_threshold),
        white_noise_sd=float(args.white_noise_sd),
        constant_offset_sd=float(args.constant_offset_sd),
        gru_hidden_size=int(args.gru_hidden_size),
        gru_num_layers=int(args.gru_num_layers),
        gru_dropout=float(args.gru_dropout_default),
        decoder_backbone_type=str(args.decoder_backbone_type),
        s5_hidden_size=int(args.s5_hidden_size),
        s5_state_size=int(args.s5_state_size),
        s5_num_layers=int(args.s5_num_layers),
        s5_dropout=float(args.s5_dropout),
        s5_direction=str(args.s5_direction),
        s5_ffn_multiplier=float(args.s5_ffn_multiplier),
        temporal_patch_kernel_size=int(args.temporal_patch_kernel_size),
        temporal_patch_stride=int(args.temporal_patch_stride),
        conv_dropout=float(args.conv_dropout_default),
    )


def make_config(base: POSSMFinetuneConfig, **overrides: Any) -> POSSMFinetuneConfig:
    payload = asdict(base)
    payload.update(overrides)
    return POSSMFinetuneConfig(**payload)


def resolve_stage1_checkpoint(args: argparse.Namespace) -> Path:
    if args.stage1_checkpoint is not None:
        checkpoint = Path(args.stage1_checkpoint)
        if not checkpoint.exists():
            raise FileNotFoundError(f"Stage-1 checkpoint does not exist: {checkpoint}")
        return checkpoint
    return resolve_possm_checkpoint_path(output_root=Path(args.stage1_output_root))


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


def trial_result_row(
    *,
    trial: "optuna.trial.FrozenTrial",
    row: dict[str, Any] | None,
    sweep_dir: Path,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "trial_number": int(trial.number),
        "trial_state": str(trial.state.name).lower(),
        "objective_name": "best_val_ctc_bpphone",
        "objective_value": trial.value,
        "gru_dropout": trial.params.get("gru_dropout"),
        "conv_dropout": trial.params.get("conv_dropout"),
        "started_utc": (
            None if trial.datetime_start is None else trial.datetime_start.replace(tzinfo=timezone.utc).isoformat()
        ),
        "completed_utc": (
            None if trial.datetime_complete is None else trial.datetime_complete.replace(tzinfo=timezone.utc).isoformat()
        ),
        "trial_dir": str(sweep_dir / "trials" / f"trial_{int(trial.number):04d}"),
    }
    if row is not None:
        payload.update(row)
    return payload


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
    parser.add_argument("--stage1-checkpoint", type=Path, default=None)
    parser.add_argument("--stage1-output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--stage2-cache-root", type=Path, default=DEFAULT_STAGE2_CACHE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT / "phoneme_finetune")
    parser.add_argument("--sweep-name", default=None)
    parser.add_argument("--study-name", default=None)
    parser.add_argument("--n-trials", type=int, default=12)
    parser.add_argument("--sampler-seed", type=int, default=7)
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse an existing Optuna SQLite study in the same sweep directory when present.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue running new trials if one trial fails.",
    )
    parser.add_argument("--dry-run", action="store_true")

    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--mode", choices=("probe_frozen", "finetune_full"), default="finetune_full")
    parser.add_argument("--dataset", default="brain2text24")
    parser.add_argument("--feature-mode", choices=("tx_only", "tx_sbp"), default="tx_only")
    parser.add_argument("--data-mode", choices=("raw", "normalized"), default="normalized")
    parser.add_argument("--boundary-key-mode", choices=("session", "subject_if_available"), default="session")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-steps", type=int, default=1500)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--encoder-learning-rate", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--checkpoint-every-steps", type=int, default=100)
    parser.add_argument("--progress-every-steps", type=int, default=20)
    parser.add_argument("--session-adapter-enabled", action="store_true")
    parser.add_argument("--input-smoothing-sigma-bins", type=float, default=2.0)
    parser.add_argument("--input-smoothing-kernel-size", type=int, default=100)
    parser.add_argument("--input-smoothing-threshold", type=float, default=0.01)
    parser.add_argument("--white-noise-sd", type=float, default=0.1)
    parser.add_argument("--constant-offset-sd", type=float, default=0.05)
    parser.add_argument("--gru-hidden-size", type=int, default=768)
    parser.add_argument("--gru-num-layers", type=int, default=5)
    parser.add_argument("--gru-dropout-default", type=float, default=0.2)
    parser.add_argument("--decoder-backbone-type", choices=("gru", "s5"), default="gru")
    parser.add_argument("--s5-hidden-size", type=int, default=768)
    parser.add_argument("--s5-state-size", type=int, default=128)
    parser.add_argument("--s5-num-layers", type=int, default=5)
    parser.add_argument("--s5-dropout", type=float, default=0.2)
    parser.add_argument("--s5-direction", choices=("causal", "bidirectional"), default="causal")
    parser.add_argument("--s5-ffn-multiplier", type=float, default=2.0)
    parser.add_argument("--temporal-patch-kernel-size", type=int, default=14)
    parser.add_argument("--temporal-patch-stride", type=int, default=4)
    parser.add_argument("--conv-dropout-default", type=float, default=0.1)
    parser.add_argument("--gru-dropout-low", type=float, default=0.0)
    parser.add_argument("--gru-dropout-high", type=float, default=0.4)
    parser.add_argument("--gru-dropout-step", type=float, default=0.05)
    parser.add_argument("--conv-dropout-low", type=float, default=0.0)
    parser.add_argument("--conv-dropout-high", type=float, default=0.3)
    parser.add_argument("--conv-dropout-step", type=float, default=0.05)
    return parser.parse_args()


def main() -> None:
    if optuna is None:
        raise ImportError(
            "optuna is not installed. In Colab, run `pip install optuna` before launching this sweep."
        )

    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base = base_config(args)

    if int(args.n_trials) <= 0:
        raise ValueError("--n-trials must be positive")
    if float(args.gru_dropout_low) > float(args.gru_dropout_high):
        raise ValueError("--gru-dropout-low must be <= --gru-dropout-high")
    if float(args.conv_dropout_low) > float(args.conv_dropout_high):
        raise ValueError("--conv-dropout-low must be <= --conv-dropout-high")

    if args.dry_run:
        print("device:", device)
        print("base config:")
        print(json.dumps(asdict(base), indent=2))
        print(
            json.dumps(
                {
                    "n_trials": int(args.n_trials),
                    "search_space": {
                        "gru_dropout": {
                            "low": float(args.gru_dropout_low),
                            "high": float(args.gru_dropout_high),
                            "step": float(args.gru_dropout_step),
                        },
                        "conv_dropout": {
                            "low": float(args.conv_dropout_low),
                            "high": float(args.conv_dropout_high),
                            "step": float(args.conv_dropout_step),
                        },
                    },
                },
                indent=2,
            )
        )
        return

    stage1_checkpoint = resolve_stage1_checkpoint(args)
    stage2_cache_root = Path(args.stage2_cache_root)
    if not stage2_cache_root.exists():
        raise FileNotFoundError(f"Stage-2 cache root does not exist: {stage2_cache_root}")

    sweep_name = args.sweep_name or f"stage2_optuna_dropout_sweep_{timestamp_utc()}"
    study_name = args.study_name or sweep_name
    sweep_dir = Path(args.output_root) / "sweeps" / sweep_name
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
            "stage1_checkpoint": str(stage1_checkpoint),
            "stage2_cache_root": str(stage2_cache_root),
            "output_root": str(args.output_root),
            "study_db_path": str(study_db_path),
            "base_config": asdict(base),
            "n_trials_per_invocation": int(args.n_trials),
            "search_space": {
                "gru_dropout": {
                    "low": float(args.gru_dropout_low),
                    "high": float(args.gru_dropout_high),
                    "step": float(args.gru_dropout_step),
                },
                "conv_dropout": {
                    "low": float(args.conv_dropout_low),
                    "high": float(args.conv_dropout_high),
                    "step": float(args.conv_dropout_step),
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

    log(f"stage1 checkpoint: {stage1_checkpoint}")
    log(f"stage2 cache root: {stage2_cache_root}")
    log(f"sweep dir: {sweep_dir}")
    log(f"study db: {study_db_path}")
    log(f"device: {device}")
    log(f"base stage2 steps: {base.num_steps}")
    log(
        "search space: "
        f"gru_dropout=[{float(args.gru_dropout_low)}, {float(args.gru_dropout_high)}] step={float(args.gru_dropout_step)}; "
        f"conv_dropout=[{float(args.conv_dropout_low)}, {float(args.conv_dropout_high)}] step={float(args.conv_dropout_step)}"
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

    def objective(trial: "optuna.Trial") -> float:
        gru_dropout = trial.suggest_float(
            "gru_dropout",
            float(args.gru_dropout_low),
            float(args.gru_dropout_high),
            step=float(args.gru_dropout_step),
        )
        conv_dropout = trial.suggest_float(
            "conv_dropout",
            float(args.conv_dropout_low),
            float(args.conv_dropout_high),
            step=float(args.conv_dropout_step),
        )

        config = make_config(
            base,
            gru_dropout=float(gru_dropout),
            conv_dropout=float(conv_dropout),
        )
        trial_dir = sweep_dir / "trials" / f"trial_{int(trial.number):04d}"
        trial_dir.mkdir(parents=True, exist_ok=True)

        log(
            f"starting trial {int(trial.number)}: "
            f"gru_dropout={float(gru_dropout):.3f}, conv_dropout={float(conv_dropout):.3f}"
        )
        print(
            json.dumps(
                {
                    "trial_number": int(trial.number),
                    "config": asdict(config),
                },
                indent=2,
            ),
            flush=True,
        )

        try:
            summary = run_possm_phoneme_finetuning(
                checkpoint_path=stage1_checkpoint,
                cache_root=stage2_cache_root,
                output_root=trial_dir,
                config=config,
                device=device,
                run_name="run",
                resume_from_latest=bool(args.resume),
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
            raise

        row = flatten_summary_row(variant_name=f"trial_{int(trial.number):04d}", summary=summary)
        row.update(
            {
                "trial_number": int(trial.number),
                "gru_dropout": float(gru_dropout),
                "conv_dropout": float(conv_dropout),
            }
        )
        rows_by_trial[int(trial.number)] = dict(row)
        trial.set_user_attr("row", dict(row))
        trial.set_user_attr("summary", summary)

        objective_value = metrics_value = summary["metrics"]["best_val_ctc_bpphone"]
        log(
            f"trial completed: {int(trial.number)} "
            f"best_val_ctc_bpphone={float(objective_value):.4f} "
            f"best_val_per={float(summary['metrics']['best_val_phoneme_error_rate']):.4f}"
        )
        print("trial result:", json.dumps(row, indent=2), flush=True)
        persist_results()
        return float(metrics_value)

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

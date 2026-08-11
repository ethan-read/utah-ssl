"""Run an isolated POSSM Stage-2 GRU-dropout ablation.

This script is intended for the specific three-run comparison discussed in the
POSSM notebook after switching Stage 2 to the Willett
``competition_train -> competition_test`` split.

Each dropout value is launched in a fresh subprocess so GPU allocator state from
one run does not leak into the next. That makes it much safer than looping over
multiple ``run_possm_phoneme_finetuning(...)`` calls inside one long-lived
notebook kernel while keeping the target batch size at 32.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.possm_style import (
    POSSMFinetuneConfig,
    resolve_possm_checkpoint_path,
    run_possm_phoneme_finetuning,
)
from experiments.possm_style.scripts.possm_stage2_hyperparam_sweep import (
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_STAGE2_CACHE_ROOT,
    flatten_summary_row,
    install_stdout_progress_hook,
    load_json,
    make_logger,
    write_json,
    write_text_atomic,
    write_results_csv,
)


def timestamp_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def base_config(args: argparse.Namespace) -> POSSMFinetuneConfig:
    return POSSMFinetuneConfig(
        seed=int(args.seed),
        mode=str(args.mode),
        dataset=str(args.dataset),
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
        conv_kernel_size=int(args.conv_kernel_size),
        conv_stride=int(args.conv_stride),
        conv_dropout=float(args.conv_dropout),
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


def default_dropouts(args: argparse.Namespace) -> list[float]:
    if args.dropout:
        return [float(value) for value in args.dropout]
    return [0.1, 0.2, 0.3]


def sanitize_dropout_label(dropout: float) -> str:
    return f"{float(dropout):.3f}".rstrip("0").rstrip(".").replace(".", "p")


def worker_result_path(variant_root: Path) -> Path:
    return variant_root / "worker_result.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage1-checkpoint", type=Path, default=None)
    parser.add_argument("--stage1-output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--stage2-cache-root", type=Path, default=DEFAULT_STAGE2_CACHE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT / "phoneme_finetune")
    parser.add_argument("--sweep-name", default=None)
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Resume completed variants when worker_result.json is present.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Keep running remaining variants if one variant fails.",
    )
    parser.add_argument(
        "--dropout",
        action="append",
        type=float,
        default=None,
        help="GRU dropout value to test. May be repeated. Defaults to 0.1, 0.2, 0.3.",
    )
    parser.add_argument("--dry-run", action="store_true")

    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--mode", choices=("probe_frozen", "finetune_full"), default="finetune_full")
    parser.add_argument("--dataset", default="brain2text24")
    parser.add_argument("--data-mode", choices=("raw", "normalized"), default="normalized")
    parser.add_argument("--boundary-key-mode", choices=("session", "subject_if_available"), default="session")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-steps", type=int, default=3000)
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
    parser.add_argument("--conv-kernel-size", type=int, default=14)
    parser.add_argument("--conv-stride", type=int, default=4)
    parser.add_argument("--conv-dropout", type=float, default=0.1)

    parser.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--_worker-variant", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--_worker-dropout", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--_worker-variant-root", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--_worker-stage1-checkpoint", type=Path, default=None, help=argparse.SUPPRESS)
    return parser.parse_args()


def run_worker(args: argparse.Namespace) -> None:
    if args._worker_variant is None or args._worker_dropout is None or args._worker_variant_root is None:
        raise ValueError("Worker mode requires variant name, dropout, and variant root.")
    if args._worker_stage1_checkpoint is None:
        raise ValueError("Worker mode requires an explicit stage-1 checkpoint path.")

    stage1_checkpoint = Path(args._worker_stage1_checkpoint)
    stage2_cache_root = Path(args.stage2_cache_root)
    variant_root = Path(args._worker_variant_root)
    variant_root.mkdir(parents=True, exist_ok=True)
    result_path = worker_result_path(variant_root)

    install_stdout_progress_hook()
    config = make_config(base_config(args), gru_dropout=float(args._worker_dropout))

    summary = run_possm_phoneme_finetuning(
        checkpoint_path=stage1_checkpoint,
        cache_root=stage2_cache_root,
        output_root=variant_root,
        config=config,
        run_name="run",
        resume_from_latest=bool(args.resume),
    )
    row = flatten_summary_row(variant_name=str(args._worker_variant), summary=summary)
    payload = {
        "variant": str(args._worker_variant),
        "gru_dropout": float(args._worker_dropout),
        "config": asdict(config),
        "summary": summary,
        "row": row,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_text_atomic(result_path, json.dumps(payload, indent=2))
    print("worker result:", json.dumps(row, indent=2), flush=True)


def run_parent(args: argparse.Namespace) -> None:
    dropouts = default_dropouts(args)
    if not dropouts:
        raise ValueError("No dropout values were provided.")

    base = base_config(args)
    sweep_name = args.sweep_name or f"stage2_dropout_ablation_{timestamp_utc()}"
    sweep_dir = Path(args.output_root) / "sweeps" / sweep_name
    manifest_path = sweep_dir / "sweep_manifest.json"
    state_path = sweep_dir / "sweep_state.json"
    csv_path = sweep_dir / "sweep_results.csv"
    results_json_path = sweep_dir / "sweep_results.json"
    sweep_log_path = sweep_dir / "sweep.log"

    if args.dry_run:
        try:
            stage1_checkpoint_preview = str(resolve_stage1_checkpoint(args))
        except Exception as exc:  # pragma: no cover - convenience path for empty local envs
            stage1_checkpoint_preview = f"<unresolved: {exc}>"
        print(
            json.dumps(
                {
                    "stage1_checkpoint": stage1_checkpoint_preview,
                    "stage2_cache_root": str(Path(args.stage2_cache_root)),
                    "sweep_dir": str(sweep_dir),
                    "dropouts": [float(value) for value in dropouts],
                    "base_config": asdict(base),
                },
                indent=2,
            )
        )
        return

    sweep_dir.mkdir(parents=True, exist_ok=True)
    log = make_logger(sweep_log_path)

    if args.resume and manifest_path.exists():
        manifest = load_json(manifest_path)
    else:
        manifest = {}
    manifest.update(
        {
            "sweep_name": sweep_name,
            "created_utc": str(manifest.get("created_utc", datetime.now(timezone.utc).isoformat())),
            "updated_utc": datetime.now(timezone.utc).isoformat(),
            "stage1_checkpoint": None,
            "stage2_cache_root": str(Path(args.stage2_cache_root)),
            "output_root": str(args.output_root),
            "base_config": asdict(base),
            "dropouts": [float(value) for value in dropouts],
            "isolated_subprocess_per_variant": True,
        }
    )

    if args.resume and state_path.exists():
        sweep_state = load_json(state_path)
    else:
        sweep_state = {}
    sweep_state.update(
        {
            "sweep_name": sweep_name,
            "created_utc": str(sweep_state.get("created_utc", datetime.now(timezone.utc).isoformat())),
            "updated_utc": datetime.now(timezone.utc).isoformat(),
            "status_by_variant": dict(sweep_state.get("status_by_variant") or {}),
            "last_error_by_variant": dict(sweep_state.get("last_error_by_variant") or {}),
        }
    )
    write_json(manifest_path, manifest)
    write_json(state_path, sweep_state)

    stage1_checkpoint = resolve_stage1_checkpoint(args)
    stage2_cache_root = Path(args.stage2_cache_root)
    if not stage2_cache_root.exists():
        raise FileNotFoundError(f"Stage-2 cache root does not exist: {stage2_cache_root}")
    manifest["stage1_checkpoint"] = str(stage1_checkpoint)
    manifest["stage2_cache_root"] = str(stage2_cache_root)
    write_json(manifest_path, manifest)

    log(f"stage1 checkpoint: {stage1_checkpoint}")
    log(f"stage2 cache root: {stage2_cache_root}")
    log(f"sweep dir: {sweep_dir}")
    log(f"dropouts: {', '.join(str(float(value)) for value in dropouts)}")
    log(f"batch size: {int(base.batch_size)}")

    rows_by_variant: dict[str, dict[str, Any]] = {}
    results_by_variant: dict[str, dict[str, Any]] = {}

    for dropout in dropouts:
        variant_name = f"gru_dropout_{sanitize_dropout_label(float(dropout))}"
        variant_root = sweep_dir / variant_name
        result_path = worker_result_path(variant_root)

        if args.resume and result_path.exists():
            payload = json.loads(result_path.read_text())
            row = dict(payload.get("row") or {})
            rows_by_variant[variant_name] = row
            results_by_variant[variant_name] = payload
            sweep_state["status_by_variant"][variant_name] = "completed"
            write_json(state_path, sweep_state)
            log(f"skipping completed variant: {variant_name}")
            continue

        sweep_state["status_by_variant"][variant_name] = "running"
        sweep_state["updated_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(state_path, sweep_state)
        log(f"starting variant: {variant_name}")

        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--_worker",
            f"--_worker-variant={variant_name}",
            f"--_worker-dropout={float(dropout)}",
            f"--_worker-variant-root={variant_root}",
            f"--_worker-stage1-checkpoint={stage1_checkpoint}",
            f"--stage2-cache-root={stage2_cache_root}",
            f"--mode={args.mode}",
            f"--dataset={args.dataset}",
            f"--data-mode={args.data_mode}",
            f"--boundary-key-mode={args.boundary_key_mode}",
            f"--batch-size={int(args.batch_size)}",
            f"--num-steps={int(args.num_steps)}",
            f"--learning-rate={float(args.learning_rate)}",
            f"--encoder-learning-rate={float(args.encoder_learning_rate)}",
            f"--weight-decay={float(args.weight_decay)}",
            f"--max-grad-norm={float(args.max_grad_norm)}",
            f"--checkpoint-every-steps={int(args.checkpoint_every_steps)}",
            f"--progress-every-steps={int(args.progress_every_steps)}",
            f"--input-smoothing-sigma-bins={float(args.input_smoothing_sigma_bins)}",
            f"--input-smoothing-kernel-size={int(args.input_smoothing_kernel_size)}",
            f"--input-smoothing-threshold={float(args.input_smoothing_threshold)}",
            f"--white-noise-sd={float(args.white_noise_sd)}",
            f"--constant-offset-sd={float(args.constant_offset_sd)}",
            f"--gru-hidden-size={int(args.gru_hidden_size)}",
            f"--gru-num-layers={int(args.gru_num_layers)}",
            f"--gru-dropout-default={float(args.gru_dropout_default)}",
            f"--decoder-backbone-type={str(args.decoder_backbone_type)}",
            f"--s5-hidden-size={int(args.s5_hidden_size)}",
            f"--s5-state-size={int(args.s5_state_size)}",
            f"--s5-num-layers={int(args.s5_num_layers)}",
            f"--s5-dropout={float(args.s5_dropout)}",
            f"--s5-direction={str(args.s5_direction)}",
            f"--s5-ffn-multiplier={float(args.s5_ffn_multiplier)}",
            f"--conv-kernel-size={int(args.conv_kernel_size)}",
            f"--conv-stride={int(args.conv_stride)}",
            f"--conv-dropout={float(args.conv_dropout)}",
            f"--seed={int(args.seed)}",
        ]
        if bool(args.resume):
            cmd.append("--resume")
        else:
            cmd.append("--no-resume")
        if bool(args.session_adapter_enabled):
            cmd.append("--session-adapter-enabled")

        try:
            subprocess.run(cmd, check=True)
        except Exception as exc:
            sweep_state["status_by_variant"][variant_name] = "failed"
            sweep_state["updated_utc"] = datetime.now(timezone.utc).isoformat()
            sweep_state["last_error_by_variant"][variant_name] = {
                "error": str(exc),
                "failed_utc": datetime.now(timezone.utc).isoformat(),
            }
            write_json(state_path, sweep_state)
            log(f"variant failed: {variant_name} error={exc}")
            if not args.continue_on_error:
                raise
            continue

        payload = json.loads(result_path.read_text())
        row = dict(payload.get("row") or {})
        row["gru_dropout"] = float(dropout)
        rows_by_variant[variant_name] = row
        results_by_variant[variant_name] = payload
        manifest["updated_utc"] = datetime.now(timezone.utc).isoformat()
        manifest["results_by_variant"] = results_by_variant
        manifest["results"] = [results_by_variant[name] for name in sorted(results_by_variant)]
        write_json(manifest_path, manifest)

        sweep_state["status_by_variant"][variant_name] = "completed"
        sweep_state["last_error_by_variant"].pop(variant_name, None)
        sweep_state["updated_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(state_path, sweep_state)
        log(f"variant completed: {variant_name}")

        ordered_rows = [rows_by_variant[name] for name in sorted(rows_by_variant)]
        write_results_csv(csv_path, ordered_rows)
        write_text_atomic(results_json_path, json.dumps(ordered_rows, indent=2))

    ordered_rows = [rows_by_variant[name] for name in sorted(rows_by_variant)]
    write_results_csv(csv_path, ordered_rows)
    write_text_atomic(results_json_path, json.dumps(ordered_rows, indent=2))
    manifest["updated_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["results_by_variant"] = results_by_variant
    manifest["results"] = [results_by_variant[name] for name in sorted(results_by_variant)]
    write_json(manifest_path, manifest)
    log(f"wrote: {manifest_path}")
    log(f"wrote: {results_json_path}")
    log(f"wrote: {csv_path}")
    log(f"wrote: {state_path}")
    log(f"wrote: {sweep_log_path}")


def main() -> None:
    args = parse_args()
    if args._worker:
        run_worker(args)
    else:
        run_parent(args)


if __name__ == "__main__":
    main()

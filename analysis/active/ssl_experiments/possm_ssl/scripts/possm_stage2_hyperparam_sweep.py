"""Run a small POSSM Stage-2 phoneme fine-tuning hyperparameter sweep.

This script is intended to be launched from Colab after the repository is
available on the Python path. It keeps the current best Stage-2 recipe fixed
except for targeted regularization / learning-rate variants while using the
released Willett ``competition_train -> competition_test`` split.
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

EXPERIMENTS_DIR = Path(__file__).resolve().parents[2]
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))

import torch

import possm_ssl.phoneme_finetune as possm_phoneme_finetune
from possm_ssl import POSSMFinetuneConfig, resolve_possm_checkpoint_path, run_possm_phoneme_finetuning


DEFAULT_DRIVE_ROOT = Path("/content/drive/MyDrive")
DEFAULT_OUTPUT_ROOT = DEFAULT_DRIVE_ROOT / "utah_ssl" / "outputs" / "ssl_experiments" / "possm_masked_reconstruction"
DEFAULT_STAGE2_CACHE_ROOT = DEFAULT_DRIVE_ROOT / "utah_ssl" / "data" / "cache_v1"


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
        gru_dropout=float(args.gru_dropout),
        decoder_backbone_type=str(args.decoder_backbone_type),
        s5_hidden_size=int(args.s5_hidden_size),
        s5_state_size=int(args.s5_state_size),
        s5_num_layers=int(args.s5_num_layers),
        s5_dropout=float(args.s5_dropout),
        s5_direction=str(args.s5_direction),
        s5_ffn_multiplier=float(args.s5_ffn_multiplier),
        temporal_patch_kernel_size=int(args.temporal_patch_kernel_size),
        temporal_patch_stride=int(args.temporal_patch_stride),
        conv_dropout=float(args.conv_dropout),
    )


def make_config(base: POSSMFinetuneConfig, **overrides: Any) -> POSSMFinetuneConfig:
    payload = asdict(base)
    payload.update(overrides)
    return POSSMFinetuneConfig(**payload)


def default_variants(base: POSSMFinetuneConfig) -> dict[str, POSSMFinetuneConfig]:
    """Targeted sweep around the current best POSSM Stage-2 run."""

    return {
        "baseline_locked": make_config(base),
        "dropout_0p3": make_config(base, gru_dropout=0.3),
        "weight_decay_3e-3": make_config(base, weight_decay=3e-3),
        "decoder_lr_1e-4": make_config(base, learning_rate=1e-4),
    }


def flatten_summary_row(*, variant_name: str, summary: dict[str, Any]) -> dict[str, Any]:
    metrics = dict(summary.get("metrics", {}))
    collapse = dict(metrics.get("collapse_diagnostics") or {})
    best_collapse = dict(metrics.get("best_collapse_diagnostics") or {})
    return {
        "variant": variant_name,
        "run_name": summary.get("run_name"),
        "run_dir": summary.get("run_dir"),
        "decoder_backbone_type": summary.get("decoder_backbone_type"),
        "s5_direction": summary.get("s5_direction"),
        "s5_hidden_size": summary.get("s5_hidden_size"),
        "s5_state_size": summary.get("s5_state_size"),
        "s5_num_layers": summary.get("s5_num_layers"),
        "s5_dropout": summary.get("s5_dropout"),
        "steps": summary.get("steps"),
        "val_ctc_bpphone": metrics.get("val_ctc_bpphone"),
        "val_phoneme_error_rate": metrics.get("val_phoneme_error_rate"),
        "best_step": metrics.get("best_step"),
        "best_val_ctc_bpphone": metrics.get("best_val_ctc_bpphone"),
        "best_val_phoneme_error_rate": metrics.get("best_val_phoneme_error_rate"),
        "predicted_to_reference_token_ratio": collapse.get("predicted_to_reference_token_ratio"),
        "blank_frame_rate": collapse.get("blank_frame_rate"),
        "best_predicted_to_reference_token_ratio": best_collapse.get("predicted_to_reference_token_ratio"),
        "best_blank_frame_rate": best_collapse.get("blank_frame_rate"),
        "checkpoint_best_path": summary.get("checkpoint_best_path"),
        "checkpoint_final_path": summary.get("checkpoint_final_path"),
    }


def install_stdout_progress_hook() -> None:
    original_emit = getattr(
        possm_phoneme_finetune,
        "_sweep_original_emit_progress",
        possm_phoneme_finetune._emit_progress,
    )
    possm_phoneme_finetune._sweep_original_emit_progress = original_emit

    def _fmt(value: Any, digits: int = 3) -> str:
        if value is None:
            return "nan"
        try:
            return f"{float(value):.{digits}f}"
        except (TypeError, ValueError):
            return str(value)

    def emit_with_stdout(progress_log_path: Path | None, **payload: Any) -> None:
        original_emit(progress_log_path, **payload)
        event = str(payload.get("event", "progress"))
        step = payload.get("step")
        elapsed = _fmt(payload.get("elapsed_seconds"), digits=1)
        if event == "phoneme_train_report":
            train_loss = _fmt(payload.get("train_ctc_bpphone"))
            print(f"[stage2 train] step={step} train_ctc_bpphone={train_loss} elapsed_s={elapsed}", flush=True)
        elif event == "phoneme_resume":
            checkpoint_path = payload.get("resumed_from_checkpoint")
            print(
                f"[stage2 resume] step={step} from={checkpoint_path} elapsed_s={elapsed}",
                flush=True,
            )
        elif event == "phoneme_val_report":
            collapse = dict(payload.get("collapse_diagnostics") or {})
            print(
                "[stage2 val] "
                f"step={step} "
                f"val_ctc_bpphone={_fmt(payload.get('val_ctc_bpphone'))} "
                f"val_PER={_fmt(payload.get('val_phoneme_error_rate'))} "
                f"pred/ref={_fmt(collapse.get('predicted_to_reference_token_ratio'))} "
                f"blank={_fmt(collapse.get('blank_frame_rate'))} "
                f"elapsed_s={elapsed}",
                flush=True,
            )

    possm_phoneme_finetune._emit_progress = emit_with_stdout


def resolve_stage1_checkpoint(args: argparse.Namespace) -> Path:
    if args.stage1_checkpoint is not None:
        checkpoint = Path(args.stage1_checkpoint)
        if not checkpoint.exists():
            raise FileNotFoundError(f"Stage-1 checkpoint does not exist: {checkpoint}")
        return checkpoint
    return resolve_possm_checkpoint_path(output_root=Path(args.stage1_output_root))


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object in {path}, got {type(payload).__name__}")
    return payload


def write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp")
    tmp_path.write_text(text)
    tmp_path.replace(path)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    write_text_atomic(path, json.dumps(payload, indent=2))


def load_saved_result_row(variant_name: str, payload: dict[str, Any]) -> dict[str, Any] | None:
    row = payload.get("row")
    if isinstance(row, dict):
        return dict(row)
    summary = payload.get("summary")
    if isinstance(summary, dict):
        restored = flatten_summary_row(variant_name=variant_name, summary=summary)
        if payload.get("elapsed_seconds") is not None:
            restored["elapsed_seconds"] = payload.get("elapsed_seconds")
        return restored
    if payload.get("run_name") is not None:
        return dict(payload)
    return None


def make_logger(log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)

    def _log(message: str) -> None:
        stamp = datetime.now(timezone.utc).isoformat()
        line = f"[{stamp}] {message}"
        print(line, flush=True)
        with log_path.open("a") as handle:
            handle.write(line + "\n")

    return _log


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
        help="Resume from an existing sweep directory and variant checkpoints when possible.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Keep running remaining variants if one variant fails.",
    )
    parser.add_argument(
        "--variant",
        action="append",
        default=None,
        help="Variant name to run. May be repeated. Defaults to all built-in variants.",
    )
    parser.add_argument("--dry-run", action="store_true")

    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--mode", choices=("probe_frozen", "finetune_full"), default="finetune_full")
    parser.add_argument("--dataset", default="brain2text24")
    parser.add_argument("--feature-mode", choices=("tx_only", "tx_sbp"), default="tx_only")
    parser.add_argument("--data-mode", choices=("raw", "normalized"), default="normalized")
    parser.add_argument("--boundary-key-mode", choices=("session", "subject_if_available"), default="session")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-steps", type=int, default=3000)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--encoder-learning-rate", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
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
    parser.add_argument("--gru-dropout", type=float, default=0.2)
    parser.add_argument("--decoder-backbone-type", choices=("gru", "s5"), default="gru")
    parser.add_argument("--s5-hidden-size", type=int, default=768)
    parser.add_argument("--s5-state-size", type=int, default=128)
    parser.add_argument("--s5-num-layers", type=int, default=5)
    parser.add_argument("--s5-dropout", type=float, default=0.2)
    parser.add_argument("--s5-direction", choices=("causal", "bidirectional"), default="causal")
    parser.add_argument("--s5-ffn-multiplier", type=float, default=2.0)
    parser.add_argument("--temporal-patch-kernel-size", type=int, default=14)
    parser.add_argument("--temporal-patch-stride", type=int, default=4)
    parser.add_argument("--conv-dropout", type=float, default=0.1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base = base_config(args)
    variants = default_variants(base)
    selected_names = list(args.variant) if args.variant else list(variants)
    missing = [name for name in selected_names if name not in variants]
    if missing:
        raise ValueError(f"Unknown variants {missing}. Available: {sorted(variants)}")

    if args.dry_run:
        print("device:", device)
        print("selected variants:", ", ".join(selected_names))
        print(json.dumps({name: asdict(variants[name]) for name in selected_names}, indent=2))
        return

    stage1_checkpoint = resolve_stage1_checkpoint(args)
    stage2_cache_root = Path(args.stage2_cache_root)
    if not stage2_cache_root.exists():
        raise FileNotFoundError(f"Stage-2 cache root does not exist: {stage2_cache_root}")

    sweep_name = args.sweep_name or f"stage2_hparam_sweep_{timestamp_utc()}"
    sweep_dir = Path(args.output_root) / "sweeps" / sweep_name
    sweep_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = sweep_dir / "sweep_manifest.json"
    state_path = sweep_dir / "sweep_state.json"
    csv_path = sweep_dir / "sweep_results.csv"
    results_json_path = sweep_dir / "sweep_results.json"
    sweep_log_path = sweep_dir / "sweep.log"
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
            "device": str(device),
            "stage1_checkpoint": str(stage1_checkpoint),
            "stage2_cache_root": str(stage2_cache_root),
            "output_root": str(args.output_root),
            "base_config": asdict(base),
            "selected_variants": selected_names,
            "variants": {name: asdict(variants[name]) for name in selected_names},
        }
    )
    results_by_variant = dict(manifest.get("results_by_variant") or {})
    normalized_results_by_variant: dict[str, dict[str, Any]] = {}
    for variant_name in selected_names:
        payload = results_by_variant.get(variant_name)
        if not isinstance(payload, dict):
            continue
        row = load_saved_result_row(variant_name, payload)
        if row is None:
            continue
        normalized_results_by_variant[variant_name] = {
            "row": row,
            "summary": payload.get("summary"),
            "elapsed_seconds": row.get("elapsed_seconds"),
            "completed_utc": payload.get("completed_utc"),
        }
    results_by_variant = normalized_results_by_variant
    manifest["results_by_variant"] = results_by_variant
    manifest["results"] = [results_by_variant[name] for name in selected_names if name in results_by_variant]

    if args.resume and state_path.exists():
        sweep_state = load_json(state_path)
    else:
        sweep_state = {}
    sweep_state.update(
        {
            "sweep_name": sweep_name,
            "created_utc": str(sweep_state.get("created_utc", datetime.now(timezone.utc).isoformat())),
            "updated_utc": datetime.now(timezone.utc).isoformat(),
            "selected_variants": selected_names,
            "status_by_variant": dict(sweep_state.get("status_by_variant") or {}),
            "last_error_by_variant": dict(sweep_state.get("last_error_by_variant") or {}),
        }
    )
    write_json(manifest_path, manifest)
    write_json(state_path, sweep_state)

    log(f"stage1 checkpoint: {stage1_checkpoint}")
    log(f"stage2 cache root: {stage2_cache_root}")
    log(f"sweep dir: {sweep_dir}")
    log(f"device: {device}")
    log(f"variants: {', '.join(selected_names)}")
    log(f"resume enabled: {bool(args.resume)}")

    install_stdout_progress_hook()

    rows_by_variant: dict[str, dict[str, Any]] = {
        str(name): dict(results_by_variant[name]["row"])
        for name in selected_names
        if name in results_by_variant and isinstance(results_by_variant[name].get("row"), dict)
    }
    for variant_name in selected_names:
        if args.resume and variant_name in rows_by_variant:
            log(f"skipping completed variant: {variant_name}")
            sweep_state["status_by_variant"][variant_name] = "completed"
            sweep_state["updated_utc"] = datetime.now(timezone.utc).isoformat()
            write_json(state_path, sweep_state)
            continue

        config = variants[variant_name]
        variant_output_root = Path(args.output_root) / sweep_name / variant_name
        variant_output_root.mkdir(parents=True, exist_ok=True)
        sweep_state["status_by_variant"][variant_name] = "running"
        sweep_state["updated_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(state_path, sweep_state)
        log(f"starting variant: {variant_name}")
        print(json.dumps(asdict(config), indent=2), flush=True)

        t0 = time.time()
        try:
            summary = run_possm_phoneme_finetuning(
                checkpoint_path=stage1_checkpoint,
                cache_root=stage2_cache_root,
                output_root=variant_output_root,
                config=config,
                device=device,
                run_name="run",
                resume_from_latest=bool(args.resume),
            )
        except Exception as exc:
            sweep_state["status_by_variant"][variant_name] = "failed"
            sweep_state["updated_utc"] = datetime.now(timezone.utc).isoformat()
            sweep_state["last_error_by_variant"][variant_name] = {
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "failed_utc": datetime.now(timezone.utc).isoformat(),
            }
            write_json(state_path, sweep_state)
            log(f"variant failed: {variant_name} error={exc}")
            if not args.continue_on_error:
                raise
            continue

        elapsed_s = round(time.time() - t0, 3)
        row = flatten_summary_row(variant_name=variant_name, summary=summary)
        row["elapsed_seconds"] = elapsed_s
        rows_by_variant[variant_name] = dict(row)
        results_by_variant[variant_name] = {
            "row": row,
            "summary": summary,
            "elapsed_seconds": elapsed_s,
            "completed_utc": datetime.now(timezone.utc).isoformat(),
        }
        manifest["updated_utc"] = datetime.now(timezone.utc).isoformat()
        manifest["results_by_variant"] = results_by_variant
        manifest["results"] = [results_by_variant[name] for name in selected_names if name in results_by_variant]
        write_json(manifest_path, manifest)

        sweep_state["status_by_variant"][variant_name] = "completed"
        sweep_state["last_error_by_variant"].pop(variant_name, None)
        sweep_state["updated_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(state_path, sweep_state)
        log(f"variant completed: {variant_name}")
        print("variant result:", json.dumps(row, indent=2), flush=True)

        ordered_rows = [rows_by_variant[name] for name in selected_names if name in rows_by_variant]
        write_results_csv(csv_path, ordered_rows)
        write_text_atomic(results_json_path, json.dumps(ordered_rows, indent=2))

    ordered_rows = [rows_by_variant[name] for name in selected_names if name in rows_by_variant]
    write_results_csv(csv_path, ordered_rows)
    write_text_atomic(results_json_path, json.dumps(ordered_rows, indent=2))
    log(f"wrote: {manifest_path}")
    log(f"wrote: {results_json_path}")
    log(f"wrote: {csv_path}")
    log(f"wrote: {state_path}")
    log(f"wrote: {sweep_log_path}")


if __name__ == "__main__":
    main()

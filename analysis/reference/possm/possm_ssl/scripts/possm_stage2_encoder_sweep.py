"""Run a frozen Stage-2 POSSM checkpoint sweep across Stage-1 encoders.

This is intended for quick Colab comparisons when full Stage-2 finetuning is
too slow. Each variant reuses one Stage-1 checkpoint and trains only the Stage-2
phoneme decoder in ``probe_frozen`` mode on the released Willett
``competition_train -> competition_test`` split.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import traceback
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

POSSM_DIR = Path(__file__).resolve().parents[1]
if str(POSSM_DIR) not in sys.path:
    sys.path.insert(0, str(POSSM_DIR))

import torch

from possm_ssl import POSSMFinetuneConfig, resolve_possm_checkpoint_path, run_possm_phoneme_finetuning
from possm_ssl.scripts.possm_stage2_hyperparam_sweep import (
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_STAGE2_CACHE_ROOT,
    flatten_summary_row,
    install_stdout_progress_hook,
    load_json,
    make_logger,
    write_json,
    write_results_csv,
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


def _sanitize_variant_label(label: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(label)).strip("._-")
    return cleaned or "checkpoint"


def _checkpoint_variant_label(checkpoint_path: Path, *, used_labels: set[str]) -> str:
    run_dir = checkpoint_path.parent.parent if checkpoint_path.parent.name == "checkpoints" else checkpoint_path.parent
    base_label = run_dir.name
    if checkpoint_path.name not in {"checkpoint_best.pt", "checkpoint_final.pt"}:
        base_label = f"{base_label}_{checkpoint_path.stem}"
    label = _sanitize_variant_label(base_label)
    resolved = label
    suffix = 2
    while resolved in used_labels:
        resolved = f"{label}_{suffix}"
        suffix += 1
    used_labels.add(resolved)
    return resolved


def _resolve_checkpoint_variants(args: argparse.Namespace) -> list[dict[str, str]]:
    checkpoint_entries: list[Path] = []
    for value in args.stage1_checkpoint:
        checkpoint_entries.append(Path(value))
    for value in args.stage1_run_dir:
        checkpoint_entries.append(resolve_possm_checkpoint_path(run_dir=Path(value)))
    if not checkpoint_entries:
        raise ValueError("Provide at least one --stage1-checkpoint or --stage1-run-dir.")

    used_labels: set[str] = set()
    variants: list[dict[str, str]] = []
    seen_paths: set[str] = set()
    for checkpoint_path in checkpoint_entries:
        resolved_path = checkpoint_path.resolve()
        if not resolved_path.exists():
            raise FileNotFoundError(f"Stage-1 checkpoint does not exist: {resolved_path}")
        resolved_key = str(resolved_path)
        if resolved_key in seen_paths:
            continue
        seen_paths.add(resolved_key)
        variants.append(
            {
                "variant": _checkpoint_variant_label(resolved_path, used_labels=used_labels),
                "checkpoint_path": resolved_key,
            }
        )
    return variants


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage1-checkpoint",
        action="append",
        default=[],
        help="Explicit Stage-1 checkpoint path. May be repeated.",
    )
    parser.add_argument(
        "--stage1-run-dir",
        action="append",
        default=[],
        help="Stage-1 run directory; resolves to checkpoint_best.pt when present. May be repeated.",
    )
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
    parser.add_argument("--dry-run", action="store_true")

    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--mode", choices=("probe_frozen", "finetune_full"), default="probe_frozen")
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
    checkpoint_variants = _resolve_checkpoint_variants(args)
    selected_names = [entry["variant"] for entry in checkpoint_variants]

    if args.dry_run:
        print("device:", device)
        print("selected variants:", ", ".join(selected_names))
        print(json.dumps({"base_config": asdict(base), "checkpoints": checkpoint_variants}, indent=2))
        return

    stage2_cache_root = Path(args.stage2_cache_root)
    if not stage2_cache_root.exists():
        raise FileNotFoundError(f"Stage-2 cache root does not exist: {stage2_cache_root}")

    sweep_name = args.sweep_name or f"stage2_encoder_sweep_{timestamp_utc()}"
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
            "stage2_cache_root": str(stage2_cache_root),
            "output_root": str(args.output_root),
            "base_config": asdict(base),
            "checkpoint_variants": checkpoint_variants,
        }
    )
    requested_checkpoint_by_variant = {
        entry["variant"]: entry["checkpoint_path"] for entry in checkpoint_variants
    }
    results_by_variant = dict(manifest.get("results_by_variant") or {})
    normalized_results_by_variant: dict[str, dict[str, Any]] = {}
    for variant_name in selected_names:
        payload = results_by_variant.get(variant_name)
        if not isinstance(payload, dict):
            continue
        if str(payload.get("checkpoint_path")) != requested_checkpoint_by_variant[variant_name]:
            continue
        row = payload.get("row")
        if not isinstance(row, dict):
            continue
        normalized_results_by_variant[variant_name] = {
            "row": dict(row),
            "summary": payload.get("summary"),
            "elapsed_seconds": row.get("elapsed_seconds"),
            "checkpoint_path": payload.get("checkpoint_path"),
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

    log(f"stage2 cache root: {stage2_cache_root}")
    log(f"sweep dir: {sweep_dir}")
    log(f"device: {device}")
    log(f"mode: {base.mode}")
    log(f"variants: {', '.join(selected_names)}")
    log(f"resume enabled: {bool(args.resume)}")

    install_stdout_progress_hook()

    rows_by_variant: dict[str, dict[str, Any]] = {
        str(name): dict(results_by_variant[name]["row"])
        for name in selected_names
        if name in results_by_variant and isinstance(results_by_variant[name].get("row"), dict)
    }
    checkpoint_by_variant = {
        entry["variant"]: Path(entry["checkpoint_path"]) for entry in checkpoint_variants
    }

    for variant_name in selected_names:
        checkpoint_path = checkpoint_by_variant[variant_name]
        if args.resume and variant_name in rows_by_variant:
            log(f"skipping completed variant: {variant_name}")
            sweep_state["status_by_variant"][variant_name] = "completed"
            sweep_state["updated_utc"] = datetime.now(timezone.utc).isoformat()
            write_json(state_path, sweep_state)
            continue

        variant_output_root = Path(args.output_root) / sweep_name / variant_name
        variant_output_root.mkdir(parents=True, exist_ok=True)
        sweep_state["status_by_variant"][variant_name] = "running"
        sweep_state["updated_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(state_path, sweep_state)
        log(f"starting variant: {variant_name}")
        log(f"stage1 checkpoint: {checkpoint_path}")
        print(json.dumps(asdict(base), indent=2), flush=True)

        t0 = time.time()
        try:
            summary = run_possm_phoneme_finetuning(
                checkpoint_path=checkpoint_path,
                cache_root=stage2_cache_root,
                output_root=variant_output_root,
                config=base,
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
        row["stage1_checkpoint_path"] = str(checkpoint_path)
        rows_by_variant[variant_name] = dict(row)
        results_by_variant[variant_name] = {
            "row": row,
            "summary": summary,
            "elapsed_seconds": elapsed_s,
            "checkpoint_path": str(checkpoint_path),
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

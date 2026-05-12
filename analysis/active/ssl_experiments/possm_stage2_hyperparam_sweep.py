"""Run a small POSSM Stage-2 phoneme fine-tuning hyperparameter sweep.

This script is intended to be launched from Colab after the repository is
available on the Python path. It keeps the current best Stage-2 recipe fixed
except for targeted regularization / learning-rate variants.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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
        session_limit=int(args.session_limit),
        target_session_count=int(args.target_session_count),
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
        "dropout_0p3": make_config(base, gru_dropout=0.3),
        "weight_decay_3e-3": make_config(base, weight_decay=3e-3),
        "weight_decay_1e-2": make_config(base, weight_decay=1e-2),
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage1-checkpoint", type=Path, default=None)
    parser.add_argument("--stage1-output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--stage2-cache-root", type=Path, default=DEFAULT_STAGE2_CACHE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT / "phoneme_finetune")
    parser.add_argument("--sweep-name", default=None)
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
    parser.add_argument("--session-limit", type=int, default=28)
    parser.add_argument("--target-session-count", type=int, default=4)
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
    parser.add_argument("--gru-dropout", type=float, default=0.2)
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

    manifest: dict[str, Any] = {
        "sweep_name": sweep_name,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "device": str(device),
        "stage1_checkpoint": str(stage1_checkpoint),
        "stage2_cache_root": str(stage2_cache_root),
        "output_root": str(args.output_root),
        "base_config": asdict(base),
        "selected_variants": selected_names,
        "variants": {name: asdict(variants[name]) for name in selected_names},
        "results": [],
    }
    (sweep_dir / "sweep_manifest.json").write_text(json.dumps(manifest, indent=2))

    print("stage1 checkpoint:", stage1_checkpoint)
    print("stage2 cache root:", stage2_cache_root)
    print("sweep dir:", sweep_dir)
    print("device:", device)
    print("variants:", ", ".join(selected_names))

    install_stdout_progress_hook()

    rows: list[dict[str, Any]] = []
    for variant_name in selected_names:
        config = variants[variant_name]
        variant_output_root = Path(args.output_root) / sweep_name / variant_name
        variant_output_root.mkdir(parents=True, exist_ok=True)
        print(f"\n=== Running variant: {variant_name} ===", flush=True)
        print(json.dumps(asdict(config), indent=2), flush=True)
        t0 = time.time()
        summary = run_possm_phoneme_finetuning(
            checkpoint_path=stage1_checkpoint,
            cache_root=stage2_cache_root,
            output_root=variant_output_root,
            config=config,
            device=device,
        )
        elapsed_s = round(time.time() - t0, 3)
        row = flatten_summary_row(variant_name=variant_name, summary=summary)
        row["elapsed_seconds"] = elapsed_s
        rows.append(row)
        manifest["results"].append({"variant": variant_name, "elapsed_seconds": elapsed_s, "summary": summary})
        (sweep_dir / "sweep_manifest.json").write_text(json.dumps(manifest, indent=2))
        print("variant result:", json.dumps(row, indent=2), flush=True)

    csv_path = sweep_dir / "sweep_results.csv"
    if rows:
        with csv_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    (sweep_dir / "sweep_results.json").write_text(json.dumps(rows, indent=2))
    print("\nwrote:", sweep_dir / "sweep_manifest.json")
    print("wrote:", sweep_dir / "sweep_results.json")
    print("wrote:", csv_path)


if __name__ == "__main__":
    main()

"""Run a six-job supervised Willett S5/S4D seed comparison.

This launcher is intended for Colab cells. It runs S5 and S4D with three seeds
each, streams the trainer's JSON progress log to stdout, and writes an aggregate
CSV/JSONL summary for quick comparison.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

EXPERIMENTS_DIR = Path(__file__).resolve().parents[2]
ACTIVE_DIR = EXPERIMENTS_DIR.parent
SSM_DIR = ACTIVE_DIR / "transfer_benchmark" / "ssl_autoresearch"
for path in (EXPERIMENTS_DIR, SSM_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from recompute_split_feature_stats import (
    recompute_split_feature_stats,
    resolve_precomputed_split_stats_path,
)


DEFAULT_DRIVE_ROOT = Path("/content/drive/MyDrive")
DEFAULT_CACHE_ROOT = DEFAULT_DRIVE_ROOT / "utah_ssl" / "data" / "cache_v1"
DEFAULT_OUTPUT_ROOT = DEFAULT_DRIVE_ROOT / "utah_ssl" / "outputs" / "willett_s5_s4d_seed_sweep"
DEFAULT_SEEDS = (7, 17, 27)
DEFAULT_BACKBONES = ("s5", "s4d")


def timestamp_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--sweep-name", type=str, default=None)
    parser.add_argument("--dataset", type=str, default="brain2text24")
    parser.add_argument("--feature-mode", choices=("tx_only", "tx_sbp"), default="tx_only")
    parser.add_argument("--boundary-key-mode", choices=("session", "subject_if_available"), default="session")
    parser.add_argument("--normalization-mode", choices=("global", "block", "per_session", "none"), default="global")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--backbones", choices=DEFAULT_BACKBONES, nargs="+", default=list(DEFAULT_BACKBONES))
    parser.add_argument("--max-steps", type=int, default=9000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--val-every-steps", type=int, default=100)
    parser.add_argument("--progress-every-steps", type=int, default=100)
    parser.add_argument("--checkpoint-every-steps", type=int, default=500)
    parser.add_argument("--checkpoint-keep-last", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--min-learning-rate", type=float, default=1e-5)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--adam-epsilon", type=float, default=1e-8)
    parser.add_argument("--max-grad-norm", type=float, default=10.0)
    parser.add_argument("--input-projection-size", type=int, default=256)
    parser.add_argument("--input-projection-dropout", type=float, default=0.2)
    parser.add_argument("--patch-size", type=int, default=14)
    parser.add_argument("--patch-stride", type=int, default=4)
    parser.add_argument("--input-smoothing-sigma-bins", type=float, default=2.0)
    parser.add_argument("--input-smoothing-kernel-size", type=int, default=100)
    parser.add_argument("--input-smoothing-threshold", type=float, default=0.01)
    parser.add_argument("--white-noise-sd", type=float, default=1.0)
    parser.add_argument("--constant-offset-sd", type=float, default=0.2)
    parser.add_argument("--s5-hidden-size", type=int, default=512)
    parser.add_argument("--s5-state-size", type=int, default=128)
    parser.add_argument("--s5-num-layers", type=int, default=5)
    parser.add_argument("--s5-dropout", type=float, default=0.2)
    parser.add_argument("--s5-direction", choices=("causal", "bidirectional"), default="causal")
    parser.add_argument("--s5-ffn-multiplier", type=float, default=2.0)
    parser.add_argument("--s4d-hidden-size", type=int, default=512)
    parser.add_argument("--s4d-state-size", type=int, default=128)
    parser.add_argument("--s4d-num-layers", type=int, default=5)
    parser.add_argument("--s4d-dropout", type=float, default=0.2)
    parser.add_argument("--s4d-direction", choices=("causal", "bidirectional"), default="causal")
    parser.add_argument("--s4d-ffn-multiplier", type=float, default=2.0)
    parser.add_argument("--precomputed-split-stats-path", type=Path, default=None)
    parser.add_argument("--force-recompute-stats", action="store_true")
    parser.add_argument("--resume-latest", action="store_true")
    parser.add_argument("--skip-completed", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def log(message: str) -> None:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    print(f"[{stamp}] {message}", flush=True)


def resolve_stats_path(args: argparse.Namespace) -> Path:
    return resolve_precomputed_split_stats_path(
        cache_root=args.cache_root,
        dataset=str(args.dataset),
        train_split_name="competition_train",
        feature_mode=str(args.feature_mode),
        preferred_path=args.precomputed_split_stats_path,
    )


def ensure_split_stats(args: argparse.Namespace) -> Path:
    stats_path = resolve_stats_path(args)
    sidecar_path = stats_path.with_suffix(".json")
    if args.dry_run:
        return stats_path
    if args.force_recompute_stats or not (stats_path.exists() and sidecar_path.exists()):
        log(f"Recomputing split stats at {stats_path}")
        recompute_split_feature_stats(
            cache_root=args.cache_root,
            output_path=stats_path,
            dataset=str(args.dataset),
            feature_mode=str(args.feature_mode),
            boundary_key_mode=str(args.boundary_key_mode),
            overwrite=True,
        )
    else:
        log(f"Reusing split stats at {stats_path}")
    return stats_path


def sweep_dir(args: argparse.Namespace) -> Path:
    resolved_name = str(args.sweep_name) if args.sweep_name else f"s5_s4d_seed_sweep_{timestamp_utc()}"
    return Path(args.output_root) / resolved_name


def planned_jobs(args: argparse.Namespace) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    for backbone in args.backbones:
        direction = args.s5_direction if backbone == "s5" else args.s4d_direction
        for seed in args.seeds:
            run_name = f"willett_{backbone}_{args.feature_mode}_{direction}_seed{int(seed)}"
            jobs.append(
                {
                    "backbone": str(backbone),
                    "direction": str(direction),
                    "seed": int(seed),
                    "run_name": run_name,
                }
            )
    return jobs


def trainer_command(
    *,
    args: argparse.Namespace,
    stats_path: Path,
    run_output_root: Path,
    job: dict[str, Any],
) -> list[str]:
    backbone = str(job["backbone"])
    cmd = [
        sys.executable,
        "-u",
        "-m",
        "willett_reconstruction.train",
        "--cache-root",
        str(args.cache_root),
        "--output-root",
        str(run_output_root),
        "--run-name",
        str(job["run_name"]),
        "--dataset",
        str(args.dataset),
        "--feature-mode",
        str(args.feature_mode),
        "--boundary-key-mode",
        str(args.boundary_key_mode),
        "--normalization-mode",
        str(args.normalization_mode),
        "--batch-size",
        str(int(args.batch_size)),
        "--max-steps",
        str(int(args.max_steps)),
        "--learning-rate",
        str(float(args.learning_rate)),
        "--min-learning-rate",
        str(float(args.min_learning_rate)),
        "--warmup-steps",
        str(int(args.warmup_steps)),
        "--weight-decay",
        str(float(args.weight_decay)),
        "--adam-epsilon",
        str(float(args.adam_epsilon)),
        "--max-grad-norm",
        str(float(args.max_grad_norm)),
        "--val-every-steps",
        str(int(args.val_every_steps)),
        "--progress-every-steps",
        str(int(args.progress_every_steps)),
        "--checkpoint-every-steps",
        str(int(args.checkpoint_every_steps)),
        "--checkpoint-keep-last",
        str(int(args.checkpoint_keep_last)),
        "--input-projection-size",
        str(int(args.input_projection_size)),
        "--input-projection-dropout",
        str(float(args.input_projection_dropout)),
        "--decoder-backbone-type",
        backbone,
        "--patch-size",
        str(int(args.patch_size)),
        "--patch-stride",
        str(int(args.patch_stride)),
        "--input-smoothing-sigma-bins",
        str(float(args.input_smoothing_sigma_bins)),
        "--input-smoothing-kernel-size",
        str(int(args.input_smoothing_kernel_size)),
        "--input-smoothing-threshold",
        str(float(args.input_smoothing_threshold)),
        "--white-noise-sd",
        str(float(args.white_noise_sd)),
        "--constant-offset-sd",
        str(float(args.constant_offset_sd)),
        "--precomputed-split-stats-path",
        str(stats_path),
        "--seed",
        str(int(job["seed"])),
    ]
    if backbone == "s5":
        cmd.extend(
            [
                "--s5-hidden-size",
                str(int(args.s5_hidden_size)),
                "--s5-state-size",
                str(int(args.s5_state_size)),
                "--s5-num-layers",
                str(int(args.s5_num_layers)),
                "--s5-dropout",
                str(float(args.s5_dropout)),
                "--s5-direction",
                str(args.s5_direction),
                "--s5-ffn-multiplier",
                str(float(args.s5_ffn_multiplier)),
            ]
        )
    else:
        cmd.extend(
            [
                "--s4d-hidden-size",
                str(int(args.s4d_hidden_size)),
                "--s4d-state-size",
                str(int(args.s4d_state_size)),
                "--s4d-num-layers",
                str(int(args.s4d_num_layers)),
                "--s4d-dropout",
                str(float(args.s4d_dropout)),
                "--s4d-direction",
                str(args.s4d_direction),
                "--s4d-ffn-multiplier",
                str(float(args.s4d_ffn_multiplier)),
            ]
        )
    if args.resume_latest:
        cmd.append("--resume-latest")
    return cmd


def pythonpath_env() -> dict[str, str]:
    env = dict(os.environ)
    additions = [str(EXPERIMENTS_DIR), str(SSM_DIR)]
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = os.pathsep.join(additions + ([existing] if existing else []))
    return env


def format_progress(run_name: str, payload: dict[str, Any]) -> str:
    event = str(payload.get("event", "progress"))
    step = payload.get("step")
    elapsed = payload.get("elapsed_seconds")
    elapsed_text = f"{float(elapsed):.1f}s" if elapsed is not None else "nan"
    if event == "willett_train_report":
        train_ctc = payload.get("train_ctc_bpphone")
        lr = payload.get("learning_rate")
        return (
            f"[{run_name}] train step={step} "
            f"ctc={float(train_ctc):.3f} lr={float(lr):.3g} elapsed={elapsed_text}"
        )
    if event == "willett_val_report":
        val_ctc = payload.get("val_ctc_bpphone")
        val_per = payload.get("val_phoneme_error_rate")
        best_ctc = payload.get("best_val_ctc_bpphone")
        best_per = payload.get("best_val_phoneme_error_rate")
        return (
            f"[{run_name}] val step={step} "
            f"ctc={float(val_ctc):.3f} per={float(val_per):.4f} "
            f"best_ctc={float(best_ctc):.3f} best_per={float(best_per):.4f} "
            f"elapsed={elapsed_text}"
        )
    return f"[{run_name}] {json.dumps(payload, sort_keys=True)}"


def stream_new_progress(run_name: str, progress_path: Path, offsets: dict[Path, int]) -> None:
    if not progress_path.exists():
        return
    offset = offsets.get(progress_path, 0)
    with progress_path.open("r") as handle:
        handle.seek(offset)
        lines = handle.readlines()
        offsets[progress_path] = handle.tell()
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError:
            print(f"[{run_name}] progress: {stripped}", flush=True)
            continue
        print(format_progress(run_name, payload), flush=True)


def flatten_summary(job: dict[str, Any], summary: dict[str, Any]) -> dict[str, Any]:
    metrics = dict(summary.get("metrics") or {})
    return {
        "backbone": job.get("backbone"),
        "direction": job.get("direction"),
        "seed": job.get("seed"),
        "run_name": summary.get("run_name"),
        "run_dir": summary.get("run_dir"),
        "steps": summary.get("steps"),
        "best_step": summary.get("best_step"),
        "device": summary.get("device"),
        "trainable_parameters": summary.get("trainable_parameters"),
        "val_ctc_bpphone": metrics.get("val_ctc_bpphone"),
        "val_phoneme_error_rate": metrics.get("val_phoneme_error_rate"),
        "best_val_ctc_bpphone": metrics.get("best_val_ctc_bpphone"),
        "best_val_phoneme_error_rate": metrics.get("best_val_phoneme_error_rate"),
        "checkpoint_best_path": summary.get("checkpoint_best_path"),
        "checkpoint_final_path": summary.get("checkpoint_final_path"),
    }


def write_results_csv(path: Path, rows: list[dict[str, Any]]) -> None:
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


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a") as handle:
        handle.write(json.dumps(payload, default=str) + "\n")


def load_summary(run_dir: Path) -> dict[str, Any]:
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary after run: {summary_path}")
    return json.loads(summary_path.read_text())


def run_job(
    *,
    args: argparse.Namespace,
    stats_path: Path,
    run_output_root: Path,
    job: dict[str, Any],
) -> dict[str, Any]:
    run_name = str(job["run_name"])
    run_dir = run_output_root / run_name
    progress_path = run_dir / "progress.jsonl"
    summary_path = run_dir / "summary.json"
    if args.skip_completed and summary_path.exists():
        log(f"Skipping completed run {run_name}")
        return flatten_summary(job, load_summary(run_dir))

    cmd = trainer_command(args=args, stats_path=stats_path, run_output_root=run_output_root, job=job)
    log(f"Starting {run_name}: {' '.join(cmd)}")
    process = subprocess.Popen(cmd, cwd=str(EXPERIMENTS_DIR), env=pythonpath_env())
    offsets: dict[Path, int] = {}
    while process.poll() is None:
        stream_new_progress(run_name, progress_path, offsets)
        time.sleep(5.0)
    stream_new_progress(run_name, progress_path, offsets)
    return_code = int(process.returncode or 0)
    if return_code != 0:
        raise RuntimeError(f"Run {run_name} failed with return code {return_code}")
    summary = load_summary(run_dir)
    row = flatten_summary(job, summary)
    log(
        f"Finished {run_name}: best_per={row.get('best_val_phoneme_error_rate')} "
        f"final_per={row.get('val_phoneme_error_rate')}"
    )
    return row


def main() -> int:
    args = parse_args()
    jobs = planned_jobs(args)
    resolved_sweep_dir = sweep_dir(args)
    run_output_root = resolved_sweep_dir / "runs"
    results_csv_path = resolved_sweep_dir / "sweep_results.csv"
    summary_jsonl_path = resolved_sweep_dir / "sweep_summary.jsonl"
    stats_path = ensure_split_stats(args)

    print(
        json.dumps(
            {
                "sweep_dir": str(resolved_sweep_dir),
                "run_output_root": str(run_output_root),
                "stats_path": str(stats_path),
                "jobs": jobs,
                "max_steps": int(args.max_steps),
            },
            indent=2,
        ),
        flush=True,
    )
    if args.dry_run:
        for job in jobs:
            cmd = trainer_command(args=args, stats_path=stats_path, run_output_root=run_output_root, job=job)
            print("DRY RUN:", " ".join(cmd), flush=True)
        return 0

    run_output_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for index, job in enumerate(jobs, start=1):
        log(f"Job {index}/{len(jobs)}")
        row = run_job(args=args, stats_path=stats_path, run_output_root=run_output_root, job=job)
        rows.append(row)
        append_jsonl(summary_jsonl_path, row)
        write_results_csv(results_csv_path, rows)
        log(f"Updated aggregate results: {results_csv_path}")

    print("\nSweep complete.", flush=True)
    print(f"CSV: {results_csv_path}", flush=True)
    print(f"JSONL: {summary_jsonl_path}", flush=True)
    print(json.dumps(rows, indent=2, default=str), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

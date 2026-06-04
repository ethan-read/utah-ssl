"""Run a small supervised Willett S4D hyperparameter sweep.

This launcher is intended for Colab cells. It tests a few targeted S4D recipe
variants at 8k steps, streams progress to stdout, and writes aggregate CSV/JSONL
results for quick comparison.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from run_s5_s4d_seed_sweep import (
    DEFAULT_CACHE_ROOT,
    ensure_split_stats,
    log,
    run_job,
    timestamp_utc,
    trainer_command,
    write_results_csv,
    append_jsonl,
)


DEFAULT_OUTPUT_ROOT = Path("/content/drive/MyDrive/utah_ssl/outputs/willett_s4d_hyperparam_sweep")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--sweep-name", type=str, default=None)
    parser.add_argument("--dataset", type=str, default="brain2text24")
    parser.add_argument("--feature-mode", choices=("tx_only", "tx_sbp"), default="tx_only")
    parser.add_argument("--boundary-key-mode", choices=("session", "subject_if_available"), default="session")
    parser.add_argument("--normalization-mode", choices=("global", "block", "per_session", "none"), default="global")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--variants", nargs="+", default=["baseline", "lower_lr", "lower_dropout", "wider"])
    parser.add_argument("--max-steps", type=int, default=8000)
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


def sweep_dir(args: argparse.Namespace) -> Path:
    name = str(args.sweep_name) if args.sweep_name else f"s4d_hyperparam_sweep_{timestamp_utc()}"
    return Path(args.output_root) / name


def variant_overrides() -> dict[str, dict[str, Any]]:
    return {
        "baseline": {},
        "lower_lr": {
            "learning_rate": 5e-4,
        },
        "lower_dropout": {
            "s4d_dropout": 0.1,
        },
        "wider": {
            "s4d_hidden_size": 768,
        },
    }


def planned_jobs(args: argparse.Namespace) -> list[dict[str, Any]]:
    available = variant_overrides()
    jobs: list[dict[str, Any]] = []
    for variant in args.variants:
        if variant not in available:
            raise ValueError(f"Unknown variant {variant!r}. Available variants: {sorted(available)}")
        overrides = dict(available[variant])
        hidden_size = int(overrides.get("s4d_hidden_size", args.s4d_hidden_size))
        dropout = float(overrides.get("s4d_dropout", args.s4d_dropout))
        learning_rate = float(overrides.get("learning_rate", args.learning_rate))
        run_name = (
            f"willett_s4d_{args.feature_mode}_{args.s4d_direction}_"
            f"{variant}_seed{int(args.seed)}"
        )
        jobs.append(
            {
                "backbone": "s4d",
                "direction": str(args.s4d_direction),
                "seed": int(args.seed),
                "variant": str(variant),
                "run_name": run_name,
                "learning_rate": learning_rate,
                "s4d_hidden_size": hidden_size,
                "s4d_dropout": dropout,
                **overrides,
            }
        )
    return jobs


def args_for_job(args: argparse.Namespace, job: dict[str, Any]) -> argparse.Namespace:
    payload = vars(args).copy()
    for key in (
        "learning_rate",
        "min_learning_rate",
        "s4d_hidden_size",
        "s4d_state_size",
        "s4d_num_layers",
        "s4d_dropout",
        "s4d_ffn_multiplier",
    ):
        if key in job:
            payload[key] = job[key]
    return argparse.Namespace(**payload)


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
            job_args = args_for_job(args, job)
            cmd = trainer_command(args=job_args, stats_path=stats_path, run_output_root=run_output_root, job=job)
            print("DRY RUN:", " ".join(cmd), flush=True)
        return 0

    run_output_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for index, job in enumerate(jobs, start=1):
        log(f"Job {index}/{len(jobs)} variant={job['variant']}")
        job_args = args_for_job(args, job)
        row = run_job(args=job_args, stats_path=stats_path, run_output_root=run_output_root, job=job)
        row.update(
            {
                "variant": job["variant"],
                "learning_rate": job_args.learning_rate,
                "s4d_hidden_size": job_args.s4d_hidden_size,
                "s4d_state_size": job_args.s4d_state_size,
                "s4d_num_layers": job_args.s4d_num_layers,
                "s4d_dropout": job_args.s4d_dropout,
                "s4d_ffn_multiplier": job_args.s4d_ffn_multiplier,
            }
        )
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

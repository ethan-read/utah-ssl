"""Run one full SSL autoresearch experiment and append to results.tsv."""

from __future__ import annotations

import argparse
import csv
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from prepare import DEFAULT_PROFILE, OUTPUT_ROOT, resolve_profile


HERE = Path(__file__).resolve().parent
TRAIN_PATH = HERE / "train.py"
RESULTS_PATH = HERE / "results.tsv"
FIELDS = [
    "timestamp",
    "profile",
    "dataset_family",
    "backbone",
    "objective_family",
    "adaptation_regime",
    "patch_size",
    "patch_stride",
    "standardize_scope",
    "post_proj_norm",
    "primary_metric_name",
    "primary_metric_value",
    "status",
    "note",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one full SSL autoresearch experiment")
    parser.add_argument("--profile", default=DEFAULT_PROFILE)
    parser.add_argument("--dataset-family", default="brain2text25")
    parser.add_argument("--backbone", required=True)
    parser.add_argument("--objective-family", required=True)
    parser.add_argument("--adaptation-regime", choices=["A", "B1", "B2"], default="A")
    parser.add_argument("--patch-size", type=int, choices=[1, 3, 5], required=True)
    parser.add_argument("--patch-stride", type=int, choices=[1, 3, 5], required=True)
    parser.add_argument("--standardize-scope", choices=["subject", "session"], default="subject")
    parser.add_argument("--post-proj-norm", choices=["none", "rms"], default="rms")
    parser.add_argument("--max-pretrain-steps", type=int, default=None)
    parser.add_argument("--max-probe-steps", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--note", default="")
    return parser.parse_args()


def ensure_results_file() -> None:
    if RESULTS_PATH.exists():
        with RESULTS_PATH.open() as handle:
            header = handle.readline().rstrip("\n").split("\t")
        if header != FIELDS:
            raise ValueError(
                f"results.tsv schema mismatch at {RESULTS_PATH}. "
                "Delete or rename the existing file before logging new runs."
            )
        return
    with RESULTS_PATH.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t")
        writer.writeheader()


def best_prior_value(metric_name: str) -> float | None:
    if not RESULTS_PATH.exists():
        return None
    best = None
    with RESULTS_PATH.open() as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            if row["status"] not in {"keep", "discard"}:
                continue
            if row["primary_metric_name"] != metric_name:
                continue
            try:
                value = float(row["primary_metric_value"])
            except ValueError:
                continue
            if not math.isfinite(value):
                continue
            if best is None or value < best:
                best = value
    return best


def parse_summary(stdout: str) -> dict[str, str]:
    metrics = {}
    for line in stdout.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        metrics[key.strip()] = value.strip()
    required = {"primary_metric_name", "primary_metric_value"}
    missing = required - set(metrics)
    if missing:
        raise ValueError(f"Missing summary metrics: {', '.join(sorted(missing))}")
    return metrics


def append_row(row: dict[str, str]) -> None:
    ensure_results_file()
    with RESULTS_PATH.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t")
        writer.writerow(row)


def main() -> int:
    args = parse_args()
    if args.patch_stride > args.patch_size:
        raise ValueError("patch_stride must be <= patch_size")

    profile = resolve_profile(args.profile)
    ensure_results_file()

    cmd = [
        sys.executable,
        "-u",
        str(TRAIN_PATH),
        "--profile",
        args.profile,
        "--dataset-family",
        args.dataset_family,
        "--backbone",
        args.backbone,
        "--objective-family",
        args.objective_family,
        "--adaptation-regime",
        args.adaptation_regime,
        "--patch-size",
        str(args.patch_size),
        "--patch-stride",
        str(args.patch_stride),
        "--standardize-scope",
        args.standardize_scope,
        "--post-proj-norm",
        args.post_proj_norm,
    ]
    if args.max_pretrain_steps is not None:
        cmd.extend(["--max-pretrain-steps", str(args.max_pretrain_steps)])
    if args.max_probe_steps is not None:
        cmd.extend(["--max-probe-steps", str(args.max_probe_steps)])
    if args.dry_run:
        cmd.append("--dry-run")

    timeout_seconds = profile.pretrain_budget_seconds + profile.probe_budget_seconds + 10 * 60
    completed = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_seconds)
    if completed.stdout:
        print(completed.stdout, end="")
    if completed.stderr:
        print(completed.stderr, end="", file=sys.stderr)

    row = {
        "timestamp": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "profile": args.profile,
        "dataset_family": args.dataset_family,
        "backbone": args.backbone,
        "objective_family": args.objective_family,
        "adaptation_regime": args.adaptation_regime,
        "patch_size": str(args.patch_size),
        "patch_stride": str(args.patch_stride),
        "standardize_scope": args.standardize_scope,
        "post_proj_norm": args.post_proj_norm,
        "primary_metric_name": "",
        "primary_metric_value": "",
        "status": "crash",
        "note": args.note,
    }

    if completed.returncode == 0:
        try:
            metrics = parse_summary(completed.stdout)
            row["primary_metric_name"] = metrics["primary_metric_name"]
            row["primary_metric_value"] = metrics["primary_metric_value"]
            value = float(metrics["primary_metric_value"])
            if math.isfinite(value):
                prior = best_prior_value(metrics["primary_metric_name"])
                row["status"] = "keep" if prior is None or value < prior else "discard"
            else:
                row["status"] = "pending"
        except ValueError as exc:
            print(f"Failed to parse summary: {exc}", file=sys.stderr)

    append_row(row)
    print(f"results_path: {RESULTS_PATH}")
    print(f"artifact_output_root: {OUTPUT_ROOT}")
    print(f"logged_status: {row['status']}")
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())

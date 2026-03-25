"""Run one local SSL autoresearch smoke-test experiment and append to results.tsv."""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


HERE = Path(__file__).resolve().parent
RESULTS_PATH = HERE / "results.tsv"
TRAIN_PATH = HERE / "train.py"
FIELDS = [
    "timestamp",
    "patch_size",
    "patch_stride",
    "hidden_size",
    "num_layers",
    "learning_rate",
    "batch_size",
    "standardize_scope",
    "post_proj_norm",
    "val_ssl_loss",
    "status",
    "note",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one local SSL autoresearch experiment")
    parser.add_argument("--patch-size", type=int, choices=[1, 3, 5], required=True)
    parser.add_argument("--patch-stride", type=int, choices=[1, 3, 5], required=True)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--standardize-scope", choices=["subject", "session"], default="subject")
    parser.add_argument("--post-proj-norm", choices=["none", "rms"], default="rms")
    parser.add_argument("--note", default="", help="Short note about what this run is testing")
    return parser.parse_args()


def ensure_results_file() -> None:
    if RESULTS_PATH.exists():
        return
    with RESULTS_PATH.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t")
        writer.writeheader()


def best_prior_loss() -> float | None:
    if not RESULTS_PATH.exists():
        return None
    best = None
    with RESULTS_PATH.open() as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            if row["status"] == "crash":
                continue
            try:
                loss = float(row["val_ssl_loss"])
            except ValueError:
                continue
            if best is None or loss < best:
                best = loss
    return best


def parse_summary(stdout: str) -> dict[str, str]:
    metrics = {}
    for line in stdout.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        metrics[key.strip()] = value.strip()
    required = {"val_ssl_loss"}
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

    ensure_results_file()
    prior_best = best_prior_loss()

    cmd = [
        sys.executable,
        "-u",
        str(TRAIN_PATH),
        "--patch-size",
        str(args.patch_size),
        "--patch-stride",
        str(args.patch_stride),
        "--hidden-size",
        str(args.hidden_size),
        "--num-layers",
        str(args.num_layers),
        "--learning-rate",
        str(args.learning_rate),
        "--batch-size",
        str(args.batch_size),
        "--standardize-scope",
        args.standardize_scope,
        "--post-proj-norm",
        args.post_proj_norm,
    ]

    completed = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=(5 * 60) + 300,
    )
    if completed.stdout:
        print(completed.stdout, end="")
    if completed.stderr:
        print(completed.stderr, end="", file=sys.stderr)

    row = {
        "timestamp": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "patch_size": str(args.patch_size),
        "patch_stride": str(args.patch_stride),
        "hidden_size": str(args.hidden_size),
        "num_layers": str(args.num_layers),
        "learning_rate": str(args.learning_rate),
        "batch_size": str(args.batch_size),
        "standardize_scope": args.standardize_scope,
        "post_proj_norm": args.post_proj_norm,
        "val_ssl_loss": "",
        "status": "crash",
        "note": args.note,
    }

    if completed.returncode == 0:
        try:
            metrics = parse_summary(completed.stdout)
            row["val_ssl_loss"] = metrics["val_ssl_loss"]
            current_loss = float(metrics["val_ssl_loss"])
            row["status"] = "keep" if prior_best is None or current_loss < prior_best else "discard"
        except ValueError as exc:
            print(f"Failed to parse summary: {exc}", file=sys.stderr)

    append_row(row)
    print(f"logged to {RESULTS_PATH.name}: status={row['status']}")
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())

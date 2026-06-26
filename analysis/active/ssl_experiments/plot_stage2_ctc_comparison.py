"""Plot stage-2 CTC fine-tuning curves from progress logs.

Example:

python analysis/active/ssl_experiments/plot_stage2_ctc_comparison.py \
  --pretrained /path/to/ctc_pretrained_tx_only/progress.jsonl \
  --random-init /path/to/ctc_random_init_tx_only/progress.jsonl \
  --output /tmp/ctc_stage2_comparison.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    resolved = Path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"Progress log does not exist: {resolved}")
    return [
        json.loads(line)
        for line in resolved.read_text().splitlines()
        if line.strip()
    ]


def _split_records(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train_records = [
        row for row in records
        if str(row.get("event", "")).startswith("ctc_train_")
        and "train_ctc_bpphone" in row
    ]
    val_records = [
        row for row in records
        if str(row.get("event", "")).startswith("ctc_val_")
        and "val_ctc_bpphone" in row
    ]
    return train_records, val_records


def _plot_run(
    *,
    axes: list[Any],
    train_records: list[dict[str, Any]],
    val_records: list[dict[str, Any]],
    label: str,
    color: str,
) -> None:
    if train_records:
        axes[0].plot(
            [int(row["step"]) for row in train_records],
            [float(row["train_ctc_bpphone"]) for row in train_records],
            label=label,
            color=color,
            alpha=0.9,
        )
    if val_records:
        axes[1].plot(
            [int(row["step"]) for row in val_records],
            [float(row["val_ctc_bpphone"]) for row in val_records],
            marker="o",
            markersize=3,
            linewidth=1.5,
            label=label,
            color=color,
        )
        axes[2].plot(
            [int(row["step"]) for row in val_records],
            [float(row["val_phoneme_error_rate"]) for row in val_records],
            marker="o",
            markersize=3,
            linewidth=1.5,
            label=label,
            color=color,
        )


def _best_by(records: list[dict[str, Any]], key: str) -> dict[str, Any] | None:
    candidates = [row for row in records if key in row]
    if not candidates:
        return None
    return min(candidates, key=lambda row: float(row[key]))


def _best_summary(val_records: list[dict[str, Any]]) -> dict[str, Any]:
    best_ctc = _best_by(val_records, "val_ctc_bpphone")
    best_per = _best_by(val_records, "val_phoneme_error_rate")
    return {
        "best_val_ctc_bpphone": None if best_ctc is None else float(best_ctc["val_ctc_bpphone"]),
        "best_val_ctc_step": None if best_ctc is None else int(best_ctc["step"]),
        "best_val_phoneme_error_rate": None if best_per is None else float(best_per["val_phoneme_error_rate"]),
        "best_val_per_step": None if best_per is None else int(best_per["step"]),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pretrained", required=True, help="Path to pretrained progress.jsonl")
    parser.add_argument("--random-init", required=True, help="Path to random-init progress.jsonl")
    parser.add_argument("--output", required=True, help="Output PNG path")
    parser.add_argument(
        "--title",
        default="Stage-2 CTC Fine-Tuning Comparison",
        help="Figure title",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    pretrained_records = _read_jsonl(args.pretrained)
    random_records = _read_jsonl(args.random_init)

    pretrained_train, pretrained_val = _split_records(pretrained_records)
    random_train, random_val = _split_records(random_records)

    fig, axes = plt.subplots(3, 1, figsize=(10, 11), sharex=True)

    _plot_run(
        axes=axes,
        train_records=pretrained_train,
        val_records=pretrained_val,
        label="pretrained",
        color="#1f77b4",
    )
    _plot_run(
        axes=axes,
        train_records=random_train,
        val_records=random_val,
        label="random-init",
        color="#d62728",
    )

    axes[0].set_ylabel("train CTC bits/phoneme")
    axes[0].set_title("Training CTC")
    axes[1].set_ylabel("val CTC bits/phoneme")
    axes[1].set_title("Validation CTC")
    axes[2].set_ylabel("val PER")
    axes[2].set_xlabel("step")
    axes[2].set_title("Validation PER")

    for ax in axes:
        ax.grid(True, alpha=0.25)
        ax.legend()

    fig.suptitle(str(args.title))
    fig.tight_layout()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    print(json.dumps({
        "output_path": str(output_path),
        "pretrained_points": {
            "train": len(pretrained_train),
            "val": len(pretrained_val),
        },
        "pretrained_best": _best_summary(pretrained_val),
        "random_init_points": {
            "train": len(random_train),
            "val": len(random_val),
        },
        "random_init_best": _best_summary(random_val),
    }, indent=2))


if __name__ == "__main__":
    main()

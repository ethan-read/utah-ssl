"""Small progress and summary helpers used by active experiment scripts."""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any, Iterable


def append_jsonl(path: str | Path, payload: dict[str, Any]) -> None:
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with resolved.open("a") as handle:
        handle.write(json.dumps(payload, default=str) + "\n")


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    resolved = Path(path)
    if not resolved.exists():
        return []
    return [json.loads(line) for line in resolved.read_text().splitlines() if line.strip()]


def write_metrics_csv(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    materialized = list(rows)
    if not materialized:
        resolved.write_text("")
        return
    fieldnames = sorted({key for row in materialized for key in row.keys()})
    with resolved.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in materialized:
            writer.writerow(row)


class ProgressPrinter:
    def __init__(self, *, every_steps: int = 25, every_seconds: float = 30.0) -> None:
        self.every_steps = max(1, int(every_steps))
        self.every_seconds = float(every_seconds)
        self._last_print_time = 0.0

    def should_print(self, step: int, *, final_step: int | None = None) -> bool:
        now = time.time()
        if int(step) == 1:
            self._last_print_time = now
            return True
        if final_step is not None and int(step) >= int(final_step):
            self._last_print_time = now
            return True
        if int(step) % self.every_steps == 0:
            self._last_print_time = now
            return True
        if now - self._last_print_time >= self.every_seconds:
            self._last_print_time = now
            return True
        return False

    def print(self, *, prefix: str, step: int, total_steps: int, metrics: dict[str, Any]) -> None:
        metric_text = " ".join(
            f"{key}={value:.4g}" if isinstance(value, float) else f"{key}={value}"
            for key, value in sorted(metrics.items())
        )
        print(f"[{prefix}] step {int(step)}/{int(total_steps)} {metric_text}", flush=True)


__all__ = [
    "ProgressPrinter",
    "append_jsonl",
    "load_jsonl",
    "write_metrics_csv",
]

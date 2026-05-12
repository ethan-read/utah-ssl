"""Compute TX-only dataset summary stats from the canonical cache."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


DEFAULT_CACHE_ROOT = Path("/Users/home/thesis/data/cache_v1")
DEFAULT_SUSPICIOUS_VALUES = (255, 65535)


@dataclass(frozen=True)
class DatasetTxStats:
    dataset: str
    modalities: list[str] | None
    num_sessions: int | None
    total_examples: int | None
    total_time_bins_meta: int | None
    total_time_bins_arrays: int
    n_channels_set: list[int]
    tx_dtype: list[str]
    mean_count_per_channel_bin: float
    p_nonzero_channel_bin: float
    p_ge2_channel_bin: float
    active_fraction_per_bin_p50_sample: float
    active_fraction_per_bin_p90_sample: float
    active_fraction_per_bin_p99_sample: float
    mean_count_per_bin_p50_sample: float
    mean_count_per_bin_p90_sample: float
    mean_count_per_bin_p99_sample: float
    example_bins_p50: int
    example_bins_p90: int
    example_bins_p99: int
    lag1_population_mean_count_corr: float
    max_tx_count: int
    suspicious_channel_bin_fraction: float
    suspicious_time_bin_fraction: float
    suspicious_shards: int
    population_sample_rows: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--sample-cap", type=int, default=50_000)
    parser.add_argument("--chunk-rows", type=int, default=4096)
    parser.add_argument(
        "--suspicious-value",
        type=int,
        action="append",
        default=None,
        help="Channel-bin values to treat as suspicious sentinels. May be passed multiple times.",
    )
    parser.add_argument(
        "--exclude-dataset",
        action="append",
        default=[],
        help="Dataset names to skip.",
    )
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def _quantile(values: np.ndarray, q: float) -> float:
    return float(np.quantile(values, q))


def _iter_datasets(cache_root: Path, exclude: Iterable[str]) -> list[Path]:
    excluded = set(exclude)
    return sorted(path for path in cache_root.iterdir() if path.is_dir() and path.name not in excluded)


def _reservoir_push(
    rng: np.random.Generator,
    reservoir_a: np.ndarray,
    reservoir_b: np.ndarray,
    sample_size: int,
    seen: int,
    values_a: np.ndarray,
    values_b: np.ndarray,
) -> tuple[int, int]:
    for value_a, value_b in zip(values_a, values_b):
        seen += 1
        if sample_size < reservoir_a.shape[0]:
            reservoir_a[sample_size] = value_a
            reservoir_b[sample_size] = value_b
            sample_size += 1
            continue
        idx = int(rng.integers(seen))
        if idx < reservoir_a.shape[0]:
            reservoir_a[idx] = value_a
            reservoir_b[idx] = value_b
    return sample_size, seen


def _summarize_dataset(
    dataset_root: Path,
    *,
    sample_cap: int,
    chunk_rows: int,
    suspicious_values: tuple[int, ...],
    rng: np.random.Generator,
) -> DatasetTxStats:
    metadata = json.loads((dataset_root / "metadata.json").read_text())
    manifest_path = dataset_root / "manifest.jsonl"

    shard_relpaths: list[str] = []
    example_lengths: list[int] = []
    with manifest_path.open() as handle:
        for line in handle:
            payload = json.loads(line)
            shard_relpaths.append(str(payload["shard_relpath"]))
            example_lengths.append(int(payload["n_time_bins"]))

    unique_shards = sorted(set(shard_relpaths))
    example_lengths_arr = np.asarray(example_lengths, dtype=np.int64)

    suspicious_values_arr = np.asarray(sorted(set(suspicious_values)), dtype=np.int64)
    total_rows = 0
    total_channel_bins = 0
    total_sum = 0
    total_positive = 0
    total_ge2 = 0
    total_suspicious = 0
    suspicious_rows = 0
    suspicious_shards = 0
    max_tx_count = 0
    dtype_names: set[str] = set()
    n_channels_set: set[int] = set()
    total_time_bins_arrays = 0

    lag_pairs = 0
    lag_sum_x = 0.0
    lag_sum_y = 0.0
    lag_sum_x2 = 0.0
    lag_sum_y2 = 0.0
    lag_sum_xy = 0.0

    reservoir_active_fraction = np.empty(sample_cap, dtype=np.float32)
    reservoir_mean_count = np.empty(sample_cap, dtype=np.float32)
    sample_size = 0
    seen = 0

    for shard_relpath in unique_shards:
        tx_path = dataset_root.parent / shard_relpath / "tx.npy"
        if not tx_path.exists():
            continue

        tx = np.load(tx_path, mmap_mode="r", allow_pickle=False)
        if tx.ndim != 2:
            raise ValueError(f"Expected 2D tx array at {tx_path}, got shape {tx.shape}")

        dtype_names.add(str(tx.dtype))
        total_time_bins_arrays += int(tx.shape[0])
        n_channels_set.add(int(tx.shape[1]))
        shard_has_suspicious = False
        prev_mean = None

        for start in range(0, tx.shape[0], chunk_rows):
            chunk = np.asarray(tx[start : start + chunk_rows], dtype=np.int64)
            n_rows, n_channels = chunk.shape

            total_rows += int(n_rows)
            total_channel_bins += int(chunk.size)
            total_sum += int(chunk.sum())
            max_tx_count = max(max_tx_count, int(chunk.max(initial=0)))

            positive = chunk > 0
            total_positive += int(positive.sum())
            total_ge2 += int((chunk >= 2).sum())

            suspicious_mask = np.isin(chunk, suspicious_values_arr)
            suspicious_count = int(suspicious_mask.sum())
            total_suspicious += suspicious_count
            suspicious_rows += int(suspicious_mask.any(axis=1).sum())
            shard_has_suspicious = shard_has_suspicious or suspicious_count > 0

            active_fraction = positive.mean(axis=1, dtype=np.float64).astype(np.float32)
            mean_count = chunk.mean(axis=1, dtype=np.float64).astype(np.float32)
            sample_size, seen = _reservoir_push(
                rng,
                reservoir_active_fraction,
                reservoir_mean_count,
                sample_size,
                seen,
                active_fraction,
                mean_count,
            )

            if prev_mean is not None and len(mean_count) > 0:
                x = np.concatenate([[prev_mean], mean_count[:-1]])
                y = mean_count
            else:
                x = mean_count[:-1]
                y = mean_count[1:]
            if len(x) > 0:
                lag_pairs += len(x)
                lag_sum_x += float(x.sum())
                lag_sum_y += float(y.sum())
                lag_sum_x2 += float(np.square(x).sum())
                lag_sum_y2 += float(np.square(y).sum())
                lag_sum_xy += float((x * y).sum())
            if len(mean_count) > 0:
                prev_mean = float(mean_count[-1])

        if shard_has_suspicious:
            suspicious_shards += 1

    if total_channel_bins == 0:
        raise ValueError(f"No tx.npy arrays found under {dataset_root}")

    active_sample = reservoir_active_fraction[:sample_size]
    mean_count_sample = reservoir_mean_count[:sample_size]

    lag_cov = lag_sum_xy - (lag_sum_x * lag_sum_y / lag_pairs)
    lag_var_x = lag_sum_x2 - (lag_sum_x * lag_sum_x / lag_pairs)
    lag_var_y = lag_sum_y2 - (lag_sum_y * lag_sum_y / lag_pairs)
    lag_corr = lag_cov / max((lag_var_x * lag_var_y) ** 0.5, 1e-12)

    return DatasetTxStats(
        dataset=dataset_root.name,
        modalities=metadata.get("modalities"),
        num_sessions=metadata.get("num_sessions"),
        total_examples=metadata.get("total_examples"),
        total_time_bins_meta=metadata.get("total_time_bins"),
        total_time_bins_arrays=total_time_bins_arrays,
        n_channels_set=sorted(n_channels_set),
        tx_dtype=sorted(dtype_names),
        mean_count_per_channel_bin=round(total_sum / total_channel_bins, 6),
        p_nonzero_channel_bin=round(total_positive / total_channel_bins, 6),
        p_ge2_channel_bin=round(total_ge2 / total_channel_bins, 6),
        active_fraction_per_bin_p50_sample=round(_quantile(active_sample, 0.50), 6),
        active_fraction_per_bin_p90_sample=round(_quantile(active_sample, 0.90), 6),
        active_fraction_per_bin_p99_sample=round(_quantile(active_sample, 0.99), 6),
        mean_count_per_bin_p50_sample=round(_quantile(mean_count_sample, 0.50), 6),
        mean_count_per_bin_p90_sample=round(_quantile(mean_count_sample, 0.90), 6),
        mean_count_per_bin_p99_sample=round(_quantile(mean_count_sample, 0.99), 6),
        example_bins_p50=int(np.quantile(example_lengths_arr, 0.50)),
        example_bins_p90=int(np.quantile(example_lengths_arr, 0.90)),
        example_bins_p99=int(np.quantile(example_lengths_arr, 0.99)),
        lag1_population_mean_count_corr=round(float(lag_corr), 6),
        max_tx_count=max_tx_count,
        suspicious_channel_bin_fraction=round(total_suspicious / total_channel_bins, 6),
        suspicious_time_bin_fraction=round(suspicious_rows / total_rows, 6),
        suspicious_shards=suspicious_shards,
        population_sample_rows=int(sample_size),
    )


def _format_table(rows: list[DatasetTxStats]) -> str:
    headers = [
        "dataset",
        "chan",
        "mean",
        "p>0",
        "p>=2",
        "active_p50",
        "active_p90",
        "lag1",
        "max",
        "susp_frac",
    ]
    body = []
    for row in rows:
        chan = ",".join(str(value) for value in row.n_channels_set)
        body.append(
            [
                row.dataset,
                chan,
                f"{row.mean_count_per_channel_bin:.3f}",
                f"{row.p_nonzero_channel_bin:.3f}",
                f"{row.p_ge2_channel_bin:.3f}",
                f"{row.active_fraction_per_bin_p50_sample:.3f}",
                f"{row.active_fraction_per_bin_p90_sample:.3f}",
                f"{row.lag1_population_mean_count_corr:.3f}",
                str(row.max_tx_count),
                f"{row.suspicious_channel_bin_fraction:.4f}",
            ]
        )

    widths = [
        max(len(headers[idx]), max((len(row[idx]) for row in body), default=0))
        for idx in range(len(headers))
    ]
    lines = [
        " ".join(header.ljust(widths[idx]) for idx, header in enumerate(headers)),
        " ".join("-" * widths[idx] for idx in range(len(headers))),
    ]
    lines.extend(" ".join(cell.ljust(widths[idx]) for idx, cell in enumerate(row)) for row in body)
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    suspicious_values = tuple(args.suspicious_value or DEFAULT_SUSPICIOUS_VALUES)
    rng = np.random.default_rng(7)

    rows = [
        _summarize_dataset(
            dataset_root,
            sample_cap=int(args.sample_cap),
            chunk_rows=int(args.chunk_rows),
            suspicious_values=suspicious_values,
            rng=rng,
        )
        for dataset_root in _iter_datasets(args.cache_root, args.exclude_dataset)
    ]

    print(_format_table(rows))
    payload = [asdict(row) for row in rows]
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2))
        print(f"\nwrote {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

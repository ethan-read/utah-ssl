"""Generate TX comparison plots and a short-session SBP trace from the canonical cache."""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import sys

EXPERIMENTS_DIR = Path(__file__).resolve().parents[2]
REPO_ROOT = Path(__file__).resolve().parents[5]
for _path in (REPO_ROOT, EXPERIMENTS_DIR):
    _path_str = str(_path)
    if _path_str not in sys.path:
        sys.path.insert(0, _path_str)

os.environ.setdefault("MPLCONFIGDIR", str(Path(".mplconfig").resolve()))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from ssl_core.scripts.tx_dataset_stats import DEFAULT_CACHE_ROOT, DEFAULT_SUSPICIOUS_VALUES


TX_PLOT_DATASETS_DEFAULT = (
    "000950",
    "brain2text24",
    "brain2text25",
    "plug_n_play",
    "unsupervised_cursor_recalibration_offline",
    "unsupervised_cursor_recalibration_online",
    "willett_handwriting",
)
SBP_DATASETS_DEFAULT = ("brain2text24", "brain2text25")


@dataclass(frozen=True)
class PlotDatasetSummary:
    dataset: str
    n_channels_set: tuple[int, ...]
    tx_count_hist: np.ndarray
    active_fraction_hist: np.ndarray
    active_fraction_edges: np.ndarray
    session_sparsity: list[tuple[str, float]]


@dataclass(frozen=True)
class SbpSessionTrace:
    dataset: str
    session_id: str
    time_ms: np.ndarray
    avg_sbp: np.ndarray
    avg_sbp_smooth: np.ndarray
    total_bins: int
    n_channels: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=list(TX_PLOT_DATASETS_DEFAULT),
        help="Datasets to include in the TX comparison plots.",
    )
    parser.add_argument(
        "--sbp-datasets",
        nargs="*",
        default=list(SBP_DATASETS_DEFAULT),
        help="Datasets to search when picking a short SBP session.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/transfer_benchmark/ssl_autoresearch/figures/tx_signal_stats"),
    )
    parser.add_argument("--chunk-rows", type=int, default=4096)
    parser.add_argument("--tx-max-count", type=int, default=5)
    parser.add_argument("--active-fraction-bins", type=int, default=40)
    parser.add_argument(
        "--suspicious-value",
        type=int,
        action="append",
        default=None,
        help="Channel-bin values treated as suspicious sentinels.",
    )
    parser.add_argument(
        "--max-suspicious-fraction",
        type=float,
        default=0.0,
        help="Skip datasets with suspicious sentinel prevalence above this threshold.",
    )
    parser.add_argument("--sbp-smooth-bins", type=int, default=25)
    parser.add_argument("--sbp-max-total-bins", type=int, default=8000)
    return parser.parse_args()


def _load_manifest_rows(dataset_root: Path) -> list[dict]:
    with (dataset_root / "manifest.jsonl").open() as handle:
        return [json.loads(line) for line in handle]


def _dataset_suspicious_fraction(
    dataset_root: Path,
    *,
    chunk_rows: int,
    suspicious_values: tuple[int, ...],
) -> float:
    rows = _load_manifest_rows(dataset_root)
    unique_shards = sorted(set(str(row["shard_relpath"]) for row in rows))
    suspicious_values_arr = np.asarray(sorted(set(suspicious_values)), dtype=np.int64)

    suspicious = 0
    total = 0
    for shard_relpath in unique_shards:
        tx = np.load(dataset_root.parent / shard_relpath / "tx.npy", mmap_mode="r", allow_pickle=False)
        for start in range(0, tx.shape[0], chunk_rows):
            chunk = np.asarray(tx[start : start + chunk_rows], dtype=np.int64)
            suspicious += int(np.isin(chunk, suspicious_values_arr).sum())
            total += int(chunk.size)
    return float(suspicious / max(total, 1))


def _collect_tx_plot_summary(
    dataset_root: Path,
    *,
    chunk_rows: int,
    tx_max_count: int,
    active_fraction_bins: int,
) -> PlotDatasetSummary:
    rows = _load_manifest_rows(dataset_root)
    unique_shards = sorted(set(str(row["shard_relpath"]) for row in rows))
    session_id_by_shard = {
        str(row["shard_relpath"]): str(row["session_id"])
        for row in rows
    }
    metadata = json.loads((dataset_root / "metadata.json").read_text())

    tx_count_hist = np.zeros(tx_max_count + 2, dtype=np.int64)
    active_fraction_edges = np.linspace(0.0, 1.0, active_fraction_bins + 1)
    active_fraction_hist = np.zeros(active_fraction_bins, dtype=np.int64)
    session_totals: dict[str, dict[str, int]] = defaultdict(lambda: {"positive": 0, "total": 0})
    n_channels_set: set[int] = set()

    for shard_relpath in unique_shards:
        shard_path = dataset_root.parent / shard_relpath
        tx = np.load(shard_path / "tx.npy", mmap_mode="r", allow_pickle=False)
        n_channels_set.add(int(tx.shape[1]))
        session_id = session_id_by_shard[shard_relpath]

        for start in range(0, tx.shape[0], chunk_rows):
            chunk = np.asarray(tx[start : start + chunk_rows], dtype=np.int64)
            clipped = np.clip(chunk, 0, tx_max_count + 1)
            tx_count_hist += np.bincount(clipped.ravel(), minlength=tx_max_count + 2)

            positive = chunk > 0
            active_fraction = positive.mean(axis=1, dtype=np.float64)
            hist, _ = np.histogram(active_fraction, bins=active_fraction_edges)
            active_fraction_hist += hist

            session_totals[session_id]["positive"] += int(positive.sum())
            session_totals[session_id]["total"] += int(chunk.size)

    session_sparsity = sorted(
        (
            session_id,
            session_totals[session_id]["positive"] / max(session_totals[session_id]["total"], 1),
        )
        for session_id in session_totals
    )

    return PlotDatasetSummary(
        dataset=str(metadata.get("dataset_family", dataset_root.name)),
        n_channels_set=tuple(sorted(n_channels_set)),
        tx_count_hist=tx_count_hist,
        active_fraction_hist=active_fraction_hist,
        active_fraction_edges=active_fraction_edges,
        session_sparsity=session_sparsity,
    )

def _pick_short_sbp_session(
    cache_root: Path,
    datasets: list[str],
    *,
    sbp_max_total_bins: int,
    smooth_bins: int,
) -> SbpSessionTrace:
    candidates: list[tuple[int, str, str, list[str]]] = []
    for dataset_name in datasets:
        dataset_root = cache_root / dataset_name
        if not dataset_root.exists():
            continue
        rows = _load_manifest_rows(dataset_root)
        session_shards: dict[str, set[str]] = defaultdict(set)
        for row in rows:
            if not bool(row.get("has_sbp", False)):
                continue
            session_shards[str(row["session_id"])].add(str(row["shard_relpath"]))

        for session_id, shard_relpaths in session_shards.items():
            total_bins = 0
            valid = True
            for shard_relpath in shard_relpaths:
                sbp_path = cache_root / shard_relpath / "sbp.npy"
                if not sbp_path.exists():
                    valid = False
                    break
                sbp = np.load(sbp_path, mmap_mode="r", allow_pickle=False)
                total_bins += int(sbp.shape[0])
            if valid and total_bins <= sbp_max_total_bins:
                candidates.append((total_bins, dataset_name, session_id, sorted(shard_relpaths)))

    if not candidates:
        raise RuntimeError("No short SBP session found under the requested datasets.")

    total_bins, dataset_name, session_id, shard_relpaths = sorted(candidates)[0]
    sbp_parts = []
    for shard_relpath in shard_relpaths:
        sbp = np.load(cache_root / shard_relpath / "sbp.npy", mmap_mode="r", allow_pickle=False)
        sbp_parts.append(np.asarray(sbp, dtype=np.float32))
    sbp_full = np.concatenate(sbp_parts, axis=0)

    avg_sbp = sbp_full.mean(axis=1, dtype=np.float64).astype(np.float32)
    kernel = np.ones(int(max(1, smooth_bins)), dtype=np.float32) / float(max(1, smooth_bins))
    avg_sbp_smooth = np.convolve(avg_sbp, kernel, mode="same")
    time_ms = np.arange(len(avg_sbp), dtype=np.float32) * 20.0

    return SbpSessionTrace(
        dataset=dataset_name,
        session_id=session_id,
        time_ms=time_ms,
        avg_sbp=avg_sbp,
        avg_sbp_smooth=avg_sbp_smooth.astype(np.float32),
        total_bins=int(sbp_full.shape[0]),
        n_channels=int(sbp_full.shape[1]),
    )


def _plot_tx_count_hist(summaries: list[PlotDatasetSummary], out_path: Path, tx_max_count: int) -> None:
    labels = [str(k) for k in range(tx_max_count + 1)] + [f">={tx_max_count + 1}"]
    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(10, 6))
    for summary in summaries:
        probs = summary.tx_count_hist / summary.tx_count_hist.sum()
        ax.plot(x, probs, marker="o", linewidth=2, label=summary.dataset)

    ax.set_xticks(x, labels)
    ax.set_xlabel("TX count per channel-bin")
    ax.set_ylabel("Probability")
    ax.set_title("TX Count Histogram")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_active_fraction_hist(summaries: list[PlotDatasetSummary], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    for summary in summaries:
        mids = 0.5 * (summary.active_fraction_edges[:-1] + summary.active_fraction_edges[1:])
        probs = summary.active_fraction_hist / summary.active_fraction_hist.sum()
        ax.plot(mids, probs, linewidth=2, label=summary.dataset)

    ax.set_xlabel("Active-channel fraction per 20 ms bin")
    ax.set_ylabel("Probability")
    ax.set_title("Population Sparsity Per Bin")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_session_sparsity(summaries: list[PlotDatasetSummary], out_path: Path) -> None:
    n_rows = len(summaries)
    fig, axes = plt.subplots(n_rows, 1, figsize=(11, max(2.5 * n_rows, 6)), sharex=False)
    if n_rows == 1:
        axes = [axes]

    for ax, summary in zip(axes, summaries):
        values = np.array([value for _, value in summary.session_sparsity], dtype=np.float32)
        order = np.arange(len(values))
        ax.plot(order, values, marker="o", markersize=3, linewidth=1.5)
        ax.set_ylabel("P(tx > 0)")
        ax.set_title(f"{summary.dataset} ({len(summary.session_sparsity)} sessions)")
        ax.grid(alpha=0.25)
        if len(values) > 0:
            ax.axhline(float(values.mean()), color="tab:red", linestyle="--", alpha=0.8, linewidth=1)
        ax.set_xlim(0, max(len(values) - 1, 1))

    axes[-1].set_xlabel("Session index after sorting by session id")
    fig.suptitle("Per-Session TX Sparsity", y=0.995)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_sbp_trace(trace: SbpSessionTrace, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 4.5))
    time_s = trace.time_ms / 1000.0
    ax.plot(time_s, trace.avg_sbp, linewidth=0.8, alpha=0.35, label="raw avg SBP")
    ax.plot(time_s, trace.avg_sbp_smooth, linewidth=2.0, label="smoothed avg SBP")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Average SBP across channels")
    ax.set_title(
        f"Average SBP Per Bin: {trace.dataset} / {trace.session_id} "
        f"({trace.total_bins} bins, {trace.n_channels} channels)"
    )
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    suspicious_values = tuple(args.suspicious_value or DEFAULT_SUSPICIOUS_VALUES)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summaries: list[PlotDatasetSummary] = []
    skipped: list[tuple[str, float]] = []
    for dataset_name in args.datasets:
        dataset_root = args.cache_root / dataset_name
        if not dataset_root.exists():
            continue
        suspicious_fraction = _dataset_suspicious_fraction(
            dataset_root,
            chunk_rows=int(args.chunk_rows),
            suspicious_values=suspicious_values,
        )
        if suspicious_fraction > float(args.max_suspicious_fraction):
            skipped.append((dataset_name, suspicious_fraction))
            continue
        summaries.append(
            _collect_tx_plot_summary(
                dataset_root,
                chunk_rows=int(args.chunk_rows),
                tx_max_count=int(args.tx_max_count),
                active_fraction_bins=int(args.active_fraction_bins),
            )
        )

    if not summaries:
        raise RuntimeError("No datasets were eligible for plotting.")

    _plot_tx_count_hist(
        summaries,
        args.output_dir / "tx_count_histogram.png",
        tx_max_count=int(args.tx_max_count),
    )
    _plot_active_fraction_hist(
        summaries,
        args.output_dir / "tx_active_fraction_histogram.png",
    )
    _plot_session_sparsity(
        summaries,
        args.output_dir / "tx_session_sparsity.png",
    )

    sbp_trace = _pick_short_sbp_session(
        args.cache_root,
        list(args.sbp_datasets),
        sbp_max_total_bins=int(args.sbp_max_total_bins),
        smooth_bins=int(args.sbp_smooth_bins),
    )
    _plot_sbp_trace(sbp_trace, args.output_dir / "short_session_avg_sbp.png")

    manifest = {
        "datasets_plotted": [summary.dataset for summary in summaries],
        "datasets_skipped_for_suspicious_values": [
            {"dataset": name, "suspicious_channel_bin_fraction": fraction}
            for name, fraction in skipped
        ],
        "sbp_trace": {
            "dataset": sbp_trace.dataset,
            "session_id": sbp_trace.session_id,
            "total_bins": sbp_trace.total_bins,
            "n_channels": sbp_trace.n_channels,
        },
        "output_dir": str(args.output_dir),
    }
    (args.output_dir / "plot_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

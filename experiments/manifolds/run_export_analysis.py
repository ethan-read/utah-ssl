"""Run a first-pass, alignment-aware trajectory analysis on a saved export."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .analysis import (
    extract_aligned_events,
    fit_shared_pca,
    split_half_reliability,
    trajectory_separation,
)
from .io import load_representation_export


def _parse_ids(value: object) -> list[int]:
    if pd.isna(value) or not str(value).strip():
        return []
    return [int(token) for token in str(value).split()]


def _symbol_map(metadata: dict) -> dict[int, str]:
    vocab = metadata.get("vocab", {})
    if isinstance(vocab.get("id_to_symbol"), dict):
        return {int(key): str(value) for key, value in vocab["id_to_symbol"].items()}
    return {idx: str(symbol) for idx, symbol in enumerate(vocab.get("index_to_symbol", []))}


def run(args: argparse.Namespace) -> dict:
    if args.before < 0 or args.after < 0:
        raise ValueError("--before and --after must be non-negative")
    if args.min_trials < 2 or args.max_events_per_phoneme < 2:
        raise ValueError("--min-trials and --max-events-per-phoneme must be at least 2")
    if args.components < 2:
        raise ValueError("--components must be at least 2 for trajectory plotting")
    payload = load_representation_export(args.model_dir, representation=args.representation)
    examples = payload["examples"]
    references = {}
    for row in examples.itertuples():
        reference_ids = _parse_ids(row.reference_ids)
        if reference_ids:
            references[int(row.example_export_index)] = reference_ids
    blank_index = int(payload["metadata"]["vocab"]["blank_index"])
    events = extract_aligned_events(
        payload["values"],
        payload["logits"],
        payload["example_indices"],
        references,
        blank_index=blank_index,
        before=args.before,
        after=args.after,
    )
    if not events:
        raise RuntimeError("No complete alignment-centered windows were extracted")
    counts = pd.Series([event.label_id for event in events]).value_counts()
    retained_ids = set(counts[counts >= args.min_trials].index.astype(int))
    if len(retained_ids) < 2:
        raise RuntimeError("Fewer than two phonemes meet --min-trials")

    # Balance conditions and bound memory use. This matters particularly for
    # flattened input windows, where one seven-step event can contain tens of
    # thousands of values. Sampling is deterministic and the unsampled counts
    # remain in the output table.
    rng = np.random.default_rng(args.seed)
    retained = []
    for label_id in sorted(retained_ids):
        label_events = [event for event in events if event.label_id == label_id]
        if len(label_events) > args.max_events_per_phoneme:
            selected = np.sort(
                rng.choice(len(label_events), size=args.max_events_per_phoneme, replace=False)
            )
            label_events = [label_events[index] for index in selected]
        retained.extend(label_events)

    projected, _, pca = fit_shared_pca(
        [event.trajectory for event in retained], n_components=args.components
    )
    conditions = [event.label_id for event in retained]
    reliability = split_half_reliability(
        projected, conditions, repetitions=args.reliability_repetitions, seed=args.seed
    )
    separation = trajectory_separation(
        projected, conditions, permutations=args.permutations, seed=args.seed
    )
    symbols = _symbol_map(payload["metadata"])
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for label_id in sorted(retained_ids):
        rows.append(
            {
                "label_id": label_id,
                "symbol": symbols.get(label_id, str(label_id)),
                "event_count": int(counts[label_id]),
                "analyzed_event_count": sum(event.label_id == label_id for event in retained),
                "split_half_reliability": reliability.get(label_id, np.nan),
            }
        )
    pd.DataFrame(rows).to_csv(output_dir / "phoneme_repeatability.csv", index=False)

    top_ids = [int(idx) for idx in counts[counts.index.isin(retained_ids)].head(args.plot_phonemes).index]
    fig, axes = plt.subplots(1, len(top_ids), figsize=(4.0 * len(top_ids), 3.8), squeeze=False)
    for ax, label_id in zip(axes[0], top_ids):
        paths = [path for path, event in zip(projected, retained) if event.label_id == label_id]
        for path in paths[: args.max_plot_trials]:
            ax.plot(path[:, 0], path[:, 1], color="0.65", alpha=0.18, linewidth=0.7)
        mean_path = np.mean(paths, axis=0)
        ax.plot(mean_path[:, 0], mean_path[:, 1], "o-", color="tab:blue", linewidth=2.2, markersize=3)
        ax.scatter(mean_path[args.before, 0], mean_path[args.before, 1], color="tab:red", s=30, zorder=3)
        ax.set_title(f"{symbols.get(label_id, label_id)} (n={len(paths)})")
        ax.set_xlabel("shared PC1")
        ax.set_ylabel("shared PC2")
    fig.suptitle("CTC-aligned phoneme trajectories; red = aligned label center")
    fig.tight_layout()
    fig.savefig(output_dir / "phoneme_trajectories.png", dpi=180)
    plt.close(fig)

    summary = {
        "model_dir": str(payload["model_dir"]),
        "representation": args.representation,
        "patch_stride_ms": payload["metadata"].get("patch_stride_ms"),
        "event_count": len(events),
        "retained_event_count": len(retained),
        "eligible_event_count": int(counts[counts.index.isin(retained_ids)].sum()),
        "retained_phoneme_count": len(retained_ids),
        "pca_explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "separation": separation,
        "alignment_warning": "CTC timing is model-assisted, not independent ground truth.",
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--representation", choices=("hidden", "input_windows", "adapted_input_windows"), default="hidden")
    parser.add_argument("--before", type=int, default=3)
    parser.add_argument("--after", type=int, default=3)
    parser.add_argument("--min-trials", type=int, default=20)
    parser.add_argument("--max-events-per-phoneme", type=int, default=100)
    parser.add_argument("--components", type=int, default=6, help="Shared PCA dimensions; must be at least 2")
    parser.add_argument("--reliability-repetitions", type=int, default=200)
    parser.add_argument("--permutations", type=int, default=1000)
    parser.add_argument("--plot-phonemes", type=int, default=6)
    parser.add_argument("--max-plot-trials", type=int, default=100)
    parser.add_argument("--seed", type=int, default=7)
    return parser


if __name__ == "__main__":
    print(json.dumps(run(build_parser().parse_args()), indent=2))

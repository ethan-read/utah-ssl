"""Prepare a smoothed cache and matching stats for the broad BIT recipe.

The source cache is read-only. The exact dataset plan and TX signal contract
are recorded in the preparation summary and normalization artifact.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from typing import Any, Sequence

from utah_ssl.bit_cache_contract import (
    BIT_STAGE1_BOUNDARY_KEY_MODE,
    BIT_STAGE1_DATASET_SPLITS,
    BIT_STAGE1_SIGMA_BINS,
    BIT_STAGE1_TX_DIM,
)
from utah_ssl.experiment_contract import DatasetPlan, SignalSpec
from utah_ssl.scripts.build_smoothed_cache import build_smoothed_cache
from utah_ssl.stats import (
    recompute_session_feature_stats,
    resolve_precomputed_session_stats_path,
)


DEFAULT_SIGMA_BINS = BIT_STAGE1_SIGMA_BINS
DEFAULT_BOUNDARY_KEY_MODE = BIT_STAGE1_BOUNDARY_KEY_MODE


def _bit_signal_spec() -> SignalSpec:
    return SignalSpec.tx_only(
        tx_dim=int(BIT_STAGE1_TX_DIM),
        missing_channel_policy="zero_pad",
    )


@dataclass(frozen=True)
class DatasetInventoryEntry:
    dataset: str
    modalities: tuple[str, ...]
    shard_count: int
    num_sessions: int | None
    num_examples: int | None
    n_tx_features: int
    n_sbp_features: int
    metadata_path: str

    @property
    def has_tx(self) -> bool:
        return "tx" in set(self.modalities) or int(self.n_tx_features) > 0

    @property
    def has_sbp(self) -> bool:
        return "sbp" in set(self.modalities) or int(self.n_sbp_features) > 0


def _normalize_name_list(values: Sequence[str] | None) -> tuple[str, ...]:
    if values is None:
        return tuple()
    normalized: list[str] = []
    seen: set[str] = set()
    for item in values:
        value = str(item).strip()
        if not value or value in seen:
            continue
        seen.add(value)
        normalized.append(value)
    return tuple(normalized)


def _load_dataset_metadata(dataset_root: Path) -> dict[str, Any]:
    metadata_path = dataset_root / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing dataset metadata: {metadata_path}")
    return json.loads(metadata_path.read_text())


def _count_shards(dataset_root: Path) -> int:
    shard_root = dataset_root / "shards"
    if shard_root.is_dir():
        return sum(1 for path in shard_root.iterdir() if path.is_dir())
    return sum(
        1
        for path in dataset_root.iterdir()
        if path.is_dir() and path.name != "shards"
    )


def _inventory_cache_root(cache_root: Path) -> dict[str, DatasetInventoryEntry]:
    inventory: dict[str, DatasetInventoryEntry] = {}
    for dataset_root in sorted(path for path in cache_root.iterdir() if path.is_dir()):
        metadata_path = dataset_root / "metadata.json"
        manifest_path = dataset_root / "manifest.jsonl"
        if not metadata_path.exists() or not manifest_path.exists():
            continue
        metadata = _load_dataset_metadata(dataset_root)
        feature_layout = metadata.get("feature_layout", {})
        shard_entries = metadata.get("shards", [])
        shard_tx_widths = [
            int(item.get("n_tx_features", 0) or 0)
            for item in shard_entries
            if isinstance(item, dict)
        ]
        shard_sbp_widths = [
            int(item.get("n_sbp_features", 0) or 0)
            for item in shard_entries
            if isinstance(item, dict)
        ]
        n_tx_features = int(feature_layout.get("n_tx_features", 0) or 0)
        n_sbp_features = int(feature_layout.get("n_sbp_features", 0) or 0)
        if n_tx_features <= 0 and shard_tx_widths:
            n_tx_features = max(shard_tx_widths)
        if n_sbp_features <= 0 and shard_sbp_widths:
            n_sbp_features = max(shard_sbp_widths)
        modalities = tuple(str(item) for item in metadata.get("modalities", []))
        inventory[dataset_root.name] = DatasetInventoryEntry(
            dataset=dataset_root.name,
            modalities=modalities,
            shard_count=_count_shards(dataset_root),
            num_sessions=(
                int(metadata["num_sessions"])
                if metadata.get("num_sessions") is not None
                else None
            ),
            num_examples=(
                int(metadata["total_examples"])
                if metadata.get("total_examples") is not None
                else None
            ),
            n_tx_features=int(n_tx_features),
            n_sbp_features=int(n_sbp_features),
            metadata_path=str(metadata_path),
        )
    return inventory


def _select_datasets(
    *,
    inventory: dict[str, DatasetInventoryEntry],
    requested_datasets: Sequence[str],
) -> list[DatasetInventoryEntry]:
    names = _normalize_name_list(requested_datasets)
    if not names:
        raise ValueError("dataset_names must contain at least one dataset")
    missing = [name for name in names if name not in inventory]
    if missing:
        raise FileNotFoundError(
            f"Requested dataset(s) not found under the selected cache root: {missing}"
        )
    without_tx = [name for name in names if not inventory[name].has_tx]
    if without_tx:
        raise ValueError(
            "The BIT TX signal contract is incompatible with dataset(s) that lack "
            f"TX: {without_tx}"
        )
    return [inventory[name] for name in names]


def _recommended_output_root(src_root: Path, *, sigma_bins: float) -> Path:
    sigma_label = str(float(sigma_bins)).replace(".", "p")
    if "smoothed_sigma" in src_root.name:
        return src_root.with_name(f"{src_root.name}_bit_stage1")
    return src_root.with_name(f"{src_root.name}_smoothed_sigma{sigma_label}")


def _recommended_stats_output_path(
    *,
    cache_root: Path,
    dataset_plan: DatasetPlan,
    signal_spec: SignalSpec,
    boundary_key_mode: str,
) -> Path:
    return resolve_precomputed_session_stats_path(
        cache_root=cache_root,
        signal_spec=signal_spec,
        dataset_plan=dataset_plan,
        boundary_key_mode=boundary_key_mode,
    )


def _write_summary(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def prepare_bit_cache(
    *,
    cache_root: str | Path,
    output_root: str | Path | None = None,
    sigma_bins: float = DEFAULT_SIGMA_BINS,
    overwrite: bool = False,
    dry_run: bool = False,
    recompute_session_stats: bool = True,
    seed: int = 7,
) -> dict[str, Any]:
    cache_root = Path(cache_root)
    if not cache_root.is_dir():
        raise FileNotFoundError(f"Cache root does not exist: {cache_root}")

    dataset_plan = DatasetPlan.from_mapping(BIT_STAGE1_DATASET_SPLITS)
    requested_datasets = dataset_plan.dataset_names
    source_inventory = _inventory_cache_root(cache_root)
    selected_entries = _select_datasets(
        inventory=source_inventory,
        requested_datasets=requested_datasets,
    )
    selected_dataset_names = [entry.dataset for entry in selected_entries]
    signal_spec = _bit_signal_spec()
    output_root = (
        Path(output_root)
        if output_root is not None
        else _recommended_output_root(cache_root, sigma_bins=float(sigma_bins))
    )
    if output_root.resolve() == cache_root.resolve():
        raise ValueError(
            "BIT cache preparation requires a destination separate from the source cache."
        )
    stats_output_path = _recommended_stats_output_path(
        cache_root=output_root,
        dataset_plan=dataset_plan,
        signal_spec=signal_spec,
        boundary_key_mode=DEFAULT_BOUNDARY_KEY_MODE,
    )

    smoothing_summary = build_smoothed_cache(
        src_root=cache_root,
        dst_root=output_root,
        sigma_bins=float(sigma_bins),
        datasets=selected_dataset_names,
        overwrite=bool(overwrite),
        dry_run=bool(dry_run),
    )

    stats_summary: dict[str, Any] | None = None
    if recompute_session_stats and not dry_run:
        stats_summary = recompute_session_feature_stats(
            cache_root=output_root,
            output_path=stats_output_path,
            signal_spec=signal_spec,
            dataset_plan=dataset_plan,
            boundary_key_mode=DEFAULT_BOUNDARY_KEY_MODE,
            seed=int(seed),
            overwrite=bool(overwrite),
        )

    summary: dict[str, Any] = {
        "bit_prep": {
            "source_root": str(cache_root),
            "output_root": str(output_root),
            "sigma_bins": float(sigma_bins),
            "overwrite": bool(overwrite),
            "dry_run": bool(dry_run),
            "dataset_plan": dataset_plan.to_dict(),
            "signal_spec": signal_spec.to_dict(),
            "dataset_count": len(selected_dataset_names),
            "boundary_key_mode": DEFAULT_BOUNDARY_KEY_MODE,
            "stats_output_path": str(stats_output_path),
            "datasets": [asdict(entry) for entry in selected_entries],
        },
        "smoothing_summary": smoothing_summary,
        "stats_summary": (
            None
            if stats_summary is None
            else {
                key: str(value) if isinstance(value, Path) else value
                for key, value in stats_summary.items()
            }
        ),
    }

    if not dry_run:
        _write_summary(output_root / "bit_prep_summary.json", summary)

    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-root",
        type=Path,
        required=True,
        help="Canonical cache root to prepare from.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Destination smoothed cache root. Defaults to a sigma-tagged sibling.",
    )
    parser.add_argument(
        "--sigma-bins",
        type=float,
        default=DEFAULT_SIGMA_BINS,
        help="Gaussian smoothing width in bins.",
    )
    parser.add_argument(
        "--skip-session-stats",
        action="store_true",
        help="Build the smoothed cache but skip recomputing canonical tx-only session stats.",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    summary = prepare_bit_cache(
        cache_root=args.cache_root,
        output_root=args.output_root,
        sigma_bins=float(args.sigma_bins),
        overwrite=bool(args.overwrite),
        dry_run=bool(args.dry_run),
        recompute_session_stats=not bool(args.skip_session_stats),
        seed=int(args.seed),
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

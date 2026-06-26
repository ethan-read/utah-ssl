"""Prepare a BIT-style pre-smoothed cache root and matching session stats.

This orchestration script is intentionally conservative:

- it discovers datasets from an existing canonical cache root
- it harmonizes the canonical raw cache to the BIT stage-1 contract
- it builds a sigma-smoothed sibling cache for the selected datasets
- it recomputes canonical session-level tx-only z-scoring stats over the prepared cache
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

from bit_cache_contract import (
    BIT_STAGE1_BOUNDARY_KEY_MODE,
    BIT_STAGE1_DEFAULT_EXCLUDED_DATASETS,
    BIT_STAGE1_FEATURE_MODE,
    BIT_STAGE1_SBP_DIM,
    BIT_STAGE1_SIGMA_BINS,
    BIT_STAGE1_TX_DIM,
    canonical_stage1_stats_stem,
)
from build_smoothed_cache import build_smoothed_cache
from harmonize_bit_cache import harmonize_bit_cache
from masked_ssl.cache import (
    _canonical_session_stats_dir,
    _canonical_session_stats_stem_from_included,
)
from recompute_session_feature_stats import recompute_session_feature_stats


DEFAULT_SIGMA_BINS = BIT_STAGE1_SIGMA_BINS
DEFAULT_BOUNDARY_KEY_MODE = BIT_STAGE1_BOUNDARY_KEY_MODE
DEFAULT_FEATURE_MODE = BIT_STAGE1_FEATURE_MODE
DEFAULT_EXCLUDED_DATASETS = BIT_STAGE1_DEFAULT_EXCLUDED_DATASETS


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


def _default_fused_candidate(cache_root: Path) -> Path:
    if cache_root.name.endswith("_fused"):
        return cache_root
    if cache_root.name == "cache_v1":
        return cache_root.with_name("cache_v1_fused")
    return cache_root.with_name(f"{cache_root.name}_fused")


def _resolve_source_root(
    *,
    cache_root: Path,
    requested_datasets: Sequence[str],
    prefer_fused: bool,
) -> tuple[Path, str]:
    reason = "requested source root"
    if not prefer_fused:
        return cache_root, reason

    fused_candidate = _default_fused_candidate(cache_root)
    if not fused_candidate.exists():
        return cache_root, "fused sibling not found"

    raw_inventory = _inventory_cache_root(cache_root)
    fused_inventory = _inventory_cache_root(fused_candidate)
    if requested_datasets:
        missing = [name for name in requested_datasets if name not in fused_inventory]
        if missing:
            return cache_root, f"fused sibling missing datasets {missing}"
    else:
        missing = sorted(name for name in raw_inventory if name not in fused_inventory)
        if missing:
            return cache_root, f"fused sibling missing raw datasets {missing}"

    return fused_candidate, "using fused sibling to keep shard counts low"


def _select_datasets(
    *,
    inventory: dict[str, DatasetInventoryEntry],
    requested_datasets: Sequence[str],
    excluded_datasets: Sequence[str],
    require_tx: bool,
) -> list[DatasetInventoryEntry]:
    excluded = set(_normalize_name_list(excluded_datasets))
    if requested_datasets:
        missing = [name for name in requested_datasets if name not in inventory]
        if missing:
            raise FileNotFoundError(
                f"Requested dataset(s) not found under the selected cache root: {missing}"
            )
        names = [name for name in requested_datasets if name not in excluded]
    else:
        names = [name for name in sorted(inventory) if name not in excluded]

    selected: list[DatasetInventoryEntry] = []
    for name in names:
        entry = inventory[name]
        if require_tx and not entry.has_tx:
            continue
        selected.append(entry)

    if not selected:
        raise RuntimeError("No datasets matched the BIT-prep selection criteria.")
    return selected


def _recommended_output_root(src_root: Path, *, sigma_bins: float) -> Path:
    sigma_label = str(float(sigma_bins)).replace(".", "p")
    if "smoothed_sigma" in src_root.name:
        return src_root
    return src_root.with_name(f"{src_root.name}_smoothed_sigma{sigma_label}")


def _recommended_stats_output_path(
    *,
    cache_root: Path,
    dataset_names: Sequence[str],
    excluded_datasets: Sequence[str],
    feature_mode: str,
    boundary_key_mode: str,
) -> Path:
    stats_dir = _canonical_session_stats_dir(
        cache_root=cache_root,
        feature_mode=feature_mode,
        boundary_key_mode=boundary_key_mode,
    )
    stem = canonical_stage1_stats_stem(
        included_datasets=tuple(dataset_names),
        excluded_datasets=tuple(excluded_datasets),
        fallback_stem=_canonical_session_stats_stem_from_included(
            dataset_names=tuple(dataset_names)
        ),
    )
    return stats_dir / f"{stem}.pt"


def _write_summary(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def prepare_bit_cache(
    *,
    cache_root: str | Path,
    output_root: str | Path | None = None,
    sigma_bins: float = DEFAULT_SIGMA_BINS,
    dataset_names: Sequence[str] | None = None,
    excluded_datasets: Sequence[str] | None = None,
    prefer_fused: bool = True,
    overwrite: bool = False,
    dry_run: bool = False,
    recompute_session_stats: bool = True,
    harmonize_canonical_raw_cache: bool = True,
    segment_bins: int = 80,
    examples_per_shard: int = 8,
    seed: int = 7,
) -> dict[str, Any]:
    cache_root = Path(cache_root)
    if not cache_root.is_dir():
        raise FileNotFoundError(f"Cache root does not exist: {cache_root}")

    requested_datasets = _normalize_name_list(dataset_names)
    excluded = (
        _normalize_name_list(DEFAULT_EXCLUDED_DATASETS)
        if excluded_datasets is None
        else _normalize_name_list(excluded_datasets)
    )
    harmonization_summary: dict[str, Any] | None = None
    if harmonize_canonical_raw_cache:
        harmonization_summary = harmonize_bit_cache(
            cache_root=cache_root,
            dry_run=bool(dry_run),
        )
    selected_source_root, source_reason = _resolve_source_root(
        cache_root=cache_root,
        requested_datasets=requested_datasets,
        prefer_fused=bool(prefer_fused),
    )
    source_inventory = _inventory_cache_root(selected_source_root)
    selected_entries = _select_datasets(
        inventory=source_inventory,
        requested_datasets=requested_datasets,
        excluded_datasets=excluded,
        require_tx=True,
    )
    selected_dataset_names = [entry.dataset for entry in selected_entries]
    output_root = (
        Path(output_root)
        if output_root is not None
        else _recommended_output_root(selected_source_root, sigma_bins=float(sigma_bins))
    )
    stats_output_path = _recommended_stats_output_path(
        cache_root=output_root,
        dataset_names=selected_dataset_names,
        excluded_datasets=excluded,
        feature_mode=DEFAULT_FEATURE_MODE,
        boundary_key_mode=DEFAULT_BOUNDARY_KEY_MODE,
    )

    smoothing_summary = build_smoothed_cache(
        src_root=selected_source_root,
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
            feature_mode=DEFAULT_FEATURE_MODE,
            boundary_key_mode=DEFAULT_BOUNDARY_KEY_MODE,
            datasets=tuple(selected_dataset_names),
            tx_dim=int(BIT_STAGE1_TX_DIM),
            sbp_dim=int(BIT_STAGE1_SBP_DIM),
            segment_bins=int(segment_bins),
            seed=int(seed),
            examples_per_shard=int(examples_per_shard),
            excluded_datasets=tuple(excluded),
            overwrite=bool(overwrite),
        )

    summary: dict[str, Any] = {
        "bit_prep": {
            "requested_cache_root": str(cache_root),
            "selected_source_root": str(selected_source_root),
            "selected_source_reason": str(source_reason),
            "output_root": str(output_root),
            "sigma_bins": float(sigma_bins),
            "prefer_fused": bool(prefer_fused),
            "overwrite": bool(overwrite),
            "dry_run": bool(dry_run),
            "dataset_names": selected_dataset_names,
            "excluded_dataset_names": list(excluded),
            "dataset_count": len(selected_dataset_names),
            "recommended_ssl_feature_mode": DEFAULT_FEATURE_MODE,
            "recommended_boundary_key_mode": DEFAULT_BOUNDARY_KEY_MODE,
            "recommended_tx_dim": int(BIT_STAGE1_TX_DIM),
            "recommended_sbp_dim": int(BIT_STAGE1_SBP_DIM),
            "recommended_sbp_dim_note": "Only relevant for downstream tx_sbp speech fine-tuning; BIT stage-1 defaults to tx_only.",
            "recommended_stats_output_path": str(stats_output_path),
            "datasets": [asdict(entry) for entry in selected_entries],
        },
        "harmonization_summary": harmonization_summary,
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
        "--dataset",
        action="append",
        default=None,
        help="Dataset to include. Repeat to select a subset.",
    )
    parser.add_argument(
        "--exclude-dataset",
        action="append",
        default=None,
        help="Dataset to exclude. Repeat as needed.",
    )
    parser.add_argument(
        "--prefer-fused",
        dest="prefer_fused",
        action="store_true",
        help="Prefer a sibling fused cache root when present.",
    )
    parser.add_argument(
        "--no-prefer-fused",
        dest="prefer_fused",
        action="store_false",
        help="Use the provided cache root directly even if a fused sibling exists.",
    )
    parser.set_defaults(prefer_fused=False)
    parser.add_argument(
        "--skip-session-stats",
        action="store_true",
        help="Build the smoothed cache but skip recomputing canonical tx-only session stats.",
    )
    parser.add_argument(
        "--skip-canonical-harmonization",
        action="store_true",
        help="Do not rewrite the raw canonical cache before building the smoothed root.",
    )
    parser.add_argument("--segment-bins", type=int, default=80)
    parser.add_argument("--examples-per-shard", type=int, default=8)
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
        dataset_names=tuple(args.dataset) if args.dataset else None,
        excluded_datasets=tuple(args.exclude_dataset) if args.exclude_dataset else None,
        prefer_fused=bool(args.prefer_fused),
        overwrite=bool(args.overwrite),
        dry_run=bool(args.dry_run),
        recompute_session_stats=not bool(args.skip_session_stats),
        harmonize_canonical_raw_cache=not bool(args.skip_canonical_harmonization),
        segment_bins=int(args.segment_bins),
        examples_per_shard=int(args.examples_per_shard),
        seed=int(args.seed),
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

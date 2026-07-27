"""Recompute session-level featurewise z-scoring stats from a cache root."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import sys

EXPERIMENTS_DIR = Path(__file__).resolve().parents[2]
REPO_ROOT = Path(__file__).resolve().parents[5]
for _path in (REPO_ROOT, EXPERIMENTS_DIR):
    _path_str = str(_path)
    if _path_str not in sys.path:
        sys.path.insert(0, _path_str)
from typing import Any, Sequence

import torch

from ssl_core.bit_cache_contract import (
    BIT_STAGE1_DEFAULT_EXCLUDED_DATASETS,
    BIT_STAGE1_FEATURE_MODE,
    BIT_STAGE1_SBP_DIM,
    BIT_STAGE1_TX_DIM,
)
from masked_ssl.cache import (
    FEATURE_POLICY,
    CacheAccessConfig,
    _cache_variant_name,
    _compute_session_feature_stats,
    prepare_cache_context,
)


DEFAULT_TX_DIM = BIT_STAGE1_TX_DIM
DEFAULT_SBP_DIM = BIT_STAGE1_SBP_DIM
DEFAULT_EXCLUDED_DATASETS = BIT_STAGE1_DEFAULT_EXCLUDED_DATASETS


def _timestamp_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _normalize_dataset_args(datasets: Sequence[str] | None) -> tuple[str, ...] | None:
    if datasets is None:
        return None
    normalized = tuple(str(item) for item in datasets if str(item).strip())
    return normalized if normalized else None


def _normalize_excluded_dataset_args(datasets: Sequence[str] | None) -> tuple[str, ...]:
    if datasets is None:
        return tuple(DEFAULT_EXCLUDED_DATASETS)
    deduped: list[str] = []
    seen: set[str] = set()
    for item in datasets:
        value = str(item).strip()
        if not value or value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    if not deduped:
        return tuple(DEFAULT_EXCLUDED_DATASETS)
    return tuple(sorted(deduped))


def recompute_session_feature_stats(
    *,
    cache_root: str | Path,
    output_path: str | Path,
    feature_mode: str = BIT_STAGE1_FEATURE_MODE,
    boundary_key_mode: str = "session",
    datasets: Sequence[str] | None = None,
    tx_dim: int = DEFAULT_TX_DIM,
    sbp_dim: int = DEFAULT_SBP_DIM,
    segment_bins: int = 80,
    seed: int = 7,
    examples_per_shard: int = 8,
    excluded_datasets: Sequence[str] = DEFAULT_EXCLUDED_DATASETS,
    source_splits: Sequence[str] | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Recompute and save session-featurewise stats for a cache root."""

    cache_root = Path(cache_root)
    output_path = Path(output_path)
    if not cache_root.is_dir():
        raise FileNotFoundError(f"Cache root does not exist: {cache_root}")
    metadata_path = output_path.with_suffix(".json")
    if (output_path.exists() or metadata_path.exists()) and not overwrite:
        raise FileExistsError(
            f"Output already exists: {output_path} or {metadata_path}. "
            "Pass overwrite=True to replace existing artifacts."
        )
    if overwrite:
        output_path.unlink(missing_ok=True)
        metadata_path.unlink(missing_ok=True)

    excluded_datasets = _normalize_excluded_dataset_args(excluded_datasets)
    requested_datasets = _normalize_dataset_args(datasets)
    available_datasets = sorted(
        path.name for path in cache_root.iterdir() if path.is_dir() and (path / "metadata.json").exists()
    )
    if requested_datasets is not None:
        missing = [name for name in requested_datasets if name not in available_datasets]
        if missing:
            raise FileNotFoundError(
                f"Requested dataset(s) not found under {cache_root}: {missing}. "
                f"Available datasets: {available_datasets}"
            )
        excluded_datasets = tuple(name for name in available_datasets if name not in set(requested_datasets))

    config = CacheAccessConfig(
        mode="drive_direct",
        local_cache_base="/content/utah_ssl_cache",
        force_recopy_local_cache=False,
        excluded_datasets=tuple(str(name) for name in excluded_datasets),
        seed=int(seed),
        segment_bins=int(segment_bins),
        use_normalization=False,
        examples_per_shard=int(examples_per_shard),
        tx_dim=int(tx_dim),
        sbp_dim=int(sbp_dim),
        feature_mode=str(feature_mode),
        boundary_key_mode=str(boundary_key_mode),
        pretrain_source_splits=(
            tuple(str(item).strip().lower() for item in source_splits if str(item).strip())
            if source_splits is not None
            else None
        ),
    )

    context = prepare_cache_context(cache_candidates=[cache_root], config=config)

    session_feature_stats = _compute_session_feature_stats(
        shard_store=context.shard_store,
        rows_by_dataset=context.rows_by_dataset,
        config=config,
    )
    if not session_feature_stats:
        raise RuntimeError("No session feature stats were computed from the selected cache root.")

    metadata: dict[str, Any] = {
        "kind": "session_featurewise_zscore_stats",
        "created_utc": _timestamp_utc(),
        "source_cache_root": str(cache_root.resolve()),
        "source_cache_name": cache_root.name,
        "source_cache_variant": _cache_variant_name(cache_root),
        "source_cache_signature": context.source_cache_signature,
        "feature_mode": str(context.feature_mode),
        "boundary_key_mode": str(boundary_key_mode),
        "requested_datasets": list(requested_datasets) if requested_datasets is not None else None,
        "tx_dim": int(tx_dim),
        "sbp_dim": int(sbp_dim),
        "full_dim": int(config.full_dim),
        "feature_policy": FEATURE_POLICY,
        "segment_bins": int(segment_bins),
        "examples_per_shard": int(examples_per_shard),
        "excluded_datasets": list(excluded_datasets),
        "pretrain_source_splits": list(context.config.pretrain_source_splits or ()),
        "dataset_names": list(context.pretrain_datasets),
        "dataset_count": int(len(context.pretrain_datasets)),
        "session_count": int(len(session_feature_stats)),
        "session_stats_bin_stride": 2,
        "cache_copy_used": bool(context.cache_copy_used),
    }

    payload = {
        "session_feature_stats": session_feature_stats,
        "metadata": metadata,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")

    return {
        "output_path": output_path,
        "metadata_path": metadata_path,
        "metadata": metadata,
        "session_count": int(len(session_feature_stats)),
        "dataset_count": int(len(context.pretrain_datasets)),
        "cache_root": cache_root,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, required=True, help="Cache root containing dataset folders.")
    parser.add_argument("--output-path", type=Path, required=True, help="Destination .pt file for the stats.")
    parser.add_argument(
        "--feature-mode",
        choices=("tx_only", "tx_sbp"),
        default=BIT_STAGE1_FEATURE_MODE,
        help="Feature layout to match when computing stats.",
    )
    parser.add_argument(
        "--boundary-key-mode",
        choices=("session", "subject_if_available"),
        default="session",
        help="Keying mode used when normalizing sampled windows.",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        default=None,
        help="Dataset name to include. May be repeated. If omitted, all datasets except excluded ones are used.",
    )
    parser.add_argument("--tx-dim", type=int, default=DEFAULT_TX_DIM)
    parser.add_argument("--sbp-dim", type=int, default=DEFAULT_SBP_DIM)
    parser.add_argument("--segment-bins", type=int, default=80)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--examples-per-shard", type=int, default=8)
    parser.add_argument(
        "--excluded-dataset",
        action="append",
        default=None,
        help="Dataset names to exclude. May be repeated.",
    )
    parser.add_argument(
        "--source-split",
        action="append",
        default=None,
        help="Source split(s) to include in session stats and SSL sampling. May be repeated.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output file.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    excluded_datasets = _normalize_excluded_dataset_args(args.excluded_dataset)
    result = recompute_session_feature_stats(
        cache_root=args.cache_root,
        output_path=args.output_path,
        feature_mode=args.feature_mode,
        boundary_key_mode=args.boundary_key_mode,
        datasets=tuple(args.dataset) if args.dataset else None,
        tx_dim=int(args.tx_dim),
        sbp_dim=int(args.sbp_dim),
        segment_bins=int(args.segment_bins),
        seed=int(args.seed),
        examples_per_shard=int(args.examples_per_shard),
        excluded_datasets=excluded_datasets,
        source_splits=tuple(args.source_split) if args.source_split else None,
        overwrite=bool(args.overwrite),
    )
    print(json.dumps({k: str(v) if isinstance(v, Path) else v for k, v in result.items()}, indent=2))


if __name__ == "__main__":
    main()

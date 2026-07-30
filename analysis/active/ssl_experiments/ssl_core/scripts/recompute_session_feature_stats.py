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

from ssl_core.feature_contract import SUPPORTED_FEATURE_MODES
from ssl_core.experiment_contract import DatasetPlan, SignalSpec
from masked_ssl.cache import (
    CacheAccessConfig,
    SESSION_STATS_BIN_STRIDE,
    _cache_variant_name,
    _compute_session_feature_stats,
    prepare_cache_context,
)


def _parse_dataset_source_split_args(
    values: Sequence[str] | None,
) -> dict[str, tuple[str, ...]] | None:
    if not values:
        return None
    parsed: dict[str, set[str]] = {}
    for value in values:
        dataset, separator, source_split = str(value).partition("=")
        dataset = dataset.strip()
        source_split = source_split.strip().lower()
        if not separator or not dataset or not source_split:
            raise ValueError(
                "--dataset-source-split values must have the form DATASET=SOURCE_SPLIT"
            )
        parsed.setdefault(dataset, set()).add(source_split)
    return {dataset: tuple(sorted(source_splits)) for dataset, source_splits in sorted(parsed.items())}


def _parse_dataset_cache_root_args(
    values: Sequence[str] | None,
) -> dict[str, Path] | None:
    if not values:
        return None
    parsed: dict[str, Path] = {}
    for value in values:
        dataset, separator, cache_root = str(value).partition("=")
        dataset = dataset.strip()
        cache_root = cache_root.strip()
        if not separator or not dataset or not cache_root:
            raise ValueError(
                "--dataset-cache-root values must have the form DATASET=CACHE_ROOT"
            )
        parsed[dataset] = Path(cache_root)
    return dict(sorted(parsed.items()))


def _timestamp_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def recompute_session_feature_stats(
    *,
    cache_root: str | Path,
    output_path: str | Path,
    signal_spec: SignalSpec,
    dataset_plan: DatasetPlan,
    boundary_key_mode: str = "session",
    seed: int = 7,
    dataset_cache_roots: dict[str, str | Path] | None = None,
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

    signal_spec = SignalSpec.from_value(signal_spec)
    dataset_plan = DatasetPlan.from_value(dataset_plan)
    normalized_dataset_cache_roots = {
        str(dataset): Path(root)
        for dataset, root in sorted((dataset_cache_roots or {}).items())
    }
    config = CacheAccessConfig(
        dataset_plan=dataset_plan,
        signal_spec=signal_spec,
        mode="drive_direct",
        local_cache_base="/content/utah_ssl_cache",
        force_recopy_local_cache=False,
        seed=int(seed),
        segment_bins=1,
        use_normalization=False,
        boundary_key_mode=str(boundary_key_mode),
        dataset_cache_roots=normalized_dataset_cache_roots or None,
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
        "signal_spec": signal_spec.to_dict(),
        "dataset_plan": dataset_plan.to_dict(),
        "boundary_key_mode": str(boundary_key_mode),
        "dataset_names": list(context.pretrain_datasets),
        "dataset_count": int(len(context.pretrain_datasets)),
        "session_count": int(len(session_feature_stats)),
        "session_stats_bin_stride": SESSION_STATS_BIN_STRIDE,
        "cache_copy_used": bool(context.cache_copy_used),
    }
    metadata["source_cache_roots"] = {
        dataset: str(context.drive_dataset_cache_roots[dataset].resolve())
        for dataset in context.pretrain_datasets
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
        choices=SUPPORTED_FEATURE_MODES,
        required=True,
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
        required=True,
        help="Dataset name to include. May be repeated; the complete set must be explicit.",
    )
    parser.add_argument(
        "--tx-dim",
        type=int,
        default=None,
        help="Selected TX width. Defaults to the maximum declared width in the dataset plan.",
    )
    parser.add_argument(
        "--sbp-dim",
        type=int,
        default=None,
        help="Selected SBP width. Defaults to the maximum declared width in the dataset plan.",
    )
    parser.add_argument(
        "--column-start",
        type=int,
        default=0,
        help="First cache column selected for each requested modality.",
    )
    parser.add_argument(
        "--missing-channel-policy",
        choices=("error", "zero_pad"),
        default="error",
        help="How to handle datasets narrower than the requested signal width.",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--dataset-source-split",
        action="append",
        default=None,
        help="Dataset-specific source split in DATASET=SOURCE_SPLIT form. May be repeated.",
    )
    parser.add_argument(
        "--dataset-cache-root",
        action="append",
        default=None,
        help=(
            "Read one dataset from another cache root, in DATASET=CACHE_ROOT form. "
            "May be repeated."
        ),
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output file.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    splits_by_dataset = _parse_dataset_source_split_args(args.dataset_source_split) or {}
    dataset_cache_roots = _parse_dataset_cache_root_args(args.dataset_cache_root) or {}
    dataset_plan = DatasetPlan.from_mapping(
        {
            dataset: splits_by_dataset.get(dataset, ())
            for dataset in args.dataset
        }
    )
    declared_widths: list[tuple[int, int]] = []
    for dataset in dataset_plan.dataset_names:
        dataset_root = dataset_cache_roots.get(dataset, Path(args.cache_root))
        metadata = json.loads(
            (dataset_root / dataset / "metadata.json").read_text()
        )
        feature_layout = dict(metadata.get("feature_layout") or {})
        declared_widths.append(
            (
                int(metadata.get("n_tx_features", feature_layout.get("n_tx_features", 0)) or 0),
                int(metadata.get("n_sbp_features", feature_layout.get("n_sbp_features", 0)) or 0),
            )
        )
    tx_dim = (
        int(args.tx_dim)
        if args.tx_dim is not None
        else max(width[0] for width in declared_widths)
    )
    sbp_dim = (
        int(args.sbp_dim)
        if args.sbp_dim is not None
        else max(width[1] for width in declared_widths)
    )
    result = recompute_session_feature_stats(
        cache_root=args.cache_root,
        output_path=args.output_path,
        signal_spec=SignalSpec.from_mode(
            args.feature_mode,
            tx_dim=tx_dim,
            sbp_dim=sbp_dim,
            column_start=int(args.column_start),
            missing_channel_policy=str(args.missing_channel_policy),
        ),
        dataset_plan=dataset_plan,
        boundary_key_mode=args.boundary_key_mode,
        seed=int(args.seed),
        dataset_cache_roots=dataset_cache_roots or None,
        overwrite=bool(args.overwrite),
    )
    print(json.dumps({k: str(v) if isinstance(v, Path) else v for k, v in result.items()}, indent=2))


if __name__ == "__main__":
    main()

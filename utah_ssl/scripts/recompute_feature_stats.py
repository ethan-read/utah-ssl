"""Compute model-independent normalization statistics in one canonical format.

The command preserves the established Brain2Text24 payload keys:

- session scope: ``session_feature_stats``
- global scope: top-level ``mean`` and ``std``

New artifacts also contain a shared ``feature_stats`` map and schema metadata,
so generic tooling can consume both scopes through one interface.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utah_ssl.experiment_contract import DatasetPlan, SignalSpec
from utah_ssl.feature_contract import SUPPORTED_FEATURE_MODES
from utah_ssl.cache import resolve_precomputed_session_stats_path
from utah_ssl.scripts.recompute_session_feature_stats import (
    _parse_dataset_cache_root_args,
    _parse_dataset_source_split_args,
    recompute_session_feature_stats,
)
from utah_ssl.scripts.recompute_split_feature_stats import (
    recompute_split_feature_stats,
)


def _dataset_feature_widths(
    *,
    cache_root: Path,
    dataset_plan: DatasetPlan,
    dataset_cache_roots: dict[str, Path],
) -> tuple[int, int]:
    widths: list[tuple[int, int]] = []
    for dataset in dataset_plan.dataset_names:
        dataset_root = dataset_cache_roots.get(dataset, cache_root)
        metadata_path = dataset_root / dataset / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Dataset metadata does not exist: {metadata_path}")
        metadata = json.loads(metadata_path.read_text())
        feature_layout = dict(metadata.get("feature_layout") or {})
        widths.append(
            (
                int(metadata.get("n_tx_features", feature_layout.get("n_tx_features", 0)) or 0),
                int(metadata.get("n_sbp_features", feature_layout.get("n_sbp_features", 0)) or 0),
            )
        )
    return max(width[0] for width in widths), max(width[1] for width in widths)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", choices=("session", "global"), required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Optional destination. Without it, the canonical data-derived path is used.",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        required=True,
        help="Dataset to include. Repeat for pooled session statistics.",
    )
    parser.add_argument("--feature-mode", choices=SUPPORTED_FEATURE_MODES, required=True)
    parser.add_argument("--tx-dim", type=int, default=None)
    parser.add_argument("--sbp-dim", type=int, default=None)
    parser.add_argument("--column-start", type=int, default=0)
    parser.add_argument(
        "--missing-channel-policy",
        choices=("error", "zero_pad"),
        default="error",
    )
    parser.add_argument(
        "--boundary-key-mode",
        choices=("session", "subject_if_available"),
        default="session",
    )
    parser.add_argument(
        "--dataset-source-split",
        action="append",
        default=None,
        help="Session-scope selection in DATASET=SOURCE_SPLIT form.",
    )
    parser.add_argument(
        "--dataset-cache-root",
        action="append",
        default=None,
        help="Session-scope cache override in DATASET=CACHE_ROOT form.",
    )
    parser.add_argument(
        "--split-policy",
        choices=("competition_train_test", "source_train_val"),
        default="competition_train_test",
        help="Global-scope train/evaluation split policy.",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_cache_roots = _parse_dataset_cache_root_args(args.dataset_cache_root) or {}
    splits_by_dataset = _parse_dataset_source_split_args(args.dataset_source_split) or {}
    requested_datasets = tuple(str(dataset) for dataset in args.dataset)
    if len(set(requested_datasets)) != len(requested_datasets):
        raise ValueError("Each --dataset may be specified only once.")
    unknown_split_datasets = sorted(set(splits_by_dataset) - set(requested_datasets))
    if unknown_split_datasets:
        raise ValueError(
            "--dataset-source-split references unselected datasets: "
            + ", ".join(unknown_split_datasets)
        )
    unknown_cache_datasets = sorted(set(dataset_cache_roots) - set(requested_datasets))
    if unknown_cache_datasets:
        raise ValueError(
            "--dataset-cache-root references unselected datasets: "
            + ", ".join(unknown_cache_datasets)
        )
    if args.scope == "session":
        missing_split_datasets = sorted(set(requested_datasets) - set(splits_by_dataset))
        if missing_split_datasets:
            raise ValueError(
                "Session scope requires an explicit --dataset-source-split for every "
                "dataset; missing: " + ", ".join(missing_split_datasets)
            )
    dataset_plan = DatasetPlan.from_mapping(
        {dataset: splits_by_dataset.get(dataset, ()) for dataset in requested_datasets}
    )
    declared_tx_dim, declared_sbp_dim = _dataset_feature_widths(
        cache_root=Path(args.cache_root),
        dataset_plan=dataset_plan,
        dataset_cache_roots=dataset_cache_roots,
    )
    signal_spec = SignalSpec.from_mode(
        args.feature_mode,
        tx_dim=declared_tx_dim if args.tx_dim is None else int(args.tx_dim),
        sbp_dim=declared_sbp_dim if args.sbp_dim is None else int(args.sbp_dim),
        column_start=int(args.column_start),
        missing_channel_policy=str(args.missing_channel_policy),
    )

    if args.scope == "session":
        output_path = args.output_path or resolve_precomputed_session_stats_path(
            cache_root=args.cache_root,
            signal_spec=signal_spec,
            dataset_plan=dataset_plan,
            boundary_key_mode=str(args.boundary_key_mode),
            dataset_cache_roots=dataset_cache_roots or None,
        )
        result = recompute_session_feature_stats(
            cache_root=args.cache_root,
            output_path=output_path,
            signal_spec=signal_spec,
            dataset_plan=dataset_plan,
            boundary_key_mode=str(args.boundary_key_mode),
            seed=int(args.seed),
            dataset_cache_roots=dataset_cache_roots or None,
            overwrite=bool(args.overwrite),
        )
    else:
        if len(dataset_plan.dataset_names) != 1:
            raise ValueError("Global scope requires exactly one --dataset.")
        if splits_by_dataset:
            raise ValueError(
                "Global scope derives its training rows from --split-policy; "
                "do not pass --dataset-source-split."
            )
        if dataset_cache_roots:
            raise ValueError(
                "Global scope reads one dataset directly; pass its root as --cache-root."
            )
        result = recompute_split_feature_stats(
            cache_root=args.cache_root,
            output_path=args.output_path,
            dataset=dataset_plan.dataset_names[0],
            signal_spec=signal_spec,
            boundary_key_mode=str(args.boundary_key_mode),
            split_policy=str(args.split_policy),
            overwrite=bool(args.overwrite),
        )

    print(
        json.dumps(
            {
                key: str(value) if isinstance(value, Path) else value
                for key, value in result.items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

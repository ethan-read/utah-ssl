from __future__ import annotations

from pathlib import Path

import torch

from utah_ssl.cache import (
    _cache_variant_name,
    _compute_dataset_cache_source_signature,
)
from utah_ssl.experiment_contract import (
    DatasetPlan,
    SignalSpec,
)
from utah_ssl.normalization_stats import (
    write_feature_stats_artifact,
)


def write_valid_split_stats_artifact(
    *,
    cache_root: Path,
    stats_path: Path,
    dataset: str,
    signal_spec: SignalSpec,
    boundary_key_mode: str,
    split_policy: str,
    train_split_name: str,
    val_split_name: str,
    mean: torch.Tensor | None = None,
    std: torch.Tensor | None = None,
) -> None:
    signal_spec = SignalSpec.from_value(signal_spec)
    resolved_dim = signal_spec.full_dim
    metadata = {
        "kind": "split_feature_stats",
        "source_cache_root": str(cache_root.resolve()),
        "source_cache_name": cache_root.name,
        "source_cache_variant": _cache_variant_name(cache_root),
        "source_cache_signature": _compute_dataset_cache_source_signature(
            {str(dataset): cache_root}
        ),
        "dataset": str(dataset),
        "feature_mode": signal_spec.mode,
        "signal_spec": signal_spec.to_dict(),
        "boundary_key_mode": str(boundary_key_mode),
        "split_policy": str(split_policy),
        "train_split_name": str(train_split_name),
        "val_split_name": str(val_split_name),
        "feature_dim": resolved_dim,
    }
    write_feature_stats_artifact(
        output_path=stats_path,
        scope="global",
        entries={
            "global": (
                torch.zeros(resolved_dim) if mean is None else torch.as_tensor(mean).clone(),
                torch.ones(resolved_dim) if std is None else torch.as_tensor(std).clone(),
            )
        },
        metadata=metadata,
    )


def write_valid_session_stats_artifact(
    *,
    cache_root: Path,
    stats_path: Path,
    stats_entries: dict[str, tuple[torch.Tensor, torch.Tensor]],
    signal_spec: SignalSpec,
    dataset_plan: DatasetPlan,
    boundary_key_mode: str,
) -> None:
    signal_spec = SignalSpec.from_value(signal_spec)
    dataset_plan = DatasetPlan.from_value(dataset_plan)
    source_cache_roots = {
        dataset: cache_root for dataset in dataset_plan.dataset_names
    }
    metadata = {
        "kind": "session_featurewise_zscore_stats",
        "source_cache_root": str(cache_root.resolve()),
        "source_cache_name": cache_root.name,
        "source_cache_variant": _cache_variant_name(cache_root),
        "source_cache_signature": _compute_dataset_cache_source_signature(
            source_cache_roots
        ),
        "source_cache_roots": {
            dataset: str(cache_root.resolve())
            for dataset in dataset_plan.dataset_names
        },
        "signal_spec": signal_spec.to_dict(),
        "dataset_plan": dataset_plan.to_dict(),
        "boundary_key_mode": str(boundary_key_mode),
        "session_stats_bin_stride": 2,
    }
    write_feature_stats_artifact(
        output_path=stats_path,
        scope="session",
        entries=stats_entries,
        metadata=metadata,
    )

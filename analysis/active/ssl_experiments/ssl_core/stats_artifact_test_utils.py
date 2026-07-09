from __future__ import annotations

import json
from pathlib import Path

import torch

from analysis.active.ssl_experiments.masked_ssl.cache import (
    FEATURE_POLICY as SESSION_FEATURE_POLICY,
    _cache_variant_name,
    _compute_cache_source_signature,
)
from analysis.active.ssl_experiments.ssl_core.scripts.recompute_split_feature_stats import (
    FEATURE_POLICY as SPLIT_FEATURE_POLICY,
)


def write_valid_split_stats_artifact(
    *,
    cache_root: Path,
    stats_path: Path,
    dataset: str,
    feature_mode: str,
    boundary_key_mode: str,
    train_split_name: str,
    val_split_name: str,
    dim: int,
    mean: torch.Tensor | None = None,
    std: torch.Tensor | None = None,
) -> None:
    resolved_dim = int(dim)
    metadata = {
        "kind": "split_feature_stats",
        "source_cache_root": str(cache_root.resolve()),
        "source_cache_name": cache_root.name,
        "source_cache_variant": _cache_variant_name(cache_root),
        "source_cache_signature": _compute_cache_source_signature(cache_root),
        "dataset": str(dataset),
        "feature_mode": str(feature_mode),
        "boundary_key_mode": str(boundary_key_mode),
        "train_split_name": str(train_split_name),
        "val_split_name": str(val_split_name),
        "feature_dim": resolved_dim,
        "feature_policy": SPLIT_FEATURE_POLICY,
    }
    payload = {
        "mean": torch.zeros(resolved_dim) if mean is None else torch.as_tensor(mean).clone(),
        "std": torch.ones(resolved_dim) if std is None else torch.as_tensor(std).clone(),
        "metadata": metadata,
    }
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, stats_path)
    stats_path.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")


def write_valid_session_stats_artifact(
    *,
    cache_root: Path,
    stats_path: Path,
    stats_entries: dict[str, tuple[torch.Tensor, torch.Tensor]],
    feature_mode: str,
    boundary_key_mode: str,
    tx_dim: int,
    sbp_dim: int,
    excluded_datasets: tuple[str, ...],
) -> None:
    resolved_tx_dim = int(tx_dim)
    resolved_sbp_dim = int(sbp_dim)
    metadata = {
        "kind": "session_featurewise_zscore_stats",
        "source_cache_root": str(cache_root.resolve()),
        "source_cache_name": cache_root.name,
        "source_cache_variant": _cache_variant_name(cache_root),
        "source_cache_signature": _compute_cache_source_signature(cache_root),
        "feature_mode": str(feature_mode),
        "boundary_key_mode": str(boundary_key_mode),
        "tx_dim": resolved_tx_dim,
        "sbp_dim": resolved_sbp_dim,
        "full_dim": int(resolved_tx_dim if feature_mode == "tx_only" else resolved_tx_dim + resolved_sbp_dim),
        "feature_policy": SESSION_FEATURE_POLICY,
        "excluded_datasets": list(
            sorted({str(item).strip() for item in excluded_datasets if str(item).strip()})
        ),
    }
    payload = {
        "session_feature_stats": stats_entries,
        "metadata": metadata,
    }
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, stats_path)
    stats_path.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")

"""Normalization-stat helpers for SSL and downstream CTC experiments."""

from __future__ import annotations

from .cache import (
    build_recompute_session_feature_stats_command,
    load_precomputed_session_feature_stats_into_cache_context,
    resolve_precomputed_session_stats_path,
)

try:
    from masked_ssl.probe import apply_feature_stats, compute_feature_stats
except ModuleNotFoundError:  # pragma: no cover - repo-root unittest fallback
    from analysis.active.ssl_experiments.masked_ssl.probe import (
        apply_feature_stats,
        compute_feature_stats,
    )

try:
    from recompute_split_feature_stats import (
        load_precomputed_split_feature_stats,
        resolve_precomputed_split_stats_path,
    )
except ModuleNotFoundError:  # pragma: no cover - repo-root unittest fallback
    from analysis.active.ssl_experiments.recompute_split_feature_stats import (
        load_precomputed_split_feature_stats,
        resolve_precomputed_split_stats_path,
    )


__all__ = [
    "apply_feature_stats",
    "build_recompute_session_feature_stats_command",
    "compute_feature_stats",
    "load_precomputed_session_feature_stats_into_cache_context",
    "load_precomputed_split_feature_stats",
    "resolve_precomputed_session_stats_path",
    "resolve_precomputed_split_stats_path",
]

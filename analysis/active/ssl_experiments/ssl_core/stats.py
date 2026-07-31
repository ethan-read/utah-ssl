"""Public normalization-stat API for analysis and model training."""

from __future__ import annotations

from .cache import (
    build_recompute_session_feature_stats_command,
    load_precomputed_session_feature_stats_into_cache_context,
    resolve_precomputed_session_stats_path,
)
from .normalization_stats import (
    FEATURE_STATS_SCHEMA,
    SUPPORTED_NORMALIZATION_SCOPES,
    build_feature_stats_payload,
    extract_feature_stats_entries,
    write_feature_stats_artifact,
)

try:
    from masked_ssl.probe import apply_feature_stats, compute_feature_stats
except ModuleNotFoundError:  # pragma: no cover - repo-root unittest fallback
    from analysis.active.ssl_experiments.masked_ssl.probe import (
        apply_feature_stats,
        compute_feature_stats,
    )

try:
    from ssl_core.scripts.recompute_split_feature_stats import (
        build_recompute_split_feature_stats_command,
        load_precomputed_split_feature_stats,
        recompute_split_feature_stats,
        resolve_precomputed_split_stats_path,
    )
except ModuleNotFoundError:  # pragma: no cover - repo-root unittest fallback
    from analysis.active.ssl_experiments.ssl_core.scripts.recompute_split_feature_stats import (
        build_recompute_split_feature_stats_command,
        load_precomputed_split_feature_stats,
        recompute_split_feature_stats,
        resolve_precomputed_split_stats_path,
    )

try:
    from ssl_core.scripts.recompute_session_feature_stats import (
        recompute_session_feature_stats,
    )
except ModuleNotFoundError:  # pragma: no cover - repo-root unittest fallback
    from analysis.active.ssl_experiments.ssl_core.scripts.recompute_session_feature_stats import (
        recompute_session_feature_stats,
    )


__all__ = [
    "apply_feature_stats",
    "FEATURE_STATS_SCHEMA",
    "SUPPORTED_NORMALIZATION_SCOPES",
    "build_recompute_split_feature_stats_command",
    "build_recompute_session_feature_stats_command",
    "build_feature_stats_payload",
    "compute_feature_stats",
    "extract_feature_stats_entries",
    "load_precomputed_session_feature_stats_into_cache_context",
    "load_precomputed_split_feature_stats",
    "recompute_session_feature_stats",
    "recompute_split_feature_stats",
    "resolve_precomputed_session_stats_path",
    "resolve_precomputed_split_stats_path",
    "write_feature_stats_artifact",
]

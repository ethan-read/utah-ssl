"""Compatibility shim for contrastive SSL cache access.

This module now delegates to ``utah_ssl.cache`` so contrastive S5 notebooks use
the same area-6v feature policy, cache validation, and segment sampler behavior
as POSSM/Willett workflows.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

try:
    from utah_ssl.cache import (
        CacheAccessConfig,
        CacheContext,
        prepare_cache_context as _prepare_cache_context,
    )
    from utah_ssl.sampling import SegmentBatchSampler, build_segment_sampler
    from utah_ssl.stats import (
        load_precomputed_session_feature_stats_into_cache_context,
        resolve_precomputed_session_stats_path,
    )
except ModuleNotFoundError:  # pragma: no cover - repo-root unittest fallback
    from utah_ssl.cache import (
        CacheAccessConfig,
        CacheContext,
        prepare_cache_context as _prepare_cache_context,
    )
    from utah_ssl.sampling import SegmentBatchSampler, build_segment_sampler
    from utah_ssl.stats import (
        load_precomputed_session_feature_stats_into_cache_context,
        resolve_precomputed_session_stats_path,
    )


def prepare_cache_context(
    *,
    cache_candidates: Sequence[Path],
    config: CacheAccessConfig,
) -> CacheContext:
    """Build a cache context using the shared utah_ssl cache backend."""
    return _prepare_cache_context(cache_candidates=cache_candidates, config=config)


__all__ = [
    "CacheAccessConfig",
    "CacheContext",
    "SegmentBatchSampler",
    "build_segment_sampler",
    "load_precomputed_session_feature_stats_into_cache_context",
    "resolve_precomputed_session_stats_path",
    "prepare_cache_context",
]

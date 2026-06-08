"""Shared cache access compatibility layer.

The implementation still lives in ``masked_ssl.cache`` during this first
refactor pass. New code should import from ``ssl_core.cache`` so the backing
module can be split without touching every experiment again.
"""

from __future__ import annotations

try:
    from masked_ssl.cache import (
        AREA6V_FEATURE_DIM,
        FEATURE_POLICY,
        SESSION_STATS_BIN_STRIDE,
        CacheAccessConfig,
        CacheContext,
        ExampleRow,
        SamplingPlan,
        SegmentBatchSampler,
        build_recompute_session_feature_stats_command,
        build_segment_sampler,
        ensure_runtime_smoothing_disabled,
        load_cache_smoothing_provenance,
        load_dataset_metadata,
        load_precomputed_session_feature_stats_into_cache_context,
        prepare_cache_context,
        resolve_boundary_key,
        resolve_precomputed_session_stats_path,
        runtime_smoothing_requested,
        sample_base_segment,
        stack_segment_batch,
    )
except ModuleNotFoundError:  # pragma: no cover - repo-root unittest fallback
    from analysis.active.ssl_experiments.masked_ssl.cache import (
        AREA6V_FEATURE_DIM,
        FEATURE_POLICY,
        SESSION_STATS_BIN_STRIDE,
        CacheAccessConfig,
        CacheContext,
        ExampleRow,
        SamplingPlan,
        SegmentBatchSampler,
        build_recompute_session_feature_stats_command,
        build_segment_sampler,
        ensure_runtime_smoothing_disabled,
        load_cache_smoothing_provenance,
        load_dataset_metadata,
        load_precomputed_session_feature_stats_into_cache_context,
        prepare_cache_context,
        resolve_boundary_key,
        resolve_precomputed_session_stats_path,
        runtime_smoothing_requested,
        sample_base_segment,
        stack_segment_batch,
    )


__all__ = [
    "AREA6V_FEATURE_DIM",
    "FEATURE_POLICY",
    "SESSION_STATS_BIN_STRIDE",
    "CacheAccessConfig",
    "CacheContext",
    "ExampleRow",
    "SamplingPlan",
    "SegmentBatchSampler",
    "build_recompute_session_feature_stats_command",
    "build_segment_sampler",
    "ensure_runtime_smoothing_disabled",
    "load_cache_smoothing_provenance",
    "load_dataset_metadata",
    "load_precomputed_session_feature_stats_into_cache_context",
    "prepare_cache_context",
    "resolve_boundary_key",
    "resolve_precomputed_session_stats_path",
    "runtime_smoothing_requested",
    "sample_base_segment",
    "stack_segment_batch",
]

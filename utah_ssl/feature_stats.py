"""Compute and apply normalization statistics for canonical sequence data."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .canonical_data import CanonicalProbeManifestRow, CanonicalShardAccessor
from .experiment_contract import SignalSpec


FeatureStats = (
    dict[str, tuple[np.ndarray, np.ndarray]]
    | tuple[np.ndarray, np.ndarray]
)


def compute_feature_stats(
    rows: tuple[CanonicalProbeManifestRow, ...] | list[CanonicalProbeManifestRow],
    *,
    cache_root: str | Path,
    mode: str,
    signal_spec: SignalSpec | dict[str, Any],
) -> FeatureStats:
    signal = SignalSpec.from_value(signal_spec)
    if mode == "per_session":
        grouped: dict[str, list[CanonicalProbeManifestRow]] = {}
        for row in rows:
            grouped.setdefault(row.session_id, []).append(row)
        return {
            session_id: compute_feature_stats(
                session_rows,
                cache_root=cache_root,
                mode="global",
                signal_spec=signal,
            )  # type: ignore[dict-item]
            for session_id, session_rows in grouped.items()
        }
    if mode != "global":
        raise ValueError("mode must be either 'global' or 'per_session'")

    accessor = CanonicalShardAccessor(cache_root)
    try:
        total_count = 0
        sum_x: np.ndarray | None = None
        sum_x2: np.ndarray | None = None
        for row in rows:
            x64 = accessor.load_features(row, signal_spec=signal).astype(np.float64, copy=False)
            if sum_x is None:
                sum_x = x64.sum(axis=0)
                sum_x2 = np.square(x64).sum(axis=0)
            else:
                sum_x += x64.sum(axis=0)
                assert sum_x2 is not None
                sum_x2 += np.square(x64).sum(axis=0)
            total_count += x64.shape[0]
        if sum_x is None or sum_x2 is None or total_count == 0:
            raise ValueError("Cannot compute global feature stats on an empty record set.")
        mean = sum_x / total_count
        variance = np.maximum(sum_x2 / total_count - np.square(mean), 1e-6)
        return mean.astype(np.float32), np.sqrt(variance).astype(np.float32)
    finally:
        accessor.close()


def apply_feature_stats(
    x: np.ndarray,
    *,
    row: CanonicalProbeManifestRow,
    stats: FeatureStats,
) -> np.ndarray:
    if isinstance(stats, dict):
        candidate_keys: list[str] = []
        if row.block_num is not None:
            candidate_keys.append(f"{row.session_id}::block:{int(row.block_num)}")
        if row.normalization_group is not None:
            candidate_keys.append(str(row.normalization_group))
        candidate_keys.append(row.session_id)
        pair = next((stats.get(key) for key in candidate_keys if stats.get(key) is not None), None)
        if pair is None:
            raise KeyError(
                f"No feature stats found for row {row.example_id} using keys {candidate_keys!r}."
            )
        mean, std = pair
    else:
        mean, std = stats
    return ((x - mean) / std).astype(np.float32, copy=False)


__all__ = ["FeatureStats", "apply_feature_stats", "compute_feature_stats"]

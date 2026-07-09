"""Shared canonical cache contract for BIT-style stage-1 pretraining."""

from __future__ import annotations

from typing import Sequence


BIT_CANONICAL_FEATURE_POLICY = "bit_stage1_tx_only_v1"
BIT_STAGE1_FEATURE_MODE = "tx_only"
BIT_STAGE1_BOUNDARY_KEY_MODE = "session"
BIT_STAGE1_TX_DIM = 256
BIT_STAGE1_SBP_DIM = 128
BIT_STAGE1_SIGMA_BINS = 2.0

BIT_STAGE1_DEFAULT_INCLUDED_DATASETS = (
    "000950",
    "brain2text24",
    "motor_data",
    "plug_n_play",
    "unsupervised_cursor_recalibration_offline",
    "unsupervised_cursor_recalibration_online",
    "willett_handwriting",
)
BIT_STAGE1_DEFAULT_EXCLUDED_DATASETS = ("brain2text25",)
BIT_STAGE1_DEFAULT_STATS_STEM = (
    "ssl_pretrain_including_brain2text24_excluding_brain2text25_v1"
)


def _normalize_dataset_names(values: Sequence[str]) -> tuple[str, ...]:
    normalized: list[str] = []
    seen: set[str] = set()
    for item in values:
        value = str(item).strip()
        if not value or value in seen:
            continue
        seen.add(value)
        normalized.append(value)
    return tuple(sorted(normalized))


def matches_default_bit_stage1_selection(
    *,
    included_datasets: Sequence[str],
    excluded_datasets: Sequence[str],
) -> bool:
    included = _normalize_dataset_names(included_datasets)
    excluded = _normalize_dataset_names(excluded_datasets)
    return (
        included == _normalize_dataset_names(BIT_STAGE1_DEFAULT_INCLUDED_DATASETS)
        and excluded == _normalize_dataset_names(BIT_STAGE1_DEFAULT_EXCLUDED_DATASETS)
    )


def canonical_stage1_stats_stem(
    *,
    included_datasets: Sequence[str],
    excluded_datasets: Sequence[str],
    fallback_stem: str,
) -> str:
    if matches_default_bit_stage1_selection(
        included_datasets=included_datasets,
        excluded_datasets=excluded_datasets,
    ):
        return BIT_STAGE1_DEFAULT_STATS_STEM
    return str(fallback_stem)


__all__ = [
    "BIT_CANONICAL_FEATURE_POLICY",
    "BIT_STAGE1_BOUNDARY_KEY_MODE",
    "BIT_STAGE1_DEFAULT_EXCLUDED_DATASETS",
    "BIT_STAGE1_DEFAULT_INCLUDED_DATASETS",
    "BIT_STAGE1_DEFAULT_STATS_STEM",
    "BIT_STAGE1_FEATURE_MODE",
    "BIT_STAGE1_SBP_DIM",
    "BIT_STAGE1_SIGMA_BINS",
    "BIT_STAGE1_TX_DIM",
    "canonical_stage1_stats_stem",
    "matches_default_bit_stage1_selection",
]

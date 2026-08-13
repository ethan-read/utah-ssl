"""Segment extraction and balanced sampling over prepared Utah-array caches.

This module consumes a prepared ``CacheContext`` but does not participate in
cache discovery, copying, validation, or shard-store construction.
"""

from __future__ import annotations

import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from .cache import CacheContext, ExampleRow
from .experiment_contract import SignalSpec
from .session_keys import resolve_boundary_key


@dataclass(frozen=True)
class SamplingPlan:
    split_name: str
    segment_bins: int
    dataset_weight_alpha: float
    dataset_names: tuple[str, ...]
    dataset_probs: np.ndarray
    shard_rows_by_dataset: dict[str, dict[str, list[ExampleRow]]]
    shard_keys_by_dataset: dict[str, list[str]]
    shard_probs_by_dataset: dict[str, np.ndarray]
    row_probs_within_shard_by_dataset: dict[str, dict[str, np.ndarray]]


def normalize_segment(
    x_seq: torch.Tensor,
    feature_mask: torch.Tensor,
    *,
    session_feature_stats: dict[str, tuple[torch.Tensor, torch.Tensor]] | None = None,
    session_key: str | None = None,
    clip_value: float = 20.0,
    use_normalization: bool = True,
) -> torch.Tensor:
    """Apply the context's session-featurewise normalization to one segment."""

    if not bool(use_normalization):
        return x_seq
    if session_feature_stats is None or session_key is None:
        raise ValueError("Session feature stats are required when normalization is enabled.")
    if session_key not in session_feature_stats:
        raise KeyError(f"Missing session feature stats for {session_key}")

    x_norm = x_seq.clone()
    present_idx = torch.nonzero(feature_mask.bool(), as_tuple=False).squeeze(1)
    if present_idx.numel() == 0:
        return x_norm

    mean, std = session_feature_stats[session_key]
    mean = mean.to(device=x_norm.device, dtype=x_norm.dtype)
    std = std.to(device=x_norm.device, dtype=x_norm.dtype).clamp_min(1e-6)
    centered = x_norm[:, present_idx] - mean[present_idx]
    x_norm[:, present_idx] = (centered / std[present_idx]).clamp(
        min=-clip_value,
        max=clip_value,
    )
    return x_norm


def sample_base_segment(
    cache_context: CacheContext,
    example: ExampleRow,
    segment_bins: int,
    py_rng: random.Random,
) -> dict[str, Any]:
    """Sample and normalize one fixed-length segment from a manifest row."""

    signal_spec = cache_context.config.signal_spec
    assert isinstance(signal_spec, SignalSpec)
    feature_contract = signal_spec.contract
    boundary_key = resolve_boundary_key(
        dataset=example.dataset,
        session_id=example.session_id,
        subject_id=example.subject_id,
        boundary_key_mode=cache_context.boundary_key_mode,
    )
    shard = cache_context.shard_store.get(example.shard_relpath)
    time_offsets = shard["time_offsets"]
    assert isinstance(time_offsets, np.ndarray)
    start = int(time_offsets[example.example_index])
    stop = int(time_offsets[example.example_index + 1])
    length = stop - start
    total_needed = int(segment_bins)
    max_start = length - total_needed
    if max_start < 0:
        raise ValueError(
            f"Example {example.dataset}:{example.session_id} length={length} "
            f"cannot support segment_bins={segment_bins}"
        )

    offset = py_rng.randrange(max_start + 1)
    src_start = start + offset
    src_stop = src_start + total_needed
    x_seq = np.zeros((total_needed, cache_context.full_dim), dtype=np.float32)
    feature_mask = np.zeros((cache_context.full_dim,), dtype=np.float32)

    tx = shard["tx"]
    if feature_contract.uses_tx and isinstance(tx, np.ndarray):
        tx_column_start, tx_column_stop = signal_spec.selected_columns_for_width(
            "tx", tx.shape[1]
        )
        tx_window = np.asarray(
            tx[src_start:src_stop, tx_column_start:tx_column_stop],
            dtype=np.float32,
        )
        tx_dim = min(tx_window.shape[1], cache_context.tx_dim)
        x_seq[:, :tx_dim] = tx_window[:, :tx_dim]
        feature_mask[:tx_dim] = 1.0

    sbp = shard["sbp"]
    if feature_contract.uses_sbp and isinstance(sbp, np.ndarray):
        sbp_column_start, sbp_column_stop = signal_spec.selected_columns_for_width(
            "sbp", sbp.shape[1]
        )
        sbp_window = np.asarray(
            sbp[src_start:src_stop, sbp_column_start:sbp_column_stop],
            dtype=np.float32,
        )
        sbp_dim = min(sbp_window.shape[1], cache_context.sbp_dim)
        sbp_start = feature_contract.feature_start(
            "sbp",
            tx_dim=int(cache_context.tx_dim),
        )
        x_seq[:, sbp_start : sbp_start + sbp_dim] = sbp_window[:, :sbp_dim]
        feature_mask[sbp_start : sbp_start + sbp_dim] = 1.0

    x_seq_t = torch.from_numpy(x_seq)
    feature_mask_t = torch.from_numpy(feature_mask)
    x_norm = normalize_segment(
        x_seq_t,
        feature_mask_t,
        session_feature_stats=cache_context.session_feature_stats,
        session_key=boundary_key,
        use_normalization=cache_context.use_normalization,
    )
    return {
        "x": x_norm,
        "feature_mask": feature_mask_t,
        "length": total_needed,
        "dataset": example.dataset,
        "session_id": example.session_id,
        "session_key": boundary_key,
        "boundary_key": boundary_key,
        "shard_relpath": example.shard_relpath,
        "has_tx": example.has_tx,
        "has_sbp": example.has_sbp,
        "orig_len": length,
    }


def stack_segment_batch(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Stack sampled segment dictionaries into the canonical training batch."""

    return {
        "x": torch.stack([item["x"] for item in samples], dim=0),
        "feature_mask": torch.stack([item["feature_mask"] for item in samples], dim=0),
        "lengths": torch.tensor([item["length"] for item in samples], dtype=torch.long),
        "datasets": [item["dataset"] for item in samples],
        "session_keys": [item["boundary_key"] for item in samples],
        "boundary_keys": [item["boundary_key"] for item in samples],
        "sessions": [item["session_id"] for item in samples],
        "shard_relpaths": [item["shard_relpath"] for item in samples],
    }


def _valid_row_weights(rows: list[ExampleRow], segment_bins: int) -> np.ndarray:
    return np.array(
        [max(0, row.n_time_bins - segment_bins + 1) for row in rows],
        dtype=np.float64,
    )


def get_sampling_plan(
    cache_context: CacheContext,
    split_name: str,
    segment_bins: int,
    dataset_weight_alpha: float,
) -> SamplingPlan:
    """Build or reuse the hierarchical dataset/shard/row sampling plan."""

    key = (split_name, int(segment_bins), float(dataset_weight_alpha))
    cached = cache_context.sampling_plan_cache.get(key)
    if cached is not None:
        return cached

    shard_rows_by_dataset: dict[str, dict[str, list[ExampleRow]]] = {}
    shard_keys_by_dataset: dict[str, list[str]] = {}
    shard_probs_by_dataset: dict[str, np.ndarray] = {}
    row_probs_within_shard_by_dataset: dict[str, dict[str, np.ndarray]] = {}
    dataset_mass: dict[str, float] = {}

    for dataset in cache_context.pretrain_datasets:
        rows = cache_context.split_rows_by_dataset[split_name][dataset]
        weights = _valid_row_weights(rows, segment_bins)
        keep_mask = weights > 0
        kept_rows = [row for row, keep in zip(rows, keep_mask) if keep]
        kept_weights = weights[keep_mask]
        if not kept_rows:
            continue

        dataset_mass[dataset] = float(kept_weights.sum())
        shard_rows: dict[str, list[ExampleRow]] = defaultdict(list)
        shard_weights: dict[str, list[float]] = defaultdict(list)
        for row, weight in zip(kept_rows, kept_weights):
            shard_rows[row.shard_relpath].append(row)
            shard_weights[row.shard_relpath].append(float(weight))

        shard_keys = list(shard_rows)
        shard_mass = np.array(
            [sum(shard_weights[name]) for name in shard_keys],
            dtype=np.float64,
        )
        shard_rows_by_dataset[dataset] = dict(shard_rows)
        shard_keys_by_dataset[dataset] = shard_keys
        shard_probs_by_dataset[dataset] = shard_mass / shard_mass.sum()
        row_probs_within_shard_by_dataset[dataset] = {
            name: np.array(weight_list, dtype=np.float64) / np.sum(weight_list)
            for name, weight_list in shard_weights.items()
        }

    dataset_names = tuple(
        dataset for dataset in cache_context.pretrain_datasets if dataset in dataset_mass
    )
    if not dataset_names:
        raise RuntimeError(
            f"Split {split_name} has no datasets with enough bins for "
            f"segment_bins={segment_bins}"
        )
    dataset_probs = np.array(
        [dataset_mass[dataset] ** dataset_weight_alpha for dataset in dataset_names],
        dtype=np.float64,
    )
    dataset_probs = dataset_probs / dataset_probs.sum()

    plan = SamplingPlan(
        split_name=split_name,
        segment_bins=int(segment_bins),
        dataset_weight_alpha=float(dataset_weight_alpha),
        dataset_names=dataset_names,
        dataset_probs=dataset_probs,
        shard_rows_by_dataset=shard_rows_by_dataset,
        shard_keys_by_dataset=shard_keys_by_dataset,
        shard_probs_by_dataset=shard_probs_by_dataset,
        row_probs_within_shard_by_dataset=row_probs_within_shard_by_dataset,
    )
    cache_context.sampling_plan_cache[key] = plan
    return plan


class SegmentBatchSampler:
    """Stateful reproducible sampler for fixed-length segment batches."""

    def __init__(
        self,
        cache_context: CacheContext,
        split_name: str,
        segment_bins: int,
        batch_size: int,
        seed: int,
        dataset_weight_alpha: float,
        examples_per_shard: int,
    ) -> None:
        self.cache_context = cache_context
        self.split_name = split_name
        self.segment_bins = int(segment_bins)
        self.batch_size = int(batch_size)
        self.examples_per_shard = max(1, int(examples_per_shard))
        self.seed = int(seed)
        self.plan = get_sampling_plan(
            cache_context,
            split_name,
            self.segment_bins,
            dataset_weight_alpha,
        )
        self.py_rng = random.Random(self.seed)
        self.np_rng = np.random.default_rng(self.seed)

    def sample_batch(self, batch_size: int | None = None) -> dict[str, Any]:
        batch_size = self.batch_size if batch_size is None else int(batch_size)
        requested_dataset_idx = self.np_rng.choice(
            len(self.plan.dataset_names),
            size=batch_size,
            p=self.plan.dataset_probs,
        )
        dataset_counts = Counter(
            self.plan.dataset_names[int(idx)] for idx in requested_dataset_idx
        )

        samples: list[dict[str, Any]] = []
        for dataset, n_examples in dataset_counts.items():
            shard_keys = self.plan.shard_keys_by_dataset[dataset]
            shard_probs = self.plan.shard_probs_by_dataset[dataset]
            n_shards = max(1, math.ceil(n_examples / self.examples_per_shard))
            sampled_shard_idx = self.np_rng.choice(
                len(shard_keys),
                size=n_shards,
                replace=n_shards > len(shard_keys),
                p=shard_probs,
            )

            remaining = int(n_examples)
            for shard_choice_idx, shard_idx in enumerate(np.atleast_1d(sampled_shard_idx)):
                take = min(self.examples_per_shard, remaining)
                if shard_choice_idx == n_shards - 1:
                    take = remaining
                shard_key = shard_keys[int(shard_idx)]
                shard_rows = self.plan.shard_rows_by_dataset[dataset][shard_key]
                row_probs = self.plan.row_probs_within_shard_by_dataset[dataset][shard_key]
                row_choices = self.np_rng.choice(
                    len(shard_rows),
                    size=take,
                    replace=True,
                    p=row_probs,
                )
                for row_idx in np.atleast_1d(row_choices):
                    samples.append(
                        sample_base_segment(
                            self.cache_context,
                            shard_rows[int(row_idx)],
                            segment_bins=self.segment_bins,
                            py_rng=self.py_rng,
                        )
                    )
                remaining -= take
                if remaining <= 0:
                    break

        order = self.np_rng.permutation(len(samples))
        return stack_segment_batch([samples[int(idx)] for idx in order])


def build_segment_sampler(
    cache_context: CacheContext,
    split_name: str,
    batch_size: int,
    *,
    seed: int,
    segment_bins: int,
    dataset_weight_alpha: float,
    examples_per_shard: int,
) -> SegmentBatchSampler:
    """Construct a segment sampler after validating split eligibility."""

    if split_name == "val" and not cache_context.has_val_datasets:
        raise RuntimeError("No validation datasets are eligible for session-disjoint validation.")
    return SegmentBatchSampler(
        cache_context=cache_context,
        split_name=split_name,
        segment_bins=segment_bins,
        batch_size=batch_size,
        seed=seed,
        dataset_weight_alpha=dataset_weight_alpha,
        examples_per_shard=examples_per_shard,
    )


__all__ = [
    "SamplingPlan",
    "SegmentBatchSampler",
    "build_segment_sampler",
    "get_sampling_plan",
    "normalize_segment",
    "sample_base_segment",
    "stack_segment_batch",
]

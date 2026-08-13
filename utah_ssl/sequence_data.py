"""PyTorch datasets, collation, and length-aware batching for neural sequences."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
import torch

from .canonical_data import CanonicalProbeManifestRow, CanonicalShardAccessor
from .experiment_contract import SignalSpec
from .feature_stats import FeatureStats, apply_feature_stats
from .session_keys import resolve_boundary_key


def canonical_row_input_length(row: CanonicalProbeManifestRow) -> int:
    if row.n_time_bins is None:
        raise ValueError(f"Canonical row is missing n_time_bins metadata: {row.example_id}")
    length = int(row.n_time_bins)
    if length <= 0:
        raise ValueError(
            f"Canonical row has non-positive n_time_bins={length}: {row.example_id}"
        )
    return length


def canonical_rows_padded_time_percentile(
    rows: Sequence[CanonicalProbeManifestRow],
    *,
    percentile: float,
) -> int:
    if not rows:
        raise ValueError("Cannot compute a padded-time percentile on an empty row set.")
    if not 0.0 < float(percentile) <= 100.0:
        raise ValueError("percentile must be in (0, 100].")
    lengths = np.array([canonical_row_input_length(row) for row in rows], dtype=np.float64)
    value = np.percentile(lengths, float(percentile), method="linear")
    return max(1, int(math.ceil(float(value))))


class LengthAwareBatchSampler(torch.utils.data.Sampler[list[int]]):
    """Form deterministic batches bounded by example count and padded time."""

    def __init__(
        self,
        rows: Sequence[CanonicalProbeManifestRow],
        *,
        max_examples_per_microbatch: int,
        max_padded_time_per_microbatch: int,
        shuffle: bool,
        seed: int,
    ) -> None:
        if int(max_examples_per_microbatch) <= 0:
            raise ValueError("max_examples_per_microbatch must be positive")
        if int(max_padded_time_per_microbatch) <= 0:
            raise ValueError("max_padded_time_per_microbatch must be positive")
        self.rows = tuple(rows)
        self.max_examples_per_microbatch = int(max_examples_per_microbatch)
        self.max_padded_time_per_microbatch = int(max_padded_time_per_microbatch)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self._iteration_count = 0

    def state_dict(self) -> dict[str, int]:
        return {"iteration_count": self._iteration_count}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        iteration_count = int(state.get("iteration_count", 0))
        if iteration_count < 0:
            raise ValueError("iteration_count must be non-negative")
        self._iteration_count = iteration_count

    def _ordered_indices(self, *, iteration_count: int | None = None) -> list[int]:
        indices = list(range(len(self.rows)))
        if not self.shuffle:
            return indices
        iteration = self._iteration_count if iteration_count is None else int(iteration_count)
        if iteration < 0:
            raise ValueError("iteration_count must be non-negative")
        generator = torch.Generator()
        generator.manual_seed(self.seed + iteration)
        return torch.randperm(len(indices), generator=generator).tolist()

    def _build_batches(self, ordered_indices: Sequence[int]) -> list[list[int]]:
        batches: list[list[int]] = []
        current: list[int] = []
        current_max_length = 0
        for row_idx in ordered_indices:
            row_length = canonical_row_input_length(self.rows[row_idx])
            proposed_count = len(current) + 1
            proposed_max_length = max(current_max_length, row_length)
            if current and (
                proposed_count > self.max_examples_per_microbatch
                or proposed_count * proposed_max_length > self.max_padded_time_per_microbatch
            ):
                batches.append(current)
                current = []
                current_max_length = 0
            current.append(int(row_idx))
            current_max_length = max(current_max_length, row_length)
        if current:
            batches.append(current)
        return batches

    def num_batches_for_iteration(self, iteration_count: int) -> int:
        return len(self._build_batches(self._ordered_indices(iteration_count=iteration_count)))

    def __iter__(self) -> Iterator[list[int]]:
        batches = self._build_batches(self._ordered_indices())
        if self.shuffle:
            self._iteration_count += 1
        yield from batches

    def __len__(self) -> int:
        return self.num_batches_for_iteration(self._iteration_count)


class CanonicalSequenceDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        rows: tuple[CanonicalProbeManifestRow, ...] | list[CanonicalProbeManifestRow],
        *,
        cache_root: str | Path,
        signal_spec: SignalSpec | dict[str, Any],
        stats: FeatureStats | None = None,
        boundary_key_mode: str = "session",
        dataset: str = "brain2text25",
        input_tail_bins: int | None = None,
        pad_feature_dim_to: int | None = None,
    ) -> None:
        self.rows = list(rows)
        self.stats = stats
        self.signal_spec = SignalSpec.from_value(signal_spec)
        self.feature_mode = self.signal_spec.mode
        self.boundary_key_mode = str(boundary_key_mode)
        self.dataset = str(dataset)
        self.input_tail_bins = int(input_tail_bins) if input_tail_bins is not None else None
        self.pad_feature_dim_to = (
            int(pad_feature_dim_to) if pad_feature_dim_to is not None else None
        )
        self._accessor = CanonicalShardAccessor(cache_root)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.rows[idx]
        x = self._accessor.load_features(row, signal_spec=self.signal_spec)
        if self.stats is not None:
            x = apply_feature_stats(x, row=row, stats=self.stats)
        else:
            x = np.array(x, dtype=np.float32, copy=True)
        if self.pad_feature_dim_to is not None:
            if x.shape[1] > self.pad_feature_dim_to:
                raise ValueError(
                    f"Example {row.example_id} has feature dim {x.shape[1]}, "
                    f"which exceeds requested padded dim {self.pad_feature_dim_to}."
                )
            if x.shape[1] < self.pad_feature_dim_to:
                x = np.pad(x, ((0, 0), (0, self.pad_feature_dim_to - x.shape[1])))
        if self.input_tail_bins is not None and x.shape[0] > self.input_tail_bins:
            x = x[-self.input_tail_bins :, :]
        labels = self._accessor.load_labels(row)
        if labels is None:
            labels = np.zeros((0,), dtype=np.int64)
        return {
            "x": torch.from_numpy(x),
            "input_length": int(x.shape[0]),
            "labels": torch.from_numpy(labels),
            "label_length": int(labels.shape[0]),
            "session_id": row.session_id,
            "boundary_key": resolve_boundary_key(
                dataset=self.dataset,
                session_id=row.session_id,
                subject_id=row.subject_id,
                boundary_key_mode=self.boundary_key_mode,
            ),
            "example_id": row.example_id,
        }

    def __del__(self) -> None:
        accessor = getattr(self, "_accessor", None)
        if accessor is not None:
            accessor.close()


def collate_sequence_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    if not batch:
        raise ValueError("Cannot collate an empty sequence batch.")
    max_time = max(item["input_length"] for item in batch)
    max_label = max(item["label_length"] for item in batch)
    input_dim = int(batch[0]["x"].shape[1])
    x = torch.zeros((len(batch), max_time, input_dim), dtype=torch.float32)
    labels = torch.zeros((len(batch), max_label), dtype=torch.int64)
    input_lengths = torch.empty((len(batch),), dtype=torch.long)
    label_lengths = torch.empty((len(batch),), dtype=torch.long)

    for idx, item in enumerate(batch):
        input_length = int(item["input_length"])
        label_length = int(item["label_length"])
        x[idx, :input_length] = item["x"]
        if label_length:
            labels[idx, :label_length] = item["labels"]
        input_lengths[idx] = input_length
        label_lengths[idx] = label_length
    return {
        "x": x,
        "labels": labels,
        "input_lengths": input_lengths,
        "label_lengths": label_lengths,
        "session_ids": [item["session_id"] for item in batch],
        "boundary_keys": [item["boundary_key"] for item in batch],
        "example_ids": [item["example_id"] for item in batch],
    }


__all__ = [
    "CanonicalSequenceDataset",
    "LengthAwareBatchSampler",
    "canonical_row_input_length",
    "canonical_rows_padded_time_percentile",
    "collate_sequence_batch",
]

"""Shared CTC data and phoneme metric helpers."""

from __future__ import annotations

import math
from collections import Counter
from typing import Any

import torch
import torch.nn.functional as F

try:
    from masked_ssl.probe import (
        DEFAULT_PHONEME_VOCABULARY,
        CanonicalProbeManifestRow,
        CanonicalSequenceDataset,
        LengthAwareBatchSampler,
        build_competition_split_problem,
        build_source_split_problem,
        canonical_rows_padded_time_percentile,
        collate_sequence_batch,
    )
except ModuleNotFoundError:  # pragma: no cover - repo-root unittest fallback
    from analysis.active.ssl_experiments.masked_ssl.probe import (
        DEFAULT_PHONEME_VOCABULARY,
        CanonicalProbeManifestRow,
        CanonicalSequenceDataset,
        LengthAwareBatchSampler,
        build_competition_split_problem,
        build_source_split_problem,
        canonical_rows_padded_time_percentile,
        collate_sequence_batch,
    )


def flatten_ctc_targets(labels: torch.Tensor, label_lengths: torch.Tensor) -> torch.Tensor:
    pieces = [
        labels[row_idx, : int(length)]
        for row_idx, length in enumerate(label_lengths.tolist())
        if int(length) > 0
    ]
    if not pieces:
        return labels.new_zeros((0,), dtype=torch.long)
    return torch.cat(pieces, dim=0)


def compute_ctc_loss_sum(
    logits: torch.Tensor,
    token_lengths: torch.Tensor,
    labels: torch.Tensor,
    label_lengths: torch.Tensor,
    *,
    blank_index: int,
) -> tuple[torch.Tensor, int]:
    target_count = int(label_lengths.sum().item())
    if target_count <= 0:
        return logits.new_zeros(()), 0
    log_probs = F.log_softmax(logits, dim=-1).transpose(0, 1)
    loss_sum = F.ctc_loss(
        log_probs,
        flatten_ctc_targets(labels, label_lengths),
        token_lengths,
        label_lengths,
        blank=int(blank_index),
        reduction="sum",
        zero_infinity=True,
    )
    return loss_sum, target_count


def ctc_bits_per_target(loss_sum: torch.Tensor | float, target_count: int) -> float:
    if int(target_count) <= 0:
        raise ValueError("target_count must be positive")
    value = float(loss_sum.item()) if isinstance(loss_sum, torch.Tensor) else float(loss_sum)
    return value / int(target_count) / math.log(2.0)


def ctc_greedy_decode(
    logits: torch.Tensor,
    token_lengths: torch.Tensor,
    *,
    blank_index: int,
) -> list[list[int]]:
    token_ids = logits.argmax(dim=-1)
    decoded: list[list[int]] = []
    for batch_idx, length in enumerate(token_lengths.tolist()):
        sequence: list[int] = []
        prev_token: int | None = None
        for token in token_ids[batch_idx, : int(length)].tolist():
            token = int(token)
            if token == int(blank_index):
                prev_token = None
                continue
            if token != prev_token:
                sequence.append(token)
            prev_token = token
        decoded.append(sequence)
    return decoded


def edit_counts(reference: list[int], hypothesis: list[int]) -> tuple[int, int, int]:
    rows = len(reference) + 1
    cols = len(hypothesis) + 1
    distances = [[0] * cols for _ in range(rows)]
    ops = [[(0, 0, 0)] * cols for _ in range(rows)]
    for row_idx in range(1, rows):
        distances[row_idx][0] = row_idx
        ops[row_idx][0] = (0, row_idx, 0)
    for col_idx in range(1, cols):
        distances[0][col_idx] = col_idx
        ops[0][col_idx] = (col_idx, 0, 0)
    for row_idx in range(1, rows):
        for col_idx in range(1, cols):
            if reference[row_idx - 1] == hypothesis[col_idx - 1]:
                distances[row_idx][col_idx] = distances[row_idx - 1][col_idx - 1]
                ops[row_idx][col_idx] = ops[row_idx - 1][col_idx - 1]
                continue
            substitution = distances[row_idx - 1][col_idx - 1] + 1
            insertion = distances[row_idx][col_idx - 1] + 1
            deletion = distances[row_idx - 1][col_idx] + 1
            best = min(substitution, insertion, deletion)
            distances[row_idx][col_idx] = best
            if best == substitution:
                prev = ops[row_idx - 1][col_idx - 1]
                ops[row_idx][col_idx] = (prev[0], prev[1], prev[2] + 1)
            elif best == insertion:
                prev = ops[row_idx][col_idx - 1]
                ops[row_idx][col_idx] = (prev[0] + 1, prev[1], prev[2])
            else:
                prev = ops[row_idx - 1][col_idx]
                ops[row_idx][col_idx] = (prev[0], prev[1] + 1, prev[2])
    return ops[-1][-1]


def phoneme_error_diagnostics(
    *,
    logits: torch.Tensor,
    token_lengths: torch.Tensor,
    labels: torch.Tensor,
    label_lengths: torch.Tensor,
    blank_index: int,
    top_k: int = 10,
) -> dict[str, Any]:
    predictions = ctc_greedy_decode(logits, token_lengths, blank_index=int(blank_index))
    total_insertions = 0
    total_deletions = 0
    total_substitutions = 0
    total_reference_tokens = 0
    total_predicted_tokens = 0
    reference_counter: Counter[int] = Counter()
    prediction_counter: Counter[int] = Counter()
    for row_idx, prediction in enumerate(predictions):
        reference = labels[row_idx, : int(label_lengths[row_idx].item())].tolist()
        insertions, deletions, substitutions = edit_counts(reference, prediction)
        total_insertions += int(insertions)
        total_deletions += int(deletions)
        total_substitutions += int(substitutions)
        total_reference_tokens += len(reference)
        total_predicted_tokens += len(prediction)
        reference_counter.update(int(token) for token in reference)
        prediction_counter.update(int(token) for token in prediction)
    if total_reference_tokens <= 0:
        raise ValueError("Reference token count is zero; cannot compute PER.")
    total_errors = total_insertions + total_deletions + total_substitutions
    return {
        "phoneme_error_rate": float(total_errors / total_reference_tokens),
        "total_reference_tokens": int(total_reference_tokens),
        "total_predicted_tokens": int(total_predicted_tokens),
        "insertions": int(total_insertions),
        "deletions": int(total_deletions),
        "substitutions": int(total_substitutions),
        "reference_top_ids": [[int(k), int(v)] for k, v in reference_counter.most_common(top_k)],
        "prediction_top_ids": [[int(k), int(v)] for k, v in prediction_counter.most_common(top_k)],
    }


__all__ = [
    "DEFAULT_PHONEME_VOCABULARY",
    "CanonicalProbeManifestRow",
    "CanonicalSequenceDataset",
    "LengthAwareBatchSampler",
    "build_competition_split_problem",
    "build_source_split_problem",
    "canonical_rows_padded_time_percentile",
    "collate_sequence_batch",
    "compute_ctc_loss_sum",
    "ctc_bits_per_target",
    "ctc_greedy_decode",
    "edit_counts",
    "flatten_ctc_targets",
    "phoneme_error_diagnostics",
]

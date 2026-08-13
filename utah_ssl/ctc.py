"""Shared operations around PyTorch CTC loss and phoneme decoding."""

from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn.functional as F


def _flatten_ctc_targets(
    labels: torch.Tensor,
    label_lengths: torch.Tensor,
    *,
    label_length_values: Sequence[int] | None = None,
) -> torch.Tensor:
    resolved_lengths = (
        [int(length) for length in label_length_values]
        if label_length_values is not None
        else [int(length) for length in label_lengths.tolist()]
    )
    pieces = [
        labels[row_idx, :length]
        for row_idx, length in enumerate(resolved_lengths)
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
    target_count: int | None = None,
    label_length_values: Sequence[int] | None = None,
) -> tuple[torch.Tensor, int]:
    """Return PyTorch's summed CTC loss and the validated target count."""
    resolved_lengths = (
        [int(length) for length in label_length_values]
        if label_length_values is not None
        else [int(length) for length in label_lengths.tolist()]
    )
    resolved_target_count = int(sum(resolved_lengths))
    if target_count is not None and int(target_count) != resolved_target_count:
        raise ValueError(
            "target_count must equal the sum of label_length_values: "
            f"target_count={int(target_count)} length_sum={resolved_target_count}"
        )
    if resolved_target_count <= 0:
        return logits.new_zeros(()), 0
    log_probs = F.log_softmax(logits, dim=-1).transpose(0, 1)
    loss_sum = F.ctc_loss(
        log_probs,
        _flatten_ctc_targets(
            labels,
            label_lengths,
            label_length_values=resolved_lengths,
        ),
        token_lengths,
        label_lengths,
        blank=int(blank_index),
        reduction="sum",
        zero_infinity=True,
    )
    return loss_sum, resolved_target_count


def ctc_bits_per_target(loss_sum: torch.Tensor | float, target_count: int) -> float:
    """Convert summed natural-log CTC loss to bits per target phoneme."""
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
    """Collapse argmax paths into token sequences using standard CTC rules."""
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
    """Return insertion, deletion, and substitution counts."""
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


__all__ = [
    "compute_ctc_loss_sum",
    "ctc_bits_per_target",
    "ctc_greedy_decode",
    "edit_counts",
]

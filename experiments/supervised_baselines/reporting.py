"""Reporting and diagnostics helpers for the Willett reconstruction baseline."""

from __future__ import annotations

import math
from collections import Counter
from typing import Any

import pandas as pd
import torch
from torch.utils.data import DataLoader

from utah_ssl.ctc import compute_ctc_loss_sum, ctc_greedy_decode, edit_counts
from utah_ssl.decoding_preprocessing import (
    WillettInputTransformConfig,
    prepare_willett_inputs,
)
from utah_ssl.reporting import load_jsonl


def _top_counter_items(counter: Counter[int], *, top_k: int = 10) -> list[list[int]]:
    return [[int(item), int(count)] for item, count in counter.most_common(top_k)]


def evaluate_willett_phoneme_metrics(
    *,
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    blank_index: int,
    input_transform_config: WillettInputTransformConfig | None = None,
) -> dict[str, Any]:
    model.eval()
    total_loss_sum = 0.0
    total_targets = 0
    total_reference_tokens = 0
    total_predicted_tokens = 0
    total_blank_frames = 0
    total_frames = 0
    total_insertions = 0
    total_deletions = 0
    total_substitutions = 0
    reference_counter: Counter[int] = Counter()
    prediction_counter: Counter[int] = Counter()
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            input_lengths = batch["input_lengths"].to(device)
            if input_transform_config is not None:
                x = prepare_willett_inputs(
                    x,
                    input_lengths,
                    config=input_transform_config,
                    is_training=False,
                )
            labels = batch["labels"].to(device)
            label_lengths = batch["label_lengths"].to(device)
            outputs = model(x, input_lengths, session_ids=batch["boundary_keys"])
            loss_sum, target_count = compute_ctc_loss_sum(
                outputs["logits"],
                outputs["token_lengths"],
                labels,
                label_lengths,
                blank_index=int(blank_index),
            )
            total_loss_sum += float(loss_sum.item())
            total_targets += int(target_count)
            predictions = ctc_greedy_decode(
                outputs["logits"],
                outputs["token_lengths"],
                blank_index=int(blank_index),
            )
            frame_ids = outputs["logits"].argmax(dim=-1)
            for row_idx, prediction in enumerate(predictions):
                reference_length = int(label_lengths[row_idx].item())
                reference = labels[row_idx, :reference_length].tolist()
                insertions, deletions, substitutions = edit_counts(reference, prediction)
                total_insertions += int(insertions)
                total_deletions += int(deletions)
                total_substitutions += int(substitutions)
                total_reference_tokens += len(reference)
                total_predicted_tokens += len(prediction)
                total_frames += int(outputs["token_lengths"][row_idx].item())
                total_blank_frames += int(
                    (frame_ids[row_idx, : int(outputs["token_lengths"][row_idx].item())] == int(blank_index)).sum().item()
                )
                reference_counter.update(int(token) for token in reference)
                prediction_counter.update(int(token) for token in prediction)
    if total_targets <= 0:
        raise ValueError("Validation target count is zero; cannot compute CTC diagnostics.")
    if total_reference_tokens <= 0:
        raise ValueError("Validation reference token count is zero; cannot compute PER.")
    total_errors = total_insertions + total_deletions + total_substitutions
    return {
        "val_ctc_bpphone": float(total_loss_sum / total_targets / math.log(2.0)),
        "val_phoneme_error_rate": float(total_errors / total_reference_tokens),
        "edit_diagnostics": {
            "insertions": int(total_insertions),
            "deletions": int(total_deletions),
            "substitutions": int(total_substitutions),
        },
        "collapse_diagnostics": {
            "total_reference_tokens": int(total_reference_tokens),
            "total_predicted_tokens": int(total_predicted_tokens),
            "predicted_to_reference_token_ratio": float(total_predicted_tokens / total_reference_tokens),
            "blank_frame_rate": float(total_blank_frames / total_frames) if total_frames > 0 else float("nan"),
            "reference_top_ids": _top_counter_items(reference_counter),
            "prediction_top_ids": _top_counter_items(prediction_counter),
        },
    }


def display_willett_report(summary: dict[str, Any] | None) -> dict[str, Any]:
    if summary is None:
        return {"progress_df": pd.DataFrame(), "summary_df": pd.DataFrame()}
    records = load_jsonl(summary.get("progress_log_path"))
    progress_df = pd.DataFrame(records)
    train_df = progress_df[progress_df.get("event") == "willett_train_report"].copy()
    val_df = progress_df[progress_df.get("event") == "willett_val_report"].copy()
    summary_df = pd.DataFrame(
        [
            {
                "run_name": summary.get("run_name"),
                "steps": summary.get("steps"),
                "val_ctc_bpphone": dict(summary.get("metrics", {})).get("val_ctc_bpphone"),
                "val_phoneme_error_rate": dict(summary.get("metrics", {})).get("val_phoneme_error_rate"),
                "best_val_ctc_bpphone": dict(summary.get("metrics", {})).get("best_val_ctc_bpphone"),
                "best_val_phoneme_error_rate": dict(summary.get("metrics", {})).get("best_val_phoneme_error_rate"),
            }
        ]
    )
    return {
        "progress_df": progress_df,
        "train_df": train_df,
        "val_df": val_df,
        "summary_df": summary_df,
    }


__all__ = [
    "display_willett_report",
    "evaluate_willett_phoneme_metrics",
]

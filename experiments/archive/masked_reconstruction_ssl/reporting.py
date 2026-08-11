"""Notebook reporting helpers for masked SSL experiments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SSL_RECON_SCORECARD_COLUMNS = [
    "split",
    "zero_baseline_masked_mse",
    "model_masked_mse",
    "relative_improvement_over_zero",
    "masked_target_std",
    "masked_prediction_std",
    "prediction_target_corr",
]

PROBE_VITAL_COLUMNS = [
    "model_variant",
    "downstream_ctc_bpphone",
    "downstream_per",
    "reference_output_len",
    "actual_output_len",
    "actual_over_reference_len",
    "most_common_prediction",
    "most_common_prediction_rate",
]


def _display(frame: pd.DataFrame) -> None:
    try:
        from IPython.display import display

        display(frame)
    except Exception:
        print(frame)


def _latest_record(records: list[dict[str, Any]]) -> dict[str, Any] | None:
    return records[-1] if records else None


def _zero_mse_for_split(zero_baseline_df: pd.DataFrame | None, split_name: str) -> float:
    if zero_baseline_df is None:
        return float("nan")
    matches = zero_baseline_df[zero_baseline_df["split"] == split_name]
    if matches.empty:
        return float("nan")
    return float(matches.iloc[0]["masked_zero_mse"])


def build_ssl_reconstruction_scorecard(
    run_state: dict[str, Any],
    *,
    zero_baseline_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    train_records = list(run_state.get("train_history", []))
    val_records = list(run_state.get("val_history", []))

    def row(split_name: str, record: dict[str, Any] | None) -> dict[str, Any]:
        zero_mse = _zero_mse_for_split(zero_baseline_df, split_name)
        model_mse = float(record["loss"]) if record is not None else np.nan
        relative_improvement = np.nan
        if np.isfinite(zero_mse) and zero_mse > 0 and np.isfinite(model_mse):
            relative_improvement = 1.0 - model_mse / zero_mse
        return {
            "split": split_name,
            "zero_baseline_masked_mse": zero_mse,
            "model_masked_mse": model_mse,
            "relative_improvement_over_zero": relative_improvement,
            "masked_target_std": record.get("masked_target_std", np.nan) if record is not None else np.nan,
            "masked_prediction_std": record.get("masked_prediction_std", np.nan) if record is not None else np.nan,
            "prediction_target_corr": (
                record.get("masked_prediction_target_corr", np.nan) if record is not None else np.nan
            ),
        }

    return pd.DataFrame(
        [
            row("train", _latest_record(train_records)),
            row("val", _latest_record(val_records)),
        ],
        columns=SSL_RECON_SCORECARD_COLUMNS,
    )


def plot_ssl_reconstruction_history(run_state: dict[str, Any]) -> None:
    import matplotlib.pyplot as plt

    train_records = list(run_state.get("train_history", []))
    val_records = list(run_state.get("val_history", []))
    if not train_records:
        print("No train history available.")
        return

    plt.figure(figsize=(7, 4))
    plt.plot([r["step"] for r in train_records], [r["loss"] for r in train_records], label="train")
    if val_records:
        plt.plot([r["step"] for r in val_records], [r["loss"] for r in val_records], marker="o", label="val")
    plt.xlabel("step")
    plt.ylabel("masked MSE")
    plt.title("Masked Reconstruction")
    plt.legend()
    plt.grid(True, alpha=0.25)
    plt.show()


def display_ssl_reconstruction_report(
    run_state: dict[str, Any],
    *,
    zero_baseline_df: pd.DataFrame | None = None,
    plot: bool = True,
) -> pd.DataFrame:
    if plot:
        plot_ssl_reconstruction_history(run_state)
    scorecard = build_ssl_reconstruction_scorecard(
        run_state,
        zero_baseline_df=zero_baseline_df,
    )
    _display(scorecard)
    print("Run:", run_state.get("run_name"))
    print("Checkpoint:", run_state.get("checkpoint_path"))
    return scorecard


def _load_probe_alignment_stats(summary: dict[str, Any]) -> dict[str, Any]:
    stats_path = summary.get("alignment_stats_path")
    if stats_path is None:
        return {}
    stats_path = Path(stats_path)
    if not stats_path.exists():
        return {}
    return json.loads(stats_path.read_text())


def _top_symbol_and_rate(items: list[dict[str, Any]]) -> tuple[object, object, object]:
    if not items:
        return None, None, None
    item = items[0]
    return item.get("symbol", item.get("id")), item.get("rate"), item.get("count")


def build_probe_vital_summary(summary: dict[str, Any]) -> dict[str, Any]:
    stats = _load_probe_alignment_stats(summary)
    reference_len = stats.get("total_reference_tokens")
    actual_len = stats.get("total_predicted_tokens")
    length_ratio = None
    if reference_len is not None and actual_len is not None:
        length_ratio = float(actual_len) / max(int(reference_len), 1)

    predicted_symbol = summary.get("most_common_prediction")
    predicted_rate = summary.get("most_common_prediction_rate")
    predicted_count = None
    if predicted_symbol is None or predicted_rate is None:
        predicted_symbol, predicted_rate, predicted_count = _top_symbol_and_rate(
            stats.get("prediction_histogram_top", [])
        )

    return {
        "model_variant": summary.get("model_variant"),
        "downstream_ctc_bpphone": summary.get("val_ctc_bpphone"),
        "downstream_per": summary.get("val_phoneme_error_rate"),
        "reference_output_len": reference_len,
        "actual_output_len": actual_len,
        "actual_over_reference_len": length_ratio,
        "most_common_prediction": predicted_symbol,
        "most_common_prediction_rate": predicted_rate,
        "most_common_prediction_count": predicted_count,
        "summary_path": summary.get("summary_path"),
        "alignment_stats_path": summary.get("alignment_stats_path"),
    }


def _fmt_pct(value: object) -> str:
    return "n/a" if value is None else f"{100.0 * float(value):.2f}%"


def _fmt_float(value: object, digits: int = 4) -> str:
    return "n/a" if value is None else f"{float(value):.{digits}f}"


def print_probe_vital_stats(summary: dict[str, Any]) -> dict[str, Any]:
    row = build_probe_vital_summary(summary)
    name = row.get("model_variant") or "probe_summary"
    print(
        f"{name}: CTC={_fmt_float(row['downstream_ctc_bpphone'])} bits/phoneme, "
        f"PER={_fmt_pct(row['downstream_per'])}, "
        f"len={row['actual_output_len']}/{row['reference_output_len']} "
        f"({_fmt_float(row['actual_over_reference_len'], digits=3)}x), "
        f"top_pred={row['most_common_prediction']} "
        f"({_fmt_pct(row['most_common_prediction_rate'])})"
    )
    return row


def display_probe_summaries(*summary_dicts: dict[str, Any] | None) -> pd.DataFrame:
    rows = [summary for summary in summary_dicts if summary is not None]
    if not rows:
        print("No summaries to display.")
        return pd.DataFrame()

    vital_rows = [build_probe_vital_summary(summary) for summary in rows]
    frame = pd.DataFrame(vital_rows)
    _display(frame[PROBE_VITAL_COLUMNS])
    for summary in rows:
        print_probe_vital_stats(summary)
    return frame

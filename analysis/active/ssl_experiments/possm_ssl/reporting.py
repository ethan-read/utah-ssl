"""Notebook reporting helpers for POSSM experiments."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import DataLoader

from masked_ssl.probe import CanonicalSequenceDataset, compute_feature_stats

from .phoneme_finetune import (
    POSSMFinetuneConfig,
    _build_problem,
    _ctc_greedy_decode,
    _loader_kwargs,
    _prepare_stage2_inputs,
    recover_possm_stage1_sequence_components,
)
from .model import POSSMPhonemeModel


def _maybe_display(value: Any) -> None:
    try:
        display(value)  # type: ignore[name-defined]
    except NameError:
        return


def _read_jsonl(path: str | Path | None) -> list[dict[str, Any]]:
    if path is None:
        return []
    resolved = Path(path)
    if not resolved.exists():
        return []
    return [json.loads(line) for line in resolved.read_text().splitlines() if line.strip()]


def display_possm_stage1_report(
    run_state: dict[str, Any] | None,
    *,
    plot: bool = True,
    tail: int = 10,
) -> dict[str, Any]:
    if run_state is None:
        print("No Stage-1 run state available.")
        return {"progress_df": pd.DataFrame(), "latest_train": None, "latest_val": None}

    records = _read_jsonl(run_state.get("progress_path"))
    progress_df = pd.DataFrame(records)
    if progress_df.empty:
        print("No Stage-1 progress rows found.")
        return {"progress_df": progress_df, "latest_train": None, "latest_val": None}

    train_df = progress_df[progress_df.get("event") == "train"].copy()
    val_df = progress_df[progress_df.get("event") == "val"].copy()
    _maybe_display(progress_df.tail(int(tail)))

    if plot and not train_df.empty:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(8, 4))
        plt.plot(train_df["step"], train_df["loss"], label="train MSE")
        if not val_df.empty:
            plt.plot(val_df["step"], val_df["loss"], marker="o", label="val MSE")
        plt.title("Stage-1 POSSM Reconstruction Loss")
        plt.xlabel("step")
        plt.ylabel("MSE")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.show()

    latest_train = train_df.iloc[-1].to_dict() if not train_df.empty else None
    latest_val = val_df.iloc[-1].to_dict() if not val_df.empty else None
    return {
        "progress_df": progress_df,
        "latest_train": latest_train,
        "latest_val": latest_val,
    }


def display_possm_stage2_report(
    summary: dict[str, Any] | None,
    *,
    plot: bool = True,
    tail: int = 10,
) -> dict[str, Any]:
    if summary is None:
        print("No Stage-2 summary available.")
        return {"progress_df": pd.DataFrame(), "latest_train": None, "latest_val": None}

    records = _read_jsonl(summary.get("progress_log_path"))
    progress_df = pd.DataFrame(records)
    if progress_df.empty:
        print("No Stage-2 progress rows found.")
        return {"progress_df": progress_df, "latest_train": None, "latest_val": None}

    train_df = progress_df[progress_df.get("event") == "phoneme_train_report"].copy()
    val_df = progress_df[progress_df.get("event") == "phoneme_val_report"].copy()
    _maybe_display(progress_df.tail(int(tail)))

    if plot:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
        if not train_df.empty:
            axes[0].plot(train_df["step"], train_df["train_ctc_bpphone"], label="train CTC")
        if not val_df.empty:
            axes[0].plot(val_df["step"], val_df["val_ctc_bpphone"], marker="o", label="val CTC")
            axes[1].plot(val_df["step"], val_df["val_phoneme_error_rate"], marker="o", label="val PER")
        axes[0].set_title("Stage-2 CTC Loss")
        axes[0].set_xlabel("step")
        axes[0].set_ylabel("bits / phoneme")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()
        axes[1].set_title("Stage-2 PER")
        axes[1].set_xlabel("step")
        axes[1].set_ylabel("PER")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()
        plt.tight_layout()
        plt.show()

    latest_train = train_df.iloc[-1].to_dict() if not train_df.empty else None
    latest_val = val_df.iloc[-1].to_dict() if not val_df.empty else None
    return {
        "progress_df": progress_df,
        "latest_train": latest_train,
        "latest_val": latest_val,
    }


def display_possm_stage2_summary(summary: dict[str, Any] | None) -> dict[str, pd.DataFrame]:
    if summary is None:
        print("No Stage-2 summary available.")
        empty = pd.DataFrame()
        return {"summary": empty, "collapse": empty}

    metrics = dict(summary.get("metrics", {}))
    summary_df = pd.DataFrame(
        [
            {
                "run_name": summary.get("run_name"),
                "mode": summary.get("mode") or dict(summary.get("config", {})).get("mode"),
                "decoder_backbone_type": summary.get("decoder_backbone_type")
                or dict(summary.get("config", {})).get("decoder_backbone_type"),
                "s5_direction": summary.get("s5_direction")
                or dict(summary.get("config", {})).get("s5_direction"),
                "s5_implementation": summary.get("s5_implementation")
                or dict(summary.get("config", {})).get("s5_implementation"),
                "steps": summary.get("steps"),
                "val_ctc_bpphone": metrics.get("val_ctc_bpphone"),
                "val_phoneme_error_rate": metrics.get("val_phoneme_error_rate"),
                "best_val_ctc_bpphone": metrics.get("best_val_ctc_bpphone"),
                "best_val_phoneme_error_rate": metrics.get("best_val_phoneme_error_rate"),
                "resume_checkpoint_path": summary.get("resume_checkpoint_path"),
                "checkpoint_best_path": summary.get("checkpoint_best_path"),
                "checkpoint_final_path": summary.get("checkpoint_final_path"),
            }
        ]
    )
    _maybe_display(summary_df)

    collapse = metrics.get("collapse_diagnostics") or metrics.get("best_collapse_diagnostics")
    collapse_df = pd.DataFrame()
    if isinstance(collapse, dict):
        collapse_df = pd.DataFrame(
            [
                {
                    "blank_frame_rate": collapse.get("blank_frame_rate"),
                    "predicted_to_reference_token_ratio": collapse.get(
                        "predicted_to_reference_token_ratio"
                    ),
                    "total_predicted_tokens": collapse.get("total_predicted_tokens"),
                    "total_reference_tokens": collapse.get("total_reference_tokens"),
                    "prediction_top_ids": collapse.get("prediction_top_ids"),
                    "reference_top_ids": collapse.get("reference_top_ids"),
                }
            ]
        )
        _maybe_display(collapse_df)
    return {"summary": summary_df, "collapse": collapse_df}


def run_possm_stage1_prediction_diagnostics(
    run_state: dict[str, Any],
    *,
    device: torch.device,
    batches: int = 4,
) -> pd.DataFrame:
    sampler = run_state.get("val_sampler") or run_state.get("train_sampler")
    if sampler is None:
        raise RuntimeError("No Stage-1 sampler available for diagnostics.")
    model = run_state["model"]
    objective = run_state["objective"]
    config_payload = dict(run_state.get("config", {}))
    was_training = bool(model.training)
    model.eval()
    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for batch_idx in range(int(batches)):
            raw_batch = sampler.sample_batch()
            batch = objective.prepare_batch(raw_batch, device=device, config=config_payload)
            outputs = model(batch.x_input, batch.lengths, session_ids=batch.session_ids)
            valid_time = (
                torch.arange(batch.x_target.shape[1], device=batch.lengths.device).unsqueeze(0)
                < batch.lengths.unsqueeze(1)
            )
            valid = valid_time.unsqueeze(-1) & batch.feature_mask.bool().unsqueeze(1)
            pred = outputs["reconstruction"].masked_select(valid)
            target = batch.x_target.masked_select(valid)
            if pred.numel() == 0:
                continue
            corr = float("nan")
            if pred.numel() > 1:
                pred_centered = pred - pred.mean()
                target_centered = target - target.mean()
                denom = pred_centered.norm() * target_centered.norm()
                if float(denom.item()) > 0.0:
                    corr = float((pred_centered * target_centered).sum().div(denom).item())
            rows.append(
                {
                    "batch": int(batch_idx),
                    "mse": float(torch.mean((pred - target).pow(2)).item()),
                    "zero_mse": float(torch.mean(target.pow(2)).item()),
                    "pred_mean": float(pred.mean().item()),
                    "target_mean": float(target.mean().item()),
                    "pred_std": float(pred.std(unbiased=False).item()) if pred.numel() > 1 else 0.0,
                    "target_std": float(target.std(unbiased=False).item()) if target.numel() > 1 else 0.0,
                    "prediction_target_corr": corr,
                }
            )
    if was_training:
        model.train()
    frame = pd.DataFrame(rows)
    _maybe_display(frame)
    return frame


def run_possm_stage2_prediction_diagnostics(
    summary: dict[str, Any],
    *,
    device: torch.device,
    batches: int = 4,
) -> pd.DataFrame:
    checkpoint_path = summary.get("checkpoint_best_path") or summary.get("resume_checkpoint_path")
    if checkpoint_path is None:
        raise ValueError("Stage-2 summary does not include a checkpoint path for diagnostics.")
    payload = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=False)
    config = POSSMFinetuneConfig(**dict(payload["config"]))
    vocab = dict(payload["vocab"])
    blank_index = int(vocab["blank_index"])
    id_to_symbol = {int(k): v for k, v in vocab.get("id_to_symbol", {}).items()}
    if not id_to_symbol and "symbols" in vocab:
        id_to_symbol = {idx: symbol for idx, symbol in enumerate(vocab["symbols"])}

    base_encoder, pre_decoder_backbone, _, _ = recover_possm_stage1_sequence_components(
        checkpoint_path=Path(payload["stage1_checkpoint_path"]),
        map_location="cpu",
    )
    problem = _build_problem(
        cache_root=Path(payload["cache_root"]),
        config=config,
        feature_mode=str(payload.get("feature_mode", config.feature_mode)),
        boundary_key_mode=str(dict(payload["config"]).get("boundary_key_mode", config.boundary_key_mode)),
    )
    stats = (
        compute_feature_stats(
            problem["train_rows"],
            cache_root=Path(problem["cache_root"]),
            mode="global",
            feature_mode=str(problem["feature_mode"]),
        )
        if str(config.data_mode) == "normalized"
        else None
    )
    loader = DataLoader(
        CanonicalSequenceDataset(
            problem["val_rows"],
            cache_root=Path(problem["cache_root"]),
            stats=stats,
            feature_mode=str(problem["feature_mode"]),
            boundary_key_mode=str(problem.get("boundary_key_mode", "session")),
            dataset=str(problem.get("dataset", config.dataset)),
        ),
        batch_size=1,
        shuffle=False,
        **_loader_kwargs(device),
    )
    model = POSSMPhonemeModel(
        base_encoder=base_encoder,
        pre_decoder_backbone=pre_decoder_backbone,
        vocab_size=int(vocab["num_classes"]),
        gru_hidden_size=int(config.gru_hidden_size),
        gru_num_layers=int(config.gru_num_layers),
        gru_dropout=float(config.gru_dropout),
        decoder_backbone_type=str(config.decoder_backbone_type),
        s5_hidden_size=int(config.s5_hidden_size),
        s5_state_size=int(config.s5_state_size),
        s5_num_layers=int(config.s5_num_layers),
        s5_dropout=float(config.s5_dropout),
        s5_direction=str(config.s5_direction),
        s5_ffn_multiplier=float(config.s5_ffn_multiplier),
        s5_implementation=str(config.s5_implementation),
        conv_hidden_size=config.conv_hidden_size,
        conv_kernel_size=int(config.conv_kernel_size),
        conv_stride=int(config.conv_stride),
        conv_dropout=float(config.conv_dropout),
        session_adapter_keys=tuple(payload.get("session_adapter_keys", ())),
        session_adapter_enabled=bool(config.session_adapter_enabled),
    )
    model.load_state_dict(payload["model_state"])
    model.to(device)
    model.eval()

    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if batch_idx >= int(batches):
                break
            x = batch["x"].to(device)
            input_lengths = batch["input_lengths"].to(device)
            x = _prepare_stage2_inputs(x, input_lengths, config=config, is_training=False)
            outputs = model(x, input_lengths, session_ids=batch["boundary_keys"])
            predictions = _ctc_greedy_decode(outputs["logits"], outputs["token_lengths"], blank_index=blank_index)
            labels = batch["labels"]
            label_lengths = batch["label_lengths"]
            for row_idx, prediction in enumerate(predictions):
                reference_ids = labels[row_idx, : int(label_lengths[row_idx].item())].tolist()
                rows.append(
                    {
                        "batch": int(batch_idx),
                        "reference": " ".join(id_to_symbol.get(int(token), str(int(token))) for token in reference_ids),
                        "prediction": " ".join(id_to_symbol.get(int(token), str(int(token))) for token in prediction),
                        "reference_len": int(len(reference_ids)),
                        "prediction_len": int(len(prediction)),
                    }
                )
    frame = pd.DataFrame(rows)
    _maybe_display(frame)
    return frame


def summarize_possm_stage2_progress(summary: dict[str, Any] | None) -> pd.DataFrame:
    if summary is None:
        return pd.DataFrame()
    records = _read_jsonl(summary.get("progress_log_path"))
    if not records:
        return pd.DataFrame()
    frame = pd.DataFrame(records)
    return frame.tail(1)


def bits_to_nats(bits: float) -> float:
    return float(bits) * math.log(2.0)

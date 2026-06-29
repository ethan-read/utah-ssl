"""Export Willett-style decoder representations for manifold analyses."""

from __future__ import annotations

import json
import math
import shutil
import subprocess
from dataclasses import asdict, dataclass
from dataclasses import replace as dataclass_replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

try:
    from recompute_split_feature_stats import (
        load_precomputed_split_feature_stats,
        resolve_precomputed_split_stats_path,
    )
    from ssl_core.ctc import CanonicalSequenceDataset, ctc_greedy_decode
except ModuleNotFoundError:  # pragma: no cover - repo-root unittest fallback
    from analysis.active.ssl_experiments.recompute_split_feature_stats import (
        load_precomputed_split_feature_stats,
        resolve_precomputed_split_stats_path,
    )
    from analysis.active.ssl_experiments.ssl_core.ctc import (
        CanonicalSequenceDataset,
        ctc_greedy_decode,
    )

from .data import (
    ConcatenatedPredictedTxSequenceDataset,
    FuturePredictionExportAccessor,
    WillettInputTransformConfig,
    adapter_keys_from_rows,
    build_willett_problem,
    compute_predicted_tx_normalization_stats,
    compute_willett_normalization_stats,
    loader_kwargs,
    make_length_aware_batch_sampler,
    normalization_stats_missing_rows,
    prepare_willett_inputs,
)
from .model import WillettPhonemeModel
from .train import WillettReconstructionConfig


PHONEME_CATEGORY_BY_SYMBOL: dict[str, str] = {
    "BLANK": "blank",
    "SIL": "silence",
    "AA": "vowel",
    "AE": "vowel",
    "AH": "vowel",
    "AO": "vowel",
    "AW": "vowel",
    "AY": "vowel",
    "EH": "vowel",
    "ER": "vowel",
    "EY": "vowel",
    "IH": "vowel",
    "IY": "vowel",
    "OW": "vowel",
    "OY": "vowel",
    "UH": "vowel",
    "UW": "vowel",
    "B": "stop",
    "D": "stop",
    "G": "stop",
    "K": "stop",
    "P": "stop",
    "T": "stop",
    "DH": "fricative",
    "F": "fricative",
    "HH": "fricative",
    "S": "fricative",
    "SH": "fricative",
    "TH": "fricative",
    "V": "fricative",
    "Z": "fricative",
    "ZH": "fricative",
    "CH": "affricate",
    "JH": "affricate",
    "M": "nasal",
    "N": "nasal",
    "NG": "nasal",
    "L": "liquid",
    "R": "liquid",
    "W": "glide",
    "Y": "glide",
}
PHONEME_CATEGORY_ORDER: tuple[str, ...] = (
    "blank",
    "silence",
    "vowel",
    "stop",
    "fricative",
    "affricate",
    "nasal",
    "liquid",
    "glide",
    "other",
)
CONSONANT_CATEGORIES = frozenset(("stop", "fricative", "affricate", "nasal", "liquid", "glide"))


@dataclass(frozen=True)
class RepresentationExportConfig:
    checkpoint_path: str | Path
    export_root: str | Path
    model_key: str
    split: str = "val"
    allowed_session_ids: tuple[str, ...] | list[str] | None = None
    max_examples: int | None = None
    batch_size: int = 32
    shard_size_tokens: int = 50000
    bin_size_ms: int = 20
    overwrite: bool = False
    device: str | None = None
    repo_dir: str | Path | None = None
    cache_root_override: str | Path | None = None
    precomputed_split_stats_path_override: str | Path | None = None
    save_input_windows: bool = False


def _timestamp_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _detect_device(preferred: str | None = None) -> torch.device:
    if preferred:
        return torch.device(str(preferred))
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def id_to_symbol_from_vocab(vocab: dict[str, Any]) -> dict[int, str]:
    if isinstance(vocab.get("id_to_symbol"), dict):
        return {int(key): str(value) for key, value in dict(vocab["id_to_symbol"]).items()}
    if isinstance(vocab.get("index_to_symbol"), list):
        return {idx: str(symbol) for idx, symbol in enumerate(vocab["index_to_symbol"])}
    if isinstance(vocab.get("symbols"), list):
        return {idx: str(symbol) for idx, symbol in enumerate(vocab["symbols"])}
    return {}


def category_for_symbol(symbol: str) -> str:
    return PHONEME_CATEGORY_BY_SYMBOL.get(str(symbol), "other")


def category_index_matrix(vocab: dict[str, Any]) -> np.ndarray:
    id_to_symbol = id_to_symbol_from_vocab(vocab)
    num_classes = int(vocab.get("num_classes", len(id_to_symbol)))
    matrix = np.zeros((num_classes, len(PHONEME_CATEGORY_ORDER)), dtype=np.float32)
    category_to_idx = {category: idx for idx, category in enumerate(PHONEME_CATEGORY_ORDER)}
    for token_id in range(num_classes):
        symbol = id_to_symbol.get(token_id, str(token_id))
        category = category_for_symbol(symbol)
        matrix[token_id, category_to_idx.get(category, category_to_idx["other"])] = 1.0
    return matrix


def category_probability_frame(
    probs: np.ndarray,
    *,
    vocab: dict[str, Any],
) -> pd.DataFrame:
    category_probs = np.asarray(probs, dtype=np.float32) @ category_index_matrix(vocab)
    frame = pd.DataFrame(
        category_probs,
        columns=[f"{category}_prob" for category in PHONEME_CATEGORY_ORDER],
    )
    frame["consonant_prob"] = frame[[f"{category}_prob" for category in CONSONANT_CATEGORIES]].sum(axis=1)
    category_columns = [f"{category}_prob" for category in PHONEME_CATEGORY_ORDER]
    frame["top_category"] = (
        frame[category_columns]
        .idxmax(axis=1)
        .str.replace("_prob", "", regex=False)
    )
    return frame


def add_transition_columns(token_frame: pd.DataFrame) -> pd.DataFrame:
    frame = token_frame.copy()
    frame["prev_top_category"] = frame.groupby("example_id")["top_category"].shift(1)
    frame["next_top_category"] = frame.groupby("example_id")["top_category"].shift(-1)
    frame["category_transition"] = (
        frame["prev_top_category"].notna()
        & (frame["prev_top_category"] != frame["top_category"])
    )
    frame["transition_type"] = "stable_or_first"
    frame.loc[frame["category_transition"], "transition_type"] = (
        frame.loc[frame["category_transition"], "prev_top_category"].astype(str)
        + "_to_"
        + frame.loc[frame["category_transition"], "top_category"].astype(str)
    )
    frame.loc[
        (frame["prev_top_category"] == "vowel") & frame["top_category"].isin(CONSONANT_CATEGORIES),
        "transition_type",
    ] = "vowel_to_consonant"
    frame.loc[
        frame["prev_top_category"].isin(CONSONANT_CATEGORIES) & (frame["top_category"] == "vowel"),
        "transition_type",
    ] = "consonant_to_vowel"
    return frame


def patch_timing_for_token(
    token_index: int,
    *,
    patch_size: int,
    patch_stride: int,
    bin_size_ms: int,
) -> dict[str, int]:
    start_bin = int(token_index) * int(patch_stride)
    end_bin = start_bin + int(patch_size)
    return {
        "patch_start_bin": start_bin,
        "patch_end_bin": end_bin,
        "patch_center_bin": start_bin + int(patch_size) // 2,
        "patch_start_ms": start_bin * int(bin_size_ms),
        "patch_end_ms": end_bin * int(bin_size_ms),
        "patch_center_ms": (start_bin + int(patch_size) // 2) * int(bin_size_ms),
    }


def _build_config_from_checkpoint(checkpoint_payload: dict[str, Any]) -> WillettReconstructionConfig:
    config_payload = dict(checkpoint_payload.get("config", {}))
    if not config_payload:
        raise KeyError("Checkpoint is missing a Willett reconstruction 'config' payload.")
    valid_keys = set(WillettReconstructionConfig.__dataclass_fields__)
    return WillettReconstructionConfig(
        **{
            key: value
            for key, value in config_payload.items()
            if key in valid_keys
        }
    )


def _feature_dim_from_problem(problem: dict[str, Any], feature_mode: str) -> int:
    row = problem["train_rows"][0]
    if str(feature_mode) == "tx_only":
        return int(row.n_tx_features)
    return int(row.n_tx_features + row.n_sbp_features)


def _make_model_from_checkpoint(
    *,
    checkpoint_payload: dict[str, Any],
    config: WillettReconstructionConfig,
    problem: dict[str, Any],
    sample_dim: int,
) -> WillettPhonemeModel:
    train_adapter_keys = adapter_keys_from_rows(
        problem["train_rows"],
        dataset=str(problem["dataset"]),
        boundary_key_mode=str(problem["boundary_key_mode"]),
    )
    val_adapter_keys = adapter_keys_from_rows(
        problem["val_rows"],
        dataset=str(problem["dataset"]),
        boundary_key_mode=str(problem["boundary_key_mode"]),
    )
    session_adapter_keys = tuple(dict.fromkeys(train_adapter_keys + val_adapter_keys))
    model = WillettPhonemeModel(
        input_dim=int(sample_dim),
        vocab_size=int(problem["vocab"]["num_classes"]),
        patch_size=int(config.patch_size),
        patch_stride=int(config.patch_stride),
        input_projection_size=int(config.input_projection_size),
        input_projection_dropout=float(config.input_projection_dropout),
        decoder_backbone_type=str(config.decoder_backbone_type),
        gru_hidden_size=int(config.gru_hidden_size),
        gru_num_layers=int(config.gru_num_layers),
        gru_dropout=float(config.gru_dropout),
        s5_hidden_size=int(config.s5_hidden_size),
        s5_state_size=int(config.s5_state_size),
        s5_num_layers=int(config.s5_num_layers),
        s5_dropout=float(config.s5_dropout),
        s5_direction=str(config.s5_direction),
        s5_ffn_multiplier=float(config.s5_ffn_multiplier),
        s4d_hidden_size=int(config.s4d_hidden_size),
        s4d_state_size=int(config.s4d_state_size),
        s4d_num_layers=int(config.s4d_num_layers),
        s4d_dropout=float(config.s4d_dropout),
        s4d_direction=str(config.s4d_direction),
        s4d_ffn_multiplier=float(config.s4d_ffn_multiplier),
        session_adapter_keys=session_adapter_keys,
        session_adapter_enabled=bool(config.session_adapter_enabled),
    )
    model.load_state_dict(checkpoint_payload["model_state"])
    return model


def _load_train_stats(
    *,
    config: WillettReconstructionConfig,
    problem: dict[str, Any],
    base_sample_dim: int,
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]] | tuple[np.ndarray, np.ndarray] | None, str | None]:
    if str(config.normalization_mode) == "global" and (
        str(config.split_policy) == "competition_train_test"
        or config.precomputed_split_stats_path is not None
    ):
        stats_path = resolve_precomputed_split_stats_path(
            cache_root=Path(config.cache_root),
            dataset=str(config.dataset),
            train_split_name=str(problem["train_split_name"]),
            feature_mode=str(config.feature_mode),
            preferred_path=config.precomputed_split_stats_path,
        )
        if stats_path.exists():
            (mean_t, std_t), _, loaded_path = load_precomputed_split_feature_stats(
                stats_path=stats_path,
                cache_root=Path(problem["cache_root"]),
                dataset=str(problem["dataset"]),
                feature_mode=str(problem["feature_mode"]),
                boundary_key_mode=str(problem["boundary_key_mode"]),
                train_split_name=str(problem["train_split_name"]),
                val_split_name=str(problem["val_split_name"]),
                expected_dim=int(base_sample_dim),
            )
            return (
                mean_t.numpy().astype(np.float32, copy=False),
                std_t.numpy().astype(np.float32, copy=False),
            ), str(loaded_path)
    return compute_willett_normalization_stats(
        problem["train_rows"],
        cache_root=Path(problem["cache_root"]),
        mode=str(config.normalization_mode),
        feature_mode=str(problem["feature_mode"]),
    ), None


def _select_rows(
    problem: dict[str, Any],
    split: str,
    max_examples: int | None,
    allowed_session_ids: tuple[str, ...] | list[str] | None = None,
) -> tuple[Any, ...]:
    if str(split) == "train":
        rows = tuple(problem["train_rows"])
    elif str(split) in {"val", "test", "competition_test"}:
        rows = tuple(problem["val_rows"])
    else:
        raise ValueError("split must be one of {'train', 'val', 'test', 'competition_test'}")
    if allowed_session_ids is not None:
        allowed = {str(session_id) for session_id in allowed_session_ids}
        rows = tuple(row for row in rows if str(row.session_id) in allowed)
    if max_examples is not None:
        rows = rows[: int(max_examples)]
    if not rows:
        raise ValueError("No rows selected for representation export.")
    return rows


def _write_table(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path.with_suffix(".csv"), index=False)
    try:
        frame.to_parquet(path.with_suffix(".parquet"), index=False)
    except Exception:
        pass


def _git_state(repo_dir: str | Path | None) -> dict[str, Any]:
    if repo_dir is None:
        return {}
    root = Path(repo_dir)
    if not root.exists():
        return {"repo_dir": str(root), "exists": False}
    result: dict[str, Any] = {"repo_dir": str(root), "exists": True}
    for key, cmd in {
        "commit": ["git", "rev-parse", "HEAD"],
        "status_short": ["git", "status", "--short"],
    }.items():
        try:
            proc = subprocess.run(cmd, cwd=root, text=True, capture_output=True, check=False, timeout=10)
            result[key] = proc.stdout.strip()
        except Exception as exc:
            result[key] = f"unavailable: {exc}"
    return result


def _flush_shard(
    *,
    shard_dir: Path,
    shard_index: int,
    arrays: dict[str, list[np.ndarray]],
) -> dict[str, Any] | None:
    if not arrays["hidden"]:
        return None
    hidden = np.concatenate(arrays["hidden"], axis=0).astype(np.float32, copy=False)
    logits = np.concatenate(arrays["logits"], axis=0).astype(np.float32, copy=False)
    token_example_index = np.concatenate(arrays["token_example_index"], axis=0).astype(np.int64, copy=False)
    token_index = np.concatenate(arrays["token_index"], axis=0).astype(np.int64, copy=False)
    patch_start_bin = np.concatenate(arrays["patch_start_bin"], axis=0).astype(np.int64, copy=False)
    patch_end_bin = np.concatenate(arrays["patch_end_bin"], axis=0).astype(np.int64, copy=False)
    shard_dir.mkdir(parents=True, exist_ok=True)
    shard_name = f"part-{int(shard_index):05d}.npz"
    shard_path = shard_dir / shard_name
    shard_arrays: dict[str, np.ndarray] = {
        "hidden": hidden,
        "logits": logits,
        "token_example_index": token_example_index,
        "token_index": token_index,
        "patch_start_bin": patch_start_bin,
        "patch_end_bin": patch_end_bin,
    }
    shard_payload: dict[str, Any] = {
        "shard": shard_name,
        "token_count": int(hidden.shape[0]),
        "hidden_dim": int(hidden.shape[1]),
        "vocab_size": int(logits.shape[1]),
    }
    if arrays.get("input_windows"):
        input_windows = np.concatenate(arrays["input_windows"], axis=0).astype(np.float32, copy=False)
        shard_arrays["input_windows"] = input_windows
        shard_payload["input_window_dim"] = int(input_windows.shape[1])
    if arrays.get("adapted_input_windows"):
        adapted_input_windows = np.concatenate(arrays["adapted_input_windows"], axis=0).astype(np.float32, copy=False)
        shard_arrays["adapted_input_windows"] = adapted_input_windows
        shard_payload["adapted_input_window_dim"] = int(adapted_input_windows.shape[1])
    np.savez_compressed(shard_path, **shard_arrays)
    return shard_payload


def export_willett_representations(config: RepresentationExportConfig) -> dict[str, Any]:
    checkpoint_path = Path(config.checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")
    export_dir = Path(config.export_root) / str(config.model_key)
    if export_dir.exists():
        if not bool(config.overwrite):
            metadata_path = export_dir / "metadata.json"
            if metadata_path.exists():
                metadata = json.loads(metadata_path.read_text())
                if bool(config.save_input_windows) and metadata.get("input_window_dim") is None:
                    raise FileExistsError(
                        "Existing representation export does not contain saved input windows. "
                        f"Set overwrite=True to regenerate it: {export_dir}"
                    )
                return metadata
            raise FileExistsError(f"Export directory exists without metadata.json: {export_dir}")
        shutil.rmtree(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_config = _build_config_from_checkpoint(payload)
    if config.cache_root_override is not None or config.precomputed_split_stats_path_override is not None:
        model_config = dataclass_replace(
            model_config,
            **{
                key: value
                for key, value in {
                    "cache_root": config.cache_root_override,
                    "precomputed_split_stats_path": config.precomputed_split_stats_path_override,
                }.items()
                if value is not None
            },
        )
    problem = build_willett_problem(
        cache_root=Path(model_config.cache_root),
        dataset=str(model_config.dataset),
        feature_mode=str(model_config.feature_mode),
        boundary_key_mode=str(model_config.boundary_key_mode),
        split_policy=str(model_config.split_policy),
        cv_num_folds=int(model_config.cv_num_folds),
        cv_fold_index=int(model_config.cv_fold_index),
    )
    base_sample_dim = _feature_dim_from_problem(problem, str(model_config.feature_mode))
    export_accessor = (
        FuturePredictionExportAccessor(model_config.predicted_export_root)
        if str(model_config.input_feature_source) == "raw_plus_predicted_tx"
        else None
    )
    sample_dim = int(base_sample_dim * 2 if export_accessor is not None else base_sample_dim)
    train_stats, loaded_stats_path = _load_train_stats(
        config=model_config,
        problem=problem,
        base_sample_dim=base_sample_dim,
    )
    predicted_stats = (
        compute_predicted_tx_normalization_stats(
            problem["train_rows"],
            export_accessor=export_accessor,
            mode=str(model_config.normalization_mode),
        )
        if export_accessor is not None
        else None
    )
    selected_rows = _select_rows(
        problem,
        config.split,
        config.max_examples,
        config.allowed_session_ids,
    )
    missing_rows = normalization_stats_missing_rows(train_stats, selected_rows)
    if missing_rows:
        raise ValueError(
            "Train-derived normalization stats do not cover selected rows. "
            f"First missing examples: {', '.join(missing_rows[:5])}"
        )
    dataset = (
        CanonicalSequenceDataset(
            selected_rows,
            cache_root=Path(problem["cache_root"]),
            stats=train_stats,
            feature_mode=str(problem["feature_mode"]),
            boundary_key_mode=str(problem["boundary_key_mode"]),
            dataset=str(problem["dataset"]),
        )
        if export_accessor is None
        else ConcatenatedPredictedTxSequenceDataset(
            selected_rows,
            cache_root=Path(problem["cache_root"]),
            raw_stats=train_stats,
            predicted_stats=predicted_stats,
            export_accessor=export_accessor,
            boundary_key_mode=str(problem["boundary_key_mode"]),
            dataset=str(problem["dataset"]),
        )
    )
    device = _detect_device(config.device)
    loader = DataLoader(
        dataset,
        batch_sampler=make_length_aware_batch_sampler(
            selected_rows,
            batch_size=int(config.batch_size),
            shuffle=False,
            seed=int(model_config.seed) + 17,
        ),
        **loader_kwargs(device),
    )
    model = _make_model_from_checkpoint(
        checkpoint_payload=payload,
        config=model_config,
        problem=problem,
        sample_dim=sample_dim,
    ).to(device)
    model.eval()
    transform_config = WillettInputTransformConfig(
        input_smoothing_sigma_bins=float(model_config.input_smoothing_sigma_bins),
        input_smoothing_kernel_size=int(model_config.input_smoothing_kernel_size),
        input_smoothing_threshold=float(model_config.input_smoothing_threshold),
        white_noise_sd=float(model_config.white_noise_sd),
        constant_offset_sd=float(model_config.constant_offset_sd),
    )
    vocab = dict(problem["vocab"])
    id_to_symbol = id_to_symbol_from_vocab(vocab)
    blank_index = int(vocab["blank_index"])

    row_by_example_id = {str(row.example_id): row for row in selected_rows}
    example_index_by_id = {str(row.example_id): idx for idx, row in enumerate(selected_rows)}
    token_rows: list[dict[str, Any]] = []
    example_rows: list[dict[str, Any]] = []
    shard_rows: list[dict[str, Any]] = []
    arrays: dict[str, list[np.ndarray]] = {
        "hidden": [],
        "logits": [],
        "token_example_index": [],
        "token_index": [],
        "patch_start_bin": [],
        "patch_end_bin": [],
    }
    if bool(config.save_input_windows):
        arrays["input_windows"] = []
        arrays["adapted_input_windows"] = []
    shard_index = 0
    buffered_tokens = 0
    total_tokens = 0
    total_examples = 0
    token_global_offset = 0

    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            input_lengths = batch["input_lengths"].to(device)
            x = prepare_willett_inputs(
                x,
                input_lengths,
                config=transform_config,
                is_training=False,
            )
            input_windows = None
            if bool(config.save_input_windows):
                input_windows = model._patch_batch(x, input_lengths)[0].detach().cpu().numpy()
            outputs = model(x, input_lengths, session_ids=batch["boundary_keys"])
            hidden = outputs["hidden"].detach().cpu().numpy()
            logits = outputs["logits"].detach().cpu().numpy()
            adapted_input_windows = (
                outputs["patched_inputs"].detach().cpu().numpy()
                if bool(config.save_input_windows)
                else None
            )
            token_lengths = outputs["token_lengths"].detach().cpu().numpy().astype(np.int64)
            decoded = ctc_greedy_decode(
                outputs["logits"].detach().cpu(),
                outputs["token_lengths"].detach().cpu(),
                blank_index=blank_index,
            )
            labels = batch["labels"].detach().cpu().numpy()
            label_lengths = batch["label_lengths"].detach().cpu().numpy().astype(np.int64)
            probs = F.softmax(outputs["logits"], dim=-1).detach().cpu().numpy()

            for batch_idx, example_id in enumerate(batch["example_ids"]):
                example_id = str(example_id)
                row = row_by_example_id[example_id]
                example_export_index = int(example_index_by_id[example_id])
                length = int(token_lengths[batch_idx])
                if length <= 0:
                    continue
                row_hidden = hidden[batch_idx, :length]
                row_logits = logits[batch_idx, :length]
                row_input_windows = (
                    input_windows[batch_idx, :length]
                    if input_windows is not None
                    else None
                )
                row_adapted_input_windows = (
                    adapted_input_windows[batch_idx, :length]
                    if adapted_input_windows is not None
                    else None
                )
                row_probs = probs[batch_idx, :length]
                row_category_frame = category_probability_frame(row_probs, vocab=vocab)
                top_ids = row_probs.argmax(axis=1).astype(np.int64)
                top_probs = row_probs.max(axis=1)
                entropy = -np.sum(row_probs * np.log(np.maximum(row_probs, 1e-12)), axis=1) / math.log(2.0)
                token_indices = np.arange(length, dtype=np.int64)
                patch_start = token_indices * int(model_config.patch_stride)
                patch_end = patch_start + int(model_config.patch_size)

                arrays["hidden"].append(row_hidden)
                arrays["logits"].append(row_logits)
                if row_input_windows is not None:
                    arrays["input_windows"].append(row_input_windows)
                if row_adapted_input_windows is not None:
                    arrays["adapted_input_windows"].append(row_adapted_input_windows)
                arrays["token_example_index"].append(np.full((length,), example_export_index, dtype=np.int64))
                arrays["token_index"].append(token_indices)
                arrays["patch_start_bin"].append(patch_start.astype(np.int64, copy=False))
                arrays["patch_end_bin"].append(patch_end.astype(np.int64, copy=False))

                for token_idx in range(length):
                    token_payload = {
                        "global_token_index": int(token_global_offset + token_idx),
                        "example_export_index": example_export_index,
                        "example_id": example_id,
                        "source_split": str(getattr(row, "source_split", "")),
                        "session_id": str(row.session_id),
                        "subject_id": "" if getattr(row, "subject_id", None) is None else str(row.subject_id),
                        "boundary_key": str(batch["boundary_keys"][batch_idx]),
                        "token_index": int(token_idx),
                        **patch_timing_for_token(
                            int(token_idx),
                            patch_size=int(model_config.patch_size),
                            patch_stride=int(model_config.patch_stride),
                            bin_size_ms=int(config.bin_size_ms),
                        ),
                        "top1_id": int(top_ids[token_idx]),
                        "top1_symbol": id_to_symbol.get(int(top_ids[token_idx]), str(int(top_ids[token_idx]))),
                        "top1_prob": float(top_probs[token_idx]),
                        "entropy_bits": float(entropy[token_idx]),
                    }
                    for column, value in row_category_frame.iloc[token_idx].to_dict().items():
                        token_payload[str(column)] = value
                    token_rows.append(token_payload)
                reference_ids = labels[batch_idx, : int(label_lengths[batch_idx])].astype(int).tolist()
                prediction_ids = [int(token_id) for token_id in decoded[batch_idx]]
                example_rows.append(
                    {
                        "example_export_index": example_export_index,
                        "example_id": example_id,
                        "source_split": str(getattr(row, "source_split", "")),
                        "session_id": str(row.session_id),
                        "subject_id": "" if getattr(row, "subject_id", None) is None else str(row.subject_id),
                        "boundary_key": str(batch["boundary_keys"][batch_idx]),
                        "input_length_bins": int(input_lengths[batch_idx].detach().cpu().item()),
                        "token_length": int(length),
                        "label_length": int(label_lengths[batch_idx]),
                        "reference_ids": " ".join(str(token_id) for token_id in reference_ids),
                        "reference_symbols": " ".join(id_to_symbol.get(token_id, str(token_id)) for token_id in reference_ids),
                        "prediction_ids": " ".join(str(token_id) for token_id in prediction_ids),
                        "prediction_symbols": " ".join(id_to_symbol.get(token_id, str(token_id)) for token_id in prediction_ids),
                    }
                )
                token_global_offset += length
                buffered_tokens += length
                total_tokens += length
                total_examples += 1

                if buffered_tokens >= int(config.shard_size_tokens):
                    shard_payload = _flush_shard(
                        shard_dir=export_dir / "shards",
                        shard_index=shard_index,
                        arrays=arrays,
                    )
                    if shard_payload is not None:
                        shard_rows.append(shard_payload)
                    shard_index += 1
                    buffered_tokens = 0
                    arrays = {key: [] for key in arrays}

    shard_payload = _flush_shard(
        shard_dir=export_dir / "shards",
        shard_index=shard_index,
        arrays=arrays,
    )
    if shard_payload is not None:
        shard_rows.append(shard_payload)

    token_frame = add_transition_columns(pd.DataFrame(token_rows))
    example_frame = pd.DataFrame(example_rows).sort_values("example_export_index")
    _write_table(token_frame, export_dir / "tokens")
    _write_table(example_frame, export_dir / "examples")
    shard_manifest_path = export_dir / "shards.json"
    shard_manifest_path.write_text(json.dumps(shard_rows, indent=2))
    metadata = {
        "export_kind": "willett_representation_tokens",
        "created_utc": _timestamp_utc(),
        "model_key": str(config.model_key),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_step": int(payload.get("step", payload.get("steps", -1))),
        "dataset": str(problem["dataset"]),
        "feature_mode": str(problem["feature_mode"]),
        "boundary_key_mode": str(problem["boundary_key_mode"]),
        "split_policy": str(problem["split_policy"]),
        "export_split": str(config.split),
        "train_split_name": str(problem["train_split_name"]),
        "val_split_name": str(problem["val_split_name"]),
        "cache_root": str(problem["cache_root"]),
        "precomputed_split_stats_path": loaded_stats_path,
        "input_feature_source": str(model_config.input_feature_source),
        "predicted_export_root": None if model_config.predicted_export_root is None else str(model_config.predicted_export_root),
        "decoder_backbone_type": str(model_config.decoder_backbone_type),
        "patch_size_bins": int(model_config.patch_size),
        "patch_stride_bins": int(model_config.patch_stride),
        "bin_size_ms": int(config.bin_size_ms),
        "patch_size_ms": int(model_config.patch_size) * int(config.bin_size_ms),
        "patch_stride_ms": int(model_config.patch_stride) * int(config.bin_size_ms),
        "example_count": int(total_examples),
        "token_count": int(total_tokens),
        "hidden_dim": int(model.decoder_output_size),
        "input_window_dim": (
            int(sample_dim) * int(model_config.patch_size)
            if bool(config.save_input_windows)
            else None
        ),
        "adapted_input_window_dim": (
            int(model.adapter_output_dim) * int(model_config.patch_size)
            if bool(config.save_input_windows)
            else None
        ),
        "vocab": _json_safe(vocab),
        "category_order": list(PHONEME_CATEGORY_ORDER),
        "consonant_categories": sorted(CONSONANT_CATEGORIES),
        "config": _json_safe(asdict(model_config)),
        "representation_export_config": _json_safe(asdict(config)),
        "shards": shard_rows,
        "git": _git_state(config.repo_dir),
        "token_table_csv": str(export_dir / "tokens.csv"),
        "example_table_csv": str(export_dir / "examples.csv"),
        "shard_manifest_path": str(shard_manifest_path),
    }
    metadata_path = export_dir / "metadata.json"
    metadata_path.write_text(json.dumps(_json_safe(metadata), indent=2))
    return metadata


__all__ = [
    "CONSONANT_CATEGORIES",
    "PHONEME_CATEGORY_BY_SYMBOL",
    "PHONEME_CATEGORY_ORDER",
    "RepresentationExportConfig",
    "add_transition_columns",
    "category_for_symbol",
    "category_index_matrix",
    "category_probability_frame",
    "export_willett_representations",
    "id_to_symbol_from_vocab",
    "patch_timing_for_token",
]

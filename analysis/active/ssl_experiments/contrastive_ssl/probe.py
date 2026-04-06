"""Held-out phoneme probe helpers for contrastive SSL notebook experiments."""

from __future__ import annotations

import copy
import importlib
import json
import math
import time
from collections import Counter
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .model import ContrastiveSSLModel, S5ContrastiveEncoder


DEFAULT_PROBE_SUMMARY_BASENAME = "downstream_probe_summary.json"


@dataclass
class DownstreamProbeConfig:
    enabled: bool = True
    seed: int = 7
    comparison_mode: str = "ssl_only"
    session_limit: int = 4
    target_session_count: int = 1
    adaptation_regime: str = "A"
    probe_batch_size: int = 8
    probe_budget_seconds: int = 240
    max_probe_steps: int = 400
    progress_every_steps: int = 25
    progress_every_seconds: float = 15.0
    probe_head_type: str = "linear"
    probe_lstm_hidden_size: int = 64
    probe_conv_hidden_size: int = 128
    probe_conv_kernel_size: int = 3
    checkpoint_source: str = "most_recent_valid_then_in_memory"
    summary_basename: str = DEFAULT_PROBE_SUMMARY_BASENAME

    def __post_init__(self) -> None:
        if self.comparison_mode != "ssl_only":
            raise ValueError("comparison_mode must currently be 'ssl_only'")
        if self.adaptation_regime != "A":
            raise ValueError("adaptation_regime must currently be 'A'")
        if self.target_session_count != 1:
            raise ValueError("target_session_count must currently be 1")
        if self.probe_head_type not in {"linear", "lstm", "conv1d"}:
            raise ValueError("probe_head_type must be one of {'linear', 'lstm', 'conv1d'}")
        if self.checkpoint_source not in {
            "most_recent_valid_then_in_memory",
            "in_memory_then_most_recent_valid",
        }:
            raise ValueError("checkpoint_source must be a supported selection policy")


class NotebookProbeEncoderAdapter(nn.Module):
    def __init__(self, encoder: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.input_dim = int(getattr(encoder, "input_dim"))
        self.hidden_size = int(getattr(encoder, "hidden_size"))

    def encode(
        self,
        x: torch.Tensor,
        input_lengths: torch.Tensor,
        session_ids: list[str],
        *,
        use_source_affines: bool,
        target_affines=None,
    ) -> SimpleNamespace:
        del use_source_affines
        if target_affines is not None:
            x = target_affines(x, session_ids)
        outputs = self.encoder(x, input_lengths)
        return SimpleNamespace(
            hidden=outputs["hidden"],
            token_lengths=outputs["token_lengths"],
            tokens=outputs["tokens"],
        )


class LinearCTCProbe(nn.Module):
    def __init__(self, hidden_size: int, vocab_size: int):
        super().__init__()
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.linear(hidden)


class TinyLSTMCTCProbe(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, vocab_size: int):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )
        self.classifier = nn.Linear(hidden_size, vocab_size)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        outputs, _ = self.lstm(hidden)
        return self.classifier(outputs)


class CausalConvCTCProbe(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, vocab_size: int, kernel_size: int):
        super().__init__()
        self.kernel_size = int(kernel_size)
        self.conv = nn.Conv1d(
            in_channels=input_size,
            out_channels=hidden_size,
            kernel_size=self.kernel_size,
        )
        self.activation = nn.GELU()
        self.classifier = nn.Linear(hidden_size, vocab_size)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        x = hidden.transpose(1, 2)
        x = F.pad(x, (self.kernel_size - 1, 0))
        x = self.conv(x)
        x = self.activation(x)
        x = x.transpose(1, 2)
        return self.classifier(x)


def _load_benchmark_modules() -> tuple[Any, Any]:
    prepare_module = importlib.import_module("prepare")
    prepare_module = importlib.reload(prepare_module)
    data_module = importlib.import_module("data")
    data_module = importlib.reload(data_module)
    return prepare_module, data_module


def _canonical_probe_paths(cache_root: Path) -> tuple[Path, Path, Path]:
    canonical_root = Path(cache_root) / "brain2text25"
    return canonical_root, canonical_root / "manifest.jsonl", canonical_root / "metadata.json"


def _validate_canonical_probe_assets(cache_root: Path) -> tuple[Path, Path, Path]:
    canonical_root, manifest_path, metadata_path = _canonical_probe_paths(cache_root)
    if not manifest_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(
            "Canonical Brain2Text25 cache manifest / metadata is missing from the mounted cache. "
            f"Expected {manifest_path} and {metadata_path}."
        )
    return canonical_root, manifest_path, metadata_path


def _default_checkpoint_config(default_checkpoint_config: dict[str, Any] | None) -> dict[str, Any]:
    if default_checkpoint_config is None:
        raise ValueError("default_checkpoint_config is required for downstream probe recovery.")
    return {
        "patch_size": int(default_checkpoint_config["patch_size"]),
        "patch_stride": int(default_checkpoint_config["patch_stride"]),
        "hidden_size": int(default_checkpoint_config["hidden_size"]),
        "s5_state_size": int(default_checkpoint_config["s5_state_size"]),
        "num_layers": int(default_checkpoint_config["num_layers"]),
        "dropout": float(default_checkpoint_config["dropout"]),
        "post_proj_norm": str(default_checkpoint_config.get("post_proj_norm", "rms")),
    }


def _resolve_candidate_checkpoint_path(
    *,
    output_root: Path,
    probe_config: DownstreamProbeConfig,
    current_checkpoint_path: Path | None,
) -> Path | None:
    del probe_config

    valid_candidates = sorted(
        [
            candidate
            for candidate in Path(output_root).glob("colab_s5_*/checkpoint_final.pt")
            if candidate.exists() and (candidate.parent / "config.json").exists()
        ],
        key=lambda candidate: candidate.stat().st_mtime,
    )
    if valid_candidates:
        return valid_candidates[-1]

    if current_checkpoint_path is not None:
        candidate = Path(current_checkpoint_path)
        if candidate.exists():
            return candidate

    return None


def _recover_encoder_from_notebook_checkpoint(
    *,
    path: Path,
    input_dim: int,
) -> tuple[nn.Module, dict[str, Any]]:
    payload = torch.load(path, map_location="cpu")
    checkpoint_cfg = dict(payload.get("config", {}))
    required_keys = [
        "patch_size",
        "patch_stride",
        "hidden_size",
        "s5_state_size",
        "num_layers",
        "dropout",
    ]
    missing_keys = [key for key in required_keys if key not in checkpoint_cfg]
    if missing_keys:
        raise KeyError(
            f"Checkpoint config is missing keys needed for encoder recovery: {missing_keys}"
        )

    recovered_encoder = S5ContrastiveEncoder(
        input_dim=input_dim,
        hidden_size=int(checkpoint_cfg["hidden_size"]),
        s5_state_size=int(checkpoint_cfg["s5_state_size"]),
        num_layers=int(checkpoint_cfg["num_layers"]),
        dropout=float(checkpoint_cfg["dropout"]),
        patch_size=int(checkpoint_cfg["patch_size"]),
        patch_stride=int(checkpoint_cfg["patch_stride"]),
        post_proj_norm=str(checkpoint_cfg.get("post_proj_norm", "rms")),
    )
    model_state = payload.get("model_state")
    if model_state is None:
        raise KeyError("Notebook checkpoint is missing 'model_state'.")

    encoder_state = {
        key.split("encoder.", 1)[1]: value
        for key, value in model_state.items()
        if key.startswith("encoder.")
    }
    if not encoder_state:
        raise KeyError("Notebook checkpoint does not contain encoder weights.")
    recovered_encoder.load_state_dict(encoder_state)
    return recovered_encoder, checkpoint_cfg


def _resolve_downstream_probe_base_run_dir(
    *,
    output_root: Path,
    checkpoint_candidate: Path | None,
    current_run_dir: Path | None,
) -> Path:
    if checkpoint_candidate is not None:
        candidate = Path(checkpoint_candidate)
        if candidate.exists():
            return candidate.parent
    if current_run_dir is not None:
        candidate = Path(current_run_dir)
        if candidate.exists():
            return candidate
    fallback = Path(output_root) / (
        f"colab_s5_downstream_probe_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    )
    fallback.mkdir(parents=True, exist_ok=True)
    return fallback


def _build_probe_state(
    *,
    base_encoder: nn.Module,
    source: str,
    checkpoint_config: dict[str, Any],
    checkpoint_path: Path | None,
    base_run_dir: Path,
) -> dict[str, Any]:
    base_encoder = base_encoder.cpu()
    return {
        "base_encoder": base_encoder,
        "encoder": NotebookProbeEncoderAdapter(base_encoder),
        "source": source,
        "checkpoint_config": dict(checkpoint_config),
        "checkpoint_path": checkpoint_path,
        "base_run_dir": Path(base_run_dir),
    }


def recover_downstream_probe_state(
    *,
    probe_config: DownstreamProbeConfig,
    output_root: Path,
    input_dim: int,
    default_checkpoint_config: dict[str, Any],
    in_memory_model: ContrastiveSSLModel | None = None,
    current_checkpoint_path: Path | None = None,
    current_run_dir: Path | None = None,
) -> dict[str, Any]:
    default_cfg = _default_checkpoint_config(default_checkpoint_config)

    def load_checkpoint_state() -> dict[str, Any] | None:
        checkpoint_candidate = _resolve_candidate_checkpoint_path(
            output_root=output_root,
            probe_config=probe_config,
            current_checkpoint_path=current_checkpoint_path,
        )
        if checkpoint_candidate is None:
            return None
        base_encoder, checkpoint_cfg = _recover_encoder_from_notebook_checkpoint(
            path=checkpoint_candidate,
            input_dim=input_dim,
        )
        return _build_probe_state(
            base_encoder=base_encoder,
            source="checkpoint",
            checkpoint_config=checkpoint_cfg,
            checkpoint_path=checkpoint_candidate,
            base_run_dir=_resolve_downstream_probe_base_run_dir(
                output_root=output_root,
                checkpoint_candidate=checkpoint_candidate,
                current_run_dir=current_run_dir,
            ),
        )

    def load_in_memory_state() -> dict[str, Any] | None:
        if in_memory_model is None or not hasattr(in_memory_model, "encoder"):
            return None
        checkpoint_candidate = _resolve_candidate_checkpoint_path(
            output_root=output_root,
            probe_config=probe_config,
            current_checkpoint_path=current_checkpoint_path,
        )
        return _build_probe_state(
            base_encoder=copy.deepcopy(in_memory_model.encoder),
            source="in_memory",
            checkpoint_config=default_cfg,
            checkpoint_path=None,
            base_run_dir=_resolve_downstream_probe_base_run_dir(
                output_root=output_root,
                checkpoint_candidate=checkpoint_candidate,
                current_run_dir=current_run_dir,
            ),
        )

    state_loaders = (
        (load_in_memory_state, load_checkpoint_state)
        if probe_config.checkpoint_source == "in_memory_then_most_recent_valid"
        else (load_checkpoint_state, load_in_memory_state)
    )
    for loader in state_loaders:
        state = loader()
        if state is not None:
            return state

    raise RuntimeError(
        "No in-memory encoder is available and no checkpoint_final.pt was found under OUTPUT_ROOT. "
        "Run the training cell first or make the checkpoint available under the notebook output root."
    )


def build_random_init_probe_state(
    *,
    reference_config: dict[str, Any],
    input_dim: int,
    seed: int,
    base_run_dir: Path,
) -> dict[str, Any]:
    checkpoint_cfg = _default_checkpoint_config(reference_config)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        base_encoder = S5ContrastiveEncoder(
            input_dim=input_dim,
            hidden_size=int(checkpoint_cfg["hidden_size"]),
            s5_state_size=int(checkpoint_cfg["s5_state_size"]),
            num_layers=int(checkpoint_cfg["num_layers"]),
            dropout=float(checkpoint_cfg["dropout"]),
            patch_size=int(checkpoint_cfg["patch_size"]),
            patch_stride=int(checkpoint_cfg["patch_stride"]),
            post_proj_norm=str(checkpoint_cfg.get("post_proj_norm", "rms")),
        )

    return _build_probe_state(
        base_encoder=base_encoder,
        source="random_init",
        checkpoint_config=checkpoint_cfg,
        checkpoint_path=None,
        base_run_dir=Path(base_run_dir),
    )


def build_downstream_probe_problem(
    *,
    cache_root: Path,
    probe_config: DownstreamProbeConfig,
) -> dict[str, Any]:
    _, data_module = _load_benchmark_modules()
    canonical_root, manifest_path, metadata_path = _validate_canonical_probe_assets(cache_root)

    inventory = data_module.load_b2t25_canonical_inventory(canonical_root)
    eligible_entries = [entry for entry in inventory if entry.has_tx and entry.has_sbp]
    split = data_module.split_latest_sessions(
        eligible_entries,
        session_limit=int(probe_config.session_limit),
        val_session_count=int(probe_config.target_session_count),
    )
    source_session_ids, target_session_ids = data_module.session_ids_from_cache_split(split)
    if len(target_session_ids) != 1:
        raise ValueError(
            "The benchmark-lite notebook probe expects exactly one held-out target session. "
            f"Found {len(target_session_ids)} target sessions."
        )

    manifest_rows = data_module.load_probe_manifest(manifest_path)
    metadata = data_module.load_probe_metadata(metadata_path)
    partitions = data_module.partition_probe_records(
        manifest_rows,
        source_session_ids=source_session_ids,
        target_session_ids=target_session_ids,
    )

    target_session_id = target_session_ids[0]
    target_train_rows = partitions.target_train_by_session[target_session_id]
    target_val_rows = partitions.target_val_by_session[target_session_id]
    if len(target_train_rows) == 0 or len(target_val_rows) == 0:
        raise ValueError(
            "The held-out target session does not have both train and val examples with phoneme labels. "
            f"train={len(target_train_rows)} val={len(target_val_rows)}"
        )

    return {
        "canonical_root": canonical_root,
        "manifest_path": manifest_path,
        "metadata_path": metadata_path,
        "inventory": inventory,
        "eligible_entries": eligible_entries,
        "split": split,
        "manifest_rows": manifest_rows,
        "metadata": metadata,
        "partitions": partitions,
        "source_session_ids": source_session_ids,
        "target_session_ids": target_session_ids,
        "target_session_id": target_session_id,
        "target_train_rows": target_train_rows,
        "target_val_rows": target_val_rows,
        "vocab": metadata["phoneme_vocabulary"],
        "data_module": data_module,
    }


def _loader_kwargs(device: torch.device, batch_size: int, *, shuffle: bool, collate_fn: Any) -> dict[str, Any]:
    return {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": 0,
        "pin_memory": device.type == "cuda",
        "collate_fn": collate_fn,
    }


def _make_loader_generator(seed: int) -> torch.Generator:
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return generator


def _flatten_targets(labels: torch.Tensor, label_lengths: torch.Tensor) -> torch.Tensor:
    pieces = []
    for row_idx, length in enumerate(label_lengths.tolist()):
        if length > 0:
            pieces.append(labels[row_idx, :length])
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
    targets = _flatten_targets(labels, label_lengths)
    loss_sum = F.ctc_loss(
        log_probs,
        targets,
        token_lengths,
        label_lengths,
        blank=blank_index,
        reduction="sum",
        zero_infinity=True,
    )
    return loss_sum, target_count


def _ctc_greedy_decode(
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
        for token in token_ids[batch_idx, :length].tolist():
            if token == blank_index:
                prev_token = None
                continue
            if token != prev_token:
                sequence.append(int(token))
            prev_token = int(token)
        decoded.append(sequence)
    return decoded


def _edit_distance(reference: list[int], hypothesis: list[int]) -> int:
    if not reference:
        return len(hypothesis)
    if not hypothesis:
        return len(reference)
    previous = list(range(len(hypothesis) + 1))
    for ref_idx, ref_token in enumerate(reference, start=1):
        current = [ref_idx]
        for hyp_idx, hyp_token in enumerate(hypothesis, start=1):
            substitution_cost = 0 if ref_token == hyp_token else 1
            current.append(
                min(
                    previous[hyp_idx] + 1,
                    current[hyp_idx - 1] + 1,
                    previous[hyp_idx - 1] + substitution_cost,
                )
            )
        previous = current
    return previous[-1]


def _align_sequences(
    reference: list[int],
    hypothesis: list[int],
) -> list[tuple[str, int | None, int | None]]:
    num_ref = len(reference)
    num_hyp = len(hypothesis)
    dp = [[0] * (num_hyp + 1) for _ in range(num_ref + 1)]
    back: list[list[tuple[str, int | None, int | None] | None]] = [
        [None] * (num_hyp + 1) for _ in range(num_ref + 1)
    ]

    for ref_idx in range(1, num_ref + 1):
        dp[ref_idx][0] = ref_idx
        back[ref_idx][0] = ("deletion", int(reference[ref_idx - 1]), None)
    for hyp_idx in range(1, num_hyp + 1):
        dp[0][hyp_idx] = hyp_idx
        back[0][hyp_idx] = ("insertion", None, int(hypothesis[hyp_idx - 1]))

    for ref_idx in range(1, num_ref + 1):
        ref_token = int(reference[ref_idx - 1])
        for hyp_idx in range(1, num_hyp + 1):
            hyp_token = int(hypothesis[hyp_idx - 1])
            candidates = []
            if ref_token == hyp_token:
                candidates.append((dp[ref_idx - 1][hyp_idx - 1], 0, "correct", ref_token, hyp_token))
            else:
                candidates.append((dp[ref_idx - 1][hyp_idx - 1] + 1, 0, "substitution", ref_token, hyp_token))
            candidates.append((dp[ref_idx - 1][hyp_idx] + 1, 1, "deletion", ref_token, None))
            candidates.append((dp[ref_idx][hyp_idx - 1] + 1, 2, "insertion", None, hyp_token))
            best_cost, _, op_name, ref_value, hyp_value = min(candidates, key=lambda item: (item[0], item[1]))
            dp[ref_idx][hyp_idx] = best_cost
            back[ref_idx][hyp_idx] = (op_name, ref_value, hyp_value)

    aligned: list[tuple[str, int | None, int | None]] = []
    ref_idx = num_ref
    hyp_idx = num_hyp
    while ref_idx > 0 or hyp_idx > 0:
        step = back[ref_idx][hyp_idx]
        if step is None:
            break
        op_name, ref_value, hyp_value = step
        aligned.append((op_name, ref_value, hyp_value))
        if op_name in {"correct", "substitution"}:
            ref_idx -= 1
            hyp_idx -= 1
        elif op_name == "deletion":
            ref_idx -= 1
        elif op_name == "insertion":
            hyp_idx -= 1
        else:
            raise ValueError(f"Unexpected alignment op: {op_name}")

    aligned.reverse()
    return aligned


def _top_counter_items(counter: Counter[int], *, top_k: int = 10) -> list[list[int]]:
    return [[int(item), int(count)] for item, count in counter.most_common(top_k)]


def _top_pair_items(counter: Counter[tuple[int, int]], *, top_k: int = 10) -> list[dict[str, int]]:
    return [
        {
            "reference_id": int(reference_id),
            "predicted_id": int(predicted_id),
            "count": int(count),
        }
        for (reference_id, predicted_id), count in counter.most_common(top_k)
    ]


def _emit_progress(progress_log_path: Path | None, **payload: Any) -> None:
    if progress_log_path is None:
        return
    progress_log_path.parent.mkdir(parents=True, exist_ok=True)
    with progress_log_path.open("a") as handle:
        handle.write(json.dumps(payload) + "\n")


def evaluate_probe_session_metrics(
    *,
    encoder: nn.Module,
    probe_head: nn.Module,
    loader: DataLoader,
    device: torch.device,
    blank_index: int,
) -> dict[str, Any]:
    encoder.eval()
    probe_head.eval()

    total_loss_sum = 0.0
    total_targets = 0
    total_edit_distance = 0
    total_reference_tokens = 0
    total_predicted_tokens = 0

    reference_counter: Counter[int] = Counter()
    prediction_counter: Counter[int] = Counter()
    insertion_counter: Counter[int] = Counter()
    deletion_counter: Counter[int] = Counter()
    substitution_counter: Counter[tuple[int, int]] = Counter()
    error_prediction_counter: Counter[int] = Counter()

    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            input_lengths = batch["input_lengths"].to(device)
            labels = batch["labels"].to(device)
            label_lengths = batch["label_lengths"].to(device)

            outputs = encoder.encode(
                x,
                input_lengths,
                batch["session_ids"],
                use_source_affines=False,
                target_affines=None,
            )
            logits = probe_head(outputs.hidden)
            loss_sum, target_count = compute_ctc_loss_sum(
                logits,
                outputs.token_lengths,
                labels,
                label_lengths,
                blank_index=blank_index,
            )
            total_loss_sum += float(loss_sum.item())
            total_targets += target_count

            predictions = _ctc_greedy_decode(
                logits,
                outputs.token_lengths,
                blank_index=blank_index,
            )
            for row_idx, prediction in enumerate(predictions):
                reference_length = int(label_lengths[row_idx].item())
                reference = labels[row_idx, :reference_length].tolist()
                total_edit_distance += _edit_distance(reference, prediction)
                total_reference_tokens += len(reference)
                total_predicted_tokens += len(prediction)
                reference_counter.update(int(token) for token in reference)
                prediction_counter.update(int(token) for token in prediction)

                for op_name, ref_token, hyp_token in _align_sequences(reference, prediction):
                    if op_name == "substitution":
                        assert ref_token is not None and hyp_token is not None
                        substitution_counter[(int(ref_token), int(hyp_token))] += 1
                        error_prediction_counter[int(hyp_token)] += 1
                    elif op_name == "insertion":
                        assert hyp_token is not None
                        insertion_counter[int(hyp_token)] += 1
                        error_prediction_counter[int(hyp_token)] += 1
                    elif op_name == "deletion":
                        assert ref_token is not None
                        deletion_counter[int(ref_token)] += 1

    if total_targets <= 0:
        raise ValueError("Validation target count is zero; cannot compute val_bpphone.")
    if total_reference_tokens <= 0:
        raise ValueError("Validation reference token count is zero; cannot compute phoneme error rate.")

    total_insertions = int(sum(insertion_counter.values()))
    total_deletions = int(sum(deletion_counter.values()))
    total_substitutions = int(sum(substitution_counter.values()))
    total_wrong_predictions = int(sum(error_prediction_counter.values()))

    val_ctc_bpphone = total_loss_sum / total_targets / math.log(2.0)
    val_phoneme_error_rate = total_edit_distance / total_reference_tokens
    return {
        "val_ctc_bpphone": float(val_ctc_bpphone),
        "val_phoneme_error_rate": float(val_phoneme_error_rate),
        "alignment_diagnostics": {
            "total_reference_tokens": int(total_reference_tokens),
            "total_predicted_tokens": int(total_predicted_tokens),
            "total_edit_distance": int(total_edit_distance),
            "total_insertions": total_insertions,
            "total_deletions": total_deletions,
            "total_substitutions": total_substitutions,
            "total_wrong_predictions": total_wrong_predictions,
            "reference_top_ids": _top_counter_items(reference_counter),
            "prediction_top_ids": _top_counter_items(prediction_counter),
            "insertion_top_ids": _top_counter_items(insertion_counter),
            "deletion_top_ids": _top_counter_items(deletion_counter),
            "false_prediction_top_ids": _top_counter_items(error_prediction_counter),
            "substitution_top_pairs": _top_pair_items(substitution_counter),
        },
    }


def _build_probe_run_config(
    probe_config: DownstreamProbeConfig,
    probe_overrides: dict[str, Any] | None,
) -> DownstreamProbeConfig:
    return replace(probe_config, **(probe_overrides or {}))


def resolve_probe_head_type(config: DownstreamProbeConfig) -> str:
    return str(config.probe_head_type)


def probe_head_suffix(probe_head_type: str) -> str:
    return {
        "linear": "linear_probe",
        "lstm": "lstm_probe",
        "conv1d": "conv1d_probe",
    }[probe_head_type]


def build_probe_head(
    *,
    encoder_hidden_size: int,
    probe_vocab_size: int,
    probe_config: DownstreamProbeConfig,
) -> nn.Module:
    probe_head_type = resolve_probe_head_type(probe_config)
    if probe_head_type == "linear":
        return LinearCTCProbe(encoder_hidden_size, probe_vocab_size)
    if probe_head_type == "lstm":
        return TinyLSTMCTCProbe(
            input_size=encoder_hidden_size,
            hidden_size=int(probe_config.probe_lstm_hidden_size),
            vocab_size=probe_vocab_size,
        )
    return CausalConvCTCProbe(
        input_size=encoder_hidden_size,
        hidden_size=int(probe_config.probe_conv_hidden_size),
        vocab_size=probe_vocab_size,
        kernel_size=int(probe_config.probe_conv_kernel_size),
    )


def count_trainable_parameters(module: nn.Module) -> int:
    return int(sum(param.numel() for param in module.parameters() if param.requires_grad))


def train_probe_with_metrics(
    *,
    problem: dict[str, Any],
    pretrained_encoder: nn.Module,
    probe_config: DownstreamProbeConfig,
    device: torch.device,
    progress_log_path: Path | None,
    train_encoder: bool,
) -> tuple[dict[str, Any], int]:
    data_module = problem["data_module"]
    target_stats = data_module.compute_feature_stats(problem["target_train_rows"], mode="global")
    train_loader = DataLoader(
        data_module.CanonicalSequenceDataset(problem["target_train_rows"], stats=target_stats),
        **_loader_kwargs(
            device,
            int(probe_config.probe_batch_size),
            shuffle=True,
            collate_fn=data_module.collate_sequence_batch,
        ),
        generator=_make_loader_generator(int(probe_config.seed)),
    )
    val_loader = DataLoader(
        data_module.CanonicalSequenceDataset(problem["target_val_rows"], stats=target_stats),
        **_loader_kwargs(
            device,
            int(probe_config.probe_batch_size),
            shuffle=False,
            collate_fn=data_module.collate_sequence_batch,
        ),
        generator=_make_loader_generator(int(probe_config.seed) + 1),
    )

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(probe_config.seed))
        encoder = copy.deepcopy(pretrained_encoder).to(device)
    for parameter in encoder.parameters():
        parameter.requires_grad = bool(train_encoder)
    if train_encoder:
        encoder.train()
    else:
        encoder.eval()

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(probe_config.seed) + 2)
        probe_head = build_probe_head(
            encoder_hidden_size=encoder.hidden_size,
            probe_vocab_size=int(problem["vocab"]["num_classes"]),
            probe_config=probe_config,
        ).to(device)
    probe_head_num_parameters = count_trainable_parameters(probe_head)
    trainable_parameters = [param for param in probe_head.parameters()]
    if train_encoder:
        trainable_parameters.extend(param for param in encoder.parameters() if param.requires_grad)

    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=1e-3,
        weight_decay=1e-2,
    )

    start_time = time.time()
    last_report_elapsed = 0.0
    steps = 0
    while True:
        elapsed = time.time() - start_time
        if elapsed >= float(probe_config.probe_budget_seconds):
            break
        if probe_config.max_probe_steps is not None and steps >= int(probe_config.max_probe_steps):
            break

        made_progress = False
        for batch in train_loader:
            elapsed = time.time() - start_time
            if elapsed >= float(probe_config.probe_budget_seconds):
                break
            if probe_config.max_probe_steps is not None and steps >= int(probe_config.max_probe_steps):
                break

            if train_encoder:
                encoder.train()
            else:
                encoder.eval()
            probe_head.train()

            x = batch["x"].to(device)
            input_lengths = batch["input_lengths"].to(device)
            labels = batch["labels"].to(device)
            label_lengths = batch["label_lengths"].to(device)

            if train_encoder:
                outputs = encoder.encode(
                    x,
                    input_lengths,
                    batch["session_ids"],
                    use_source_affines=False,
                    target_affines=None,
                )
            else:
                with torch.no_grad():
                    outputs = encoder.encode(
                        x,
                        input_lengths,
                        batch["session_ids"],
                        use_source_affines=False,
                        target_affines=None,
                    )
            logits = probe_head(outputs.hidden)
            loss_sum, target_count = compute_ctc_loss_sum(
                logits,
                outputs.token_lengths,
                labels,
                label_lengths,
                blank_index=int(problem["vocab"]["blank_index"]),
            )
            if target_count <= 0:
                continue

            loss = loss_sum / target_count
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_parameters, max_norm=1.0)
            optimizer.step()
            steps += 1
            made_progress = True

            elapsed = time.time() - start_time
            should_report = (
                steps == 1
                or steps % int(probe_config.progress_every_steps) == 0
                or elapsed - last_report_elapsed >= float(probe_config.progress_every_seconds)
            )
            if should_report:
                last_report_elapsed = elapsed
                _emit_progress(
                    progress_log_path,
                    event="probe_train_report",
                    stage="probe_train",
                    session_id=problem["target_session_id"],
                    step=steps,
                    elapsed_seconds=round(elapsed, 3),
                    train_ctc_bpphone=float(loss.item()) / math.log(2.0),
                    train_encoder=bool(train_encoder),
                    seed=int(probe_config.seed),
                    probe_head_type=resolve_probe_head_type(probe_config),
                    budget_seconds=float(probe_config.probe_budget_seconds),
                )
        if not made_progress:
            break

    metrics = evaluate_probe_session_metrics(
        encoder=encoder,
        probe_head=probe_head,
        loader=val_loader,
        device=device,
        blank_index=int(problem["vocab"]["blank_index"]),
    )
    metrics["probe_head_num_parameters"] = int(probe_head_num_parameters)
    _emit_progress(
        progress_log_path,
        event="probe_session_complete",
        stage="probe_train",
        session_id=problem["target_session_id"],
        step=steps,
        elapsed_seconds=round(time.time() - start_time, 3),
        train_encoder=bool(train_encoder),
        seed=int(probe_config.seed),
        probe_head_type=resolve_probe_head_type(probe_config),
        **metrics,
    )
    return metrics, steps


def run_downstream_probe(
    *,
    probe_state: dict[str, Any],
    probe_config: DownstreamProbeConfig,
    cache_root: Path,
    device: torch.device,
    variant_prefix: str,
    artifact_prefix: str,
    train_encoder: bool,
    probe_overrides: dict[str, Any] | None = None,
    comparison_mode: str | None = None,
) -> dict[str, Any]:
    effective_probe_config = _build_probe_run_config(probe_config, probe_overrides)
    problem = build_downstream_probe_problem(
        cache_root=cache_root,
        probe_config=effective_probe_config,
    )

    suffix = probe_head_suffix(resolve_probe_head_type(effective_probe_config))
    model_variant = f"{variant_prefix}_{suffix}"
    artifact_name = f"{artifact_prefix}_{suffix}"
    artifact_dir = Path(probe_state["base_run_dir"]) / artifact_name
    artifact_dir.mkdir(parents=True, exist_ok=True)

    progress_path = artifact_dir / "progress.jsonl"
    if progress_path.exists():
        progress_path.unlink()

    start_time = time.time()
    metrics, steps = train_probe_with_metrics(
        problem=problem,
        pretrained_encoder=probe_state["encoder"],
        probe_config=effective_probe_config,
        device=device,
        progress_log_path=progress_path,
        train_encoder=train_encoder,
    )
    elapsed_seconds = time.time() - start_time

    index_to_symbol = list(problem["vocab"].get("index_to_symbol", []))

    def symbol_for(token_id: int | None) -> str | None:
        if token_id is None:
            return None
        if 0 <= int(token_id) < len(index_to_symbol):
            return str(index_to_symbol[int(token_id)])
        return str(token_id)

    alignment = dict(metrics.get("alignment_diagnostics", {}))

    def format_ranked_ids(items: list[list[int]], *, denominator: int) -> list[dict[str, Any]]:
        formatted = []
        for token_id, count in items:
            formatted.append(
                {
                    "id": int(token_id),
                    "symbol": symbol_for(int(token_id)),
                    "count": int(count),
                    "rate": float(count / denominator) if denominator > 0 else None,
                }
            )
        return formatted

    def format_ranked_pairs(items: list[dict[str, int]], *, denominator: int) -> list[dict[str, Any]]:
        formatted = []
        for item in items:
            count = int(item["count"])
            reference_id = int(item["reference_id"])
            predicted_id = int(item["predicted_id"])
            formatted.append(
                {
                    "reference_id": reference_id,
                    "reference_symbol": symbol_for(reference_id),
                    "predicted_id": predicted_id,
                    "predicted_symbol": symbol_for(predicted_id),
                    "count": count,
                    "rate": float(count / denominator) if denominator > 0 else None,
                }
            )
        return formatted

    prediction_histogram = format_ranked_ids(
        list(alignment.get("prediction_top_ids", [])),
        denominator=int(alignment.get("total_predicted_tokens", 0)),
    )
    reference_histogram = format_ranked_ids(
        list(alignment.get("reference_top_ids", [])),
        denominator=int(alignment.get("total_reference_tokens", 0)),
    )
    insertion_histogram = format_ranked_ids(
        list(alignment.get("insertion_top_ids", [])),
        denominator=int(alignment.get("total_insertions", 0)),
    )
    deletion_histogram = format_ranked_ids(
        list(alignment.get("deletion_top_ids", [])),
        denominator=int(alignment.get("total_deletions", 0)),
    )
    false_prediction_histogram = format_ranked_ids(
        list(alignment.get("false_prediction_top_ids", [])),
        denominator=int(alignment.get("total_wrong_predictions", 0)),
    )
    substitution_pairs = format_ranked_pairs(
        list(alignment.get("substitution_top_pairs", [])),
        denominator=int(alignment.get("total_substitutions", 0)),
    )

    alignment_stats = {
        "total_reference_tokens": int(alignment.get("total_reference_tokens", 0)),
        "total_predicted_tokens": int(alignment.get("total_predicted_tokens", 0)),
        "total_edit_distance": int(alignment.get("total_edit_distance", 0)),
        "total_insertions": int(alignment.get("total_insertions", 0)),
        "total_deletions": int(alignment.get("total_deletions", 0)),
        "total_substitutions": int(alignment.get("total_substitutions", 0)),
        "prediction_histogram_top": prediction_histogram,
        "reference_histogram_top": reference_histogram,
        "false_prediction_histogram_top": false_prediction_histogram,
        "insertion_histogram_top": insertion_histogram,
        "deletion_histogram_top": deletion_histogram,
        "substitution_pairs_top": substitution_pairs,
        "most_common_prediction": prediction_histogram[0] if prediction_histogram else None,
        "most_common_false_prediction": false_prediction_histogram[0] if false_prediction_histogram else None,
        "top_substitution_pair": substitution_pairs[0] if substitution_pairs else None,
    }

    alignment_stats_path = artifact_dir / "val_alignment_stats.json"
    alignment_stats_path.write_text(json.dumps(alignment_stats, indent=2))

    summary_path = artifact_dir / effective_probe_config.summary_basename
    summary = {
        "model_variant": model_variant,
        "comparison_mode": comparison_mode or model_variant,
        "probe_head_type": resolve_probe_head_type(effective_probe_config),
        "seed": int(effective_probe_config.seed),
        "probe_head_num_parameters": int(metrics.get("probe_head_num_parameters", 0)),
        "train_encoder": bool(train_encoder),
        "adaptation_regime": str(effective_probe_config.adaptation_regime),
        "session_limit": int(effective_probe_config.session_limit),
        "target_session_count": int(effective_probe_config.target_session_count),
        "heldout_target_session_id": problem["target_session_id"],
        "source_session_ids": list(problem["source_session_ids"]),
        "target_session_ids": list(problem["target_session_ids"]),
        "selected_session_bases": [
            entry.session_base for entry in problem["split"].train + problem["split"].val
        ],
        "target_train_examples": len(problem["target_train_rows"]),
        "target_val_examples": len(problem["target_val_rows"]),
        "val_bpphone": float(metrics["val_ctc_bpphone"]),
        "val_ctc_bpphone": float(metrics["val_ctc_bpphone"]),
        "val_phoneme_error_rate": float(metrics["val_phoneme_error_rate"]),
        "most_common_prediction": (
            alignment_stats["most_common_prediction"]["symbol"]
            if alignment_stats["most_common_prediction"] is not None
            else None
        ),
        "most_common_prediction_rate": (
            float(alignment_stats["most_common_prediction"]["rate"])
            if alignment_stats["most_common_prediction"] is not None
            else None
        ),
        "most_common_false_prediction": (
            alignment_stats["most_common_false_prediction"]["symbol"]
            if alignment_stats["most_common_false_prediction"] is not None
            else None
        ),
        "most_common_false_prediction_rate": (
            float(alignment_stats["most_common_false_prediction"]["rate"])
            if alignment_stats["most_common_false_prediction"] is not None
            else None
        ),
        "top_substitution_pair": (
            f"{alignment_stats['top_substitution_pair']['reference_symbol']}->{alignment_stats['top_substitution_pair']['predicted_symbol']}"
            if alignment_stats["top_substitution_pair"] is not None
            else None
        ),
        "top_substitution_pair_rate": (
            float(alignment_stats["top_substitution_pair"]["rate"])
            if alignment_stats["top_substitution_pair"] is not None
            else None
        ),
        "probe_steps": int(steps),
        "probe_elapsed_seconds": float(elapsed_seconds),
        "checkpoint_source_used": str(probe_state["source"]),
        "checkpoint_path": (
            str(probe_state["checkpoint_path"]) if probe_state["checkpoint_path"] is not None else None
        ),
        "run_dir": str(artifact_dir),
        "progress_log_path": str(progress_path),
        "alignment_stats_path": str(alignment_stats_path),
        "summary_path": str(summary_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    return summary


def run_probe_head_sweep(
    *,
    probe_state: dict[str, Any],
    probe_config: DownstreamProbeConfig,
    cache_root: Path,
    device: torch.device,
    variant_prefix: str,
    artifact_prefix: str,
    train_encoder: bool = False,
    comparison_mode_prefix: str | None = None,
    head_specs: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    specs = head_specs or [
        {"probe_head_type": "linear"},
        {"probe_head_type": "lstm", "probe_lstm_hidden_size": 64},
        {
            "probe_head_type": "conv1d",
            "probe_conv_hidden_size": probe_state["encoder"].hidden_size,
            "probe_conv_kernel_size": 3,
        },
    ]
    results = []
    for spec in specs:
        effective_probe_config = _build_probe_run_config(probe_config, spec)
        suffix = probe_head_suffix(resolve_probe_head_type(effective_probe_config))
        results.append(
            run_downstream_probe(
                probe_state=probe_state,
                probe_config=effective_probe_config,
                cache_root=cache_root,
                device=device,
                variant_prefix=variant_prefix,
                artifact_prefix=artifact_prefix,
                train_encoder=train_encoder,
                comparison_mode=(
                    f"{comparison_mode_prefix}_{suffix}"
                    if comparison_mode_prefix is not None
                    else None
                ),
            )
        )
    return results

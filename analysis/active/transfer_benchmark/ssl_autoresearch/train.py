"""Run the first full SSL transfer benchmark loop.

This benchmark is intentionally narrow:

- dataset: Brain2Text25
- reference backbone: s5
- bootstrap SSL objective: future prediction
- downstream benchmark: held-out-session causal phoneme probe
- primary metric: session-averaged validation bits per target phoneme

The scaffold is designed so S5/Mamba and additional SSL objectives can replace
the placeholder components later without changing the benchmark contract.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader

from s5 import S5SequenceBackbone
from data import (
    CanonicalSequenceDataset,
    collate_sequence_batch,
    compute_feature_stats,
    discover_b2t25_sources,
    filter_matched_tx_sbp,
    inventory_summary,
    load_b2t25_canonical_inventory,
    load_probe_manifest,
    load_probe_metadata,
    partition_probe_records,
    probe_partition_summary,
    session_ids_from_cache_split,
    split_latest_sessions,
)
from prepare import (
    BRAINTOTEXT25_ROOT,
    CACHE_ROOT,
    DEFAULT_ADAPTATION_REGIME,
    DEFAULT_DATASET_FAMILY,
    DEFAULT_PRIMARY_METRIC_NAME,
    DEFAULT_PROFILE,
    BenchmarkSummary,
    count_parameters,
    detect_device,
    ensure_artifact_dirs,
    format_summary,
    make_run_slug,
    now,
    resolve_profile,
    set_seed,
)


MANIFEST_BASENAME = "manifest.jsonl"
METADATA_BASENAME = "metadata.json"
DEFAULT_S5_STATE_SIZE = 64


@dataclass
class RunConfig:
    profile: str = DEFAULT_PROFILE
    dataset_family: str = DEFAULT_DATASET_FAMILY
    backbone: str = "s5"
    objective_family: str = "future_prediction"
    adaptation_regime: str = DEFAULT_ADAPTATION_REGIME
    patch_size: int = 1
    patch_stride: int = 1
    standardize_scope: str = "subject"
    post_proj_norm: str = "rms"
    session_limit: int = 8
    val_session_count: int = 2
    hidden_size: int = 128
    s5_state_size: int = DEFAULT_S5_STATE_SIZE
    num_layers: int = 1
    dropout: float = 0.1
    pretrain_learning_rate: float = 3e-4
    probe_learning_rate: float = 1e-3
    weight_decay: float = 1e-2
    pretrain_batch_size: int = 8
    probe_batch_size: int = 8
    future_horizons: tuple[int, ...] = (1, 3)
    pretrain_budget_seconds: int | None = None
    probe_budget_seconds: int | None = None
    max_pretrain_steps: int | None = None
    max_probe_steps: int | None = None
    progress_every_steps: int = 25
    progress_every_seconds: float = 15.0
    pretrain_loss_ema_alpha: float = 0.2
    pretrain_plateau_min_steps: int = 200
    pretrain_plateau_patience_reports: int = 12
    pretrain_plateau_min_delta: float = 1e-4
    disable_pretrain_plateau_stop: bool = False
    dry_run: bool = False

    def __post_init__(self) -> None:
        if self.patch_stride > self.patch_size:
            raise ValueError("patch_stride must be <= patch_size")
        if self.patch_size not in {1, 3, 5}:
            raise ValueError("patch_size must be one of {1, 3, 5}")
        if self.patch_stride not in {1, 3, 5}:
            raise ValueError("patch_stride must be one of {1, 3, 5}")
        if self.standardize_scope not in {"subject", "session"}:
            raise ValueError("standardize_scope must be one of {'subject', 'session'}")
        if self.post_proj_norm not in {"none", "rms"}:
            raise ValueError("post_proj_norm must be one of {'none', 'rms'}")
        if self.adaptation_regime not in {"A", "B1", "B2"}:
            raise ValueError("adaptation_regime must be one of {'A', 'B1', 'B2'}")
        if self.dataset_family != "brain2text25":
            raise ValueError("Only dataset_family='brain2text25' is wired in this benchmark.")
        if self.hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if self.s5_state_size <= 0:
            raise ValueError("s5_state_size must be positive")
        if self.num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if self.pretrain_batch_size <= 0 or self.probe_batch_size <= 0:
            raise ValueError("batch sizes must be positive")
        if not self.future_horizons:
            raise ValueError("future_horizons must contain at least one value")
        if any(horizon <= 0 for horizon in self.future_horizons):
            raise ValueError("future_horizons must contain only positive integers")
        if self.pretrain_budget_seconds is not None and self.pretrain_budget_seconds <= 0:
            raise ValueError("pretrain_budget_seconds must be positive when provided")
        if self.probe_budget_seconds is not None and self.probe_budget_seconds <= 0:
            raise ValueError("probe_budget_seconds must be positive when provided")
        if self.progress_every_steps <= 0:
            raise ValueError("progress_every_steps must be positive")
        if self.progress_every_seconds <= 0:
            raise ValueError("progress_every_seconds must be positive")
        if not (0.0 < self.pretrain_loss_ema_alpha <= 1.0):
            raise ValueError("pretrain_loss_ema_alpha must be in (0, 1]")
        if self.pretrain_plateau_min_steps < 0:
            raise ValueError("pretrain_plateau_min_steps must be non-negative")
        if self.pretrain_plateau_patience_reports < 0:
            raise ValueError("pretrain_plateau_patience_reports must be non-negative")
        if self.pretrain_plateau_min_delta < 0:
            raise ValueError("pretrain_plateau_min_delta must be non-negative")


@dataclass(frozen=True)
class EncoderOutputs:
    hidden: torch.Tensor
    token_lengths: torch.Tensor
    tokens: torch.Tensor


@dataclass(frozen=True)
class PretrainProgressSummary:
    stop_reason: str
    last_loss: float | None
    last_ema_loss: float | None
    best_ema_loss: float | None
    report_count: int
    plateau_reports_since_improvement: int


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return (x / rms) * self.scale


class SessionAffineBank(nn.Module):
    def __init__(self, session_ids: tuple[str, ...], dim: int):
        super().__init__()
        self._name_map = {session_id: self._module_key(session_id) for session_id in session_ids}
        self.layers = nn.ModuleDict(
            {
                module_key: self._identity_affine(dim)
                for module_key in self._name_map.values()
            }
        )

    @staticmethod
    def _module_key(session_id: str) -> str:
        return session_id.replace(".", "_dot_")

    @staticmethod
    def _identity_affine(dim: int) -> nn.Linear:
        layer = nn.Linear(dim, dim)
        with torch.no_grad():
            layer.weight.zero_()
            layer.weight += torch.eye(dim)
            layer.bias.zero_()
        return layer

    def forward(self, x: torch.Tensor, session_ids: list[str]) -> torch.Tensor:
        transformed: list[torch.Tensor] = []
        for sample, session_id in zip(x, session_ids):
            module_key = self._name_map.get(session_id)
            if module_key is None:
                transformed.append(sample)
            else:
                transformed.append(self.layers[module_key](sample))
        return torch.stack(transformed, dim=0)


class ProjectedCausalEncoderBase(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        hidden_size: int,
        patch_size: int,
        patch_stride: int,
        post_proj_norm: str,
        source_session_ids: tuple[str, ...],
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.token_dim = input_dim * patch_size
        self.source_affines = SessionAffineBank(source_session_ids, input_dim)
        self.proj = nn.Linear(self.token_dim, hidden_size)
        self.post_proj_norm = RMSNorm(hidden_size) if post_proj_norm == "rms" else nn.Identity()

    def _patch_one(self, sample: torch.Tensor, length: int) -> torch.Tensor:
        valid = sample[:length]
        if self.patch_size == 1:
            return valid

        # Intentionally use only stride-aligned valid patches. Any trailing bins that
        # do not fit the chosen (patch_size, stride) schedule are dropped rather than
        # padded into an extra terminal patch.
        starts = list(range(0, max(length - self.patch_size + 1, 1), self.patch_stride))
        patches: list[torch.Tensor] = []
        for start in starts:
            patch = valid[start : start + self.patch_size]
            if patch.shape[0] < self.patch_size:
                pad = valid.new_zeros((self.patch_size - patch.shape[0], valid.shape[1]))
                patch = torch.cat([patch, pad], dim=0)
            patches.append(patch.reshape(-1))
        return torch.stack(patches, dim=0)

    def _patch_batch(self, x: torch.Tensor, input_lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        token_sequences: list[torch.Tensor] = []
        token_lengths: list[int] = []
        for sample, length_tensor in zip(x, input_lengths):
            length = int(length_tensor.item())
            tokens = self._patch_one(sample, length)
            token_sequences.append(tokens)
            token_lengths.append(int(tokens.shape[0]))

        max_tokens = max(token_lengths)
        batch_size = len(token_sequences)
        tokens = x.new_zeros((batch_size, max_tokens, self.token_dim))
        for idx, token_sequence in enumerate(token_sequences):
            tokens[idx, : token_sequence.shape[0]] = token_sequence
        return tokens, torch.tensor(token_lengths, device=input_lengths.device, dtype=torch.long)

    def _align_inputs(
        self,
        x: torch.Tensor,
        session_ids: list[str],
        *,
        use_source_affines: bool,
        target_affines: SessionAffineBank | None = None,
    ) -> torch.Tensor:
        if target_affines is not None:
            aligned = target_affines(x, session_ids)
        elif use_source_affines:
            aligned = self.source_affines(x, session_ids)
        else:
            aligned = x
        return aligned


class DebugCausalEncoder(ProjectedCausalEncoderBase):
    def __init__(
        self,
        *,
        input_dim: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        patch_size: int,
        patch_stride: int,
        post_proj_norm: str,
        source_session_ids: tuple[str, ...],
    ):
        super().__init__(
            input_dim=input_dim,
            hidden_size=hidden_size,
            patch_size=patch_size,
            patch_stride=patch_stride,
            post_proj_norm=post_proj_norm,
            source_session_ids=source_session_ids,
        )
        self.backbone = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )

    def encode(
        self,
        x: torch.Tensor,
        input_lengths: torch.Tensor,
        session_ids: list[str],
        *,
        use_source_affines: bool,
        target_affines: SessionAffineBank | None = None,
    ) -> EncoderOutputs:
        aligned = self._align_inputs(
            x,
            session_ids,
            use_source_affines=use_source_affines,
            target_affines=target_affines,
        )
        tokens, token_lengths = self._patch_batch(aligned, input_lengths)
        hidden = self.post_proj_norm(self.proj(tokens))
        packed = pack_padded_sequence(
            hidden,
            token_lengths.detach().cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_hidden, _ = self.backbone(packed)
        hidden, _ = pad_packed_sequence(
            packed_hidden,
            batch_first=True,
            total_length=tokens.shape[1],
        )
        return EncoderOutputs(hidden=hidden, token_lengths=token_lengths, tokens=tokens)


class S5CausalEncoder(ProjectedCausalEncoderBase):
    def __init__(
        self,
        *,
        input_dim: int,
        hidden_size: int,
        s5_state_size: int,
        num_layers: int,
        dropout: float,
        patch_size: int,
        patch_stride: int,
        post_proj_norm: str,
        source_session_ids: tuple[str, ...],
    ):
        super().__init__(
            input_dim=input_dim,
            hidden_size=hidden_size,
            patch_size=patch_size,
            patch_stride=patch_stride,
            post_proj_norm=post_proj_norm,
            source_session_ids=source_session_ids,
        )
        self.backbone = S5SequenceBackbone(
            d_model=hidden_size,
            d_state=s5_state_size,
            num_layers=num_layers,
            dropout=dropout,
            ffn_multiplier=2.0,
        )

    def encode(
        self,
        x: torch.Tensor,
        input_lengths: torch.Tensor,
        session_ids: list[str],
        *,
        use_source_affines: bool,
        target_affines: SessionAffineBank | None = None,
    ) -> EncoderOutputs:
        aligned = self._align_inputs(
            x,
            session_ids,
            use_source_affines=use_source_affines,
            target_affines=target_affines,
        )
        tokens, token_lengths = self._patch_batch(aligned, input_lengths)
        hidden = self.post_proj_norm(self.proj(tokens))
        hidden = self.backbone(hidden, token_lengths)
        return EncoderOutputs(hidden=hidden, token_lengths=token_lengths, tokens=tokens)


class FuturePredictionHead(nn.Module):
    def __init__(self, hidden_size: int, token_dim: int, horizons: tuple[int, ...]):
        super().__init__()
        self.horizons = horizons
        self.heads = nn.ModuleDict(
            {
                str(horizon): nn.Sequential(
                    nn.Linear(hidden_size, hidden_size),
                    nn.GELU(),
                    nn.Linear(hidden_size, token_dim),
                )
                for horizon in horizons
            }
        )

    def forward(self, hidden: torch.Tensor) -> dict[int, torch.Tensor]:
        return {
            int(horizon): head(hidden)
            for horizon, head in self.heads.items()
        }


class LinearCTCProbe(nn.Module):
    def __init__(self, hidden_size: int, vocab_size: int):
        super().__init__()
        self.classifier = nn.Linear(hidden_size, vocab_size)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.classifier(hidden)


def parse_args() -> RunConfig:
    parser = argparse.ArgumentParser(description="Run the full SSL autoresearch benchmark loop")
    parser.add_argument("--profile", default=DEFAULT_PROFILE)
    parser.add_argument("--dataset-family", default=DEFAULT_DATASET_FAMILY)
    parser.add_argument("--backbone", default="s5")
    parser.add_argument("--objective-family", default="future_prediction")
    parser.add_argument("--adaptation-regime", choices=["A", "B1", "B2"], default=DEFAULT_ADAPTATION_REGIME)
    parser.add_argument("--patch-size", type=int, choices=[1, 3, 5], default=1)
    parser.add_argument("--patch-stride", type=int, choices=[1, 3, 5], default=1)
    parser.add_argument("--standardize-scope", choices=["subject", "session"], default="subject")
    parser.add_argument("--post-proj-norm", choices=["none", "rms"], default="rms")
    parser.add_argument("--session-limit", type=int, default=8)
    parser.add_argument("--val-session-count", type=int, default=2)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--s5-state-size", type=int, default=DEFAULT_S5_STATE_SIZE)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--pretrain-learning-rate", type=float, default=3e-4)
    parser.add_argument("--probe-learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--pretrain-batch-size", type=int, default=8)
    parser.add_argument("--probe-batch-size", type=int, default=8)
    parser.add_argument(
        "--future-horizons",
        default="1,3",
        help="Comma-separated token horizons for the future-prediction SSL objective.",
    )
    parser.add_argument("--pretrain-budget-seconds", type=int, default=None)
    parser.add_argument("--probe-budget-seconds", type=int, default=None)
    parser.add_argument("--max-pretrain-steps", type=int, default=None)
    parser.add_argument("--max-probe-steps", type=int, default=None)
    parser.add_argument("--progress-every-steps", type=int, default=25)
    parser.add_argument("--progress-every-seconds", type=float, default=15.0)
    parser.add_argument("--pretrain-loss-ema-alpha", type=float, default=0.2)
    parser.add_argument("--pretrain-plateau-min-steps", type=int, default=200)
    parser.add_argument("--pretrain-plateau-patience-reports", type=int, default=12)
    parser.add_argument("--pretrain-plateau-min-delta", type=float, default=1e-4)
    parser.add_argument("--disable-pretrain-plateau-stop", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    future_horizons = tuple(sorted({int(value) for value in args.future_horizons.split(",") if value}))
    return RunConfig(
        profile=args.profile,
        dataset_family=args.dataset_family,
        backbone=args.backbone,
        objective_family=args.objective_family,
        adaptation_regime=args.adaptation_regime,
        patch_size=args.patch_size,
        patch_stride=args.patch_stride,
        standardize_scope=args.standardize_scope,
        post_proj_norm=args.post_proj_norm,
        session_limit=args.session_limit,
        val_session_count=args.val_session_count,
        hidden_size=args.hidden_size,
        s5_state_size=args.s5_state_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        pretrain_learning_rate=args.pretrain_learning_rate,
        probe_learning_rate=args.probe_learning_rate,
        weight_decay=args.weight_decay,
        pretrain_batch_size=args.pretrain_batch_size,
        probe_batch_size=args.probe_batch_size,
        future_horizons=future_horizons,
        pretrain_budget_seconds=args.pretrain_budget_seconds,
        probe_budget_seconds=args.probe_budget_seconds,
        max_pretrain_steps=args.max_pretrain_steps,
        max_probe_steps=args.max_probe_steps,
        progress_every_steps=args.progress_every_steps,
        progress_every_seconds=args.progress_every_seconds,
        pretrain_loss_ema_alpha=args.pretrain_loss_ema_alpha,
        pretrain_plateau_min_steps=args.pretrain_plateau_min_steps,
        pretrain_plateau_patience_reports=args.pretrain_plateau_patience_reports,
        pretrain_plateau_min_delta=args.pretrain_plateau_min_delta,
        disable_pretrain_plateau_stop=args.disable_pretrain_plateau_stop,
        dry_run=args.dry_run,
    )


def _checkpoint_path(checkpoint_dir: Path, run_id: str) -> Path:
    return checkpoint_dir / f"{run_id}.pt"


def _run_record_path(run_dir: Path, run_id: str) -> Path:
    return run_dir / f"{run_id}.json"


def _manifest_paths(manifest_dir: Path) -> tuple[Path, Path]:
    cache_dataset_root = CACHE_ROOT / "brain2text25"
    return cache_dataset_root / MANIFEST_BASENAME, cache_dataset_root / METADATA_BASENAME


def _loader_kwargs(device: torch.device, batch_size: int, shuffle: bool) -> dict[str, Any]:
    return {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": 0,
        "pin_memory": device.type == "cuda",
        "collate_fn": collate_sequence_batch,
    }


def _build_model(
    config: RunConfig,
    *,
    input_dim: int,
    source_session_ids: tuple[str, ...],
) -> ProjectedCausalEncoderBase:
    if config.backbone == "debug_gru":
        return DebugCausalEncoder(
            input_dim=input_dim,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            dropout=config.dropout,
            patch_size=config.patch_size,
            patch_stride=config.patch_stride,
            post_proj_norm=config.post_proj_norm,
            source_session_ids=source_session_ids,
        )
    if config.backbone == "s5":
        return S5CausalEncoder(
            input_dim=input_dim,
            hidden_size=config.hidden_size,
            s5_state_size=config.s5_state_size,
            num_layers=config.num_layers,
            dropout=config.dropout,
            patch_size=config.patch_size,
            patch_stride=config.patch_stride,
            post_proj_norm=config.post_proj_norm,
            source_session_ids=source_session_ids,
        )
    if config.backbone == "mamba":
        raise NotImplementedError(
            "Backbone 'mamba' is still pending in the benchmark loop. "
            "Use 's5' or 'debug_gru' for now."
        )
    raise ValueError(f"Unknown backbone: {config.backbone}")


def _validate_objective_family(config: RunConfig) -> None:
    if config.objective_family != "future_prediction":
        raise NotImplementedError(
            f"Objective family '{config.objective_family}' is documented but not wired into the first benchmark loop. "
            "Use 'future_prediction' for now."
        )


def _flatten_targets(labels: torch.Tensor, label_lengths: torch.Tensor) -> torch.Tensor:
    parts = [labels[idx, : int(length.item())] for idx, length in enumerate(label_lengths) if int(length.item()) > 0]
    if not parts:
        return labels.new_empty((0,), dtype=torch.long)
    return torch.cat(parts, dim=0)


def compute_future_prediction_loss(
    predictions: dict[int, torch.Tensor],
    tokens: torch.Tensor,
    token_lengths: torch.Tensor,
) -> torch.Tensor:
    total_sse = tokens.new_tensor(0.0)
    total_count = 0
    for horizon, prediction in predictions.items():
        for idx, length_tensor in enumerate(token_lengths):
            length = int(length_tensor.item())
            usable = length - horizon
            if usable <= 0:
                continue
            diff = prediction[idx, :usable] - tokens[idx, horizon:length]
            total_sse = total_sse + diff.pow(2).sum()
            total_count += diff.numel()
    if total_count == 0:
        raise ValueError("No valid future-prediction targets remain. Reduce the horizon values.")
    return total_sse / total_count


def compute_ctc_loss_sum(
    logits: torch.Tensor,
    input_lengths: torch.Tensor,
    labels: torch.Tensor,
    label_lengths: torch.Tensor,
    *,
    blank_index: int,
) -> tuple[torch.Tensor, int]:
    targets = _flatten_targets(labels, label_lengths)
    total_targets = int(label_lengths.sum().item())
    if total_targets == 0:
        return logits.new_tensor(0.0), 0
    log_probs = F.log_softmax(logits, dim=-1).transpose(0, 1)
    criterion = nn.CTCLoss(blank=blank_index, reduction="sum", zero_infinity=True)
    loss_sum = criterion(
        log_probs,
        targets,
        input_lengths.detach().cpu(),
        label_lengths.detach().cpu(),
    )
    return loss_sum, total_targets


def _set_requires_grad(module: nn.Module | None, value: bool) -> None:
    if module is None:
        return
    for param in module.parameters():
        param.requires_grad = value


def _emit_progress(
    *,
    progress_log_path: Path | None,
    event: str,
    **fields: Any,
) -> None:
    payload = {"event": event, **fields}
    serialized = json.dumps(payload, sort_keys=True)
    print(f"progress_json: {serialized}", flush=True)
    if progress_log_path is not None:
        with progress_log_path.open("a") as handle:
            handle.write(serialized + "\n")


def train_ssl_pretrain(
    *,
    encoder: ProjectedCausalEncoderBase,
    future_head: FuturePredictionHead,
    loader: DataLoader,
    device: torch.device,
    learning_rate: float,
    weight_decay: float,
    budget_seconds: int,
    max_steps: int | None,
    progress_log_path: Path | None,
    progress_every_steps: int,
    progress_every_seconds: float,
    pretrain_loss_ema_alpha: float,
    pretrain_plateau_min_steps: int,
    pretrain_plateau_patience_reports: int,
    pretrain_plateau_min_delta: float,
    disable_pretrain_plateau_stop: bool,
) -> tuple[float, int, PretrainProgressSummary]:
    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(future_head.parameters()),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    encoder.train()
    future_head.train()
    start = now()
    steps = 0
    last_report_elapsed = 0.0
    last_loss: float | None = None
    ema_loss: float | None = None
    best_ema_loss: float | None = None
    report_count = 0
    plateau_reports_since_improvement = 0
    stop_reason = "budget_seconds"

    while True:
        if now() - start >= budget_seconds:
            stop_reason = "budget_seconds"
            break
        if max_steps is not None and steps >= max_steps:
            stop_reason = "max_pretrain_steps"
            break
        for batch in loader:
            if now() - start >= budget_seconds:
                stop_reason = "budget_seconds"
                break
            if max_steps is not None and steps >= max_steps:
                stop_reason = "max_pretrain_steps"
                break

            x = batch["x"].to(device)
            input_lengths = batch["input_lengths"].to(device)
            outputs = encoder.encode(
                x,
                input_lengths,
                batch["session_ids"],
                use_source_affines=True,
                target_affines=None,
            )
            predictions = future_head(outputs.hidden)
            loss = compute_future_prediction_loss(predictions, outputs.tokens, outputs.token_lengths)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(encoder.parameters()) + list(future_head.parameters()),
                max_norm=1.0,
            )
            optimizer.step()
            steps += 1

            loss_value = float(loss.item())
            last_loss = loss_value
            ema_loss = (
                loss_value
                if ema_loss is None
                else ((1.0 - pretrain_loss_ema_alpha) * ema_loss) + (pretrain_loss_ema_alpha * loss_value)
            )
            elapsed = now() - start
            should_report = (
                steps == 1
                or steps % progress_every_steps == 0
                or elapsed - last_report_elapsed >= progress_every_seconds
            )
            if should_report:
                improved = False
                if best_ema_loss is None or (best_ema_loss - ema_loss) >= pretrain_plateau_min_delta:
                    best_ema_loss = ema_loss
                    plateau_reports_since_improvement = 0
                    improved = True
                elif steps >= pretrain_plateau_min_steps:
                    plateau_reports_since_improvement += 1

                report_count += 1
                last_report_elapsed = elapsed
                _emit_progress(
                    progress_log_path=progress_log_path,
                    event="pretrain_report",
                    stage="pretrain",
                    step=steps,
                    elapsed_seconds=round(elapsed, 3),
                    loss=loss_value,
                    ema_loss=ema_loss,
                    best_ema_loss=best_ema_loss,
                    improved=improved,
                    plateau_reports_since_improvement=plateau_reports_since_improvement,
                    budget_seconds=budget_seconds,
                )

                plateau_enabled = not disable_pretrain_plateau_stop and pretrain_plateau_patience_reports > 0
                if (
                    plateau_enabled
                    and steps >= pretrain_plateau_min_steps
                    and plateau_reports_since_improvement >= pretrain_plateau_patience_reports
                ):
                    stop_reason = "plateau"
                    _emit_progress(
                        progress_log_path=progress_log_path,
                        event="pretrain_stop",
                        stage="pretrain",
                        step=steps,
                        elapsed_seconds=round(elapsed, 3),
                        stop_reason=stop_reason,
                        ema_loss=ema_loss,
                        best_ema_loss=best_ema_loss,
                        plateau_reports_since_improvement=plateau_reports_since_improvement,
                    )
                    break
        else:
            continue
        break

    elapsed = now() - start
    _emit_progress(
        progress_log_path=progress_log_path,
        event="pretrain_complete",
        stage="pretrain",
        step=steps,
        elapsed_seconds=round(elapsed, 3),
        stop_reason=stop_reason,
        last_loss=last_loss,
        last_ema_loss=ema_loss,
        best_ema_loss=best_ema_loss,
        report_count=report_count,
        plateau_reports_since_improvement=plateau_reports_since_improvement,
    )
    return elapsed, steps, PretrainProgressSummary(
        stop_reason=stop_reason,
        last_loss=last_loss,
        last_ema_loss=ema_loss,
        best_ema_loss=best_ema_loss,
        report_count=report_count,
        plateau_reports_since_improvement=plateau_reports_since_improvement,
    )


@torch.no_grad()
def evaluate_probe_session(
    *,
    encoder: ProjectedCausalEncoderBase,
    probe_head: LinearCTCProbe,
    target_affines: SessionAffineBank | None,
    loader: DataLoader,
    device: torch.device,
    blank_index: int,
) -> float:
    encoder.eval()
    probe_head.eval()
    if target_affines is not None:
        target_affines.eval()

    total_loss_sum = 0.0
    total_targets = 0
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
            target_affines=target_affines,
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

    if total_targets <= 0:
        raise ValueError("Validation target count is zero; cannot compute val_bpphone.")
    return total_loss_sum / total_targets / math.log(2.0)


def train_probe_for_session(
    *,
    pretrained_encoder: ProjectedCausalEncoderBase,
    session_id: str,
    train_rows: tuple[Any, ...],
    val_rows: tuple[Any, ...],
    probe_vocab_size: int,
    blank_index: int,
    config: RunConfig,
    device: torch.device,
    budget_seconds: float,
    progress_log_path: Path | None,
) -> tuple[float, int]:
    target_stats = compute_feature_stats(train_rows, mode="global")
    train_loader = DataLoader(
        CanonicalSequenceDataset(train_rows, stats=target_stats),
        **_loader_kwargs(device, config.probe_batch_size, shuffle=True),
    )
    val_loader = DataLoader(
        CanonicalSequenceDataset(val_rows, stats=target_stats),
        **_loader_kwargs(device, config.probe_batch_size, shuffle=False),
    )

    encoder = copy.deepcopy(pretrained_encoder).to(device)
    target_affines = SessionAffineBank((session_id,), encoder.input_dim).to(device) if config.adaptation_regime in {"B1", "B2"} else None
    probe_head = LinearCTCProbe(encoder.hidden_size, probe_vocab_size).to(device)

    if config.adaptation_regime == "A":
        _set_requires_grad(encoder, False)
        _set_requires_grad(target_affines, False)
    elif config.adaptation_regime == "B1":
        _set_requires_grad(encoder, False)
        _set_requires_grad(target_affines, True)
    elif config.adaptation_regime == "B2":
        _set_requires_grad(encoder, True)
        _set_requires_grad(target_affines, True)
    _set_requires_grad(probe_head, True)

    parameters = [param for param in list(encoder.parameters()) + list(probe_head.parameters()) + list(target_affines.parameters() if target_affines is not None else []) if param.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=config.probe_learning_rate, weight_decay=config.weight_decay)

    start = now()
    steps = 0
    last_report_elapsed = 0.0
    while True:
        if now() - start >= budget_seconds:
            break
        if config.max_probe_steps is not None and steps >= config.max_probe_steps:
            break
        for batch in train_loader:
            if now() - start >= budget_seconds:
                break
            if config.max_probe_steps is not None and steps >= config.max_probe_steps:
                break

            encoder.train(any(param.requires_grad for param in encoder.parameters()))
            probe_head.train()
            if target_affines is not None:
                target_affines.train()

            x = batch["x"].to(device)
            input_lengths = batch["input_lengths"].to(device)
            labels = batch["labels"].to(device)
            label_lengths = batch["label_lengths"].to(device)

            outputs = encoder.encode(
                x,
                input_lengths,
                batch["session_ids"],
                use_source_affines=False,
                target_affines=target_affines,
            )
            logits = probe_head(outputs.hidden)
            loss_sum, target_count = compute_ctc_loss_sum(
                logits,
                outputs.token_lengths,
                labels,
                label_lengths,
                blank_index=blank_index,
            )
            if target_count <= 0:
                continue
            loss = loss_sum / target_count

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, max_norm=1.0)
            optimizer.step()
            steps += 1

            elapsed = now() - start
            should_report = (
                steps == 1
                or steps % config.progress_every_steps == 0
                or elapsed - last_report_elapsed >= config.progress_every_seconds
            )
            if should_report:
                last_report_elapsed = elapsed
                _emit_progress(
                    progress_log_path=progress_log_path,
                    event="probe_train_report",
                    stage="probe_train",
                    session_id=session_id,
                    step=steps,
                    elapsed_seconds=round(elapsed, 3),
                    loss_bpphone=float(loss.item()) / math.log(2.0),
                    budget_seconds=budget_seconds,
                )
        else:
            continue
        break

    session_bpphone = evaluate_probe_session(
        encoder=encoder,
        probe_head=probe_head,
        target_affines=target_affines,
        loader=val_loader,
        device=device,
        blank_index=blank_index,
    )
    _emit_progress(
        progress_log_path=progress_log_path,
        event="probe_session_complete",
        stage="probe_train",
        session_id=session_id,
        step=steps,
        elapsed_seconds=round(now() - start, 3),
        val_bpphone=session_bpphone,
    )
    return session_bpphone, steps


def _write_checkpoint(
    *,
    checkpoint_path: Path,
    encoder: ProjectedCausalEncoderBase,
    future_head: FuturePredictionHead,
    config: RunConfig,
    source_session_ids: tuple[str, ...],
    target_session_ids: tuple[str, ...],
) -> None:
    torch.save(
        {
            "config": {
                "profile": config.profile,
                "dataset_family": config.dataset_family,
                "backbone": config.backbone,
                "objective_family": config.objective_family,
                "adaptation_regime": config.adaptation_regime,
                "patch_size": config.patch_size,
                "patch_stride": config.patch_stride,
                "standardize_scope": config.standardize_scope,
                "post_proj_norm": config.post_proj_norm,
                "hidden_size": config.hidden_size,
                "s5_state_size": config.s5_state_size,
                "num_layers": config.num_layers,
                "dropout": config.dropout,
                "future_horizons": list(config.future_horizons),
            },
            "source_session_ids": list(source_session_ids),
            "target_session_ids": list(target_session_ids),
            "encoder_state_dict": encoder.state_dict(),
            "future_head_state_dict": future_head.state_dict(),
        },
        checkpoint_path,
    )


def _write_run_record(
    *,
    path: Path,
    config: RunConfig,
    source_session_ids: tuple[str, ...],
    target_session_ids: tuple[str, ...],
    session_metrics: dict[str, float],
    pretrain_steps: int,
    pretrain_progress: PretrainProgressSummary,
    probe_steps_by_session: dict[str, int],
    summary: BenchmarkSummary,
    progress_log_path: Path,
) -> None:
    payload = {
        "config": {
            "profile": config.profile,
            "dataset_family": config.dataset_family,
            "backbone": config.backbone,
            "objective_family": config.objective_family,
            "adaptation_regime": config.adaptation_regime,
            "patch_size": config.patch_size,
            "patch_stride": config.patch_stride,
            "standardize_scope": config.standardize_scope,
            "post_proj_norm": config.post_proj_norm,
            "hidden_size": config.hidden_size,
            "s5_state_size": config.s5_state_size,
            "num_layers": config.num_layers,
            "dropout": config.dropout,
            "pretrain_learning_rate": config.pretrain_learning_rate,
            "probe_learning_rate": config.probe_learning_rate,
            "weight_decay": config.weight_decay,
            "pretrain_batch_size": config.pretrain_batch_size,
            "probe_batch_size": config.probe_batch_size,
            "future_horizons": list(config.future_horizons),
            "pretrain_budget_seconds": config.pretrain_budget_seconds,
            "probe_budget_seconds": config.probe_budget_seconds,
            "progress_every_steps": config.progress_every_steps,
            "progress_every_seconds": config.progress_every_seconds,
            "pretrain_loss_ema_alpha": config.pretrain_loss_ema_alpha,
            "pretrain_plateau_min_steps": config.pretrain_plateau_min_steps,
            "pretrain_plateau_patience_reports": config.pretrain_plateau_patience_reports,
            "pretrain_plateau_min_delta": config.pretrain_plateau_min_delta,
            "disable_pretrain_plateau_stop": config.disable_pretrain_plateau_stop,
        },
        "source_session_ids": list(source_session_ids),
        "target_session_ids": list(target_session_ids),
        "session_metrics": session_metrics,
        "pretrain_steps": pretrain_steps,
        "pretrain_progress": {
            "stop_reason": pretrain_progress.stop_reason,
            "last_loss": pretrain_progress.last_loss,
            "last_ema_loss": pretrain_progress.last_ema_loss,
            "best_ema_loss": pretrain_progress.best_ema_loss,
            "report_count": pretrain_progress.report_count,
            "plateau_reports_since_improvement": pretrain_progress.plateau_reports_since_improvement,
        },
        "probe_steps_by_session": probe_steps_by_session,
        "progress_log_path": str(progress_log_path),
        "summary": {
            "benchmark_state": summary.benchmark_state,
            "primary_metric_name": summary.primary_metric_name,
            "primary_metric_value": summary.primary_metric_value,
            "total_seconds": summary.total_seconds,
            "pretrain_seconds": summary.pretrain_seconds,
            "probe_seconds": summary.probe_seconds,
            "device": summary.device,
            "profile": summary.profile,
            "dataset_family": summary.dataset_family,
            "backbone": summary.backbone,
            "objective_family": summary.objective_family,
            "adaptation_regime": summary.adaptation_regime,
            "patch_size": summary.patch_size,
            "patch_stride": summary.patch_stride,
            "standardize_scope": summary.standardize_scope,
            "post_proj_norm": summary.post_proj_norm,
            "num_source_sessions": summary.num_source_sessions,
            "num_target_sessions": summary.num_target_sessions,
            "checkpoint_path": summary.checkpoint_path,
        },
    }
    path.write_text(json.dumps(payload, indent=2))


def main() -> int:
    config = parse_args()
    _validate_objective_family(config)
    set_seed()
    overall_start = now()

    profile = resolve_profile(config.profile)
    device = detect_device(profile)
    if device.type == "mps":
        print("device_note: MPS is not preferred for the CTC benchmark path; falling back to CPU.")
        device = torch.device("cpu")

    artifacts = ensure_artifact_dirs()
    manifest_path, metadata_path = _manifest_paths(artifacts.manifest_dir)
    if not manifest_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(
            "Canonical Brain2Text25 cache metadata is missing. Build data/cache_v1/brain2text25 before train.py."
        )

    run_slug = make_run_slug(
        dataset_family=config.dataset_family,
        backbone=config.backbone,
        objective_family=config.objective_family,
        adaptation_regime=config.adaptation_regime,
        patch_size=config.patch_size,
        patch_stride=config.patch_stride,
        standardize_scope=config.standardize_scope,
        post_proj_norm=config.post_proj_norm,
    )
    run_id = f"{run_slug}__{int(overall_start)}"
    checkpoint_path = _checkpoint_path(artifacts.checkpoint_dir, run_id)
    run_record_path = _run_record_path(artifacts.run_dir, run_id)
    progress_log_path = artifacts.log_dir / f"{run_id}.progress.jsonl"
    if progress_log_path.exists():
        progress_log_path.unlink()
    print(f"checkpoint_path: {checkpoint_path}")
    print(f"progress_log_path: {progress_log_path}")

    effective_pretrain_budget_seconds = (
        config.pretrain_budget_seconds
        if config.pretrain_budget_seconds is not None
        else profile.pretrain_budget_seconds
    )
    effective_probe_budget_seconds = (
        config.probe_budget_seconds
        if config.probe_budget_seconds is not None
        else profile.probe_budget_seconds
    )

    entries = load_b2t25_canonical_inventory()
    matched_entries = filter_matched_tx_sbp(entries)
    split = split_latest_sessions(
        matched_entries,
        session_limit=config.session_limit,
        val_session_count=config.val_session_count,
    )
    source_session_ids, target_session_ids = session_ids_from_cache_split(split)

    manifest_rows = load_probe_manifest(manifest_path)
    manifest_metadata = load_probe_metadata(metadata_path)
    partitions = partition_probe_records(
        manifest_rows,
        source_session_ids=source_session_ids,
        target_session_ids=target_session_ids,
    )
    sources = discover_b2t25_sources(BRAINTOTEXT25_ROOT)

    if config.dry_run:
        print("dry_run: true")
        print(f"profile_name: {profile.name}")
        print(f"profile_pretrain_budget_seconds: {profile.pretrain_budget_seconds}")
        print(f"profile_probe_budget_seconds: {profile.probe_budget_seconds}")
        print(f"effective_pretrain_budget_seconds: {effective_pretrain_budget_seconds}")
        print(f"effective_probe_budget_seconds: {effective_probe_budget_seconds}")
        print(f"progress_every_steps: {config.progress_every_steps}")
        print(f"progress_every_seconds: {config.progress_every_seconds}")
        print(f"pretrain_loss_ema_alpha: {config.pretrain_loss_ema_alpha}")
        print(f"pretrain_plateau_min_steps: {config.pretrain_plateau_min_steps}")
        print(f"pretrain_plateau_patience_reports: {config.pretrain_plateau_patience_reports}")
        print(f"pretrain_plateau_min_delta: {config.pretrain_plateau_min_delta}")
        print(f"disable_pretrain_plateau_stop: {config.disable_pretrain_plateau_stop}")
        print(f"detected_device: {device}")
        print(f"artifact_output_root: {artifacts.output_root}")
        print(f"cache_manifest_path: {manifest_path}")
        print(f"metadata_path: {metadata_path}")
        print(f"checkpoint_path: {checkpoint_path}")
        print(f"progress_log_path: {progress_log_path}")
        print(f"inventory_summary: {inventory_summary(entries)}")
        print(f"partition_summary: {probe_partition_summary(partitions)}")
        print(f"train_sessions: {[entry.session_base for entry in split.train]}")
        print(f"val_sessions: {[entry.session_base for entry in split.val]}")
        print(f"sources: {[source.__dict__ for source in sources]}")
        summary = BenchmarkSummary(
            benchmark_state="benchmark_dry_run",
            primary_metric_name=DEFAULT_PRIMARY_METRIC_NAME,
            primary_metric_value=float("nan"),
            total_seconds=now() - overall_start,
            pretrain_seconds=0.0,
            probe_seconds=0.0,
            device=str(device),
            profile=profile.name,
            dataset_family=config.dataset_family,
            backbone=config.backbone,
            objective_family=config.objective_family,
            adaptation_regime=config.adaptation_regime,
            patch_size=config.patch_size,
            patch_stride=config.patch_stride,
            standardize_scope=config.standardize_scope,
            post_proj_norm=config.post_proj_norm,
            num_source_sessions=len(source_session_ids),
            num_target_sessions=len(target_session_ids),
            checkpoint_path=str(checkpoint_path),
        )
        print(format_summary(summary))
        print(f"run_record_path: {run_record_path}")
        return 0

    if not partitions.source_pretrain:
        raise ValueError("No source-session pretraining examples were found in the probe manifest.")
    if any(len(rows) == 0 for rows in partitions.target_train_by_session.values()):
        raise ValueError("At least one target session has zero probe-train examples.")
    if any(len(rows) == 0 for rows in partitions.target_val_by_session.values()):
        raise ValueError("At least one target session has zero probe-val examples.")

    input_dim = partitions.source_pretrain[0].n_features
    encoder = _build_model(config, input_dim=input_dim, source_session_ids=source_session_ids).to(device)
    future_head = FuturePredictionHead(
        hidden_size=encoder.hidden_size,
        token_dim=encoder.token_dim,
        horizons=config.future_horizons,
    ).to(device)

    if config.standardize_scope == "subject":
        source_stats: Any = compute_feature_stats(partitions.source_pretrain, mode="global")
    else:
        source_stats = compute_feature_stats(partitions.source_pretrain, mode="per_session")

    pretrain_loader = DataLoader(
        CanonicalSequenceDataset(partitions.source_pretrain, stats=source_stats),
        **_loader_kwargs(device, config.pretrain_batch_size, shuffle=True),
    )
    pretrain_seconds, pretrain_steps, pretrain_progress = train_ssl_pretrain(
        encoder=encoder,
        future_head=future_head,
        loader=pretrain_loader,
        device=device,
        learning_rate=config.pretrain_learning_rate,
        weight_decay=config.weight_decay,
        budget_seconds=effective_pretrain_budget_seconds,
        max_steps=config.max_pretrain_steps,
        progress_log_path=progress_log_path,
        progress_every_steps=config.progress_every_steps,
        progress_every_seconds=config.progress_every_seconds,
        pretrain_loss_ema_alpha=config.pretrain_loss_ema_alpha,
        pretrain_plateau_min_steps=config.pretrain_plateau_min_steps,
        pretrain_plateau_patience_reports=config.pretrain_plateau_patience_reports,
        pretrain_plateau_min_delta=config.pretrain_plateau_min_delta,
        disable_pretrain_plateau_stop=config.disable_pretrain_plateau_stop,
    )

    _write_checkpoint(
        checkpoint_path=checkpoint_path,
        encoder=encoder,
        future_head=future_head,
        config=config,
        source_session_ids=source_session_ids,
        target_session_ids=target_session_ids,
    )

    phoneme_vocab = manifest_metadata["phoneme_vocabulary"]
    probe_vocab_size = int(phoneme_vocab["num_classes"])
    blank_index = int(phoneme_vocab["blank_index"])
    probe_budget_per_session = effective_probe_budget_seconds / max(1, len(target_session_ids))

    session_metrics: dict[str, float] = {}
    probe_steps_by_session: dict[str, int] = {}
    probe_start = now()
    for session_id in target_session_ids:
        session_bpphone, probe_steps = train_probe_for_session(
            pretrained_encoder=encoder,
            session_id=session_id,
            train_rows=partitions.target_train_by_session[session_id],
            val_rows=partitions.target_val_by_session[session_id],
            probe_vocab_size=probe_vocab_size,
            blank_index=blank_index,
            config=config,
            device=device,
            budget_seconds=probe_budget_per_session,
            progress_log_path=progress_log_path,
        )
        session_metrics[session_id] = session_bpphone
        probe_steps_by_session[session_id] = probe_steps
        print(f"session_val_bpphone[{session_id}]: {session_bpphone:.10f}")

    session_avg_val_bpphone = sum(session_metrics.values()) / len(session_metrics)
    probe_seconds = now() - probe_start

    summary = BenchmarkSummary(
        benchmark_state="benchmark_complete",
        primary_metric_name=DEFAULT_PRIMARY_METRIC_NAME,
        primary_metric_value=session_avg_val_bpphone,
        total_seconds=now() - overall_start,
        pretrain_seconds=pretrain_seconds,
        probe_seconds=probe_seconds,
        device=str(device),
        profile=profile.name,
        dataset_family=config.dataset_family,
        backbone=config.backbone,
        objective_family=config.objective_family,
        adaptation_regime=config.adaptation_regime,
        patch_size=config.patch_size,
        patch_stride=config.patch_stride,
        standardize_scope=config.standardize_scope,
        post_proj_norm=config.post_proj_norm,
        num_source_sessions=len(source_session_ids),
        num_target_sessions=len(target_session_ids),
        checkpoint_path=str(checkpoint_path),
    )

    _write_run_record(
        path=run_record_path,
        config=config,
        source_session_ids=source_session_ids,
        target_session_ids=target_session_ids,
        session_metrics=session_metrics,
        pretrain_steps=pretrain_steps,
        pretrain_progress=pretrain_progress,
        probe_steps_by_session=probe_steps_by_session,
        summary=summary,
        progress_log_path=progress_log_path,
    )

    print(format_summary(summary))
    print(f"run_record_path: {run_record_path}")
    print(f"progress_log_path: {progress_log_path}")
    print(f"pretrain_steps: {pretrain_steps}")
    print(f"pretrain_stop_reason: {pretrain_progress.stop_reason}")
    print(f"probe_steps_by_session: {probe_steps_by_session}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Editable training script for the local SSL autoresearch smoke test.

This is intentionally a small causal placeholder model for local loop validation.
It is not the intended final S5/Mamba thesis architecture.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from data import load_local_smoke_bundle
from prepare import (
    BenchmarkSummary,
    EVAL_INTERVAL_STEPS,
    LOCAL_PROFILE,
    RANDOM_SEED,
    SBP_CACHE_DIR,
    TIME_BUDGET_SECONDS,
    TX_CACHE_DIR,
    count_parameters,
    detect_device,
    format_summary,
    now,
    set_seed,
)


@dataclass
class RunConfig:
    batch_size: int = 32
    learning_rate: float = 3e-4
    hidden_size: int = 128
    num_layers: int = 1
    dropout: float = 0.1
    patch_size: int = 1
    patch_stride: int = 1
    standardize_scope: str = LOCAL_PROFILE["default_standardize_scope"]
    post_proj_norm: str = "rms"
    horizons: tuple[int, ...] = LOCAL_PROFILE["default_horizons"]

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


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return (x / rms) * self.scale


class CausalSmokeModel(nn.Module):
    def __init__(self, input_dim: int, num_sessions: int, config: RunConfig):
        super().__init__()
        self.input_dim = input_dim
        self.patch_size = config.patch_size
        self.patch_stride = config.patch_stride
        self.horizons = config.horizons

        self.session_layers = nn.ModuleList([nn.Linear(input_dim, input_dim) for _ in range(num_sessions)])
        for layer in self.session_layers:
            nn.init.eye_(layer.weight)
            nn.init.zeros_(layer.bias)

        token_dim = input_dim * config.patch_size
        self.token_dim = token_dim
        self.proj = nn.Linear(token_dim, config.hidden_size)
        self.post_proj_norm = RMSNorm(config.hidden_size) if config.post_proj_norm == "rms" else nn.Identity()
        self.backbone = nn.GRU(
            input_size=config.hidden_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            dropout=config.dropout if config.num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.future_heads = nn.ModuleDict(
            {
                str(horizon): nn.Sequential(
                    nn.Linear(config.hidden_size, config.hidden_size),
                    nn.GELU(),
                    nn.Linear(config.hidden_size, token_dim),
                )
                for horizon in self.horizons
            }
        )

    def _apply_session_affine(self, x: torch.Tensor, session_idx: torch.Tensor) -> torch.Tensor:
        transformed = []
        for sample, idx in zip(x, session_idx.tolist()):
            transformed.append(self.session_layers[idx](sample))
        return torch.stack(transformed, dim=0)

    def _patch_sequence(self, x: torch.Tensor) -> torch.Tensor:
        if self.patch_size == 1:
            return x
        patched = []
        for sample in x:
            windows = sample.unfold(0, self.patch_size, self.patch_stride)
            patched.append(windows.reshape(windows.shape[0], -1))
        return torch.stack(patched, dim=0)

    def forward(self, x: torch.Tensor, session_idx: torch.Tensor) -> tuple[dict[int, torch.Tensor], torch.Tensor]:
        aligned = self._apply_session_affine(x, session_idx)
        tokens = self._patch_sequence(aligned)
        hidden = self.post_proj_norm(self.proj(tokens))
        hidden, _ = self.backbone(hidden)
        predictions = {
            horizon: head(hidden)
            for horizon, head in ((int(key), head) for key, head in self.future_heads.items())
        }
        return predictions, tokens


def parse_args() -> RunConfig:
    parser = argparse.ArgumentParser(description="Train the local SSL smoke-test benchmark")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--patch-size", type=int, choices=[1, 3, 5], default=1)
    parser.add_argument("--patch-stride", type=int, choices=[1, 3, 5], default=1)
    parser.add_argument("--standardize-scope", choices=["subject", "session"], default="subject")
    parser.add_argument("--post-proj-norm", choices=["none", "rms"], default="rms")
    parser.add_argument("--horizons", default="1,3", help="Comma-separated token horizons, e.g. '1,3'")
    args = parser.parse_args()
    horizons = tuple(sorted({int(value) for value in args.horizons.split(",") if value}))
    return RunConfig(
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        patch_size=args.patch_size,
        patch_stride=args.patch_stride,
        standardize_scope=args.standardize_scope,
        post_proj_norm=args.post_proj_norm,
        horizons=horizons,
    )


def compute_future_loss(
    predictions: dict[int, torch.Tensor],
    tokens: torch.Tensor,
) -> torch.Tensor:
    losses = []
    for horizon, prediction in predictions.items():
        if tokens.shape[1] <= horizon:
            continue
        pred_slice = prediction[:, :-horizon, :]
        target_slice = tokens[:, horizon:, :]
        losses.append(F.mse_loss(pred_slice, target_slice))
    if not losses:
        raise ValueError("No valid horizons remain after patching. Reduce the horizon values.")
    return sum(losses) / len(losses)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    total_batches = 0
    for x, session_idx in loader:
        x = x.to(device)
        session_idx = session_idx.to(device)
        predictions, tokens = model(x, session_idx)
        loss = compute_future_loss(predictions, tokens)
        total_loss += float(loss.item())
        total_batches += 1
    return total_loss / max(1, total_batches)


def main() -> int:
    config = parse_args()
    set_seed(RANDOM_SEED)
    overall_start = now()
    device = detect_device()

    bundle = load_local_smoke_bundle(
        TX_CACHE_DIR,
        SBP_CACHE_DIR,
        session_limit=LOCAL_PROFILE["session_limit"],
        session_selection=LOCAL_PROFILE["session_selection"],
        val_session_count=LOCAL_PROFILE["val_session_count"],
        train_windows_per_session=LOCAL_PROFILE["train_windows_per_session"],
        val_windows_per_session=LOCAL_PROFILE["val_windows_per_session"],
        standardize_scope=config.standardize_scope,
        seed=RANDOM_SEED,
    )

    train_loader = DataLoader(
        TensorDataset(bundle.train_x, bundle.train_session_idx),
        batch_size=config.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(bundle.val_x, bundle.val_session_idx),
        batch_size=config.batch_size,
        shuffle=False,
    )

    model = CausalSmokeModel(bundle.input_dim, bundle.num_sessions, config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    best_val_loss = float("inf")
    num_steps = 0
    train_start = now()
    keep_training = True

    while keep_training:
        for x, session_idx in train_loader:
            if now() - train_start >= TIME_BUDGET_SECONDS:
                keep_training = False
                break

            model.train()
            x = x.to(device)
            session_idx = session_idx.to(device)

            predictions, tokens = model(x, session_idx)
            loss = compute_future_loss(predictions, tokens)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            num_steps += 1
            if num_steps % EVAL_INTERVAL_STEPS == 0:
                best_val_loss = min(best_val_loss, evaluate(model, val_loader, device))

    final_val_loss = evaluate(model, val_loader, device)
    best_val_loss = min(best_val_loss, final_val_loss)

    summary = BenchmarkSummary(
        val_ssl_loss=best_val_loss,
        training_seconds=now() - train_start,
        total_seconds=now() - overall_start,
        device=str(device),
        num_steps=num_steps,
        num_params=count_parameters(model),
        patch_size=config.patch_size,
        patch_stride=config.patch_stride,
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        standardize_scope=config.standardize_scope,
        post_proj_norm=config.post_proj_norm,
    )
    print(format_summary(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

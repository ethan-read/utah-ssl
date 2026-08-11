"""Frozen benchmark harness for the local SSL autoresearch smoke test."""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


ROOT_DIR = Path(__file__).resolve().parents[4]
TX_CACHE_DIR = ROOT_DIR / "code" / "ssl" / "cache"
SBP_CACHE_DIR = ROOT_DIR / "code" / "ssl" / "cache_b2t25_sbp"

TIME_BUDGET_SECONDS = 5 * 60
RANDOM_SEED = 7
VAL_METRIC_NAME = "val_ssl_loss"
EVAL_INTERVAL_STEPS = 50

LOCAL_PROFILE = {
    "session_limit": 8,
    "session_selection": "latest",
    "val_session_count": 2,
    "train_windows_per_session": 256,
    "val_windows_per_session": 64,
    "default_standardize_scope": "subject",
    "default_horizons": (1, 3),
}


@dataclass(frozen=True)
class BenchmarkSummary:
    val_ssl_loss: float
    training_seconds: float
    total_seconds: float
    device: str
    num_steps: int
    num_params: int
    patch_size: int
    patch_stride: int
    hidden_size: int
    num_layers: int
    standardize_scope: str
    post_proj_norm: str


def set_seed(seed: int = RANDOM_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def detect_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def count_parameters(model: torch.nn.Module) -> int:
    return sum(param.numel() for param in model.parameters())


def format_summary(summary: BenchmarkSummary) -> str:
    lines = [
        f"{VAL_METRIC_NAME}: {summary.val_ssl_loss:.6f}",
        f"training_seconds: {summary.training_seconds:.1f}",
        f"total_seconds: {summary.total_seconds:.1f}",
        f"device: {summary.device}",
        f"num_steps: {summary.num_steps}",
        f"num_params: {summary.num_params}",
        f"patch_size: {summary.patch_size}",
        f"patch_stride: {summary.patch_stride}",
        f"hidden_size: {summary.hidden_size}",
        f"num_layers: {summary.num_layers}",
        f"standardize_scope: {summary.standardize_scope}",
        f"post_proj_norm: {summary.post_proj_norm}",
    ]
    return "\n".join(lines)


def now() -> float:
    return time.time()

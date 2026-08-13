"""Small runtime helpers shared by experiment training loops."""

from __future__ import annotations

import random

import numpy as np
import torch


def seed_torch(seed: int) -> None:
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    seed_torch(seed)


def resolve_device(requested: str | torch.device | None = None) -> torch.device:
    if requested is not None:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


__all__ = ["resolve_device", "seed_everything", "seed_torch"]

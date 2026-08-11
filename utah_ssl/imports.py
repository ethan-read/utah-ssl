"""Repository path helpers shared by experiment packages."""

from __future__ import annotations

from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


__all__ = [
    "repo_root",
]

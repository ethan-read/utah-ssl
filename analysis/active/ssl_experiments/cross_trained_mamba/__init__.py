"""Cross-trained area-6v Mamba phoneme decoding experiments."""

from __future__ import annotations

import sys
from pathlib import Path


def _ensure_repo_import_paths() -> None:
    package_dir = Path(__file__).resolve().parent
    repo_root = package_dir.parents[3]
    experiments_dir = repo_root / "analysis" / "active" / "ssl_experiments"
    benchmark_dir = repo_root / "analysis" / "active" / "transfer_benchmark" / "ssl_autoresearch"
    for path in (repo_root, experiments_dir, benchmark_dir):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


_ensure_repo_import_paths()

from .config import CrossTrainedMambaConfig
from .model import CrossTrainedMambaPhonemeModel
from .train import run_cross_trained_mamba

__all__ = [
    "CrossTrainedMambaConfig",
    "CrossTrainedMambaPhonemeModel",
    "run_cross_trained_mamba",
]

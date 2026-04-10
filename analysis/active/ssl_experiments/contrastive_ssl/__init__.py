"""Reusable contrastive SSL helpers for Colab notebook experiments."""

from __future__ import annotations

import sys
from pathlib import Path


def _ensure_repo_import_paths() -> None:
    package_dir = Path(__file__).resolve().parent
    repo_root = package_dir.parents[3]
    benchmark_dir = repo_root / "analysis" / "active" / "transfer_benchmark" / "ssl_autoresearch"
    for path in (repo_root, benchmark_dir):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


_ensure_repo_import_paths()

from .cache import CacheAccessConfig, CacheContext, build_segment_sampler, prepare_cache_context
from .probe import (
    DownstreamProbeConfig,
    build_random_init_probe_state,
    recover_downstream_probe_state,
    run_downstream_probe,
    run_probe_head_sweep,
)
from .training import SSLTrainingConfig, list_ssl_checkpoints, plot_ssl_training_history, run_ssl_training


__all__ = [
    "CacheAccessConfig",
    "CacheContext",
    "DownstreamProbeConfig",
    "SSLTrainingConfig",
    "build_random_init_probe_state",
    "build_segment_sampler",
    "list_ssl_checkpoints",
    "plot_ssl_training_history",
    "prepare_cache_context",
    "recover_downstream_probe_state",
    "run_downstream_probe",
    "run_probe_head_sweep",
    "run_ssl_training",
]

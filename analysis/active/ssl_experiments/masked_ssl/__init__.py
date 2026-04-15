"""Reusable causal masked-reconstruction SSL helpers for Colab notebooks."""

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

from .cache import (
    CacheAccessConfig,
    CacheContext,
    build_segment_sampler,
    load_precomputed_session_feature_stats_into_cache_context,
    prepare_cache_context,
)
from .probe import (
    DownstreamProbeConfig,
    build_random_init_probe_state,
    recover_downstream_probe_state,
    run_downstream_probe,
    run_probe_head_sweep,
)
from .phoneme_finetune import (
    PhonemeFinetuneConfig,
    run_phoneme_finetuning,
)
from .training import (
    SSLTrainingConfig,
    list_ssl_checkpoints,
    plot_ssl_training_history,
    recover_ssl_run_state_from_checkpoint,
    resolve_ssl_checkpoint_path,
    run_ssl_training,
)
from .training_mae import (
    SSLTrainingConfig as SSLTrainingConfigMAE,
    list_ssl_checkpoints as list_ssl_checkpoints_mae,
    plot_ssl_training_history as plot_ssl_training_history_mae,
    recover_ssl_run_state_from_checkpoint as recover_ssl_run_state_from_checkpoint_mae,
    resolve_ssl_checkpoint_path as resolve_ssl_checkpoint_path_mae,
    run_ssl_training as run_ssl_training_mae,
)


__all__ = [
    "CacheAccessConfig",
    "CacheContext",
    "DownstreamProbeConfig",
    "PhonemeFinetuneConfig",
    "SSLTrainingConfig",
    "build_random_init_probe_state",
    "build_segment_sampler",
    "list_ssl_checkpoints",
    "load_precomputed_session_feature_stats_into_cache_context",
    "plot_ssl_training_history",
    "prepare_cache_context",
    "recover_downstream_probe_state",
    "recover_ssl_run_state_from_checkpoint",
    "resolve_ssl_checkpoint_path",
    "run_downstream_probe",
    "run_phoneme_finetuning",
    "run_probe_head_sweep",
    "run_ssl_training",
    "SSLTrainingConfigMAE",
    "list_ssl_checkpoints_mae",
    "plot_ssl_training_history_mae",
    "recover_ssl_run_state_from_checkpoint_mae",
    "resolve_ssl_checkpoint_path_mae",
    "run_ssl_training_mae",
]

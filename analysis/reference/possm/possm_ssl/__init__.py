"""Reusable POSSM-style SSL helpers for notebook experiments."""

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

from masked_ssl.cache import (
    CacheAccessConfig,
    CacheContext,
    load_precomputed_session_feature_stats_into_cache_context,
    prepare_cache_context,
)

from .model import (
    POSSMEncoder,
    POSSMPhonemeModel,
    POSSMReconstructionModel,
    SessionInputAdapterBank,
    causal_conv_output_lengths,
    list_registered_temporal_backbones,
    register_temporal_backbone,
)
from .phoneme_finetune import (
    POSSMFinetuneConfig,
    find_latest_possm_stage2_run_dir,
    recover_possm_stage1_encoder,
    recover_possm_stage1_sequence_components,
    recover_possm_stage2_summary,
    run_possm_phoneme_finetuning,
)
from .reporting import (
    display_possm_stage1_report,
    display_possm_stage2_report,
    display_possm_stage2_summary,
    run_possm_stage1_prediction_diagnostics,
    run_possm_stage2_prediction_diagnostics,
    summarize_possm_stage2_progress,
)
from .training import (
    POSSMTrainingConfig,
    build_possm_segment_sampler,
    find_latest_possm_step_checkpoint,
    list_possm_checkpoints,
    prune_possm_resumable_checkpoints,
    recover_possm_run_state_from_checkpoint,
    resolve_latest_possm_checkpoint_path,
    resolve_possm_checkpoint_path,
    resume_possm_training,
    run_possm_training,
)

__all__ = [
    "CacheAccessConfig",
    "CacheContext",
    "POSSMEncoder",
    "POSSMFinetuneConfig",
    "POSSMPhonemeModel",
    "POSSMReconstructionModel",
    "POSSMTrainingConfig",
    "SessionInputAdapterBank",
    "build_possm_segment_sampler",
    "display_possm_stage1_report",
    "display_possm_stage2_report",
    "display_possm_stage2_summary",
    "find_latest_possm_stage2_run_dir",
    "find_latest_possm_step_checkpoint",
    "list_possm_checkpoints",
    "list_registered_temporal_backbones",
    "load_precomputed_session_feature_stats_into_cache_context",
    "prepare_cache_context",
    "prune_possm_resumable_checkpoints",
    "recover_possm_run_state_from_checkpoint",
    "recover_possm_stage1_encoder",
    "recover_possm_stage1_sequence_components",
    "recover_possm_stage2_summary",
    "resolve_latest_possm_checkpoint_path",
    "resolve_possm_checkpoint_path",
    "resume_possm_training",
    "run_possm_stage1_prediction_diagnostics",
    "run_possm_stage2_prediction_diagnostics",
    "run_possm_phoneme_finetuning",
    "run_possm_training",
    "summarize_possm_stage2_progress",
    "causal_conv_output_lengths",
    "register_temporal_backbone",
]

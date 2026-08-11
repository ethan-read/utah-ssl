"""Paper-derived POSSM-style reconstruction and transfer experiments."""

from utah_ssl.experiment_contract import (
    DatasetPlan,
    ExperimentRecipe,
    SignalSpec,
)
from utah_ssl.feature_contract import (
    FeatureContract,
    SUPPORTED_FEATURE_MODES,
    resolve_feature_contract,
)
from utah_ssl.cache import (
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
from .recipes import (
    POSSM_B2T24_B2T25_SBP,
    POSSM_B2T24_SBP,
    POSSM_BROAD_TX,
    POSSM_RECIPES,
    get_possm_recipe,
    possm_single_dataset_plan,
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
    "DatasetPlan",
    "ExperimentRecipe",
    "FeatureContract",
    "POSSMEncoder",
    "POSSMFinetuneConfig",
    "POSSMPhonemeModel",
    "POSSMReconstructionModel",
    "POSSMTrainingConfig",
    "POSSM_B2T24_B2T25_SBP",
    "POSSM_B2T24_SBP",
    "POSSM_BROAD_TX",
    "POSSM_RECIPES",
    "SessionInputAdapterBank",
    "SignalSpec",
    "SUPPORTED_FEATURE_MODES",
    "build_possm_segment_sampler",
    "display_possm_stage1_report",
    "display_possm_stage2_report",
    "display_possm_stage2_summary",
    "find_latest_possm_stage2_run_dir",
    "find_latest_possm_step_checkpoint",
    "get_possm_recipe",
    "possm_single_dataset_plan",
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
    "resolve_feature_contract",
    "resume_possm_training",
    "run_possm_stage1_prediction_diagnostics",
    "run_possm_stage2_prediction_diagnostics",
    "run_possm_phoneme_finetuning",
    "run_possm_training",
    "summarize_possm_stage2_progress",
    "causal_conv_output_lengths",
    "register_temporal_backbone",
]

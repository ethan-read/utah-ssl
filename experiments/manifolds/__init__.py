"""Time-resolved trajectory analyses for Willett representation exports."""

from .analysis import (
    AlignedEvent,
    ctc_forced_align,
    extract_aligned_events,
    fit_shared_pca,
    split_half_reliability,
    trajectory_separation,
)
from .io import load_representation_export
from .bigram_trajectories import (
    BigramTrajectoryConfig,
    analyze_bigram_event_set,
    build_bigram_event_set,
    make_bigram_trajectory_figures,
    prepare_bigram_sources,
    run_bigram_trajectory_analysis,
    save_bigram_trajectory_result,
)
from .robustness import (
    RobustnessConfig,
    build_reconstructed_event_set,
    generate_unique_session_splits,
    reconstruct_overlapping_bins,
    run_robustness_analysis,
    save_robustness_result,
)

__all__ = [
    "AlignedEvent",
    "BigramTrajectoryConfig",
    "analyze_bigram_event_set",
    "build_bigram_event_set",
    "ctc_forced_align",
    "extract_aligned_events",
    "fit_shared_pca",
    "load_representation_export",
    "make_bigram_trajectory_figures",
    "prepare_bigram_sources",
    "RobustnessConfig",
    "build_reconstructed_event_set",
    "generate_unique_session_splits",
    "reconstruct_overlapping_bins",
    "run_robustness_analysis",
    "run_bigram_trajectory_analysis",
    "save_bigram_trajectory_result",
    "save_robustness_result",
    "split_half_reliability",
    "trajectory_separation",
]

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
    "ctc_forced_align",
    "extract_aligned_events",
    "fit_shared_pca",
    "load_representation_export",
    "RobustnessConfig",
    "build_reconstructed_event_set",
    "generate_unique_session_splits",
    "reconstruct_overlapping_bins",
    "run_robustness_analysis",
    "save_robustness_result",
    "split_half_reliability",
    "trajectory_separation",
]

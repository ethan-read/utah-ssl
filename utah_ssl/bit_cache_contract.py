"""Named dataset and signal constants for the broad BIT stage-1 recipe."""

from __future__ import annotations

from types import MappingProxyType

BIT_STAGE1_BOUNDARY_KEY_MODE = "session"
BIT_STAGE1_TX_DIM = 256
BIT_STAGE1_SIGMA_BINS = 2.0

BIT_STAGE1_DATASET_SPLITS = MappingProxyType(
    {
        "000950": ("held-in-calib", "held-in-minival", "held-out-calib"),
        "brain2text24": ("competition_train", "none"),
        "motor_data": ("eval", "none"),
        "plug_n_play": (
            "no_recalibration",
            "recalibration",
            "seed_model_training",
        ),
        "unsupervised_cursor_recalibration_offline": ("historical", "new"),
        "unsupervised_cursor_recalibration_online": ("one_month_recal",),
        "willett_handwriting": ("none",),
    }
)
BIT_STAGE1_DATASETS = tuple(BIT_STAGE1_DATASET_SPLITS)


__all__ = [
    "BIT_STAGE1_BOUNDARY_KEY_MODE",
    "BIT_STAGE1_DATASET_SPLITS",
    "BIT_STAGE1_DATASETS",
    "BIT_STAGE1_SIGMA_BINS",
    "BIT_STAGE1_TX_DIM",
]

"""Named POSSM data/signal recipes.

Recipes define scientific data boundaries. Model hyperparameters remain in the
training configs so changing an architecture cannot silently change the data.
"""

from __future__ import annotations

from utah_ssl.bit_cache_contract import BIT_STAGE1_DATASET_SPLITS
from utah_ssl.experiment_contract import DatasetPlan, ExperimentRecipe, SignalSpec


POSSM_B2T24_SBP = ExperimentRecipe(
    name="possm_b2t24_sbp",
    description="Brain2Text24 competition-train pretraining using area-6v SBP.",
    dataset_plan=DatasetPlan.from_mapping(
        {"brain2text24": ("competition_train",)}
    ),
    signal_spec=SignalSpec.sbp_only(sbp_dim=128),
)

POSSM_B2T24_B2T25_SBP = ExperimentRecipe(
    name="possm_b2t24_b2t25_sbp",
    description=(
        "Speech-focused pooled pretraining using Brain2Text24 competition-train "
        "and Brain2Text25 train+val area-6v SBP."
    ),
    dataset_plan=DatasetPlan.from_mapping(
        {
            "brain2text24": ("competition_train",),
            "brain2text25": ("train", "val"),
        }
    ),
    signal_spec=SignalSpec.sbp_only(sbp_dim=128),
    settings={"segment_bins": 100, "dataset_weight_alpha": 0.25},
)

POSSM_BROAD_TX = ExperimentRecipe(
    name="possm_broad_tx",
    description=(
        "Broad heterogeneous pretraining using the TX signal available across "
        "the canonical BIT dataset inventory. Narrower datasets are zero-padded "
        "to the shared 256-channel model width."
    ),
    dataset_plan=DatasetPlan.from_mapping(BIT_STAGE1_DATASET_SPLITS),
    signal_spec=SignalSpec.tx_only(
        tx_dim=256,
        missing_channel_policy="zero_pad",
    ),
)

POSSM_RECIPES = {
    recipe.name: recipe
    for recipe in (
        POSSM_B2T24_SBP,
        POSSM_B2T24_B2T25_SBP,
        POSSM_BROAD_TX,
    )
}


def possm_single_dataset_plan(dataset: str) -> DatasetPlan:
    """Return the leakage-safe Stage-1 split plan for a supported dataset."""
    source_splits_by_dataset = {
        "brain2text24": ("competition_train",),
        "brain2text25": ("train", "val"),
    }
    dataset_name = str(dataset)
    try:
        source_splits = source_splits_by_dataset[dataset_name]
    except KeyError as exc:
        raise ValueError(
            f"No default POSSM Stage-1 source-split policy for {dataset_name!r}. "
            f"Choose a named recipe or add an explicit policy."
        ) from exc
    return DatasetPlan.from_mapping({dataset_name: source_splits})


def get_possm_recipe(name: str) -> ExperimentRecipe:
    try:
        return POSSM_RECIPES[str(name)]
    except KeyError as exc:
        raise ValueError(
            f"Unknown POSSM recipe {name!r}; choose one of {tuple(POSSM_RECIPES)}"
        ) from exc


__all__ = [
    "POSSM_B2T24_B2T25_SBP",
    "POSSM_B2T24_SBP",
    "POSSM_BROAD_TX",
    "POSSM_RECIPES",
    "get_possm_recipe",
    "possm_single_dataset_plan",
]

"""Current in-scope search space for the full SSL autoresearch scaffold."""

from __future__ import annotations


SEARCH_SPACE = {
    "dataset_family": ["brain2text25"],
    "backbone": ["s5"],
    "objective_family": ["future_prediction"],
    "adaptation_regime": ["A", "B1", "B2"],
    "patch_size": [1, 3, 5],
    "patch_stride": [1, 3, 5],
    "standardize_scope": ["subject", "session"],
    "post_proj_norm": ["none", "rms"],
}


def is_valid_config(config: dict[str, object]) -> bool:
    patch_size = int(config["patch_size"])
    patch_stride = int(config["patch_stride"])
    if patch_stride > patch_size:
        return False
    if config["backbone"] not in SEARCH_SPACE["backbone"]:
        return False
    if config["objective_family"] not in SEARCH_SPACE["objective_family"]:
        return False
    if config["adaptation_regime"] not in SEARCH_SPACE["adaptation_regime"]:
        return False
    if config["standardize_scope"] not in SEARCH_SPACE["standardize_scope"]:
        return False
    if config["post_proj_norm"] not in SEARCH_SPACE["post_proj_norm"]:
        return False
    return True

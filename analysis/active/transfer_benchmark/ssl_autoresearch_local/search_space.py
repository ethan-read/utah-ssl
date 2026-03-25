"""Local smoke-test search space for ssl_autoresearch_local."""

from __future__ import annotations


SEARCH_SPACE = {
    "patch_size": [1, 3, 5],
    "patch_stride": [1, 3, 5],
    "hidden_size": [64, 128],
    "num_layers": [1, 2],
    "learning_rate": [3e-4, 1e-3],
    "batch_size": [16, 32],
    "standardize_scope": ["subject", "session"],
    "post_proj_norm": ["none", "rms"],
}


def is_valid_config(config: dict[str, object]) -> bool:
    patch_size = int(config["patch_size"])
    patch_stride = int(config["patch_stride"])
    return patch_stride <= patch_size

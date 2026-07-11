"""Load durable arrays produced by ``willett_reconstruction`` exports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def load_representation_export(
    model_dir: str | Path,
    *,
    representation: str = "hidden",
) -> dict[str, Any]:
    """Load one model export while preserving its per-example time ordering."""
    model_dir = Path(model_dir)
    metadata = json.loads((model_dir / "metadata.json").read_text())
    tokens = pd.read_csv(model_dir / "tokens.csv")
    examples = pd.read_csv(model_dir / "examples.csv")
    shards = json.loads((model_dir / "shards.json").read_text())

    representation_parts: list[np.ndarray] = []
    logit_parts: list[np.ndarray] = []
    example_index_parts: list[np.ndarray] = []
    token_index_parts: list[np.ndarray] = []
    for shard in shards:
        with np.load(model_dir / "shards" / shard["shard"]) as arrays:
            if representation not in arrays:
                available = ", ".join(sorted(arrays.files))
                raise KeyError(
                    f"Representation {representation!r} is absent from {shard['shard']}; "
                    f"available arrays: {available}"
                )
            representation_parts.append(np.asarray(arrays[representation], dtype=np.float32))
            logit_parts.append(np.asarray(arrays["logits"], dtype=np.float32))
            example_index_parts.append(np.asarray(arrays["token_example_index"], dtype=np.int64))
            token_index_parts.append(np.asarray(arrays["token_index"], dtype=np.int64))

    def concatenate(parts: list[np.ndarray], *, shape: tuple[int, ...], dtype: np.dtype) -> np.ndarray:
        return np.concatenate(parts, axis=0) if parts else np.zeros(shape, dtype=dtype)

    values = concatenate(representation_parts, shape=(0, 0), dtype=np.float32)
    logits = concatenate(logit_parts, shape=(0, 0), dtype=np.float32)
    example_indices = concatenate(example_index_parts, shape=(0,), dtype=np.int64)
    token_indices = concatenate(token_index_parts, shape=(0,), dtype=np.int64)
    if not (len(values) == len(logits) == len(tokens) == len(example_indices) == len(token_indices)):
        raise ValueError("Export tables and shard arrays have inconsistent token counts")
    if len(tokens):
        if not np.array_equal(tokens["example_export_index"].to_numpy(), example_indices):
            raise ValueError("tokens.csv example indices disagree with shard arrays")
        if not np.array_equal(tokens["token_index"].to_numpy(), token_indices):
            raise ValueError("tokens.csv token indices disagree with shard arrays")
    return {
        "model_dir": model_dir,
        "metadata": metadata,
        "tokens": tokens,
        "examples": examples,
        "values": values,
        "logits": logits,
        "example_indices": example_indices,
        "token_indices": token_indices,
        "representation": representation,
    }

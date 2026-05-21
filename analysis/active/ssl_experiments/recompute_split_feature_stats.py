"""Recompute train-split global feature stats from a canonical cache root."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from masked_ssl.cache import _cache_variant_name, _canonical_stats_root_for_cache
from masked_ssl.probe import build_competition_split_problem, compute_feature_stats


DEFAULT_DATASET = "brain2text24"
DEFAULT_FEATURE_MODE = "tx_only"
DEFAULT_BOUNDARY_KEY_MODE = "session"
DEFAULT_SPLIT_NAME = "competition_train"


def _timestamp_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _default_output_path(
    *,
    cache_root: Path,
    dataset: str,
    train_split_name: str,
    feature_mode: str,
) -> Path:
    return (
        _canonical_stats_root_for_cache(cache_root)
        / "split_feature_stats"
        / _cache_variant_name(cache_root)
        / str(dataset)
        / str(train_split_name)
        / str(feature_mode)
        / "global_v1.pt"
    )


def recompute_split_feature_stats(
    *,
    cache_root: str | Path,
    output_path: str | Path | None = None,
    dataset: str = DEFAULT_DATASET,
    feature_mode: str = DEFAULT_FEATURE_MODE,
    boundary_key_mode: str = DEFAULT_BOUNDARY_KEY_MODE,
    overwrite: bool = False,
) -> dict[str, Any]:
    cache_root = Path(cache_root)
    if not cache_root.is_dir():
        raise FileNotFoundError(f"Cache root does not exist: {cache_root}")

    problem = build_competition_split_problem(
        cache_root=cache_root,
        dataset=str(dataset),
        feature_mode=str(feature_mode),
        boundary_key_mode=str(boundary_key_mode),
    )
    train_split_name = str(problem.get("train_split_name", DEFAULT_SPLIT_NAME))
    val_split_name = str(problem.get("val_split_name", "competition_test"))
    resolved_output_path = (
        Path(output_path)
        if output_path is not None
        else _default_output_path(
            cache_root=cache_root,
            dataset=str(dataset),
            train_split_name=train_split_name,
            feature_mode=str(feature_mode),
        )
    )
    metadata_path = resolved_output_path.with_suffix(".json")
    if (resolved_output_path.exists() or metadata_path.exists()) and not overwrite:
        raise FileExistsError(
            f"Output already exists: {resolved_output_path} or {metadata_path}. "
            "Pass overwrite=True to replace existing artifacts."
        )

    stats = compute_feature_stats(
        problem["train_rows"],
        cache_root=cache_root,
        mode="global",
        feature_mode=str(feature_mode),
    )
    if not isinstance(stats, tuple) or len(stats) != 2:
        raise TypeError("Expected compute_feature_stats(..., mode='global') to return (mean, std).")
    mean, std = stats

    metadata: dict[str, Any] = {
        "kind": "split_feature_stats",
        "created_utc": _timestamp_utc(),
        "source_cache_root": str(cache_root.resolve()),
        "source_cache_name": cache_root.name,
        "source_cache_variant": _cache_variant_name(cache_root),
        "dataset": str(dataset),
        "feature_mode": str(feature_mode),
        "boundary_key_mode": str(boundary_key_mode),
        "split_policy": str(problem.get("split_policy", "competition_train_test")),
        "train_split_name": train_split_name,
        "val_split_name": val_split_name,
        "train_examples": int(len(problem["train_rows"])),
        "val_examples": int(len(problem["val_rows"])),
        "train_session_ids": list(problem.get("train_session_ids", ())),
        "val_session_ids": list(problem.get("val_session_ids", ())),
        "feature_dim": int(mean.shape[0]),
    }

    payload = {
        "mean": mean,
        "std": std,
        "metadata": metadata,
    }
    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, resolved_output_path)
    metadata_path.write_text(json.dumps(metadata, indent=2))

    return {
        "output_path": resolved_output_path,
        "metadata_path": metadata_path,
        "metadata": metadata,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, required=True, help="Canonical cache root.")
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Destination .pt file. Defaults to the canonical split_feature_stats path.",
    )
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument(
        "--feature-mode",
        choices=("tx_only", "tx_sbp"),
        default=DEFAULT_FEATURE_MODE,
        help="Feature layout to match when computing stats.",
    )
    parser.add_argument(
        "--boundary-key-mode",
        choices=("session", "subject_if_available"),
        default=DEFAULT_BOUNDARY_KEY_MODE,
        help="Boundary-key mode used when building the competition split.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output file.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = recompute_split_feature_stats(
        cache_root=args.cache_root,
        output_path=args.output_path,
        dataset=str(args.dataset),
        feature_mode=str(args.feature_mode),
        boundary_key_mode=str(args.boundary_key_mode),
        overwrite=bool(args.overwrite),
    )
    print(
        json.dumps(
            {key: (str(value) if isinstance(value, Path) else value) for key, value in result.items()},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

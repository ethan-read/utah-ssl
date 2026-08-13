#!/usr/bin/env python
"""Evaluate the converted released Stanford/Willett RNN checkpoint on validation PER."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

EXPERIMENTS_DIR = Path(__file__).resolve().parents[2]
REPO_ROOT = Path(__file__).resolve().parents[3]
for path in (REPO_ROOT, EXPERIMENTS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiments.supervised_baselines.checkpointing import (  # noqa: E402
    config_from_checkpoint,
    load_willett_model_from_checkpoint,
)
from experiments.supervised_baselines.data import (  # noqa: E402
    CanonicalSequenceDataset,
    build_willett_problem,
    loader_kwargs,
    make_length_aware_batch_sampler,
)
from experiments.supervised_baselines.released_tf_checkpoint import (
    RELEASED_SESSIONS,  # noqa: E402
)
from experiments.supervised_baselines.reporting import (
    evaluate_willett_phoneme_metrics,  # noqa: E402
)
from utah_ssl.decoding_preprocessing import (  # noqa: E402
    willett_input_transform_config_from,
)
from utah_ssl.runtime import resolve_device  # noqa: E402


def _selected_val_rows(problem: dict[str, Any], *, released_sessions_only: bool, max_examples: int | None) -> tuple[Any, ...]:
    rows = tuple(problem["val_rows"])
    if released_sessions_only:
        allowed = set(RELEASED_SESSIONS)
        rows = tuple(row for row in rows if str(row.session_id) in allowed)
    if max_examples is not None:
        rows = rows[: int(max_examples)]
    if not rows:
        raise ValueError("No validation rows selected.")
    return rows


def evaluate_checkpoint(
    *,
    checkpoint_path: Path,
    cache_root: Path | None,
    batch_size: int,
    device_name: str | None,
    max_examples: int | None,
    released_sessions_only: bool,
) -> dict[str, Any]:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = config_from_checkpoint(payload)
    if cache_root is not None:
        config = replace(config, cache_root=cache_root)

    problem = build_willett_problem(
        cache_root=Path(config.cache_root),
        dataset=str(config.dataset),
        feature_mode=str(config.feature_mode),
        boundary_key_mode=str(config.boundary_key_mode),
        split_policy=str(config.split_policy),
        cv_num_folds=int(config.cv_num_folds),
        cv_fold_index=int(config.cv_fold_index),
    )
    rows = _selected_val_rows(
        problem,
        released_sessions_only=bool(released_sessions_only),
        max_examples=max_examples,
    )
    stats = None if str(config.normalization_mode) == "none" else None
    if str(config.normalization_mode) != "none":
        raise ValueError(
            "This compact released-checkpoint evaluator currently expects normalization_mode='none'. "
            f"Got {config.normalization_mode!r}."
        )

    dataset = CanonicalSequenceDataset(
        rows,
        cache_root=Path(problem["cache_root"]),
        signal_spec=problem["signal_spec"],
        stats=stats,
        boundary_key_mode=str(problem["boundary_key_mode"]),
        dataset=str(problem["dataset"]),
    )
    device = resolve_device(device_name)
    loader = DataLoader(
        dataset,
        batch_sampler=make_length_aware_batch_sampler(
            rows,
            batch_size=int(batch_size),
            shuffle=False,
            seed=int(config.seed) + 1,
        ),
        **loader_kwargs(device),
    )
    model, _, _ = load_willett_model_from_checkpoint(
        payload,
        config=config,
        problem=problem,
        input_dim=256,
        vocab_size=int(problem["vocab"]["num_classes"]),
        device=device,
    )

    metrics = evaluate_willett_phoneme_metrics(
        model=model,
        loader=loader,
        device=device,
        blank_index=int(problem["vocab"]["blank_index"]),
        input_transform_config=willett_input_transform_config_from(config),
    )
    return {
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_step": int(payload.get("step", payload.get("steps", -1))),
        "cache_root": str(config.cache_root),
        "dataset": str(problem["dataset"]),
        "feature_mode": str(problem["feature_mode"]),
        "split": str(problem["val_split_name"]),
        "released_sessions_only": bool(released_sessions_only),
        "val_examples": int(len(rows)),
        "val_sessions": sorted({str(row.session_id) for row in rows}),
        "device": str(device),
        "metrics": metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("/content/drive/MyDrive/utah_ssl/outputs/willett_reconstruction/stanford_released_baseline_rnn/checkpoint_released_ckpt9950.pt"),
        help="Converted PyTorch checkpoint produced by convert_released_tf_checkpoint_to_pytorch.",
    )
    parser.add_argument("--cache-root", type=Path, default=None, help="Override cache root from the checkpoint config.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default=None, help="Optional torch device, e.g. cuda, mps, or cpu.")
    parser.add_argument("--max-examples", type=int, default=None, help="Optional smoke-test cap.")
    parser.add_argument(
        "--include-unreleased-sessions",
        action="store_true",
        help="Evaluate all local validation sessions instead of only sessions present in the released checkpoint.",
    )
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    summary = evaluate_checkpoint(
        checkpoint_path=Path(args.checkpoint),
        cache_root=args.cache_root,
        batch_size=int(args.batch_size),
        device_name=args.device,
        max_examples=args.max_examples,
        released_sessions_only=not bool(args.include_unreleased_sessions),
    )
    print(json.dumps(summary, indent=2))
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()

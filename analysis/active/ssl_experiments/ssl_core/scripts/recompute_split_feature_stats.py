"""Recompute train-split global feature stats from a canonical cache root."""

from __future__ import annotations

import argparse
import json
import shlex
from datetime import datetime, timezone
from pathlib import Path
import sys

EXPERIMENTS_DIR = Path(__file__).resolve().parents[2]
REPO_ROOT = Path(__file__).resolve().parents[5]
for _path in (REPO_ROOT, EXPERIMENTS_DIR):
    _path_str = str(_path)
    if _path_str not in sys.path:
        sys.path.insert(0, _path_str)
from typing import Any

import torch

from ssl_core.feature_contract import SUPPORTED_FEATURE_MODES
from ssl_core.experiment_contract import SignalSpec

try:
    from masked_ssl.cache import (
        _cache_variant_name,
        _canonical_stats_root_for_cache,
        _compute_dataset_cache_source_signature,
        _load_artifact_payload_and_sidecar,
        _validate_common_artifact_metadata,
    )
    from masked_ssl.probe import (
        build_competition_split_problem,
        build_source_split_problem,
        compute_feature_stats,
    )
except ModuleNotFoundError:  # pragma: no cover - repo-root unittest fallback
    from analysis.active.ssl_experiments.masked_ssl.cache import (
        _cache_variant_name,
        _canonical_stats_root_for_cache,
        _compute_dataset_cache_source_signature,
        _load_artifact_payload_and_sidecar,
        _validate_common_artifact_metadata,
    )
    from analysis.active.ssl_experiments.masked_ssl.probe import (
        build_competition_split_problem,
        build_source_split_problem,
        compute_feature_stats,
    )


DEFAULT_DATASET = "brain2text24"
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


def build_recompute_split_feature_stats_command(
    *,
    cache_root: str | Path,
    dataset: str,
    signal_spec: SignalSpec,
    boundary_key_mode: str,
    split_policy: str,
    output_path: str | Path,
) -> str:
    resolved_signal_spec = SignalSpec.from_value(signal_spec)
    return shlex.join(
        [
            "python",
            "analysis/active/ssl_experiments/ssl_core/scripts/recompute_split_feature_stats.py",
            "--cache-root",
            str(Path(cache_root)),
            "--dataset",
            str(dataset),
            "--feature-mode",
            resolved_signal_spec.mode,
            "--tx-dim",
            str(resolved_signal_spec.tx_dim),
            "--sbp-dim",
            str(resolved_signal_spec.sbp_dim),
            "--column-start",
            str(resolved_signal_spec.column_start),
            "--missing-channel-policy",
            str(resolved_signal_spec.missing_channel_policy),
            "--boundary-key-mode",
            str(boundary_key_mode),
            "--split-policy",
            str(split_policy),
            "--output-path",
            str(Path(output_path)),
            "--overwrite",
        ]
    )


def resolve_precomputed_split_stats_path(
    *,
    cache_root: str | Path,
    dataset: str,
    train_split_name: str,
    signal_spec: SignalSpec,
    preferred_path: str | Path | None,
) -> Path:
    resolved_signal_spec = SignalSpec.from_value(signal_spec)
    if preferred_path is not None:
        return Path(preferred_path)
    return _default_output_path(
        cache_root=Path(cache_root),
        dataset=str(dataset),
        train_split_name=str(train_split_name),
        feature_mode=resolved_signal_spec.mode,
    )


def load_precomputed_split_feature_stats(
    *,
    stats_path: str | Path,
    cache_root: str | Path,
    dataset: str,
    signal_spec: SignalSpec,
    boundary_key_mode: str,
    train_split_name: str,
    val_split_name: str,
    split_policy: str = "competition_train_test",
) -> tuple[tuple[torch.Tensor, torch.Tensor], dict[str, Any], Path]:
    resolved_signal_spec = SignalSpec.from_value(signal_spec)
    feature_mode = resolved_signal_spec.mode
    expected_dim = resolved_signal_spec.full_dim
    path = Path(stats_path)
    canonical_path = _default_output_path(
        cache_root=Path(cache_root),
        dataset=str(dataset),
        train_split_name=str(train_split_name),
        feature_mode=feature_mode,
    )
    recompute_cmd = build_recompute_split_feature_stats_command(
        cache_root=cache_root,
        dataset=str(dataset),
        signal_spec=resolved_signal_spec,
        boundary_key_mode=str(boundary_key_mode),
        split_policy=str(split_policy),
        output_path=canonical_path,
    )
    payload, metadata, path, _ = _load_artifact_payload_and_sidecar(
        path=path,
        canonical_path=canonical_path,
        recompute_cmd=recompute_cmd,
        artifact_name="split stats",
        expected_kind="split_feature_stats",
    )
    if "mean" not in payload or "std" not in payload:
        raise ValueError(
            "Precomputed split stats payload is missing mean/std tensors.\n"
            f"expected_path: {canonical_path}\n"
            f"requested_path: {path}\n"
            f"recompute_command: {recompute_cmd}"
        )
    mean_t = torch.as_tensor(payload.get("mean")).float().cpu()
    std_t = torch.as_tensor(payload.get("std")).float().cpu()

    expected_cache_root = str(Path(cache_root).resolve())
    expected_cache_variant = _cache_variant_name(cache_root)
    expected_cache_signature = _compute_dataset_cache_source_signature(
        {str(dataset): Path(cache_root)}
    )
    common_metadata = {
        "source_cache_root": expected_cache_root,
        "source_cache_variant": expected_cache_variant,
        "source_cache_signature": expected_cache_signature,
        "feature_mode": feature_mode,
        "signal_spec": resolved_signal_spec.to_dict(),
        "boundary_key_mode": str(boundary_key_mode),
        "split_policy": str(split_policy),
    }
    split_metadata = {
        "dataset": str(dataset),
        "train_split_name": str(train_split_name),
        "val_split_name": str(val_split_name),
        "feature_dim": int(expected_dim),
    }
    mismatches = _validate_common_artifact_metadata(
        metadata=metadata,
        expected_metadata=common_metadata,
    )
    mismatches.extend(
        _validate_common_artifact_metadata(
            metadata=metadata,
            expected_metadata=split_metadata,
        )
    )
    if mean_t.numel() != expected_dim or std_t.numel() != expected_dim:
        mismatches.append(
            f"tensor_dim=mean:{mean_t.numel()} std:{std_t.numel()} expected {expected_dim}"
        )
    if mismatches:
        mismatch_text = "; ".join(mismatches)
        raise ValueError(
            "Precomputed split stats artifact is stale or incompatible.\n"
            f"expected_path: {canonical_path}\n"
            f"requested_path: {path}\n"
            f"reason: {mismatch_text}\n"
            f"recompute_command: {recompute_cmd}"
        )
    return (mean_t, std_t), metadata, path


def recompute_split_feature_stats(
    *,
    cache_root: str | Path,
    output_path: str | Path | None = None,
    dataset: str = DEFAULT_DATASET,
    signal_spec: SignalSpec | dict[str, Any],
    boundary_key_mode: str = DEFAULT_BOUNDARY_KEY_MODE,
    split_policy: str = "competition_train_test",
    overwrite: bool = False,
) -> dict[str, Any]:
    cache_root = Path(cache_root)
    if not cache_root.is_dir():
        raise FileNotFoundError(f"Cache root does not exist: {cache_root}")

    signal_spec = SignalSpec.from_value(signal_spec)
    feature_mode = signal_spec.mode
    if str(split_policy) == "competition_train_test":
        problem = build_competition_split_problem(
            cache_root=cache_root,
            dataset=str(dataset),
            signal_spec=signal_spec,
            boundary_key_mode=str(boundary_key_mode),
        )
    elif str(split_policy) == "source_train_val":
        problem = build_source_split_problem(
            cache_root=cache_root,
            dataset=str(dataset),
            signal_spec=signal_spec,
            boundary_key_mode=str(boundary_key_mode),
            train_split_name="train",
            val_split_name="val",
        )
        problem = {**problem, "split_policy": "source_train_val"}
    else:
        raise ValueError("split_policy must be one of {'competition_train_test', 'source_train_val'}")
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
        signal_spec=signal_spec,
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
        "source_cache_signature": _compute_dataset_cache_source_signature(
            {str(dataset): cache_root}
        ),
        "dataset": str(dataset),
        "feature_mode": str(feature_mode),
        "signal_spec": signal_spec.to_dict(),
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
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")

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
        choices=SUPPORTED_FEATURE_MODES,
        required=True,
        help="Feature layout to match when computing stats.",
    )
    parser.add_argument("--tx-dim", type=int, default=None)
    parser.add_argument("--sbp-dim", type=int, default=None)
    parser.add_argument("--column-start", type=int, default=0)
    parser.add_argument(
        "--missing-channel-policy",
        choices=("error", "zero_pad"),
        default="error",
    )
    parser.add_argument(
        "--boundary-key-mode",
        choices=("session", "subject_if_available"),
        default=DEFAULT_BOUNDARY_KEY_MODE,
        help="Boundary-key mode used when building the competition split.",
    )
    parser.add_argument(
        "--split-policy",
        choices=("competition_train_test", "source_train_val"),
        default="competition_train_test",
        help="Source split policy used when computing train-split stats.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output file.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata = json.loads(
        (Path(args.cache_root) / str(args.dataset) / "metadata.json").read_text()
    )
    feature_layout = dict(metadata.get("feature_layout") or {})
    tx_dim = (
        int(args.tx_dim)
        if args.tx_dim is not None
        else int(metadata.get("n_tx_features", feature_layout.get("n_tx_features", 0)))
    )
    sbp_dim = (
        int(args.sbp_dim)
        if args.sbp_dim is not None
        else int(metadata.get("n_sbp_features", feature_layout.get("n_sbp_features", 0)))
    )
    result = recompute_split_feature_stats(
        cache_root=args.cache_root,
        output_path=args.output_path,
        dataset=str(args.dataset),
        signal_spec=SignalSpec.from_mode(
            str(args.feature_mode),
            tx_dim=tx_dim,
            sbp_dim=sbp_dim,
            column_start=int(args.column_start),
            missing_channel_policy=str(args.missing_channel_policy),
        ),
        boundary_key_mode=str(args.boundary_key_mode),
        split_policy=str(args.split_policy),
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

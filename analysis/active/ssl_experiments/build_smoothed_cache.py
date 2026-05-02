"""Build a sibling pre-smoothed cache root from a canonical Utah SSL cache."""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from masked_ssl.cache import _apply_gaussian_smoothing


SMOOTHED_CACHE_IMPL = "masked_ssl.cache._apply_gaussian_smoothing"


def _timestamp_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _iter_dataset_names(src_root: Path, requested_datasets: Sequence[str] | None) -> list[str]:
    if requested_datasets:
        dataset_names = [str(name) for name in requested_datasets]
        for dataset in dataset_names:
            dataset_root = src_root / dataset
            if not dataset_root.is_dir():
                raise FileNotFoundError(f"Dataset not found under source cache: {dataset_root}")
        return dataset_names
    return sorted(path.name for path in src_root.iterdir() if path.is_dir())


def _iter_shard_dirs(dataset_root: Path) -> list[Path]:
    shard_parent = dataset_root / "shards"
    if shard_parent.is_dir():
        return sorted(path for path in shard_parent.iterdir() if path.is_dir())
    return sorted(
        path
        for path in dataset_root.iterdir()
        if path.is_dir() and path.name != "shards"
    )


def smooth_feature_array(array: np.ndarray, *, sigma_bins: float) -> np.ndarray:
    if array.ndim != 2:
        raise ValueError(f"Expected a rank-2 feature array, got shape {tuple(array.shape)}")
    if array.size == 0 or float(sigma_bins) <= 0.0:
        return np.array(array, copy=True)
    tensor = torch.from_numpy(np.asarray(array, dtype=np.float32))
    feature_mask = torch.ones(int(tensor.shape[1]), dtype=torch.float32)
    smoothed = _apply_gaussian_smoothing(tensor, feature_mask, sigma_bins=float(sigma_bins))
    return smoothed.cpu().numpy().astype(array.dtype, copy=False)


def _copy_or_smooth_shard(
    *,
    src_shard_dir: Path,
    dst_shard_dir: Path,
    sigma_bins: float,
    dry_run: bool,
) -> dict[str, int]:
    counters = {"files_copied": 0, "feature_files_smoothed": 0}
    if not dry_run:
        dst_shard_dir.mkdir(parents=True, exist_ok=True)
    for src_path in sorted(src_shard_dir.iterdir(), key=lambda item: item.name):
        if not src_path.is_file():
            continue
        dst_path = dst_shard_dir / src_path.name
        if src_path.name in {"tx.npy", "sbp.npy"}:
            counters["feature_files_smoothed"] += 1
            if not dry_run:
                array = np.load(src_path)
                np.save(dst_path, smooth_feature_array(array, sigma_bins=sigma_bins))
            continue
        counters["files_copied"] += 1
        if not dry_run:
            shutil.copy2(src_path, dst_path)
    return counters


def build_smoothed_cache(
    *,
    src_root: str | Path,
    dst_root: str | Path,
    sigma_bins: float,
    datasets: Sequence[str] | None = None,
    overwrite: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    src_root = Path(src_root)
    dst_root = Path(dst_root)
    sigma_bins = float(sigma_bins)
    if sigma_bins <= 0.0:
        raise ValueError("sigma_bins must be positive for a smoothed cache build.")
    if not src_root.is_dir():
        raise FileNotFoundError(f"Source cache root does not exist: {src_root}")
    if src_root.resolve() == dst_root.resolve():
        raise ValueError("Destination cache root must be different from the source cache root.")

    dataset_names = _iter_dataset_names(src_root, datasets)
    if dst_root.exists() and not overwrite and not dry_run:
        raise FileExistsError(
            f"Destination cache root already exists: {dst_root}. Pass overwrite=True to replace it."
        )
    if dst_root.exists() and overwrite and not dry_run:
        shutil.rmtree(dst_root)
    if not dry_run:
        dst_root.mkdir(parents=True, exist_ok=True)

    root_file_count = 0
    for src_path in sorted(src_root.iterdir(), key=lambda item: item.name):
        if not src_path.is_file():
            continue
        root_file_count += 1
        if not dry_run:
            shutil.copy2(src_path, dst_root / src_path.name)

    summary: dict[str, Any] = {
        "src_root": str(src_root),
        "dst_root": str(dst_root),
        "sigma_bins": sigma_bins,
        "datasets": {},
        "root_files_copied": root_file_count,
        "dry_run": bool(dry_run),
    }
    for dataset in dataset_names:
        src_dataset_root = src_root / dataset
        dst_dataset_root = dst_root / dataset
        shard_dirs = _iter_shard_dirs(src_dataset_root)
        manifest_path = src_dataset_root / "manifest.jsonl"
        metadata_path = src_dataset_root / "metadata.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Missing manifest for dataset {dataset}: {manifest_path}")
        if not metadata_path.exists():
            raise FileNotFoundError(f"Missing metadata for dataset {dataset}: {metadata_path}")

        if not dry_run:
            dst_dataset_root.mkdir(parents=True, exist_ok=True)
            shutil.copy2(manifest_path, dst_dataset_root / "manifest.jsonl")

        metadata = json.loads(metadata_path.read_text())
        metadata["smoothing_provenance"] = {
            "source_cache_root": str(src_root.resolve()),
            "source_cache_name": src_root.name,
            "sigma_bins": sigma_bins,
            "implementation": SMOOTHED_CACHE_IMPL,
            "created_utc": _timestamp_utc(),
        }
        if not dry_run:
            (dst_dataset_root / "metadata.json").write_text(json.dumps(metadata, indent=2))

        files_copied = 0
        feature_files_smoothed = 0
        for src_shard_dir in shard_dirs:
            relative_shard_dir = src_shard_dir.relative_to(src_dataset_root)
            dst_shard_dir = dst_dataset_root / relative_shard_dir
            counts = _copy_or_smooth_shard(
                src_shard_dir=src_shard_dir,
                dst_shard_dir=dst_shard_dir,
                sigma_bins=sigma_bins,
                dry_run=dry_run,
            )
            files_copied += int(counts["files_copied"])
            feature_files_smoothed += int(counts["feature_files_smoothed"])

        summary["datasets"][dataset] = {
            "manifest_copied": True,
            "metadata_copied": True,
            "shard_count": len(shard_dirs),
            "files_copied": files_copied,
            "feature_files_smoothed": feature_files_smoothed,
        }
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", required=True, help="Source canonical cache root.")
    parser.add_argument("--dst", required=True, help="Destination smoothed cache root.")
    parser.add_argument("--sigma-bins", required=True, type=float, help="Gaussian sigma in bins.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Optional dataset names to smooth. Defaults to every dataset under --src.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing destination cache root.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report the planned work without writing files.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    summary = build_smoothed_cache(
        src_root=args.src,
        dst_root=args.dst,
        sigma_bins=float(args.sigma_bins),
        datasets=args.datasets,
        overwrite=bool(args.overwrite),
        dry_run=bool(args.dry_run),
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

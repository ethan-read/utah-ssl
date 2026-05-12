#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect the canonical brain2text24 cache and test whether channels are "
            "structurally present for every example or whether masking for missing "
            "units might be needed."
        )
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("data/cache_v1/brain2text24"),
        help="Path to the cached brain2text24 dataset root.",
    )
    parser.add_argument(
        "--chunk-rows",
        type=int,
        default=4096,
        help="Rows per chunk when scanning shard arrays.",
    )
    return parser.parse_args()


def iter_manifest_rows(manifest_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with manifest_path.open() as handle:
        for line in handle:
            rows.append(json.loads(line))
    return rows


def scan_dense_array(array: np.ndarray, *, chunk_rows: int) -> tuple[np.ndarray, np.ndarray]:
    if array.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {array.shape}")
    n_rows, n_features = array.shape
    any_nonzero = np.zeros(n_features, dtype=bool)
    any_nonfinite = np.zeros(n_features, dtype=bool)

    is_float = np.issubdtype(array.dtype, np.floating)
    for start in range(0, n_rows, chunk_rows):
        stop = min(start + chunk_rows, n_rows)
        chunk = np.asarray(array[start:stop])
        any_nonzero |= np.any(chunk != 0, axis=0)
        if is_float:
            any_nonfinite |= ~np.isfinite(chunk).all(axis=0)

    all_zero = ~any_nonzero
    return all_zero, any_nonfinite


def format_indices(indices: np.ndarray, *, limit: int = 16) -> str:
    if indices.size == 0:
        return "[]"
    trimmed = indices[:limit].tolist()
    suffix = "" if indices.size <= limit else f", ... (+{indices.size - limit} more)"
    return f"{trimmed}{suffix}"


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    metadata_path = dataset_root / "metadata.json"
    manifest_path = dataset_root / "manifest.jsonl"
    shard_root = dataset_root / "shards"

    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata file: {metadata_path}")
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest file: {manifest_path}")
    if not shard_root.exists():
        raise FileNotFoundError(f"Missing shard directory: {shard_root}")

    metadata = json.loads(metadata_path.read_text())
    manifest_rows = iter_manifest_rows(manifest_path)

    expected_tx = int(metadata["feature_layout"]["n_tx_features"])
    expected_sbp = int(metadata["feature_layout"]["n_sbp_features"])

    manifest_has_tx_false = sum(not bool(row.get("has_tx", False)) for row in manifest_rows)
    manifest_has_sbp_false = sum(not bool(row.get("has_sbp", False)) for row in manifest_rows)
    manifest_tx_dims = sorted({int(row.get("n_tx_features", -1)) for row in manifest_rows})
    manifest_sbp_dims = sorted({int(row.get("n_sbp_features", -1)) for row in manifest_rows})

    tx_any_nonzero_dataset = np.zeros(expected_tx, dtype=bool)
    sbp_any_nonzero_dataset = np.zeros(expected_sbp, dtype=bool)
    sbp_any_nonfinite_dataset = np.zeros(expected_sbp, dtype=bool)

    shard_tx_width_mismatches: list[str] = []
    shard_sbp_width_mismatches: list[str] = []
    shard_time_mismatches: list[str] = []
    shard_missing_files: list[str] = []
    shards_with_zero_tx: list[tuple[str, int]] = []
    shards_with_zero_sbp: list[tuple[str, int]] = []
    shards_with_nonfinite_sbp: list[tuple[str, int]] = []

    shard_rows = metadata.get("shards", [])
    for shard in shard_rows:
        shard_id = str(shard["shard_id"])
        shard_dir = shard_root / shard_id
        tx_path = shard_dir / "tx.npy"
        sbp_path = shard_dir / "sbp.npy"
        time_offsets_path = shard_dir / "time_offsets.npy"

        if not tx_path.exists() or not sbp_path.exists() or not time_offsets_path.exists():
            shard_missing_files.append(shard_id)
            continue

        tx = np.load(tx_path, mmap_mode="r")
        sbp = np.load(sbp_path, mmap_mode="r")
        time_offsets = np.load(time_offsets_path, mmap_mode="r")

        if tx.ndim != 2 or tx.shape[1] != expected_tx:
            shard_tx_width_mismatches.append(f"{shard_id}:{tx.shape}")
        if sbp.ndim != 2 or sbp.shape[1] != expected_sbp:
            shard_sbp_width_mismatches.append(f"{shard_id}:{sbp.shape}")

        expected_rows = int(time_offsets[-1])
        if tx.shape[0] != expected_rows or sbp.shape[0] != expected_rows:
            shard_time_mismatches.append(
                f"{shard_id}: tx_rows={tx.shape[0]} sbp_rows={sbp.shape[0]} offsets_end={expected_rows}"
            )

        tx_all_zero, _ = scan_dense_array(tx, chunk_rows=args.chunk_rows)
        sbp_all_zero, sbp_nonfinite = scan_dense_array(sbp, chunk_rows=args.chunk_rows)

        tx_any_nonzero_dataset |= ~tx_all_zero
        sbp_any_nonzero_dataset |= ~sbp_all_zero
        sbp_any_nonfinite_dataset |= sbp_nonfinite

        n_tx_all_zero = int(tx_all_zero.sum())
        n_sbp_all_zero = int(sbp_all_zero.sum())
        n_sbp_nonfinite = int(sbp_nonfinite.sum())
        if n_tx_all_zero:
            shards_with_zero_tx.append((shard_id, n_tx_all_zero))
        if n_sbp_all_zero:
            shards_with_zero_sbp.append((shard_id, n_sbp_all_zero))
        if n_sbp_nonfinite:
            shards_with_nonfinite_sbp.append((shard_id, n_sbp_nonfinite))

    dataset_tx_all_zero = np.flatnonzero(~tx_any_nonzero_dataset)
    dataset_sbp_all_zero = np.flatnonzero(~sbp_any_nonzero_dataset)
    dataset_sbp_nonfinite = np.flatnonzero(sbp_any_nonfinite_dataset)

    print("brain2text24 channel presence check")
    print(f"dataset_root: {dataset_root}")
    print(f"manifest_rows: {len(manifest_rows)}")
    print(f"metadata_num_shards: {len(shard_rows)}")
    print()
    print("manifest consistency")
    print(f"  expected_tx_features: {expected_tx}")
    print(f"  expected_sbp_features: {expected_sbp}")
    print(f"  manifest_tx_dims: {manifest_tx_dims}")
    print(f"  manifest_sbp_dims: {manifest_sbp_dims}")
    print(f"  rows_with_has_tx_false: {manifest_has_tx_false}")
    print(f"  rows_with_has_sbp_false: {manifest_has_sbp_false}")
    print()
    print("shard consistency")
    print(f"  missing_shard_files: {len(shard_missing_files)}")
    print(f"  tx_width_mismatches: {len(shard_tx_width_mismatches)}")
    print(f"  sbp_width_mismatches: {len(shard_sbp_width_mismatches)}")
    print(f"  time_offset_mismatches: {len(shard_time_mismatches)}")
    print()
    print("dataset-level structural checks")
    print(f"  dataset_tx_all_zero_channels: {dataset_tx_all_zero.size} {format_indices(dataset_tx_all_zero)}")
    print(f"  dataset_sbp_all_zero_channels: {dataset_sbp_all_zero.size} {format_indices(dataset_sbp_all_zero)}")
    print(f"  dataset_sbp_nonfinite_channels: {dataset_sbp_nonfinite.size} {format_indices(dataset_sbp_nonfinite)}")
    print()
    print("shard-level anomalies")
    print(f"  shards_with_any_all_zero_tx_channel: {len(shards_with_zero_tx)}")
    if shards_with_zero_tx:
        print(f"    first_few: {shards_with_zero_tx[:8]}")
    print(f"  shards_with_any_all_zero_sbp_channel: {len(shards_with_zero_sbp)}")
    if shards_with_zero_sbp:
        print(f"    first_few: {shards_with_zero_sbp[:8]}")
    print(f"  shards_with_any_nonfinite_sbp_channel: {len(shards_with_nonfinite_sbp)}")
    if shards_with_nonfinite_sbp:
        print(f"    first_few: {shards_with_nonfinite_sbp[:8]}")
    print()
    print("interpretation")
    if (
        manifest_has_tx_false == 0
        and manifest_has_sbp_false == 0
        and not shard_missing_files
        and not shard_tx_width_mismatches
        and not shard_sbp_width_mismatches
        and not shard_time_mismatches
        and dataset_sbp_nonfinite.size == 0
    ):
        print(
            "  The cached dataset is structurally dense: every example points into fixed-width "
            "tx/sbp arrays with consistent feature counts. Missing-unit masking does not appear "
            "necessary for structural absence. Any all-zero channels should be treated as inactive "
            "or dead channels, not absent columns."
        )
    else:
        print(
            "  Structural inconsistencies were found. Review the mismatch counts above before "
            "assuming dense fixed-width features."
        )


if __name__ == "__main__":
    main()

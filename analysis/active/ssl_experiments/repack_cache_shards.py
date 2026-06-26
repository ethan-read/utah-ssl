"""Repack canonical cache datasets into larger fused shards.

This utility reads an existing cache root, slices examples using the manifest
and per-shard offset arrays, and writes a new cache root with larger shards.

It is intended for cases where some datasets have many tiny session/block
shards that create excessive file-handle or metadata overhead compared with a
reference layout like Brain2Text24.
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


TIME_OFFSETS_NAME = "time_offsets.npy"


@dataclass(frozen=True)
class ExampleRecord:
    row: dict[str, Any]
    shard_relpath: str
    example_index: int
    n_time_bins: int


@dataclass(frozen=True)
class DatasetRepackSummary:
    dataset: str
    src_shards: int
    dst_shards: int
    examples: int
    target_mb: float


def _normalize_names(values: list[str] | tuple[str, ...] | None) -> list[str]:
    if not values:
        return []
    out: list[str] = []
    seen: set[str] = set()
    for item in values:
        value = str(item).strip()
        if not value or value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _iter_dataset_names(src_root: Path, requested: list[str] | None) -> list[str]:
    if requested:
        names = _normalize_names(requested)
        for name in names:
            if not (src_root / name).is_dir():
                raise FileNotFoundError(f"Dataset not found under source root: {src_root / name}")
        return names
    return sorted(path.name for path in src_root.iterdir() if path.is_dir())


def _load_manifest(dataset_root: Path) -> list[dict[str, Any]]:
    manifest_path = dataset_root / "manifest.jsonl"
    rows: list[dict[str, Any]] = []
    with manifest_path.open() as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _load_shard_arrays(shard_path: Path) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    for path in sorted(shard_path.iterdir()):
        if path.is_file() and path.suffix == ".npy":
            arrays[path.name] = np.load(path, allow_pickle=False)
    if TIME_OFFSETS_NAME not in arrays:
        raise FileNotFoundError(f"Missing {TIME_OFFSETS_NAME} in {shard_path}")
    return arrays


def _classify_arrays(arrays: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], dict[str, tuple[np.ndarray, np.ndarray]]]:
    time_offsets = arrays[TIME_OFFSETS_NAME]
    if time_offsets.ndim != 1:
        raise ValueError(f"{TIME_OFFSETS_NAME} must be 1D, got shape {time_offsets.shape}")
    total_time = int(time_offsets[-1]) if int(time_offsets.shape[0]) > 0 else 0

    time_aligned: dict[str, np.ndarray] = {}
    offset_paired: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    for name, arr in arrays.items():
        if name == TIME_OFFSETS_NAME:
            continue
        if name.endswith("_offsets.npy"):
            data_name = name.replace("_offsets.npy", "_ids.npy")
            if data_name not in arrays:
                continue
            offsets = arr
            data = arrays[data_name]
            if offsets.ndim != 1:
                raise ValueError(f"{name} must be 1D, got shape {offsets.shape}")
            if int(offsets[-1]) != int(data.shape[0]):
                raise ValueError(
                    f"Offset/data mismatch in shard arrays: {name} ends at {int(offsets[-1])}, "
                    f"but {data_name} has leading dim {int(data.shape[0])}"
                )
            offset_paired[name] = (offsets, data)
            continue
        if int(arr.shape[0]) == total_time:
            time_aligned[name] = arr

    return time_aligned, offset_paired


def _pad_array_to_shape(array: np.ndarray, target_shape: tuple[int, ...]) -> np.ndarray:
    if tuple(array.shape[1:]) == tuple(target_shape):
        return np.asarray(array)
    if len(array.shape) - 1 != len(target_shape):
        raise ValueError(
            f"Cannot pad array of shape {array.shape} to target trailing shape {target_shape}"
        )
    out_shape = (int(array.shape[0]),) + tuple(int(x) for x in target_shape)
    padded = np.zeros(out_shape, dtype=array.dtype)
    slices = (slice(None),) + tuple(slice(0, int(x)) for x in array.shape[1:])
    padded[slices] = array
    return padded


def _scan_time_aligned_specs(
    *,
    src_root: Path,
    shard_relpaths: list[str],
) -> dict[str, tuple[tuple[int, ...], np.dtype]]:
    specs: dict[str, tuple[tuple[int, ...], np.dtype]] = {}
    for shard_relpath in shard_relpaths:
        arrays = _load_shard_arrays(src_root / shard_relpath)
        time_aligned, _ = _classify_arrays(arrays)
        for name, arr in time_aligned.items():
            trailing = tuple(int(x) for x in arr.shape[1:])
            current = specs.get(name)
            if current is None:
                specs[name] = (trailing, arr.dtype)
                continue
            prev_trailing, prev_dtype = current
            if len(prev_trailing) != len(trailing):
                raise ValueError(
                    f"Inconsistent ranks for array {name}: {prev_trailing} vs {trailing}"
                )
            merged = tuple(max(int(a), int(b)) for a, b in zip(prev_trailing, trailing))
            if prev_dtype != arr.dtype:
                raise ValueError(
                    f"Inconsistent dtypes for array {name}: {prev_dtype} vs {arr.dtype}"
                )
            specs[name] = (merged, prev_dtype)
    return specs


def _example_bytes(
    row: ExampleRecord,
    *,
    time_offsets: np.ndarray,
    time_aligned: dict[str, np.ndarray],
    offset_paired: dict[str, tuple[np.ndarray, np.ndarray]],
) -> int:
    start = int(time_offsets[row.example_index])
    stop = int(time_offsets[row.example_index + 1])
    total = 0
    for arr in time_aligned.values():
        total += int(arr[start:stop].nbytes)
    for offsets, data in offset_paired.values():
        o_start = int(offsets[row.example_index])
        o_stop = int(offsets[row.example_index + 1])
        total += int(data[o_start:o_stop].nbytes)
        total += int(np.asarray([0, o_stop - o_start], dtype=offsets.dtype).nbytes)
    total += int(np.asarray([0, stop - start], dtype=time_offsets.dtype).nbytes)
    return total


def _copy_example_into_builders(
    row: ExampleRecord,
    *,
    time_offsets: np.ndarray,
    time_aligned: dict[str, np.ndarray],
    offset_paired: dict[str, tuple[np.ndarray, np.ndarray]],
    time_aligned_specs: dict[str, tuple[tuple[int, ...], np.dtype]],
    time_buffers: dict[str, list[np.ndarray]],
    label_buffers: dict[str, list[np.ndarray]],
    offsets_buffers: dict[str, list[int]],
    label_offset_buffers: dict[str, list[int]],
) -> None:
    start = int(time_offsets[row.example_index])
    stop = int(time_offsets[row.example_index + 1])
    for name, arr in time_aligned.items():
        piece = np.asarray(arr[start:stop])
        target_shape, _ = time_aligned_specs[name]
        time_buffers[name].append(_pad_array_to_shape(piece, target_shape))
    offsets_buffers[TIME_OFFSETS_NAME].append(offsets_buffers[TIME_OFFSETS_NAME][-1] + (stop - start))

    for offsets_name, (offsets, data) in offset_paired.items():
        o_start = int(offsets[row.example_index])
        o_stop = int(offsets[row.example_index + 1])
        data_name = offsets_name.replace("_offsets.npy", "_ids.npy")
        label_buffers[data_name].append(np.asarray(data[o_start:o_stop]))
        label_offset_buffers[offsets_name].append(label_offset_buffers[offsets_name][-1] + (o_stop - o_start))


def _flush_shard(
    *,
    dataset_root: Path,
    dataset: str,
    shard_idx: int,
    manifest_rows: list[dict[str, Any]],
    time_buffers: dict[str, list[np.ndarray]],
    label_buffers: dict[str, list[np.ndarray]],
    offsets_buffers: dict[str, list[int]],
    label_offset_buffers: dict[str, list[int]],
) -> dict[str, Any]:
    shard_id = f"fused_{shard_idx:05d}"
    shard_dir = dataset_root / "shards" / shard_id
    shard_dir.mkdir(parents=True, exist_ok=True)

    for name, pieces in time_buffers.items():
        if not pieces:
            continue
        np.save(shard_dir / name, np.concatenate(pieces, axis=0))
    np.save(
        shard_dir / TIME_OFFSETS_NAME,
        np.asarray(offsets_buffers[TIME_OFFSETS_NAME], dtype=np.int64),
    )

    for data_name, pieces in label_buffers.items():
        if not pieces:
            continue
        np.save(shard_dir / data_name, np.concatenate(pieces, axis=0))
    for offsets_name, values in label_offset_buffers.items():
        np.save(shard_dir / offsets_name, np.asarray(values, dtype=np.int64))

    total_time_bins = int(offsets_buffers[TIME_OFFSETS_NAME][-1])
    shard_meta: dict[str, Any] = {
        "shard_id": shard_id,
        "shard_relpath": f"{dataset}/shards/{shard_id}",
        "example_count": len(manifest_rows),
        "total_time_bins": total_time_bins,
    }
    if manifest_rows:
        first = manifest_rows[0]
        if first.get("session_id") and all(row.get("session_id") == first.get("session_id") for row in manifest_rows):
            shard_meta["session_id"] = first.get("session_id")
        if first.get("subject_id") and all(row.get("subject_id") == first.get("subject_id") for row in manifest_rows):
            shard_meta["subject_id"] = first.get("subject_id")
        if first.get("source_split") and all(row.get("source_split") == first.get("source_split") for row in manifest_rows):
            shard_meta["source_split"] = first.get("source_split")
        if first.get("n_tx_features") is not None:
            shard_meta["n_tx_features"] = int(first.get("n_tx_features") or 0)
        if first.get("n_sbp_features") is not None and int(first.get("n_sbp_features") or 0) > 0:
            shard_meta["n_sbp_features"] = int(first.get("n_sbp_features") or 0)
    return shard_meta


def repack_dataset(
    *,
    src_root: Path,
    dst_root: Path,
    dataset: str,
    target_mb: float,
) -> DatasetRepackSummary:
    src_dataset_root = src_root / dataset
    dst_dataset_root = dst_root / dataset
    dst_dataset_root.mkdir(parents=True, exist_ok=True)

    rows_raw = _load_manifest(src_dataset_root)
    example_rows = [
        ExampleRecord(
            row=row,
            shard_relpath=str(row["shard_relpath"]),
            example_index=int(row["example_index"]),
            n_time_bins=int(row.get("n_time_bins", 0) or 0),
        )
        for row in rows_raw
    ]
    rows_by_shard: dict[str, list[ExampleRecord]] = {}
    for row in example_rows:
        rows_by_shard.setdefault(row.shard_relpath, []).append(row)
    time_aligned_specs = _scan_time_aligned_specs(
        src_root=src_root,
        shard_relpaths=sorted(rows_by_shard),
    )

    target_bytes = int(float(target_mb) * 1024 * 1024)
    new_manifest_rows: list[dict[str, Any]] = []
    new_shards_meta: list[dict[str, Any]] = []
    shard_idx = 0
    current_bytes = 0
    current_rows: list[dict[str, Any]] = []
    time_buffers: dict[str, list[np.ndarray]] = {}
    label_buffers: dict[str, list[np.ndarray]] = {}
    offsets_buffers: dict[str, list[int]] = {TIME_OFFSETS_NAME: [0]}
    label_offset_buffers: dict[str, list[int]] = {}

    def flush_current() -> None:
        nonlocal shard_idx, current_bytes, current_rows, time_buffers, label_buffers, offsets_buffers, label_offset_buffers
        if not current_rows:
            return
        shard_meta = _flush_shard(
            dataset_root=dst_dataset_root,
            dataset=dataset,
            shard_idx=shard_idx,
            manifest_rows=current_rows,
            time_buffers=time_buffers,
            label_buffers=label_buffers,
            offsets_buffers=offsets_buffers,
            label_offset_buffers=label_offset_buffers,
        )
        new_shards_meta.append(shard_meta)
        shard_idx += 1
        current_bytes = 0
        current_rows = []
        time_buffers = {}
        label_buffers = {}
        offsets_buffers = {TIME_OFFSETS_NAME: [0]}
        label_offset_buffers = {}

    for shard_relpath in sorted(rows_by_shard):
        shard_path = src_root / shard_relpath
        arrays = _load_shard_arrays(shard_path)
        time_offsets = arrays[TIME_OFFSETS_NAME]
        time_aligned, offset_paired = _classify_arrays(arrays)

        if not time_buffers:
            time_buffers = {name: [] for name in time_aligned_specs}
            label_buffers = {
                offsets_name.replace("_offsets.npy", "_ids.npy"): []
                for offsets_name in offset_paired
            }
            label_offset_buffers = {offsets_name: [0] for offsets_name in offset_paired}

        for record in rows_by_shard[shard_relpath]:
            ex_bytes = _example_bytes(
                record,
                time_offsets=time_offsets,
                time_aligned=time_aligned,
                offset_paired=offset_paired,
            )
            if current_rows and current_bytes + ex_bytes > target_bytes:
                flush_current()
                time_buffers = {name: [] for name in time_aligned_specs}
                label_buffers = {
                    offsets_name.replace("_offsets.npy", "_ids.npy"): []
                    for offsets_name in offset_paired
                }
                label_offset_buffers = {offsets_name: [0] for offsets_name in offset_paired}

            row_copy = dict(record.row)
            row_copy["shard_id"] = f"fused_{shard_idx:05d}"
            row_copy["shard_relpath"] = f"{dataset}/shards/fused_{shard_idx:05d}"
            row_copy["example_index"] = len(current_rows)
            new_manifest_rows.append(row_copy)
            current_rows.append(row_copy)
            _copy_example_into_builders(
                record,
                time_offsets=time_offsets,
                time_aligned=time_aligned,
                offset_paired=offset_paired,
                time_aligned_specs=time_aligned_specs,
                time_buffers=time_buffers,
                label_buffers=label_buffers,
                offsets_buffers=offsets_buffers,
                label_offset_buffers=label_offset_buffers,
            )
            current_bytes += ex_bytes

    flush_current()

    metadata = _load_json(src_dataset_root / "metadata.json")
    metadata["num_shards"] = len(new_shards_meta)
    metadata["shards"] = new_shards_meta
    notes = metadata.setdefault("build_notes", [])
    if isinstance(notes, list):
        notes.append(
            f"Repacked into fused shards targeting ~{float(target_mb):.1f} MB using analysis/active/ssl_experiments/repack_cache_shards.py."
        )
    metadata["repack_provenance"] = {
        "source_cache_root": str(src_root),
        "source_dataset_root": str(src_dataset_root),
        "target_mb": float(target_mb),
        "dst_shards": len(new_shards_meta),
        "src_shards": len(rows_by_shard),
    }

    _write_json(dst_dataset_root / "metadata.json", metadata)
    _write_jsonl(dst_dataset_root / "manifest.jsonl", new_manifest_rows)

    return DatasetRepackSummary(
        dataset=dataset,
        src_shards=len(rows_by_shard),
        dst_shards=len(new_shards_meta),
        examples=len(new_manifest_rows),
        target_mb=float(target_mb),
    )


def copy_dataset_tree(*, src_root: Path, dst_root: Path, dataset: str) -> None:
    shutil.copytree(src_root / dataset, dst_root / dataset, dirs_exist_ok=False)


def repack_cache_root(
    *,
    src_root: str | Path,
    dst_root: str | Path,
    repack_datasets: list[str],
    copy_datasets: list[str],
    target_mb: float,
    overwrite: bool = False,
) -> dict[str, Any]:
    src_root = Path(src_root)
    dst_root = Path(dst_root)
    if not src_root.is_dir():
        raise FileNotFoundError(f"Source cache root does not exist: {src_root}")
    if dst_root.exists():
        if not overwrite:
            raise FileExistsError(f"Destination already exists: {dst_root}")
        shutil.rmtree(dst_root)
    dst_root.mkdir(parents=True, exist_ok=True)

    summaries: list[dict[str, Any]] = []
    for dataset in _normalize_names(copy_datasets):
        copy_dataset_tree(src_root=src_root, dst_root=dst_root, dataset=dataset)
        summaries.append({"dataset": dataset, "mode": "copied_unchanged"})
    for dataset in _normalize_names(repack_datasets):
        summary = repack_dataset(
            src_root=src_root,
            dst_root=dst_root,
            dataset=dataset,
            target_mb=float(target_mb),
        )
        summaries.append(
            {
                "dataset": summary.dataset,
                "mode": "repacked",
                "src_shards": summary.src_shards,
                "dst_shards": summary.dst_shards,
                "examples": summary.examples,
                "target_mb": summary.target_mb,
            }
        )
    return {
        "src_root": str(src_root),
        "dst_root": str(dst_root),
        "target_mb": float(target_mb),
        "datasets": summaries,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, required=True)
    parser.add_argument("--dst", type=Path, required=True)
    parser.add_argument("--target-mb", type=float, default=65.0)
    parser.add_argument("--repack-dataset", action="append", default=[])
    parser.add_argument("--copy-dataset", action="append", default=[])
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = repack_cache_root(
        src_root=args.src,
        dst_root=args.dst,
        repack_datasets=list(args.repack_dataset),
        copy_datasets=list(args.copy_dataset),
        target_mb=float(args.target_mb),
        overwrite=bool(args.overwrite),
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

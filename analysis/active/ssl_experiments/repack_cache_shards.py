from __future__ import annotations

import argparse
import json
import re
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Repack canonical cache shards into larger fused shards.")
    parser.add_argument("--src", type=Path, required=True, help="Source cache_v1 root")
    parser.add_argument("--dst", type=Path, required=True, help="Destination repacked cache root")
    parser.add_argument(
        "--target-mb",
        type=float,
        default=128.0,
        help="Target fused shard size in MiB. Existing shards larger than this stay alone.",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help="Optional dataset names to repack. Default is all datasets under src.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Delete dst before writing.")
    parser.add_argument("--dry-run", action="store_true", help="Print the plan without writing output.")
    return parser.parse_args()


def format_bytes(num_bytes: int) -> str:
    value = float(num_bytes)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if value < 1024.0 or unit == "TB":
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{value:.1f} TB"


def sanitize_component(text: str | None) -> str:
    if not text:
        return "none"
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text)


@dataclass
class SourceShard:
    dataset: str
    shard_relpath: str
    shard_id: str
    source_split: str
    session_date: str | None
    rows: list[dict[str, Any]]
    shard_dir: Path
    bytes: int
    time_bins: int
    example_count: int
    file_names: tuple[str, ...]
    shape_signature: tuple[tuple[str, tuple[int, ...]], ...]


def load_manifest_rows(dataset_root: Path) -> list[dict[str, Any]]:
    manifest_path = dataset_root / "manifest.jsonl"
    rows: list[dict[str, Any]] = []
    with manifest_path.open() as handle:
        for line in handle:
            rows.append(json.loads(line))
    return rows


def build_source_shards(src_root: Path, dataset_name: str) -> tuple[list[SourceShard], dict[str, Any]]:
    dataset_root = src_root / dataset_name
    rows = load_manifest_rows(dataset_root)
    metadata = json.loads((dataset_root / "metadata.json").read_text())

    rows_by_shard: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        rows_by_shard[str(row["shard_relpath"])].append(row)

    shards: list[SourceShard] = []
    for shard_relpath, shard_rows in rows_by_shard.items():
        shard_rows = sorted(shard_rows, key=lambda row: int(row["example_index"]))
        first = shard_rows[0]
        shard_dir = src_root / shard_relpath
        file_names = tuple(sorted(p.name for p in shard_dir.iterdir() if p.is_file()))
        total_bytes = sum(p.stat().st_size for p in shard_dir.iterdir() if p.is_file())
        time_offsets = np.load(shard_dir / "time_offsets.npy", mmap_mode="r")
        shape_signature = []
        for array_path in sorted(shard_dir.glob("*.npy")):
            arr = np.load(array_path, mmap_mode="r")
            shape_signature.append((array_path.name, tuple(int(v) for v in arr.shape[1:])))
        shards.append(
            SourceShard(
                dataset=dataset_name,
                shard_relpath=shard_relpath,
                shard_id=str(first["shard_id"]),
                source_split=str(first.get("source_split") or "none"),
                session_date=str(first["session_date"]) if first.get("session_date") is not None else None,
                rows=shard_rows,
                shard_dir=shard_dir,
                bytes=int(total_bytes),
                time_bins=int(time_offsets[-1]),
                example_count=len(shard_rows),
                file_names=file_names,
                shape_signature=tuple(shape_signature),
            )
        )
    return sorted(
        shards,
        key=lambda shard: (
            shard.source_split,
            "" if shard.session_date is None else shard.session_date,
            shard.shard_relpath,
        ),
    ), metadata


def group_shards(shards: list[SourceShard], target_bytes: int) -> list[list[SourceShard]]:
    buckets: dict[tuple[str, tuple[str, ...], tuple[tuple[str, tuple[int, ...]], ...]], list[SourceShard]] = defaultdict(list)
    for shard in shards:
        buckets[(shard.source_split, shard.file_names, shard.shape_signature)].append(shard)

    groups: list[list[SourceShard]] = []
    for _, bucket in sorted(buckets.items(), key=lambda item: (item[0][0], item[0][1])):
        current: list[SourceShard] = []
        current_bytes = 0
        for shard in bucket:
            if current and current_bytes + shard.bytes > target_bytes:
                groups.append(current)
                current = []
                current_bytes = 0
            current.append(shard)
            current_bytes += shard.bytes
        if current:
            groups.append(current)
    return groups


def classify_alignment(
    array_name: str,
    array: np.ndarray,
    *,
    time_total: int,
    label_total: int | None,
    example_total: int,
) -> str:
    if array_name == "time_offsets.npy":
        return "time_offsets"
    if array_name == "phoneme_offsets.npy":
        return "label_offsets"
    if array_name == "phoneme_ids.npy":
        return "label"
    if array.ndim >= 1 and int(array.shape[0]) == time_total:
        return "time"
    if label_total is not None and array.ndim >= 1 and int(array.shape[0]) == label_total:
        return "label"
    if array.ndim >= 1 and int(array.shape[0]) == example_total:
        return "example"
    raise ValueError(f"Could not classify alignment for {array_name} with shape {array.shape}")


def concatenate_group(group: list[SourceShard], dst_shard_dir: Path) -> dict[str, Any]:
    arrays_by_kind: dict[str, list[np.ndarray]] = defaultdict(list)
    time_lengths: list[np.ndarray] = []
    label_lengths: list[np.ndarray] = []
    time_offset_dtype = None
    label_offset_dtype = None

    for shard in group:
        time_offsets = np.load(shard.shard_dir / "time_offsets.npy")
        time_offset_dtype = time_offsets.dtype if time_offset_dtype is None else time_offset_dtype
        time_lengths.append(np.diff(time_offsets).astype(np.int64, copy=False))
        example_total = int(len(time_offsets) - 1)
        time_total = int(time_offsets[-1])

        phoneme_offsets_path = shard.shard_dir / "phoneme_offsets.npy"
        label_total = None
        if phoneme_offsets_path.exists():
            phoneme_offsets = np.load(phoneme_offsets_path)
            label_offset_dtype = phoneme_offsets.dtype if label_offset_dtype is None else label_offset_dtype
            label_lengths.append(np.diff(phoneme_offsets).astype(np.int64, copy=False))
            label_total = int(phoneme_offsets[-1])

        for array_path in sorted(shard.shard_dir.glob("*.npy")):
            array = np.load(array_path)
            kind = classify_alignment(
                array_path.name,
                array,
                time_total=time_total,
                label_total=label_total,
                example_total=example_total,
            )
            if kind in {"time", "label", "example"}:
                arrays_by_kind[array_path.name].append(array)

    dst_shard_dir.mkdir(parents=True, exist_ok=True)

    all_time_lengths = np.concatenate(time_lengths, axis=0)
    new_time_offsets = np.zeros(len(all_time_lengths) + 1, dtype=time_offset_dtype or np.int64)
    new_time_offsets[1:] = np.cumsum(all_time_lengths, dtype=np.int64)
    np.save(dst_shard_dir / "time_offsets.npy", new_time_offsets)

    if label_lengths:
        all_label_lengths = np.concatenate(label_lengths, axis=0)
        new_label_offsets = np.zeros(len(all_label_lengths) + 1, dtype=label_offset_dtype or np.int64)
        new_label_offsets[1:] = np.cumsum(all_label_lengths, dtype=np.int64)
        np.save(dst_shard_dir / "phoneme_offsets.npy", new_label_offsets)

    for array_name, parts in arrays_by_kind.items():
        np.save(dst_shard_dir / array_name, np.concatenate(parts, axis=0))

    total_bytes = sum(p.stat().st_size for p in dst_shard_dir.iterdir() if p.is_file())
    return {
        "example_count": int(len(all_time_lengths)),
        "total_time_bins": int(new_time_offsets[-1]),
        "bytes": int(total_bytes),
    }


def summarize_group(group: list[SourceShard], target_bytes: int) -> dict[str, Any]:
    return {
        "source_split": group[0].source_split,
        "source_shards": len(group),
        "source_examples": sum(shard.example_count for shard in group),
        "source_time_bins": sum(shard.time_bins for shard in group),
        "source_bytes": sum(shard.bytes for shard in group),
        "target_bytes": int(target_bytes),
    }


def copy_dataset_extras(src_dataset_root: Path, dst_dataset_root: Path) -> None:
    for path in src_dataset_root.iterdir():
        if path.name in {"manifest.jsonl", "metadata.json", "shards"}:
            continue
        if path.is_file() and not path.name.startswith(".DS_Store"):
            shutil.copy2(path, dst_dataset_root / path.name)


def rebuild_dataset(
    src_root: Path,
    dst_root: Path,
    dataset_name: str,
    *,
    target_bytes: int,
    dry_run: bool,
) -> dict[str, Any]:
    shards, metadata = build_source_shards(src_root, dataset_name)
    groups = group_shards(shards, target_bytes=target_bytes)
    dst_dataset_root = dst_root / dataset_name
    dst_shards_root = dst_dataset_root / "shards"

    rows_by_shard = {shard.shard_relpath: shard.rows for shard in shards}
    meta_by_relpath = {
        str(item.get("shard_relpath")): item
        for item in metadata.get("shards", [])
        if isinstance(item, dict) and item.get("shard_relpath") is not None
    }

    new_rows: list[dict[str, Any]] = []
    new_shard_meta: list[dict[str, Any]] = []
    dataset_group_counter = 0

    for group in groups:
        group_split = sanitize_component(group[0].source_split)
        new_shard_id = f"fused_{group_split}_{dataset_group_counter:04d}"
        dataset_group_counter += 1
        new_shard_relpath = f"{dataset_name}/shards/{new_shard_id}"
        summary = summarize_group(group, target_bytes=target_bytes)

        if not dry_run:
            fused_summary = concatenate_group(group, dst_root / new_shard_relpath)
            summary.update(fused_summary)

        next_example_index = 0
        for shard in group:
            for row in rows_by_shard[shard.shard_relpath]:
                new_row = dict(row)
                new_row["shard_id"] = new_shard_id
                new_row["shard_relpath"] = new_shard_relpath
                new_row["example_index"] = next_example_index
                new_rows.append(new_row)
                next_example_index += 1

        source_meta_items = [meta_by_relpath.get(shard.shard_relpath, {}) for shard in group]
        aggregated_meta = {
            "shard_id": new_shard_id,
            "shard_relpath": new_shard_relpath,
            "example_count": int(sum(shard.example_count for shard in group)),
            "total_time_bins": int(sum(shard.time_bins for shard in group)),
            "source_shard_count": int(len(group)),
            "source_split": group[0].source_split,
        }
        if source_meta_items:
            all_keys = set().union(*(item.keys() for item in source_meta_items))
            for key in sorted(all_keys):
                if key in aggregated_meta:
                    continue
                values = [item.get(key) for item in source_meta_items]
                if all(value == values[0] for value in values):
                    aggregated_meta[key] = values[0]
        if not dry_run:
            aggregated_meta.update(summary)
        new_shard_meta.append(aggregated_meta)

    if dry_run:
        return {
            "dataset": dataset_name,
            "old_shards": len(shards),
            "new_shards": len(groups),
            "old_bytes": int(sum(shard.bytes for shard in shards)),
            "new_bytes": None,
            "examples": len(new_rows),
        }

    dst_dataset_root.mkdir(parents=True, exist_ok=True)
    dst_shards_root.mkdir(parents=True, exist_ok=True)
    copy_dataset_extras(src_root / dataset_name, dst_dataset_root)
    with (dst_dataset_root / "manifest.jsonl").open("w") as handle:
        for row in new_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    new_metadata = dict(metadata)
    new_metadata["num_shards"] = len(new_shard_meta)
    new_metadata["shards"] = new_shard_meta
    build_notes = list(new_metadata.get("build_notes", []))
    build_notes.append(
        f"Repacked from existing canonical cache into larger fused shards with target size {target_bytes / (1024 ** 2):.1f} MiB."
    )
    new_metadata["build_notes"] = build_notes
    (dst_dataset_root / "metadata.json").write_text(json.dumps(new_metadata, indent=2))

    return {
        "dataset": dataset_name,
        "old_shards": len(shards),
        "new_shards": len(groups),
        "old_bytes": int(sum(shard.bytes for shard in shards)),
        "new_bytes": int(sum(item["bytes"] for item in new_shard_meta if "bytes" in item)),
        "examples": len(new_rows),
    }


def main() -> None:
    args = parse_args()
    src_root = args.src.resolve()
    dst_root = args.dst.resolve()
    target_bytes = int(args.target_mb * (1024 ** 2))

    if not src_root.exists():
        raise FileNotFoundError(f"Source cache root not found: {src_root}")

    datasets = (
        args.datasets
        if args.datasets
        else sorted(path.name for path in src_root.iterdir() if path.is_dir())
    )

    if dst_root.exists():
        if args.overwrite:
            shutil.rmtree(dst_root)
        elif not args.dry_run:
            raise FileExistsError(f"Destination already exists: {dst_root}")

    if not args.dry_run:
        dst_root.mkdir(parents=True, exist_ok=True)

    print(f"source: {src_root}")
    print(f"dest: {dst_root}")
    print(f"target shard size: {args.target_mb:.1f} MiB")
    print(f"datasets: {datasets}")

    summaries = []
    for dataset_name in datasets:
        print(f"\n=== {dataset_name} ===")
        summary = rebuild_dataset(
            src_root,
            dst_root,
            dataset_name,
            target_bytes=target_bytes,
            dry_run=args.dry_run,
        )
        summaries.append(summary)
        old_bytes = format_bytes(summary["old_bytes"])
        new_bytes = "n/a" if summary["new_bytes"] is None else format_bytes(summary["new_bytes"])
        print(
            f"{dataset_name}: shards {summary['old_shards']} -> {summary['new_shards']}, "
            f"bytes {old_bytes} -> {new_bytes}, examples={summary['examples']}"
        )

    if not args.dry_run:
        (dst_root / "repack_summary.json").write_text(json.dumps(summaries, indent=2))
    print("\nDone.")


if __name__ == "__main__":
    main()

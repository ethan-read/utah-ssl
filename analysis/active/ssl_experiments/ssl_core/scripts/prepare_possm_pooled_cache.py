"""Build and validate a versioned area-6v Brain2Text25 cache.

Brain2Text24 remains in its canonical cache. The raw and pre-smoothed
Brain2Text25 sources are repacked independently, so this script never
regenerates smoothing after examples have been fused into new shards. TX and
SBP are projected to their first 128 columns. Every retained array preserves
its source dtype and exact values.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import statistics
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any

import numpy as np

EXPERIMENTS_DIR = Path(__file__).resolve().parents[2]
REPO_ROOT = Path(__file__).resolve().parents[5]
for _path in (REPO_ROOT, EXPERIMENTS_DIR):
    _path_str = str(_path)
    if _path_str not in sys.path:
        sys.path.insert(0, _path_str)

from masked_ssl.cache import _compute_dataset_cache_source_signature  # noqa: E402
from ssl_core.scripts.repack_cache_shards import (  # noqa: E402
    AREA6V_FEATURES,
    TIME_OFFSETS_NAME,
    _classify_arrays,
    _load_shard_arrays,
    repack_cache_root,
)


DATASETS = ("brain2text25",)
SUMMARY_NAME = "possm_pooled_cache_prep_summary.json"


def _timestamp_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_inventory(root: Path) -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    for dataset in DATASETS:
        dataset_root = root / dataset
        if not dataset_root.is_dir():
            raise FileNotFoundError(f"Dataset missing from source cache: {dataset_root}")
        for relative in ("manifest.jsonl", "metadata.json"):
            path = dataset_root / relative
            files.append(
                {
                    "path": str(path.relative_to(root)),
                    "size": int(path.stat().st_size),
                    "mtime_ns": int(path.stat().st_mtime_ns),
                    "sha256": _sha256_file(path),
                }
            )
        for path in sorted((dataset_root / "shards").glob("*/*.npy")):
            files.append(
                {
                    "path": str(path.relative_to(root)),
                    "size": int(path.stat().st_size),
                    "mtime_ns": int(path.stat().st_mtime_ns),
                }
            )
    return {
        "root": str(root),
        "source_signature": _compute_dataset_cache_source_signature(
            {"brain2text25": root}
        ),
        "files": files,
    }


def _read_manifest(root: Path, dataset: str) -> list[dict[str, Any]]:
    path = root / dataset / "manifest.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _rows_by_example_id(rows: list[dict[str, Any]], *, dataset: str) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        example_id = str(row.get("example_id", "")).strip()
        if not example_id:
            raise ValueError(f"{dataset} manifest row is missing example_id: {row}")
        if example_id in indexed:
            raise ValueError(f"{dataset} has duplicate example_id={example_id!r}")
        indexed[example_id] = row
    return indexed


def _normalized_manifest_row(row: dict[str, Any]) -> dict[str, Any]:
    normalized = {
        key: value
        for key, value in row.items()
        if key not in {"shard_id", "shard_relpath", "example_index"}
    }
    if bool(normalized.get("has_tx", False)):
        normalized["n_tx_features"] = AREA6V_FEATURES
    if bool(normalized.get("has_sbp", False)):
        normalized["n_sbp_features"] = AREA6V_FEATURES
    if "n_total_features" in normalized:
        normalized["n_total_features"] = AREA6V_FEATURES + (
            AREA6V_FEATURES if bool(normalized.get("has_sbp", False)) else 0
        )
    return normalized


class _LogicalExampleReader:
    def __init__(self, root: Path, *, project_area6v: bool):
        self.root = Path(root)
        self.project_area6v = bool(project_area6v)
        self._shard_relpath: str | None = None
        self._arrays: dict[str, np.ndarray] | None = None

    def read(self, row: dict[str, Any]) -> dict[str, np.ndarray]:
        shard_relpath = str(row["shard_relpath"])
        if self._arrays is None or self._shard_relpath != shard_relpath:
            self._arrays = _load_shard_arrays(
                self.root / shard_relpath,
                project_area6v=self.project_area6v,
            )
            self._shard_relpath = shard_relpath
        arrays = self._arrays
        assert arrays is not None
        example_index = int(row["example_index"])
        time_offsets = arrays[TIME_OFFSETS_NAME]
        start = int(time_offsets[example_index])
        stop = int(time_offsets[example_index + 1])
        time_aligned, offset_paired = _classify_arrays(arrays)
        logical = {
            name: np.asarray(array[start:stop])
            for name, array in time_aligned.items()
        }
        logical[TIME_OFFSETS_NAME] = np.asarray(
            [0, stop - start],
            dtype=time_offsets.dtype,
        )
        for offsets_name, (offsets, data) in offset_paired.items():
            data_name = offsets_name.replace("_offsets.npy", "_ids.npy")
            offset_start = int(offsets[example_index])
            offset_stop = int(offsets[example_index + 1])
            logical[data_name] = np.asarray(data[offset_start:offset_stop])
            logical[offsets_name] = np.asarray(
                [0, offset_stop - offset_start],
                dtype=offsets.dtype,
            )
        return logical


def _update_logical_hash(
    digest: Any,
    *,
    example_id: str,
    name: str,
    array: np.ndarray,
) -> None:
    digest.update(str(example_id).encode("utf-8"))
    digest.update(str(name).encode("utf-8"))
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(json.dumps(list(array.shape)).encode("utf-8"))
    digest.update(np.ascontiguousarray(array).tobytes())


def _shard_size_summary(root: Path, dataset: str) -> dict[str, Any]:
    sizes = [
        sum(path.stat().st_size for path in shard_dir.glob("*.npy"))
        for shard_dir in sorted((root / dataset / "shards").iterdir())
        if shard_dir.is_dir()
    ]
    if not sizes:
        raise ValueError(f"No destination shards found for {dataset}")
    return {
        "count": len(sizes),
        "min_mb": min(sizes) / (1024 ** 2),
        "median_mb": statistics.median(sizes) / (1024 ** 2),
        "max_mb": max(sizes) / (1024 ** 2),
        "total_gb": sum(sizes) / (1024 ** 3),
    }


def validate_versioned_cache(
    *,
    source_root: str | Path,
    destination_root: str | Path,
    target_mb: float,
) -> dict[str, Any]:
    source_root = Path(source_root)
    destination_root = Path(destination_root)
    dataset_summaries: dict[str, Any] = {}
    aggregate_source = hashlib.sha256()
    aggregate_destination = hashlib.sha256()

    for dataset in DATASETS:
        source_rows = _read_manifest(source_root, dataset)
        destination_rows = _read_manifest(destination_root, dataset)
        source_by_id = _rows_by_example_id(source_rows, dataset=dataset)
        destination_by_id = _rows_by_example_id(destination_rows, dataset=dataset)
        if set(source_by_id) != set(destination_by_id):
            missing = sorted(set(source_by_id).difference(destination_by_id))[:5]
            extra = sorted(set(destination_by_id).difference(source_by_id))[:5]
            raise ValueError(
                f"{dataset} example IDs changed during repack; missing={missing} extra={extra}"
            )

        source_reader = _LogicalExampleReader(source_root, project_area6v=True)
        destination_reader = _LogicalExampleReader(destination_root, project_area6v=False)
        dataset_source = hashlib.sha256()
        dataset_destination = hashlib.sha256()
        for source_row in source_rows:
            example_id = str(source_row["example_id"])
            destination_row = destination_by_id[example_id]
            if _normalized_manifest_row(source_row) != _normalized_manifest_row(destination_row):
                raise ValueError(f"{dataset}:{example_id} manifest metadata changed during repack")
            source_logical = source_reader.read(source_row)
            destination_logical = destination_reader.read(destination_row)
            if set(source_logical) != set(destination_logical):
                raise ValueError(
                    f"{dataset}:{example_id} logical arrays changed: "
                    f"source={sorted(source_logical)} destination={sorted(destination_logical)}"
                )
            for name in sorted(source_logical):
                source_array = source_logical[name]
                destination_array = destination_logical[name]
                expected_array = source_array
                if destination_array.dtype != expected_array.dtype:
                    raise ValueError(
                        f"{dataset}:{example_id}:{name} changed dtype from "
                        f"{expected_array.dtype} to {destination_array.dtype}"
                    )
                if not np.array_equal(expected_array, destination_array):
                    raise ValueError(
                        f"{dataset}:{example_id}:{name} differs from the expected "
                        "lossless area-6v representation"
                    )
                _update_logical_hash(
                    dataset_source,
                    example_id=example_id,
                    name=name,
                    array=expected_array,
                )
                _update_logical_hash(
                    dataset_destination,
                    example_id=example_id,
                    name=name,
                    array=destination_array,
                )

        source_hash = dataset_source.hexdigest()
        destination_hash = dataset_destination.hexdigest()
        if source_hash != destination_hash:
            raise ValueError(f"{dataset} logical hash mismatch after validation")
        aggregate_source.update(source_hash.encode("utf-8"))
        aggregate_destination.update(destination_hash.encode("utf-8"))

        source_splits = Counter(str(row.get("source_split", "")) for row in source_rows)
        destination_splits = Counter(str(row.get("source_split", "")) for row in destination_rows)
        if source_splits != destination_splits:
            raise ValueError(f"{dataset} source-split counts changed during repack")

        metadata = json.loads((destination_root / dataset / "metadata.json").read_text())
        feature_layout = metadata.get("feature_layout", {})
        if int(feature_layout.get("n_tx_features", 0) or 0) != AREA6V_FEATURES:
            raise ValueError(f"{dataset} destination metadata does not declare 128 TX features")
        if int(feature_layout.get("n_sbp_features", 0) or 0) != AREA6V_FEATURES:
            raise ValueError(f"{dataset} destination metadata does not declare 128 SBP features")
        if list(feature_layout.get("tx_slice", [])) != [0, AREA6V_FEATURES]:
            raise ValueError(f"{dataset} destination metadata has the wrong TX slice")
        if list(feature_layout.get("sbp_slice", [])) != [
            AREA6V_FEATURES,
            2 * AREA6V_FEATURES,
        ]:
            raise ValueError(f"{dataset} destination metadata has the wrong SBP slice")
        if not bool(metadata.get("area6v_migration", {}).get("area6v_only", False)):
            raise ValueError(f"{dataset} destination metadata lacks area-6v provenance")
        if "tx_storage_conversion" in metadata:
            raise ValueError(
                f"{dataset} destination metadata unexpectedly declares TX conversion"
            )

        shard_sizes = _shard_size_summary(destination_root, dataset)
        if float(shard_sizes["median_mb"]) > float(target_mb) + 1.0:
            raise ValueError(
                f"{dataset} median shard exceeds target: "
                f"{shard_sizes['median_mb']:.2f} MB > {float(target_mb):.2f} MB"
            )
        dataset_summaries[dataset] = {
            "examples": len(source_rows),
            "source_split_counts": dict(sorted(source_splits.items())),
            "logical_sha256": source_hash,
            "storage_policy": "preserve_projected_source_dtypes_and_values",
            "shards": shard_sizes,
        }

    source_hash = aggregate_source.hexdigest()
    destination_hash = aggregate_destination.hexdigest()
    if source_hash != destination_hash:
        raise ValueError("Aggregate logical hash mismatch")
    return {
        "source_root": str(source_root),
        "destination_root": str(destination_root),
        "target_mb": float(target_mb),
        "logical_sha256": source_hash,
        "datasets": dataset_summaries,
    }


def _prepare_one_root(
    *,
    source_root: Path,
    destination_root: Path,
    target_mb: float,
    resume_completed: bool,
    replace_partial: bool,
) -> dict[str, Any]:
    summary_path = destination_root / SUMMARY_NAME
    if destination_root.exists():
        if not resume_completed:
            raise FileExistsError(
                f"Completed destination already exists: {destination_root}. "
                "Use --resume-completed to validate and reuse it."
            )
        if not summary_path.exists():
            raise FileNotFoundError(
                f"Destination exists without completion summary: {summary_path}"
            )
        validation = validate_versioned_cache(
            source_root=source_root,
            destination_root=destination_root,
            target_mb=target_mb,
        )
        return {
            "status": "reused_completed",
            "summary_path": str(summary_path),
            "validation": validation,
        }

    partial_root = destination_root.with_name(destination_root.name + ".partial")
    if partial_root.exists():
        if not replace_partial:
            raise FileExistsError(
                f"Partial destination exists: {partial_root}. "
                "Inspect it or pass --replace-partial."
            )
        shutil.rmtree(partial_root)

    source_before = _source_inventory(source_root)
    repack_summary = repack_cache_root(
        src_root=source_root,
        dst_root=partial_root,
        repack_datasets=list(DATASETS),
        copy_datasets=[],
        target_mb=float(target_mb),
        area6v_datasets=list(DATASETS),
        tx_float16_datasets=[],
    )
    validation = validate_versioned_cache(
        source_root=source_root,
        destination_root=partial_root,
        target_mb=target_mb,
    )
    source_after = _source_inventory(source_root)
    if source_before != source_after:
        raise RuntimeError(f"Source cache changed during preparation: {source_root}")

    repack_summary["dst_root"] = str(destination_root)
    validation["destination_root"] = str(destination_root)
    completed_summary = {
        "kind": "possm_brain2text25_area6v_cache",
        "created_utc": _timestamp_utc(),
        "source_root": str(source_root),
        "destination_root": str(destination_root),
        "datasets": list(DATASETS),
        "area6v_columns": [0, AREA6V_FEATURES],
        "tx_storage_policy": "preserve_source_dtype_exactly",
        "sbp_storage_policy": "preserve_source_dtype_exactly",
        "target_mb": float(target_mb),
        "source_inventory": source_before,
        "repack": repack_summary,
        "validation": validation,
    }
    (partial_root / SUMMARY_NAME).write_text(
        json.dumps(completed_summary, indent=2) + "\n"
    )
    partial_root.replace(destination_root)
    return {
        "status": "built",
        "summary_path": str(destination_root / SUMMARY_NAME),
        "validation": validation,
    }


def prepare_possm_pooled_caches(
    *,
    raw_source_root: str | Path,
    smoothed_source_root: str | Path,
    raw_destination_root: str | Path,
    smoothed_destination_root: str | Path,
    target_mb: float = 65.0,
    dry_run: bool = False,
    resume_completed: bool = False,
    replace_partial: bool = False,
) -> dict[str, Any]:
    raw_source_root = Path(raw_source_root)
    smoothed_source_root = Path(smoothed_source_root)
    raw_destination_root = Path(raw_destination_root)
    smoothed_destination_root = Path(smoothed_destination_root)
    if raw_destination_root == smoothed_destination_root:
        raise ValueError("Raw and smoothed destination roots must be different")

    dry_run_payload = {
        "kind": "possm_brain2text25_area6v_cache_plan",
        "created_utc": _timestamp_utc(),
        "target_mb": float(target_mb),
        "datasets": list(DATASETS),
        "area6v_columns": [0, AREA6V_FEATURES],
        "tx_storage_policy": "preserve_source_dtype_exactly",
        "sbp_storage_policy": "preserve_source_dtype_exactly",
        "raw": {
            "source": str(raw_source_root),
            "destination": str(raw_destination_root),
            "source_inventory": _source_inventory(raw_source_root),
        },
        "smoothed": {
            "source": str(smoothed_source_root),
            "destination": str(smoothed_destination_root),
            "source_inventory": _source_inventory(smoothed_source_root),
        },
        "dry_run": bool(dry_run),
    }
    if dry_run:
        return dry_run_payload

    return {
        **dry_run_payload,
        "dry_run": False,
        "raw_result": _prepare_one_root(
            source_root=raw_source_root,
            destination_root=raw_destination_root,
            target_mb=target_mb,
            resume_completed=resume_completed,
            replace_partial=replace_partial,
        ),
        "smoothed_result": _prepare_one_root(
            source_root=smoothed_source_root,
            destination_root=smoothed_destination_root,
            target_mb=target_mb,
            resume_completed=resume_completed,
            replace_partial=replace_partial,
        ),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-source-root", type=Path, required=True)
    parser.add_argument("--smoothed-source-root", type=Path, required=True)
    parser.add_argument("--raw-destination-root", type=Path, required=True)
    parser.add_argument("--smoothed-destination-root", type=Path, required=True)
    parser.add_argument("--target-mb", type=float, default=65.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume-completed", action="store_true")
    parser.add_argument("--replace-partial", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    summary = prepare_possm_pooled_caches(
        raw_source_root=args.raw_source_root,
        smoothed_source_root=args.smoothed_source_root,
        raw_destination_root=args.raw_destination_root,
        smoothed_destination_root=args.smoothed_destination_root,
        target_mb=float(args.target_mb),
        dry_run=bool(args.dry_run),
        resume_completed=bool(args.resume_completed),
        replace_partial=bool(args.replace_partial),
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

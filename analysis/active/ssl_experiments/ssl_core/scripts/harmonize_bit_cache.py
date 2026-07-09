"""Harmonize a canonical Utah SSL cache root for BIT-style stage-1 pretraining."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
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

import numpy as np

from ssl_core.bit_cache_contract import (
    BIT_CANONICAL_FEATURE_POLICY,
    BIT_STAGE1_BOUNDARY_KEY_MODE,
    BIT_STAGE1_DEFAULT_EXCLUDED_DATASETS,
    BIT_STAGE1_DEFAULT_INCLUDED_DATASETS,
    BIT_STAGE1_FEATURE_MODE,
    BIT_STAGE1_TX_DIM,
)
from ssl_core.scripts.trim_area6v_cache import trim_area6v_cache


@dataclass(frozen=True)
class MotorDataPaddingSummary:
    shard_count: int
    tx_arrays_padded: int
    manifest_rows_updated: int
    metadata_updated: bool


def _timestamp_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _iter_dataset_roots(cache_root: Path) -> list[Path]:
    return sorted(
        path for path in cache_root.iterdir() if path.is_dir() and (path / "metadata.json").exists()
    )


def _iter_shard_dirs(dataset_root: Path) -> list[Path]:
    shard_root = dataset_root / "shards"
    if shard_root.is_dir():
        return sorted(path for path in shard_root.iterdir() if path.is_dir())
    return sorted(path for path in dataset_root.iterdir() if path.is_dir() and path.name != "shards")


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    tmp_path = path.with_name(f".{path.name}.tmp")
    tmp_path.write_text(json.dumps(payload, indent=2) + "\n")
    tmp_path.replace(path)


def _atomic_write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    tmp_path = path.with_name(f".{path.name}.tmp")
    with tmp_path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    tmp_path.replace(path)


def _atomic_save_npy(path: Path, array: np.ndarray) -> None:
    tmp_path = path.with_name(f".{path.name}.tmp.npy")
    np.save(tmp_path, array)
    tmp_path.replace(path)


def _load_metadata(dataset_root: Path) -> dict[str, Any]:
    metadata_path = dataset_root / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata: {metadata_path}")
    return json.loads(metadata_path.read_text())


def _save_metadata(dataset_root: Path, metadata: dict[str, Any], *, dry_run: bool) -> None:
    if not dry_run:
        _atomic_write_json(dataset_root / "metadata.json", metadata)


def _pad_feature_array_file(
    path: Path,
    *,
    target_width: int,
    dry_run: bool,
) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Missing cache array: {path}")
    arr = np.load(path, mmap_mode="r", allow_pickle=False)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D feature array at {path}, got shape {arr.shape}")
    width = int(arr.shape[1])
    if width == int(target_width):
        return "already_harmonized"
    if width > int(target_width):
        raise ValueError(
            f"Feature array at {path} has width {width}, which exceeds target width {target_width}"
        )
    if not dry_run:
        padded = np.zeros((int(arr.shape[0]), int(target_width)), dtype=arr.dtype)
        padded[:, :width] = np.asarray(arr)
        _atomic_save_npy(path, padded)
    return "padded"


def _rewrite_manifest_feature_widths(
    manifest_path: Path,
    *,
    tx_dim: int | None = None,
    sbp_dim: int | None = None,
    dry_run: bool,
) -> int:
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")
    rows: list[dict[str, Any]] = []
    changed = 0
    with manifest_path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            row_changed = False
            if tx_dim is not None and int(row.get("n_tx_features", 0) or 0) != int(tx_dim):
                row["n_tx_features"] = int(tx_dim)
                row_changed = True
            if sbp_dim is not None and int(row.get("n_sbp_features", 0) or 0) != int(sbp_dim):
                row["n_sbp_features"] = int(sbp_dim)
                row_changed = True
            if row_changed:
                changed += 1
            rows.append(row)
    if changed and not dry_run:
        _atomic_write_jsonl(manifest_path, rows)
    return changed


def _update_feature_layout_dims(
    metadata: dict[str, Any],
    *,
    tx_dim: int | None = None,
    sbp_dim: int | None = None,
) -> bool:
    changed = False
    feature_layout = metadata.setdefault("feature_layout", {})
    if isinstance(feature_layout, dict):
        if tx_dim is not None and int(feature_layout.get("n_tx_features", 0) or 0) != int(tx_dim):
            feature_layout["n_tx_features"] = int(tx_dim)
            changed = True
        if sbp_dim is not None and int(feature_layout.get("n_sbp_features", 0) or 0) != int(sbp_dim):
            feature_layout["n_sbp_features"] = int(sbp_dim)
            changed = True
        if tx_dim is not None and sbp_dim is not None:
            total = int(tx_dim) + int(sbp_dim)
            if int(feature_layout.get("n_total_features", 0) or 0) != total:
                feature_layout["n_total_features"] = total
                changed = True
    shards = metadata.get("shards")
    if isinstance(shards, list):
        for shard in shards:
            if not isinstance(shard, dict):
                continue
            if tx_dim is not None and int(shard.get("n_tx_features", 0) or 0) != int(tx_dim):
                shard["n_tx_features"] = int(tx_dim)
                changed = True
            if sbp_dim is not None and int(shard.get("n_sbp_features", 0) or 0) != int(sbp_dim):
                shard["n_sbp_features"] = int(sbp_dim)
                changed = True
    return changed


def _annotate_dataset_metadata(
    *,
    dataset_root: Path,
    default_stage1_included: bool,
    dry_run: bool,
) -> bool:
    metadata = _load_metadata(dataset_root)
    feature_layout = metadata.get("feature_layout", {})
    on_disk_tx = int(feature_layout.get("n_tx_features", 0) or 0)
    on_disk_sbp = int(feature_layout.get("n_sbp_features", 0) or 0)
    annotation = {
        "policy_name": BIT_CANONICAL_FEATURE_POLICY,
        "stage1_feature_mode": BIT_STAGE1_FEATURE_MODE,
        "stage1_boundary_key_mode": BIT_STAGE1_BOUNDARY_KEY_MODE,
        "stage1_tx_dim": int(BIT_STAGE1_TX_DIM),
        "stage1_default_included": bool(default_stage1_included),
        "stage1_default_excluded_datasets": list(BIT_STAGE1_DEFAULT_EXCLUDED_DATASETS),
        "native_modalities": list(metadata.get("modalities", [])),
        "effective_on_disk_feature_widths": {
            "tx": int(on_disk_tx),
            "sbp": int(on_disk_sbp),
        },
    }
    current = metadata.get("canonical_cache_policy")
    current_without_timestamp = (
        {key: value for key, value in current.items() if key != "updated_utc"}
        if isinstance(current, dict)
        else current
    )
    changed = current_without_timestamp != annotation
    if changed:
        annotation["updated_utc"] = _timestamp_utc()
        metadata["canonical_cache_policy"] = annotation
        _save_metadata(dataset_root, metadata, dry_run=dry_run)
    return changed


def harmonize_motor_data(
    cache_root: str | Path,
    *,
    dry_run: bool = False,
) -> MotorDataPaddingSummary:
    root = Path(cache_root)
    dataset_root = root / "motor_data"
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    tx_arrays_padded = 0
    shard_count = 0
    for shard_dir in _iter_shard_dirs(dataset_root):
        shard_count += 1
        tx_result = _pad_feature_array_file(
            shard_dir / "tx.npy",
            target_width=BIT_STAGE1_TX_DIM,
            dry_run=dry_run,
        )
        if tx_result == "padded":
            tx_arrays_padded += 1

    manifest_rows_updated = _rewrite_manifest_feature_widths(
        dataset_root / "manifest.jsonl",
        tx_dim=BIT_STAGE1_TX_DIM,
        dry_run=dry_run,
    )
    metadata = _load_metadata(dataset_root)
    metadata_changed = _update_feature_layout_dims(
        metadata,
        tx_dim=BIT_STAGE1_TX_DIM,
    )
    harmonization = {
        "kind": "motor_data_tx_only_width_padding",
        "target_tx_dim": int(BIT_STAGE1_TX_DIM),
        "stage1_feature_mode": BIT_STAGE1_FEATURE_MODE,
    }
    current_harmonization = metadata.get("bit_harmonization")
    current_without_timestamp = (
        {key: value for key, value in current_harmonization.items() if key != "updated_utc"}
        if isinstance(current_harmonization, dict)
        else current_harmonization
    )
    if current_without_timestamp != harmonization:
        harmonization["updated_utc"] = _timestamp_utc()
        metadata["bit_harmonization"] = harmonization
        metadata_changed = True
    if metadata_changed:
        _save_metadata(dataset_root, metadata, dry_run=dry_run)

    return MotorDataPaddingSummary(
        shard_count=shard_count,
        tx_arrays_padded=tx_arrays_padded,
        manifest_rows_updated=manifest_rows_updated,
        metadata_updated=bool(metadata_changed),
    )


def harmonize_bit_cache(
    cache_root: str | Path,
    *,
    dry_run: bool = False,
) -> dict[str, Any]:
    root = Path(cache_root)
    if not root.is_dir():
        raise FileNotFoundError(f"Cache root does not exist: {root}")

    trim_summaries = {
        dataset: {
            key: str(value) if isinstance(value, Path) else value
            for key, value in asdict(
                trim_area6v_cache(root, dataset=dataset, dry_run=dry_run)
            ).items()
        }
        for dataset in ("brain2text24", "brain2text25")
        if (root / dataset).is_dir()
    }
    motor_summary = (
        None
        if not (root / "motor_data").is_dir()
        else asdict(harmonize_motor_data(root, dry_run=dry_run))
    )

    metadata_annotation_updates: dict[str, bool] = {}
    for dataset_root in _iter_dataset_roots(root):
        metadata_annotation_updates[dataset_root.name] = _annotate_dataset_metadata(
            dataset_root=dataset_root,
            default_stage1_included=dataset_root.name in BIT_STAGE1_DEFAULT_INCLUDED_DATASETS,
            dry_run=dry_run,
        )

    return {
        "cache_root": str(root),
        "dry_run": bool(dry_run),
        "feature_policy": BIT_CANONICAL_FEATURE_POLICY,
        "default_stage1_included_datasets": list(BIT_STAGE1_DEFAULT_INCLUDED_DATASETS),
        "default_stage1_excluded_datasets": list(BIT_STAGE1_DEFAULT_EXCLUDED_DATASETS),
        "area6v_trim_summaries": trim_summaries,
        "motor_data_padding_summary": motor_summary,
        "metadata_annotation_updates": metadata_annotation_updates,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    print(
        json.dumps(
            harmonize_bit_cache(
                cache_root=args.cache_root,
                dry_run=bool(args.dry_run),
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

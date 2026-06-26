"""Trim Brain2Text canonical cache feature arrays to area-6v columns.

The Stanford speechBCI TFRecord converter explicitly uses the first 128 TX
columns and first 128 spike-band-power columns as area 6v. This utility applies
that hard migration to existing canonical cache roots in place.
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


AREA6V_FEATURES = 128
FULL_RELEASE_FEATURES = 256
SOURCE_NOTE = (
    "Stanford speechBCI AnalysisExamples/makeTFRecordsFromSession.py states "
    "'first 128 columns = area 6v only' for tx1 and spikePow."
)


@dataclass(frozen=True)
class TrimSummary:
    cache_root: Path
    dataset: str
    dry_run: bool
    shard_count: int
    arrays_trimmed: int
    arrays_already_trimmed: int
    manifest_rows_updated: int


def _timestamp_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _atomic_save_npy(path: Path, array: np.ndarray) -> None:
    tmp_path = path.with_name(f".{path.name}.tmp.npy")
    np.save(tmp_path, array)
    tmp_path.replace(path)


def _trim_array_file(path: Path, *, dry_run: bool) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Missing cache array: {path}")
    arr = np.load(path, mmap_mode="r", allow_pickle=False)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D feature array at {path}, got shape {arr.shape}")
    width = int(arr.shape[1])
    if width == AREA6V_FEATURES:
        return "already_trimmed"
    if width < AREA6V_FEATURES:
        raise ValueError(f"Feature array at {path} has only {width} columns; cannot trim to {AREA6V_FEATURES}")
    if width != FULL_RELEASE_FEATURES:
        raise ValueError(
            f"Feature array at {path} has unexpected width {width}; expected {FULL_RELEASE_FEATURES} or {AREA6V_FEATURES}"
        )
    if not dry_run:
        trimmed = np.asarray(arr[:, :AREA6V_FEATURES])
        _atomic_save_npy(path, trimmed.astype(arr.dtype, copy=False))
    return "trimmed"


def _update_feature_counts(payload: dict[str, Any]) -> bool:
    changed = False
    for key in ("n_tx_features", "n_sbp_features"):
        if key in payload and int(payload.get(key) or 0) != AREA6V_FEATURES:
            payload[key] = AREA6V_FEATURES
            changed = True
    if "n_total_features" in payload and int(payload.get("n_total_features") or 0) != AREA6V_FEATURES * 2:
        payload["n_total_features"] = AREA6V_FEATURES * 2
        changed = True
    return changed


def _rewrite_manifest(manifest_path: Path, *, dry_run: bool) -> int:
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")
    rows: list[dict[str, Any]] = []
    changed = 0
    with manifest_path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if _update_feature_counts(row):
                changed += 1
            rows.append(row)
    if changed and not dry_run:
        tmp_path = manifest_path.with_name(f".{manifest_path.name}.tmp")
        with tmp_path.open("w") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")
        tmp_path.replace(manifest_path)
    return changed


def _rewrite_metadata(metadata_path: Path, *, dry_run: bool) -> None:
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata: {metadata_path}")
    metadata = json.loads(metadata_path.read_text())
    changed = False
    feature_layout = metadata.setdefault("feature_layout", {})
    if isinstance(feature_layout, dict):
        changed = _update_feature_counts(feature_layout) or changed

    shards = metadata.get("shards")
    if isinstance(shards, list):
        for shard in shards:
            if isinstance(shard, dict):
                changed = _update_feature_counts(shard) or changed

    provenance = metadata.setdefault("area6v_migration", {})
    if not isinstance(provenance, dict):
        provenance = {}
        metadata["area6v_migration"] = provenance
        changed = True
    migration_payload = {
        "area6v_only": True,
        "trimmed_feature_columns": [0, AREA6V_FEATURES],
        "removed_feature_columns": [AREA6V_FEATURES, FULL_RELEASE_FEATURES],
        "source_note": SOURCE_NOTE,
    }
    for key, value in migration_payload.items():
        if provenance.get(key) != value:
            provenance[key] = value
            changed = True
    if "migrated_utc" not in provenance:
        provenance["migrated_utc"] = _timestamp_utc()
        changed = True

    notes = metadata.setdefault("build_notes", [])
    if isinstance(notes, list):
        note = "Area-6v-only migration retained columns [0, 128) and removed BA44/IFG columns [128, 256)."
        if note not in notes:
            notes.append(note)
            changed = True

    if changed and not dry_run:
        tmp_path = metadata_path.with_name(f".{metadata_path.name}.tmp")
        tmp_path.write_text(json.dumps(metadata, indent=2) + "\n")
        tmp_path.replace(metadata_path)


def trim_area6v_cache(
    cache_root: str | Path,
    *,
    dataset: str = "brain2text24",
    dry_run: bool = False,
    backup_metadata: bool = True,
) -> TrimSummary:
    root = Path(cache_root)
    dataset_root = root / dataset
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")
    shard_root = dataset_root / "shards"
    if not shard_root.is_dir():
        raise FileNotFoundError(f"Shard root not found: {shard_root}")

    metadata_path = dataset_root / "metadata.json"
    manifest_path = dataset_root / "manifest.jsonl"
    if backup_metadata and not dry_run:
        for path in (metadata_path, manifest_path):
            backup_path = path.with_suffix(path.suffix + ".pre_area6v_backup")
            if path.exists() and not backup_path.exists():
                shutil.copy2(path, backup_path)

    arrays_trimmed = 0
    arrays_already_trimmed = 0
    shard_count = 0
    for shard_dir in sorted(path for path in shard_root.iterdir() if path.is_dir()):
        shard_count += 1
        for name in ("tx.npy", "sbp.npy"):
            result = _trim_array_file(shard_dir / name, dry_run=dry_run)
            if result == "trimmed":
                arrays_trimmed += 1
            elif result == "already_trimmed":
                arrays_already_trimmed += 1

    manifest_rows_updated = _rewrite_manifest(manifest_path, dry_run=dry_run)
    _rewrite_metadata(metadata_path, dry_run=dry_run)

    return TrimSummary(
        cache_root=root,
        dataset=dataset,
        dry_run=bool(dry_run),
        shard_count=shard_count,
        arrays_trimmed=arrays_trimmed,
        arrays_already_trimmed=arrays_already_trimmed,
        manifest_rows_updated=manifest_rows_updated,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--dataset", default="brain2text24")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-backup-metadata", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    summary = trim_area6v_cache(
        args.cache_root,
        dataset=str(args.dataset),
        dry_run=bool(args.dry_run),
        backup_metadata=not bool(args.no_backup_metadata),
    )
    print(
        json.dumps(
            {
                "cache_root": str(summary.cache_root),
                "dataset": summary.dataset,
                "dry_run": summary.dry_run,
                "shard_count": summary.shard_count,
                "arrays_trimmed": summary.arrays_trimmed,
                "arrays_already_trimmed": summary.arrays_already_trimmed,
                "manifest_rows_updated": summary.manifest_rows_updated,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

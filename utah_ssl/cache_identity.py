"""Stable identities for cache roots and dataset-to-root mappings.

Cache identities are shared infrastructure: cache copying uses them to detect
stale local mirrors, while normalization artifacts use them to prove which
physical data they describe.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Mapping


def cache_variant_name(cache_root: str | Path) -> str:
    """Return the stable storage-view name embedded in artifact paths."""

    name = Path(cache_root).name
    if "smoothed_sigma2p0" in name:
        return "smoothed_sigma2p0"
    if name == "cache_v1":
        return "raw"
    return name.replace("cache_v1_", "").replace("/", "_")


def list_directory_with_retries(path: Path, *, max_retries: int = 5) -> list[Path]:
    """List a directory with bounded retries for occasionally stalled mounts."""

    last_error: OSError | None = None
    for attempt in range(1, max_retries + 1):
        try:
            return sorted(path.iterdir(), key=lambda child: child.name)
        except OSError as exc:  # pragma: no cover - exercised when Drive stalls
            last_error = exc
            if attempt == max_retries:
                break
            print(f"directory scan retry {attempt}/{max_retries} failed for {path}: {exc}")
            time.sleep(min(10.0, float(attempt)))
    assert last_error is not None
    raise last_error


def _path_signature(path: Path) -> dict[str, int] | None:
    if not path.exists():
        return None
    stat = path.stat()
    return {
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _dataset_signature_payload(dataset_root: Path) -> dict[str, Any]:
    shard_root = dataset_root / "shards"
    shard_names: list[str] = []
    shard_scan_error: str | None = None
    if shard_root.exists():
        try:
            shard_names = [
                path.name
                for path in list_directory_with_retries(shard_root)
                if path.is_dir()
            ]
        except OSError as exc:  # pragma: no cover - exercised when Drive stalls
            shard_scan_error = str(exc)
            print(
                f"warning: failed to enumerate shards for signature under {shard_root}; "
                f"falling back to metadata-only signature fields: {exc}"
            )
    return {
        "dataset": dataset_root.name,
        "manifest": _path_signature(dataset_root / "manifest.jsonl"),
        "metadata": _path_signature(dataset_root / "metadata.json"),
        "shard_count": len(shard_names),
        "first_shard": shard_names[0] if shard_names else None,
        "last_shard": shard_names[-1] if shard_names else None,
        "shard_scan_error": shard_scan_error,
    }


def compute_cache_source_signature(cache_root: str | Path) -> str:
    """Identify a complete cache root from its datasets and storage metadata."""

    root = Path(cache_root)
    datasets = [
        _dataset_signature_payload(dataset_root)
        for dataset_root in (
            path
            for path in list_directory_with_retries(root)
            if path.is_dir() and (path / "metadata.json").exists()
        )
    ]
    payload = {
        "root": str(root),
        "datasets": datasets,
        "repack_summary": _path_signature(root / "repack_summary.json"),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def compute_dataset_cache_source_signature(
    dataset_cache_roots: Mapping[str, str | Path],
) -> str:
    """Identify an explicit dataset-to-cache-root mapping."""

    normalized = {
        str(dataset): Path(cache_root)
        for dataset, cache_root in sorted(dataset_cache_roots.items())
    }
    payload = {
        "kind": "dataset_cache_root_map_v1",
        "dataset_roots": {
            dataset: str(cache_root.resolve())
            for dataset, cache_root in normalized.items()
        },
        "datasets": [
            _dataset_signature_payload(cache_root / dataset)
            for dataset, cache_root in normalized.items()
        ],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


__all__ = [
    "cache_variant_name",
    "compute_cache_source_signature",
    "compute_dataset_cache_source_signature",
    "list_directory_with_retries",
]

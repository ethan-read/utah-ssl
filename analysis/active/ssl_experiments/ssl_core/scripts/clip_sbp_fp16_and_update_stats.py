"""Clip SBP caches to the published ceiling, store them as float16, and update stats.

This is intended for the POSSM SBP-only cache variants.  It is deliberately
non-destructive: each source cache is copied to a new destination root, with
only ``sbp.npy`` changed.  If a stats artifact is supplied, its finalized
mean/std values are updated from the exact source-to-destination first- and
second-moment deltas, so the full corpus does not need to be scanned twice.

The source and destination caches must have identical manifests and shard
topology.  The script preserves the existing cache layout and applies the
same row selection used by the stats artifacts:

* session scope: every other time bin (the current POSSM stride-2 contract);
* global scope: every selected time bin.

Example, pooled Stage 1 session stats::

    python analysis/active/ssl_experiments/ssl_core/scripts/clip_sbp_fp16_and_update_stats.py \
      --dataset-source-root brain2text24=/path/cache_v1_smoothed_sigma2p0 \
      --dataset-destination-root brain2text24=/path/cache_v1_sbpclip12500_fp16_smoothed \
      --dataset-source-root brain2text25=/path/cache_v1_possm_b2t25_sigma2p0 \
      --dataset-destination-root brain2text25=/path/cache_v1_possm_b2t25_sbpclip12500_fp16_smoothed \
      --dataset-source-split brain2text24=competition_train \
      --dataset-source-split brain2text25=train \
      --dataset-source-split brain2text25=val \
      --stats-path /path/old_session_stats.pt \
      --stats-output-path /path/new_session_stats.pt \
      --scope session

Run the command once more with only the B2T24 raw root and ``--scope global``
for the Stage 2 artifact.  The old stats artifact must correspond to the
source roots supplied on the command line.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch

EXPERIMENTS_DIR = Path(__file__).resolve().parents[2]
REPO_ROOT = Path(__file__).resolve().parents[5]
for _path in (REPO_ROOT, EXPERIMENTS_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from masked_ssl.cache import (  # noqa: E402
    _cache_variant_name,
    _compute_dataset_cache_source_signature,
)
from ssl_core.normalization_stats import (  # noqa: E402
    extract_feature_stats_entries,
    write_feature_stats_artifact,
)


DEFAULT_CLIP_THRESHOLD = 12_500.0
DEFAULT_SBP_DIM = 128
SESSION_STATS_BIN_STRIDE = 2
TIME_OFFSETS_NAME = "time_offsets.npy"


@dataclass(frozen=True)
class CacheMap:
    dataset: str
    source_root: Path
    destination_root: Path


@dataclass(frozen=True)
class ManifestRow:
    dataset: str
    session_id: str
    subject_id: str | None
    shard_relpath: str
    example_index: int


def _timestamp_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def _parse_dataset_root_args(values: Sequence[str] | None, *, option: str) -> dict[str, Path]:
    parsed: dict[str, Path] = {}
    for value in values or ():
        dataset, separator, root = str(value).partition("=")
        dataset = dataset.strip()
        root = root.strip()
        if not separator or not dataset or not root:
            raise ValueError(f"{option} values must have the form DATASET=ROOT")
        if dataset in parsed:
            raise ValueError(f"{option} specified more than once for dataset {dataset!r}")
        parsed[dataset] = Path(root).expanduser()
    return dict(sorted(parsed.items()))


def _parse_dataset_split_args(values: Sequence[str] | None) -> dict[str, tuple[str, ...]]:
    parsed: dict[str, set[str]] = defaultdict(set)
    for value in values or ():
        dataset, separator, source_split = str(value).partition("=")
        dataset = dataset.strip()
        source_split = source_split.strip().lower()
        if not separator or not dataset or not source_split:
            raise ValueError(
                "--dataset-source-split values must have the form DATASET=SOURCE_SPLIT"
            )
        parsed[dataset].add(source_split)
    return {
        dataset: tuple(sorted(source_splits))
        for dataset, source_splits in sorted(parsed.items())
    }


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _load_manifest_rows(
    *,
    dataset: str,
    root: Path,
    source_splits: Iterable[str],
    scope: str,
    sbp_dim: int,
) -> list[ManifestRow]:
    dataset_root = root / dataset
    manifest_path = dataset_root / "manifest.jsonl"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")
    allowed_splits = {str(value).strip().lower() for value in source_splits}
    rows: list[ManifestRow] = []
    for payload in _load_jsonl(manifest_path):
        source_split = str(payload.get("source_split", "")).strip().lower()
        if allowed_splits and source_split not in allowed_splits:
            continue
        n_sbp_features = int(payload.get("n_sbp_features", 0) or 0)
        has_sbp = bool(payload.get("has_sbp", n_sbp_features > 0))
        compatible = has_sbp and n_sbp_features >= int(sbp_dim)
        if scope == "global":
            # Match build_competition_split_problem: Stage 2 global stats use
            # labeled rows compatible with the requested signal.
            if not bool(payload.get("has_labels", False)) or not compatible:
                continue
        elif scope == "session":
            # Match prepare_cache_context's strict preflight for Stage 1.
            if not compatible:
                raise ValueError(
                    f"Selected session row is incompatible with SBP-{sbp_dim}: "
                    f"dataset={dataset!r}, source_split={source_split!r}, "
                    f"example_index={payload.get('example_index')!r}, "
                    f"has_sbp={has_sbp}, n_sbp_features={n_sbp_features}"
                )
        else:
            raise ValueError(f"Unsupported stats scope: {scope!r}")
        rows.append(
            ManifestRow(
                dataset=str(dataset),
                session_id=str(payload["session_id"]),
                subject_id=(
                    str(payload["subject_id"])
                    if payload.get("subject_id") is not None
                    else None
                ),
                shard_relpath=str(payload["shard_relpath"]),
                example_index=int(payload["example_index"]),
            )
        )
    if not rows:
        raise ValueError(
            f"No manifest rows selected for {dataset!r} under {root}; "
            f"source_splits={sorted(allowed_splits)}"
        )
    return rows


def _resolve_boundary_key(row: ManifestRow, boundary_key_mode: str) -> str:
    if boundary_key_mode == "session":
        boundary_id = row.session_id
    elif boundary_key_mode == "subject_if_available":
        boundary_id = row.subject_id or row.session_id
    else:
        raise ValueError(f"Unsupported boundary key mode: {boundary_key_mode!r}")
    return f"{row.dataset}:{boundary_id}"


def _dataset_name_for_relative_path(relative_path: Path) -> str | None:
    return relative_path.parts[0] if relative_path.parts else None


def _clip_sbp_array(
    source: np.ndarray,
    *,
    clip_threshold: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    if source.ndim != 2:
        raise ValueError(f"SBP arrays must be 2D, got shape {source.shape}")
    with np.errstate(over="ignore", invalid="ignore"):
        source_float32 = np.asarray(source, dtype=np.float32)
    if not bool(np.all(np.isfinite(source_float32))):
        raise ValueError("Cannot convert non-finite SBP values to float16")
    clipped = np.minimum(source_float32, np.float32(clip_threshold))
    converted = np.asarray(clipped, dtype=np.float16)
    if not bool(np.all(np.isfinite(converted))):
        raise ValueError("SBP conversion produced non-finite float16 values")
    roundtrip = np.asarray(converted, dtype=np.float32)
    summary = {
        "values": int(source_float32.size),
        "values_above_clip_threshold": int(np.count_nonzero(source_float32 > clip_threshold)),
        "values_changed_after_clip_and_fp16": int(
            np.count_nonzero(roundtrip != source_float32)
        ),
        "source_dtype": str(source.dtype),
        "destination_dtype": str(converted.dtype),
        "source_min": float(np.min(source_float32)) if source_float32.size else None,
        "source_max": float(np.max(source_float32)) if source_float32.size else None,
        "destination_min": float(np.min(roundtrip)) if roundtrip.size else None,
        "destination_max": float(np.max(roundtrip)) if roundtrip.size else None,
    }
    return converted, summary


def _transform_cache_pair(
    *,
    source_root: Path,
    destination_root: Path,
    datasets: Sequence[str],
    clip_threshold: float,
) -> dict[str, Any]:
    source_root = source_root.expanduser().resolve()
    destination_root = destination_root.expanduser().resolve()
    if not source_root.is_dir():
        raise FileNotFoundError(f"Source cache root does not exist: {source_root}")
    if source_root == destination_root:
        raise ValueError("Source and destination cache roots must be different")
    if destination_root.exists():
        raise FileExistsError(
            f"Destination already exists: {destination_root}. "
            "Choose a new versioned root; the script never overwrites caches."
        )
    try:
        destination_root.relative_to(source_root)
    except ValueError:
        pass
    else:
        raise ValueError(
            "Destination cache root must not be inside the source cache root: "
            f"source={source_root}, destination={destination_root}"
        )
    try:
        source_root.relative_to(destination_root)
    except ValueError:
        pass
    else:
        raise ValueError(
            "Source cache root must not be inside the destination cache root: "
            f"source={source_root}, destination={destination_root}"
        )

    dataset_set = {str(dataset) for dataset in datasets}
    for dataset in sorted(dataset_set):
        dataset_root = source_root / dataset
        if not (dataset_root / "metadata.json").exists():
            raise FileNotFoundError(f"Missing dataset metadata: {dataset_root / 'metadata.json'}")
        if not (dataset_root / "manifest.jsonl").exists():
            raise FileNotFoundError(f"Missing dataset manifest: {dataset_root / 'manifest.jsonl'}")

    partial_root = destination_root.with_name(f".{destination_root.name}.partial")
    if partial_root.exists():
        raise FileExistsError(f"Temporary destination already exists: {partial_root}")

    dataset_summaries: dict[str, dict[str, Any]] = {
        dataset: {
            "sbp_files": 0,
            "values": 0,
            "values_above_clip_threshold": 0,
            "values_changed_after_clip_and_fp16": 0,
            "source_min": None,
            "source_max": None,
            "destination_min": None,
            "destination_max": None,
        }
        for dataset in sorted(dataset_set)
    }
    try:
        for source_path in sorted(source_root.rglob("*")):
            relative_path = source_path.relative_to(source_root)
            destination_path = partial_root / relative_path
            if source_path.is_dir():
                destination_path.mkdir(parents=True, exist_ok=True)
                continue
            destination_path.parent.mkdir(parents=True, exist_ok=True)
            dataset = _dataset_name_for_relative_path(relative_path)
            if source_path.name == "sbp.npy" and dataset in dataset_set:
                source_array = np.load(source_path, allow_pickle=False)
                converted, file_summary = _clip_sbp_array(
                    source_array,
                    clip_threshold=float(clip_threshold),
                )
                np.save(destination_path, converted, allow_pickle=False)
                summary = dataset_summaries[dataset]
                summary["sbp_files"] += 1
                for key in (
                    "values",
                    "values_above_clip_threshold",
                    "values_changed_after_clip_and_fp16",
                ):
                    summary[key] += int(file_summary[key])
                for key in ("source_min", "source_max", "destination_min", "destination_max"):
                    value = file_summary[key]
                    if value is None:
                        continue
                    previous = summary[key]
                    if previous is None:
                        summary[key] = value
                    elif key.endswith("min"):
                        summary[key] = min(float(previous), float(value))
                    else:
                        summary[key] = max(float(previous), float(value))
            else:
                shutil.copy2(source_path, destination_path)

        for dataset in sorted(dataset_set):
            summary = dataset_summaries[dataset]
            if int(summary["sbp_files"]) == 0:
                raise FileNotFoundError(f"No sbp.npy files found for {dataset} under {source_root}")
            metadata_path = partial_root / dataset / "metadata.json"
            metadata = json.loads(metadata_path.read_text())
            metadata["sbp_storage_policy"] = "clip_upper_then_float16"
            metadata["sbp_storage_dtype"] = "float16"
            metadata["sbp_clip_threshold"] = float(clip_threshold)
            metadata["sbp_transform_provenance"] = {
                "source_cache_root": str(source_root),
                "source_cache_name": source_root.name,
                "clip_threshold": float(clip_threshold),
                "destination_dtype": "float16",
                "created_utc": _timestamp_utc(),
            }
            metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")

        root_summary = {
            "kind": "sbp_clip_fp16_cache_transform",
            "created_utc": _timestamp_utc(),
            "source_cache_root": str(source_root),
            "destination_cache_root": str(destination_root),
            "datasets": sorted(dataset_set),
            "clip_threshold": float(clip_threshold),
            "destination_dtype": "float16",
            "dataset_summaries": dataset_summaries,
        }
        (partial_root / "sbp_clip_fp16_transform_summary.json").write_text(
            json.dumps(root_summary, indent=2, default=_json_default) + "\n"
        )
        partial_root.rename(destination_root)
    except Exception:
        shutil.rmtree(partial_root, ignore_errors=True)
        raise

    return {
        "source_root": source_root,
        "destination_root": destination_root,
        "datasets": sorted(dataset_set),
        "dataset_summaries": dataset_summaries,
    }


def transform_cache_maps(
    cache_maps: Sequence[CacheMap],
    *,
    clip_threshold: float,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Path, Path], list[str]] = defaultdict(list)
    for cache_map in cache_maps:
        grouped[
            (cache_map.source_root.expanduser().resolve(), cache_map.destination_root.expanduser().resolve())
        ].append(cache_map.dataset)
    return [
        _transform_cache_pair(
            source_root=source_root,
            destination_root=destination_root,
            datasets=sorted(datasets),
            clip_threshold=float(clip_threshold),
        )
        for (source_root, destination_root), datasets in sorted(grouped.items(), key=lambda item: str(item[0]))
    ]


def _load_sbp_shard(
    *,
    source_shard: Path,
    destination_shard: Path,
    clip_threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    source_offsets = np.load(source_shard / TIME_OFFSETS_NAME, allow_pickle=False)
    destination_offsets = np.load(destination_shard / TIME_OFFSETS_NAME, allow_pickle=False)
    if not np.array_equal(source_offsets, destination_offsets):
        raise ValueError(f"Time offsets differ between source and destination: {source_shard}")
    source_sbp = np.load(source_shard / "sbp.npy", allow_pickle=False)
    destination_sbp = np.load(destination_shard / "sbp.npy", allow_pickle=False)
    if source_sbp.shape != destination_sbp.shape:
        raise ValueError(f"SBP shapes differ between source and destination: {source_shard}")
    if destination_sbp.dtype != np.dtype(np.float16):
        raise ValueError(
            f"Destination SBP must be float16, got {destination_sbp.dtype}: "
            f"{destination_shard / 'sbp.npy'}"
        )
    if destination_sbp.size and not bool(np.all(destination_sbp <= np.float16(clip_threshold))):
        raise ValueError(
            f"Destination SBP contains values above the expected clip ceiling: "
            f"{destination_shard / 'sbp.npy'}"
        )
    return source_offsets, source_sbp, destination_sbp


def _compute_moment_deltas(
    *,
    cache_maps: Mapping[str, CacheMap],
    source_splits_by_dataset: Mapping[str, Sequence[str]],
    scope: str,
    boundary_key_mode: str,
    stride: int,
    feature_dim: int,
    sbp_feature_start: int,
    sbp_dim: int,
    clip_threshold: float,
    stats_path: Path,
) -> tuple[
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, Any],
]:
    delta_sum: dict[str, np.ndarray] = defaultdict(lambda: np.zeros(feature_dim, dtype=np.float64))
    delta_sum2: dict[str, np.ndarray] = defaultdict(lambda: np.zeros(feature_dim, dtype=np.float64))
    counts: dict[str, np.ndarray] = defaultdict(
        lambda: np.zeros(feature_dim, dtype=np.float64)
    )
    selected_summary: dict[str, Any] = {"scope": scope, "stride": int(stride), "datasets": {}}

    for dataset, cache_map in sorted(cache_maps.items()):
        source_manifest = cache_map.source_root / dataset / "manifest.jsonl"
        destination_manifest = cache_map.destination_root / dataset / "manifest.jsonl"
        if not destination_manifest.exists():
            raise FileNotFoundError(f"Missing destination manifest: {destination_manifest}")
        if source_manifest.read_bytes() != destination_manifest.read_bytes():
            raise ValueError(
                f"Source and destination manifests differ for {dataset}; "
                "algebraic stats updates require identical row topology."
            )
        rows = _load_manifest_rows(
            dataset=dataset,
            root=cache_map.source_root,
            source_splits=source_splits_by_dataset.get(dataset, ()),
            scope=scope,
            sbp_dim=int(sbp_dim),
        )
        rows_by_shard: dict[str, list[ManifestRow]] = defaultdict(list)
        for row in rows:
            rows_by_shard[row.shard_relpath].append(row)
        dataset_bins = 0
        for shard_relpath, shard_rows in sorted(rows_by_shard.items()):
            source_shard = cache_map.source_root / shard_relpath
            destination_shard = cache_map.destination_root / shard_relpath
            if not destination_shard.is_dir():
                raise FileNotFoundError(f"Missing destination shard: {destination_shard}")
            source_offsets, source_sbp, destination_sbp = _load_sbp_shard(
                source_shard=source_shard,
                destination_shard=destination_shard,
                clip_threshold=float(clip_threshold),
            )
            sbp_column_start = 0
            stop_column = sbp_column_start + int(sbp_dim)
            if source_sbp.ndim != 2 or stop_column > source_sbp.shape[1]:
                raise ValueError(
                    f"SBP column selection [{sbp_column_start}:{stop_column}] is invalid for "
                    f"{source_shard / 'sbp.npy'} with shape {source_sbp.shape}"
                )
            for row in shard_rows:
                start = int(source_offsets[row.example_index])
                stop = int(source_offsets[row.example_index + 1])
                source_window = np.asarray(
                    source_sbp[start:stop:int(stride), sbp_column_start:stop_column],
                    dtype=np.float64,
                )
                destination_window = np.asarray(
                    destination_sbp[start:stop:int(stride), sbp_column_start:stop_column],
                    dtype=np.float64,
                )
                if source_window.shape != destination_window.shape:
                    raise ValueError(f"SBP windows differ for {row.shard_relpath}:{row.example_index}")
                if not bool(np.all(np.isfinite(source_window))) or not bool(np.all(np.isfinite(destination_window))):
                    raise ValueError(f"Non-finite SBP values found in {row.shard_relpath}:{row.example_index}")
                key = "global" if scope == "global" else _resolve_boundary_key(row, boundary_key_mode)
                delta = destination_window - source_window
                delta_squared = np.square(destination_window) - np.square(source_window)
                feature_slice = slice(int(sbp_feature_start), int(sbp_feature_start) + int(sbp_dim))
                delta_sum[key][feature_slice] += delta.sum(axis=0)
                delta_sum2[key][feature_slice] += delta_squared.sum(axis=0)
                counts[key][feature_slice] += int(source_window.shape[0])
                dataset_bins += int(source_window.shape[0])
        selected_summary["datasets"][dataset] = {
            "manifest_rows": int(len(rows)),
            "selected_bins": int(dataset_bins),
            "source_splits": sorted(str(value) for value in source_splits_by_dataset.get(dataset, ())),
        }

    if not counts:
        raise RuntimeError(f"No stats rows were selected from sources for {stats_path}")
    selected_summary["keys"] = int(len(counts))
    selected_summary["selected_bins_total"] = int(
        sum(dataset_summary["selected_bins"] for dataset_summary in selected_summary["datasets"].values())
    )
    return dict(delta_sum), dict(delta_sum2), dict(counts), selected_summary


def update_stats_artifact_algebraically(
    *,
    stats_path: str | Path,
    output_path: str | Path,
    cache_maps: Mapping[str, CacheMap],
    source_splits_by_dataset: Mapping[str, Sequence[str]],
    scope: str,
    boundary_key_mode: str = "session",
    stride: int | None = None,
    sbp_feature_start: int = 0,
    sbp_dim: int = DEFAULT_SBP_DIM,
    clip_threshold: float = DEFAULT_CLIP_THRESHOLD,
) -> dict[str, Any]:
    stats_path = Path(stats_path).expanduser().resolve()
    output_path = Path(output_path).expanduser().resolve()
    if not stats_path.exists():
        raise FileNotFoundError(f"Stats artifact does not exist: {stats_path}")
    if output_path.exists() or output_path.with_suffix(".json").exists():
        raise FileExistsError(
            f"Stats output already exists: {output_path}. "
            "Choose a new path; the script never overwrites stats by default."
        )
    if scope not in {"session", "global"}:
        raise ValueError(f"Unsupported stats scope: {scope!r}")
    resolved_stride = int(stride if stride is not None else (SESSION_STATS_BIN_STRIDE if scope == "session" else 1))
    if resolved_stride <= 0:
        raise ValueError("stride must be positive")
    if scope == "global" and resolved_stride != 1:
        raise ValueError("Global stats must use stride=1")

    try:
        payload = torch.load(stats_path, map_location="cpu", weights_only=False)
    except TypeError:  # pragma: no cover - compatibility with older PyTorch releases
        payload = torch.load(stats_path, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise ValueError(f"Stats artifact must contain a mapping: {stats_path}")
    payload_scope, old_entries = extract_feature_stats_entries(payload)
    if payload_scope != scope:
        raise ValueError(
            f"Stats artifact scope is {payload_scope!r}, but requested update scope is {scope!r}"
        )
    metadata = dict(payload.get("metadata") or {})
    unknown_split_datasets = sorted(set(source_splits_by_dataset) - set(cache_maps))
    if unknown_split_datasets:
        raise ValueError(
            "Source-split selections reference unmapped datasets: "
            + ", ".join(unknown_split_datasets)
        )
    requested_dataset_plan = {
        dataset: sorted(str(split).strip().lower() for split in splits)
        for dataset, splits in sorted(source_splits_by_dataset.items())
    }
    if scope == "session":
        artifact_dataset_plan = metadata.get("dataset_plan")
        if isinstance(artifact_dataset_plan, Mapping):
            normalized_artifact_plan = {
                str(dataset): sorted(str(split).strip().lower() for split in splits)
                for dataset, splits in sorted(artifact_dataset_plan.items())
            }
            if normalized_artifact_plan != requested_dataset_plan:
                raise ValueError(
                    "Requested session source splits do not match the old stats artifact "
                    f"dataset_plan: artifact={normalized_artifact_plan}, "
                    f"requested={requested_dataset_plan}"
                )
        artifact_boundary_mode = metadata.get("boundary_key_mode")
        if artifact_boundary_mode is not None and str(artifact_boundary_mode) != str(boundary_key_mode):
            raise ValueError(
                "Requested boundary key mode does not match the old stats artifact: "
                f"artifact={artifact_boundary_mode!r}, requested={boundary_key_mode!r}"
            )
        artifact_stride = metadata.get("session_stats_bin_stride")
        if artifact_stride is not None and int(artifact_stride) != resolved_stride:
            raise ValueError(
                "Requested session stride does not match the old stats artifact: "
                f"artifact={artifact_stride}, requested={resolved_stride}"
            )
    elif scope == "global":
        artifact_dataset = metadata.get("dataset")
        if artifact_dataset is not None and set(cache_maps) != {str(artifact_dataset)}:
            raise ValueError(
                "Mapped datasets do not match the old global stats artifact: "
                f"artifact={artifact_dataset!r}, requested={sorted(cache_maps)}"
            )
        artifact_train_split = metadata.get("train_split_name")
        if artifact_train_split is not None:
            expected_global_plan = {
                str(artifact_dataset or next(iter(cache_maps))): [str(artifact_train_split).lower()]
            }
            if requested_dataset_plan != expected_global_plan:
                raise ValueError(
                    "Requested global source split does not match the old stats artifact: "
                    f"artifact={expected_global_plan}, requested={requested_dataset_plan}"
                )
    old_signature = metadata.get("source_cache_signature")
    expected_source_signature = _compute_dataset_cache_source_signature(
        {dataset: cache_map.source_root for dataset, cache_map in sorted(cache_maps.items())}
    )
    if old_signature and str(old_signature) != expected_source_signature:
        raise ValueError(
            "Stats artifact source signature does not match the supplied source roots. "
            f"artifact={old_signature}, supplied={expected_source_signature}"
        )

    feature_dims = {int(torch.as_tensor(mean).numel()) for mean, _ in old_entries.values()}
    if len(feature_dims) != 1:
        raise ValueError(f"Stats artifact entries have inconsistent feature dimensions: {feature_dims}")
    feature_dim = next(iter(feature_dims))
    if int(sbp_feature_start) < 0 or int(sbp_feature_start) + int(sbp_dim) > feature_dim:
        raise ValueError(
            f"SBP feature slice [{sbp_feature_start}:{sbp_feature_start + sbp_dim}] "
            f"does not fit stats feature dimension {feature_dim}"
        )

    delta_sum, delta_sum2, counts, selected_summary = _compute_moment_deltas(
        cache_maps=cache_maps,
        source_splits_by_dataset=source_splits_by_dataset,
        scope=scope,
        boundary_key_mode=boundary_key_mode,
        stride=resolved_stride,
        feature_dim=feature_dim,
        sbp_feature_start=int(sbp_feature_start),
        sbp_dim=int(sbp_dim),
        clip_threshold=float(clip_threshold),
        stats_path=stats_path,
    )
    unknown_delta_keys = sorted(set(delta_sum) - set(old_entries))
    if unknown_delta_keys:
        raise ValueError(
            "Selected source rows produced stats keys absent from the old artifact: "
            + ", ".join(unknown_delta_keys[:10])
        )
    missing_delta_keys = sorted(set(old_entries) - set(delta_sum))
    if missing_delta_keys:
        raise ValueError(
            "Selected source rows did not cover stats keys in the old artifact: "
            + ", ".join(missing_delta_keys[:10])
        )

    new_entries: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for key, (old_mean, old_std) in old_entries.items():
        mean64 = np.asarray(old_mean.detach().cpu().numpy(), dtype=np.float64)
        std64 = np.asarray(old_std.detach().cpu().numpy(), dtype=np.float64)
        if key not in counts or not bool(np.any(counts[key] > 0)):
            new_entries[key] = (old_mean.clone(), old_std.clone())
            continue
        present = counts[key] > 0
        new_mean64 = mean64.copy()
        old_second_moment64 = np.square(mean64) + np.square(std64)
        new_second_moment64 = old_second_moment64.copy()
        new_mean64[present] += delta_sum[key][present] / counts[key][present]
        new_second_moment64[present] += delta_sum2[key][present] / counts[key][present]
        new_variance64 = np.square(std64)
        new_variance64[present] = np.maximum(
            new_second_moment64[present] - np.square(new_mean64[present]),
            1e-6,
        )
        new_entries[key] = (
            torch.from_numpy(np.asarray(new_mean64, dtype=np.float32)),
            torch.from_numpy(np.asarray(np.sqrt(new_variance64), dtype=np.float32)),
        )

    destination_roots = {
        dataset: cache_map.destination_root.resolve()
        for dataset, cache_map in sorted(cache_maps.items())
    }
    primary_destination = destination_roots[sorted(destination_roots)[0]]
    updated_metadata = dict(metadata)
    updated_metadata.update(
        {
            "created_utc": _timestamp_utc(),
            "source_cache_root": str(primary_destination),
            "source_cache_name": primary_destination.name,
            "source_cache_variant": _cache_variant_name(primary_destination),
            "source_cache_signature": _compute_dataset_cache_source_signature(destination_roots),
            "source_cache_roots": {dataset: str(root) for dataset, root in destination_roots.items()},
            "sbp_transform": {
                "clip_threshold": float(clip_threshold),
                "storage_dtype": "float16",
                "update_method": "algebraic_moment_update",
                "source_stats_artifact": str(stats_path),
                "source_cache_signature": expected_source_signature,
                "destination_cache_signature": _compute_dataset_cache_source_signature(destination_roots),
                "selected_rows": selected_summary,
            },
        }
    )
    output_payload = write_feature_stats_artifact(
        output_path=output_path,
        scope=scope,
        entries=new_entries,
        metadata=updated_metadata,
    )
    return {
        "output_path": output_path,
        "metadata_path": output_path.with_suffix(".json"),
        "scope": scope,
        "entries": len(new_entries),
        "selected_summary": selected_summary,
        "source_cache_signature": expected_source_signature,
        "destination_cache_signature": updated_metadata["source_cache_signature"],
        "payload": output_payload,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-source-root",
        action="append",
        required=True,
        help="Source cache root in DATASET=ROOT form. Repeat once per dataset.",
    )
    parser.add_argument(
        "--dataset-destination-root",
        action="append",
        required=True,
        help="New cache root in DATASET=ROOT form. Repeat once per dataset.",
    )
    parser.add_argument(
        "--dataset-source-split",
        action="append",
        default=None,
        help="Stats row selection in DATASET=SOURCE_SPLIT form; repeat for multiple splits.",
    )
    parser.add_argument("--stats-path", type=Path, default=None)
    parser.add_argument("--stats-output-path", type=Path, default=None)
    parser.add_argument("--scope", choices=("session", "global"), default=None)
    parser.add_argument(
        "--boundary-key-mode",
        choices=("session", "subject_if_available"),
        default="session",
    )
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--sbp-dim", type=int, default=DEFAULT_SBP_DIM)
    parser.add_argument("--sbp-feature-start", type=int, default=0)
    parser.add_argument("--clip-threshold", type=float, default=DEFAULT_CLIP_THRESHOLD)
    parser.add_argument(
        "--skip-cache-transform",
        action="store_true",
        help="Use existing destination roots instead of creating them.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    source_roots = _parse_dataset_root_args(
        args.dataset_source_root,
        option="--dataset-source-root",
    )
    destination_roots = _parse_dataset_root_args(
        args.dataset_destination_root,
        option="--dataset-destination-root",
    )
    if set(source_roots) != set(destination_roots):
        raise ValueError(
            "Source and destination roots must name the same datasets: "
            f"source={sorted(source_roots)}, destination={sorted(destination_roots)}"
        )
    if float(args.clip_threshold) <= 0:
        raise ValueError("clip threshold must be positive")
    cache_maps = {
        dataset: CacheMap(
            dataset=dataset,
            source_root=source_roots[dataset],
            destination_root=destination_roots[dataset],
        )
        for dataset in sorted(source_roots)
    }

    transform_results: list[dict[str, Any]] = []
    if not args.skip_cache_transform:
        transform_results = transform_cache_maps(
            list(cache_maps.values()),
            clip_threshold=float(args.clip_threshold),
        )
    else:
        for cache_map in cache_maps.values():
            if not cache_map.destination_root.is_dir():
                raise FileNotFoundError(f"Destination cache root does not exist: {cache_map.destination_root}")

    stats_result: dict[str, Any] | None = None
    if args.stats_path is not None or args.stats_output_path is not None:
        if args.stats_path is None or args.stats_output_path is None or args.scope is None:
            raise ValueError("--stats-path, --stats-output-path, and --scope must be supplied together")
        source_splits = _parse_dataset_split_args(args.dataset_source_split)
        missing_splits = sorted(set(cache_maps) - set(source_splits))
        if missing_splits:
            raise ValueError(
                "Stats updates require --dataset-source-split for every mapped dataset; "
                "missing: " + ", ".join(missing_splits)
            )
        stats_result = update_stats_artifact_algebraically(
            stats_path=args.stats_path,
            output_path=args.stats_output_path,
            cache_maps=cache_maps,
            source_splits_by_dataset=source_splits,
            scope=args.scope,
            boundary_key_mode=args.boundary_key_mode,
            stride=args.stride,
            sbp_feature_start=int(args.sbp_feature_start),
            sbp_dim=int(args.sbp_dim),
            clip_threshold=float(args.clip_threshold),
        )

    print(
        json.dumps(
            {
                "transform_results": transform_results,
                "stats_result": (
                    {key: value for key, value in stats_result.items() if key != "payload"}
                    if stats_result is not None
                    else None
                ),
            },
            indent=2,
            default=_json_default,
        )
    )


if __name__ == "__main__":
    main()

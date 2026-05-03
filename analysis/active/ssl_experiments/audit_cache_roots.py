from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch

from masked_ssl.cache import (
    CacheAccessConfig,
    _compute_cache_source_signature,
    get_sampling_plan,
    load_cache_smoothing_provenance,
    load_dataset_metadata,
    prepare_cache_context,
)


DEFAULT_DATASET = "brain2text24"
DEFAULT_SEGMENT_BINS = 80
DEFAULT_FEATURE_MODES = ("tx_only", "tx_sbp")


@dataclass(frozen=True)
class RootAuditInput:
    cache_root: Path
    dataset_names: tuple[str, ...]
    segment_bins: int
    feature_modes: tuple[str, ...]
    sample_shards: int = 3
    deep_array_check: bool = False


@dataclass(frozen=True)
class StatsAuditInput:
    stats_path: Path


def _timestamp_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _sha256_file(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _compute_structural_root_signature(root: Path) -> str | None:
    if not root.exists() or not root.is_dir():
        return None
    datasets = []
    for dataset_root in sorted(path for path in root.iterdir() if path.is_dir()):
        shard_root = dataset_root / "shards"
        shard_names = sorted(path.name for path in shard_root.iterdir() if path.is_dir()) if shard_root.exists() else []
        datasets.append(
            {
                "dataset": dataset_root.name,
                "manifest_sha256": _sha256_file(dataset_root / "manifest.jsonl"),
                "metadata_sha256": _sha256_file(dataset_root / "metadata.json"),
                "shard_count": len(shard_names),
                "first_shard": shard_names[0] if shard_names else None,
                "last_shard": shard_names[-1] if shard_names else None,
            }
        )
    payload = {
        "datasets": datasets,
        "repack_summary_sha256": _sha256_file(root / "repack_summary.json"),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _quantiles(values: Sequence[int | float]) -> dict[str, float | None]:
    if not values:
        return {"min": None, "p10": None, "p50": None, "p90": None, "p99": None, "max": None}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "min": float(np.min(arr)),
        "p10": float(np.quantile(arr, 0.10)),
        "p50": float(np.quantile(arr, 0.50)),
        "p90": float(np.quantile(arr, 0.90)),
        "p99": float(np.quantile(arr, 0.99)),
        "max": float(np.max(arr)),
    }


def _iter_manifest_rows(manifest_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with manifest_path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _infer_shard_dir(dataset_root: Path, shard_id: str) -> Path:
    shard_root = dataset_root / "shards"
    if shard_root.exists():
        return shard_root / shard_id
    return dataset_root / shard_id


def _sample_shard_ids(
    metadata: dict[str, Any] | None,
    manifest_rows: Sequence[dict[str, Any]],
    *,
    sample_shards: int,
    deep_array_check: bool,
) -> list[str]:
    shard_ids: list[str] = []
    if isinstance(metadata, dict):
        for shard in metadata.get("shards", []):
            shard_id = shard.get("shard_id")
            if shard_id is not None:
                shard_ids.append(str(shard_id))
    if not shard_ids:
        shard_ids = sorted(
            {
                Path(str(row.get("shard_relpath", ""))).name
                for row in manifest_rows
                if row.get("shard_relpath")
            }
        )
    if deep_array_check:
        return sorted(dict.fromkeys(shard_ids))
    ordered = sorted(dict.fromkeys(shard_ids))
    if len(ordered) <= sample_shards:
        return ordered
    if sample_shards <= 1:
        return [ordered[0]]
    idxs = np.linspace(0, len(ordered) - 1, num=sample_shards, dtype=int)
    return [ordered[int(idx)] for idx in idxs]


def _scan_dense_array(array: np.ndarray, *, chunk_rows: int = 4096) -> tuple[np.ndarray, np.ndarray]:
    if array.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {array.shape}")
    n_features = int(array.shape[1])
    any_nonzero = np.zeros(n_features, dtype=bool)
    any_nonfinite = np.zeros(n_features, dtype=bool)
    is_float = np.issubdtype(array.dtype, np.floating)
    for start in range(0, int(array.shape[0]), int(chunk_rows)):
        stop = min(start + int(chunk_rows), int(array.shape[0]))
        chunk = np.asarray(array[start:stop])
        any_nonzero |= np.any(chunk != 0, axis=0)
        if is_float:
            any_nonfinite |= ~np.isfinite(chunk).all(axis=0)
    return ~any_nonzero, any_nonfinite


def _summarize_array_consistency(
    dataset_root: Path,
    manifest_rows: Sequence[dict[str, Any]],
    metadata: dict[str, Any] | None,
    *,
    sample_shards: int,
    deep_array_check: bool,
    dataset_name: str,
) -> dict[str, Any]:
    shard_ids = _sample_shard_ids(
        metadata,
        manifest_rows,
        sample_shards=sample_shards,
        deep_array_check=deep_array_check,
    )
    rows_by_shard: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in manifest_rows:
        shard_relpath = row.get("shard_relpath")
        if shard_relpath:
            rows_by_shard[Path(str(shard_relpath)).name].append(row)

    checks: list[dict[str, Any]] = []
    totals = {
        "checked_shards": 0,
        "missing_tx_files": 0,
        "missing_sbp_files": 0,
        "missing_time_offsets_files": 0,
        "tx_width_mismatches": 0,
        "sbp_width_mismatches": 0,
        "row_count_mismatches": 0,
        "manifest_length_mismatches": 0,
        "bad_example_index": 0,
    }

    feature_layout = metadata.get("feature_layout", {}) if isinstance(metadata, dict) else {}
    expected_tx = feature_layout.get("n_tx_features")
    expected_sbp = feature_layout.get("n_sbp_features")

    for shard_id in shard_ids:
        shard_dir = _infer_shard_dir(dataset_root, shard_id)
        tx_path = shard_dir / "tx.npy"
        sbp_path = shard_dir / "sbp.npy"
        time_offsets_path = shard_dir / "time_offsets.npy"
        shard_rows = rows_by_shard.get(shard_id, [])
        shard_report: dict[str, Any] = {
            "shard_id": shard_id,
            "path": str(shard_dir),
            "exists": shard_dir.exists(),
            "tx_exists": tx_path.exists(),
            "sbp_exists": sbp_path.exists(),
            "time_offsets_exists": time_offsets_path.exists(),
            "manifest_examples": len(shard_rows),
            "tx_shape": None,
            "sbp_shape": None,
            "time_offsets_len": None,
            "tx_width_match": None,
            "sbp_width_match": None,
            "row_count_match": None,
            "manifest_length_mismatch_count": 0,
            "bad_example_index_count": 0,
        }
        totals["checked_shards"] += 1
        if not tx_path.exists():
            totals["missing_tx_files"] += 1
        if not sbp_path.exists():
            totals["missing_sbp_files"] += 1
        if not time_offsets_path.exists():
            totals["missing_time_offsets_files"] += 1
        if not (tx_path.exists() and sbp_path.exists() and time_offsets_path.exists()):
            checks.append(shard_report)
            continue

        tx = np.load(tx_path, mmap_mode="r", allow_pickle=False)
        sbp = np.load(sbp_path, mmap_mode="r", allow_pickle=False)
        time_offsets = np.load(time_offsets_path, mmap_mode="r", allow_pickle=False)

        shard_report["tx_shape"] = list(tx.shape)
        shard_report["sbp_shape"] = list(sbp.shape)
        shard_report["time_offsets_len"] = int(time_offsets.shape[0])

        tx_width_match = bool(tx.ndim == 2 and (expected_tx is None or int(tx.shape[1]) == int(expected_tx)))
        sbp_width_match = bool(sbp.ndim == 2 and (expected_sbp is None or int(sbp.shape[1]) == int(expected_sbp)))
        row_count_match = bool(
            tx.ndim == 2
            and sbp.ndim == 2
            and time_offsets.ndim == 1
            and time_offsets.size > 0
            and int(tx.shape[0]) == int(sbp.shape[0]) == int(time_offsets[-1])
        )
        shard_report["tx_width_match"] = tx_width_match
        shard_report["sbp_width_match"] = sbp_width_match
        shard_report["row_count_match"] = row_count_match
        if not tx_width_match:
            totals["tx_width_mismatches"] += 1
        if not sbp_width_match:
            totals["sbp_width_mismatches"] += 1
        if not row_count_match:
            totals["row_count_mismatches"] += 1

        bad_example_index_count = 0
        manifest_length_mismatch_count = 0
        for row in shard_rows:
            example_index = int(row.get("example_index", -1))
            if example_index < 0 or example_index + 1 >= int(time_offsets.shape[0]):
                bad_example_index_count += 1
                continue
            start = int(time_offsets[example_index])
            stop = int(time_offsets[example_index + 1])
            implied = max(0, stop - start)
            if implied != int(row.get("n_time_bins", -1)):
                manifest_length_mismatch_count += 1
        shard_report["bad_example_index_count"] = bad_example_index_count
        shard_report["manifest_length_mismatch_count"] = manifest_length_mismatch_count
        totals["bad_example_index"] += bad_example_index_count
        totals["manifest_length_mismatches"] += manifest_length_mismatch_count
        checks.append(shard_report)

    dense_check = None
    if deep_array_check and dataset_name == "brain2text24":
        tx_any_nonzero = None
        sbp_any_nonzero = None
        sbp_any_nonfinite = None
        shards_with_zero_tx = 0
        shards_with_zero_sbp = 0
        shards_with_nonfinite_sbp = 0
        for shard_check in checks:
            if not (shard_check.get("tx_exists") and shard_check.get("sbp_exists") and shard_check.get("time_offsets_exists")):
                continue
            shard_dir = Path(str(shard_check["path"]))
            tx = np.load(shard_dir / "tx.npy", mmap_mode="r", allow_pickle=False)
            sbp = np.load(shard_dir / "sbp.npy", mmap_mode="r", allow_pickle=False)
            tx_all_zero, _ = _scan_dense_array(tx)
            sbp_all_zero, sbp_nonfinite = _scan_dense_array(sbp)
            tx_any_nonzero = (~tx_all_zero) if tx_any_nonzero is None else (tx_any_nonzero | (~tx_all_zero))
            sbp_any_nonzero = (~sbp_all_zero) if sbp_any_nonzero is None else (sbp_any_nonzero | (~sbp_all_zero))
            sbp_any_nonfinite = sbp_nonfinite if sbp_any_nonfinite is None else (sbp_any_nonfinite | sbp_nonfinite)
            shards_with_zero_tx += int(tx_all_zero.any())
            shards_with_zero_sbp += int(sbp_all_zero.any())
            shards_with_nonfinite_sbp += int(sbp_nonfinite.any())
        dense_check = {
            "dataset_tx_all_zero_channels": int((~tx_any_nonzero).sum()) if tx_any_nonzero is not None else None,
            "dataset_sbp_all_zero_channels": int((~sbp_any_nonzero).sum()) if sbp_any_nonzero is not None else None,
            "dataset_sbp_nonfinite_channels": int(sbp_any_nonfinite.sum()) if sbp_any_nonfinite is not None else None,
            "shards_with_any_all_zero_tx_channel": int(shards_with_zero_tx),
            "shards_with_any_all_zero_sbp_channel": int(shards_with_zero_sbp),
            "shards_with_any_nonfinite_sbp_channel": int(shards_with_nonfinite_sbp),
        }

    return {
        "sampled_shard_ids": shard_ids,
        "checked_shards": checks,
        "totals": totals,
        "deep_dense_check": dense_check,
    }


def _build_context_for_mode(
    cache_root: Path,
    dataset_name: str,
    *,
    segment_bins: int,
    feature_mode: str,
) -> dict[str, Any]:
    available_datasets = sorted(path.name for path in cache_root.iterdir() if path.is_dir()) if cache_root.exists() else []
    excluded = tuple(name for name in available_datasets if name != dataset_name)
    config = CacheAccessConfig(
        mode="drive_direct",
        excluded_datasets=excluded,
        seed=7,
        segment_bins=int(segment_bins),
        use_normalization=False,
        examples_per_shard=8,
        feature_mode=str(feature_mode),
        boundary_key_mode="session",
        gaussian_smoothing_sigma_bins=0.0,
    )
    try:
        context = prepare_cache_context(cache_candidates=[cache_root], config=config)
    except Exception as exc:
        return {
            "feature_mode": feature_mode,
            "prepare_cache_context_ok": False,
            "prepare_cache_context_error": f"{type(exc).__name__}: {exc}",
        }

    summary = dict(context.session_split_summary.get(dataset_name, {}))
    train_rows = list(context.split_rows_by_dataset.get("train", {}).get(dataset_name, []))
    val_rows = list(context.split_rows_by_dataset.get("val", {}).get(dataset_name, []))
    summary["train_session_ids"] = sorted({str(row.session_id) for row in train_rows})
    summary["val_session_ids"] = sorted({str(row.session_id) for row in val_rows})
    train_eligible = sum(int(row.n_time_bins >= int(segment_bins)) for row in train_rows)
    val_eligible = sum(int(row.n_time_bins >= int(segment_bins)) for row in val_rows)

    sampling_error = None
    train_sampling_ok = False
    val_sampling_ok = False
    try:
        _ = get_sampling_plan(context, "train", int(segment_bins), 1.0)
        train_sampling_ok = True
    except Exception as exc:
        sampling_error = f"train: {type(exc).__name__}: {exc}"
    try:
        _ = get_sampling_plan(context, "val", int(segment_bins), 1.0)
        val_sampling_ok = True
    except Exception as exc:
        sampling_error = (sampling_error + " | " if sampling_error else "") + f"val: {type(exc).__name__}: {exc}"

    return {
        "feature_mode": feature_mode,
        "prepare_cache_context_ok": True,
        "cache_root": str(context.cache_root),
        "source_cache_signature": str(context.source_cache_signature),
        "session_split_summary": summary,
        "train_rows": int(len(train_rows)),
        "val_rows": int(len(val_rows)),
        "train_rows_supporting_segment_bins": int(train_eligible),
        "val_rows_supporting_segment_bins": int(val_eligible),
        "train_sampling_ok": bool(train_sampling_ok),
        "val_sampling_ok": bool(val_sampling_ok),
        "sampling_error": sampling_error,
    }


def audit_cache_root(spec: RootAuditInput) -> dict[str, Any]:
    root = Path(spec.cache_root)
    root_report: dict[str, Any] = {
        "cache_root": str(root),
        "exists": root.exists(),
        "source_signature": None,
        "structural_signature": None,
        "datasets": {},
    }
    if root.exists():
        try:
            root_report["source_signature"] = _compute_cache_source_signature(root)
            root_report["structural_signature"] = _compute_structural_root_signature(root)
        except Exception as exc:
            root_report["source_signature_error"] = f"{type(exc).__name__}: {exc}"

    for dataset_name in spec.dataset_names:
        dataset_root = root / dataset_name
        manifest_path = dataset_root / "manifest.jsonl"
        metadata_path = dataset_root / "metadata.json"
        shard_root = dataset_root / "shards"
        dataset_report: dict[str, Any] = {
            "dataset_root": str(dataset_root),
            "exists": dataset_root.exists(),
            "manifest_exists": manifest_path.exists(),
            "metadata_exists": metadata_path.exists(),
            "shards_dir_exists": shard_root.exists(),
            "legacy_flat_shard_layout": dataset_root.exists() and not shard_root.exists(),
            "manifest_sha256": _sha256_file(manifest_path),
            "metadata_sha256": _sha256_file(metadata_path),
            "smoothing_provenance": None,
            "metadata_feature_layout": None,
            "metadata_modalities": None,
            "metadata_declared_shards": None,
            "manifest_summary": None,
            "array_check": None,
            "feature_mode_audits": {},
            "findings": [],
        }
        manifest_rows: list[dict[str, Any]] = []
        metadata: dict[str, Any] | None = None

        if metadata_path.exists():
            try:
                metadata = load_dataset_metadata(root, dataset_name)
                dataset_report["smoothing_provenance"] = load_cache_smoothing_provenance(root, dataset=dataset_name)
                dataset_report["metadata_feature_layout"] = metadata.get("feature_layout")
                dataset_report["metadata_modalities"] = metadata.get("modalities")
                dataset_report["metadata_declared_shards"] = len(metadata.get("shards", [])) if isinstance(metadata.get("shards"), list) else None
            except Exception as exc:
                dataset_report["findings"].append(f"metadata_read_failed: {type(exc).__name__}: {exc}")
        else:
            dataset_report["findings"].append("missing_metadata_json")

        if manifest_path.exists():
            try:
                manifest_rows = _iter_manifest_rows(manifest_path)
            except Exception as exc:
                dataset_report["findings"].append(f"manifest_read_failed: {type(exc).__name__}: {exc}")
        else:
            dataset_report["findings"].append("missing_manifest_jsonl")

        if manifest_rows:
            lengths = [int(row.get("n_time_bins", 0)) for row in manifest_rows]
            sessions = {str(row.get("session_id")) for row in manifest_rows}
            tx_features = sorted({int(row.get("n_tx_features", -1)) for row in manifest_rows})
            sbp_features = sorted({int(row.get("n_sbp_features", -1)) for row in manifest_rows})
            has_tx_values = sorted({bool(row.get("has_tx", False)) for row in manifest_rows})
            has_sbp_values = sorted({bool(row.get("has_sbp", False)) for row in manifest_rows})
            supporting = sum(int(length >= int(spec.segment_bins)) for length in lengths)
            dataset_report["manifest_summary"] = {
                "row_count": int(len(manifest_rows)),
                "session_count": int(len(sessions)),
                "unique_n_tx_features": tx_features,
                "unique_n_sbp_features": sbp_features,
                "unique_has_tx": has_tx_values,
                "unique_has_sbp": has_sbp_values,
                "n_time_bins": _quantiles(lengths),
                "rows_supporting_segment_bins": int(supporting),
                "rows_failing_segment_bins": int(len(manifest_rows) - supporting),
            }
            if supporting == 0:
                dataset_report["findings"].append(f"no_manifest_rows_support_segment_bins_{int(spec.segment_bins)}")

        if manifest_rows:
            dataset_report["array_check"] = _summarize_array_consistency(
                dataset_root,
                manifest_rows,
                metadata,
                sample_shards=int(spec.sample_shards),
                deep_array_check=bool(spec.deep_array_check),
                dataset_name=dataset_name,
            )
            totals = dataset_report["array_check"]["totals"]
            if totals["manifest_length_mismatches"] > 0:
                dataset_report["findings"].append("manifest_vs_time_offsets_length_mismatch")
            if totals["missing_tx_files"] > 0 or totals["missing_sbp_files"] > 0 or totals["missing_time_offsets_files"] > 0:
                dataset_report["findings"].append("missing_array_files")
            if totals["tx_width_mismatches"] > 0 or totals["sbp_width_mismatches"] > 0:
                dataset_report["findings"].append("feature_width_mismatch")

        if root.exists() and dataset_root.exists() and manifest_rows:
            for feature_mode in spec.feature_modes:
                dataset_report["feature_mode_audits"][feature_mode] = _build_context_for_mode(
                    root,
                    dataset_name,
                    segment_bins=int(spec.segment_bins),
                    feature_mode=feature_mode,
                )
        else:
            for feature_mode in spec.feature_modes:
                dataset_report["feature_mode_audits"][feature_mode] = {
                    "feature_mode": feature_mode,
                    "prepare_cache_context_ok": False,
                    "prepare_cache_context_error": "dataset_missing_or_unreadable",
                }

        root_report["datasets"][dataset_name] = dataset_report
    return root_report


def _classify_stats_feature_modes(unique_dims: Iterable[int]) -> dict[str, bool]:
    dims = sorted({int(dim) for dim in unique_dims if int(dim) > 0})
    smallest_dim = min(dims) if dims else 0
    return {
        "tx_only": bool(smallest_dim >= 256),
        "tx_sbp": bool(smallest_dim >= 512),
    }


def audit_stats_artifact(spec: StatsAuditInput, dataset_names: Sequence[str]) -> dict[str, Any]:
    path = Path(spec.stats_path)
    report: dict[str, Any] = {
        "stats_path": str(path),
        "exists": path.exists(),
        "session_count": None,
        "metadata": None,
        "unique_mean_dims": [],
        "unique_std_dims": [],
        "compatible_feature_modes": None,
        "dataset_session_counts": {},
        "dataset_session_keys": {},
        "findings": [],
    }
    if not path.exists():
        report["findings"].append("missing_stats_file")
        return report

    payload = torch.load(path, map_location="cpu")
    raw_stats = payload.get("session_feature_stats")
    if not isinstance(raw_stats, dict):
        report["findings"].append("missing_session_feature_stats_dict")
        return report

    metadata = payload.get("metadata")
    report["metadata"] = dict(metadata) if isinstance(metadata, dict) else {}
    mean_dims: list[int] = []
    std_dims: list[int] = []
    dataset_session_counts = defaultdict(int)
    dataset_session_keys = defaultdict(set)
    for key, value in raw_stats.items():
        if not isinstance(value, (tuple, list)) or len(value) != 2:
            report["findings"].append(f"bad_stats_entry:{key}")
            continue
        mean, std = value
        mean_dims.append(int(torch.as_tensor(mean).numel()))
        std_dims.append(int(torch.as_tensor(std).numel()))
        key_str = str(key)
        for dataset_name in dataset_names:
            if key_str.startswith(f"{dataset_name}:"):
                dataset_session_counts[dataset_name] += 1
                session_suffix = key_str[len(dataset_name) + 1 :]
                if session_suffix:
                    dataset_session_keys[dataset_name].add(session_suffix)

    report["session_count"] = int(len(raw_stats))
    report["unique_mean_dims"] = sorted(set(mean_dims))
    report["unique_std_dims"] = sorted(set(std_dims))
    report["compatible_feature_modes"] = _classify_stats_feature_modes(set(mean_dims) | set(std_dims))
    report["dataset_session_counts"] = {name: int(dataset_session_counts.get(name, 0)) for name in dataset_names}
    report["dataset_session_keys"] = {
        name: sorted(dataset_session_keys.get(name, set()))
        for name in dataset_names
    }
    if report["compatible_feature_modes"]["tx_only"] and not report["compatible_feature_modes"]["tx_sbp"]:
        report["findings"].append("tx_only_only_not_tx_sbp")
    if not report["compatible_feature_modes"]["tx_only"]:
        report["findings"].append("insufficient_dims_for_tx_only")
    return report


def compare_root_audits(root_audits: Sequence[dict[str, Any]], dataset_names: Sequence[str]) -> list[dict[str, Any]]:
    if not root_audits:
        return []
    baseline = root_audits[0]
    comparisons: list[dict[str, Any]] = []
    for other in root_audits[1:]:
        report: dict[str, Any] = {
            "baseline_cache_root": baseline.get("cache_root"),
            "candidate_cache_root": other.get("cache_root"),
            "baseline_source_signature": baseline.get("source_signature"),
            "candidate_source_signature": other.get("source_signature"),
            "baseline_structural_signature": baseline.get("structural_signature"),
            "candidate_structural_signature": other.get("structural_signature"),
            "source_signature_match": baseline.get("source_signature") == other.get("source_signature"),
            "structural_signature_match": baseline.get("structural_signature") == other.get("structural_signature"),
            "datasets": {},
            "findings": [],
        }
        for dataset_name in dataset_names:
            base_ds = baseline.get("datasets", {}).get(dataset_name, {})
            cand_ds = other.get("datasets", {}).get(dataset_name, {})
            base_manifest = base_ds.get("manifest_summary") or {}
            cand_manifest = cand_ds.get("manifest_summary") or {}
            ds_findings: list[str] = []
            manifest_match = base_ds.get("manifest_sha256") == cand_ds.get("manifest_sha256")
            metadata_match = base_ds.get("metadata_sha256") == cand_ds.get("metadata_sha256")
            if manifest_match and not metadata_match:
                ds_findings.append("same_manifest_different_metadata")
            if metadata_match and not manifest_match:
                ds_findings.append("same_metadata_different_manifest")
            if base_ds.get("metadata_exists") and not cand_ds.get("metadata_exists"):
                ds_findings.append("baseline_has_metadata_candidate_missing_metadata")
            if cand_manifest.get("row_count") == base_manifest.get("row_count") and cand_manifest.get("n_time_bins") != base_manifest.get("n_time_bins"):
                ds_findings.append("same_row_count_different_length_distribution")
            if cand_ds.get("smoothing_provenance") is None and "smoothed" in str(other.get("cache_root", "")):
                ds_findings.append("smoothed_root_missing_copied_metadata")
            if cand_manifest.get("row_count") != base_manifest.get("row_count"):
                ds_findings.append("example_count_drift")
            if cand_manifest.get("session_count") != base_manifest.get("session_count"):
                ds_findings.append("session_count_drift")
            if (not cand_ds.get("metadata_exists")) or ((cand_ds.get("array_check") or {}).get("totals", {}).get("missing_tx_files", 0) > 0) or ((cand_ds.get("array_check") or {}).get("totals", {}).get("missing_sbp_files", 0) > 0) or ((cand_ds.get("array_check") or {}).get("totals", {}).get("missing_time_offsets_files", 0) > 0):
                ds_findings.append("candidate_root_appears_partial_or_incomplete")
            report["datasets"][dataset_name] = {
                "manifest_sha256_match": manifest_match,
                "metadata_sha256_match": metadata_match,
                "baseline_row_count": base_manifest.get("row_count"),
                "candidate_row_count": cand_manifest.get("row_count"),
                "baseline_session_count": base_manifest.get("session_count"),
                "candidate_session_count": cand_manifest.get("session_count"),
                "baseline_length_summary": base_manifest.get("n_time_bins"),
                "candidate_length_summary": cand_manifest.get("n_time_bins"),
                "baseline_declared_shards": base_ds.get("metadata_declared_shards"),
                "candidate_declared_shards": cand_ds.get("metadata_declared_shards"),
                "findings": ds_findings,
            }
            report["findings"].extend(f"{dataset_name}:{finding}" for finding in ds_findings)
        comparisons.append(report)
    return comparisons


def compare_stats_to_roots(
    stats_audits: Sequence[dict[str, Any]],
    root_audits: Sequence[dict[str, Any]],
    dataset_names: Sequence[str],
) -> list[dict[str, Any]]:
    comparisons: list[dict[str, Any]] = []
    for stats_report in stats_audits:
        for root_report in root_audits:
            dataset_checks: dict[str, Any] = {}
            findings: list[str] = []
            for dataset_name in dataset_names:
                ds = root_report.get("datasets", {}).get(dataset_name, {})
                provenance = ds.get("smoothing_provenance") or {}
                stats_meta = stats_report.get("metadata") or {}
                stats_sigma = stats_meta.get("gaussian_smoothing_sigma_bins")
                root_sigma = provenance.get("sigma_bins")
                sigma_match = None
                if stats_sigma is not None or root_sigma is not None:
                    try:
                        sigma_match = math.isclose(float(stats_sigma), float(root_sigma), rel_tol=0.0, abs_tol=1e-6)
                    except (TypeError, ValueError):
                        sigma_match = False
                session_count = stats_report.get("dataset_session_counts", {}).get(dataset_name)
                stats_session_keys = set(stats_report.get("dataset_session_keys", {}).get(dataset_name, []))
                manifest_sessions = (ds.get("manifest_summary") or {}).get("session_count")
                manifest_session_keys = set()
                mode_audits = ds.get("feature_mode_audits") or {}
                mode_report = mode_audits.get("tx_only") or next(iter(mode_audits.values()), {})
                summary = mode_report.get("session_split_summary") or {}
                for split_name in ("train_session_ids", "val_session_ids"):
                    values = summary.get(split_name)
                    if isinstance(values, list):
                        manifest_session_keys.update(str(value) for value in values)
                missing_in_stats = sorted(manifest_session_keys - stats_session_keys)
                extra_in_stats = sorted(stats_session_keys - manifest_session_keys)
                dataset_checks[dataset_name] = {
                    "stats_session_count": session_count,
                    "manifest_session_count": manifest_sessions,
                    "stats_sigma": stats_sigma,
                    "root_sigma": root_sigma,
                    "sigma_match": sigma_match,
                    "manifest_session_keys": sorted(manifest_session_keys),
                    "stats_session_keys": sorted(stats_session_keys),
                    "missing_session_keys_in_stats": missing_in_stats,
                    "extra_session_keys_in_stats": extra_in_stats,
                }
                if sigma_match is False:
                    findings.append(f"{dataset_name}:sigma_mismatch")
                if session_count in (None, 0):
                    findings.append(f"{dataset_name}:stats_missing_sessions")
                if missing_in_stats or extra_in_stats:
                    findings.append(f"{dataset_name}:session_key_mismatch")
            comparisons.append(
                {
                    "stats_path": stats_report.get("stats_path"),
                    "cache_root": root_report.get("cache_root"),
                    "datasets": dataset_checks,
                    "findings": findings,
                }
            )
    return comparisons


def run_audit(
    *,
    cache_roots: Sequence[str | Path],
    stats_paths: Sequence[str | Path] = (),
    dataset: str = DEFAULT_DATASET,
    compare_datasets: Sequence[str] = (),
    segment_bins: int = DEFAULT_SEGMENT_BINS,
    feature_modes: Sequence[str] = DEFAULT_FEATURE_MODES,
    sample_shards: int = 3,
    deep_array_check: bool = False,
) -> dict[str, Any]:
    dataset_names = tuple(dict.fromkeys([str(dataset), *[str(name) for name in compare_datasets]]))
    feature_modes = tuple(dict.fromkeys(str(mode) for mode in feature_modes))
    root_specs = [
        RootAuditInput(
            cache_root=Path(root),
            dataset_names=dataset_names,
            segment_bins=int(segment_bins),
            feature_modes=feature_modes,
            sample_shards=int(sample_shards),
            deep_array_check=bool(deep_array_check),
        )
        for root in cache_roots
    ]
    stats_specs = [StatsAuditInput(stats_path=Path(path)) for path in stats_paths]

    root_audits = [audit_cache_root(spec) for spec in root_specs]
    stats_audits = [audit_stats_artifact(spec, dataset_names) for spec in stats_specs]

    report = {
        "generated_at_utc": _timestamp_utc(),
        "dataset_names": list(dataset_names),
        "segment_bins": int(segment_bins),
        "feature_modes": list(feature_modes),
        "sample_shards": int(sample_shards),
        "deep_array_check": bool(deep_array_check),
        "root_audits": root_audits,
        "root_comparisons": compare_root_audits(root_audits, dataset_names),
        "stats_audits": stats_audits,
        "stats_root_comparisons": compare_stats_to_roots(stats_audits, root_audits, dataset_names),
    }
    return report


def print_report(report: dict[str, Any]) -> None:
    print("cache audit")
    print(f"generated_at_utc: {report['generated_at_utc']}")
    print(f"datasets: {report['dataset_names']}")
    print(f"segment_bins: {report['segment_bins']}")
    print()
    print("cache roots")
    for root_report in report.get("root_audits", []):
        print(
            f"- {root_report.get('cache_root')} | exists={root_report.get('exists')} "
            f"| source_signature={root_report.get('source_signature')} "
            f"| structural_signature={root_report.get('structural_signature')}"
        )
        for dataset_name, ds in root_report.get("datasets", {}).items():
            manifest = ds.get("manifest_summary") or {}
            print(
                f"  {dataset_name}: rows={manifest.get('row_count')} sessions={manifest.get('session_count')} "
                f"support_segment_bins={manifest.get('rows_supporting_segment_bins')} metadata={ds.get('metadata_exists')}"
            )
            for mode, mode_report in ds.get("feature_mode_audits", {}).items():
                if mode_report.get("prepare_cache_context_ok"):
                    print(
                        f"    [{mode}] train_rows={mode_report.get('train_rows')} "
                        f"train_support={mode_report.get('train_rows_supporting_segment_bins')} "
                        f"train_sampling_ok={mode_report.get('train_sampling_ok')}"
                    )
                else:
                    print(f"    [{mode}] prepare_failed={mode_report.get('prepare_cache_context_error')}")
            if ds.get("findings"):
                print(f"    findings: {', '.join(ds['findings'])}")
    if report.get("stats_audits"):
        print()
        print("stats artifacts")
        for stats_report in report["stats_audits"]:
            compat = stats_report.get("compatible_feature_modes") or {}
            print(
                f"- {stats_report.get('stats_path')} | exists={stats_report.get('exists')} "
                f"sessions={stats_report.get('session_count')} tx_only={compat.get('tx_only')} tx_sbp={compat.get('tx_sbp')}"
            )
            if stats_report.get("findings"):
                print(f"  findings: {', '.join(stats_report['findings'])}")
    if report.get("root_comparisons"):
        print()
        print("root comparisons")
        for comparison in report["root_comparisons"]:
            print(
                f"- baseline={comparison.get('baseline_cache_root')} vs candidate={comparison.get('candidate_cache_root')} "
                f"source_signature_match={comparison.get('source_signature_match')} "
                f"structural_signature_match={comparison.get('structural_signature_match')}"
            )
            for dataset_name, ds in comparison.get("datasets", {}).items():
                print(
                    f"  {dataset_name}: manifest_match={ds.get('manifest_sha256_match')} "
                    f"metadata_match={ds.get('metadata_sha256_match')} findings={ds.get('findings')}"
                )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit Utah SSL cache roots and session-stats artifacts.")
    parser.add_argument("--cache-root", action="append", required=True, help="Cache root to audit. Pass multiple times.")
    parser.add_argument("--stats-path", action="append", default=[], help="Optional session-stats artifact to audit. Pass multiple times.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="Primary dataset to audit.")
    parser.add_argument("--compare-dataset", action="append", default=[], help="Additional dataset(s) to include.")
    parser.add_argument("--segment-bins", type=int, default=DEFAULT_SEGMENT_BINS, help="Segment length to test against sampling eligibility.")
    parser.add_argument("--feature-mode", action="append", choices=sorted(DEFAULT_FEATURE_MODES), default=[], help="Feature mode(s) to audit. Defaults to both tx_only and tx_sbp.")
    parser.add_argument("--sample-shards", type=int, default=3, help="Number of shards to inspect per dataset when not doing a deep array check.")
    parser.add_argument("--deep-array-check", action="store_true", help="Scan every shard for array-level consistency and dense-channel checks.")
    parser.add_argument("--output-json", type=Path, default=None, help="Optional path to save the JSON report.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run_audit(
        cache_roots=args.cache_root,
        stats_paths=args.stats_path,
        dataset=args.dataset,
        compare_datasets=args.compare_dataset,
        segment_bins=args.segment_bins,
        feature_modes=args.feature_mode or DEFAULT_FEATURE_MODES,
        sample_shards=args.sample_shards,
        deep_array_check=args.deep_array_check,
    )
    print_report(report)
    output_path = args.output_json
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2, default=_json_default))
        print()
        print(f"wrote_json: {output_path}")


if __name__ == "__main__":
    main()

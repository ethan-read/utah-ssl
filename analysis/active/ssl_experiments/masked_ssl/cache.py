"""Cache access, shard loading, and segment sampling for contrastive SSL runs."""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import shlex
import shutil
import time
from collections import Counter, OrderedDict, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from ssl_core.experiment_contract import (
    DatasetPlan,
    SignalSpec,
)

try:
    import psutil
except ImportError:  # pragma: no cover - optional dependency
    psutil = None


RUNTIME_SMOOTHING_MIGRATION_MESSAGE = (
    "Runtime Gaussian smoothing has been removed from masked_ssl. "
    "Build or select a pre-smoothed cache root instead and keep "
    "gaussian_smoothing_sigma_bins=0.0 during training."
)

# Fixed stride for session-stat computation to match the normalized cache artifacts.
SESSION_STATS_BIN_STRIDE = 2
AREA6V_FEATURE_DIM = 128


@dataclass
class CacheAccessConfig:
    dataset_plan: DatasetPlan | Mapping[str, Sequence[str]]
    signal_spec: SignalSpec | Mapping[str, Any]
    mode: str = "copy_to_local"
    local_cache_base: str = "/content/utah_ssl_cache"
    force_recopy_local_cache: bool = False
    seed: int = 7
    segment_bins: int = 64
    use_normalization: bool = True
    examples_per_shard: int = 8
    boundary_key_mode: str = "session"
    gaussian_smoothing_sigma_bins: float = 0.0
    shard_cache_ram_gb: float | None = None
    precomputed_session_stats_path: str | Path | None = None
    dataset_cache_roots: dict[str, str | Path] | None = None

    def __post_init__(self) -> None:
        self.signal_spec = SignalSpec.from_value(self.signal_spec)
        self.dataset_plan = DatasetPlan.from_value(self.dataset_plan)
        if self.dataset_cache_roots is not None:
            normalized_cache_roots: dict[str, Path] = {}
            for dataset, cache_root in self.dataset_cache_roots.items():
                dataset_name = str(dataset).strip()
                if not dataset_name:
                    raise ValueError("dataset_cache_roots contains an empty dataset name")
                normalized_cache_roots[dataset_name] = Path(cache_root)
            self.dataset_cache_roots = normalized_cache_roots or None
        if self.mode not in {"copy_to_local", "drive_direct"}:
            raise ValueError("mode must be either 'copy_to_local' or 'drive_direct'")
        if self.segment_bins <= 0:
            raise ValueError("segment_bins must be positive")
        if self.examples_per_shard <= 0:
            raise ValueError("examples_per_shard must be positive")
        if self.boundary_key_mode not in {"session", "subject_if_available"}:
            raise ValueError(
                "boundary_key_mode must be one of {'session', 'subject_if_available'}"
            )
        if float(self.gaussian_smoothing_sigma_bins) < 0.0:
            raise ValueError("gaussian_smoothing_sigma_bins must be non-negative")
        if not isinstance(self.use_normalization, bool):
            raise ValueError("use_normalization must be a boolean")

    @property
    def full_dim(self) -> int:
        assert isinstance(self.signal_spec, SignalSpec)
        return int(self.signal_spec.full_dim)

    @property
    def feature_mode(self) -> str:
        assert isinstance(self.signal_spec, SignalSpec)
        return self.signal_spec.mode

    @property
    def tx_dim(self) -> int:
        assert isinstance(self.signal_spec, SignalSpec)
        return int(self.signal_spec.tx_dim)

    @property
    def sbp_dim(self) -> int:
        assert isinstance(self.signal_spec, SignalSpec)
        return int(self.signal_spec.sbp_dim)


@dataclass(frozen=True)
class ExampleRow:
    dataset: str
    session_id: str
    subject_id: str | None
    shard_relpath: str
    example_index: int
    n_time_bins: int
    has_tx: bool
    has_sbp: bool
    n_tx_features: int
    n_sbp_features: int


@dataclass(frozen=True)
class SamplingPlan:
    split_name: str
    segment_bins: int
    dataset_weight_alpha: float
    dataset_names: tuple[str, ...]
    dataset_probs: np.ndarray
    shard_rows_by_dataset: dict[str, dict[str, list[ExampleRow]]]
    shard_keys_by_dataset: dict[str, list[str]]
    shard_probs_by_dataset: dict[str, np.ndarray]
    row_probs_within_shard_by_dataset: dict[str, dict[str, np.ndarray]]


@dataclass
class CacheContext:
    config: CacheAccessConfig
    drive_cache_root: Path
    cache_root: Path
    cache_copy_used: bool
    source_cache_signature: str
    available_datasets: list[str]
    pretrain_datasets: list[str]
    rows_by_dataset: dict[str, list[ExampleRow]]
    split_rows_by_dataset: dict[str, dict[str, list[ExampleRow]]]
    session_split_summary: dict[str, dict[str, Any]]
    shard_store: "ShardStore"
    has_val_datasets: bool
    drive_dataset_cache_roots: dict[str, Path] = field(default_factory=dict)
    dataset_cache_roots: dict[str, Path] = field(default_factory=dict)
    session_feature_stats: dict[str, tuple[torch.Tensor, torch.Tensor]] = field(default_factory=dict)
    sampling_plan_cache: dict[tuple[str, int, float], SamplingPlan] = field(default_factory=dict)

    @property
    def tx_dim(self) -> int:
        return int(self.config.tx_dim)

    @property
    def sbp_dim(self) -> int:
        return int(self.config.sbp_dim)

    @property
    def full_dim(self) -> int:
        return int(self.config.full_dim)

    @property
    def feature_mode(self) -> str:
        return str(self.config.feature_mode)

    @property
    def signal_spec(self) -> SignalSpec:
        assert isinstance(self.config.signal_spec, SignalSpec)
        return self.config.signal_spec

    @property
    def boundary_key_mode(self) -> str:
        return str(self.config.boundary_key_mode)

    @property
    def use_normalization(self) -> bool:
        return bool(self.config.use_normalization)

    @property
    def gaussian_smoothing_sigma_bins(self) -> float:
        return float(self.config.gaussian_smoothing_sigma_bins)


def runtime_smoothing_requested(config: CacheAccessConfig) -> bool:
    return float(config.gaussian_smoothing_sigma_bins) > 0.0


def ensure_runtime_smoothing_disabled(
    config: CacheAccessConfig,
    *,
    context: str,
) -> None:
    if runtime_smoothing_requested(config):
        raise RuntimeError(
            f"{context}: {RUNTIME_SMOOTHING_MIGRATION_MESSAGE} "
            f"Requested gaussian_smoothing_sigma_bins={float(config.gaussian_smoothing_sigma_bins):.6g}."
        )


def load_dataset_metadata(
    cache_root: str | Path,
    dataset: str,
) -> dict[str, Any]:
    metadata_path = Path(cache_root) / str(dataset) / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing dataset metadata: {metadata_path}")
    return json.loads(metadata_path.read_text())


def load_cache_smoothing_provenance(
    cache_root: str | Path,
    *,
    dataset: str | None = None,
) -> dict[str, Any] | None:
    root = Path(cache_root)
    if dataset is not None:
        metadata = load_dataset_metadata(root, dataset)
        provenance = metadata.get("smoothing_provenance")
        return dict(provenance) if isinstance(provenance, dict) else None

    for dataset_root in sorted(path for path in root.iterdir() if path.is_dir()):
        metadata_path = dataset_root / "metadata.json"
        if not metadata_path.exists():
            continue
        metadata = json.loads(metadata_path.read_text())
        provenance = metadata.get("smoothing_provenance")
        if isinstance(provenance, dict):
            return dict(provenance)
    return None


def _cache_variant_name(cache_root: str | Path) -> str:
    name = Path(cache_root).name
    if "smoothed_sigma2p0" in name:
        return "smoothed_sigma2p0"
    if name == "cache_v1":
        return "raw"
    return name.replace("cache_v1_", "").replace("/", "_")


def _canonical_stats_root_for_cache(cache_root: str | Path) -> Path:
    cache_root = Path(cache_root)
    if cache_root.parent.name == "data":
        return cache_root.parent / "stats"
    local_stats_root = cache_root / "stats"
    if local_stats_root.exists():
        return local_stats_root
    return cache_root.parent / "stats"


def _canonical_session_stats_dir(
    *,
    cache_root: str | Path,
    feature_mode: str,
    boundary_key_mode: str,
) -> Path:
    return (
        _canonical_stats_root_for_cache(cache_root)
        / "session_feature_stats"
        / _cache_variant_name(cache_root)
        / str(feature_mode)
        / str(boundary_key_mode)
    )


def _session_stats_plan_stem(dataset_plan: DatasetPlan) -> str:
    dataset_names = "_".join(
        name.replace("/", "_") for name in dataset_plan.dataset_names
    )
    plan_json = json.dumps(dataset_plan.to_dict(), sort_keys=True, separators=(",", ":"))
    plan_hash = hashlib.sha256(plan_json.encode("utf-8")).hexdigest()[:10]
    return f"ssl_pretrain_{dataset_names}_plan_{plan_hash}_v2"


def resolve_precomputed_session_stats_path(
    *,
    cache_root: str | Path,
    signal_spec: SignalSpec | Mapping[str, Any],
    dataset_plan: DatasetPlan | Mapping[str, Sequence[str]],
    boundary_key_mode: str,
) -> Path:
    resolved_signal_spec = SignalSpec.from_value(signal_spec)
    resolved_dataset_plan = DatasetPlan.from_value(dataset_plan)
    stats_dir = _canonical_session_stats_dir(
        cache_root=cache_root,
        feature_mode=resolved_signal_spec.mode,
        boundary_key_mode=boundary_key_mode,
    )
    return stats_dir / f"{_session_stats_plan_stem(resolved_dataset_plan)}.pt"


def _load_artifact_payload_and_sidecar(
    *,
    path: str | Path,
    canonical_path: str | Path,
    recompute_cmd: str,
    artifact_name: str,
    expected_kind: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any], Path, Path]:
    resolved_path = Path(path)
    expected_path = Path(canonical_path)
    if not resolved_path.exists():
        raise FileNotFoundError(
            f"Precomputed {artifact_name} file does not exist.\n"
            f"expected_path: {expected_path}\n"
            f"requested_path: {resolved_path}\n"
            f"recompute_command: {recompute_cmd}"
        )
    metadata_path = resolved_path.with_suffix(".json")
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Precomputed {artifact_name} sidecar is missing.\n"
            f"expected_path: {expected_path}\n"
            f"requested_path: {resolved_path}\n"
            f"missing_sidecar: {metadata_path}\n"
            f"recompute_command: {recompute_cmd}"
        )

    payload = torch.load(resolved_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"Precomputed {artifact_name} payload must be a dict: {resolved_path}")

    sidecar_metadata = json.loads(metadata_path.read_text())
    if not isinstance(sidecar_metadata, dict):
        raise ValueError(
            f"Precomputed {artifact_name} sidecar must be a JSON object.\n"
            f"requested_path: {resolved_path}\n"
            f"metadata_path: {metadata_path}\n"
            f"recompute_command: {recompute_cmd}"
        )

    metadata = dict(payload.get("metadata", {}))
    if metadata != sidecar_metadata:
        raise ValueError(
            f"Precomputed {artifact_name} payload metadata does not match the JSON sidecar.\n"
            f"expected_path: {expected_path}\n"
            f"requested_path: {resolved_path}\n"
            f"metadata_path: {metadata_path}\n"
            f"recompute_command: {recompute_cmd}"
        )
    if expected_kind is not None and metadata.get("kind") != str(expected_kind):
        raise ValueError(
            f"Precomputed {artifact_name} artifact has the wrong kind.\n"
            f"expected_path: {expected_path}\n"
            f"requested_path: {resolved_path}\n"
            f"reason: kind={metadata.get('kind')!r} expected {str(expected_kind)!r}\n"
            f"recompute_command: {recompute_cmd}"
        )
    return payload, metadata, resolved_path, metadata_path


def _validate_common_artifact_metadata(
    *,
    metadata: dict[str, Any],
    expected_metadata: dict[str, Any],
) -> list[str]:
    return [
        f"{key}={metadata.get(key)!r} expected {value!r}"
        for key, value in expected_metadata.items()
        if metadata.get(key) != value
    ]


def build_recompute_session_feature_stats_command(
    *,
    cache_root: str | Path,
    output_path: str | Path,
    signal_spec: SignalSpec | Mapping[str, Any],
    dataset_plan: DatasetPlan | Mapping[str, Sequence[str]],
    boundary_key_mode: str,
    dataset_cache_roots: Mapping[str, str | Path] | None = None,
) -> str:
    resolved_signal_spec = SignalSpec.from_value(signal_spec)
    resolved_dataset_plan = DatasetPlan.from_value(dataset_plan)
    cmd = [
        "python",
        "analysis/active/ssl_experiments/ssl_core/scripts/recompute_session_feature_stats.py",
        "--cache-root",
        str(Path(cache_root)),
        "--output-path",
        str(Path(output_path)),
        "--feature-mode",
        str(resolved_signal_spec.mode),
        "--boundary-key-mode",
        str(boundary_key_mode),
        "--tx-dim",
        str(int(resolved_signal_spec.tx_dim)),
        "--sbp-dim",
        str(int(resolved_signal_spec.sbp_dim)),
        "--column-start",
        str(int(resolved_signal_spec.column_start)),
        "--missing-channel-policy",
        str(resolved_signal_spec.missing_channel_policy),
    ]
    for selection in resolved_dataset_plan.datasets:
        cmd.extend(["--dataset", selection.name])
        for source_split in selection.source_splits:
            cmd.extend(
                ["--dataset-source-split", f"{selection.name}={source_split}"]
            )
    for dataset, dataset_cache_root in sorted((dataset_cache_roots or {}).items()):
        cmd.extend(["--dataset-cache-root", f"{str(dataset)}={str(Path(dataset_cache_root))}"])
    cmd.append("--overwrite")
    return shlex.join(cmd)


def _format_bytes(num_bytes: int) -> str:
    value = float(num_bytes)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if value < 1024.0 or unit == "TB":
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{value:.1f} TB"


def _path_signature(path: Path) -> dict[str, int] | None:
    if not path.exists():
        return None
    stat = path.stat()
    return {
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _list_dir_with_retries(path: Path, *, max_retries: int = 5) -> list[Path]:
    last_error: OSError | None = None
    for attempt in range(1, max_retries + 1):
        try:
            return sorted(path.iterdir(), key=lambda child: child.name)
        except OSError as exc:  # pragma: no cover - exercised in Colab when Drive stalls
            last_error = exc
            if attempt == max_retries:
                break
            print(f"directory scan retry {attempt}/{max_retries} failed for {path}: {exc}")
            time.sleep(min(10.0, float(attempt)))
    assert last_error is not None
    raise last_error


def _dataset_signature_payload(dataset_root: Path) -> dict[str, Any]:
    shard_root = dataset_root / "shards"
    shard_names: list[str] = []
    shard_scan_error: str | None = None
    if shard_root.exists():
        try:
            shard_names = [path.name for path in _list_dir_with_retries(shard_root) if path.is_dir()]
        except OSError as exc:  # pragma: no cover - exercised in Colab when Drive stalls
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


def _compute_cache_source_signature(src_root: Path) -> str:
    datasets = [
        _dataset_signature_payload(dataset_root)
        for dataset_root in (
            path
            for path in _list_dir_with_retries(src_root)
            if path.is_dir() and (path / "metadata.json").exists()
        )
    ]
    payload = {
        "root": str(src_root),
        "datasets": datasets,
        "repack_summary": _path_signature(src_root / "repack_summary.json"),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _resolve_drive_dataset_cache_roots(
    primary_root: Path,
    overrides: Mapping[str, str | Path] | None,
) -> dict[str, Path]:
    resolved = {
        path.name: primary_root
        for path in _list_dir_with_retries(primary_root)
        if path.is_dir() and (path / "metadata.json").exists()
    }
    for dataset, cache_root_value in sorted((overrides or {}).items()):
        dataset_name = str(dataset).strip()
        cache_root = Path(cache_root_value)
        dataset_root = cache_root / dataset_name
        if not (dataset_root / "metadata.json").exists():
            raise FileNotFoundError(
                f"Dataset override {dataset_name!r} is missing metadata: "
                f"{dataset_root / 'metadata.json'}"
            )
        if not (dataset_root / "manifest.jsonl").exists():
            raise FileNotFoundError(
                f"Dataset override {dataset_name!r} is missing manifest: "
                f"{dataset_root / 'manifest.jsonl'}"
            )
        resolved[dataset_name] = cache_root
    return resolved


def _compute_dataset_cache_source_signature(
    dataset_cache_roots: Mapping[str, str | Path],
) -> str:
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


def _pretrain_source_splits_for_dataset(
    config: CacheAccessConfig,
    dataset: str,
) -> tuple[str, ...] | None:
    assert isinstance(config.dataset_plan, DatasetPlan)
    source_splits = config.dataset_plan.source_splits_by_dataset.get(str(dataset))
    return tuple(source_splits) if source_splits else None


def _load_copy_status(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _copy_complete_for_current_source(
    *,
    status: dict[str, Any] | None,
    drive_cache_root: Path,
    local_cache_root: Path,
    source_signature: str,
) -> bool:
    return bool(
        status
        and status.get("complete") is True
        and status.get("source") == str(drive_cache_root)
        and status.get("source_signature") == source_signature
        and Path(status.get("dest", str(local_cache_root))).exists()
    )


def _write_copy_status(
    path: Path,
    *,
    drive_cache_root: Path,
    local_cache_root: Path,
    source_signature: str,
    file_count: int,
    total_bytes: int,
) -> None:
    payload = {
        "complete": True,
        "source": str(drive_cache_root),
        "source_signature": source_signature,
        "dest": str(local_cache_root),
        "file_count": int(file_count),
        "total_bytes": int(total_bytes),
        "written_at": time.time(),
    }
    path.write_text(json.dumps(payload, indent=2))


def copy_tree_with_progress(
    src_root: Path,
    dst_root: Path,
    *,
    print_every_files: int = 250,
    max_copy_retries: int = 5,
) -> tuple[int, int]:
    entries = sorted(src_root.iterdir(), key=lambda path: path.name)
    file_count = 0
    total_bytes = 0
    for path in entries:
        if path.is_file():
            file_count += 1
            total_bytes += path.stat().st_size
            continue
        for child in path.rglob("*"):
            if child.is_file():
                file_count += 1
                total_bytes += child.stat().st_size

    print(
        f"copy plan: {len(entries)} top-level entries, {file_count} files, {_format_bytes(total_bytes)} total"
    )

    copied_files = 0
    copied_bytes = 0
    last_report = time.time()
    start_time = last_report

    def report(force: bool = False, label: str | None = None) -> None:
        nonlocal last_report
        now = time.time()
        if not force and copied_files > 0 and copied_files % print_every_files != 0 and (now - last_report) < 15.0:
            return
        elapsed = max(now - start_time, 1e-6)
        rate = copied_bytes / elapsed
        prefix = "progress" if label is None else f"progress [{label}]"
        print(
            f"{prefix}: files={copied_files}/{file_count} bytes={_format_bytes(copied_bytes)}/{_format_bytes(total_bytes)} "
            f"rate={_format_bytes(int(rate))}/s elapsed={elapsed:.1f}s"
        )
        last_report = now

    def copy_path(src_path: Path, dst_path: Path, *, label: str) -> None:
        nonlocal copied_files, copied_bytes
        if src_path.is_dir():
            dst_path.mkdir(parents=True, exist_ok=True)
            for child in sorted(src_path.iterdir(), key=lambda path: path.name):
                copy_path(child, dst_path / child.name, label=label)
            return

        src_size = src_path.stat().st_size
        if dst_path.exists() and dst_path.stat().st_size == src_size:
            copied_files += 1
            copied_bytes += src_size
            report(label=label)
            return

        dst_path.parent.mkdir(parents=True, exist_ok=True)
        last_error = None
        for attempt in range(1, max_copy_retries + 1):
            try:
                shutil.copy2(src_path, dst_path)
                copied_files += 1
                copied_bytes += src_size
                report(label=label)
                return
            except OSError as exc:  # pragma: no cover - exercised in Colab when Drive stalls
                last_error = exc
                print(f"copy retry {attempt}/{max_copy_retries} failed for {src_path}: {exc}")
                time.sleep(min(10, 2 * attempt))

        raise OSError(f"Failed to copy {src_path} after {max_copy_retries} retries") from last_error

    dst_root.mkdir(parents=True, exist_ok=True)
    for entry_idx, src_path in enumerate(entries, start=1):
        label = f"{entry_idx}/{len(entries)} {src_path.name}"
        print(f"starting {label}")
        copy_path(src_path, dst_root / src_path.name, label=label)
        report(force=True, label=label)

    return file_count, total_bytes


def stable_text_seed(text: str, base_seed: int) -> int:
    return int(base_seed + sum((idx + 1) * ord(ch) for idx, ch in enumerate(text)))


def _choose_shard_cache_gb() -> float:
    if psutil is None:
        return 4.0
    available_gb = psutil.virtual_memory().available / (1024 ** 3)
    return float(min(8.0, max(2.0, 0.35 * available_gb)))


class ShardStore:
    def __init__(
        self,
        cache_root: Path,
        ram_cache_gb: float,
        *,
        modalities: Sequence[str] = ("tx", "sbp"),
        dataset_cache_roots: Mapping[str, str | Path] | None = None,
    ):
        self.cache_root = Path(cache_root)
        self.dataset_cache_roots = {
            str(dataset): Path(root)
            for dataset, root in (dataset_cache_roots or {}).items()
        }
        self.max_bytes = int(ram_cache_gb * (1024 ** 3))
        requested_modalities = (
            (str(modalities),)
            if isinstance(modalities, str)
            else tuple(str(item) for item in modalities)
        )
        requested_modality_names = {
            item.strip().lower() for item in requested_modalities
        }
        normalized_modalities = tuple(
            name
            for name in ("tx", "sbp")
            if name in requested_modality_names
        )
        unsupported = requested_modality_names.difference({"tx", "sbp"})
        if unsupported:
            raise ValueError(f"Unsupported ShardStore modalities: {sorted(unsupported)}")
        if not normalized_modalities:
            raise ValueError("ShardStore modalities must include at least one of {'tx', 'sbp'}")
        self.modalities = normalized_modalities
        self._cache: OrderedDict[str, dict[str, np.ndarray | None | int]] = OrderedDict()
        self._cached_bytes = 0
        self._cache_hits = 0
        self._cache_misses = 0
        self._evictions = 0
        self._bytes_read = 0

    def clear(self) -> None:
        self._cache.clear()
        self._cached_bytes = 0
        self._cache_hits = 0
        self._cache_misses = 0
        self._evictions = 0
        self._bytes_read = 0

    def summary(self) -> dict[str, Any]:
        total_requests = self._cache_hits + self._cache_misses
        return {
            "modalities": list(self.modalities),
            "dataset_cache_roots": {
                dataset: str(root)
                for dataset, root in sorted(self.dataset_cache_roots.items())
            },
            "cached_shards": float(len(self._cache)),
            "cached_gb": self._cached_bytes / (1024 ** 3),
            "budget_gb": self.max_bytes / (1024 ** 3),
            "cache_hits": int(self._cache_hits),
            "cache_misses": int(self._cache_misses),
            "cache_hit_rate": (
                float(self._cache_hits / total_requests) if total_requests else 0.0
            ),
            "evictions": int(self._evictions),
            "bytes_read": int(self._bytes_read),
            "gb_read": self._bytes_read / (1024 ** 3),
        }

    def _load_array(self, path: Path) -> np.ndarray | None:
        if not path.exists():
            return None
        # Use eager in-memory loads here rather than memmaps. ShardStore already
        # enforces a RAM budget, and keeping many memmapped arrays alive at once
        # can exhaust per-process file-descriptor limits during full-corpus
        # stats recomputation on laptops.
        return np.load(path, allow_pickle=False)

    def _load_shard(self, shard_relpath: str) -> dict[str, np.ndarray | None | int]:
        relative_path = Path(shard_relpath)
        dataset = relative_path.parts[0] if relative_path.parts else ""
        shard_path = self.dataset_cache_roots.get(dataset, self.cache_root) / relative_path
        shard = {
            "time_offsets": self._load_array(shard_path / "time_offsets.npy"),
            "tx": (
                self._load_array(shard_path / "tx.npy")
                if "tx" in self.modalities
                else None
            ),
            "sbp": (
                self._load_array(shard_path / "sbp.npy")
                if "sbp" in self.modalities
                else None
            ),
        }
        time_offsets = shard["time_offsets"]
        if time_offsets is None:
            raise FileNotFoundError(f"Missing time_offsets.npy for shard {shard_path}")
        for modality in self.modalities:
            if shard[modality] is None:
                raise FileNotFoundError(
                    f"Requested modality {modality!r} is missing from shard {shard_path}"
                )
        shard["bytes"] = int(
            time_offsets.nbytes
            + (0 if shard["tx"] is None else shard["tx"].nbytes)
            + (0 if shard["sbp"] is None else shard["sbp"].nbytes)
        )
        self._bytes_read += int(shard["bytes"])
        return shard

    def get(self, shard_relpath: str) -> dict[str, np.ndarray | None | int]:
        key = str(shard_relpath)
        cached = self._cache.get(key)
        if cached is not None:
            self._cache_hits += 1
            self._cache.move_to_end(key)
            return cached

        self._cache_misses += 1
        shard = self._load_shard(key)
        shard_bytes = int(shard["bytes"])
        if shard_bytes <= self.max_bytes:
            while self._cache and self._cached_bytes + shard_bytes > self.max_bytes:
                _, evicted = self._cache.popitem(last=False)
                self._cached_bytes -= int(evicted["bytes"])
                self._evictions += 1
            self._cache[key] = shard
            self._cached_bytes += shard_bytes
        return shard


def _normalize_segment(
    x_seq: torch.Tensor,
    feature_mask: torch.Tensor,
    *,
    session_feature_stats: dict[str, tuple[torch.Tensor, torch.Tensor]] | None = None,
    session_key: str | None = None,
    min_scale_std: float = 0.1,
    clip_value: float = 20.0,
    use_normalization: bool = True,
) -> torch.Tensor:
    if not bool(use_normalization):
        return x_seq
    return _normalize_segment_session_featurewise(
        x_seq,
        feature_mask,
        session_feature_stats=session_feature_stats,
        session_key=session_key,
        clip_value=clip_value,
    )


def _normalize_segment_session_featurewise(
    x_seq: torch.Tensor,
    feature_mask: torch.Tensor,
    *,
    session_feature_stats: dict[str, tuple[torch.Tensor, torch.Tensor]] | None,
    session_key: str | None,
    clip_value: float = 20.0,
) -> torch.Tensor:
    if session_feature_stats is None or session_key is None:
        raise ValueError("Session feature stats are required when normalization is enabled.")
    if session_key not in session_feature_stats:
        raise KeyError(f"Missing session feature stats for {session_key}")

    x_norm = x_seq.clone()
    present_idx = torch.nonzero(feature_mask.bool(), as_tuple=False).squeeze(1)
    if present_idx.numel() == 0:
        return x_norm

    mean, std = session_feature_stats[session_key]
    mean = mean.to(device=x_norm.device, dtype=x_norm.dtype)
    std = std.to(device=x_norm.device, dtype=x_norm.dtype).clamp_min(1e-6)
    centered = x_norm[:, present_idx] - mean[present_idx]
    x_norm[:, present_idx] = (centered / std[present_idx]).clamp(min=-clip_value, max=clip_value)
    return x_norm


def _gaussian_kernel_1d(
    sigma_bins: float,
    *,
    device: torch.device,
    dtype: torch.dtype,
    radius: int | None = None,
) -> torch.Tensor:
    sigma = float(sigma_bins)
    if sigma <= 0.0:
        return torch.ones((1,), device=device, dtype=dtype)
    effective_radius = (
        int(radius)
        if radius is not None
        else max(1, int(math.ceil(4.0 * sigma)))
    )
    positions = torch.arange(
        -effective_radius,
        effective_radius + 1,
        device=device,
        dtype=dtype,
    )
    kernel = torch.exp(-0.5 * (positions / sigma).pow(2))
    return kernel / kernel.sum().clamp_min(1e-8)


def _apply_gaussian_smoothing(
    x_seq: torch.Tensor,
    feature_mask: torch.Tensor,
    *,
    sigma_bins: float,
) -> torch.Tensor:
    sigma = float(sigma_bins)
    if sigma <= 0.0 or x_seq.shape[0] <= 1:
        return x_seq
    present_idx = torch.nonzero(feature_mask.bool(), as_tuple=False).squeeze(1)
    if present_idx.numel() == 0:
        return x_seq

    max_reflect_radius = int(x_seq.shape[0] - 1)
    if max_reflect_radius <= 0:
        return x_seq
    kernel_radius = min(max(1, int(math.ceil(4.0 * sigma))), max_reflect_radius)
    kernel = _gaussian_kernel_1d(
        sigma,
        device=x_seq.device,
        dtype=x_seq.dtype,
        radius=kernel_radius,
    )
    kernel = kernel / kernel.sum().clamp_min(1e-8)

    selected = x_seq[:, present_idx].transpose(0, 1).unsqueeze(0)  # (1, C, T)
    padded = F.pad(selected, (kernel_radius, kernel_radius), mode="reflect")
    weight = kernel.view(1, 1, -1).expand(selected.shape[1], 1, -1)
    smoothed = F.conv1d(padded, weight, groups=selected.shape[1]).squeeze(0).transpose(0, 1)

    out = x_seq.clone()
    out[:, present_idx] = smoothed
    return out


def _session_stat_key(dataset: str, session_id: str) -> str:
    return f"{dataset}:{session_id}"


def resolve_boundary_key(
    *,
    dataset: str,
    session_id: str,
    subject_id: str | None,
    boundary_key_mode: str,
) -> str:
    if boundary_key_mode == "session":
        return f"{dataset}:{session_id}"
    if boundary_key_mode == "subject_if_available":
        if subject_id:
            return f"{dataset}:{subject_id}"
        return f"{dataset}:{session_id}"
    raise ValueError(f"Unsupported boundary_key_mode: {boundary_key_mode}")


def _compute_session_feature_stats(
    shard_store: ShardStore,
    rows_by_dataset: dict[str, list[ExampleRow]],
    config: CacheAccessConfig,
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    assert isinstance(config.signal_spec, SignalSpec)
    feature_contract = config.signal_spec.contract
    print("computing SSL session-level featurewise z-scoring stats...")
    session_rows: dict[str, list[ExampleRow]] = defaultdict(list)
    for dataset, rows in rows_by_dataset.items():
        for row in rows:
            session_rows[
                resolve_boundary_key(
                    dataset=dataset,
                    session_id=row.session_id,
                    subject_id=row.subject_id,
                    boundary_key_mode=config.boundary_key_mode,
                )
            ].append(row)

    full_dim = int(config.full_dim)
    session_stats: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    total_sessions = len(session_rows)
    bin_stride = int(SESSION_STATS_BIN_STRIDE)

    for session_idx, session_key in enumerate(sorted(session_rows), start=1):
        rows = session_rows[session_key]
        sum_x = np.zeros((full_dim,), dtype=np.float64)
        sum_x2 = np.zeros((full_dim,), dtype=np.float64)
        count_x = np.zeros((full_dim,), dtype=np.float64)

        for row in rows:
            shard = shard_store.get(row.shard_relpath)
            time_offsets = shard["time_offsets"]
            assert isinstance(time_offsets, np.ndarray)
            start = int(time_offsets[row.example_index])
            stop = int(time_offsets[row.example_index + 1])
            raw_len = int(max(stop - start, 0))
            if raw_len <= 0:
                continue

            tx = shard["tx"]
            if feature_contract.uses_tx and isinstance(tx, np.ndarray):
                tx_start, tx_stop = config.signal_spec.selected_columns_for_width(
                    "tx", tx.shape[1]
                )
                tx_window = np.asarray(
                    tx[start:stop:bin_stride, tx_start:tx_stop],
                    dtype=np.float64,
                )
                tx_dim = min(tx_window.shape[1], int(config.tx_dim))
                sum_x[:tx_dim] += tx_window[:, :tx_dim].sum(axis=0)
                sum_x2[:tx_dim] += np.square(tx_window[:, :tx_dim]).sum(axis=0)
                count_x[:tx_dim] += tx_window.shape[0]

            sbp = shard["sbp"]
            if feature_contract.uses_sbp and isinstance(sbp, np.ndarray):
                sbp_column_start, sbp_column_stop = (
                    config.signal_spec.selected_columns_for_width("sbp", sbp.shape[1])
                )
                sbp_window = np.asarray(
                    sbp[
                        start:stop:bin_stride,
                        sbp_column_start:sbp_column_stop,
                    ],
                    dtype=np.float64,
                )
                sbp_dim = min(sbp_window.shape[1], int(config.sbp_dim))
                sbp_start = feature_contract.feature_start(
                    "sbp",
                    tx_dim=int(config.tx_dim),
                )
                sbp_slice = slice(sbp_start, sbp_start + sbp_dim)
                sum_x[sbp_slice] += sbp_window[:, :sbp_dim].sum(axis=0)
                sum_x2[sbp_slice] += np.square(sbp_window[:, :sbp_dim]).sum(axis=0)
                count_x[sbp_slice] += sbp_window.shape[0]

        mean = np.zeros((full_dim,), dtype=np.float32)
        std = np.ones((full_dim,), dtype=np.float32)
        present_mask = count_x > 0
        if present_mask.any():
            mean64 = sum_x[present_mask] / count_x[present_mask]
            var64 = np.maximum(sum_x2[present_mask] / count_x[present_mask] - np.square(mean64), 1e-6)
            mean[present_mask] = mean64.astype(np.float32)
            std[present_mask] = np.sqrt(var64).astype(np.float32)
        session_stats[session_key] = (torch.from_numpy(mean), torch.from_numpy(std))

        if session_idx == 1 or session_idx % 25 == 0 or session_idx == total_sessions:
            print(f" session_stats={session_idx}/{total_sessions} current={session_key}")

    return session_stats


def load_precomputed_session_feature_stats_into_cache_context(
    *,
    cache_context: CacheContext,
    stats_path: str | Path,
) -> dict[str, Any]:
    session_feature_stats, metadata, path = _load_precomputed_session_feature_stats(
        stats_path=stats_path,
        cache_root=cache_context.drive_cache_root,
        signal_spec=cache_context.signal_spec,
        dataset_plan=cache_context.config.dataset_plan,
        boundary_key_mode=str(cache_context.boundary_key_mode),
        dataset_cache_roots=cache_context.drive_dataset_cache_roots,
    )
    cache_context.session_feature_stats = dict(session_feature_stats)
    return {
        "stats_path": path,
        "metadata": metadata,
        "session_feature_stats": session_feature_stats,
        "session_count": int(len(session_feature_stats)),
        "use_normalization": cache_context.use_normalization,
    }


def _load_precomputed_session_feature_stats(
    *,
    stats_path: str | Path,
    cache_root: str | Path,
    signal_spec: SignalSpec | Mapping[str, Any],
    dataset_plan: DatasetPlan | Mapping[str, Sequence[str]],
    boundary_key_mode: str,
    dataset_cache_roots: Mapping[str, str | Path] | None = None,
) -> tuple[dict[str, tuple[torch.Tensor, torch.Tensor]], dict[str, Any], Path]:
    resolved_signal_spec = SignalSpec.from_value(signal_spec)
    resolved_dataset_plan = DatasetPlan.from_value(dataset_plan)
    expected_dim = int(resolved_signal_spec.full_dim)
    path = Path(stats_path)
    canonical_path = resolve_precomputed_session_stats_path(
        cache_root=cache_root,
        signal_spec=resolved_signal_spec,
        dataset_plan=resolved_dataset_plan,
        boundary_key_mode=str(boundary_key_mode),
    )
    recompute_cmd = build_recompute_session_feature_stats_command(
        cache_root=cache_root,
        output_path=canonical_path,
        signal_spec=resolved_signal_spec,
        dataset_plan=resolved_dataset_plan,
        boundary_key_mode=str(boundary_key_mode),
        dataset_cache_roots=dataset_cache_roots,
    )
    payload, metadata, path, _ = _load_artifact_payload_and_sidecar(
        path=path,
        canonical_path=canonical_path,
        recompute_cmd=recompute_cmd,
        artifact_name="session stats",
        expected_kind="session_featurewise_zscore_stats",
    )
    raw_stats = payload.get("session_feature_stats")
    if not isinstance(raw_stats, dict):
        raise KeyError("Precomputed session stats payload is missing 'session_feature_stats'.")

    session_feature_stats: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for key, value in raw_stats.items():
        if not isinstance(value, (tuple, list)) or len(value) != 2:
            raise ValueError(
                f"Session stats entry for {key!r} must be a 2-item (mean, std) tuple/list."
            )
        mean, std = value
        mean_t = torch.as_tensor(mean).float().cpu()
        std_t = torch.as_tensor(std).float().cpu()
        if mean_t.numel() != expected_dim or std_t.numel() != expected_dim:
            raise ValueError(
                "Precomputed session stats artifact is stale or incompatible.\n"
                f"expected_path: {canonical_path}\n"
                f"requested_path: {path}\n"
                f"reason: entry {key!r} has dim mean:{mean_t.numel()} std:{std_t.numel()} expected {expected_dim}\n"
                f"recompute_command: {recompute_cmd}"
            )
        session_feature_stats[str(key)] = (mean_t, std_t)

    expected_cache_root = str(Path(cache_root).resolve())
    expected_cache_variant = _cache_variant_name(cache_root)
    normalized_dataset_cache_roots = {
        str(dataset): Path(root)
        for dataset, root in sorted((dataset_cache_roots or {}).items())
    }
    expected_cache_signature = (
        _compute_dataset_cache_source_signature(normalized_dataset_cache_roots)
        if normalized_dataset_cache_roots
        else _compute_cache_source_signature(Path(cache_root))
    )
    common_metadata = {
        "source_cache_root": expected_cache_root,
        "source_cache_variant": expected_cache_variant,
        "source_cache_signature": expected_cache_signature,
        "signal_spec": resolved_signal_spec.to_dict(),
        "dataset_plan": resolved_dataset_plan.to_dict(),
        "boundary_key_mode": str(boundary_key_mode),
        "session_stats_bin_stride": SESSION_STATS_BIN_STRIDE,
    }
    if normalized_dataset_cache_roots:
        common_metadata["source_cache_roots"] = {
            dataset: str(root.resolve())
            for dataset, root in normalized_dataset_cache_roots.items()
        }
    mismatches = _validate_common_artifact_metadata(
        metadata=metadata,
        expected_metadata=common_metadata,
    )
    if mismatches:
        mismatch_text = "; ".join(mismatches)
        raise ValueError(
            "Precomputed session stats artifact is stale or incompatible.\n"
            f"expected_path: {canonical_path}\n"
            f"requested_path: {path}\n"
            f"reason: {mismatch_text}\n"
            f"recompute_command: {recompute_cmd}"
        )
    return session_feature_stats, metadata, path


def sample_base_segment(
    cache_context: CacheContext,
    example: ExampleRow,
    segment_bins: int,
    py_rng: random.Random,
) -> dict[str, Any]:
    signal_spec = cache_context.config.signal_spec
    assert isinstance(signal_spec, SignalSpec)
    feature_contract = signal_spec.contract
    boundary_key = resolve_boundary_key(
        dataset=example.dataset,
        session_id=example.session_id,
        subject_id=example.subject_id,
        boundary_key_mode=cache_context.boundary_key_mode,
    )
    session_key = boundary_key
    shard = cache_context.shard_store.get(example.shard_relpath)
    time_offsets = shard["time_offsets"]
    assert isinstance(time_offsets, np.ndarray)
    start = int(time_offsets[example.example_index])
    stop = int(time_offsets[example.example_index + 1])
    length = stop - start
    total_needed = int(segment_bins)
    max_start = length - total_needed
    if max_start < 0:
        raise ValueError(
            f"Example {example.dataset}:{example.session_id} length={length} cannot support segment_bins={segment_bins}"
        )

    offset = py_rng.randrange(max_start + 1)
    src_start = start + offset
    src_stop = src_start + total_needed
    x_seq = np.zeros((total_needed, cache_context.full_dim), dtype=np.float32)
    feature_mask = np.zeros((cache_context.full_dim,), dtype=np.float32)

    tx = shard["tx"]
    if feature_contract.uses_tx and isinstance(tx, np.ndarray):
        tx_column_start, tx_column_stop = signal_spec.selected_columns_for_width(
            "tx", tx.shape[1]
        )
        tx_window = np.asarray(
            tx[
                src_start:src_stop,
                tx_column_start:tx_column_stop,
            ],
            dtype=np.float32,
        )
        tx_dim = min(tx_window.shape[1], cache_context.tx_dim)
        x_seq[:, :tx_dim] = tx_window[:, :tx_dim]
        feature_mask[:tx_dim] = 1.0

    sbp = shard["sbp"]
    if feature_contract.uses_sbp and isinstance(sbp, np.ndarray):
        sbp_column_start, sbp_column_stop = signal_spec.selected_columns_for_width(
            "sbp", sbp.shape[1]
        )
        sbp_window = np.asarray(
            sbp[
                src_start:src_stop,
                sbp_column_start:sbp_column_stop,
            ],
            dtype=np.float32,
        )
        sbp_dim = min(sbp_window.shape[1], cache_context.sbp_dim)
        sbp_start = feature_contract.feature_start(
            "sbp",
            tx_dim=int(cache_context.tx_dim),
        )
        x_seq[:, sbp_start : sbp_start + sbp_dim] = sbp_window[:, :sbp_dim]
        feature_mask[sbp_start : sbp_start + sbp_dim] = 1.0

    x_seq_t = torch.from_numpy(x_seq)
    feature_mask_t = torch.from_numpy(feature_mask)
    x_norm = _normalize_segment(
        x_seq_t,
        feature_mask_t,
        session_feature_stats=cache_context.session_feature_stats,
        session_key=session_key,
        use_normalization=cache_context.use_normalization,
    )

    return {
        "x": x_norm,
        "feature_mask": feature_mask_t,
        "length": int(segment_bins),
        "dataset": example.dataset,
        "session_id": example.session_id,
        "session_key": session_key,
        "boundary_key": boundary_key,
        "shard_relpath": example.shard_relpath,
        "has_tx": example.has_tx,
        "has_sbp": example.has_sbp,
        "orig_len": length,
    }


def stack_segment_batch(samples: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "x": torch.stack([item["x"] for item in samples], dim=0),
        "feature_mask": torch.stack([item["feature_mask"] for item in samples], dim=0),
        "lengths": torch.tensor([item["length"] for item in samples], dtype=torch.long),
        "datasets": [item["dataset"] for item in samples],
        "session_keys": [item["boundary_key"] for item in samples],
        "boundary_keys": [item["boundary_key"] for item in samples],
        "sessions": [item["session_id"] for item in samples],
        "shard_relpaths": [item["shard_relpath"] for item in samples],
    }


def _valid_row_weights(rows: list[ExampleRow], segment_bins: int) -> np.ndarray:
    return np.array([max(0, row.n_time_bins - segment_bins + 1) for row in rows], dtype=np.float64)


def get_sampling_plan(
    cache_context: CacheContext,
    split_name: str,
    segment_bins: int,
    dataset_weight_alpha: float,
) -> SamplingPlan:
    key = (split_name, int(segment_bins), float(dataset_weight_alpha))
    cached = cache_context.sampling_plan_cache.get(key)
    if cached is not None:
        return cached

    shard_rows_by_dataset = {}
    shard_keys_by_dataset = {}
    shard_probs_by_dataset = {}
    row_probs_within_shard_by_dataset = {}
    dataset_mass = {}

    for dataset in cache_context.pretrain_datasets:
        rows = cache_context.split_rows_by_dataset[split_name][dataset]
        weights = _valid_row_weights(rows, segment_bins)
        keep_mask = weights > 0
        kept_rows = [row for row, keep in zip(rows, keep_mask) if keep]
        kept_weights = weights[keep_mask]
        if len(kept_rows) == 0:
            continue

        dataset_mass[dataset] = float(kept_weights.sum())
        shard_rows = defaultdict(list)
        shard_weights = defaultdict(list)
        for row, weight in zip(kept_rows, kept_weights):
            shard_rows[row.shard_relpath].append(row)
            shard_weights[row.shard_relpath].append(float(weight))

        shard_keys = list(shard_rows.keys())
        shard_mass = np.array([sum(shard_weights[name]) for name in shard_keys], dtype=np.float64)
        shard_probs = shard_mass / shard_mass.sum()

        shard_rows_by_dataset[dataset] = dict(shard_rows)
        shard_keys_by_dataset[dataset] = shard_keys
        shard_probs_by_dataset[dataset] = shard_probs
        row_probs_within_shard_by_dataset[dataset] = {
            name: np.array(weight_list, dtype=np.float64) / np.sum(weight_list)
            for name, weight_list in shard_weights.items()
        }

    dataset_names = tuple(dataset for dataset in cache_context.pretrain_datasets if dataset in dataset_mass)
    if not dataset_names:
        raise RuntimeError(f"Split {split_name} has no datasets with enough bins for segment_bins={segment_bins}")

    dataset_probs = np.array(
        [dataset_mass[dataset] ** dataset_weight_alpha for dataset in dataset_names],
        dtype=np.float64,
    )
    dataset_probs = dataset_probs / dataset_probs.sum()

    plan = SamplingPlan(
        split_name=split_name,
        segment_bins=int(segment_bins),
        dataset_weight_alpha=float(dataset_weight_alpha),
        dataset_names=dataset_names,
        dataset_probs=dataset_probs,
        shard_rows_by_dataset=shard_rows_by_dataset,
        shard_keys_by_dataset=shard_keys_by_dataset,
        shard_probs_by_dataset=shard_probs_by_dataset,
        row_probs_within_shard_by_dataset=row_probs_within_shard_by_dataset,
    )
    cache_context.sampling_plan_cache[key] = plan
    return plan


class SegmentBatchSampler:
    def __init__(
        self,
        cache_context: CacheContext,
        split_name: str,
        segment_bins: int,
        batch_size: int,
        seed: int,
        dataset_weight_alpha: float,
        examples_per_shard: int,
    ):
        self.cache_context = cache_context
        self.split_name = split_name
        self.segment_bins = int(segment_bins)
        self.batch_size = int(batch_size)
        self.examples_per_shard = max(1, int(examples_per_shard))
        self.seed = int(seed)
        self.plan = get_sampling_plan(cache_context, split_name, self.segment_bins, dataset_weight_alpha)
        self.py_rng = random.Random(self.seed)
        self.np_rng = np.random.default_rng(self.seed)

    def sample_batch(self, batch_size: int | None = None) -> dict[str, Any]:
        batch_size = self.batch_size if batch_size is None else int(batch_size)
        requested_dataset_idx = self.np_rng.choice(
            len(self.plan.dataset_names),
            size=batch_size,
            p=self.plan.dataset_probs,
        )
        dataset_counts = Counter(self.plan.dataset_names[int(idx)] for idx in requested_dataset_idx)

        samples = []
        for dataset, n_examples in dataset_counts.items():
            shard_keys = self.plan.shard_keys_by_dataset[dataset]
            shard_probs = self.plan.shard_probs_by_dataset[dataset]
            n_shards = max(1, math.ceil(n_examples / self.examples_per_shard))
            replace_shards = n_shards > len(shard_keys)
            sampled_shard_idx = self.np_rng.choice(
                len(shard_keys),
                size=n_shards,
                replace=replace_shards,
                p=shard_probs,
            )

            remaining = int(n_examples)
            for shard_choice_idx, shard_idx in enumerate(np.atleast_1d(sampled_shard_idx)):
                take = min(self.examples_per_shard, remaining)
                if shard_choice_idx == n_shards - 1:
                    take = remaining

                shard_key = shard_keys[int(shard_idx)]
                shard_rows = self.plan.shard_rows_by_dataset[dataset][shard_key]
                row_probs = self.plan.row_probs_within_shard_by_dataset[dataset][shard_key]
                row_choices = self.np_rng.choice(len(shard_rows), size=take, replace=True, p=row_probs)
                for row_idx in np.atleast_1d(row_choices):
                    example = shard_rows[int(row_idx)]
                    samples.append(
                        sample_base_segment(
                            self.cache_context,
                            example,
                            segment_bins=self.segment_bins,
                            py_rng=self.py_rng,
                        )
                    )

                remaining -= take
                if remaining <= 0:
                    break

        order = self.np_rng.permutation(len(samples))
        samples = [samples[int(idx)] for idx in order]
        return stack_segment_batch(samples)


def build_segment_sampler(
    cache_context: CacheContext,
    split_name: str,
    batch_size: int,
    *,
    seed: int,
    segment_bins: int,
    dataset_weight_alpha: float,
    examples_per_shard: int,
) -> SegmentBatchSampler:
    if split_name == "val" and not cache_context.has_val_datasets:
        raise RuntimeError("No validation datasets are eligible for session-disjoint validation.")
    return SegmentBatchSampler(
        cache_context=cache_context,
        split_name=split_name,
        segment_bins=segment_bins,
        batch_size=batch_size,
        seed=seed,
        dataset_weight_alpha=dataset_weight_alpha,
        examples_per_shard=examples_per_shard,
    )


def _preflight_cache_inputs(
    *,
    dataset_cache_roots: Mapping[str, Path],
    config: CacheAccessConfig,
) -> None:
    """Validate the complete dataset/signal plan before copying or loading shards."""

    assert isinstance(config.signal_spec, SignalSpec)
    for dataset, cache_root in sorted(dataset_cache_roots.items()):
        manifest_path = Path(cache_root) / dataset / "manifest.jsonl"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest missing for dataset {dataset}: {manifest_path}")
        source_split_filter = _pretrain_source_splits_for_dataset(config, dataset)
        allowed_splits = set(source_split_filter or ())
        selected_rows = 0
        with manifest_path.open() as handle:
            for line in handle:
                payload = json.loads(line)
                source_split = str(payload.get("source_split", "")).strip().lower()
                if allowed_splits and source_split not in allowed_splits:
                    continue
                selected_rows += 1
                n_tx_features = int(payload.get("n_tx_features", 0) or 0)
                n_sbp_features = int(payload.get("n_sbp_features", 0) or 0)
                if not config.signal_spec.row_is_compatible(
                    has_tx=bool(payload.get("has_tx", n_tx_features > 0)),
                    has_sbp=bool(payload.get("has_sbp", n_sbp_features > 0)),
                    n_tx_features=n_tx_features,
                    n_sbp_features=n_sbp_features,
                ):
                    raise ValueError(
                        "Dataset plan is incompatible with the selected signal before "
                        "cache copying begins. "
                        f"dataset={dataset!r}, source_split={source_split!r}, "
                        f"example={payload.get('example_id')!r}, "
                        f"signal_spec={config.signal_spec.to_dict()}, "
                        f"n_tx_features={n_tx_features}, n_sbp_features={n_sbp_features}."
                    )
        if selected_rows == 0:
            raise ValueError(
                f"Dataset plan selected no manifest rows for {dataset!r}; "
                f"requested source splits={sorted(allowed_splits)}"
            )


def prepare_cache_context(
    *,
    cache_candidates: Sequence[Path],
    config: CacheAccessConfig,
) -> CacheContext:
    ensure_runtime_smoothing_disabled(config, context="prepare_cache_context")
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    candidate_paths = [Path(path) for path in cache_candidates]
    drive_cache_root = next((path for path in candidate_paths if path.exists()), None)
    if drive_cache_root is None:
        raise FileNotFoundError(
            "No cache root found. Candidates checked: " + ", ".join(str(path) for path in candidate_paths)
        )

    all_drive_dataset_cache_roots = _resolve_drive_dataset_cache_roots(
        drive_cache_root,
        config.dataset_cache_roots,
    )
    available_datasets = sorted(all_drive_dataset_cache_roots)
    assert isinstance(config.dataset_plan, DatasetPlan)
    missing_datasets = sorted(
        set(config.dataset_plan.dataset_names).difference(available_datasets)
    )
    if missing_datasets:
        raise FileNotFoundError(
            "Dataset plan contains unavailable dataset(s): "
            f"{missing_datasets}. Available datasets: {available_datasets}"
        )
    pretrain_datasets = list(config.dataset_plan.dataset_names)
    drive_dataset_cache_roots = {
        dataset: all_drive_dataset_cache_roots[dataset]
        for dataset in pretrain_datasets
    }
    _preflight_cache_inputs(
        dataset_cache_roots=drive_dataset_cache_roots,
        config=config,
    )
    source_signature = _compute_dataset_cache_source_signature(
        drive_dataset_cache_roots
    )
    local_cache_name = f"{drive_cache_root.name}_plan_{source_signature[:12]}"

    local_cache_root = Path(config.local_cache_base) / local_cache_name
    local_cache_status_path = local_cache_root.parent / f"{local_cache_name}_copy_status.json"

    if config.force_recopy_local_cache and config.mode == "copy_to_local" and local_cache_root.exists():
        print("removing existing local cache:", local_cache_root)
        shutil.rmtree(local_cache_root)
    if config.force_recopy_local_cache and config.mode == "copy_to_local" and local_cache_status_path.exists():
        local_cache_status_path.unlink()

    copy_status = _load_copy_status(local_cache_status_path)
    if config.mode == "copy_to_local":
        copy_is_current = (
            _copy_complete_for_current_source(
                status=copy_status,
                drive_cache_root=drive_cache_root,
                local_cache_root=local_cache_root,
                source_signature=source_signature,
            )
            and copy_status.get("source_cache_roots")
            == {
                dataset: str(root)
                for dataset, root in sorted(drive_dataset_cache_roots.items())
            }
        )
        if (not local_cache_root.exists()) or not copy_is_current:
            if local_cache_root.exists():
                print("removing stale local cache:", local_cache_root)
                shutil.rmtree(local_cache_root)
            if local_cache_status_path.exists():
                local_cache_status_path.unlink()
            local_cache_root.parent.mkdir(parents=True, exist_ok=True)
            print("copying cache to local disk...")
            print("source:", drive_cache_root)
            print("source signature:", source_signature[:12])
            print("dest  :", local_cache_root)
            t0 = time.time()
            file_count = 0
            total_bytes = 0
            local_cache_root.mkdir(parents=True, exist_ok=True)
            for dataset, source_root in sorted(drive_dataset_cache_roots.items()):
                copied_files, copied_bytes = copy_tree_with_progress(
                    source_root / dataset,
                    local_cache_root / dataset,
                )
                file_count += copied_files
                total_bytes += copied_bytes
            _write_copy_status(
                local_cache_status_path,
                drive_cache_root=drive_cache_root,
                local_cache_root=local_cache_root,
                source_signature=source_signature,
                file_count=file_count,
                total_bytes=total_bytes,
            )
            status_payload = _load_copy_status(local_cache_status_path) or {}
            status_payload["source_cache_roots"] = {
                dataset: str(root)
                for dataset, root in sorted(drive_dataset_cache_roots.items())
            }
            local_cache_status_path.write_text(json.dumps(status_payload, indent=2))
            print(f"copy complete in {time.time() - t0:.1f}s")
        else:
            print("using existing local cache:", local_cache_root)
            print("source signature:", source_signature[:12])
        cache_root = local_cache_root
        dataset_cache_roots = {
            dataset: local_cache_root for dataset in pretrain_datasets
        }
        cache_copy_used = True
    else:
        cache_root = drive_cache_root
        dataset_cache_roots = dict(drive_dataset_cache_roots)
        cache_copy_used = False
        print("using Drive-backed cache directly; skipping local copy")
        print("source:", drive_cache_root)
        print("source signature:", source_signature[:12])

    os.environ["SSL_AUTORESEARCH_CACHE_ROOT"] = str(cache_root)

    rows_by_dataset: dict[str, list[ExampleRow]] = {}
    split_rows_by_dataset: dict[str, dict[str, list[ExampleRow]]] = {"train": {}, "val": {}}
    session_split_summary: dict[str, dict[str, Any]] = {}

    for dataset in pretrain_datasets:
        ds_root = dataset_cache_roots[dataset] / dataset
        manifest_path = ds_root / "manifest.jsonl"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest missing for dataset {dataset}: {manifest_path}")

        rows: list[ExampleRow] = []
        source_split_filter = _pretrain_source_splits_for_dataset(config, dataset)
        with manifest_path.open() as handle:
            for line in handle:
                payload = json.loads(line)
                source_split = str(payload.get("source_split", "")).strip().lower()
                if source_split_filter and source_split not in set(source_split_filter):
                    continue
                rows.append(
                    ExampleRow(
                        dataset=dataset,
                        session_id=str(payload["session_id"]),
                        subject_id=(
                            str(payload["subject_id"])
                            if payload.get("subject_id") is not None
                            else None
                        ),
                        shard_relpath=str(payload["shard_relpath"]),
                        example_index=int(payload["example_index"]),
                        n_time_bins=int(payload["n_time_bins"]),
                        has_tx=bool(payload.get("has_tx", False)),
                        has_sbp=bool(payload.get("has_sbp", False)),
                        n_tx_features=int(payload.get("n_tx_features", 0) or 0),
                        n_sbp_features=int(payload.get("n_sbp_features", 0) or 0),
                    )
                )
        rows_by_dataset[dataset] = rows

        session_ids = sorted({row.session_id for row in rows})
        if len(session_ids) < 2:
            train_session_ids = list(session_ids)
            val_session_ids: list[str] = []
        else:
            split_rng = random.Random(stable_text_seed(dataset, config.seed))
            shuffled = list(session_ids)
            split_rng.shuffle(shuffled)
            val_count = max(1, int(math.ceil(0.2 * len(shuffled))))
            val_count = min(val_count, len(shuffled) - 1)
            val_session_ids = sorted(shuffled[:val_count])
            train_session_ids = sorted(shuffled[val_count:])

        train_set = set(train_session_ids)
        val_set = set(val_session_ids)
        split_rows_by_dataset["train"][dataset] = [row for row in rows if row.session_id in train_set]
        split_rows_by_dataset["val"][dataset] = [row for row in rows if row.session_id in val_set]
        session_split_summary[dataset] = {
            "total_sessions": len(session_ids),
            "train_sessions": len(train_session_ids),
            "val_sessions": len(val_session_ids),
            "val_eligible": len(session_ids) >= 2,
            "train_examples": len(split_rows_by_dataset["train"][dataset]),
            "val_examples": len(split_rows_by_dataset["val"][dataset]),
        }

    shard_cache_ram_gb = (
        float(config.shard_cache_ram_gb)
        if config.shard_cache_ram_gb is not None
        else float(round(_choose_shard_cache_gb(), 2))
    )
    shard_store = ShardStore(
        cache_root,
        shard_cache_ram_gb,
        modalities=config.signal_spec.modalities,
        dataset_cache_roots=dataset_cache_roots,
    )
    has_val_datasets = any(
        session_split_summary[dataset]["val_examples"] > 0
        for dataset in pretrain_datasets
    )
    session_feature_stats: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    if config.use_normalization:
        resolved_stats_path = (
            Path(config.precomputed_session_stats_path)
            if config.precomputed_session_stats_path is not None
            else resolve_precomputed_session_stats_path(
                cache_root=drive_cache_root,
                signal_spec=config.signal_spec,
                dataset_plan=config.dataset_plan,
                boundary_key_mode=str(config.boundary_key_mode),
            )
        )
        session_feature_stats, _, stats_path = _load_precomputed_session_feature_stats(
            stats_path=resolved_stats_path,
            cache_root=drive_cache_root,
            signal_spec=config.signal_spec,
            dataset_plan=config.dataset_plan,
            boundary_key_mode=str(config.boundary_key_mode),
            dataset_cache_roots=(
                drive_dataset_cache_roots
            ),
        )
        print(f"loaded precomputed SSL session-level featurewise z-scoring stats: {stats_path}")

    return CacheContext(
        config=config,
        drive_cache_root=drive_cache_root,
        cache_root=cache_root,
        cache_copy_used=cache_copy_used,
        source_cache_signature=source_signature,
        drive_dataset_cache_roots=drive_dataset_cache_roots,
        dataset_cache_roots=dataset_cache_roots,
        available_datasets=available_datasets,
        pretrain_datasets=pretrain_datasets,
        rows_by_dataset=rows_by_dataset,
        split_rows_by_dataset=split_rows_by_dataset,
        session_split_summary=session_split_summary,
        shard_store=shard_store,
        has_val_datasets=has_val_datasets,
        session_feature_stats=session_feature_stats,
    )

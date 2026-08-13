"""Cache discovery, copying, shard access, and context preparation."""

from __future__ import annotations

import json
import math
import os
import random
import shutil
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from utah_ssl.experiment_contract import (
    DatasetPlan,
    SignalSpec,
)
from utah_ssl.cache_identity import (
    compute_dataset_cache_source_signature,
    list_directory_with_retries,
)
from utah_ssl.stats import (
    load_precomputed_session_feature_stats,
    resolve_precomputed_session_stats_path,
)

try:
    import psutil
except ImportError:  # pragma: no cover - optional dependency
    psutil = None


RUNTIME_SMOOTHING_MIGRATION_MESSAGE = (
    "Runtime Gaussian smoothing has been removed from CacheAccessConfig. "
    "Build or select a pre-smoothed cache root instead and keep "
    "gaussian_smoothing_sigma_bins=0.0 during training."
)

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
    sampling_plan_cache: dict[tuple[str, int, float], Any] = field(default_factory=dict)

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


def _format_bytes(num_bytes: int) -> str:
    value = float(num_bytes)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if value < 1024.0 or unit == "TB":
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{value:.1f} TB"


def _resolve_drive_dataset_cache_roots(
    primary_root: Path,
    overrides: Mapping[str, str | Path] | None,
) -> dict[str, Path]:
    resolved = {
        path.name: primary_root
        for path in list_directory_with_retries(primary_root)
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
    source_signature = compute_dataset_cache_source_signature(
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
                dataset_cache_roots=drive_dataset_cache_roots,
            )
        )
        session_feature_stats, _, stats_path = load_precomputed_session_feature_stats(
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

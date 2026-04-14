"""Cache access, shard loading, and segment sampling for contrastive SSL runs."""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import shutil
import time
from collections import Counter, OrderedDict, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

try:
    import psutil
except ImportError:  # pragma: no cover - optional dependency
    psutil = None


@dataclass
class CacheAccessConfig:
    mode: str = "copy_to_local"
    local_cache_base: str = "/content/utah_ssl_cache"
    force_recopy_local_cache: bool = False
    excluded_datasets: tuple[str, ...] = ("brain2text25",)
    seed: int = 7
    segment_bins: int = 64
    normalize_context_bins: int | None = None
    normalize_impl_version: str = "segment_prefix_v1"
    examples_per_shard: int = 8
    tx_dim: int = 256
    sbp_dim: int = 256
    feature_mode: str = "tx_only"
    boundary_key_mode: str = "session"
    shard_cache_ram_gb: float | None = None

    def __post_init__(self) -> None:
        if self.mode not in {"copy_to_local", "drive_direct"}:
            raise ValueError("mode must be either 'copy_to_local' or 'drive_direct'")
        if self.segment_bins <= 0:
            raise ValueError("segment_bins must be positive")
        if self.examples_per_shard <= 0:
            raise ValueError("examples_per_shard must be positive")
        if self.tx_dim <= 0 or self.sbp_dim <= 0:
            raise ValueError("tx_dim and sbp_dim must be positive")
        if self.feature_mode not in {"tx_only", "tx_sbp"}:
            raise ValueError("feature_mode must be one of {'tx_only', 'tx_sbp'}")
        if self.boundary_key_mode not in {"session", "subject_if_available"}:
            raise ValueError(
                "boundary_key_mode must be one of {'session', 'subject_if_available'}"
            )
        if self.normalize_impl_version not in {"segment_prefix_v1", "session_featurewise_v1"}:
            raise ValueError(
                "normalize_impl_version must be one of {'segment_prefix_v1', 'session_featurewise_v1'}"
            )
        if self.normalize_context_bins is None:
            self.normalize_context_bins = min(16, int(self.segment_bins))

    @property
    def full_dim(self) -> int:
        if self.feature_mode == "tx_only":
            return int(self.tx_dim)
        return int(self.tx_dim + self.sbp_dim)


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
    def boundary_key_mode(self) -> str:
        return str(self.config.boundary_key_mode)

    @property
    def normalize_context_bins(self) -> int:
        return int(self.config.normalize_context_bins or 1)

    @property
    def normalize_impl_version(self) -> str:
        return str(self.config.normalize_impl_version)


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


def _compute_cache_source_signature(src_root: Path) -> str:
    def list_dir_with_retries(path: Path, *, max_retries: int = 5) -> list[Path]:
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

    datasets = []
    for dataset_root in (path for path in list_dir_with_retries(src_root) if path.is_dir()):
        shard_root = dataset_root / "shards"
        shard_names: list[str] = []
        shard_scan_error: str | None = None
        if shard_root.exists():
            try:
                shard_names = [path.name for path in list_dir_with_retries(shard_root) if path.is_dir()]
            except OSError as exc:  # pragma: no cover - exercised in Colab when Drive stalls
                shard_scan_error = str(exc)
                print(
                    f"warning: failed to enumerate shards for signature under {shard_root}; "
                    f"falling back to metadata-only signature fields: {exc}"
                )
        datasets.append(
            {
                "dataset": dataset_root.name,
                "manifest": _path_signature(dataset_root / "manifest.jsonl"),
                "metadata": _path_signature(dataset_root / "metadata.json"),
                "shard_count": len(shard_names),
                "first_shard": shard_names[0] if shard_names else None,
                "last_shard": shard_names[-1] if shard_names else None,
                "shard_scan_error": shard_scan_error,
            }
        )
    payload = {
        "root": str(src_root),
        "datasets": datasets,
        "repack_summary": _path_signature(src_root / "repack_summary.json"),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


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
    def __init__(self, cache_root: Path, ram_cache_gb: float):
        self.cache_root = Path(cache_root)
        self.max_bytes = int(ram_cache_gb * (1024 ** 3))
        self._cache: OrderedDict[str, dict[str, np.ndarray | None | int]] = OrderedDict()
        self._cached_bytes = 0

    def clear(self) -> None:
        self._cache.clear()
        self._cached_bytes = 0

    def summary(self) -> dict[str, float]:
        return {
            "cached_shards": float(len(self._cache)),
            "cached_gb": self._cached_bytes / (1024 ** 3),
            "budget_gb": self.max_bytes / (1024 ** 3),
        }

    def _load_array(self, path: Path) -> np.ndarray | None:
        if not path.exists():
            return None
        return np.load(path, mmap_mode="r", allow_pickle=False)

    def _load_shard(self, shard_relpath: str) -> dict[str, np.ndarray | None | int]:
        shard_path = self.cache_root / shard_relpath
        shard = {
            "time_offsets": self._load_array(shard_path / "time_offsets.npy"),
            "tx": self._load_array(shard_path / "tx.npy"),
            "sbp": self._load_array(shard_path / "sbp.npy"),
        }
        time_offsets = shard["time_offsets"]
        if time_offsets is None:
            raise FileNotFoundError(f"Missing time_offsets.npy for shard {shard_path}")
        shard["bytes"] = int(
            time_offsets.nbytes
            + (0 if shard["tx"] is None else shard["tx"].nbytes)
            + (0 if shard["sbp"] is None else shard["sbp"].nbytes)
        )
        return shard

    def get(self, shard_relpath: str) -> dict[str, np.ndarray | None | int]:
        key = str(shard_relpath)
        cached = self._cache.get(key)
        if cached is not None:
            self._cache.move_to_end(key)
            return cached

        shard = self._load_shard(key)
        shard_bytes = int(shard["bytes"])
        if shard_bytes <= self.max_bytes:
            while self._cache and self._cached_bytes + shard_bytes > self.max_bytes:
                _, evicted = self._cache.popitem(last=False)
                self._cached_bytes -= int(evicted["bytes"])
            self._cache[key] = shard
            self._cached_bytes += shard_bytes
        return shard


def _normalize_segment(
    x_seq: torch.Tensor,
    feature_mask: torch.Tensor,
    context_bins: int,
    *,
    normalize_impl_version: str,
    session_feature_stats: dict[str, tuple[torch.Tensor, torch.Tensor]] | None = None,
    session_key: str | None = None,
    min_scale_std: float = 0.1,
    clip_value: float = 20.0,
) -> torch.Tensor:
    if normalize_impl_version == "segment_prefix_v1":
        return _normalize_segment_prefix(
            x_seq,
            feature_mask,
            context_bins=context_bins,
            min_scale_std=min_scale_std,
            clip_value=clip_value,
        )
    if normalize_impl_version == "session_featurewise_v1":
        return _normalize_segment_session_featurewise(
            x_seq,
            feature_mask,
            session_feature_stats=session_feature_stats,
            session_key=session_key,
            clip_value=clip_value,
        )
    raise ValueError(f"Unsupported normalize_impl_version: {normalize_impl_version}")


def _normalize_segment_prefix(
    x_seq: torch.Tensor,
    feature_mask: torch.Tensor,
    context_bins: int,
    *,
    min_scale_std: float = 0.1,
    clip_value: float = 20.0,
) -> torch.Tensor:
    x_norm = x_seq.clone()
    present_idx = torch.nonzero(feature_mask.bool(), as_tuple=False).squeeze(1)
    if present_idx.numel() == 0:
        return x_norm

    context_x = x_norm[:context_bins, present_idx]
    mean = context_x.mean(dim=0)
    centered = x_norm[:, present_idx] - mean
    std = context_x.std(dim=0, unbiased=False)
    scale_mask = std >= min_scale_std
    if scale_mask.any():
        centered[:, scale_mask] = centered[:, scale_mask] / std[scale_mask]
    x_norm[:, present_idx] = centered.clamp(min=-clip_value, max=clip_value)
    return x_norm


def _normalize_segment_session_featurewise(
    x_seq: torch.Tensor,
    feature_mask: torch.Tensor,
    *,
    session_feature_stats: dict[str, tuple[torch.Tensor, torch.Tensor]] | None,
    session_key: str | None,
    clip_value: float = 20.0,
) -> torch.Tensor:
    if session_feature_stats is None or session_key is None:
        raise ValueError("Session feature stats are required for session_featurewise_v1 normalization.")
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
    print("computing SSL session-level featurewise z-scoring stats...")
    session_rows: dict[str, list[ExampleRow]] = defaultdict(list)
    for dataset, rows in rows_by_dataset.items():
        for row in rows:
            session_rows[_session_stat_key(dataset, row.session_id)].append(row)

    full_dim = int(config.full_dim)
    session_stats: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    total_sessions = len(session_rows)
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

            tx = shard["tx"]
            if isinstance(tx, np.ndarray):
                tx_window = np.asarray(tx[start:stop], dtype=np.float64)
                tx_dim = min(tx_window.shape[1], config.tx_dim)
                sum_x[:tx_dim] += tx_window[:, :tx_dim].sum(axis=0)
                sum_x2[:tx_dim] += np.square(tx_window[:, :tx_dim]).sum(axis=0)
                count_x[:tx_dim] += tx_window.shape[0]

            sbp = shard["sbp"]
            if config.feature_mode == "tx_sbp" and isinstance(sbp, np.ndarray):
                sbp_window = np.asarray(sbp[start:stop], dtype=np.float64)
                sbp_dim = min(sbp_window.shape[1], config.sbp_dim)
                sbp_slice = slice(config.tx_dim, config.tx_dim + sbp_dim)
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
    normalize_impl_version: str = "session_featurewise_v1",
) -> dict[str, Any]:
    path = Path(stats_path)
    if not path.exists():
        raise FileNotFoundError(f"Precomputed session stats file does not exist: {path}")

    payload = torch.load(path, map_location="cpu")
    raw_stats = payload.get("session_feature_stats")
    if not isinstance(raw_stats, dict):
        raise KeyError("Precomputed session stats payload is missing 'session_feature_stats'.")

    session_feature_stats: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    expected_dim = int(cache_context.full_dim)
    for key, value in raw_stats.items():
        if not isinstance(value, (tuple, list)) or len(value) != 2:
            raise ValueError(
                f"Session stats entry for {key!r} must be a 2-item (mean, std) tuple/list."
            )
        mean, std = value
        mean_t = torch.as_tensor(mean).float().cpu()
        std_t = torch.as_tensor(std).float().cpu()
        if mean_t.numel() < expected_dim or std_t.numel() < expected_dim:
            raise ValueError(
                f"Session stats entry for {key!r} is too small for feature_mode={cache_context.feature_mode!r}: "
                f"expected at least {expected_dim} values, got mean={mean_t.numel()} std={std_t.numel()}."
            )
        if mean_t.numel() != expected_dim:
            mean_t = mean_t[:expected_dim].clone()
        if std_t.numel() != expected_dim:
            std_t = std_t[:expected_dim].clone()
        session_feature_stats[str(key)] = (mean_t, std_t)

    cache_context.session_feature_stats = dict(session_feature_stats)
    cache_context.config.normalize_impl_version = str(normalize_impl_version)

    metadata = dict(payload.get("metadata", {}))
    return {
        "stats_path": path,
        "metadata": metadata,
        "session_feature_stats": session_feature_stats,
        "session_count": int(len(session_feature_stats)),
        "normalize_impl_version": cache_context.normalize_impl_version,
    }


def sample_base_segment(
    cache_context: CacheContext,
    example: ExampleRow,
    segment_bins: int,
    py_rng: random.Random,
) -> dict[str, Any]:
    session_key = _session_stat_key(example.dataset, example.session_id)
    boundary_key = resolve_boundary_key(
        dataset=example.dataset,
        session_id=example.session_id,
        subject_id=example.subject_id,
        boundary_key_mode=cache_context.boundary_key_mode,
    )
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
    if isinstance(tx, np.ndarray):
        tx_window = np.asarray(tx[src_start:src_stop], dtype=np.float32)
        tx_dim = min(tx_window.shape[1], cache_context.tx_dim)
        x_seq[:, :tx_dim] = tx_window[:, :tx_dim]
        feature_mask[:tx_dim] = 1.0

    sbp = shard["sbp"]
    if cache_context.feature_mode == "tx_sbp" and isinstance(sbp, np.ndarray):
        sbp_window = np.asarray(sbp[src_start:src_stop], dtype=np.float32)
        sbp_dim = min(sbp_window.shape[1], cache_context.sbp_dim)
        x_seq[:, cache_context.tx_dim : cache_context.tx_dim + sbp_dim] = sbp_window[:, :sbp_dim]
        feature_mask[cache_context.tx_dim : cache_context.tx_dim + sbp_dim] = 1.0

    x_seq_t = torch.from_numpy(x_seq)
    feature_mask_t = torch.from_numpy(feature_mask)
    x_norm = _normalize_segment(
        x_seq_t,
        feature_mask_t,
        context_bins=cache_context.normalize_context_bins,
        normalize_impl_version=cache_context.normalize_impl_version,
        session_feature_stats=cache_context.session_feature_stats,
        session_key=session_key,
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


def prepare_cache_context(
    *,
    cache_candidates: Sequence[Path],
    config: CacheAccessConfig,
) -> CacheContext:
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    candidate_paths = [Path(path) for path in cache_candidates]
    drive_cache_root = next((path for path in candidate_paths if path.exists()), None)
    if drive_cache_root is None:
        raise FileNotFoundError(
            "No cache root found. Candidates checked: " + ", ".join(str(path) for path in candidate_paths)
        )

    local_cache_root = Path(config.local_cache_base) / drive_cache_root.name
    local_cache_status_path = local_cache_root.parent / f"{drive_cache_root.name}_copy_status.json"
    source_signature = _compute_cache_source_signature(drive_cache_root)

    if config.force_recopy_local_cache and config.mode == "copy_to_local" and local_cache_root.exists():
        print("removing existing local cache:", local_cache_root)
        shutil.rmtree(local_cache_root)
    if config.force_recopy_local_cache and config.mode == "copy_to_local" and local_cache_status_path.exists():
        local_cache_status_path.unlink()

    copy_status = _load_copy_status(local_cache_status_path)
    if config.mode == "copy_to_local":
        if (not local_cache_root.exists()) or (
            not _copy_complete_for_current_source(
                status=copy_status,
                drive_cache_root=drive_cache_root,
                local_cache_root=local_cache_root,
                source_signature=source_signature,
            )
        ):
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
            file_count, total_bytes = copy_tree_with_progress(drive_cache_root, local_cache_root)
            _write_copy_status(
                local_cache_status_path,
                drive_cache_root=drive_cache_root,
                local_cache_root=local_cache_root,
                source_signature=source_signature,
                file_count=file_count,
                total_bytes=total_bytes,
            )
            print(f"copy complete in {time.time() - t0:.1f}s")
        else:
            print("using existing local cache:", local_cache_root)
            print("source signature:", source_signature[:12])
        cache_root = local_cache_root
        cache_copy_used = True
    else:
        cache_root = drive_cache_root
        cache_copy_used = False
        print("using Drive-backed cache directly; skipping local copy")
        print("source:", drive_cache_root)
        print("source signature:", source_signature[:12])

    os.environ["SSL_AUTORESEARCH_CACHE_ROOT"] = str(cache_root)

    dataset_roots = sorted(path for path in cache_root.iterdir() if path.is_dir())
    available_datasets = [path.name for path in dataset_roots]
    pretrain_datasets = [name for name in available_datasets if name not in set(config.excluded_datasets)]

    rows_by_dataset: dict[str, list[ExampleRow]] = {}
    split_rows_by_dataset: dict[str, dict[str, list[ExampleRow]]] = {"train": {}, "val": {}}
    session_split_summary: dict[str, dict[str, Any]] = {}

    for dataset in pretrain_datasets:
        ds_root = cache_root / dataset
        manifest_path = ds_root / "manifest.jsonl"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest missing for dataset {dataset}: {manifest_path}")

        rows: list[ExampleRow] = []
        with manifest_path.open() as handle:
            for line in handle:
                payload = json.loads(line)
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
    shard_store = ShardStore(cache_root, shard_cache_ram_gb)
    has_val_datasets = any(
        session_split_summary[dataset]["val_examples"] > 0
        for dataset in pretrain_datasets
    )
    session_feature_stats: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    if config.normalize_impl_version == "session_featurewise_v1":
        session_feature_stats = _compute_session_feature_stats(
            shard_store=shard_store,
            rows_by_dataset=rows_by_dataset,
            config=config,
        )

    return CacheContext(
        config=config,
        drive_cache_root=drive_cache_root,
        cache_root=cache_root,
        cache_copy_used=cache_copy_used,
        source_cache_signature=source_signature,
        available_datasets=available_datasets,
        pretrain_datasets=pretrain_datasets,
        rows_by_dataset=rows_by_dataset,
        split_rows_by_dataset=split_rows_by_dataset,
        session_split_summary=session_split_summary,
        shard_store=shard_store,
        has_val_datasets=has_val_datasets,
        session_feature_stats=session_feature_stats,
    )

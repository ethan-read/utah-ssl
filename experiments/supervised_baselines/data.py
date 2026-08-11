"""Native cache data helpers for Willett reconstruction."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from utah_ssl.cache import resolve_boundary_key
from utah_ssl.ctc import (
    CanonicalSequenceDataset,
    LengthAwareBatchSampler,
    build_competition_split_problem,
    build_source_split_problem,
    canonical_rows_padded_time_percentile,
    collate_sequence_batch,
)
from utah_ssl.datasets import apply_feature_stats
from utah_ssl.experiment_contract import SignalSpec
from utah_ssl.stats import compute_feature_stats


@dataclass(frozen=True)
class WillettInputTransformConfig:
    input_smoothing_sigma_bins: float = 2.0
    input_smoothing_kernel_size: int = 100
    input_smoothing_threshold: float = 0.01
    white_noise_sd: float = 1.0
    constant_offset_sd: float = 0.2


class FuturePredictionExportAccessor:
    def __init__(self, export_root: str | Path) -> None:
        self.export_root = Path(export_root)
        manifest_path = self.export_root / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Future-prediction export manifest not found: {manifest_path}")
        payload = json.loads(manifest_path.read_text())
        self.temporal_bin_stride = int(payload.get("temporal_bin_stride", 1))
        self.future_bins = int(payload.get("future_bins", 1))
        self.tx_dim = int(payload.get("tx_dim", 0))
        self._rows_by_key: dict[tuple[str, int], dict[str, Any]] = {
            (str(row["shard_relpath"]), int(row["example_index"])): dict(row)
            for row in payload.get("rows", [])
        }
        self._shard_cache: dict[str, dict[int, dict[str, Any]]] = {}

    def _prediction_shard_path(self, *, dataset: str, shard_relpath: str) -> Path:
        return self.export_root / "predictions" / str(dataset) / Path(shard_relpath).with_suffix(".future_pred.pt")

    def _load_shard_rows(self, *, dataset: str, shard_relpath: str) -> dict[int, dict[str, Any]]:
        cache_key = f"{dataset}:{shard_relpath}"
        cached = self._shard_cache.get(cache_key)
        if cached is not None:
            return cached
        shard_path = self._prediction_shard_path(dataset=str(dataset), shard_relpath=str(shard_relpath))
        if not shard_path.exists():
            raise FileNotFoundError(f"Future-prediction shard export not found: {shard_path}")
        payload = torch.load(shard_path, map_location="cpu", weights_only=False)
        rows_by_example_index = {
            int(row_payload["example_index"]): row_payload
            for row_payload in payload.get("rows", [])
        }
        self._shard_cache[cache_key] = rows_by_example_index
        return rows_by_example_index

    def duplicated_predicted_tx_for_row(self, row: Any) -> np.ndarray:
        manifest_row = self._rows_by_key.get((str(row.shard_relpath), int(row.example_index)))
        if manifest_row is None:
            raise KeyError(
                "Future-prediction export does not contain row "
                f"{row.shard_relpath}:{row.example_index} ({getattr(row, 'example_id', 'unknown')})."
            )
        shard_rows = self._load_shard_rows(
            dataset=str(manifest_row["dataset"]),
            shard_relpath=str(row.shard_relpath),
        )
        payload = shard_rows.get(int(row.example_index))
        if payload is None:
            raise KeyError(f"Missing exported prediction payload for {row.shard_relpath}:{row.example_index}.")
        forecast_raw = payload["forecast_raw"]
        tensor = (
            forecast_raw.detach().cpu().to(dtype=torch.float32)
            if isinstance(forecast_raw, torch.Tensor)
            else torch.as_tensor(forecast_raw, dtype=torch.float32)
        )
        if tensor.ndim == 3:
            tensor = tensor[:, 0, :]
        predicted_tx = tensor[:, : int(self.tx_dim)].numpy()
        return np.repeat(np.asarray(predicted_tx, dtype=np.float32), max(1, int(self.temporal_bin_stride)), axis=0)


def normalization_key_for_row(row: Any) -> str:
    block_num = getattr(row, "block_num", None)
    if block_num is not None:
        return f"{row.session_id}::block:{int(block_num)}"
    normalization_group = getattr(row, "normalization_group", None)
    if normalization_group is not None:
        return str(normalization_group)
    return str(row.session_id)


def _stats_group_key(row: Any, *, mode: str) -> str:
    if str(mode) == "block":
        return normalization_key_for_row(row)
    if str(mode) == "per_session":
        return str(row.session_id)
    raise ValueError(f"Unsupported grouped stats mode: {mode!r}")


def normalization_stats_missing_rows(
    stats: dict[str, tuple[np.ndarray, np.ndarray]] | tuple[np.ndarray, np.ndarray] | None,
    rows: tuple[Any, ...] | list[Any],
) -> list[str]:
    if stats is None or not isinstance(stats, dict):
        return []
    missing: list[str] = []
    for row in rows:
        candidate_keys: list[str] = []
        block_num = getattr(row, "block_num", None)
        if block_num is not None:
            candidate_keys.append(f"{row.session_id}::block:{int(block_num)}")
        normalization_group = getattr(row, "normalization_group", None)
        if normalization_group is not None:
            candidate_keys.append(str(normalization_group))
        candidate_keys.append(str(row.session_id))
        if any(candidate_key in stats for candidate_key in candidate_keys):
            continue
        missing.append(str(getattr(row, "example_id", row)))
    return missing


def compute_predicted_tx_normalization_stats(
    rows: tuple[Any, ...] | list[Any],
    *,
    export_accessor: FuturePredictionExportAccessor,
    mode: str,
) -> dict[str, tuple[np.ndarray, np.ndarray]] | tuple[np.ndarray, np.ndarray] | None:
    resolved_mode = str(mode)
    if resolved_mode == "none":
        return None
    if resolved_mode == "global":
        total_count = 0
        sum_x = None
        sum_x2 = None
        for row in rows:
            x64 = export_accessor.duplicated_predicted_tx_for_row(row).astype(np.float64, copy=False)
            if x64.shape[0] <= 0:
                continue
            if sum_x is None:
                sum_x = x64.sum(axis=0)
                sum_x2 = np.square(x64).sum(axis=0)
            else:
                sum_x += x64.sum(axis=0)
                sum_x2 += np.square(x64).sum(axis=0)
            total_count += int(x64.shape[0])
        if sum_x is None or sum_x2 is None or total_count <= 0:
            raise ValueError("Cannot compute predicted-tx global stats on an empty record set.")
        mean = sum_x / total_count
        var = np.maximum(sum_x2 / total_count - np.square(mean), 1e-6)
        return mean.astype(np.float32), np.sqrt(var).astype(np.float32)
    if resolved_mode not in {"block", "per_session"}:
        raise ValueError("mode must be one of {'block', 'global', 'per_session', 'none'}")
    grouped: dict[str, list[Any]] = {}
    for row in rows:
        grouped.setdefault(_stats_group_key(row, mode=resolved_mode), []).append(row)
    stats: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for key, group_rows in grouped.items():
        total_count = 0
        sum_x = None
        sum_x2 = None
        for row in group_rows:
            x64 = export_accessor.duplicated_predicted_tx_for_row(row).astype(np.float64, copy=False)
            if x64.shape[0] <= 0:
                continue
            if sum_x is None:
                sum_x = x64.sum(axis=0)
                sum_x2 = np.square(x64).sum(axis=0)
            else:
                sum_x += x64.sum(axis=0)
                sum_x2 += np.square(x64).sum(axis=0)
            total_count += int(x64.shape[0])
        if sum_x is None or sum_x2 is None or total_count <= 0:
            raise ValueError(f"Cannot compute predicted-tx stats for empty group {key!r}.")
        mean = sum_x / total_count
        var = np.maximum(sum_x2 / total_count - np.square(mean), 1e-6)
        stats[str(key)] = (
            mean.astype(np.float32, copy=False),
            np.sqrt(var).astype(np.float32, copy=False),
        )
    return stats


def adapter_keys_from_rows(
    rows: tuple[Any, ...] | list[Any],
    *,
    dataset: str,
    boundary_key_mode: str,
) -> tuple[str, ...]:
    keys: list[str] = []
    seen: set[str] = set()
    for row in rows:
        key = resolve_boundary_key(
            dataset=str(dataset),
            session_id=str(row.session_id),
            subject_id=None if getattr(row, "subject_id", None) is None else str(row.subject_id),
            boundary_key_mode=str(boundary_key_mode),
        )
        if key in seen:
            continue
        seen.add(key)
        keys.append(key)
    return tuple(keys)


def group_rows_by_adapter_key(
    rows: tuple[Any, ...] | list[Any],
    *,
    dataset: str,
    boundary_key_mode: str,
) -> dict[str, tuple[Any, ...]]:
    grouped: dict[str, list[Any]] = {}
    for row in rows:
        key = resolve_boundary_key(
            dataset=str(dataset),
            session_id=str(row.session_id),
            subject_id=None if getattr(row, "subject_id", None) is None else str(row.subject_id),
            boundary_key_mode=str(boundary_key_mode),
        )
        grouped.setdefault(key, []).append(row)
    return {key: tuple(group_rows) for key, group_rows in grouped.items()}


def _willett_gaussian_kernel_1d(
    *,
    sigma_bins: float,
    kernel_size: int,
    threshold: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    sigma = float(sigma_bins)
    if sigma <= 0.0:
        return torch.ones((1,), device=device, dtype=dtype)
    positions = torch.arange(int(kernel_size), device=device, dtype=dtype) - float(int(kernel_size) // 2)
    kernel = torch.exp(-0.5 * (positions / sigma).pow(2))
    kernel = kernel / kernel.sum().clamp_min(1e-8)
    keep = kernel > float(threshold)
    if not bool(keep.any().item()):
        keep[int(kernel.numel() // 2)] = True
    kept_positions = torch.nonzero(keep, as_tuple=False).squeeze(1)
    start = int(kept_positions.min().item())
    stop = int(kept_positions.max().item()) + 1
    kernel = kernel[start:stop]
    if int(kernel.numel()) % 2 == 0:
        kernel = torch.cat([kernel, kernel.new_zeros((1,))], dim=0)
    return kernel / kernel.sum().clamp_min(1e-8)


def _sequence_mask_from_lengths(lengths: torch.Tensor, max_time: int) -> torch.Tensor:
    return torch.arange(max_time, device=lengths.device).unsqueeze(0) < lengths.unsqueeze(1)


def smooth_batch_like_willett(
    x: torch.Tensor,
    input_lengths: torch.Tensor,
    *,
    sigma_bins: float,
    kernel_size: int,
    threshold: float,
) -> torch.Tensor:
    if float(sigma_bins) <= 0.0 or int(x.shape[1]) <= 1:
        return x
    kernel = _willett_gaussian_kernel_1d(
        sigma_bins=float(sigma_bins),
        kernel_size=int(kernel_size),
        threshold=float(threshold),
        device=x.device,
        dtype=x.dtype,
    )
    channels = int(x.shape[-1])
    weight = kernel.view(1, 1, -1).expand(channels, 1, -1)
    smoothed = torch.nn.functional.conv1d(
        x.transpose(1, 2),
        weight,
        padding=int(kernel.numel() // 2),
        groups=channels,
    ).transpose(1, 2)
    valid = _sequence_mask_from_lengths(input_lengths.to(x.device), int(x.shape[1]))
    return smoothed * valid.unsqueeze(-1).to(smoothed.dtype)


def prepare_willett_inputs(
    x: torch.Tensor,
    input_lengths: torch.Tensor,
    *,
    config: WillettInputTransformConfig,
    is_training: bool,
) -> torch.Tensor:
    transformed = x
    if is_training and float(config.white_noise_sd) > 0.0:
        transformed = transformed + torch.randn_like(transformed) * float(config.white_noise_sd)
    if is_training and float(config.constant_offset_sd) > 0.0:
        transformed = transformed + torch.randn(
            (int(transformed.shape[0]), 1, int(transformed.shape[2])),
            device=transformed.device,
            dtype=transformed.dtype,
        ) * float(config.constant_offset_sd)
    return smooth_batch_like_willett(
        transformed,
        input_lengths,
        sigma_bins=float(config.input_smoothing_sigma_bins),
        kernel_size=int(config.input_smoothing_kernel_size),
        threshold=float(config.input_smoothing_threshold),
    )


def build_willett_problem(
    *,
    cache_root: str | Path,
    dataset: str,
    feature_mode: str,
    boundary_key_mode: str,
    split_policy: str = "competition_train_test",
    cv_num_folds: int = 5,
    cv_fold_index: int = 0,
) -> dict[str, Any]:
    metadata_path = Path(cache_root) / str(dataset) / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    feature_layout = dict(metadata.get("feature_layout") or {})
    tx_dim = int(metadata.get("n_tx_features", feature_layout.get("n_tx_features", 0)))
    sbp_dim = int(metadata.get("n_sbp_features", feature_layout.get("n_sbp_features", 0)))
    if tx_dim <= 0 or sbp_dim <= 0:
        manifest_path = Path(cache_root) / str(dataset) / "manifest.jsonl"
        first_manifest_row = next(
            json.loads(line)
            for line in manifest_path.read_text().splitlines()
            if line.strip()
        )
        tx_dim = max(tx_dim, int(first_manifest_row.get("n_tx_features", 0)))
        sbp_dim = max(sbp_dim, int(first_manifest_row.get("n_sbp_features", 0)))
    signal_spec = SignalSpec.from_mode(
        str(feature_mode),
        tx_dim=tx_dim,
        sbp_dim=sbp_dim,
    )
    if str(split_policy) == "competition_train_test":
        problem = build_competition_split_problem(
            cache_root=Path(cache_root),
            dataset=str(dataset),
            signal_spec=signal_spec,
            boundary_key_mode=str(boundary_key_mode),
        )
        return problem
    if str(split_policy) == "source_train_val":
        problem = build_source_split_problem(
            cache_root=Path(cache_root),
            dataset=str(dataset),
            signal_spec=signal_spec,
            boundary_key_mode=str(boundary_key_mode),
            train_split_name="train",
            val_split_name="val",
        )
        updated = dict(problem)
        updated["split_policy"] = "source_train_val"
        return updated
    if str(split_policy) != "competition_train_kfold":
        raise ValueError(
            "split_policy must be one of "
            "{'competition_train_test', 'competition_train_kfold', 'source_train_val'}"
        )
    problem = build_competition_split_problem(
        cache_root=Path(cache_root),
        dataset=str(dataset),
        signal_spec=signal_spec,
        boundary_key_mode=str(boundary_key_mode),
    )

    num_folds = int(cv_num_folds)
    fold_index = int(cv_fold_index)
    if num_folds < 2:
        raise ValueError("cv_num_folds must be at least 2")
    if fold_index < 0 or fold_index >= num_folds:
        raise ValueError("cv_fold_index must satisfy 0 <= cv_fold_index < cv_num_folds")

    rows_by_session: dict[str, list[Any]] = {}
    for row in problem["train_rows"]:
        rows_by_session.setdefault(str(row.session_id), []).append(row)

    train_candidates: list[Any] = []
    val_candidates: list[Any] = []
    for session_id in sorted(rows_by_session):
        session_rows = sorted(
            rows_by_session[session_id],
            key=lambda row: (
                int(getattr(row, "block_num", -1) or -1),
                str(getattr(row, "example_id", "")),
                int(getattr(row, "example_index", -1)),
            ),
        )
        if len(session_rows) < num_folds:
            raise ValueError(
                f"Session {session_id!r} has only {len(session_rows)} train rows, "
                f"which is fewer than cv_num_folds={num_folds}."
            )
        for row_index, row in enumerate(session_rows):
            if row_index % num_folds == fold_index:
                val_candidates.append(row)
            else:
                train_candidates.append(row)

    train_rows, train_examples_by_session, train_session_ids = _group_willett_rows_by_session(train_candidates)
    val_rows, val_examples_by_session, val_session_ids = _group_willett_rows_by_session(val_candidates)
    if not train_rows or not val_rows:
        raise ValueError(
            "Cross-validation split produced an empty train or validation set. "
            f"cv_num_folds={num_folds}, cv_fold_index={fold_index}"
        )

    updated = dict(problem)
    updated.update(
        {
            "split_policy": "competition_train_kfold",
            "train_split_name": f"competition_train_cv{num_folds}_fold{fold_index}_train",
            "val_split_name": f"competition_train_cv{num_folds}_fold{fold_index}_val",
            "cv_num_folds": num_folds,
            "cv_fold_index": fold_index,
            "train_rows": train_rows,
            "val_rows": val_rows,
            "train_examples_by_session": train_examples_by_session,
            "val_examples_by_session": val_examples_by_session,
            "train_session_ids": train_session_ids,
            "val_session_ids": val_session_ids,
        }
    )
    return updated


def _group_willett_rows_by_session(
    rows: list[Any],
) -> tuple[tuple[Any, ...], dict[str, int], tuple[str, ...]]:
    grouped: dict[str, list[Any]] = {}
    for row in rows:
        grouped.setdefault(str(row.session_id), []).append(row)
    session_ids = tuple(sorted(grouped))
    flattened = tuple(row for session_id in session_ids for row in grouped[session_id])
    counts = {session_id: len(grouped[session_id]) for session_id in session_ids}
    return flattened, counts, session_ids


def compute_willett_normalization_stats(
    rows: tuple[Any, ...] | list[Any],
    *,
    cache_root: Path,
    feature_mode: str,
    mode: str,
) -> dict[str, tuple[np.ndarray, np.ndarray]] | tuple[np.ndarray, np.ndarray] | None:
    resolved_mode = str(mode)
    if not rows:
        raise ValueError("Cannot compute normalization stats for an empty row set")
    first_row = rows[0]
    signal_spec = SignalSpec.from_mode(
        str(feature_mode),
        tx_dim=int(first_row.n_tx_features),
        sbp_dim=int(first_row.n_sbp_features),
    )
    if resolved_mode == "none":
        return None
    if resolved_mode == "global":
        return compute_feature_stats(
            rows,
            cache_root=cache_root,
            mode="global",
            signal_spec=signal_spec,
        )
    if resolved_mode == "per_session":
        return compute_feature_stats(
            rows,
            cache_root=cache_root,
            mode="per_session",
            signal_spec=signal_spec,
        )
    if resolved_mode != "block":
        raise ValueError("mode must be one of {'block', 'global', 'per_session', 'none'}")

    accessor = CanonicalSequenceDataset(
        rows,
        cache_root=cache_root,
        signal_spec=signal_spec,
        stats=None,
    )._accessor
    try:
        grouped: dict[str, list[Any]] = {}
        for row in rows:
            grouped.setdefault(normalization_key_for_row(row), []).append(row)
        stats: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for key, group_rows in grouped.items():
            total_count = 0
            sum_x = None
            sum_x2 = None
            for row in group_rows:
                x = accessor.load_features(row, signal_spec=signal_spec)
                x64 = x.astype(np.float64, copy=False)
                if sum_x is None:
                    sum_x = x64.sum(axis=0)
                    sum_x2 = np.square(x64).sum(axis=0)
                else:
                    sum_x += x64.sum(axis=0)
                    sum_x2 += np.square(x64).sum(axis=0)
                total_count += int(x.shape[0])
            if sum_x is None or sum_x2 is None or total_count <= 0:
                raise ValueError(f"Cannot compute block stats for empty group {key!r}.")
            mean = sum_x / total_count
            var = np.maximum(sum_x2 / total_count - np.square(mean), 1e-6)
            stats[str(key)] = (
                mean.astype(np.float32, copy=False),
                np.sqrt(var).astype(np.float32, copy=False),
            )
        return stats
    finally:
        accessor.close()


class ConcatenatedPredictedTxSequenceDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        rows: tuple[Any, ...] | list[Any],
        *,
        cache_root: Path,
        raw_stats: dict[str, tuple[np.ndarray, np.ndarray]] | tuple[np.ndarray, np.ndarray] | None,
        predicted_stats: dict[str, tuple[np.ndarray, np.ndarray]] | tuple[np.ndarray, np.ndarray] | None,
        export_accessor: FuturePredictionExportAccessor,
        boundary_key_mode: str = "session",
        dataset: str = "brain2text24",
    ) -> None:
        self.rows = list(rows)
        self.raw_stats = raw_stats
        self.predicted_stats = predicted_stats
        self.boundary_key_mode = str(boundary_key_mode)
        self.dataset = str(dataset)
        self.export_accessor = export_accessor
        signal_spec = SignalSpec.tx_only(tx_dim=int(self.rows[0].n_tx_features))
        self._base = CanonicalSequenceDataset(
            self.rows,
            cache_root=cache_root,
            signal_spec=signal_spec,
            stats=None,
            boundary_key_mode=self.boundary_key_mode,
            dataset=self.dataset,
        )
        self._accessor = self._base._accessor
        self.signal_spec = signal_spec

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.rows[idx]
        raw_tx = np.asarray(
            self._accessor.load_features(row, signal_spec=self.signal_spec),
            dtype=np.float32,
        )
        predicted_tx = np.asarray(self.export_accessor.duplicated_predicted_tx_for_row(row), dtype=np.float32)
        usable = min(int(raw_tx.shape[0]), int(predicted_tx.shape[0]))
        if usable <= 0:
            raise ValueError(f"Concatenated predicted-tx example has no usable frames: {row.example_id}")
        raw_tx = np.asarray(raw_tx[:usable], dtype=np.float32)
        predicted_tx = np.asarray(predicted_tx[:usable], dtype=np.float32)
        if self.raw_stats is not None:
            raw_tx = apply_feature_stats(raw_tx, row=row, stats=self.raw_stats)
        if self.predicted_stats is not None:
            predicted_tx = apply_feature_stats(predicted_tx, row=row, stats=self.predicted_stats)
        x = np.concatenate([raw_tx, predicted_tx], axis=1).astype(np.float32, copy=False)
        labels = self._accessor.load_labels(row)
        labels = np.zeros((0,), dtype=np.int64) if labels is None else np.array(labels, dtype=np.int64, copy=True)
        return {
            "x": torch.from_numpy(x),
            "input_length": int(x.shape[0]),
            "labels": torch.from_numpy(labels),
            "label_length": int(labels.shape[0]),
            "session_id": row.session_id,
            "boundary_key": resolve_boundary_key(
                dataset=self.dataset,
                session_id=row.session_id,
                subject_id=None if getattr(row, "subject_id", None) is None else str(row.subject_id),
                boundary_key_mode=self.boundary_key_mode,
            ),
            "example_id": row.example_id,
        }

def make_length_aware_batch_sampler(
    rows: tuple[Any, ...] | list[Any],
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> LengthAwareBatchSampler:
    p95_train_input_length = canonical_rows_padded_time_percentile(rows, percentile=95.0)
    max_examples_per_microbatch = int(batch_size)
    max_padded_time_per_microbatch = int(max_examples_per_microbatch * p95_train_input_length)
    return LengthAwareBatchSampler(
        rows,
        max_examples_per_microbatch=max_examples_per_microbatch,
        max_padded_time_per_microbatch=max_padded_time_per_microbatch,
        shuffle=bool(shuffle),
        seed=int(seed),
    )


def loader_kwargs(device: torch.device) -> dict[str, Any]:
    return {
        "num_workers": 0,
        "pin_memory": device.type == "cuda",
        "collate_fn": collate_sequence_batch,
    }


__all__ = [
    "CanonicalSequenceDataset",
    "ConcatenatedPredictedTxSequenceDataset",
    "FuturePredictionExportAccessor",
    "WillettInputTransformConfig",
    "adapter_keys_from_rows",
    "build_willett_problem",
    "compute_predicted_tx_normalization_stats",
    "compute_willett_normalization_stats",
    "collate_sequence_batch",
    "compute_feature_stats",
    "group_rows_by_adapter_key",
    "loader_kwargs",
    "make_length_aware_batch_sampler",
    "normalization_key_for_row",
    "normalization_stats_missing_rows",
    "prepare_willett_inputs",
    "smooth_batch_like_willett",
]

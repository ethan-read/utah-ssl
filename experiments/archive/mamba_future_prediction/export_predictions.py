"""Export future-prediction forecasts over full cache examples."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

try:
    from utah_ssl.cache import CacheContext, resolve_boundary_key
except ModuleNotFoundError:  # pragma: no cover - repo-root unittest fallback
    from utah_ssl.cache import CacheContext, resolve_boundary_key

from .config import FuturePredictionSSLConfig
from .model import make_future_prediction_model
from .objectives import aggregate_time_bins, build_future_prediction_targets
from .training import _load_checkpoint_payload, _prepare_future_cache_context


def _load_full_example_arrays(
    cache_context: CacheContext,
    row: Any,
) -> tuple[torch.Tensor, torch.Tensor, str]:
    boundary_key = resolve_boundary_key(
        dataset=row.dataset,
        session_id=row.session_id,
        subject_id=row.subject_id,
        boundary_key_mode=cache_context.boundary_key_mode,
    )
    shard = cache_context.shard_store.get(row.shard_relpath)
    time_offsets = shard["time_offsets"]
    assert isinstance(time_offsets, np.ndarray)
    start = int(time_offsets[row.example_index])
    stop = int(time_offsets[row.example_index + 1])
    length = int(stop - start)

    x_raw = np.zeros((length, cache_context.full_dim), dtype=np.float32)
    present = np.zeros((cache_context.full_dim,), dtype=bool)

    tx = shard["tx"]
    if isinstance(tx, np.ndarray):
        tx_window = np.asarray(tx[start:stop], dtype=np.float32)
        tx_dim = min(int(tx_window.shape[1]), int(cache_context.tx_dim))
        x_raw[:, :tx_dim] = tx_window[:, :tx_dim]
        present[:tx_dim] = True

    sbp = shard["sbp"]
    if cache_context.feature_mode == "tx_sbp" and isinstance(sbp, np.ndarray):
        sbp_window = np.asarray(sbp[start:stop], dtype=np.float32)
        sbp_dim = min(int(sbp_window.shape[1]), int(cache_context.sbp_dim))
        lo = int(cache_context.tx_dim)
        hi = lo + sbp_dim
        x_raw[:, lo:hi] = sbp_window[:, :sbp_dim]
        present[lo:hi] = True

    x_raw_t = torch.from_numpy(x_raw)
    x_norm_t = x_raw_t.clone()
    if bool(cache_context.use_normalization):
        mean_t, std_t = cache_context.session_feature_stats[boundary_key]
        mean_t = mean_t.to(dtype=x_norm_t.dtype)
        std_t = std_t.to(dtype=x_norm_t.dtype).clamp_min(1e-6)
        present_idx = torch.from_numpy(np.flatnonzero(present)).long()
        if int(present_idx.numel()) > 0:
            centered = x_norm_t[:, present_idx] - mean_t[present_idx]
            x_norm_t[:, present_idx] = centered / std_t[present_idx]
    return x_raw_t, x_norm_t, boundary_key


def _shard_output_path(output_dir: Path, dataset: str, shard_relpath: str) -> Path:
    return output_dir / "predictions" / dataset / Path(shard_relpath).with_suffix(".future_pred.pt")


def _manifest_rows_from_saved_shard(shard_payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = shard_payload.get("rows", [])
    manifest_rows: list[dict[str, Any]] = []
    for row in rows:
        manifest_rows.append(
            {
                "dataset": str(row["dataset"]),
                "session_id": str(row["session_id"]),
                "subject_id": None if row.get("subject_id") is None else str(row["subject_id"]),
                "boundary_key": str(row["boundary_key"]),
                "shard_relpath": str(row["shard_relpath"]),
                "example_index": int(row["example_index"]),
                "n_time_bins_raw": int(row["n_time_bins_raw"]),
                "n_time_bins_agg": int(row["n_time_bins_agg"]),
            }
        )
    return manifest_rows


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    tmp_path = path.with_name(f"{path.name}.tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    tmp_path.replace(path)


def _write_torch_atomic(path: Path, payload: dict[str, Any]) -> None:
    tmp_path = path.with_name(f"{path.name}.tmp")
    torch.save(payload, tmp_path)
    tmp_path.replace(path)


def export_future_prediction_bins(
    *,
    checkpoint_path: str | Path,
    output_dir: str | Path,
    cache_root: str | Path,
    datasets: tuple[str, ...] = ("brain2text24",),
    resume: bool = True,
    overwrite_existing: bool = False,
    on_shard_written: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    checkpoint_path = Path(checkpoint_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = _load_checkpoint_payload(checkpoint_path)
    config = FuturePredictionSSLConfig.from_dict(dict(payload["config"]))
    config.cache_root = str(cache_root)
    config.cache_mode = "drive_direct"
    config.local_cache_base = "/tmp/utah_ssl_cache"
    config.resume = False
    config.resume_checkpoint_path = None
    config.pretrain_datasets = tuple(str(item) for item in datasets)
    cache_context = _prepare_future_cache_context(config)

    model = make_future_prediction_model(config, input_dim=int(cache_context.full_dim)).to(torch.device("cpu"))
    model.load_state_dict(payload["model_state"])
    model.eval()

    rows = []
    for dataset in config.pretrain_datasets:
        rows.extend(cache_context.rows_by_dataset[str(dataset)])
    rows = sorted(
        rows,
        key=lambda row: (
            str(row.dataset),
            str(row.shard_relpath),
            int(row.example_index),
        ),
    )

    rows_by_shard: dict[tuple[str, str], list[Any]] = defaultdict(list)
    for row in rows:
        rows_by_shard[(str(row.dataset), str(row.shard_relpath))].append(row)

    manifest_rows: list[dict[str, Any]] = []
    completed_shards: list[str] = []
    shard_export_count = 0
    example_count = 0
    total_shards = len(rows_by_shard)
    progress_path = output_dir / "progress.json"

    with torch.no_grad():
        for (dataset, shard_relpath), shard_rows in rows_by_shard.items():
            shard_output_path = _shard_output_path(output_dir, dataset, shard_relpath)
            if resume and not overwrite_existing and shard_output_path.exists():
                saved_payload = torch.load(shard_output_path, map_location="cpu", weights_only=False)
                saved_manifest_rows = _manifest_rows_from_saved_shard(saved_payload)
                manifest_rows.extend(saved_manifest_rows)
                example_count += len(saved_manifest_rows)
                shard_export_count += 1
                completed_shards.append(f"{dataset}:{shard_relpath}")
                _write_json_atomic(
                    progress_path,
                    {
                        "export_kind": "future_prediction_bins_progress",
                        "checkpoint_path": str(checkpoint_path),
                        "run_name": str(config.run_name),
                        "completed_shards": completed_shards,
                        "completed_shard_count": int(shard_export_count),
                        "total_shards": int(total_shards),
                        "example_count": int(example_count),
                        "resume": True,
                    },
                )
                continue

            shard_payload_rows: list[dict[str, Any]] = []
            for row in shard_rows:
                x_raw_t, x_norm_t, boundary_key = _load_full_example_arrays(cache_context, row)
                length_t = torch.tensor([int(x_norm_t.shape[0])], dtype=torch.long)
                x_agg, agg_lengths = aggregate_time_bins(
                    x_norm_t.unsqueeze(0),
                    length_t,
                    stride=int(config.temporal_bin_stride),
                )
                outputs = model(x_agg, agg_lengths)
                targets_norm, valid_mask = build_future_prediction_targets(
                    x_agg,
                    agg_lengths,
                    future_bins=int(config.future_bins),
                )
                x_raw_agg, _ = aggregate_time_bins(
                    x_raw_t.unsqueeze(0),
                    length_t,
                    stride=int(config.temporal_bin_stride),
                )
                targets_raw, _ = build_future_prediction_targets(
                    x_raw_agg,
                    agg_lengths,
                    future_bins=int(config.future_bins),
                )

                forecast_norm = outputs["forecast"].squeeze(0).cpu()
                target_norm = targets_norm.squeeze(0).cpu()
                valid_mask_cpu = valid_mask.squeeze(0).cpu()
                target_raw = targets_raw.squeeze(0).cpu()

                if bool(cache_context.use_normalization):
                    mean_t, std_t = cache_context.session_feature_stats[boundary_key]
                    mean_t = mean_t.to(dtype=forecast_norm.dtype).view(1, 1, -1).cpu()
                    std_t = std_t.to(dtype=forecast_norm.dtype).clamp_min(1e-6).view(1, 1, -1).cpu()
                    forecast_raw = forecast_norm * std_t + mean_t
                else:
                    forecast_raw = forecast_norm.clone()

                shard_payload_rows.append(
                    {
                        "dataset": str(row.dataset),
                        "session_id": str(row.session_id),
                        "subject_id": None if row.subject_id is None else str(row.subject_id),
                        "boundary_key": str(boundary_key),
                        "shard_relpath": str(row.shard_relpath),
                        "example_index": int(row.example_index),
                        "n_time_bins_raw": int(x_raw_t.shape[0]),
                        "n_time_bins_agg": int(x_agg.shape[1]),
                        "future_bins": int(config.future_bins),
                        "forecast_raw": forecast_raw,
                        "target_raw": target_raw,
                        "forecast_norm": forecast_norm,
                        "target_norm": target_norm,
                        "valid_mask": valid_mask_cpu,
                    }
                )
                manifest_rows.append(
                    {
                        "dataset": str(row.dataset),
                        "session_id": str(row.session_id),
                        "subject_id": None if row.subject_id is None else str(row.subject_id),
                        "boundary_key": str(boundary_key),
                        "shard_relpath": str(row.shard_relpath),
                        "example_index": int(row.example_index),
                        "n_time_bins_raw": int(x_raw_t.shape[0]),
                        "n_time_bins_agg": int(x_agg.shape[1]),
                    }
                )
                example_count += 1

            shard_output_path.parent.mkdir(parents=True, exist_ok=True)
            shard_payload = {
                "checkpoint_path": str(checkpoint_path),
                "checkpoint_step": int(payload.get("step", -1)),
                "dataset": dataset,
                "shard_relpath": shard_relpath,
                "rows": shard_payload_rows,
            }
            _write_torch_atomic(
                shard_output_path,
                shard_payload,
            )
            shard_export_count += 1
            completed_shards.append(f"{dataset}:{shard_relpath}")
            progress_payload = {
                "export_kind": "future_prediction_bins_progress",
                "checkpoint_path": str(checkpoint_path),
                "run_name": str(config.run_name),
                "completed_shards": completed_shards,
                "completed_shard_count": int(shard_export_count),
                "total_shards": int(total_shards),
                "example_count": int(example_count),
                "resume": bool(resume),
            }
            _write_json_atomic(progress_path, progress_payload)
            if on_shard_written is not None:
                on_shard_written(
                    {
                        "dataset": dataset,
                        "shard_relpath": shard_relpath,
                        "shard_output_path": str(shard_output_path),
                        "completed_shard_count": int(shard_export_count),
                        "total_shards": int(total_shards),
                        "example_count": int(example_count),
                    }
                )

    manifest = {
        "export_kind": "future_prediction_bins",
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_step": int(payload.get("step", -1)),
        "checkpoint_kind": payload.get("checkpoint_kind"),
        "run_name": str(config.run_name),
        "datasets": list(config.pretrain_datasets),
        "segment_bins": int(config.segment_bins),
        "temporal_bin_stride": int(config.temporal_bin_stride),
        "future_bins": int(config.future_bins),
        "feature_mode": str(config.feature_mode),
        "tx_dim": int(config.tx_dim),
        "sbp_dim": int(config.sbp_dim),
        "example_count": int(example_count),
        "shard_export_count": int(shard_export_count),
        "rows": manifest_rows,
    }
    manifest_path = output_dir / "manifest.json"
    _write_json_atomic(manifest_path, manifest)
    return {
        "output_dir": str(output_dir),
        "manifest_path": str(manifest_path),
        "example_count": int(example_count),
        "shard_export_count": int(shard_export_count),
        "checkpoint_step": int(payload.get("step", -1)),
        "checkpoint_kind": payload.get("checkpoint_kind"),
        "progress_path": str(progress_path),
    }


__all__ = ["export_future_prediction_bins"]

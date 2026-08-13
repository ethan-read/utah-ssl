"""Normalization-stat computation, validation, and public training APIs."""

from __future__ import annotations

import hashlib
import json
import shlex
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Sequence

import numpy as np
import torch

from .cache_identity import (
    cache_variant_name,
    compute_cache_source_signature,
    compute_dataset_cache_source_signature,
)
from .dataset_splits import (
    build_competition_split_problem,
    build_source_split_problem,
)
from .feature_stats import apply_feature_stats, compute_feature_stats
from .experiment_contract import DatasetPlan, SignalSpec
from .normalization_stats import (
    FEATURE_STATS_SCHEMA,
    SUPPORTED_NORMALIZATION_SCOPES,
    build_feature_stats_payload,
    extract_feature_stats_entries,
    write_feature_stats_artifact,
)
from .session_keys import resolve_boundary_key

if TYPE_CHECKING:
    from .cache import CacheAccessConfig, CacheContext, ExampleRow, ShardStore


DEFAULT_DATASET = "brain2text24"
DEFAULT_BOUNDARY_KEY_MODE = "session"
DEFAULT_SPLIT_NAME = "competition_train"
SESSION_STATS_BIN_STRIDE = 2


def _timestamp_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _canonical_stats_root_for_cache(cache_root: str | Path) -> Path:
    cache_root = Path(cache_root)
    if cache_root.parent.name == "data":
        return cache_root.parent / "stats"
    local_stats_root = cache_root / "stats"
    if local_stats_root.exists():
        return local_stats_root
    return cache_root.parent / "stats"


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
    dataset_cache_roots: Mapping[str, str | Path] | None = None,
) -> Path:
    """Resolve the canonical session-stat path for an exact data contract."""

    resolved_signal_spec = SignalSpec.from_value(signal_spec)
    resolved_dataset_plan = DatasetPlan.from_value(dataset_plan)
    primary_root = Path(cache_root)
    effective_roots = {
        dataset: Path((dataset_cache_roots or {}).get(dataset, primary_root))
        for dataset in resolved_dataset_plan.dataset_names
    }
    cache_variant = cache_variant_name(primary_root)
    if any(root.resolve() != primary_root.resolve() for root in effective_roots.values()):
        source_signature = compute_dataset_cache_source_signature(effective_roots)
        cache_variant = f"{cache_variant}_mixed_{source_signature[:12]}"
    stats_dir = (
        _canonical_stats_root_for_cache(primary_root)
        / "session_feature_stats"
        / cache_variant
        / resolved_signal_spec.mode
        / str(boundary_key_mode)
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


def _validate_artifact_metadata(
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
        "utah_ssl/scripts/recompute_feature_stats.py",
        "--scope",
        "session",
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
            cmd.extend(["--dataset-source-split", f"{selection.name}={source_split}"])
    for dataset, dataset_cache_root in sorted((dataset_cache_roots or {}).items()):
        cmd.extend(["--dataset-cache-root", f"{str(dataset)}={str(Path(dataset_cache_root))}"])
    cmd.append("--overwrite")
    return shlex.join(cmd)


def compute_session_feature_stats(
    shard_store: "ShardStore",
    rows_by_dataset: dict[str, list["ExampleRow"]],
    config: "CacheAccessConfig",
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    assert isinstance(config.signal_spec, SignalSpec)
    feature_contract = config.signal_spec.contract
    print("computing SSL session-level featurewise z-scoring stats...")
    session_rows: dict[str, list["ExampleRow"]] = defaultdict(list)
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
        sum_x = np.zeros((full_dim,), dtype=np.float64)
        sum_x2 = np.zeros((full_dim,), dtype=np.float64)
        count_x = np.zeros((full_dim,), dtype=np.float64)
        for row in session_rows[session_key]:
            shard = shard_store.get(row.shard_relpath)
            time_offsets = shard["time_offsets"]
            assert isinstance(time_offsets, np.ndarray)
            start = int(time_offsets[row.example_index])
            stop = int(time_offsets[row.example_index + 1])
            if stop <= start:
                continue

            tx = shard["tx"]
            if feature_contract.uses_tx and isinstance(tx, np.ndarray):
                tx_start, tx_stop = config.signal_spec.selected_columns_for_width(
                    "tx", tx.shape[1]
                )
                tx_window = np.asarray(
                    tx[start:stop:bin_stride, tx_start:tx_stop], dtype=np.float64
                )
                tx_dim = min(tx_window.shape[1], int(config.tx_dim))
                sum_x[:tx_dim] += tx_window[:, :tx_dim].sum(axis=0)
                sum_x2[:tx_dim] += np.square(tx_window[:, :tx_dim]).sum(axis=0)
                count_x[:tx_dim] += tx_window.shape[0]

            sbp = shard["sbp"]
            if feature_contract.uses_sbp and isinstance(sbp, np.ndarray):
                sbp_column_start, sbp_column_stop = config.signal_spec.selected_columns_for_width(
                    "sbp", sbp.shape[1]
                )
                sbp_window = np.asarray(
                    sbp[start:stop:bin_stride, sbp_column_start:sbp_column_stop],
                    dtype=np.float64,
                )
                sbp_dim = min(sbp_window.shape[1], int(config.sbp_dim))
                sbp_start = feature_contract.feature_start("sbp", tx_dim=int(config.tx_dim))
                sbp_slice = slice(sbp_start, sbp_start + sbp_dim)
                sum_x[sbp_slice] += sbp_window[:, :sbp_dim].sum(axis=0)
                sum_x2[sbp_slice] += np.square(sbp_window[:, :sbp_dim]).sum(axis=0)
                count_x[sbp_slice] += sbp_window.shape[0]

        mean = np.zeros((full_dim,), dtype=np.float32)
        std = np.ones((full_dim,), dtype=np.float32)
        present_mask = count_x > 0
        if present_mask.any():
            mean64 = sum_x[present_mask] / count_x[present_mask]
            var64 = np.maximum(
                sum_x2[present_mask] / count_x[present_mask] - np.square(mean64),
                1e-6,
            )
            mean[present_mask] = mean64.astype(np.float32)
            std[present_mask] = np.sqrt(var64).astype(np.float32)
        session_stats[session_key] = (torch.from_numpy(mean), torch.from_numpy(std))
        if session_idx == 1 or session_idx % 25 == 0 or session_idx == total_sessions:
            print(f" session_stats={session_idx}/{total_sessions} current={session_key}")
    return session_stats


def recompute_session_feature_stats(
    *,
    cache_root: str | Path,
    output_path: str | Path,
    signal_spec: SignalSpec,
    dataset_plan: DatasetPlan,
    boundary_key_mode: str = "session",
    seed: int = 7,
    dataset_cache_roots: dict[str, str | Path] | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Recompute and save session-featurewise statistics for a cache plan."""

    # Cache construction depends on statistics at runtime, so these types are
    # imported only for the standalone recomputation path to keep the module
    # dependency acyclic.
    from .cache import CacheAccessConfig, prepare_cache_context

    cache_root = Path(cache_root)
    output_path = Path(output_path)
    if not cache_root.is_dir():
        raise FileNotFoundError(f"Cache root does not exist: {cache_root}")
    metadata_path = output_path.with_suffix(".json")
    if (output_path.exists() or metadata_path.exists()) and not overwrite:
        raise FileExistsError(
            f"Output already exists: {output_path} or {metadata_path}. "
            "Pass overwrite=True to replace existing artifacts."
        )
    if overwrite:
        output_path.unlink(missing_ok=True)
        metadata_path.unlink(missing_ok=True)

    signal_spec = SignalSpec.from_value(signal_spec)
    dataset_plan = DatasetPlan.from_value(dataset_plan)
    normalized_dataset_cache_roots = {
        str(dataset): Path(root)
        for dataset, root in sorted((dataset_cache_roots or {}).items())
    }
    config = CacheAccessConfig(
        dataset_plan=dataset_plan,
        signal_spec=signal_spec,
        mode="drive_direct",
        local_cache_base="/content/utah_ssl_cache",
        force_recopy_local_cache=False,
        seed=int(seed),
        segment_bins=1,
        use_normalization=False,
        boundary_key_mode=str(boundary_key_mode),
        dataset_cache_roots=normalized_dataset_cache_roots or None,
    )
    context = prepare_cache_context(cache_candidates=[cache_root], config=config)
    session_feature_stats = compute_session_feature_stats(
        shard_store=context.shard_store,
        rows_by_dataset=context.rows_by_dataset,
        config=config,
    )
    if not session_feature_stats:
        raise RuntimeError("No session feature stats were computed from the selected cache root.")

    metadata: dict[str, Any] = {
        "kind": "session_featurewise_zscore_stats",
        "created_utc": _timestamp_utc(),
        "source_cache_root": str(cache_root.resolve()),
        "source_cache_name": cache_root.name,
        "source_cache_variant": cache_variant_name(cache_root),
        "source_cache_signature": context.source_cache_signature,
        "signal_spec": signal_spec.to_dict(),
        "dataset_plan": dataset_plan.to_dict(),
        "boundary_key_mode": str(boundary_key_mode),
        "dataset_names": list(context.pretrain_datasets),
        "dataset_count": int(len(context.pretrain_datasets)),
        "session_count": int(len(session_feature_stats)),
        "session_stats_bin_stride": SESSION_STATS_BIN_STRIDE,
        "cache_copy_used": bool(context.cache_copy_used),
        "source_cache_roots": {
            dataset: str(context.drive_dataset_cache_roots[dataset].resolve())
            for dataset in context.pretrain_datasets
        },
    }
    payload = write_feature_stats_artifact(
        output_path=output_path,
        scope="session",
        entries=session_feature_stats,
        metadata=metadata,
    )
    metadata = dict(payload["metadata"])
    return {
        "output_path": output_path,
        "metadata_path": metadata_path,
        "metadata": metadata,
        "session_count": int(len(session_feature_stats)),
        "dataset_count": int(len(context.pretrain_datasets)),
        "cache_root": cache_root,
    }


def load_precomputed_session_feature_stats(
    *,
    stats_path: str | Path,
    cache_root: str | Path,
    signal_spec: SignalSpec | Mapping[str, Any],
    dataset_plan: DatasetPlan | Mapping[str, Sequence[str]],
    boundary_key_mode: str,
    dataset_cache_roots: Mapping[str, str | Path] | None = None,
) -> tuple[dict[str, tuple[torch.Tensor, torch.Tensor]], dict[str, Any], Path]:
    """Load and validate session statistics against an exact cache contract."""

    resolved_signal_spec = SignalSpec.from_value(signal_spec)
    resolved_dataset_plan = DatasetPlan.from_value(dataset_plan)
    expected_dim = int(resolved_signal_spec.full_dim)
    path = Path(stats_path)
    canonical_path = resolve_precomputed_session_stats_path(
        cache_root=cache_root,
        signal_spec=resolved_signal_spec,
        dataset_plan=resolved_dataset_plan,
        boundary_key_mode=str(boundary_key_mode),
        dataset_cache_roots=dataset_cache_roots,
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
    try:
        payload_scope, raw_stats = extract_feature_stats_entries(payload)
    except ValueError as exc:
        raise KeyError(
            "Precomputed session stats payload is missing valid feature statistics."
        ) from exc
    if payload_scope != "session":
        raise ValueError("Precomputed session stats payload has global scope.")

    session_feature_stats: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for key, value in raw_stats.items():
        mean, std = value
        mean_t = torch.as_tensor(mean).float().cpu()
        std_t = torch.as_tensor(std).float().cpu()
        if mean_t.numel() != expected_dim or std_t.numel() != expected_dim:
            raise ValueError(
                "Precomputed session stats artifact is stale or incompatible.\n"
                f"expected_path: {canonical_path}\n"
                f"requested_path: {path}\n"
                f"reason: entry {key!r} has dim mean:{mean_t.numel()} "
                f"std:{std_t.numel()} expected {expected_dim}\n"
                f"recompute_command: {recompute_cmd}"
            )
        session_feature_stats[str(key)] = (mean_t, std_t)

    normalized_dataset_cache_roots = {
        str(dataset): Path(root)
        for dataset, root in sorted((dataset_cache_roots or {}).items())
    }
    expected_cache_signature = (
        compute_dataset_cache_source_signature(normalized_dataset_cache_roots)
        if normalized_dataset_cache_roots
        else compute_cache_source_signature(Path(cache_root))
    )
    expected_metadata: dict[str, Any] = {
        "source_cache_root": str(Path(cache_root).resolve()),
        "source_cache_variant": cache_variant_name(cache_root),
        "source_cache_signature": expected_cache_signature,
        "signal_spec": resolved_signal_spec.to_dict(),
        "dataset_plan": resolved_dataset_plan.to_dict(),
        "boundary_key_mode": str(boundary_key_mode),
        "session_stats_bin_stride": SESSION_STATS_BIN_STRIDE,
    }
    if normalized_dataset_cache_roots:
        expected_metadata["source_cache_roots"] = {
            dataset: str(root.resolve())
            for dataset, root in normalized_dataset_cache_roots.items()
        }
    mismatches = _validate_artifact_metadata(
        metadata=metadata,
        expected_metadata=expected_metadata,
    )
    if mismatches:
        raise ValueError(
            "Precomputed session stats artifact is stale or incompatible.\n"
            f"expected_path: {canonical_path}\n"
            f"requested_path: {path}\n"
            f"reason: {'; '.join(mismatches)}\n"
            f"recompute_command: {recompute_cmd}"
        )
    return session_feature_stats, metadata, path


def load_precomputed_session_feature_stats_into_cache_context(
    *,
    cache_context: "CacheContext",
    stats_path: str | Path,
) -> dict[str, Any]:
    """Load validated session statistics into an existing cache context."""

    session_feature_stats, metadata, path = load_precomputed_session_feature_stats(
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


def _default_split_stats_path(
    *,
    cache_root: Path,
    dataset: str,
    train_split_name: str,
    feature_mode: str,
) -> Path:
    return (
        _canonical_stats_root_for_cache(cache_root)
        / "split_feature_stats"
        / cache_variant_name(cache_root)
        / str(dataset)
        / str(train_split_name)
        / str(feature_mode)
        / "global_v1.pt"
    )


def build_recompute_split_feature_stats_command(
    *,
    cache_root: str | Path,
    dataset: str,
    signal_spec: SignalSpec,
    boundary_key_mode: str,
    split_policy: str,
    output_path: str | Path,
) -> str:
    resolved_signal_spec = SignalSpec.from_value(signal_spec)
    return shlex.join(
        [
            "python",
            "utah_ssl/scripts/recompute_feature_stats.py",
            "--scope",
            "global",
            "--cache-root",
            str(Path(cache_root)),
            "--dataset",
            str(dataset),
            "--feature-mode",
            resolved_signal_spec.mode,
            "--tx-dim",
            str(resolved_signal_spec.tx_dim),
            "--sbp-dim",
            str(resolved_signal_spec.sbp_dim),
            "--column-start",
            str(resolved_signal_spec.column_start),
            "--missing-channel-policy",
            str(resolved_signal_spec.missing_channel_policy),
            "--boundary-key-mode",
            str(boundary_key_mode),
            "--split-policy",
            str(split_policy),
            "--output-path",
            str(Path(output_path)),
            "--overwrite",
        ]
    )


def resolve_precomputed_split_stats_path(
    *,
    cache_root: str | Path,
    dataset: str,
    train_split_name: str,
    signal_spec: SignalSpec,
    preferred_path: str | Path | None,
) -> Path:
    resolved_signal_spec = SignalSpec.from_value(signal_spec)
    if preferred_path is not None:
        return Path(preferred_path)
    return _default_split_stats_path(
        cache_root=Path(cache_root),
        dataset=str(dataset),
        train_split_name=str(train_split_name),
        feature_mode=resolved_signal_spec.mode,
    )


def load_precomputed_split_feature_stats(
    *,
    stats_path: str | Path,
    cache_root: str | Path,
    dataset: str,
    signal_spec: SignalSpec,
    boundary_key_mode: str,
    train_split_name: str,
    val_split_name: str,
    split_policy: str = "competition_train_test",
) -> tuple[tuple[torch.Tensor, torch.Tensor], dict[str, Any], Path]:
    resolved_signal_spec = SignalSpec.from_value(signal_spec)
    feature_mode = resolved_signal_spec.mode
    expected_dim = resolved_signal_spec.full_dim
    path = Path(stats_path)
    canonical_path = _default_split_stats_path(
        cache_root=Path(cache_root),
        dataset=str(dataset),
        train_split_name=str(train_split_name),
        feature_mode=feature_mode,
    )
    recompute_cmd = build_recompute_split_feature_stats_command(
        cache_root=cache_root,
        dataset=str(dataset),
        signal_spec=resolved_signal_spec,
        boundary_key_mode=str(boundary_key_mode),
        split_policy=str(split_policy),
        output_path=canonical_path,
    )
    payload, metadata, path, _ = _load_artifact_payload_and_sidecar(
        path=path,
        canonical_path=canonical_path,
        recompute_cmd=recompute_cmd,
        artifact_name="split stats",
        expected_kind="split_feature_stats",
    )
    try:
        payload_scope, stats_entries = extract_feature_stats_entries(payload)
    except ValueError as exc:
        raise ValueError(
            "Precomputed split stats payload is missing valid feature statistics.\n"
            f"expected_path: {canonical_path}\n"
            f"requested_path: {path}\n"
            f"recompute_command: {recompute_cmd}"
        ) from exc
    if payload_scope != "global" or set(stats_entries) != {"global"}:
        raise ValueError(
            "Precomputed split stats artifact does not contain global statistics.\n"
            f"expected_path: {canonical_path}\n"
            f"requested_path: {path}\n"
            f"recompute_command: {recompute_cmd}"
        )
    mean_t, std_t = stats_entries["global"]

    common_metadata = {
        "source_cache_root": str(Path(cache_root).resolve()),
        "source_cache_variant": cache_variant_name(cache_root),
        "source_cache_signature": compute_dataset_cache_source_signature(
            {str(dataset): Path(cache_root)}
        ),
        "feature_mode": feature_mode,
        "signal_spec": resolved_signal_spec.to_dict(),
        "boundary_key_mode": str(boundary_key_mode),
        "split_policy": str(split_policy),
    }
    split_metadata = {
        "dataset": str(dataset),
        "train_split_name": str(train_split_name),
        "val_split_name": str(val_split_name),
        "feature_dim": int(expected_dim),
    }
    mismatches = _validate_artifact_metadata(
        metadata=metadata,
        expected_metadata=common_metadata,
    )
    mismatches.extend(
        _validate_artifact_metadata(
            metadata=metadata,
            expected_metadata=split_metadata,
        )
    )
    if mean_t.numel() != expected_dim or std_t.numel() != expected_dim:
        mismatches.append(
            f"tensor_dim=mean:{mean_t.numel()} std:{std_t.numel()} expected {expected_dim}"
        )
    if mismatches:
        raise ValueError(
            "Precomputed split stats artifact is stale or incompatible.\n"
            f"expected_path: {canonical_path}\n"
            f"requested_path: {path}\n"
            f"reason: {'; '.join(mismatches)}\n"
            f"recompute_command: {recompute_cmd}"
        )
    return (mean_t, std_t), metadata, path


def recompute_split_feature_stats(
    *,
    cache_root: str | Path,
    output_path: str | Path | None = None,
    dataset: str = DEFAULT_DATASET,
    signal_spec: SignalSpec | dict[str, Any],
    boundary_key_mode: str = DEFAULT_BOUNDARY_KEY_MODE,
    split_policy: str = "competition_train_test",
    overwrite: bool = False,
) -> dict[str, Any]:
    """Recompute global training-split statistics for one dataset."""

    cache_root = Path(cache_root)
    if not cache_root.is_dir():
        raise FileNotFoundError(f"Cache root does not exist: {cache_root}")
    signal_spec = SignalSpec.from_value(signal_spec)
    if str(split_policy) == "competition_train_test":
        problem = build_competition_split_problem(
            cache_root=cache_root,
            dataset=str(dataset),
            signal_spec=signal_spec,
            boundary_key_mode=str(boundary_key_mode),
        )
    elif str(split_policy) == "source_train_val":
        problem = build_source_split_problem(
            cache_root=cache_root,
            dataset=str(dataset),
            signal_spec=signal_spec,
            boundary_key_mode=str(boundary_key_mode),
            train_split_name="train",
            val_split_name="val",
        )
        problem = {**problem, "split_policy": "source_train_val"}
    else:
        raise ValueError("split_policy must be one of {'competition_train_test', 'source_train_val'}")

    train_split_name = str(problem.get("train_split_name", DEFAULT_SPLIT_NAME))
    val_split_name = str(problem.get("val_split_name", "competition_test"))
    resolved_output_path = Path(output_path) if output_path is not None else _default_split_stats_path(
        cache_root=cache_root,
        dataset=str(dataset),
        train_split_name=train_split_name,
        feature_mode=signal_spec.mode,
    )
    metadata_path = resolved_output_path.with_suffix(".json")
    if (resolved_output_path.exists() or metadata_path.exists()) and not overwrite:
        raise FileExistsError(
            f"Output already exists: {resolved_output_path} or {metadata_path}. "
            "Pass overwrite=True to replace existing artifacts."
        )

    stats = compute_feature_stats(
        problem["train_rows"],
        cache_root=cache_root,
        mode="global",
        signal_spec=signal_spec,
    )
    if not isinstance(stats, tuple) or len(stats) != 2:
        raise TypeError("Expected compute_feature_stats(..., mode='global') to return (mean, std).")
    mean, std = stats
    metadata: dict[str, Any] = {
        "kind": "split_feature_stats",
        "created_utc": _timestamp_utc(),
        "source_cache_root": str(cache_root.resolve()),
        "source_cache_name": cache_root.name,
        "source_cache_variant": cache_variant_name(cache_root),
        "source_cache_signature": compute_dataset_cache_source_signature(
            {str(dataset): cache_root}
        ),
        "dataset": str(dataset),
        "feature_mode": signal_spec.mode,
        "signal_spec": signal_spec.to_dict(),
        "boundary_key_mode": str(boundary_key_mode),
        "split_policy": str(problem.get("split_policy", "competition_train_test")),
        "train_split_name": train_split_name,
        "val_split_name": val_split_name,
        "train_examples": int(len(problem["train_rows"])),
        "val_examples": int(len(problem["val_rows"])),
        "train_session_ids": list(problem.get("train_session_ids", ())),
        "val_session_ids": list(problem.get("val_session_ids", ())),
        "feature_dim": int(mean.shape[0]),
    }
    payload = write_feature_stats_artifact(
        output_path=resolved_output_path,
        scope="global",
        entries={"global": (mean, std)},
        metadata=metadata,
    )
    return {
        "output_path": resolved_output_path,
        "metadata_path": metadata_path,
        "metadata": dict(payload["metadata"]),
    }


__all__ = [
    "apply_feature_stats",
    "FEATURE_STATS_SCHEMA",
    "SUPPORTED_NORMALIZATION_SCOPES",
    "build_recompute_split_feature_stats_command",
    "build_recompute_session_feature_stats_command",
    "build_feature_stats_payload",
    "compute_feature_stats",
    "compute_session_feature_stats",
    "extract_feature_stats_entries",
    "load_precomputed_session_feature_stats_into_cache_context",
    "load_precomputed_session_feature_stats",
    "load_precomputed_split_feature_stats",
    "recompute_session_feature_stats",
    "recompute_split_feature_stats",
    "resolve_precomputed_session_stats_path",
    "resolve_precomputed_split_stats_path",
    "write_feature_stats_artifact",
    "SESSION_STATS_BIN_STRIDE",
]

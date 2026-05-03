"""Sweep helpers for masked SSL Colab experiments."""

from __future__ import annotations

import json
from dataclasses import replace
from itertools import product
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from .cache import (
    CacheAccessConfig,
    SESSION_STATS_BIN_STRIDE,
    load_cache_smoothing_provenance,
    load_precomputed_session_feature_stats_into_cache_context,
    prepare_cache_context,
)
from .probe import (
    CanonicalProbeManifestRow,
    DownstreamProbeConfig,
    NotebookProbeEncoderAdapter,
    train_probe_with_metrics,
)
from .training import SSLTrainingConfig, run_ssl_training


SWEEP_VITAL_COLUMNS = [
    "sigma",
    "mask_ratio",
    "ssl_steps",
    "val_model_masked_mse",
    "val_masked_target_std",
    "val_masked_prediction_std",
    "val_prediction_target_corr",
    "downstream_ctc_bpphone",
    "downstream_per",
    "reference_output_len",
    "actual_output_len",
    "actual_over_reference_len",
    "most_common_prediction",
    "most_common_prediction_rate",
]


def sigma_tag(value: float) -> str:
    text = f"{float(value):.6f}".rstrip("0").rstrip(".")
    if "." not in text:
        text = f"{text}.0"
    return text.replace("-", "m").replace(".", "p")


def resolve_smoothed_stats_path(
    *,
    sigma: float,
    session_stats_dir: str | Path,
    active_sigma: float | None = None,
    loaded_session_stats_state: dict[str, Any] | None = None,
    stable_stats_path: str | Path | None = None,
) -> Path | None:
    sigma = float(sigma)
    active_sigma = sigma if active_sigma is None else float(active_sigma)
    tag = sigma_tag(sigma)
    session_stats_dir = Path(session_stats_dir)

    def path_matches_requested_sigma(path_obj: Path) -> bool:
        if abs(sigma) < 1e-6:
            return "smooth_sigma" not in path_obj.name
        return f"smooth_sigma{tag}" in path_obj.name

    if abs(sigma - active_sigma) < 1e-6:
        if isinstance(loaded_session_stats_state, dict):
            loaded_path = loaded_session_stats_state.get("stats_path")
            loaded_meta = (
                loaded_session_stats_state.get("metadata")
                if isinstance(loaded_session_stats_state.get("metadata"), dict)
                else {}
            )
            loaded_sigma = loaded_meta.get("gaussian_smoothing_sigma_bins")
            if loaded_path and loaded_sigma is not None:
                path = Path(loaded_path)
                if (
                    path.exists()
                    and abs(float(loaded_sigma) - sigma) < 1e-6
                    and path_matches_requested_sigma(path)
                ):
                    return path

        if stable_stats_path is not None:
            path = Path(stable_stats_path)
            if path.exists() and path_matches_requested_sigma(path):
                return path

    if abs(sigma) < 1e-6:
        candidates = [
            session_stats_dir
            / (
                "session_feature_stats_session_featurewise_v1_refds000950_cap126682_tx256_sbp256_"
                f"stride{int(SESSION_STATS_BIN_STRIDE)}_stable.pt"
            ),
            session_stats_dir
            / "session_feature_stats_session_featurewise_v1_refds000950_cap126682_tx256_sbp256_stable.pt",
        ]
    else:
        candidates = [
            session_stats_dir
            / (
                "session_feature_stats_session_featurewise_v1_refds000950_cap126682_tx256_sbp256_"
                f"smooth_sigma{tag}_stride{int(SESSION_STATS_BIN_STRIDE)}_stable.pt"
            ),
            session_stats_dir
            / (
                "session_feature_stats_session_featurewise_v1_refds000950_cap126682_tx256_sbp256_"
                f"smooth_sigma{tag}_stable.pt"
            ),
        ]
    for path in candidates:
        if path.exists():
            return path
    return None


def _cache_root_matches_sigma(
    cache_root: Path,
    *,
    sigma: float,
    dataset: str,
) -> bool:
    provenance = load_cache_smoothing_provenance(cache_root, dataset=dataset)
    if provenance is None:
        return abs(float(sigma)) < 1e-6
    try:
        provenance_sigma = float(provenance.get("sigma_bins"))
    except (TypeError, ValueError):
        return False
    return abs(provenance_sigma - float(sigma)) < 1e-6


def resolve_cache_candidates_for_sigma(
    *,
    sigma: float,
    cache_candidates: Sequence[Path],
    dataset: str,
) -> tuple[Path, ...]:
    sigma = float(sigma)
    candidate_paths = [Path(path) for path in cache_candidates]
    if abs(sigma) < 1e-6:
        return tuple(candidate_paths)

    source_roots = {str(path.resolve()) for path in candidate_paths if path.exists()}
    matches: list[tuple[int, Path]] = []
    seen: set[Path] = set()

    def maybe_add(path: Path) -> None:
        if path in seen or not path.exists() or not path.is_dir():
            return
        if not _cache_root_matches_sigma(path, sigma=sigma, dataset=dataset):
            return
        seen.add(path)
        provenance = load_cache_smoothing_provenance(path, dataset=dataset) or {}
        source_root = provenance.get("source_cache_root")
        priority = 0 if isinstance(source_root, str) and source_root in source_roots else 1
        matches.append((priority, path))

    for path in candidate_paths:
        maybe_add(path)
    for path in candidate_paths:
        parent = path.parent
        if not parent.exists():
            continue
        for sibling in sorted(parent.iterdir(), key=lambda item: item.name):
            maybe_add(sibling)

    matches.sort(key=lambda item: (item[0], item[1].name))
    return tuple(path for _, path in matches)


def apply_two_session_split(
    cache_context: Any,
    dataset: str,
    train_session_id: str,
    val_session_id: str,
) -> tuple[int, int]:
    all_rows = list(cache_context.rows_by_dataset.get(dataset, []))
    train_rows = [row for row in all_rows if row.session_id == train_session_id]
    val_rows = [row for row in all_rows if row.session_id == val_session_id]
    if not train_rows:
        raise RuntimeError(f"No rows found for train session {dataset}:{train_session_id}")
    if not val_rows:
        raise RuntimeError(f"No rows found for val session {dataset}:{val_session_id}")

    cache_context.pretrain_datasets = [dataset]
    cache_context.split_rows_by_dataset["train"][dataset] = train_rows
    cache_context.split_rows_by_dataset["val"][dataset] = val_rows
    cache_context.rows_by_dataset[dataset] = [
        row for row in all_rows if row.session_id in {train_session_id, val_session_id}
    ]
    cache_context.session_split_summary[dataset] = {
        "total_sessions": 2,
        "train_sessions": 1,
        "val_sessions": 1,
        "val_eligible": True,
        "train_examples": len(train_rows),
        "val_examples": len(val_rows),
        "train_session_ids": [train_session_id],
        "val_session_ids": [val_session_id],
    }
    cache_context.has_val_datasets = True
    cache_context.sampling_plan_cache.clear()
    return len(train_rows), len(val_rows)


def load_single_session_probe_problem(
    *,
    cache_root: str | Path,
    dataset: str,
    feature_mode: str,
    boundary_key_mode: str,
    session_id: str | None,
    allow_none_as_train: bool = False,
) -> dict[str, Any]:
    dataset_root = Path(cache_root) / dataset
    manifest_path = dataset_root / "manifest.jsonl"
    metadata_path = dataset_root / "metadata.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata: {metadata_path}")

    metadata = json.loads(metadata_path.read_text())
    vocab = metadata.get("phoneme_vocabulary")
    if not isinstance(vocab, dict):
        raise RuntimeError(
            f"Metadata is missing phoneme_vocabulary: {metadata_path}. "
            "Rebuild or relabel the cache before probing."
        )

    rows: list[CanonicalProbeManifestRow] = []
    with manifest_path.open() as handle:
        for line in handle:
            payload = json.loads(line)
            rows.append(
                CanonicalProbeManifestRow(
                    example_id=str(payload["example_id"]),
                    session_id=str(payload["session_id"]),
                    subject_id=str(payload["subject_id"]) if payload.get("subject_id") is not None else None,
                    source_split=str(payload["source_split"]),
                    has_labels=bool(payload.get("has_labels", False)),
                    shard_relpath=str(payload["shard_relpath"]),
                    example_index=int(payload["example_index"]),
                    n_tx_features=int(payload.get("n_tx_features", 0) or 0),
                    n_sbp_features=int(payload.get("n_sbp_features", 0) or 0),
                    target_length=int(payload["target_length"]) if payload.get("target_length") is not None else None,
                    transcript=str(payload.get("transcript", payload.get("transcription", ""))),
                )
            )

    labeled = [row for row in rows if row.has_labels]
    if not labeled:
        raise RuntimeError(f"No labeled rows found in {dataset} manifest.")

    train_split_aliases = {"train", "competition_train"}
    val_split_aliases = {"val", "competition_test"}

    by_session: dict[str, dict[str, list[CanonicalProbeManifestRow]]] = {}
    split_counts: dict[str, int] = {}
    for row in labeled:
        split_key = str(row.source_split).strip().lower()
        split_counts[split_key] = split_counts.get(split_key, 0) + 1

        bucket = by_session.setdefault(row.session_id, {"train": [], "val": []})
        if split_key in train_split_aliases:
            bucket["train"].append(row)
        elif split_key in val_split_aliases:
            bucket["val"].append(row)
        elif split_key == "none" and allow_none_as_train:
            bucket["train"].append(row)

    viable = [
        (session, len(values["train"]), len(values["val"]))
        for session, values in by_session.items()
        if values["train"] and values["val"]
    ]
    if not viable:
        raise RuntimeError(
            f"No {dataset} session has both labeled train and labeled val rows. "
            f"Observed split counts: {split_counts}"
        )

    if session_id is None:
        viable.sort(key=lambda item: (item[2], item[1], min(item[1], item[2])), reverse=True)
        session_id = viable[0][0]
    if session_id not in by_session:
        raise RuntimeError(f"Requested PROBE_SESSION_ID={session_id!r} not found in labeled rows.")
    if not by_session[session_id]["train"] or not by_session[session_id]["val"]:
        raise RuntimeError(
            f"Session {session_id!r} does not have both labeled train and val rows. "
            f"train={len(by_session[session_id]['train'])} val={len(by_session[session_id]['val'])}"
        )

    return {
        "target_session_ids": (session_id,),
        "target_session_label": session_id,
        "target_train_rows": tuple(by_session[session_id]["train"]),
        "target_val_rows": tuple(by_session[session_id]["val"]),
        "vocab": vocab,
        "cache_root": Path(cache_root),
        "feature_mode": str(feature_mode),
        "boundary_key_mode": str(boundary_key_mode),
        "source_session_ids": tuple(),
    }


def run_sigma_mask_probe_sweep(
    *,
    sweep_sigmas: Sequence[float],
    sweep_mask_ratios: Sequence[float],
    sweep_ssl_steps: int,
    cache_candidates: Sequence[Path],
    base_cache_config: CacheAccessConfig,
    base_training_config: SSLTrainingConfig,
    output_root: str | Path,
    device: Any,
    active_cache_context: Any | None,
    active_sigma: float,
    loaded_session_stats_state: dict[str, Any] | None,
    stable_stats_path: str | Path | None,
    session_stats_dir: str | Path,
    use_normalization: bool,
    train_dataset: str,
    train_session_id: str,
    val_session_id: str,
    probe_dataset: str,
    probe_session_id: str | None,
    probe_max_steps: int,
    probe_batch_size: int,
    probe_head_lr: float,
    probe_weight_decay: float,
    probe_train_encoder: bool,
    probe_allow_none_as_train: bool,
    skip_sigma_if_stats_missing: bool,
    require_precomputed_stats_for_sweep: bool,
    boundary_key_mode: str = "subject_if_available",
) -> tuple[pd.DataFrame, dict[float, Any]]:
    contexts_by_sigma: dict[float, Any] = {}
    for sigma_value in sweep_sigmas:
        sigma = float(sigma_value)
        sigma_cache_candidates = resolve_cache_candidates_for_sigma(
            sigma=sigma,
            cache_candidates=cache_candidates,
            dataset=train_dataset,
        )
        if not sigma_cache_candidates:
            msg = f"[sigma={sigma}] no pre-smoothed cache root found."
            if skip_sigma_if_stats_missing:
                print(f"[skip] {msg}")
                continue
            raise RuntimeError(msg)
        stats_path = None
        if use_normalization:
            stats_path = resolve_smoothed_stats_path(
                sigma=sigma,
                session_stats_dir=session_stats_dir,
                active_sigma=active_sigma,
                loaded_session_stats_state=loaded_session_stats_state,
                stable_stats_path=stable_stats_path,
            )

            if stats_path is None:
                msg = f"[sigma={sigma}] no precomputed smoothed stats found."
                if skip_sigma_if_stats_missing:
                    print(f"[skip] {msg}")
                    continue
                if require_precomputed_stats_for_sweep:
                    raise RuntimeError(msg + " Refusing to recompute session z-scores during sweep.")

        can_reuse_existing_ctx = (
            active_cache_context is not None
            and abs(sigma - float(active_sigma)) < 1e-6
            and (not use_normalization or (stats_path is not None and stats_path.exists()))
            and (not use_normalization or bool(getattr(active_cache_context, "session_feature_stats", {})))
            and _cache_root_matches_sigma(
                Path(active_cache_context.cache_root),
                sigma=sigma,
                dataset=train_dataset,
            )
        )

        if can_reuse_existing_ctx:
            context = active_cache_context
            print(f"[sigma={sigma}] reusing existing CACHE_CONTEXT with preloaded session stats.")
        else:
            cache_config = replace(
                base_cache_config,
                use_normalization=use_normalization,
                gaussian_smoothing_sigma_bins=0.0,
                boundary_key_mode=boundary_key_mode,
            )

            context = prepare_cache_context(cache_candidates=sigma_cache_candidates, config=cache_config)

            if use_normalization and stats_path is not None:
                _ = load_precomputed_session_feature_stats_into_cache_context(
                    cache_context=context,
                    stats_path=stats_path,
                )
                print(f"[sigma={sigma}] loaded stats: {stats_path.name}")
            elif use_normalization:
                print(f"[sigma={sigma}] no precomputed stats found; used recomputed in-memory stats.")

        n_train, n_val = apply_two_session_split(
            context,
            train_dataset,
            train_session_id,
            val_session_id,
        )
        print(f"[sigma={sigma}] toy split: train={n_train}, val={n_val}")
        contexts_by_sigma[float(sigma)] = context

    if not contexts_by_sigma:
        raise RuntimeError("No sweep contexts available.")

    records: list[dict[str, Any]] = []
    for sigma_value, mask_ratio in product(sweep_sigmas, sweep_mask_ratios):
        sigma = float(sigma_value)
        if sigma not in contexts_by_sigma:
            continue

        print(f"\n=== sweep run: sigma={sigma}, mask_ratio={mask_ratio} ===")
        context = contexts_by_sigma[sigma]

        run_config = replace(
            base_training_config,
            num_steps=int(sweep_ssl_steps),
            mask_ratio=float(mask_ratio),
        )

        run_state = run_ssl_training(
            cache_context=context,
            config=run_config,
            output_root=Path(output_root),
            device=device,
        )

        probe_problem = load_single_session_probe_problem(
            cache_root=Path(context.cache_root),
            dataset=probe_dataset,
            feature_mode=str(run_config.feature_mode),
            boundary_key_mode=boundary_key_mode,
            session_id=probe_session_id,
            allow_none_as_train=probe_allow_none_as_train,
        )

        probe_config = DownstreamProbeConfig(
            enabled=True,
            seed=int(run_config.seed),
            session_limit=2,
            target_session_count=1,
            probe_batch_size=int(probe_batch_size),
            probe_budget_seconds=10**9,
            max_probe_steps=int(probe_max_steps),
            probe_head_learning_rate=float(probe_head_lr),
            encoder_learning_rate=None,
            weight_decay=float(probe_weight_decay),
            probe_head_type="linear",
        )

        probe_metrics, probe_steps = train_probe_with_metrics(
            problem=probe_problem,
            pretrained_encoder=NotebookProbeEncoderAdapter(run_state["model"].encoder),
            probe_config=probe_config,
            device=device,
            progress_log_path=None,
            train_encoder=bool(probe_train_encoder),
        )

        record = _build_sweep_record(
            sigma=sigma,
            mask_ratio=float(mask_ratio),
            sweep_ssl_steps=int(sweep_ssl_steps),
            run_state=run_state,
            probe_problem=probe_problem,
            probe_metrics=probe_metrics,
            probe_steps=int(probe_steps),
        )
        records.append(record)

        most_common_rate_text = (
            "n/a"
            if record["most_common_prediction_rate"] is None
            else f"{record['most_common_prediction_rate']:.3f}"
        )
        print(
            f"sigma={sigma} mask={mask_ratio} | "
            f"model_mse={record['val_model_masked_mse']:.4f} | "
            f"ctc={record['downstream_ctc_bpphone']:.4f} | "
            f"PER={record['downstream_per']:.4f} | "
            f"len={record['actual_output_len']}/{record['reference_output_len']} "
            f"({record['actual_over_reference_len']:.3f}) | "
            f"top_pred={record['most_common_prediction']} ({most_common_rate_text})"
        )

    results = pd.DataFrame(records).sort_values(
        by=["downstream_ctc_bpphone", "downstream_per"],
        ascending=[True, True],
    ).reset_index(drop=True)
    return results, contexts_by_sigma


def _build_sweep_record(
    *,
    sigma: float,
    mask_ratio: float,
    sweep_ssl_steps: int,
    run_state: dict[str, Any],
    probe_problem: dict[str, Any],
    probe_metrics: dict[str, Any],
    probe_steps: int,
) -> dict[str, Any]:
    alignment = probe_metrics["alignment_diagnostics"]
    reference_len = int(alignment["total_reference_tokens"])
    actual_len = int(alignment["total_predicted_tokens"])
    length_ratio = float(actual_len / max(reference_len, 1))
    index_to_symbol = list(probe_problem["vocab"].get("index_to_symbol", []))

    def symbol_for(token_id: int) -> str:
        token_id = int(token_id)
        if 0 <= token_id < len(index_to_symbol):
            return str(index_to_symbol[token_id])
        return str(token_id)

    prediction_top = list(alignment.get("prediction_top_ids", []))
    most_common_prediction = None
    most_common_prediction_rate = None
    if prediction_top:
        pred_id, pred_count = prediction_top[0]
        most_common_prediction = symbol_for(pred_id)
        most_common_prediction_rate = float(pred_count / max(actual_len, 1))

    latest_train = run_state["train_history"][-1] if run_state.get("train_history") else {}
    latest_val = run_state["val_history"][-1] if run_state.get("val_history") else {}

    return {
        "sigma": sigma,
        "mask_ratio": float(mask_ratio),
        "ssl_steps": int(sweep_ssl_steps),
        "probe_steps": int(probe_steps),
        "probe_session_id": probe_problem["target_session_ids"][0],
        "train_model_masked_mse": float(latest_train.get("loss", float("nan"))),
        "val_model_masked_mse": float(latest_val.get("loss", float("nan"))),
        "val_masked_target_std": latest_val.get("masked_target_std"),
        "val_masked_prediction_std": latest_val.get("masked_prediction_std"),
        "val_prediction_target_corr": latest_val.get("masked_prediction_target_corr"),
        "downstream_ctc_bpphone": float(probe_metrics["val_ctc_bpphone"]),
        "downstream_per": float(probe_metrics["val_phoneme_error_rate"]),
        "actual_output_len": actual_len,
        "reference_output_len": reference_len,
        "actual_over_reference_len": length_ratio,
        "most_common_prediction": most_common_prediction,
        "most_common_prediction_rate": most_common_prediction_rate,
        "run_dir": str(run_state["run_dir"]),
        "checkpoint_path": str(run_state["checkpoint_path"]),
    }

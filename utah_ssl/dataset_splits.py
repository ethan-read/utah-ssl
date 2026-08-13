"""Build labeled train/validation problems from canonical dataset splits."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Sequence

from .canonical_data import (
    CanonicalProbeManifestRow,
    load_canonical_manifest,
    load_canonical_metadata,
    resolve_phoneme_vocabulary,
    validate_canonical_dataset,
)
from .experiment_contract import SignalSpec


def _group_rows_by_session(
    rows: Sequence[CanonicalProbeManifestRow],
) -> tuple[tuple[CanonicalProbeManifestRow, ...], dict[str, int], tuple[str, ...]]:
    grouped: dict[str, list[CanonicalProbeManifestRow]] = {}
    for row in rows:
        grouped.setdefault(str(row.session_id), []).append(row)
    session_ids = tuple(sorted(grouped))
    flattened = tuple(row for session_id in session_ids for row in grouped[session_id])
    counts = {session_id: len(grouped[session_id]) for session_id in session_ids}
    return flattened, counts, session_ids


def _build_split_problem(
    *,
    cache_root: str | Path,
    dataset: str,
    signal_spec: SignalSpec | dict[str, Any],
    boundary_key_mode: str,
    train_split_name: str,
    val_split_name: str,
    split_policy: str,
) -> dict[str, Any]:
    dataset_root, manifest_path, metadata_path = validate_canonical_dataset(
        cache_root,
        dataset=str(dataset),
    )
    manifest_rows = load_canonical_manifest(manifest_path)
    metadata = load_canonical_metadata(metadata_path)
    signal = SignalSpec.from_value(signal_spec)
    train_key = str(train_split_name).strip().lower()
    val_key = str(val_split_name).strip().lower()

    split_counts: Counter[str] = Counter()
    train_candidates: list[CanonicalProbeManifestRow] = []
    val_candidates: list[CanonicalProbeManifestRow] = []
    for row in manifest_rows:
        split_name = str(row.source_split).strip().lower()
        if row.has_labels:
            split_counts[split_name] += 1
        if not row.has_labels or not signal.row_is_compatible(
            has_tx=row.n_tx_features > 0,
            has_sbp=row.n_sbp_features > 0,
            n_tx_features=row.n_tx_features,
            n_sbp_features=row.n_sbp_features,
        ):
            continue
        if split_name == train_key:
            train_candidates.append(row)
        elif split_name == val_key:
            val_candidates.append(row)

    if not train_candidates:
        raise ValueError(
            f"No labeled {train_split_name!r} rows were found for {split_policy}. "
            f"Observed labeled split counts: {dict(split_counts)}"
        )
    if not val_candidates:
        raise ValueError(
            f"No labeled {val_split_name!r} rows were found for {split_policy}. "
            f"Observed labeled split counts: {dict(split_counts)}"
        )

    train_rows, train_counts, train_sessions = _group_rows_by_session(train_candidates)
    val_rows, val_counts, val_sessions = _group_rows_by_session(val_candidates)
    return {
        "canonical_root": dataset_root,
        "manifest_path": manifest_path,
        "metadata_path": metadata_path,
        "manifest_rows": manifest_rows,
        "metadata": metadata,
        "vocab": resolve_phoneme_vocabulary(metadata),
        "cache_root": Path(cache_root),
        "dataset": str(dataset),
        "feature_mode": signal.mode,
        "signal_spec": signal,
        "boundary_key_mode": str(boundary_key_mode),
        "split_policy": str(split_policy),
        "train_split_name": str(train_split_name),
        "val_split_name": str(val_split_name),
        "train_rows": train_rows,
        "val_rows": val_rows,
        "train_examples_by_session": train_counts,
        "val_examples_by_session": val_counts,
        "train_session_ids": train_sessions,
        "val_session_ids": val_sessions,
    }


def build_competition_split_problem(
    *,
    cache_root: str | Path,
    signal_spec: SignalSpec | dict[str, Any],
    dataset: str = "brain2text24",
    boundary_key_mode: str = "session",
) -> dict[str, Any]:
    return _build_split_problem(
        cache_root=cache_root,
        dataset=dataset,
        signal_spec=signal_spec,
        boundary_key_mode=boundary_key_mode,
        train_split_name="competition_train",
        val_split_name="competition_test",
        split_policy="competition_train_test",
    )


def build_source_split_problem(
    *,
    cache_root: str | Path,
    dataset: str,
    signal_spec: SignalSpec | dict[str, Any],
    boundary_key_mode: str = "session",
    train_split_name: str = "train",
    val_split_name: str = "val",
) -> dict[str, Any]:
    return _build_split_problem(
        cache_root=cache_root,
        dataset=dataset,
        signal_spec=signal_spec,
        boundary_key_mode=boundary_key_mode,
        train_split_name=train_split_name,
        val_split_name=val_split_name,
        split_policy="source_split",
    )


__all__ = ["build_competition_split_problem", "build_source_split_problem"]

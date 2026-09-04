"""GRU-timed raw-SBP trajectories around frequent phoneme bigrams.

Bigram identities come from transcript-derived reference phonemes. The frozen
Willett-style decoder supplies only a hard CTC alignment. This local decoder is
an LLM-assisted adaptation with unresolved upstream provenance; see
``experiments/supervised_baselines/PROVENANCE.md``.
"""

from __future__ import annotations

import json
import shutil
import uuid
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.manifolds.analysis import ctc_forced_align
from utah_ssl.canonical_data import (
    CanonicalProbeManifestRow,
    CanonicalShardAccessor,
    load_canonical_manifest,
    load_canonical_metadata,
    resolve_phoneme_vocabulary,
    validate_canonical_dataset,
)
from utah_ssl.experiment_contract import SignalSpec


@dataclass(frozen=True)
class BigramTrajectoryConfig:
    source_split: str = "competition_test"
    minimum_transcript_count: int = 50
    before_bins: int = 12
    after_bins: int = 16
    jitter_bins: tuple[int, ...] = (-2, 0, 2)
    primary_components: int = 6
    reported_components: tuple[int, ...] = (3, 6, 12)
    bootstrap_repetitions: int = 1_000
    standard_deviation_floor: float = 1e-6
    seed: int = 7
    expected_examples: int | None = 880
    expected_sessions: int | None = 24
    expected_bigram_count: int | None = 66
    max_examples_per_session: int | None = None
    smoke: bool = False

    def validate(self) -> None:
        if self.minimum_transcript_count < 1:
            raise ValueError("minimum_transcript_count must be positive")
        if self.before_bins < 0 or self.after_bins < 0:
            raise ValueError("before_bins and after_bins must be nonnegative")
        if 0 not in self.jitter_bins or len(set(self.jitter_bins)) != len(
            self.jitter_bins
        ):
            raise ValueError("jitter_bins must be unique and include zero")
        if self.primary_components < 1 or any(
            value < 1 for value in self.reported_components
        ):
            raise ValueError("PCA component counts must be positive")
        if self.primary_components not in self.reported_components:
            raise ValueError("primary_components must appear in reported_components")
        if self.bootstrap_repetitions < 1:
            raise ValueError("bootstrap_repetitions must be positive")
        if self.standard_deviation_floor <= 0:
            raise ValueError("standard_deviation_floor must be positive")
        if (
            self.max_examples_per_session is not None
            and self.max_examples_per_session < 1
        ):
            raise ValueError("max_examples_per_session must be positive when provided")

    @property
    def path_bins(self) -> int:
        return self.before_bins + self.after_bins + 1


@dataclass(frozen=True)
class CovariancePCA:
    mean: np.ndarray
    covariance: np.ndarray
    components: np.ndarray
    eigenvalues: np.ndarray
    explained_variance_ratio: np.ndarray


@dataclass
class BigramEventSet:
    paths_by_jitter: dict[int, np.ndarray]
    events: pd.DataFrame
    counts: pd.DataFrame
    diagnostics: pd.DataFrame
    exclusions: pd.DataFrame
    session_statistics: pd.DataFrame
    session_means: dict[str, np.ndarray]
    session_stds: dict[str, np.ndarray]
    candidate_pairs: tuple[tuple[int, int], ...]
    symbol_by_id: dict[int, str]
    time_ms: np.ndarray
    metadata: dict[str, Any]


@dataclass
class BigramSourceBundle:
    model_dir: Path
    cache_root: Path
    dataset_root: Path
    export: dict[str, Any]
    selected_examples: pd.DataFrame
    joined_rows: dict[str, CanonicalProbeManifestRow]
    references: dict[int, list[int]]
    transcript_counts: Counter[tuple[int, int]]
    candidate_pairs: tuple[tuple[int, int], ...]
    symbol_by_id: dict[int, str]
    export_vocab: dict[str, Any]
    cache_metadata: dict[str, Any]
    signal_spec: SignalSpec


@dataclass
class BigramTrajectoryResult:
    config: BigramTrajectoryConfig
    event_set: BigramEventSet
    change_pca: CovariancePCA
    state_pca: CovariancePCA
    pca_variance: pd.DataFrame
    ranking: pd.DataFrame
    jitter_sensitivity: pd.DataFrame
    session_trajectory_index: pd.DataFrame
    session_change_paths: np.ndarray
    session_state_paths: np.ndarray
    pooled_change_paths: np.ndarray
    pooled_state_paths: np.ndarray


def _parse_reference_ids(value: object) -> list[int]:
    if pd.isna(value) or not str(value).strip():
        return []
    return [int(token) for token in str(value).split()]


def _symbol_map(vocab: dict[str, Any]) -> dict[int, str]:
    if isinstance(vocab.get("id_to_symbol"), dict):
        return {int(key): str(value) for key, value in vocab["id_to_symbol"].items()}
    return {index: str(value) for index, value in enumerate(vocab["index_to_symbol"])}


def _load_logits_export(model_dir: str | Path) -> dict[str, Any]:
    root = Path(model_dir)
    required = (
        "metadata.json",
        "validation.json",
        "_SUCCESS.json",
        "examples.csv",
        "tokens.csv",
        "shards.json",
    )
    missing = [name for name in required if not (root / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing GRU export artifacts: {missing}")
    metadata = json.loads((root / "metadata.json").read_text())
    validation = json.loads((root / "validation.json").read_text())
    success = json.loads((root / "_SUCCESS.json").read_text())
    if validation.get("status") != "passed" or success.get("status") != "complete":
        raise ValueError("GRU export is not marked complete and validated")
    expected = {
        "dataset": "brain2text24",
        "feature_mode": "tx_sbp",
        "checkpoint_step": 18_300,
        "patch_size_bins": 14,
        "patch_stride_bins": 4,
        "bin_size_ms": 20,
    }
    mismatches = {
        key: (metadata.get(key), value)
        for key, value in expected.items()
        if metadata.get(key) != value
    }
    if mismatches:
        raise ValueError(f"GRU export contract mismatch: {mismatches}")
    if metadata.get("selected_source_splits") != ["competition_test"]:
        raise ValueError("GRU export must contain only competition_test")
    examples = pd.read_csv(root / "examples.csv")
    tokens = pd.read_csv(root / "tokens.csv")
    manifests = json.loads((root / "shards.json").read_text())
    logits_parts: list[np.ndarray] = []
    example_parts: list[np.ndarray] = []
    token_parts: list[np.ndarray] = []
    for shard in manifests:
        with np.load(root / "shards" / shard["shard"]) as arrays:
            logits_parts.append(np.asarray(arrays["logits"], dtype=np.float32))
            example_parts.append(
                np.asarray(arrays["token_example_index"], dtype=np.int64)
            )
            token_parts.append(np.asarray(arrays["token_index"], dtype=np.int64))
    logits = np.concatenate(logits_parts)
    example_indices = np.concatenate(example_parts)
    token_indices = np.concatenate(token_parts)
    if not (len(logits) == len(tokens) == len(example_indices) == len(token_indices)):
        raise ValueError("GRU export token arrays and table have inconsistent lengths")
    if not np.array_equal(
        example_indices, tokens.example_export_index.to_numpy(dtype=np.int64)
    ):
        raise ValueError("GRU export example indices do not align")
    if not np.array_equal(token_indices, tokens.token_index.to_numpy(dtype=np.int64)):
        raise ValueError("GRU export token indices do not align")
    if not np.isfinite(logits).all():
        raise ValueError("GRU export logits contain nonfinite values")
    return {
        "root": root,
        "metadata": metadata,
        "examples": examples,
        "tokens": tokens,
        "logits": logits,
        "example_indices": example_indices,
    }


def _select_examples(
    examples: pd.DataFrame, config: BigramTrajectoryConfig
) -> pd.DataFrame:
    required_columns = {
        "example_export_index",
        "example_id",
        "session_id",
        "source_split",
        "reference_ids",
        "input_length_bins",
    }
    missing_columns = sorted(required_columns - set(examples.columns))
    if missing_columns:
        raise ValueError(f"GRU examples table is missing columns: {missing_columns}")
    if examples.example_export_index.duplicated().any():
        raise ValueError("GRU examples table has duplicate export indices")
    if examples.example_id.astype(str).duplicated().any():
        raise ValueError("GRU examples table has duplicate example IDs")
    selected = examples.loc[
        examples.source_split.astype(str).eq(config.source_split)
    ].copy()
    selected = selected.sort_values("example_export_index")
    if config.max_examples_per_session is not None:
        selected = selected.groupby("session_id", sort=False, group_keys=False).head(
            config.max_examples_per_session
        )
    selected = selected.reset_index(drop=True)
    if (
        config.expected_examples is not None
        and len(selected) != config.expected_examples
    ):
        raise ValueError(
            f"Expected {config.expected_examples} examples, found {len(selected)}"
        )
    sessions = tuple(sorted(selected.session_id.astype(str).unique()))
    if (
        config.expected_sessions is not None
        and len(sessions) != config.expected_sessions
    ):
        raise ValueError(
            f"Expected {config.expected_sessions} sessions, found {len(sessions)}"
        )
    return selected


def count_reference_bigrams(
    references: Iterable[Sequence[int]],
    *,
    excluded_ids: set[int] | frozenset[int],
) -> Counter[tuple[int, int]]:
    counts: Counter[tuple[int, int]] = Counter()
    for reference in references:
        for first, second in zip(reference, reference[1:]):
            pair = (int(first), int(second))
            if pair[0] not in excluded_ids and pair[1] not in excluded_ids:
                counts[pair] += 1
    return counts


def transition_anchor_bin(
    first_span: tuple[int, int],
    second_span: tuple[int, int],
    *,
    patch_size: int,
    patch_stride: int,
) -> int:
    first_onset, first_offset = map(int, first_span)
    second_onset, second_offset = map(int, second_span)
    if (
        first_offset <= first_onset
        or second_offset <= second_onset
        or second_onset < first_offset
    ):
        raise ValueError("Adjacent CTC spans must be nonempty and time ordered")
    first_center = (first_offset - 1) * int(patch_stride) + int(patch_size) // 2
    second_center = second_onset * int(patch_stride) + int(patch_size) // 2
    return (first_center + second_center) // 2


def alignment_confidence(
    logits: np.ndarray,
    first_span: tuple[int, int],
    second_span: tuple[int, int],
    pair: tuple[int, int],
) -> float:
    values = np.asarray(logits, dtype=np.float64)
    shifted = values - values.max(axis=1, keepdims=True)
    log_probs = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    selected = np.concatenate(
        [
            log_probs[first_span[0] : first_span[1], int(pair[0])],
            log_probs[second_span[0] : second_span[1], int(pair[1])],
        ]
    )
    if not len(selected):
        raise ValueError("Cannot score empty aligned spans")
    return float(np.exp(selected.mean()))


def extract_transition_window(
    raw: np.ndarray,
    *,
    anchor_bin: int,
    before_bins: int,
    after_bins: int,
) -> np.ndarray | None:
    """Return one full window or None; partial windows are never truncated."""

    start = int(anchor_bin) - int(before_bins)
    stop = int(anchor_bin) + int(after_bins) + 1
    if start < 0 or stop > len(raw):
        return None
    return np.asarray(raw[start:stop]).copy()


def _validate_cache_metadata(cache_root: Path, metadata: dict[str, Any]) -> None:
    if cache_root.name != "cache_v1_sbpclip12500_fp16_raw":
        raise ValueError(f"Unexpected raw-SBP cache identity: {cache_root}")
    expected = {
        "dataset_family": "brain2text24",
        "bin_size_ms": 20,
        "sbp_storage_dtype": "float16",
        "sbp_clip_threshold": 12_500.0,
    }
    mismatches = {
        key: (metadata.get(key), value)
        for key, value in expected.items()
        if metadata.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Raw-SBP cache contract mismatch: {mismatches}")


def _join_cache_rows(
    selected_examples: pd.DataFrame,
    cache_rows: Sequence[CanonicalProbeManifestRow],
    *,
    signal_spec: SignalSpec,
) -> dict[str, CanonicalProbeManifestRow]:
    eligible = [row for row in cache_rows if row.source_split == "competition_test"]
    by_id = {row.example_id: row for row in eligible}
    if len(by_id) != len(eligible):
        raise ValueError("Raw cache contains duplicate competition_test example IDs")
    if len(selected_examples) == 880 and set(
        selected_examples.example_id.astype(str)
    ) != set(by_id):
        raise ValueError(
            "Full GRU export and raw cache do not contain the same 880 "
            "competition_test example IDs"
        )
    missing = sorted(set(selected_examples.example_id.astype(str)) - set(by_id))
    if missing:
        raise ValueError(f"Raw cache is missing GRU-export examples: {missing[:5]}")
    joined: dict[str, CanonicalProbeManifestRow] = {}
    for example in selected_examples.itertuples(index=False):
        row = by_id[str(example.example_id)]
        if row.session_id != str(example.session_id) or row.source_split != str(
            example.source_split
        ):
            raise ValueError(
                f"Cache/export session or split mismatch for {example.example_id}"
            )
        if not signal_spec.row_is_compatible(
            has_tx=False,
            has_sbp=True,
            n_tx_features=0,
            n_sbp_features=row.n_sbp_features,
        ):
            raise ValueError(
                f"Raw cache row violates the SBP signal contract: {row.example_id}"
            )
        joined[row.example_id] = row
    return joined


def normalize_paths_by_session(
    paths: np.ndarray,
    session_ids: Sequence[str],
    *,
    means: dict[str, np.ndarray],
    stds: dict[str, np.ndarray],
) -> np.ndarray:
    """Apply fixed per-session channel statistics while preserving NaN paths."""

    values = np.asarray(paths, dtype=np.float32).copy()
    sessions = np.asarray(session_ids, dtype=object)
    if len(values) != len(sessions):
        raise ValueError("session_ids must align one-to-one with paths")
    missing = sorted(set(sessions.tolist()) - set(means))
    if missing or not set(sessions.tolist()).issubset(stds):
        raise ValueError(f"Missing session normalization statistics: {missing}")
    for session_id in dict.fromkeys(sessions.tolist()):
        mask = sessions == session_id
        values[mask] = (values[mask] - np.asarray(means[session_id])) / np.asarray(
            stds[session_id]
        )
    return values


def prepare_bigram_sources(
    model_dir: str | Path,
    cache_root: str | Path,
    *,
    config: BigramTrajectoryConfig | None = None,
) -> BigramSourceBundle:
    """Validate and join the GRU export and canonical raw-SBP cache."""

    config = config or BigramTrajectoryConfig()
    config.validate()
    signal_spec = SignalSpec.sbp_only(
        sbp_dim=128,
        column_start=0,
        missing_channel_policy="error",
    )
    export = _load_logits_export(model_dir)
    selected = _select_examples(export["examples"], config)
    dataset_root, manifest_path, metadata_path = validate_canonical_dataset(
        cache_root,
        dataset="brain2text24",
    )
    cache_metadata = load_canonical_metadata(metadata_path)
    _validate_cache_metadata(Path(cache_root), cache_metadata)
    cache_vocab = resolve_phoneme_vocabulary(cache_metadata)
    export_vocab = dict(export["metadata"]["vocab"])
    vocabulary_keys = ("blank_index", "sil_index", "num_classes")
    vocab_metadata_matches = all(
        int(cache_vocab[key]) == int(export_vocab[key]) for key in vocabulary_keys
    )
    if (
        _symbol_map(cache_vocab) != _symbol_map(export_vocab)
        or not vocab_metadata_matches
    ):
        raise ValueError("Raw cache and GRU export vocabularies disagree")
    symbol_by_id = _symbol_map(export_vocab)
    joined = _join_cache_rows(
        selected,
        load_canonical_manifest(manifest_path),
        signal_spec=signal_spec,
    )
    references = {
        int(row.example_export_index): _parse_reference_ids(row.reference_ids)
        for row in selected.itertuples(index=False)
    }
    excluded = {
        int(export_vocab["blank_index"]),
        int(export_vocab["sil_index"]),
    }
    transcript_counts = count_reference_bigrams(
        references.values(),
        excluded_ids=excluded,
    )
    candidate_pairs = tuple(
        sorted(
            pair
            for pair, count in transcript_counts.items()
            if count >= config.minimum_transcript_count
        )
    )
    if (
        config.expected_bigram_count is not None
        and len(candidate_pairs) != config.expected_bigram_count
    ):
        raise ValueError(
            f"Expected {config.expected_bigram_count} qualifying bigrams, "
            f"found {len(candidate_pairs)}"
        )
    return BigramSourceBundle(
        model_dir=Path(model_dir),
        cache_root=Path(cache_root),
        dataset_root=dataset_root,
        export=export,
        selected_examples=selected,
        joined_rows=joined,
        references=references,
        transcript_counts=transcript_counts,
        candidate_pairs=candidate_pairs,
        symbol_by_id=symbol_by_id,
        export_vocab=export_vocab,
        cache_metadata=cache_metadata,
        signal_spec=signal_spec,
    )


def build_bigram_event_set(
    model_dir: str | Path,
    cache_root: str | Path,
    *,
    config: BigramTrajectoryConfig | None = None,
    progress: Callable[[str], None] | None = None,
    sources: BigramSourceBundle | None = None,
) -> BigramEventSet:
    """Join the logits export to raw SBP and extract transition-centered paths."""

    config = config or BigramTrajectoryConfig()
    config.validate()
    sources = sources or prepare_bigram_sources(
        model_dir,
        cache_root,
        config=config,
    )
    if sources.model_dir != Path(model_dir) or sources.cache_root != Path(cache_root):
        raise ValueError("Prepared source bundle does not match the requested paths")
    signal_spec = sources.signal_spec
    export = sources.export
    selected = sources.selected_examples
    dataset_root = sources.dataset_root
    cache_metadata = sources.cache_metadata
    export_vocab = sources.export_vocab
    symbol_by_id = sources.symbol_by_id
    joined = sources.joined_rows
    references = sources.references
    transcript_counts = sources.transcript_counts
    candidate_pairs = sources.candidate_pairs
    candidate_set = set(candidate_pairs)
    accessor = CanonicalShardAccessor(dataset_root.parent)
    session_moments: dict[str, dict[str, Any]] = {}
    event_rows: list[dict[str, Any]] = []
    exclusion_rows: list[dict[str, Any]] = []
    paths: dict[int, list[np.ndarray]] = {jitter: [] for jitter in config.jitter_bins}
    aligned_counts: Counter[tuple[int, int]] = Counter()
    valid_counts: dict[int, Counter[tuple[int, int]]] = {
        jitter: Counter() for jitter in config.jitter_bins
    }
    diagnostic_rows: list[dict[str, Any]] = []
    try:
        for position, example in enumerate(
            selected.itertuples(index=False),
            start=1,
        ):
            export_index = int(example.example_export_index)
            row = joined[str(example.example_id)]
            raw = accessor.load_features(row, signal_spec=signal_spec)
            labels = accessor.load_labels(row)
            reference = references[export_index]
            if labels is None or not np.array_equal(
                labels,
                np.asarray(reference, dtype=np.int64),
            ):
                raise ValueError(
                    "Raw-cache labels disagree with the GRU reference for "
                    f"{example.example_id}"
                )
            if raw.ndim != 2 or raw.shape[1] != 128 or not np.isfinite(raw).all():
                raise ValueError(
                    f"Invalid raw SBP for {example.example_id}: {raw.shape}"
                )
            expected_raw_bins = int(example.input_length_bins)
            if len(raw) != expected_raw_bins or (
                row.n_time_bins is not None and len(raw) != int(row.n_time_bins)
            ):
                raise ValueError(
                    f"Raw/cache/export length mismatch for {example.example_id}: "
                    f"raw={len(raw)}, export={expected_raw_bins}, "
                    f"manifest={row.n_time_bins}"
                )
            session_id = str(example.session_id)
            moment = session_moments.setdefault(
                session_id,
                {
                    "count": 0,
                    "sum": np.zeros(128, dtype=np.float64),
                    "sum_squares": np.zeros(128, dtype=np.float64),
                },
            )
            moment["count"] += len(raw)
            moment["sum"] += raw.sum(axis=0, dtype=np.float64)
            moment["sum_squares"] += np.square(
                raw,
                dtype=np.float64,
            ).sum(axis=0)
            token_rows = np.flatnonzero(export["example_indices"] == export_index)
            candidate_positions = [
                (index, (int(first), int(second)))
                for index, (first, second) in enumerate(zip(reference, reference[1:]))
                if (int(first), int(second)) in candidate_set
            ]
            status = "included"
            try:
                spans = ctc_forced_align(
                    export["logits"][token_rows],
                    reference,
                    blank_index=int(export_vocab["blank_index"]),
                )
            except (ValueError, IndexError) as exc:
                spans = []
                status = f"alignment_failed:{type(exc).__name__}"
                for reference_index, pair in candidate_positions:
                    exclusion_rows.append(
                        {
                            "example_export_index": export_index,
                            "example_id": str(example.example_id),
                            "session_id": session_id,
                            "reference_index": reference_index,
                            "bigram": (
                                f"{symbol_by_id[pair[0]]}-{symbol_by_id[pair[1]]}"
                            ),
                            "reason": status,
                        }
                    )
            example_candidates = len(candidate_positions)
            example_aligned = 0
            example_valid = 0
            example_boundary_excluded = 0
            if spans:
                for reference_index, pair in candidate_positions:
                    example_aligned += 1
                    aligned_counts[pair] += 1
                    anchor = transition_anchor_bin(
                        spans[reference_index],
                        spans[reference_index + 1],
                        patch_size=int(export["metadata"]["patch_size_bins"]),
                        patch_stride=int(export["metadata"]["patch_stride_bins"]),
                    )
                    confidence = alignment_confidence(
                        export["logits"][token_rows],
                        spans[reference_index],
                        spans[reference_index + 1],
                        pair,
                    )
                    extracted: dict[int, np.ndarray | None] = {}
                    for jitter in config.jitter_bins:
                        center = anchor + int(jitter)
                        extracted[jitter] = extract_transition_window(
                            raw,
                            anchor_bin=center,
                            before_bins=config.before_bins,
                            after_bins=config.after_bins,
                        )
                    if extracted[0] is None:
                        example_boundary_excluded += 1
                        exclusion_rows.append(
                            {
                                "example_export_index": export_index,
                                "example_id": str(example.example_id),
                                "session_id": session_id,
                                "reference_index": reference_index,
                                "bigram": (
                                    f"{symbol_by_id[pair[0]]}-{symbol_by_id[pair[1]]}"
                                ),
                                "reason": "nominal_window_out_of_bounds",
                                "anchor_bin": anchor,
                                "alignment_confidence": confidence,
                            }
                        )
                        continue
                    example_valid += 1
                    event_rows.append(
                        {
                            "event_index": len(event_rows),
                            "example_export_index": export_index,
                            "example_id": str(example.example_id),
                            "session_id": session_id,
                            "reference_index": reference_index,
                            "first_id": pair[0],
                            "second_id": pair[1],
                            "first_symbol": symbol_by_id[pair[0]],
                            "second_symbol": symbol_by_id[pair[1]],
                            "bigram": (
                                f"{symbol_by_id[pair[0]]}-{symbol_by_id[pair[1]]}"
                            ),
                            "anchor_bin": anchor,
                            "anchor_ms": (
                                anchor * int(export["metadata"]["bin_size_ms"])
                            ),
                            "alignment_confidence": confidence,
                            **{
                                f"jitter_{jitter:+d}_valid": (
                                    extracted[jitter] is not None
                                )
                                for jitter in config.jitter_bins
                            },
                        }
                    )
                    for jitter in config.jitter_bins:
                        value = extracted[jitter]
                        if value is None:
                            exclusion_rows.append(
                                {
                                    "example_export_index": export_index,
                                    "example_id": str(example.example_id),
                                    "session_id": session_id,
                                    "reference_index": reference_index,
                                    "bigram": (
                                        f"{symbol_by_id[pair[0]]}-"
                                        f"{symbol_by_id[pair[1]]}"
                                    ),
                                    "reason": "jitter_window_out_of_bounds",
                                    "anchor_bin": anchor,
                                    "jitter_bins": int(jitter),
                                    "alignment_confidence": confidence,
                                }
                            )
                            paths[jitter].append(
                                np.full(
                                    (config.path_bins, 128),
                                    np.nan,
                                    dtype=np.float32,
                                )
                            )
                        else:
                            paths[jitter].append(np.asarray(value, dtype=np.float32))
                            valid_counts[jitter][pair] += 1
            diagnostic_rows.append(
                {
                    "example_export_index": export_index,
                    "example_id": str(example.example_id),
                    "session_id": session_id,
                    "status": status,
                    "candidate_occurrences": example_candidates,
                    "aligned_occurrences": example_aligned,
                    "nominal_valid_occurrences": example_valid,
                    "nominal_boundary_excluded": example_boundary_excluded,
                    "raw_bins": len(raw),
                }
            )
            if progress is not None and (
                position % 100 == 0 or position == len(selected)
            ):
                progress(f"Loaded and aligned {position}/{len(selected)} examples")
    finally:
        accessor.close()
    if not event_rows:
        raise RuntimeError("No nominal bigram transition windows were extracted")

    session_means: dict[str, np.ndarray] = {}
    session_stds: dict[str, np.ndarray] = {}
    statistics_rows = []
    for session_id, moment in sorted(session_moments.items()):
        count = int(moment["count"])
        mean = moment["sum"] / count
        variance = np.maximum(
            moment["sum_squares"] / count - np.square(mean),
            0.0,
        )
        std = np.maximum(
            np.sqrt(variance),
            config.standard_deviation_floor,
        )
        session_means[session_id] = mean
        session_stds[session_id] = std
        statistics_rows.append(
            {
                "session_id": session_id,
                "n_bins": count,
                "minimum_channel_std": float(std.min()),
                "maximum_channel_std": float(std.max()),
            }
        )
    events = pd.DataFrame(event_rows)
    normalized_paths: dict[int, np.ndarray] = {}
    for jitter, raw_paths in paths.items():
        normalized_paths[jitter] = normalize_paths_by_session(
            np.stack(raw_paths),
            events.session_id.astype(str).tolist(),
            means=session_means,
            stds=session_stds,
        )
    count_rows = []
    for pair in candidate_pairs:
        count_rows.append(
            {
                "first_id": pair[0],
                "second_id": pair[1],
                "first_symbol": symbol_by_id[pair[0]],
                "second_symbol": symbol_by_id[pair[1]],
                "bigram": (f"{symbol_by_id[pair[0]]}-{symbol_by_id[pair[1]]}"),
                "transcript_count": int(transcript_counts[pair]),
                "aligned_count": int(aligned_counts[pair]),
                **{
                    f"jitter_{jitter:+d}_valid_count": int(valid_counts[jitter][pair])
                    for jitter in config.jitter_bins
                },
            }
        )
    metadata = {
        "model_dir": str(model_dir),
        "cache_root": str(cache_root),
        "dataset": "brain2text24",
        "source_split": config.source_split,
        "signal_spec": signal_spec.to_dict(),
        "raw_cache_metadata": cache_metadata,
        "gru_export_metadata": export["metadata"],
        "normalization": (
            "per-session featurewise z-score over all selected utterance bins; "
            "transductive"
        ),
        "timing": (
            "reference-constrained hard CTC Viterbi alignment from frozen GRU logits"
        ),
    }
    exclusion_columns = [
        "example_export_index",
        "example_id",
        "session_id",
        "reference_index",
        "bigram",
        "reason",
        "anchor_bin",
        "jitter_bins",
        "alignment_confidence",
    ]
    return BigramEventSet(
        paths_by_jitter=normalized_paths,
        events=events,
        counts=pd.DataFrame(count_rows),
        diagnostics=pd.DataFrame(diagnostic_rows),
        exclusions=pd.DataFrame(
            exclusion_rows,
            columns=exclusion_columns,
        ),
        session_statistics=pd.DataFrame(statistics_rows),
        session_means=session_means,
        session_stds=session_stds,
        candidate_pairs=candidate_pairs,
        symbol_by_id=symbol_by_id,
        time_ms=(np.arange(-config.before_bins, config.after_bins + 1) * 20),
        metadata=metadata,
    )


def fit_equal_bigram_pca(
    paths: np.ndarray,
    labels: Sequence[str],
    *,
    temporal_center: bool,
) -> CovariancePCA:
    """Fit exact covariance PCA while giving every bigram equal total weight."""

    values = np.asarray(paths, dtype=np.float64)
    if values.ndim != 3 or not np.isfinite(values).all():
        raise ValueError("PCA paths must be finite with shape [events, time, channels]")
    labels_array = np.asarray(labels, dtype=object)
    if len(labels_array) != len(values):
        raise ValueError("PCA labels must align with paths")
    if temporal_center:
        values = values - values.mean(axis=1, keepdims=True)
    unique = tuple(dict.fromkeys(labels_array.tolist()))
    if not unique:
        raise ValueError("At least one bigram is required")
    label_means = [
        values[labels_array == label].reshape(-1, values.shape[-1]).mean(axis=0)
        for label in unique
    ]
    grand_mean = np.mean(label_means, axis=0)
    covariance = np.zeros(
        (values.shape[-1], values.shape[-1]),
        dtype=np.float64,
    )
    for label in unique:
        points = (
            values[labels_array == label].reshape(-1, values.shape[-1]) - grand_mean
        )
        covariance += points.T @ points / len(points)
    covariance /= len(unique)
    covariance = (covariance + covariance.T) / 2.0
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    tolerance = max(1.0, float(abs(eigenvalues[0]))) * 1e-10
    if float(eigenvalues.min()) < -tolerance:
        raise ValueError(
            f"PCA covariance has a materially negative eigenvalue: {eigenvalues.min()}"
        )
    eigenvalues = np.maximum(eigenvalues, 0.0)
    components = eigenvectors[:, order].T
    total = float(eigenvalues.sum())
    if total <= 0:
        raise ValueError("PCA covariance has zero total variance")
    return CovariancePCA(
        mean=grand_mean,
        covariance=covariance,
        components=components,
        eigenvalues=eigenvalues,
        explained_variance_ratio=eigenvalues / total,
    )


def project_paths(
    paths: np.ndarray,
    pca: CovariancePCA,
    *,
    temporal_center: bool,
) -> np.ndarray:
    values = np.asarray(paths, dtype=np.float64)
    if temporal_center:
        values = values - values.mean(axis=1, keepdims=True)
    return ((values - pca.mean) @ pca.components.T).astype(np.float32)


def _session_means(
    paths: np.ndarray,
    events: pd.DataFrame,
) -> tuple[pd.DataFrame, np.ndarray]:
    rows: list[dict[str, Any]] = []
    values: list[np.ndarray] = []
    for bigram in sorted(events.bigram.unique()):
        bigram_mask = events.bigram.eq(bigram).to_numpy()
        sessions = sorted(events.loc[bigram_mask, "session_id"].unique())
        for session_id in sessions:
            mask = bigram_mask & events.session_id.eq(session_id).to_numpy()
            rows.append(
                {
                    "bigram": bigram,
                    "session_id": session_id,
                    "event_count": int(mask.sum()),
                }
            )
            values.append(np.nanmean(paths[mask], axis=0))
    if not values:
        raise ValueError("No session-mean trajectories could be computed")
    return pd.DataFrame(rows), np.stack(values)


def _captured_curve(
    path: np.ndarray,
    pca: CovariancePCA,
) -> tuple[np.ndarray, float]:
    centered = np.asarray(path, dtype=np.float64)
    centered -= centered.mean(axis=0, keepdims=True)
    denominator = float(np.square(centered).sum())
    if denominator <= 0:
        return np.full(len(pca.components), np.nan), 0.0
    scores = centered @ pca.components.T
    curve = np.cumsum(np.square(scores).sum(axis=0)) / denominator
    magnitude = denominator / max(1, len(centered) - 1)
    return np.clip(curve, 0.0, 1.0), magnitude


def _loso_correlation(
    paths: np.ndarray,
    pca: CovariancePCA,
    components: int,
) -> float:
    if len(paths) < 2:
        return float("nan")
    correlations = []
    for index in range(len(paths)):
        held = paths[index] - paths[index].mean(axis=0, keepdims=True)
        reference = np.mean(np.delete(paths, index, axis=0), axis=0)
        reference -= reference.mean(axis=0, keepdims=True)
        held_score = held @ pca.components[:components].T
        reference_score = reference @ pca.components[:components].T
        if np.std(held_score) > 0 and np.std(reference_score) > 0:
            correlations.append(
                float(
                    np.corrcoef(
                        held_score.ravel(),
                        reference_score.ravel(),
                    )[0, 1]
                )
            )
    return float(np.median(correlations)) if correlations else float("nan")


def _bootstrap_capture(
    session_paths: np.ndarray,
    pca: CovariancePCA,
    *,
    components: int,
    repetitions: int,
    seed: int,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    values = np.empty(repetitions, dtype=np.float64)
    for repetition in range(repetitions):
        sampled = session_paths[
            rng.integers(0, len(session_paths), size=len(session_paths))
        ].mean(axis=0)
        curve, _ = _captured_curve(sampled, pca)
        values[repetition] = curve[components - 1]
    return (
        float(np.nanquantile(values, 0.025)),
        float(np.nanquantile(values, 0.975)),
    )


def rank_bigram_trajectories(
    event_set: BigramEventSet,
    pca: CovariancePCA,
    *,
    config: BigramTrajectoryConfig,
    jitter: int = 0,
    bootstrap: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Rank session-equal mean paths by captured temporal-change variance."""

    valid = np.isfinite(event_set.paths_by_jitter[jitter]).all(axis=(1, 2))
    paths = event_set.paths_by_jitter[jitter][valid]
    events = event_set.events.loc[valid].reset_index(drop=True)
    paths = paths - paths.mean(axis=1, keepdims=True)
    session_index, session_paths = _session_means(paths, events)
    rows = []
    counts = event_set.counts.set_index("bigram")
    for bigram_index, bigram in enumerate(sorted(events.bigram.unique())):
        mask = session_index.bigram.eq(bigram).to_numpy()
        per_session = session_paths[mask]
        pooled = per_session.mean(axis=0)
        curve, magnitude = _captured_curve(pooled, pca)
        low, high = (float("nan"), float("nan"))
        if bootstrap:
            low, high = _bootstrap_capture(
                per_session,
                pca,
                components=config.primary_components,
                repetitions=config.bootstrap_repetitions,
                seed=config.seed + bigram_index,
            )
        row: dict[str, Any] = {
            "bigram": bigram,
            "session_count": int(mask.sum()),
            "event_count": int(session_index.loc[mask, "event_count"].sum()),
            "trajectory_magnitude": magnitude,
            "top6_projected_magnitude": (
                magnitude * float(curve[config.primary_components - 1])
            ),
            "top6_capture_ci_low": low,
            "top6_capture_ci_high": high,
            "top6_loso_session_correlation": _loso_correlation(
                per_session,
                pca,
                config.primary_components,
            ),
            "median_alignment_confidence": float(
                events.loc[
                    events.bigram.eq(bigram),
                    "alignment_confidence",
                ].median()
            ),
        }
        for component in config.reported_components:
            row[f"top{component}_trajectory_captured_fraction"] = float(
                curve[component - 1]
            )
        for component, fraction in enumerate(curve, start=1):
            row[f"k{component}_captured_fraction"] = float(fraction)
        if bigram in counts.index:
            row.update(counts.loc[bigram].to_dict())
        rows.append(row)
    primary_column = f"top{config.primary_components}_trajectory_captured_fraction"
    ranking = (
        pd.DataFrame(rows)
        .sort_values(
            primary_column,
            ascending=False,
        )
        .reset_index(drop=True)
    )
    ranking.insert(0, "rank", np.arange(1, len(ranking) + 1))
    return ranking, session_index, session_paths


def analyze_bigram_event_set(
    event_set: BigramEventSet,
    *,
    config: BigramTrajectoryConfig,
) -> BigramTrajectoryResult:
    """Fit both PCA conditions and calculate nominal and jitter rankings."""

    config.validate()
    nominal = event_set.paths_by_jitter[0]
    centered_events = nominal - nominal.mean(axis=1, keepdims=True)
    session_index, session_change = _session_means(
        centered_events,
        event_set.events,
    )
    state_index, session_state = _session_means(
        nominal,
        event_set.events,
    )
    if not state_index.equals(session_index):
        raise ValueError("Change and state session-trajectory indices disagree")
    change_pca = fit_equal_bigram_pca(
        session_change,
        session_index.bigram.tolist(),
        temporal_center=False,
    )
    state_pca = fit_equal_bigram_pca(
        session_state,
        session_index.bigram.tolist(),
        temporal_center=False,
    )
    variance_rows = []
    for condition, pca in (("change", change_pca), ("state", state_pca)):
        cumulative = np.cumsum(pca.explained_variance_ratio)
        if np.any(np.diff(cumulative) < -1e-12):
            raise ValueError("Cumulative PCA explained variance is not monotonic")
        for component, (value, cumulative_value) in enumerate(
            zip(pca.explained_variance_ratio, cumulative),
            start=1,
        ):
            variance_rows.append(
                {
                    "condition": condition,
                    "component": component,
                    "explained_variance_fraction": float(value),
                    "cumulative_explained_variance": float(cumulative_value),
                    "eigenvalue": float(pca.eigenvalues[component - 1]),
                }
            )
    ranking, session_index, session_change = rank_bigram_trajectories(
        event_set,
        change_pca,
        config=config,
    )
    if not session_index.equals(state_index):
        raise ValueError("Change and state session-trajectory indices disagree")
    jitter_frames = []
    primary_column = f"top{config.primary_components}_trajectory_captured_fraction"
    for jitter in config.jitter_bins:
        jitter_ranking, _, _ = rank_bigram_trajectories(
            event_set,
            change_pca,
            config=config,
            jitter=jitter,
            bootstrap=False,
        )
        jitter_ranking.insert(1, "jitter_bins", jitter)
        jitter_ranking.insert(2, "jitter_ms", jitter * 20)
        jitter_frames.append(
            jitter_ranking[
                [
                    "bigram",
                    "jitter_bins",
                    "jitter_ms",
                    "rank",
                    "event_count",
                    primary_column,
                ]
            ]
        )
    jitter_sensitivity = pd.concat(jitter_frames, ignore_index=True)
    ordered = ranking.bigram.tolist()
    pooled_change = np.stack(
        [
            session_change[session_index.bigram.eq(label)].mean(axis=0)
            for label in ordered
        ]
    )
    pooled_state = np.stack(
        [
            session_state[session_index.bigram.eq(label)].mean(axis=0)
            for label in ordered
        ]
    )
    return BigramTrajectoryResult(
        config=config,
        event_set=event_set,
        change_pca=change_pca,
        state_pca=state_pca,
        pca_variance=pd.DataFrame(variance_rows),
        ranking=ranking,
        jitter_sensitivity=jitter_sensitivity,
        session_trajectory_index=session_index,
        session_change_paths=session_change,
        session_state_paths=session_state,
        pooled_change_paths=pooled_change,
        pooled_state_paths=pooled_state,
    )


def run_bigram_trajectory_analysis(
    model_dir: str | Path,
    cache_root: str | Path,
    *,
    config: BigramTrajectoryConfig | None = None,
    progress: Callable[[str], None] | None = None,
) -> BigramTrajectoryResult:
    config = config or BigramTrajectoryConfig()
    event_set = build_bigram_event_set(
        model_dir,
        cache_root,
        config=config,
        progress=progress,
    )
    return analyze_bigram_event_set(event_set, config=config)


def _plot_scree(result: BigramTrajectoryResult) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for condition, frame in result.pca_variance.groupby("condition"):
        ax.plot(
            frame.component,
            frame.cumulative_explained_variance,
            label=condition,
        )
    ax.set(
        xlabel="PCA components",
        ylabel="Cumulative explained variance",
        xlim=(1, 128),
        ylim=(0, 1.01),
    )
    ax.legend()
    fig.tight_layout()
    return fig


def _plot_ranking(result: BigramTrajectoryResult) -> plt.Figure:
    column = f"top{result.config.primary_components}_trajectory_captured_fraction"
    frame = result.ranking.sort_values(column)
    fig, ax = plt.subplots(figsize=(9, 16))
    ax.barh(frame.bigram, frame[column], color="tab:blue")
    ax.set(
        xlabel=(
            "Mean-trajectory captured fraction in top "
            f"{result.config.primary_components} PCs"
        ),
        ylabel="Bigram",
        xlim=(0, 1),
    )
    fig.tight_layout()
    return fig


def _trajectory_pages(
    result: BigramTrajectoryResult,
    *,
    condition: str,
    per_page: int = 18,
) -> list[plt.Figure]:
    if condition == "change":
        source_paths = result.pooled_change_paths
        pca = result.change_pca
        temporal_center = True
    elif condition == "state":
        source_paths = result.pooled_state_paths
        pca = result.state_pca
        temporal_center = False
    else:
        raise ValueError("condition must be 'change' or 'state'")
    projected = project_paths(
        source_paths,
        pca,
        temporal_center=temporal_center,
    )
    limit = float(np.nanpercentile(np.abs(projected[:, :, :2]), 99))
    limit = max(limit, 1e-6)
    figures = []
    for start in range(0, len(projected), per_page):
        stop = min(len(projected), start + per_page)
        fig, axes = plt.subplots(3, 6, figsize=(18, 9), squeeze=False)
        for ax in axes.ravel():
            ax.set_visible(False)
        points = None
        rows = result.ranking.iloc[start:stop].itertuples(index=False)
        for ax, path, row in zip(axes.ravel(), projected[start:stop], rows):
            ax.set_visible(True)
            ax.plot(path[:, 0], path[:, 1], color="0.7", linewidth=0.8)
            points = ax.scatter(
                path[:, 0],
                path[:, 1],
                c=result.event_set.time_ms,
                cmap="coolwarm",
                s=15,
            )
            zero = int(result.config.before_bins)
            ax.scatter(
                path[zero, 0],
                path[zero, 1],
                marker="x",
                color="black",
                s=35,
            )
            ax.set(
                title=f"#{row.rank} {row.bigram}",
                xlim=(-limit, limit),
                ylim=(-limit, limit),
                xlabel="PC1",
                ylabel="PC2",
            )
        if points is not None:
            fig.colorbar(
                points,
                ax=axes.ravel().tolist(),
                label="Time from transition (ms)",
                shrink=0.7,
            )
        fig.suptitle(f"Mean {condition} trajectories ranked {start + 1}–{stop}")
        fig.subplots_adjust(top=0.92, wspace=0.35, hspace=0.45)
        figures.append(fig)
    return figures


def make_bigram_trajectory_figures(
    result: BigramTrajectoryResult,
) -> dict[str, plt.Figure]:
    """Build the scree, ranking, and paginated change/state trajectory plots."""

    figures = {
        "pca_scree.png": _plot_scree(result),
        "bigram_top6_ranking.png": _plot_ranking(result),
    }
    for condition in ("change", "state"):
        pages = _trajectory_pages(result, condition=condition)
        for index, figure in enumerate(pages, start=1):
            figures[f"{condition}_trajectory_grid_{index:02d}.png"] = figure
    return figures


def save_bigram_trajectory_result(
    result: BigramTrajectoryResult,
    output_dir: str | Path,
    *,
    git_commit: str,
    overwrite: bool = False,
) -> dict[str, str]:
    """Stage, validate, and atomically promote one complete analysis."""

    destination = Path(output_dir)
    if destination.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing analysis: {destination}")
    staging = destination.parent / f".{destination.name}.staging-{uuid.uuid4().hex}"
    backup: Path | None = None
    staging.mkdir(parents=True)
    try:
        result.event_set.counts.to_csv(
            staging / "bigram_counts.csv",
            index=False,
        )
        result.event_set.events.to_csv(
            staging / "event_manifest.csv",
            index=False,
        )
        result.event_set.diagnostics.to_csv(
            staging / "example_diagnostics.csv",
            index=False,
        )
        result.event_set.exclusions.to_csv(
            staging / "excluded_events.csv",
            index=False,
        )
        result.event_set.session_statistics.to_csv(
            staging / "session_statistics.csv",
            index=False,
        )
        result.pca_variance.to_csv(
            staging / "pca_variance.csv",
            index=False,
        )
        result.ranking.to_csv(
            staging / "bigram_ranking.csv",
            index=False,
        )
        result.jitter_sensitivity.to_csv(
            staging / "jitter_sensitivity.csv",
            index=False,
        )
        result.session_trajectory_index.to_csv(
            staging / "session_trajectory_index.csv",
            index=False,
        )
        session_ids = sorted(result.event_set.session_means)
        np.savez_compressed(
            staging / "session_sufficient_statistics.npz",
            session_ids=np.asarray(session_ids, dtype=str),
            counts=result.event_set.session_statistics.set_index("session_id")
            .loc[session_ids, "n_bins"]
            .to_numpy(dtype=np.int64),
            means=np.stack(
                [result.event_set.session_means[key] for key in session_ids]
            ),
            stds=np.stack([result.event_set.session_stds[key] for key in session_ids]),
        )
        np.savez_compressed(
            staging / "pca_parameters.npz",
            change_mean=result.change_pca.mean,
            change_covariance=result.change_pca.covariance,
            change_components=result.change_pca.components,
            change_eigenvalues=result.change_pca.eigenvalues,
            state_mean=result.state_pca.mean,
            state_covariance=result.state_pca.covariance,
            state_components=result.state_pca.components,
            state_eigenvalues=result.state_pca.eigenvalues,
        )
        change_projected = project_paths(
            result.session_change_paths,
            result.change_pca,
            temporal_center=True,
        )[:, :, :12]
        state_projected = project_paths(
            result.session_state_paths,
            result.state_pca,
            temporal_center=False,
        )[:, :, :12]
        np.savez_compressed(
            staging / "mean_trajectories.npz",
            time_ms=result.event_set.time_ms,
            session_change_128=result.session_change_paths,
            session_state_128=result.session_state_paths,
            session_change_pc12=change_projected,
            session_state_pc12=state_projected,
            pooled_change_128=result.pooled_change_paths,
            pooled_state_128=result.pooled_state_paths,
        )
        figures = make_bigram_trajectory_figures(result)
        for name, figure in figures.items():
            figure.savefig(
                staging / name,
                dpi=180,
                bbox_inches="tight",
            )
            plt.close(figure)
        config_payload = {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "git_commit": str(git_commit),
            "config": asdict(result.config),
            "source": result.event_set.metadata,
            "analysis_scope": "all-24-session descriptive/transductive",
            "timing_caveat": (
                "20 ms SBP sampling does not improve the GRU alignment's "
                "native 80 ms timing precision"
            ),
            "provenance": (
                "The frozen local GRU is an LLM-assisted Willett-style "
                "adaptation with unresolved exact upstream source and licensing; "
                "not an official Stanford implementation."
            ),
            "ai_assistance": (
                "Codex implemented the bigram alignment, raw-SBP PCA, ranking, "
                "plotting, serialization, and validation workflow; human review "
                "is required."
            ),
        }
        (staging / "config.json").write_text(
            json.dumps(config_payload, indent=2, default=str)
        )
        primary_column = (
            f"top{result.config.primary_components}_trajectory_captured_fraction"
        )
        summary = {
            "status": ("smoke_engineering_only" if result.config.smoke else "complete"),
            "example_count": int(len(result.event_set.diagnostics)),
            "session_count": int(result.event_set.events.session_id.nunique()),
            "bigram_count": int(len(result.ranking)),
            "event_count": int(len(result.event_set.events)),
            "top_ranked_bigram": str(result.ranking.iloc[0].bigram),
            "top_ranked_top6_trajectory_captured_fraction": float(
                result.ranking.iloc[0][primary_column]
            ),
        }
        (staging / "summary.json").write_text(json.dumps(summary, indent=2))
        required = {
            "config.json",
            "summary.json",
            "bigram_counts.csv",
            "event_manifest.csv",
            "example_diagnostics.csv",
            "excluded_events.csv",
            "session_statistics.csv",
            "pca_variance.csv",
            "bigram_ranking.csv",
            "jitter_sensitivity.csv",
            "session_trajectory_index.csv",
            "session_sufficient_statistics.npz",
            "pca_parameters.npz",
            "mean_trajectories.npz",
            "pca_scree.png",
            "bigram_top6_ranking.png",
        }
        required.update(figures)
        missing = sorted(name for name in required if not (staging / name).exists())
        if missing:
            raise FileNotFoundError(f"Missing staged analysis artifacts: {missing}")
        for name in (
            "bigram_counts.csv",
            "event_manifest.csv",
            "example_diagnostics.csv",
            "excluded_events.csv",
            "session_statistics.csv",
            "pca_variance.csv",
            "bigram_ranking.csv",
            "jitter_sensitivity.csv",
            "session_trajectory_index.csv",
        ):
            pd.read_csv(staging / name)
        json.loads((staging / "config.json").read_text())
        json.loads((staging / "summary.json").read_text())
        for name in figures:
            image = plt.imread(staging / name)
            if image.size == 0 or not np.isfinite(image).all():
                raise ValueError(f"Reopened figure is invalid: {name}")
        reopened_ranking = pd.read_csv(staging / "bigram_ranking.csv")
        if (
            len(reopened_ranking) != len(result.ranking)
            or reopened_ranking.bigram.tolist() != result.ranking.bigram.tolist()
        ):
            raise ValueError(
                "Reopened bigram ranking disagrees with the in-memory result"
            )
        with np.load(staging / "pca_parameters.npz") as arrays:
            if (
                arrays["change_components"].shape != (128, 128)
                or not np.isfinite(arrays["change_components"]).all()
            ):
                raise ValueError("Reopened PCA parameters are invalid")
        with np.load(staging / "mean_trajectories.npz") as arrays:
            if arrays["pooled_change_128"].shape != result.pooled_change_paths.shape:
                raise ValueError("Reopened pooled trajectories have the wrong shape")
        marker = {
            "status": "complete",
            "analysis_status": summary["status"],
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "output_dir": str(destination),
        }
        (staging / "_SUCCESS.json").write_text(json.dumps(marker, indent=2))
        if destination.exists():
            backup = (
                destination.parent / f".{destination.name}.backup-{uuid.uuid4().hex}"
            )
            destination.rename(backup)
        staging.rename(destination)
        if backup is not None:
            shutil.rmtree(backup)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        if backup is not None and backup.exists() and not destination.exists():
            backup.rename(destination)
        raise
    return {
        "output_dir": str(destination),
        "summary": str(destination / "summary.json"),
    }


__all__ = [
    "BigramEventSet",
    "BigramSourceBundle",
    "BigramTrajectoryConfig",
    "BigramTrajectoryResult",
    "CovariancePCA",
    "alignment_confidence",
    "analyze_bigram_event_set",
    "build_bigram_event_set",
    "count_reference_bigrams",
    "extract_transition_window",
    "fit_equal_bigram_pca",
    "make_bigram_trajectory_figures",
    "normalize_paths_by_session",
    "prepare_bigram_sources",
    "project_paths",
    "rank_bigram_trajectories",
    "run_bigram_trajectory_analysis",
    "save_bigram_trajectory_result",
    "transition_anchor_bin",
]

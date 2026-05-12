"""Build the canonical Brain2Text24 cache under cache_v1.

This converter ingests the raw 2024 MATLAB release and writes canonical shards
(`tx.npy`, `sbp.npy`, `time_offsets.npy`) plus optional transcript-derived
phoneme targets (`phoneme_ids.npy`, `phoneme_offsets.npy`) for probe use.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import scipy.io as sio

from build_probe_manifest import LOGIT_TO_PHONEME
from prepare import CACHE_ROOT, BRAINTOTEXT24_ROOT, relative_to_root, source_root_metadata


CACHE_VERSION = "v1"
BIN_SIZE_MS = 20
N_FEATURES = 256
TX_VARIANT = "tx2"
DEFAULT_DATASET_FAMILY = "brain2text24"
RAW_TASK_DIRS = ("sentences", "diagnosticBlocks", "tuningTasks")
COMPETITION_SPLIT_DIRS = {
    "train": "competition_train",
    "test": "competition_test",
    "competitionHoldOut": "competition_holdout",
}
LOAD_VARIABLES_BY_GROUP = {
    "sentences": [
        TX_VARIANT,
        "spikePow",
        "goTrialEpochs",
        "delayTrialEpochs",
        "blockNum",
        "blockList",
        "blockTypes",
        "sentences",
        "sentenceDurations",
        "ngramFinalOutput",
        "speakingMode",
    ],
    "diagnosticBlocks": [
        TX_VARIANT,
        "spikePow",
        "goTrialEpochs",
        "delayTrialEpochs",
        "blockNum",
        "blockList",
        "trialDelayTimes",
        "trialCues",
        "cueList",
    ],
    "tuningTasks": [
        TX_VARIANT,
        "spikePow",
        "goTrialEpochs",
        "delayTrialEpochs",
        "blockNum",
        "blockList",
        "trialDelayTimes",
        "trialCues",
        "cueList",
    ],
}

PHONEME_TO_INDEX = {symbol: idx for idx, symbol in enumerate(LOGIT_TO_PHONEME)}


@dataclass(frozen=True)
class FileHeader:
    source_path: Path
    source_relpath: str
    source_group: str
    source_filename: str
    session_id: str
    session_date: str | None
    subject_id: str
    task_family: str
    task_name: str


def _remove_punctuation(text: str) -> str:
    text = re.sub(r"[^a-zA-Z\- ']", "", text)
    text = text.replace("--", "")
    text = text.replace(" '", "'")
    text = text.strip().lower()
    return " ".join(text.split())


def _build_transcript_phonemizer() -> Any | None:
    try:
        from g2p_en import G2p  # type: ignore
    except ImportError:
        return None
    return G2p()


def _transcript_to_phoneme_ids(transcript: str, g2p_instance: Any | None) -> list[int]:
    if g2p_instance is None:
        return []
    if not transcript or not transcript.strip():
        return []

    cleaned = _remove_punctuation(transcript)
    if not cleaned:
        return []

    symbols: list[str] = []
    for token in g2p_instance(cleaned):
        if token == " ":
            symbols.append("SIL")
            continue
        token = re.sub(r"[0-9]", "", str(token)).upper()
        if token in PHONEME_TO_INDEX:
            symbols.append(token)
    symbols.append("SIL")
    return [PHONEME_TO_INDEX[symbol] for symbol in symbols]


def _iter_source_files(root: Path) -> list[Path]:
    out: list[Path] = []
    for group in RAW_TASK_DIRS:
        group_root = root / group
        if not group_root.exists():
            continue
        for path in sorted(group_root.glob("*.mat")):
            if path.name.startswith("._"):
                continue
            out.append(path)
    return out


def _session_date(session_id: str) -> str | None:
    parts = session_id.split(".")
    if len(parts) >= 4:
        return ".".join(parts[1:4])
    return None


def _subject_id(session_id: str) -> str:
    return session_id.split(".", 1)[0]


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return ""
        if value.ndim == 0:
            return _normalize_text(value.item())
        return _normalize_text(value.flat[0])
    if isinstance(value, (list, tuple)):
        if not value:
            return ""
        return _normalize_text(value[0])
    return str(value).strip()


def _text_list(value: Any, expected_len: int | None = None) -> list[str]:
    if value is None:
        return [""] * expected_len if expected_len is not None else []
    arr = np.asarray(value, dtype=object).reshape(-1)
    out = [_normalize_text(item) for item in arr.tolist()]
    if expected_len is not None and len(out) < expected_len:
        out.extend([""] * (expected_len - len(out)))
    return out


def _parse_header(path: Path) -> FileHeader:
    if "_" not in path.stem:
        raise ValueError(f"Cannot parse Brain2Text24 file stem: {path.stem}")
    session_id, task_name = path.stem.split("_", 1)
    task_family = {
        "sentences": "sentence_production",
        "diagnosticBlocks": "diagnostic_blocks",
        "tuningTasks": "tuning_tasks",
    }[path.parent.name]
    return FileHeader(
        source_path=path,
        source_relpath=relative_to_root(path, "brain2text24_root"),
        source_group=path.parent.name,
        source_filename=path.name,
        session_id=session_id,
        session_date=_session_date(session_id),
        subject_id=_subject_id(session_id),
        task_family=task_family,
        task_name=task_name,
    )


def _load_source_mat(path: Path) -> dict[str, Any]:
    variable_names = LOAD_VARIABLES_BY_GROUP[path.parent.name]
    return sio.loadmat(path, variable_names=variable_names, simplify_cells=True)


def _competition_split_map(root: Path) -> dict[str, dict[int, str]]:
    split_map: dict[str, dict[int, str]] = {}
    comp_root = root / "competitionData"
    if not comp_root.exists():
        return split_map

    for split_dir_name, split_label in COMPETITION_SPLIT_DIRS.items():
        split_dir = comp_root / split_dir_name
        if not split_dir.exists():
            continue
        for path in sorted(split_dir.glob("*.mat")):
            session_id = path.stem
            mat = sio.loadmat(path, variable_names=["blockIdx"], simplify_cells=True)
            block_values = np.asarray(mat["blockIdx"]).reshape(-1)
            session_map = split_map.setdefault(session_id, {})
            for block_num in np.unique(block_values).tolist():
                block_num = int(block_num)
                existing = session_map.get(block_num)
                if existing is not None and existing != split_label:
                    raise ValueError(
                        f"Competition split conflict for {session_id} block {block_num}: "
                        f"{existing} vs {split_label}"
                    )
                session_map[block_num] = split_label
    return split_map


def _extract_epoch_slice(array: np.ndarray, epoch: np.ndarray) -> np.ndarray:
    start = int(epoch[0])
    end = int(epoch[1])
    if start <= 0 or end < start:
        raise ValueError(f"Invalid MATLAB epoch indices: {epoch}")
    return np.asarray(array[start - 1 : end])


def _block_type_map(block_list: Any, block_types: Any) -> dict[int, str]:
    if block_list is None or block_types is None:
        return {}
    blocks = np.asarray(block_list).reshape(-1).tolist()
    types = _text_list(block_types, expected_len=len(blocks))
    return {int(block): block_type for block, block_type in zip(blocks, types, strict=False)}


def _cue_metadata(
    *,
    example_index: int,
    trial_cues: np.ndarray | None,
    cue_list: list[str],
) -> tuple[int | None, str]:
    if trial_cues is None or example_index >= len(trial_cues):
        return None, ""
    cue_index = int(trial_cues[example_index])
    if cue_index <= 0 or cue_index > len(cue_list):
        return cue_index, ""
    return cue_index, cue_list[cue_index - 1]


def _write_file_shard(
    header: FileHeader,
    mat: dict[str, Any],
    *,
    shard_dir: Path,
    shard_relpath: str,
    sentence_split_map: dict[int, str] | None,
    dataset_family: str,
    transcript_phonemizer: Any | None,
) -> tuple[list[dict[str, Any]], dict[str, Any], int, int]:
    tx_all = np.asarray(mat[TX_VARIANT])
    sbp_all = np.asarray(mat["spikePow"], dtype=np.float32)
    go_epochs = np.asarray(mat["goTrialEpochs"], dtype=np.int64)
    delay_epochs = np.asarray(mat["delayTrialEpochs"], dtype=np.int64) if "delayTrialEpochs" in mat else None
    block_num_all = np.asarray(mat["blockNum"]).reshape(-1) if "blockNum" in mat else None
    block_types = _block_type_map(mat.get("blockList"), mat.get("blockTypes"))
    trial_cues = np.asarray(mat["trialCues"]).reshape(-1) if "trialCues" in mat else None
    cue_list = _text_list(mat.get("cueList"))
    sentences = _text_list(mat.get("sentences"), expected_len=go_epochs.shape[0])
    sentence_durations = np.asarray(mat["sentenceDurations"]).reshape(-1) if "sentenceDurations" in mat else None
    realtime_outputs = _text_list(mat.get("ngramFinalOutput"), expected_len=go_epochs.shape[0])
    speaking_mode = _normalize_text(mat.get("speakingMode"))
    trial_delay_times = np.asarray(mat["trialDelayTimes"]).reshape(-1) if "trialDelayTimes" in mat else None

    if tx_all.ndim != 2 or tx_all.shape[1] != N_FEATURES:
        raise ValueError(
            f"{header.source_filename} expected {TX_VARIANT} shape (*, {N_FEATURES}), got {tx_all.shape}"
        )
    if sbp_all.ndim != 2 or sbp_all.shape != tx_all.shape:
        raise ValueError(
            f"{header.source_filename} expected spikePow shape {tx_all.shape}, got {sbp_all.shape}"
        )

    tx_examples: list[np.ndarray] = []
    sbp_examples: list[np.ndarray] = []
    phoneme_ids_flat: list[int] = []
    manifest_rows: list[dict[str, Any]] = []
    time_offsets_list = [0]
    phoneme_offsets_list = [0]
    total_time = 0
    labeled_examples = 0

    for example_index, epoch in enumerate(go_epochs):
        tx_trial = _extract_epoch_slice(tx_all, epoch).astype(np.uint8, copy=False)
        sbp_trial = _extract_epoch_slice(sbp_all, epoch).astype(np.float32, copy=False)
        if tx_trial.shape != sbp_trial.shape:
            raise ValueError(
                f"{header.source_filename} trial {example_index} TX/SBP mismatch: "
                f"{tx_trial.shape} vs {sbp_trial.shape}"
            )
        if tx_trial.shape[0] < 2:
            continue

        block_num = int(block_num_all[int(epoch[0]) - 1]) if block_num_all is not None else None
        cue_index, cue_text = _cue_metadata(
            example_index=example_index,
            trial_cues=trial_cues,
            cue_list=cue_list,
        )
        sentence_text = sentences[example_index] if example_index < len(sentences) else ""
        transcript = sentence_text or cue_text
        source_split = "none"
        if header.source_group == "sentences" and block_num is not None and sentence_split_map is not None:
            source_split = sentence_split_map.get(block_num, "none")

        delay_time_bins = None
        if delay_epochs is not None and example_index < delay_epochs.shape[0]:
            delay_epoch = delay_epochs[example_index]
            delay_time_bins = int(delay_epoch[1] - delay_epoch[0] + 1)

        sentence_duration_bins = None
        if sentence_durations is not None and example_index < len(sentence_durations):
            sentence_duration_bins = int(sentence_durations[example_index])

        trial_delay_time_bins = None
        if trial_delay_times is not None and example_index < len(trial_delay_times):
            trial_delay_time_bins = int(trial_delay_times[example_index])

        label_ids = _transcript_to_phoneme_ids(transcript, transcript_phonemizer)
        has_labels = len(label_ids) > 0
        if has_labels:
            labeled_examples += 1
            phoneme_ids_flat.extend(label_ids)

        tx_examples.append(tx_trial)
        sbp_examples.append(sbp_trial)
        total_time += int(tx_trial.shape[0])
        time_offsets_list.append(total_time)
        phoneme_offsets_list.append(len(phoneme_ids_flat))

        manifest_rows.append(
            {
                "example_id": f"{header.source_group}__{header.source_path.stem}__trial{example_index:04d}",
                "dataset_family": dataset_family,
                "subject_id": header.subject_id,
                "session_id": header.session_id,
                "session_date": header.session_date,
                "source_split": source_split,
                "example_type": "trial",
                "task_family": header.task_family,
                "task_name": header.task_name,
                "source_group": header.source_group,
                "bin_size_ms": BIN_SIZE_MS,
                "source_bin_size_ms": BIN_SIZE_MS,
                "resampled_to_20ms": False,
                "has_tx": True,
                "has_sbp": True,
                "n_time_bins": int(tx_trial.shape[0]),
                "n_time_bins_native": int(tx_trial.shape[0]),
                "target_type": "phoneme_ctc" if has_labels else "none",
                "has_labels": has_labels,
                "normalization_group": header.session_id,
                "shard_id": shard_dir.name,
                "shard_relpath": shard_relpath,
                "example_index": len(manifest_rows),
                "block_num": block_num,
                "trial_num": example_index,
                "trial_key": f"trial_{example_index:04d}",
                "target_length": len(label_ids) if has_labels else None,
                "transcript": transcript,
                "sentence_label": sentence_text,
                "feature_modalities": "tx+sbp",
                "n_tx_features": int(tx_trial.shape[1]),
                "n_sbp_features": int(sbp_trial.shape[1]),
                "source_root_key": "brain2text24_root",
                "source_relpath": header.source_relpath,
                "cache_root_key": "canonical_cache_root",
                "cache_dataset_relpath": dataset_family,
                "cue_index": cue_index,
                "cue_text": cue_text,
                "sentence_text": sentence_text,
                "realtime_decoded_text": realtime_outputs[example_index] if example_index < len(realtime_outputs) else "",
                "speaking_mode": speaking_mode,
                "block_type": block_types.get(block_num, "") if block_num is not None else "",
                "delay_time_bins": delay_time_bins,
                "sentence_duration_bins": sentence_duration_bins,
                "trial_delay_time_bins": trial_delay_time_bins,
                "go_epoch_start_1based": int(epoch[0]),
                "go_epoch_stop_1based": int(epoch[1]),
                "source_filename": header.source_filename,
                "tx_variant": TX_VARIANT,
            }
        )

    if not manifest_rows:
        return [], {
            "shard_id": shard_dir.name,
            "shard_relpath": shard_relpath,
            "session_id": header.session_id,
            "subject_id": header.subject_id,
            "task_family": header.task_family,
            "task_name": header.task_name,
            "source_group": header.source_group,
            "source_filename": header.source_filename,
            "source_split": "none",
            "example_count": 0,
            "labeled_example_count": 0,
            "total_time_bins": 0,
            "total_targets": 0,
            "n_tx_features": 0,
            "n_sbp_features": 0,
        }, 0, 0

    time_offsets = np.asarray(time_offsets_list, dtype=np.int64)
    phoneme_offsets = np.asarray(phoneme_offsets_list, dtype=np.int64)
    phoneme_ids = np.asarray(phoneme_ids_flat, dtype=np.int32)

    shard_dir.mkdir(parents=True, exist_ok=True)
    np.save(shard_dir / "time_offsets.npy", time_offsets)
    np.save(shard_dir / "tx.npy", np.concatenate(tx_examples, axis=0).astype(np.uint8, copy=False))
    np.save(shard_dir / "sbp.npy", np.concatenate(sbp_examples, axis=0).astype(np.float32, copy=False))
    np.save(shard_dir / "phoneme_offsets.npy", phoneme_offsets)
    np.save(shard_dir / "phoneme_ids.npy", phoneme_ids)

    source_split = manifest_rows[0]["source_split"]
    if header.source_group == "sentences":
        source_split = "mixed" if len({row["source_split"] for row in manifest_rows}) > 1 else source_split

    shard_meta = {
        "shard_id": shard_dir.name,
        "shard_relpath": shard_relpath,
        "session_id": header.session_id,
        "subject_id": header.subject_id,
        "task_family": header.task_family,
        "task_name": header.task_name,
        "source_group": header.source_group,
        "source_filename": header.source_filename,
        "source_split": source_split,
        "example_count": len(manifest_rows),
        "labeled_example_count": labeled_examples,
        "total_time_bins": int(time_offsets[-1]),
        "total_targets": int(phoneme_offsets[-1]),
        "n_tx_features": int(tx_examples[0].shape[1]),
        "n_sbp_features": int(sbp_examples[0].shape[1]),
    }
    return manifest_rows, shard_meta, labeled_examples, int(phoneme_offsets[-1])


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete an existing destination dataset directory before rebuilding.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Optional smoke-test limit on the number of raw .mat files to convert.",
    )
    parser.add_argument(
        "--dataset-family",
        type=str,
        default=DEFAULT_DATASET_FAMILY,
        help=f"Destination dataset family under cache_v1 (default: {DEFAULT_DATASET_FAMILY}).",
    )
    parser.add_argument(
        "--disable-phoneme-labels",
        action="store_true",
        help="Skip transcript->phoneme conversion and write unlabeled examples only.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_argparser().parse_args(argv)

    dataset_family = str(args.dataset_family).strip()
    if not dataset_family:
        raise SystemExit("--dataset-family cannot be empty.")

    if not BRAINTOTEXT24_ROOT.exists():
        raise SystemExit(
            f"Brain2Text24 source root does not exist: {BRAINTOTEXT24_ROOT}. "
            "Set SSL_AUTORESEARCH_B2T24_ROOT to the mounted raw dataset path."
        )

    source_files = _iter_source_files(BRAINTOTEXT24_ROOT)
    if args.max_files is not None:
        source_files = source_files[: args.max_files]
    if len(source_files) == 0:
        raise SystemExit(
            "No Brain2Text24 source .mat files were found under "
            f"{BRAINTOTEXT24_ROOT}. Check the source root and folder layout."
        )

    transcript_phonemizer = None if args.disable_phoneme_labels else _build_transcript_phonemizer()
    if not args.disable_phoneme_labels and transcript_phonemizer is None:
        raise SystemExit(
            "g2p_en is required for transcript-derived phoneme labels. "
            "Install it (e.g. `pip install g2p_en`) or pass --disable-phoneme-labels."
        )

    dataset_root = CACHE_ROOT / dataset_family
    shard_root = dataset_root / "shards"
    manifest_path = dataset_root / "manifest.jsonl"
    metadata_path = dataset_root / "metadata.json"

    if dataset_root.exists():
        if not args.overwrite:
            raise SystemExit(
                f"{dataset_root} already exists. Re-run with --overwrite to rebuild the cache."
            )
        shutil.rmtree(dataset_root)

    headers = [_parse_header(path) for path in source_files]
    competition_splits = _competition_split_map(BRAINTOTEXT24_ROOT)

    dataset_root.mkdir(parents=True, exist_ok=True)
    shard_root.mkdir(parents=True, exist_ok=True)

    total_examples = 0
    labeled_examples = 0
    total_time_bins = 0
    total_targets = 0
    session_ids: set[str] = set()
    subject_ids: set[str] = set()
    source_group_counts: dict[str, int] = {}
    source_split_counts: dict[str, int] = {}
    labeled_split_counts: dict[str, int] = {}
    shard_rows: list[dict[str, Any]] = []

    with manifest_path.open("w") as manifest_handle:
        for header in headers:
            mat = _load_source_mat(header.source_path)
            shard_id = header.source_path.stem
            shard_relpath = f"{dataset_family}/shards/{shard_id}"
            rows, shard_meta, shard_labeled, shard_targets = _write_file_shard(
                header,
                mat,
                shard_dir=shard_root / shard_id,
                shard_relpath=shard_relpath,
                sentence_split_map=competition_splits.get(header.session_id),
                dataset_family=dataset_family,
                transcript_phonemizer=transcript_phonemizer,
            )
            if not rows:
                continue

            for row in rows:
                manifest_handle.write(json.dumps(row) + "\n")
                source_group_counts[row["source_group"]] = source_group_counts.get(row["source_group"], 0) + 1
                source_split_counts[row["source_split"]] = source_split_counts.get(row["source_split"], 0) + 1
                if bool(row["has_labels"]):
                    labeled_split_counts[row["source_split"]] = labeled_split_counts.get(row["source_split"], 0) + 1

            shard_rows.append(shard_meta)
            total_examples += len(rows)
            labeled_examples += shard_labeled
            total_time_bins += shard_meta["total_time_bins"]
            total_targets += shard_targets
            session_ids.add(header.session_id)
            subject_ids.add(header.subject_id)

    if labeled_examples > 0 and labeled_examples < total_examples:
        target_type = "mixed"
    elif labeled_examples == total_examples and total_examples > 0:
        target_type = "phoneme_ctc"
    else:
        target_type = "none"

    metadata = {
        "dataset_family": dataset_family,
        "cache_version": CACHE_VERSION,
        "cache_root_key": "canonical_cache_root",
        "cache_dataset_relpath": dataset_family,
        "bin_size_ms": BIN_SIZE_MS,
        "modalities": ["tx", "sbp"],
        "feature_layout": {
            "n_total_features": 512,
            "n_tx_features": N_FEATURES,
            "n_sbp_features": N_FEATURES,
            "tx_variant": TX_VARIANT,
        },
        "num_subjects": len(subject_ids),
        "num_sessions": len(session_ids),
        "num_shards": len(shard_rows),
        "total_examples": total_examples,
        "labeled_examples": labeled_examples,
        "total_time_bins": total_time_bins,
        "total_targets": total_targets,
        "target_type": target_type,
        "source_group_counts": source_group_counts,
        "source_split_counts": source_split_counts,
        "labeled_split_counts": labeled_split_counts,
        "phoneme_vocabulary": {
            "index_to_symbol": list(LOGIT_TO_PHONEME),
            "num_classes": len(LOGIT_TO_PHONEME),
            "blank_index": 0,
            "sil_index": len(LOGIT_TO_PHONEME) - 1,
        },
        "source_provenance": source_root_metadata(),
        "source_root_key": "brain2text24_root",
        "source_relpath_base": ".",
        "build_notes": [
            "Canonical Brain2Text24 cache built from raw MATLAB task exports.",
            "Only sentences, diagnosticBlocks, and tuningTasks are cached as source data.",
            "competitionData is used only to recover sentence block split annotations.",
            "Utterance-level trial boundaries are preserved via time_offsets.npy.",
            "Threshold crossings and spike band power are stored as separate arrays.",
            (
                "Phoneme labels were derived from transcripts with g2p_en and stored via "
                "phoneme_offsets.npy + phoneme_ids.npy."
                if transcript_phonemizer is not None
                else "Phoneme labels were disabled with --disable-phoneme-labels."
            ),
            "Normalization and SSL windowing are intentionally deferred to loader-time.",
        ],
        "shards": shard_rows,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2))

    print(f"cache_root: {dataset_root}")
    print(f"manifest_path: {manifest_path}")
    print(f"metadata_path: {metadata_path}")
    print(f"num_subjects: {len(subject_ids)}")
    print(f"num_sessions: {len(session_ids)}")
    print(f"num_shards: {len(shard_rows)}")
    print(f"total_examples: {total_examples}")
    print(f"labeled_examples: {labeled_examples}")
    print(f"total_time_bins: {total_time_bins}")
    print(f"total_targets: {total_targets}")
    print(f"target_type: {target_type}")
    print(f"source_group_counts: {source_group_counts}")
    print(f"source_split_counts: {source_split_counts}")
    print(f"labeled_split_counts: {labeled_split_counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

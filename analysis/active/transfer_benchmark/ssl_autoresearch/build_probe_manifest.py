"""Build a descriptive Brain2Text25 phoneme-probe manifest.

This script intentionally does not define the final benchmark split.
It inventories all available utterance-level trials from the official
`data_train.hdf5`, `data_val.hdf5`, and `data_test.hdf5` files so later
benchmark code can freeze splits without reparsing the raw HDF5 structure.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import h5py

from prepare import BRAINTOTEXT25_HDF5_ROOT, ensure_artifact_dirs, relative_to_root, source_root_metadata


LOGIT_TO_PHONEME = (
    "BLANK",
    "AA",
    "AE",
    "AH",
    "AO",
    "AW",
    "AY",
    "B",
    "CH",
    "D",
    "DH",
    "EH",
    "ER",
    "EY",
    "F",
    "G",
    "HH",
    "IH",
    "IY",
    "JH",
    "K",
    "L",
    "M",
    "N",
    "NG",
    "OW",
    "OY",
    "P",
    "R",
    "S",
    "SH",
    "T",
    "TH",
    "UH",
    "UW",
    "V",
    "W",
    "Y",
    "Z",
    "ZH",
    "SIL",
)


def _decode_scalar(value: Any) -> Any:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value


def _decode_transcription(codes: list[int]) -> str:
    out = []
    for code in codes:
        if code == 0:
            break
        out.append(chr(int(code)))
    return "".join(out)


def _date_from_session(session: str) -> str | None:
    parts = session.split(".")
    if len(parts) >= 4:
        return ".".join(parts[1:4])
    return None


def _split_name_from_file(path: Path) -> str:
    stem = path.stem
    if stem.startswith("data_"):
        return stem.replace("data_", "", 1)
    return stem


def _iter_hdf5_files(root: Path) -> list[Path]:
    out: list[Path] = []
    for session_dir in sorted(root.iterdir()):
        if not session_dir.is_dir():
            continue
        for name in ("data_train.hdf5", "data_val.hdf5", "data_test.hdf5"):
            path = session_dir / name
            if path.exists():
                out.append(path)
    return out


def _build_trial_record(file_path: Path, trial_key: str, group: h5py.Group) -> dict[str, Any]:
    session = str(_decode_scalar(group.attrs["session"]))
    sentence_label = _decode_scalar(group.attrs.get("sentence_label"))
    if sentence_label is None:
        sentence_label = ""
    input_features = group["input_features"]
    n_time_steps = int(_decode_scalar(group.attrs["n_time_steps"]))
    has_labels = "seq_class_ids" in group and "seq_len" in group.attrs
    if has_labels:
        seq_len = int(_decode_scalar(group.attrs["seq_len"]))
        phoneme_ids = [int(x) for x in group["seq_class_ids"][:seq_len].tolist()]
        phoneme_symbols = [
            LOGIT_TO_PHONEME[idx] if 0 <= idx < len(LOGIT_TO_PHONEME) else f"UNK_{idx}"
            for idx in phoneme_ids
        ]
    else:
        seq_len = None
        phoneme_ids = None
        phoneme_symbols = None
    if "transcription" in group:
        transcription_codes = [int(x) for x in group["transcription"][:].tolist()]
        transcription = _decode_transcription(transcription_codes)
    else:
        transcription = ""
    block_num = int(_decode_scalar(group.attrs["block_num"]))
    trial_num = int(_decode_scalar(group.attrs["trial_num"]))
    source_split = _split_name_from_file(file_path)
    session_date = _date_from_session(session)
    example_id = f"{session}__{source_split}__block{block_num:04d}__trial{trial_num:04d}"

    return {
        "example_id": example_id,
        "dataset_family": "brain2text25",
        "has_labels": has_labels,
        "session_id": session,
        "session_date": session_date,
        "source_split": source_split,
        "source_root_key": "brain2text25_hdf5",
        "source_relpath": relative_to_root(file_path, "brain2text25_hdf5"),
        "trial_key": trial_key,
        "block_num": block_num,
        "trial_num": trial_num,
        "feature_modalities": "tx+sbp",
        "bin_size_ms": 20,
        "n_time_bins": n_time_steps,
        "n_features": int(input_features.shape[1]),
        "phoneme_ids": phoneme_ids,
        "phoneme_symbols": phoneme_symbols,
        "target_length": seq_len,
        "transcription": transcription,
        "sentence_label": sentence_label,
        "normalization_group": session,
    }


def main() -> int:
    artifacts = ensure_artifact_dirs()
    hdf5_root = BRAINTOTEXT25_HDF5_ROOT
    hdf5_files = _iter_hdf5_files(hdf5_root)

    manifest_path = artifacts.manifest_dir / "brain2text25_probe_manifest.jsonl"
    metadata_path = artifacts.manifest_dir / "brain2text25_probe_metadata.json"

    total_examples = 0
    labeled_examples = 0
    session_ids: set[str] = set()
    split_counts: dict[str, int] = {}
    labeled_split_counts: dict[str, int] = {}

    with manifest_path.open("w") as handle:
        for file_path in hdf5_files:
            source_split = _split_name_from_file(file_path)
            split_counts[source_split] = split_counts.get(source_split, 0) + 0
            labeled_split_counts[source_split] = labeled_split_counts.get(source_split, 0) + 0
            with h5py.File(file_path, "r") as f:
                for trial_key in sorted(f.keys()):
                    record = _build_trial_record(file_path, trial_key, f[trial_key])
                    handle.write(json.dumps(record) + "\n")
                    total_examples += 1
                    if record["has_labels"]:
                        labeled_examples += 1
                        labeled_split_counts[source_split] += 1
                    session_ids.add(record["session_id"])
                    split_counts[source_split] += 1

    metadata = {
        "dataset_family": "brain2text25",
        "source_root_key": "brain2text25_hdf5",
        "source_roots": source_root_metadata(),
        "manifest_path": str(manifest_path.resolve()),
        "total_examples": total_examples,
        "labeled_examples": labeled_examples,
        "num_sessions": len(session_ids),
        "source_split_counts": split_counts,
        "labeled_split_counts": labeled_split_counts,
        "phoneme_vocabulary": {
            "index_to_symbol": list(LOGIT_TO_PHONEME),
            "num_classes": len(LOGIT_TO_PHONEME),
            "blank_index": 0,
            "sil_index": len(LOGIT_TO_PHONEME) - 1,
        },
        "notes": [
            "One row per utterance/trial from the official Brain2Text25 HDF5 release.",
            "This manifest is descriptive and does not yet encode the thesis benchmark split.",
            "The benchmark can later impose held-out-session regimes on top of this manifest.",
            "Manifest rows store source_root_key plus source_relpath instead of machine-specific absolute data paths.",
        ],
    }
    metadata_path.write_text(json.dumps(metadata, indent=2))

    print(f"manifest_path: {manifest_path}")
    print(f"metadata_path: {metadata_path}")
    print(f"total_examples: {total_examples}")
    print(f"labeled_examples: {labeled_examples}")
    print(f"num_sessions: {len(session_ids)}")
    print(f"source_split_counts: {split_counts}")
    print(f"labeled_split_counts: {labeled_split_counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""Evaluate the converted released RNN on Stanford's released TFRecord test files."""

from __future__ import annotations

import argparse
import json
import sys
import tarfile
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from tfrecord.reader import tfrecord_loader

EXPERIMENTS_DIR = Path(__file__).resolve().parents[2]
REPO_ROOT = Path(__file__).resolve().parents[3]
for path in (REPO_ROOT, EXPERIMENTS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiments.supervised_baselines.checkpointing import (  # noqa: E402
    load_willett_model_from_checkpoint,
)
from experiments.supervised_baselines.released_tf_checkpoint import (
    RELEASED_SESSIONS,  # noqa: E402
)
from utah_ssl.ctc import ctc_greedy_decode, edit_counts  # noqa: E402
from utah_ssl.decoding_preprocessing import (  # noqa: E402
    prepare_willett_inputs,
    willett_input_transform_config_from,
)

TFRECORD_DESCRIPTION = {
    "inputFeatures": "float",
    "newClassSignal": "float",
    "ceMask": "float",
    "seqClassIDs": "int",
    "nTimeSteps": "int",
    "nSeqElements": "int",
    "transcription": "int",
}


def _iter_tfrecord_paths(root: Path, *, sessions: tuple[str, ...] | None) -> Iterable[tuple[str, Path]]:
    session_dirs = sorted(path for path in root.iterdir() if path.is_dir())
    allowed = None if sessions is None else set(sessions)
    for session_dir in session_dirs:
        session = session_dir.name
        if allowed is not None and session not in allowed:
            continue
        tfrecord_path = session_dir / "test" / "chunk_0.tfrecord"
        if tfrecord_path.exists():
            yield session, tfrecord_path


def _extract_tfrecords_from_archive(archive_path: Path, output_root: Path) -> Path:
    if (output_root / "tfRecords").exists():
        return output_root / "tfRecords"
    with tarfile.open(archive_path, "r:gz") as archive:
        members = [
            member for member in archive.getmembers()
            if "/tfRecords/" in member.name and member.name.endswith("/test/chunk_0.tfrecord")
        ]
        archive.extractall(output_root, members=members)
    extracted = output_root / "derived" / "tfRecords"
    if not extracted.exists():
        raise FileNotFoundError(f"Archive did not extract released tfRecords under {output_root}")
    return extracted


def _build_model(payload: dict[str, Any], *, device: torch.device) -> tuple[torch.nn.Module, Any]:
    session_adapter_keys = tuple(payload.get("session_adapter_keys") or ())
    if not session_adapter_keys:
        session_adapter_keys = tuple(f"brain2text24:{session}" for session in RELEASED_SESSIONS)
    model, config, _ = load_willett_model_from_checkpoint(
        payload,
        input_dim=256,
        vocab_size=41,
        session_adapter_keys=session_adapter_keys,
        device=device,
    )
    model.eval()
    return model, config


def evaluate_tfrecords(
    *,
    checkpoint_path: Path,
    tfrecord_root: Path,
    device_name: str,
    released_sessions_only: bool,
    max_examples: int | None,
    smooth_inputs: bool,
) -> dict[str, Any]:
    device = torch.device(device_name)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model, config = _build_model(payload, device=device)
    transform_config = willett_input_transform_config_from(config)
    sessions = RELEASED_SESSIONS if released_sessions_only else None
    total_examples = 0
    total_reference_tokens = 0
    total_predicted_tokens = 0
    total_insertions = 0
    total_deletions = 0
    total_substitutions = 0
    per_session: dict[str, dict[str, Any]] = {}
    with torch.no_grad():
        for session, path in _iter_tfrecord_paths(tfrecord_root, sessions=sessions):
            session_ref = session_pred = session_ins = session_del = session_sub = session_examples = 0
            for record in tfrecord_loader(str(path), None, description=TFRECORD_DESCRIPTION):
                n_time_steps = int(np.asarray(record["nTimeSteps"]).reshape(-1)[0])
                n_seq_elements = int(np.asarray(record["nSeqElements"]).reshape(-1)[0])
                features = np.asarray(record["inputFeatures"], dtype=np.float32).reshape(n_time_steps, 256)
                # Released TFRecords already use the repository's 0..40 CTC vocabulary
                # convention, with 0 as the blank class.
                reference = np.asarray(record["seqClassIDs"], dtype=np.int64)[:n_seq_elements].tolist()
                x = torch.from_numpy(features).unsqueeze(0).to(device)
                input_lengths = torch.tensor([n_time_steps], dtype=torch.long, device=device)
                if smooth_inputs:
                    x = prepare_willett_inputs(
                        x,
                        input_lengths,
                        config=transform_config,
                        is_training=False,
                    )
                outputs = model(x, input_lengths, session_ids=[f"brain2text24:{session}"])
                prediction = ctc_greedy_decode(
                    outputs["logits"].cpu(),
                    outputs["token_lengths"].cpu(),
                    blank_index=0,
                )[0]
                ins, dele, sub = edit_counts(reference, prediction)
                session_ref += len(reference)
                session_pred += len(prediction)
                session_ins += ins
                session_del += dele
                session_sub += sub
                session_examples += 1
                total_examples += 1
                if max_examples is not None and total_examples >= int(max_examples):
                    break
            if session_examples:
                errors = session_ins + session_del + session_sub
                per_session[session] = {
                    "examples": int(session_examples),
                    "phoneme_error_rate": float(errors / session_ref) if session_ref else float("nan"),
                    "reference_tokens": int(session_ref),
                    "predicted_tokens": int(session_pred),
                    "insertions": int(session_ins),
                    "deletions": int(session_del),
                    "substitutions": int(session_sub),
                }
                total_reference_tokens += session_ref
                total_predicted_tokens += session_pred
                total_insertions += session_ins
                total_deletions += session_del
                total_substitutions += session_sub
            if max_examples is not None and total_examples >= int(max_examples):
                break
    total_errors = total_insertions + total_deletions + total_substitutions
    return {
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_step": int(payload.get("step", payload.get("steps", -1))),
        "tfrecord_root": str(tfrecord_root),
        "released_sessions_only": bool(released_sessions_only),
        "smooth_inputs": bool(smooth_inputs),
        "examples": int(total_examples),
        "phoneme_error_rate": float(total_errors / total_reference_tokens) if total_reference_tokens else float("nan"),
        "reference_tokens": int(total_reference_tokens),
        "predicted_tokens": int(total_predicted_tokens),
        "edit_diagnostics": {
            "insertions": int(total_insertions),
            "deletions": int(total_deletions),
            "substitutions": int(total_substitutions),
        },
        "per_session": per_session,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tfrecord-root", type=Path, default=None)
    parser.add_argument("--derived-archive", type=Path, default=Path("experiments/supervised_baselines/derived.tar.gz"))
    parser.add_argument("--extract-root", type=Path, default=Path("/tmp/willett_released_tfrecord_eval"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--include-unreleased-sessions", action="store_true")
    parser.add_argument("--smooth-inputs", action="store_true", help="Apply local Willett smoothing before inference.")
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    tfrecord_root = args.tfrecord_root
    if tfrecord_root is None:
        tfrecord_root = _extract_tfrecords_from_archive(args.derived_archive, args.extract_root)
    summary = evaluate_tfrecords(
        checkpoint_path=args.checkpoint,
        tfrecord_root=tfrecord_root,
        device_name=str(args.device),
        released_sessions_only=not bool(args.include_unreleased_sessions),
        max_examples=args.max_examples,
        smooth_inputs=bool(args.smooth_inputs),
    )
    print(json.dumps(summary, indent=2))
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()

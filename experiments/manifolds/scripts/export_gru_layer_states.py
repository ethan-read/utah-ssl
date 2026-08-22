"""Export every recurrent layer of the best recorded local B2T24 GRU.

Run this from a Colab session after mounting Google Drive. The local decoder is
an LLM-assisted Willett-style adaptation with unresolved upstream provenance;
the resulting states are repository artifacts, not outputs of an official
Stanford implementation.
"""

from __future__ import annotations

import argparse
import json
import shutil
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch

from experiments.manifolds.representation_export import (
    RepresentationExportConfig,
    export_willett_representations,
)
from experiments.supervised_baselines.checkpointing import config_from_checkpoint


COLAB_UTAH_SSL_ROOT = Path("/content/drive/MyDrive/utah_ssl")
DEFAULT_CHECKPOINT_PATH = (
    COLAB_UTAH_SSL_ROOT
    / "outputs/willett_reconstruction/willett_tx_only_area6v_colab/checkpoint_best.pt"
)
DEFAULT_CACHE_ROOT = COLAB_UTAH_SSL_ROOT / "data/cache_v1"
DEFAULT_STATS_PATH = (
    COLAB_UTAH_SSL_ROOT
    / "data/stats/split_feature_stats/raw/brain2text24/competition_train/tx_sbp/global_v1.pt"
)
DEFAULT_EXPORT_ROOT = (
    COLAB_UTAH_SSL_ROOT
    / "data/representations/willett_manifolds/gru_layerwise_b2t24_step18300_v1"
)
DEFAULT_MODEL_KEY = "gru_best_step18300_all_val_sessions"
COMPLETION_MARKER_NAME = "_SUCCESS.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-path", type=Path, default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--stats-path", type=Path, default=DEFAULT_STATS_PATH)
    parser.add_argument("--export-root", type=Path, default=DEFAULT_EXPORT_ROOT)
    parser.add_argument("--model-key", default=DEFAULT_MODEL_KEY)
    parser.add_argument("--repo-dir", type=Path, default=Path.cwd())
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument(
        "--allowed-session-id",
        action="append",
        default=None,
        help="Repeat to restrict the export; omit to export every session in the split.",
    )
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Export at most 16 examples to a separate model-key suffix.",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--shard-size-tokens", type=int, default=10_000)
    parser.add_argument(
        "--layer-state-dtype",
        choices=("float16", "float32"),
        default="float16",
        help="On-disk dtype only; equivalence is checked before casting.",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--expected-checkpoint-step", type=int, default=18_300)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _validate_checkpoint_contract(args: argparse.Namespace) -> dict[str, Any]:
    required_paths = (
        Path(args.checkpoint_path),
        Path(args.cache_root) / "brain2text24/metadata.json",
        Path(args.cache_root) / "brain2text24/manifest.jsonl",
        Path(args.stats_path),
        Path(args.stats_path).with_suffix(".json"),
    )
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing checkpoint/data contract artifacts: {missing}")

    payload = torch.load(args.checkpoint_path, map_location="cpu", weights_only=False)
    config = config_from_checkpoint(payload)
    checkpoint_step = int(payload.get("step", payload.get("steps", -1)))
    expected_step = int(args.expected_checkpoint_step)
    if expected_step >= 0 and checkpoint_step != expected_step:
        raise ValueError(
            f"Checkpoint step {checkpoint_step} does not match expected step {expected_step}."
        )
    expected_fields = {
        "dataset": "brain2text24",
        "decoder_backbone_type": "gru",
        "feature_mode": "tx_sbp",
        "split_policy": "competition_train_test",
        "normalization_mode": "global",
        "patch_size": 14,
        "patch_stride": 4,
    }
    mismatches = {
        key: {"observed": getattr(config, key), "expected": expected}
        for key, expected in expected_fields.items()
        if getattr(config, key) != expected
    }
    if mismatches:
        raise ValueError(f"Checkpoint contract mismatch: {mismatches}")
    return {
        "checkpoint_step": checkpoint_step,
        "checkpoint_config": asdict(config),
        "contract": expected_fields,
    }


def validate_layerwise_export(export_dir: str | Path) -> dict[str, Any]:
    """Reopen every shard and validate layer-state shape, dtype, and finiteness."""

    root = Path(export_dir)
    metadata = json.loads((root / "metadata.json").read_text())
    shard_manifest = json.loads((root / "shards.json").read_text())
    layer_keys = tuple(str(key) for key in metadata["gru_layer_state_keys"])
    if not layer_keys or len(layer_keys) != int(metadata["gru_layer_count"]):
        raise ValueError("Layerwise metadata does not declare a complete GRU layer stack.")
    expected_dtype = np.dtype(str(metadata["gru_layer_state_dtype"]))
    hidden_dim = int(metadata["hidden_dim"])
    equivalence = dict(metadata.get("layerwise_equivalence") or {})
    if expected_dtype == np.dtype("float16"):
        storage_atol = 1e-3
        storage_rtol = 1e-3
    else:
        storage_atol = float(equivalence.get("atol", 2e-5))
        storage_rtol = float(equivalence.get("rtol", 1e-5))
    total_tokens = 0
    maximum_final_layer_cast_error = 0.0
    for shard in shard_manifest:
        shard_path = root / "shards" / str(shard["shard"])
        with np.load(shard_path) as arrays:
            row_count = int(arrays["hidden"].shape[0])
            if row_count != int(shard["token_count"]):
                raise ValueError(f"Token count mismatch in {shard_path}.")
            for key in layer_keys:
                values = arrays[key]
                if values.shape != (row_count, hidden_dim):
                    raise ValueError(f"Unexpected shape for {key!r} in {shard_path}.")
                if values.dtype != expected_dtype:
                    raise ValueError(f"Unexpected dtype for {key!r} in {shard_path}.")
                if not np.isfinite(values).all():
                    raise ValueError(f"Nonfinite values in {key!r} in {shard_path}.")
            final_layer = arrays[layer_keys[-1]].astype(np.float32)
            standard_hidden = arrays["hidden"].astype(np.float32)
            if not np.allclose(
                final_layer,
                standard_hidden,
                atol=storage_atol,
                rtol=storage_rtol,
            ):
                maximum_error = float(np.max(np.abs(final_layer - standard_hidden)))
                raise ValueError(
                    "Stored final GRU layer does not match the standard hidden states "
                    f"in {shard_path}: max_abs_error={maximum_error:.8g}, "
                    f"atol={storage_atol:.8g}, rtol={storage_rtol:.8g}."
                )
            cast_error = float(
                np.max(np.abs(final_layer - standard_hidden))
            )
            maximum_final_layer_cast_error = max(
                maximum_final_layer_cast_error,
                cast_error,
            )
            total_tokens += row_count
    if total_tokens != int(metadata["token_count"]):
        raise ValueError("Reopened shard token total does not match metadata.")

    tokens = pd.read_csv(root / "tokens.csv")
    examples = pd.read_csv(root / "examples.csv")
    if len(tokens) != total_tokens or len(examples) != int(metadata["example_count"]):
        raise ValueError("Reopened table counts do not match metadata.")
    return {
        "validated_utc": datetime.now(timezone.utc).isoformat(),
        "export_dir": str(root),
        "shard_count": len(shard_manifest),
        "example_count": int(metadata["example_count"]),
        "token_count": total_tokens,
        "gru_layer_count": len(layer_keys),
        "gru_layer_state_dtype": str(expected_dtype),
        "final_layer_storage_atol": storage_atol,
        "final_layer_storage_rtol": storage_rtol,
        "maximum_final_layer_cast_error": maximum_final_layer_cast_error,
        "status": "passed",
    }


def _rewrite_staged_metadata(
    metadata: dict[str, Any],
    *,
    export_dir: Path,
    export_root: Path,
) -> dict[str, Any]:
    """Replace staging paths with the canonical promoted artifact paths."""

    rewritten = dict(metadata)
    rewritten["token_table_csv"] = str(export_dir / "tokens.csv")
    rewritten["example_table_csv"] = str(export_dir / "examples.csv")
    rewritten["shard_manifest_path"] = str(export_dir / "shards.json")
    export_config = dict(rewritten["representation_export_config"])
    export_config["export_root"] = str(export_root)
    rewritten["representation_export_config"] = export_config
    return rewritten


def _promote_validated_export(
    *,
    staging_dir: Path,
    export_dir: Path,
    overwrite: bool,
) -> None:
    """Rename a completed sibling artifact into place, preserving old output on failure."""

    marker_path = staging_dir / COMPLETION_MARKER_NAME
    if not marker_path.exists():
        raise ValueError("Refusing to promote an export without its completion marker.")
    marker = json.loads(marker_path.read_text())
    if marker.get("status") != "complete":
        raise ValueError("Refusing to promote an export with an invalid completion marker.")
    if export_dir.exists() and not bool(overwrite):
        raise FileExistsError(f"Export directory already exists: {export_dir}")

    backup_dir: Path | None = None
    if export_dir.exists():
        backup_dir = export_dir.parent / f".{export_dir.name}.backup-{uuid.uuid4().hex}"
        export_dir.rename(backup_dir)
    try:
        staging_dir.rename(export_dir)
    except Exception:
        if backup_dir is not None and backup_dir.exists() and not export_dir.exists():
            backup_dir.rename(export_dir)
        raise
    if backup_dir is not None:
        shutil.rmtree(backup_dir)


def run(args: argparse.Namespace) -> dict[str, Any]:
    contract = _validate_checkpoint_contract(args)
    model_key = f"{args.model_key}_smoke" if bool(args.smoke) else str(args.model_key)
    if Path(model_key).name != model_key or model_key in {"", ".", ".."}:
        raise ValueError("model-key must be one nonempty directory name.")
    if bool(args.smoke):
        max_examples = 16 if args.max_examples is None else min(int(args.max_examples), 16)
    else:
        max_examples = args.max_examples
    export_dir = Path(args.export_root) / model_key
    if export_dir.exists() and not bool(args.overwrite):
        raise FileExistsError(
            f"Export directory already exists: {export_dir}. Pass --overwrite to replace it."
        )
    staging_root = Path(args.export_root) / f".staging-{model_key}-{uuid.uuid4().hex}"
    staging_dir = staging_root / model_key
    try:
        metadata = export_willett_representations(
            RepresentationExportConfig(
                checkpoint_path=Path(args.checkpoint_path),
                export_root=staging_root,
                model_key=model_key,
                split=str(args.split),
                allowed_session_ids=args.allowed_session_id,
                max_examples=max_examples,
                batch_size=int(args.batch_size),
                shard_size_tokens=int(args.shard_size_tokens),
                bin_size_ms=20,
                overwrite=False,
                device=args.device,
                repo_dir=Path(args.repo_dir),
                cache_root_override=Path(args.cache_root),
                precomputed_split_stats_path_override=Path(args.stats_path),
                save_input_windows=False,
                save_gru_layer_states=True,
                gru_layer_state_dtype=str(args.layer_state_dtype),
            )
        )
        metadata = _rewrite_staged_metadata(
            metadata,
            export_dir=export_dir,
            export_root=Path(args.export_root),
        )
        (staging_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2, default=str)
        )
        if int(metadata["checkpoint_step"]) != int(contract["checkpoint_step"]):
            raise ValueError("Saved metadata checkpoint step changed during export.")
        expected_signal_spec = {
            "mode": "tx_sbp",
            "tx_dim": 128,
            "sbp_dim": 128,
            "column_start": 0,
            "missing_channel_policy": "error",
        }
        observed_signal_spec = {
            key: metadata["signal_spec"][key] for key in expected_signal_spec
        }
        if observed_signal_spec != expected_signal_spec:
            raise ValueError(
                f"Saved signal contract mismatch: {observed_signal_spec} != {expected_signal_spec}"
            )
        if Path(metadata["cache_root"]) != Path(args.cache_root):
            raise ValueError("Saved cache root does not match the requested cache root.")
        if Path(metadata["precomputed_split_stats_path"]) != Path(args.stats_path):
            raise ValueError("Saved normalization artifact does not match the requested path.")
        if (
            max_examples is None
            and str(args.split) == "val"
            and args.allowed_session_id is None
        ):
            if int(metadata["example_count"]) != 880:
                raise ValueError(
                    "Full Brain-to-Text 2024 competition_test export must contain 880 examples."
                )
            if len(metadata["selected_session_ids"]) != 24:
                raise ValueError("Full validation export must contain all 24 sessions.")
            if metadata["selected_source_splits"] != ["competition_test"]:
                raise ValueError("Full validation export must contain only competition_test rows.")
        validation = validate_layerwise_export(staging_dir)
        validation["export_dir"] = str(export_dir)
        validation["checkpoint_contract"] = contract
        validation_path = staging_dir / "validation.json"
        validation_path.write_text(json.dumps(validation, indent=2, default=str))
        reopened_validation = json.loads(validation_path.read_text())
        if reopened_validation.get("status") != "passed":
            raise ValueError("Saved validation artifact did not reopen successfully.")
        completion_marker = {
            "status": "complete",
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "validation_path": str(export_dir / "validation.json"),
        }
        (staging_dir / COMPLETION_MARKER_NAME).write_text(
            json.dumps(completion_marker, indent=2)
        )
        if json.loads((staging_dir / COMPLETION_MARKER_NAME).read_text()).get("status") != "complete":
            raise ValueError("Completion marker did not reopen successfully.")
        _promote_validated_export(
            staging_dir=staging_dir,
            export_dir=export_dir,
            overwrite=bool(args.overwrite),
        )
        promoted_marker = json.loads(
            (export_dir / COMPLETION_MARKER_NAME).read_text()
        )
        if promoted_marker.get("status") != "complete":
            raise ValueError("Promoted completion marker did not reopen successfully.")
    finally:
        if staging_root.exists():
            shutil.rmtree(staging_root)
    return {"metadata": metadata, "validation": validation}


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    result = run(args)
    print(
        json.dumps(
            {
                "export_dir": result["validation"]["export_dir"],
                "checkpoint_step": result["metadata"]["checkpoint_step"],
                "example_count": result["validation"]["example_count"],
                "token_count": result["validation"]["token_count"],
                "gru_layer_count": result["validation"]["gru_layer_count"],
                "gru_layer_state_dtype": result["validation"]["gru_layer_state_dtype"],
                "layerwise_equivalence": result["metadata"]["layerwise_equivalence"],
                "validation": result["validation"]["status"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

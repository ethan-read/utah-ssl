"""Export full-dataset future-prediction bins on Modal."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import modal


APP_NAME = "utah-ssl-future-prediction-export"
DEFAULT_CACHE_VOLUME_NAME = "utah-ssl-cache"
DEFAULT_OUTPUT_VOLUME_NAME = "utah-ssl-outputs"

def _resolve_local_repo_root() -> Path:
    for candidate in Path(__file__).resolve().parents:
        if (candidate / ".git").exists():
            return candidate
    raise RuntimeError("Could not locate repository root from launcher path")


LOCAL_REPO_ROOT = _resolve_local_repo_root()
REMOTE_REPO_ROOT = Path("/root/utah-ssl")
REMOTE_CACHE_VOLUME_ROOT = Path("/vol/cache")
REMOTE_OUTPUT_VOLUME_ROOT = Path("/vol/outputs")

DEFAULT_REMOTE_CACHE_SUBDIR = "cache_v1"
DEFAULT_OUTPUT_SUBDIR = "ssl_experiments/modal_future_prediction_ssl_exports"
DEFAULT_CHECKPOINT_PATH = (
    REMOTE_OUTPUT_VOLUME_ROOT
    / "ssl_experiments/modal_future_prediction_ssl"
    / "future_pred_mamba_b2t24_txsbp_40ms_ctx12bins_h1_var001_slim20k"
    / "checkpoint_best.pt"
)

cache_volume = modal.Volume.from_name(DEFAULT_CACHE_VOLUME_NAME, create_if_missing=True)
output_volume = modal.Volume.from_name(DEFAULT_OUTPUT_VOLUME_NAME, create_if_missing=True)


def _base_registry_image() -> modal.Image:
    return modal.Image.from_registry(
        "nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04",
        add_python="3.10",
    ).apt_install("git").env({"PYTHONPATH": str(REMOTE_REPO_ROOT)})


def _candidate_image() -> modal.Image:
    return (
        _base_registry_image()
        .uv_pip_install(
            "numpy==2.1.2",
            "pandas==2.2.3",
            "torch==2.8.0",
            "transformers==4.57.6",
            "wheel",
            "einops",
        )
        .run_commands(
            "python -m pip install 'causal-conv1d==1.6.2.post1' --no-build-isolation",
            "python -m pip install 'mamba-ssm==2.3.2.post1' --no-build-isolation --no-deps",
        )
        .add_local_file(
            str(Path(__file__).resolve()),
            remote_path=str(REMOTE_REPO_ROOT / "export_future_prediction_bins.py"),
            copy=True,
        )
        .add_local_dir(
            str(LOCAL_REPO_ROOT / "utah_ssl"),
            remote_path=str(REMOTE_REPO_ROOT / "utah_ssl"),
            copy=True,
        )
        .add_local_dir(
            str(LOCAL_REPO_ROOT / "experiments"),
            remote_path=str(REMOTE_REPO_ROOT / "experiments"),
            copy=True,
        )
    )


app = modal.App(APP_NAME)


def _ensure_import_paths() -> None:
    repo_root = str(REMOTE_REPO_ROOT)
    for path in (repo_root,):
        if path not in sys.path:
            sys.path.insert(0, path)


@app.function(
    image=_candidate_image(),
    serialized=True,
    gpu=["L40S", "RTX-PRO-6000"],
    cpu=8,
    memory=32768,
    timeout=60 * 60 * 24,
    volumes={
        str(REMOTE_CACHE_VOLUME_ROOT): cache_volume.with_mount_options(read_only=True),
        str(REMOTE_OUTPUT_VOLUME_ROOT): output_volume,
    },
)
def export_future_prediction_bins_remote(
    *,
    export_name: str,
    checkpoint_path: str = str(DEFAULT_CHECKPOINT_PATH),
    cache_subdir: str = DEFAULT_REMOTE_CACHE_SUBDIR,
    output_subdir: str = DEFAULT_OUTPUT_SUBDIR,
    dataset: str = "brain2text24",
) -> dict[str, object]:
    _ensure_import_paths()

    from experiments.archive.mamba_future_prediction.export_predictions import (
        export_future_prediction_bins,
    )

    output_dir = REMOTE_OUTPUT_VOLUME_ROOT / output_subdir / export_name
    progress_events: list[dict[str, Any]] = []

    def _on_shard_written(event: dict[str, Any]) -> None:
        progress_events.append(dict(event))
        output_volume.commit()
        print(json.dumps({"event": "shard_written", **event}, sort_keys=True), flush=True)

    result = export_future_prediction_bins(
        checkpoint_path=checkpoint_path,
        output_dir=output_dir,
        cache_root=REMOTE_CACHE_VOLUME_ROOT / cache_subdir,
        datasets=(str(dataset),),
        resume=True,
        overwrite_existing=False,
        on_shard_written=_on_shard_written,
    )
    output_volume.commit()
    return {
        "checkpoint_path": str(checkpoint_path),
        "cache_root": str(REMOTE_CACHE_VOLUME_ROOT / cache_subdir),
        "output_dir": str(output_dir),
        "progress_events": progress_events,
        "result": result,
    }


@app.local_entrypoint()
def main(
    export_name: str = "future_pred_mamba_b2t24_txsbp_40ms_ctx12bins_h1_var001_slim20k_best_full_export",
    checkpoint_path: str = str(DEFAULT_CHECKPOINT_PATH),
    cache_subdir: str = DEFAULT_REMOTE_CACHE_SUBDIR,
    output_subdir: str = DEFAULT_OUTPUT_SUBDIR,
    dataset: str = "brain2text24",
) -> None:
    result = export_future_prediction_bins_remote.remote(
        export_name=export_name,
        checkpoint_path=checkpoint_path,
        cache_subdir=cache_subdir,
        output_subdir=output_subdir,
        dataset=dataset,
    )
    print(json.dumps(result, indent=2, sort_keys=True))

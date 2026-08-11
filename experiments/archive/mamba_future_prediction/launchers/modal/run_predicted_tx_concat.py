"""Run Willett GRU training on Modal with raw TX concatenated to duplicated predicted TX."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import modal


APP_NAME = "utah-ssl-willett-predicted-tx-concat"
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
DEFAULT_OUTPUT_SUBDIR = "ssl_experiments/modal_willett_predicted_tx_concat"
DEFAULT_EXPORT_SUBDIR = "ssl_experiments/modal_future_prediction_ssl_exports"
DEFAULT_EXPORT_NAME = "future_pred_mamba_b2t24_txsbp_40ms_ctx12bins_h1_var001_slim20k_best_full_export"

cache_volume = modal.Volume.from_name(DEFAULT_CACHE_VOLUME_NAME, create_if_missing=True)
output_volume = modal.Volume.from_name(DEFAULT_OUTPUT_VOLUME_NAME, create_if_missing=True)


def _base_registry_image() -> modal.Image:
    return modal.Image.from_registry(
        "nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04",
        add_python="3.10",
    ).apt_install("git").env({"PYTHONPATH": str(REMOTE_REPO_ROOT)})


def _image() -> modal.Image:
    return (
        _base_registry_image()
        .uv_pip_install(
            "numpy==2.1.2",
            "pandas==2.2.3",
            "torch==2.8.0",
            "einops",
        )
        .add_local_file(
            str(Path(__file__).resolve()),
            remote_path=str(REMOTE_REPO_ROOT / "run_willett_predicted_tx_concat.py"),
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


def _recompute_tx_only_stats(*, cache_root: Path) -> None:
    cmd = [
        sys.executable,
        str(REMOTE_REPO_ROOT / "utah_ssl" / "scripts" / "recompute_split_feature_stats.py"),
        "--cache-root",
        str(cache_root),
        "--dataset",
        "brain2text24",
        "--feature-mode",
        "tx_only",
        "--boundary-key-mode",
        "session",
        "--overwrite",
    ]
    subprocess.run(cmd, check=True)


@app.function(
    image=_image(),
    serialized=True,
    gpu=["L40S", "RTX-PRO-6000"],
    cpu=8,
    memory=32768,
    timeout=60 * 60 * 24,
    volumes={
        str(REMOTE_CACHE_VOLUME_ROOT): cache_volume,
        str(REMOTE_OUTPUT_VOLUME_ROOT): output_volume,
    },
)
def run_willett_predicted_tx_concat_remote(
    *,
    run_name: str,
    max_steps: int = 12000,
    batch_size: int = 64,
    export_name: str = DEFAULT_EXPORT_NAME,
    cache_subdir: str = DEFAULT_REMOTE_CACHE_SUBDIR,
    output_subdir: str = DEFAULT_OUTPUT_SUBDIR,
    export_subdir: str = DEFAULT_EXPORT_SUBDIR,
    input_smoothing_sigma_bins: float = 0.0,
    learning_rate: float = 1e-2,
    min_learning_rate: float = 1e-4,
    warmup_steps: int = 1000,
    seed: int = 7,
) -> dict[str, object]:
    _ensure_import_paths()

    from experiments.supervised_baselines.train import (
        WillettReconstructionConfig,
        run_willett_reconstruction,
    )

    cache_root = REMOTE_CACHE_VOLUME_ROOT / cache_subdir
    output_root = REMOTE_OUTPUT_VOLUME_ROOT / output_subdir
    predicted_export_root = REMOTE_OUTPUT_VOLUME_ROOT / export_subdir / export_name

    _recompute_tx_only_stats(cache_root=cache_root)
    cache_volume.commit()

    config = WillettReconstructionConfig(
        cache_root=cache_root,
        output_root=output_root,
        run_name=run_name,
        dataset="brain2text24",
        feature_mode="tx_only",
        boundary_key_mode="session",
        split_policy="competition_train_test",
        normalization_mode="global",
        decoder_backbone_type="gru",
        batch_size=int(batch_size),
        max_steps=int(max_steps),
        learning_rate=float(learning_rate),
        min_learning_rate=float(min_learning_rate),
        warmup_steps=int(warmup_steps),
        input_feature_source="raw_plus_predicted_tx",
        predicted_export_root=predicted_export_root,
        input_smoothing_sigma_bins=float(input_smoothing_sigma_bins),
        seed=int(seed),
        resume_latest=True,
    )
    summary = run_willett_reconstruction(config)
    output_volume.commit()
    return {
        "run_name": str(run_name),
        "cache_root": str(cache_root),
        "predicted_export_root": str(predicted_export_root),
        "output_root": str(output_root),
        "summary": summary,
    }


@app.local_entrypoint()
def main(
    run_name: str = "willett_gru_tx_plus_predtx_dup20ms_nosmooth_12k",
    max_steps: int = 12000,
    batch_size: int = 64,
    export_name: str = DEFAULT_EXPORT_NAME,
    cache_subdir: str = DEFAULT_REMOTE_CACHE_SUBDIR,
    output_subdir: str = DEFAULT_OUTPUT_SUBDIR,
    export_subdir: str = DEFAULT_EXPORT_SUBDIR,
    input_smoothing_sigma_bins: float = 0.0,
    learning_rate: float = 1e-2,
    min_learning_rate: float = 1e-4,
    warmup_steps: int = 1000,
    seed: int = 7,
) -> None:
    result = run_willett_predicted_tx_concat_remote.remote(
        run_name=run_name,
        max_steps=max_steps,
        batch_size=batch_size,
        export_name=export_name,
        cache_subdir=cache_subdir,
        output_subdir=output_subdir,
        export_subdir=export_subdir,
        input_smoothing_sigma_bins=input_smoothing_sigma_bins,
        learning_rate=learning_rate,
        min_learning_rate=min_learning_rate,
        warmup_steps=warmup_steps,
        seed=seed,
    )
    print(json.dumps(result, indent=2, sort_keys=True))

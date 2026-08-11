"""Run tx-only stage-2 phoneme CTC fine-tuning on Modal for a prior BIT-style S5 run."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import modal


APP_NAME = "utah-ssl-bit-s5-stage2-ctc"
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

DEFAULT_CACHE_SUBDIR = "cache_v1_smoothed_sigma2p0"
DEFAULT_STAGE1_OUTPUT_SUBDIR = "ssl_experiments/modal_bit_s5_stage1"

cache_volume = modal.Volume.from_name(DEFAULT_CACHE_VOLUME_NAME, create_if_missing=True)
output_volume = modal.Volume.from_name(DEFAULT_OUTPUT_VOLUME_NAME, create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.10")
    .uv_pip_install(
        "numpy==2.1.2",
        "pandas==2.2.3",
        "torch==2.8.0",
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

app = modal.App(APP_NAME, image=image)


def _ensure_import_paths() -> None:
    repo_root = str(REMOTE_REPO_ROOT)
    for path in (repo_root,):
        if path not in sys.path:
            sys.path.insert(0, path)


@app.function(
    gpu=["L40S", "RTX-PRO-6000"],
    cpu=8,
    memory=32768,
    timeout=60 * 60 * 24,
    volumes={
        str(REMOTE_CACHE_VOLUME_ROOT): cache_volume.with_mount_options(read_only=True),
        str(REMOTE_OUTPUT_VOLUME_ROOT): output_volume,
    },
)
def run_stage2_ctc(
    *,
    stage1_run_name: str,
    stage1_output_subdir: str = DEFAULT_STAGE1_OUTPUT_SUBDIR,
    cache_subdir: str = DEFAULT_CACHE_SUBDIR,
    ctc_steps: int = 12000,
    batch_size: int = 16,
    ctc_learning_rate: float = 1e-3,
    weight_decay: float = 1e-2,
    max_grad_norm: float = 1.0,
    val_every_steps: int = 50,
    progress_every_steps: int = 25,
    seed: int = 7,
    also_random_init: bool = False,
) -> dict[str, object]:
    _ensure_import_paths()

    from experiments.bit_style.config import GenericSSMSSLConfig
    from experiments.bit_style.training import run_ctc_finetuning
    from utah_ssl.experiment_contract import (
        DatasetPlan,
        SignalSpec,
    )

    stage1_run_dir = REMOTE_OUTPUT_VOLUME_ROOT / stage1_output_subdir / stage1_run_name
    checkpoint_path = stage1_run_dir / "checkpoint_best.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Stage-1 checkpoint not found: {checkpoint_path}")

    cache_root = REMOTE_CACHE_VOLUME_ROOT / cache_subdir
    config = GenericSSMSSLConfig(
        seed=int(seed),
        backbone_type="s5",
        input_mode="temporal_patch",
        dataset="brain2text24",
        dataset_plan=DatasetPlan.from_mapping(
            {"brain2text24": ("competition_train",)}
        ),
        signal_spec=SignalSpec.tx_only(tx_dim=256),
        boundary_key_mode="session",
        cache_root=str(cache_root),
        cache_mode="drive_direct",
        local_cache_base="/tmp/utah_ssl_cache",
        use_normalization=True,
        precomputed_session_stats_path=None,
        precomputed_split_stats_path=None,
        normalization_mode="global",
        segment_bins=256,
        batch_size=int(batch_size),
        ssl_steps=1,
        ctc_steps=int(ctc_steps),
        run_downstream_ctc=False,
        learning_rate=3e-4,
        ctc_learning_rate=float(ctc_learning_rate),
        weight_decay=float(weight_decay),
        max_grad_norm=float(max_grad_norm),
        hidden_size=256,
        state_size=64,
        num_layers=4,
        dropout=0.1,
        direction="bidirectional",
        ffn_multiplier=2.0,
        patch_size=5,
        patch_stride=5,
        patch_policy="floor",
        conv_kernel_size=14,
        conv_stride=4,
        mask_time_ratio=0.25,
        mask_channel_ratio=0.10,
        mask_chunk_size=4,
        ctc_input_smoothing_sigma_bins=0.0,
        ctc_input_smoothing_kernel_size=100,
        ctc_input_smoothing_threshold=0.01,
        ctc_white_noise_sd=0.0,
        ctc_constant_offset_sd=0.0,
        val_every_steps=int(val_every_steps),
        val_batches=4,
        progress_every_steps=int(progress_every_steps),
        progress_every_seconds=30.0,
        checkpoint_every_steps=None,
        output_root=str(stage1_run_dir),
        run_name=stage1_run_name,
    )

    print(f"Stage-1 run dir: {stage1_run_dir}", flush=True)
    print(f"Checkpoint: {checkpoint_path}", flush=True)
    print(f"Cache root: {cache_root}", flush=True)
    print("Starting pretrained tx-only CTC fine-tuning", flush=True)

    pretrained_summary = run_ctc_finetuning(
        config,
        run_dir=stage1_run_dir,
        encoder_checkpoint_path=checkpoint_path,
        label="pretrained_tx_only",
    )
    result: dict[str, object] = {
        "stage1_run_name": stage1_run_name,
        "stage1_run_dir": str(stage1_run_dir),
        "checkpoint_path": str(checkpoint_path),
        "cache_root": str(cache_root),
        "pretrained": pretrained_summary,
    }

    if bool(also_random_init):
        print("Starting random-init tx-only CTC control", flush=True)
        random_summary = run_ctc_finetuning(
            config,
            run_dir=stage1_run_dir,
            encoder_checkpoint_path=None,
            label="random_init_tx_only",
        )
        result["random_init"] = random_summary

    output_volume.commit()
    return result


@app.local_entrypoint()
def main(
    stage1_run_name: str,
    stage1_output_subdir: str = DEFAULT_STAGE1_OUTPUT_SUBDIR,
    cache_subdir: str = DEFAULT_CACHE_SUBDIR,
    ctc_steps: int = 12000,
    batch_size: int = 16,
    ctc_learning_rate: float = 1e-3,
    weight_decay: float = 1e-2,
    max_grad_norm: float = 1.0,
    val_every_steps: int = 50,
    progress_every_steps: int = 25,
    seed: int = 7,
    also_random_init: bool = False,
) -> None:
    result = run_stage2_ctc.remote(
        stage1_run_name=stage1_run_name,
        stage1_output_subdir=stage1_output_subdir,
        cache_subdir=cache_subdir,
        ctc_steps=ctc_steps,
        batch_size=batch_size,
        ctc_learning_rate=ctc_learning_rate,
        weight_decay=weight_decay,
        max_grad_norm=max_grad_norm,
        val_every_steps=val_every_steps,
        progress_every_steps=progress_every_steps,
        seed=seed,
        also_random_init=also_random_init,
    )
    print(json.dumps(result, indent=2, sort_keys=True))

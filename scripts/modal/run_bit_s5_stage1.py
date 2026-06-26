"""Run BIT-style S5 stage-1 SSL on Modal.

This script is designed to be invoked with:

    modal run scripts/modal/run_bit_s5_stage1.py

or, to upload the local smoothed cache and stats first:

    modal run scripts/modal/run_bit_s5_stage1.py --sync-cache

Notes:
- Modal's current documented GPU list does not include an RTX 4090. This script
  requests an `L40S` first and falls back to `RTX-PRO-6000`, which are the
  closest currently documented options.
- The default cache root is the BIT-style smoothed corpus prepared locally at
  `/Users/home/thesis/data/cache_v1_smoothed_sigma2p0`.
- The default stats root is `/Users/home/thesis/data/stats`.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import modal


APP_NAME = "utah-ssl-bit-s5-stage1"
DEFAULT_CACHE_VOLUME_NAME = "utah-ssl-cache"
DEFAULT_OUTPUT_VOLUME_NAME = "utah-ssl-outputs"


def _resolve_local_repo_root() -> Path:
    resolved = Path(__file__).resolve()
    if len(resolved.parents) >= 3:
        return resolved.parents[2]
    return resolved.parent


LOCAL_REPO_ROOT = _resolve_local_repo_root()

LOCAL_CACHE_ROOT = Path("/Users/home/thesis/data/cache_v1_smoothed_sigma2p0")
LOCAL_STATS_ROOT = Path("/Users/home/thesis/data/stats")

REMOTE_REPO_ROOT = Path("/root/utah-ssl")
REMOTE_CACHE_VOLUME_ROOT = Path("/vol/cache")
REMOTE_OUTPUT_VOLUME_ROOT = Path("/vol/outputs")

DEFAULT_REMOTE_CACHE_SUBDIR = "cache_v1_smoothed_sigma2p0"
DEFAULT_REMOTE_STATS_SUBDIR = "stats"
DEFAULT_OUTPUT_SUBDIR = "ssl_experiments/modal_bit_s5_stage1"


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
        str(LOCAL_REPO_ROOT / "analysis"),
        remote_path=str(REMOTE_REPO_ROOT / "analysis"),
        copy=True,
    )
)

app = modal.App(APP_NAME, image=image)


def _upload_directory_to_volume(
    *,
    volume: modal.Volume,
    local_dir: Path,
    remote_dir: str,
) -> None:
    if not local_dir.exists():
        raise FileNotFoundError(f"Local directory does not exist: {local_dir}")
    if not local_dir.is_dir():
        raise NotADirectoryError(f"Expected a directory: {local_dir}")
    with volume.batch_upload() as batch:
        batch.put_directory(str(local_dir), remote_dir)


def _build_config(
    *,
    run_name: str,
    cache_root: Path,
    output_root: Path,
    ssl_steps: int,
    ctc_steps: int,
    run_downstream_ctc: bool,
    batch_size: int,
    hidden_size: int,
    state_size: int,
    num_layers: int,
    seed: int,
) -> dict[str, object]:
    return {
        "seed": int(seed),
        "backbone_type": "s5",
        "input_mode": "temporal_patch",
        "objective": "masked_time_channel_reconstruction",
        "dataset": "brain2text24",
        "feature_mode": "tx_only",
        "boundary_key_mode": "session",
        "cache_root": str(cache_root),
        "cache_mode": "drive_direct",
        "local_cache_base": "/tmp/utah_ssl_cache",
        "excluded_datasets": ["brain2text25"],
        "use_normalization": True,
        "precomputed_session_stats_path": None,
        "precomputed_split_stats_path": None,
        "normalization_mode": "global",
        "tx_dim": 256,
        "sbp_dim": 0,
        "segment_bins": 256,
        "batch_size": int(batch_size),
        "ssl_steps": int(ssl_steps),
        "ctc_steps": int(ctc_steps),
        "run_downstream_ctc": bool(run_downstream_ctc),
        "learning_rate": 3e-4,
        "ctc_learning_rate": 1e-3,
        "weight_decay": 1e-2,
        "max_grad_norm": 1.0,
        "hidden_size": int(hidden_size),
        "state_size": int(state_size),
        "num_layers": int(num_layers),
        "dropout": 0.1,
        "direction": "bidirectional",
        "ffn_multiplier": 2.0,
        "patch_size": 5,
        "patch_stride": 5,
        "patch_policy": "floor",
        "conv_kernel_size": 14,
        "conv_stride": 4,
        "mask_time_ratio": 0.25,
        "mask_channel_ratio": 0.10,
        "mask_chunk_size": 4,
        "val_every_steps": 100,
        "val_batches": 4,
        "progress_every_steps": 25,
        "progress_every_seconds": 30.0,
        "checkpoint_every_steps": 1000,
        "output_root": str(output_root),
        "run_name": str(run_name),
    }


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
def train_bit_s5_stage1(
    *,
    run_name: str,
    cache_subdir: str = DEFAULT_REMOTE_CACHE_SUBDIR,
    output_subdir: str = DEFAULT_OUTPUT_SUBDIR,
    ssl_steps: int = 60000,
    ctc_steps: int = 12000,
    run_downstream_ctc: bool = False,
    batch_size: int = 16,
    hidden_size: int = 256,
    state_size: int = 64,
    num_layers: int = 4,
    seed: int = 7,
) -> dict[str, object]:
    cache_root = REMOTE_CACHE_VOLUME_ROOT / cache_subdir
    output_root = REMOTE_OUTPUT_VOLUME_ROOT / output_subdir
    output_root.mkdir(parents=True, exist_ok=True)

    config = _build_config(
        run_name=run_name,
        cache_root=cache_root,
        output_root=output_root,
        ssl_steps=ssl_steps,
        ctc_steps=ctc_steps,
        run_downstream_ctc=run_downstream_ctc,
        batch_size=batch_size,
        hidden_size=hidden_size,
        state_size=state_size,
        num_layers=num_layers,
        seed=seed,
    )

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
        json.dump(config, handle, indent=2, sort_keys=True)
        config_json_path = Path(handle.name)

    command = [
        sys.executable,
        "analysis/active/ssl_experiments/ssm_ssl/scripts/run_generic_ssm_ssl.py",
        "--config-json",
        str(config_json_path),
    ]
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"

    print(f"Running in {REMOTE_REPO_ROOT}", flush=True)
    print(f"Cache root: {cache_root}", flush=True)
    print(f"Output root: {output_root}", flush=True)
    print("Command:", " ".join(command), flush=True)

    try:
        subprocess.run(
            command,
            cwd=str(REMOTE_REPO_ROOT),
            env=env,
            stdout=sys.stdout,
            stderr=sys.stderr,
            check=True,
        )
    finally:
        try:
            config_json_path.unlink(missing_ok=True)
        finally:
            output_volume.commit()

    summary_path = output_root / run_name / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Expected summary file was not produced: {summary_path}")
    payload = json.loads(summary_path.read_text())
    output_volume.commit()
    return {
        "gpu_request": ["L40S", "RTX-PRO-6000"],
        "cache_root": str(cache_root),
        "output_root": str(output_root),
        "summary_path": str(summary_path),
        "summary": payload,
    }


@app.local_entrypoint()
def main(
    run_name: str = "bit_s5_stage1_l40s",
    sync_cache: bool = False,
    local_cache_root: str = str(LOCAL_CACHE_ROOT),
    local_stats_root: str = str(LOCAL_STATS_ROOT),
    remote_cache_subdir: str = DEFAULT_REMOTE_CACHE_SUBDIR,
    remote_stats_subdir: str = DEFAULT_REMOTE_STATS_SUBDIR,
    output_subdir: str = DEFAULT_OUTPUT_SUBDIR,
    ssl_steps: int = 60000,
    ctc_steps: int = 12000,
    run_downstream_ctc: bool = False,
    batch_size: int = 16,
    hidden_size: int = 256,
    state_size: int = 64,
    num_layers: int = 4,
    seed: int = 7,
) -> None:
    local_cache = Path(local_cache_root).expanduser().resolve()
    local_stats = Path(local_stats_root).expanduser().resolve()

    if sync_cache:
        print(
            f"Uploading cache {local_cache} -> {DEFAULT_CACHE_VOLUME_NAME}:{remote_cache_subdir}",
            flush=True,
        )
        _upload_directory_to_volume(
            volume=cache_volume,
            local_dir=local_cache,
            remote_dir=f"/{remote_cache_subdir}",
        )
        print(
            f"Uploading stats {local_stats} -> {DEFAULT_CACHE_VOLUME_NAME}:{remote_stats_subdir}",
            flush=True,
        )
        _upload_directory_to_volume(
            volume=cache_volume,
            local_dir=local_stats,
            remote_dir=f"/{remote_stats_subdir}",
        )
        print("Cache/stat sync complete.", flush=True)

    result = train_bit_s5_stage1.remote(
        run_name=run_name,
        cache_subdir=remote_cache_subdir,
        output_subdir=output_subdir,
        ssl_steps=ssl_steps,
        ctc_steps=ctc_steps,
        run_downstream_ctc=run_downstream_ctc,
        batch_size=batch_size,
        hidden_size=hidden_size,
        state_size=state_size,
        num_layers=num_layers,
        seed=seed,
    )
    print(json.dumps(result, indent=2, sort_keys=True))

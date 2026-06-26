"""Recompute BIT stage-1 session stats inside the Modal cache volume."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import modal


APP_NAME = "utah-ssl-recompute-bit-stage1-stats"
CACHE_VOLUME_NAME = "utah-ssl-cache"

CACHE_MOUNT = Path("/vol/cache")
REMOTE_REPO_ROOT = Path("/root/utah-ssl")

CACHE_SUBDIR = "cache_v1_smoothed_sigma2p0"
STATS_OUTPUT = (
    "stats/session_feature_stats/smoothed_sigma2p0/tx_only/session/"
    "ssl_pretrain_including_brain2text24_excluding_brain2text25_v1.pt"
)


def _resolve_local_repo_root() -> Path:
    resolved = Path(__file__).resolve()
    if len(resolved.parents) >= 3:
        return resolved.parents[2]
    return resolved.parent


LOCAL_REPO_ROOT = _resolve_local_repo_root()

cache_volume = modal.Volume.from_name(CACHE_VOLUME_NAME, create_if_missing=True)

image = modal.Image.debian_slim(python_version="3.10").uv_pip_install(
    "numpy==2.1.2",
    "pandas==2.2.3",
    "torch==2.8.0",
).add_local_dir(
    str(LOCAL_REPO_ROOT / "analysis"),
    remote_path=str(REMOTE_REPO_ROOT / "analysis"),
    copy=True,
)

app = modal.App(APP_NAME, image=image)


@app.function(
    cpu=8,
    memory=32768,
    timeout=60 * 60 * 12,
    volumes={str(CACHE_MOUNT): cache_volume},
)
def recompute_stats() -> dict[str, str]:
    cache_root = CACHE_MOUNT / CACHE_SUBDIR
    output_path = CACHE_MOUNT / STATS_OUTPUT

    command = [
        sys.executable,
        "analysis/active/ssl_experiments/recompute_session_feature_stats.py",
        "--cache-root",
        str(cache_root),
        "--output-path",
        str(output_path),
        "--feature-mode",
        "tx_only",
        "--boundary-key-mode",
        "session",
        "--tx-dim",
        "256",
        "--sbp-dim",
        "0",
        "--segment-bins",
        "256",
        "--examples-per-shard",
        "8",
        "--excluded-dataset",
        "brain2text25",
        "--overwrite",
    ]

    print("Running:", " ".join(command), flush=True)
    subprocess.run(
        command,
        cwd=str(REMOTE_REPO_ROOT),
        check=True,
        stdout=sys.stdout,
        stderr=sys.stderr,
    )
    cache_volume.commit()

    metadata_path = output_path.with_suffix(".json")
    return {
        "cache_root": str(cache_root),
        "output_path": str(output_path),
        "metadata_path": str(metadata_path),
        "metadata": json.loads(metadata_path.read_text()) if metadata_path.exists() else {},
    }


@app.local_entrypoint()
def main() -> None:
    result = recompute_stats.remote()
    print(json.dumps(result, indent=2, sort_keys=True))

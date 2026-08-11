"""Recompute BIT stage-1 session stats inside the Modal cache volume."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import modal

from utah_ssl.bit_cache_contract import (
    BIT_STAGE1_DATASET_SPLITS,
    BIT_STAGE1_TX_DIM,
)
from utah_ssl.experiment_contract import (
    DatasetPlan,
    SignalSpec,
)
from utah_ssl.cache import (
    resolve_precomputed_session_stats_path,
)

APP_NAME = "utah-ssl-recompute-bit-stage1-stats"
CACHE_VOLUME_NAME = "utah-ssl-cache"

CACHE_MOUNT = Path("/vol/cache")
REMOTE_REPO_ROOT = Path("/root/utah-ssl")

CACHE_SUBDIR = "cache_v1_smoothed_sigma2p0"
DATASET_PLAN = DatasetPlan.from_mapping(BIT_STAGE1_DATASET_SPLITS)
SIGNAL_SPEC = SignalSpec.tx_only(
    tx_dim=BIT_STAGE1_TX_DIM,
    missing_channel_policy="zero_pad",
)
STATS_OUTPUT = resolve_precomputed_session_stats_path(
    cache_root=CACHE_MOUNT / CACHE_SUBDIR,
    signal_spec=SIGNAL_SPEC,
    dataset_plan=DATASET_PLAN,
    boundary_key_mode="session",
)


def _resolve_local_repo_root() -> Path:
    for candidate in Path(__file__).resolve().parents:
        if (candidate / ".git").exists():
            return candidate
    raise RuntimeError("Could not locate repository root from launcher path")


LOCAL_REPO_ROOT = _resolve_local_repo_root()

cache_volume = modal.Volume.from_name(CACHE_VOLUME_NAME, create_if_missing=True)

image = modal.Image.debian_slim(python_version="3.10").uv_pip_install(
    "numpy==2.1.2",
    "pandas==2.2.3",
    "torch==2.8.0",
).add_local_dir(
    str(LOCAL_REPO_ROOT / "utah_ssl"),
    remote_path=str(REMOTE_REPO_ROOT / "utah_ssl"),
    copy=True,
).add_local_dir(
    str(LOCAL_REPO_ROOT / "experiments"),
    remote_path=str(REMOTE_REPO_ROOT / "experiments"),
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
    output_path = STATS_OUTPUT

    command = [
        sys.executable,
        "utah_ssl/scripts/recompute_session_feature_stats.py",
        "--cache-root",
        str(cache_root),
        "--output-path",
        str(output_path),
        "--feature-mode",
        SIGNAL_SPEC.mode,
        "--boundary-key-mode",
        "session",
        "--tx-dim",
        str(SIGNAL_SPEC.tx_dim),
        "--sbp-dim",
        str(SIGNAL_SPEC.sbp_dim),
        "--column-start",
        str(SIGNAL_SPEC.column_start),
        "--missing-channel-policy",
        SIGNAL_SPEC.missing_channel_policy,
        "--overwrite",
    ]
    for selection in DATASET_PLAN.datasets:
        command.extend(["--dataset", selection.name])
        for source_split in selection.source_splits:
            command.extend(
                ["--dataset-source-split", f"{selection.name}={source_split}"]
            )

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

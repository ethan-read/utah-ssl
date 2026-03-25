"""Frozen runtime and benchmark scaffold for the full SSL autoresearch setup."""

from __future__ import annotations

import os
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


SOURCE_DIR = Path(__file__).resolve().parent
ROOT_DIR = SOURCE_DIR.parents[3]


def _env_path(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    if value:
        return Path(value).expanduser().resolve()
    return default.resolve()


OUTPUT_ROOT = _env_path(
    "SSL_AUTORESEARCH_OUTPUT_ROOT",
    ROOT_DIR / "outputs" / "transfer_benchmark" / "ssl_autoresearch",
)
TX_CACHE_DIR = _env_path("SSL_AUTORESEARCH_TX_CACHE_DIR", ROOT_DIR / "code" / "ssl" / "cache")
SBP_CACHE_DIR = _env_path("SSL_AUTORESEARCH_SBP_CACHE_DIR", ROOT_DIR / "code" / "ssl" / "cache_b2t25_sbp")
BRAINTOTEXT25_ROOT = _env_path("SSL_AUTORESEARCH_B2T25_ROOT", ROOT_DIR / "code" / "brain2text25")
BRAINTOTEXT25_HDF5_ROOT = _env_path(
    "SSL_AUTORESEARCH_B2T25_HDF5_ROOT",
    BRAINTOTEXT25_ROOT / "hdf5_data_final",
)

RANDOM_SEED = 7
DEFAULT_PROFILE = "single_gpu"
DEFAULT_DATASET_FAMILY = "brain2text25"
DEFAULT_PRIMARY_METRIC_NAME = "session_avg_val_bpphone"
DEFAULT_ADAPTATION_REGIME = "A"


@dataclass(frozen=True)
class RuntimeProfile:
    name: str
    device_priority: tuple[str, ...]
    preferred_batch_size: int
    num_workers: int
    pin_memory: bool
    persistent_workers: bool
    pretrain_budget_seconds: int
    probe_budget_seconds: int
    allow_torch_compile: bool


@dataclass(frozen=True)
class RunArtifacts:
    output_root: Path
    checkpoint_dir: Path
    run_dir: Path
    log_dir: Path
    inventory_dir: Path
    manifest_dir: Path


@dataclass(frozen=True)
class BenchmarkSummary:
    benchmark_state: str
    primary_metric_name: str
    primary_metric_value: float
    total_seconds: float
    pretrain_seconds: float
    probe_seconds: float
    device: str
    profile: str
    dataset_family: str
    backbone: str
    objective_family: str
    adaptation_regime: str
    patch_size: int
    patch_stride: int
    standardize_scope: str
    post_proj_norm: str
    num_source_sessions: int
    num_target_sessions: int
    checkpoint_path: str


RUNTIME_PROFILES: dict[str, RuntimeProfile] = {
    "local_debug": RuntimeProfile(
        name="local_debug",
        device_priority=("mps", "cuda", "cpu"),
        preferred_batch_size=16,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
        pretrain_budget_seconds=10 * 60,
        probe_budget_seconds=2 * 60,
        allow_torch_compile=False,
    ),
    "single_gpu": RuntimeProfile(
        name="single_gpu",
        device_priority=("cuda", "cpu"),
        preferred_batch_size=64,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        pretrain_budget_seconds=60 * 60,
        probe_budget_seconds=10 * 60,
        allow_torch_compile=True,
    ),
    "colab_cuda": RuntimeProfile(
        name="colab_cuda",
        device_priority=("cuda", "cpu"),
        preferred_batch_size=32,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
        pretrain_budget_seconds=40 * 60,
        probe_budget_seconds=8 * 60,
        allow_torch_compile=False,
    ),
    "cluster_cuda": RuntimeProfile(
        name="cluster_cuda",
        device_priority=("cuda", "cpu"),
        preferred_batch_size=64,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        pretrain_budget_seconds=2 * 60 * 60,
        probe_budget_seconds=15 * 60,
        allow_torch_compile=True,
    ),
}

SOURCE_ROOT_SPECS = {
    "ssl_tx_cache": {
        "env_var": "SSL_AUTORESEARCH_TX_CACHE_DIR",
        "path": TX_CACHE_DIR,
    },
    "ssl_sbp_cache": {
        "env_var": "SSL_AUTORESEARCH_SBP_CACHE_DIR",
        "path": SBP_CACHE_DIR,
    },
    "brain2text25_root": {
        "env_var": "SSL_AUTORESEARCH_B2T25_ROOT",
        "path": BRAINTOTEXT25_ROOT,
    },
    "brain2text25_hdf5": {
        "env_var": "SSL_AUTORESEARCH_B2T25_HDF5_ROOT",
        "path": BRAINTOTEXT25_HDF5_ROOT,
    },
}


def set_seed(seed: int = RANDOM_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def now() -> float:
    return time.time()


def resolve_profile(name: str) -> RuntimeProfile:
    try:
        return RUNTIME_PROFILES[name]
    except KeyError as exc:
        known = ", ".join(sorted(RUNTIME_PROFILES))
        raise ValueError(f"Unknown profile '{name}'. Expected one of: {known}") from exc


def detect_device(profile: RuntimeProfile) -> torch.device:
    for name in profile.device_priority:
        if name == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        if name == "mps" and torch.backends.mps.is_available():
            return torch.device("mps")
        if name == "cpu":
            return torch.device("cpu")
    return torch.device("cpu")


def resolve_source_root(root_key: str) -> Path:
    try:
        return SOURCE_ROOT_SPECS[root_key]["path"]
    except KeyError as exc:
        known = ", ".join(sorted(SOURCE_ROOT_SPECS))
        raise ValueError(f"Unknown source root key '{root_key}'. Expected one of: {known}") from exc


def source_root_metadata() -> dict[str, dict[str, str]]:
    return {
        key: {
            "env_var": str(spec["env_var"]),
            "resolved_path": str(spec["path"]),
        }
        for key, spec in SOURCE_ROOT_SPECS.items()
    }


def relative_to_root(path: Path, root_key: str) -> str:
    root = resolve_source_root(root_key)
    return str(path.resolve().relative_to(root))


def resolve_relative_path(root_key: str, relpath: str) -> Path:
    return resolve_source_root(root_key) / relpath


def ensure_artifact_dirs() -> RunArtifacts:
    checkpoint_dir = OUTPUT_ROOT / "checkpoints"
    run_dir = OUTPUT_ROOT / "runs"
    log_dir = OUTPUT_ROOT / "logs"
    inventory_dir = OUTPUT_ROOT / "inventories"
    manifest_dir = OUTPUT_ROOT / "manifests"
    for path in (checkpoint_dir, run_dir, log_dir, inventory_dir, manifest_dir):
        path.mkdir(parents=True, exist_ok=True)
    return RunArtifacts(
        output_root=OUTPUT_ROOT,
        checkpoint_dir=checkpoint_dir,
        run_dir=run_dir,
        log_dir=log_dir,
        inventory_dir=inventory_dir,
        manifest_dir=manifest_dir,
    )


def count_parameters(model: torch.nn.Module) -> int:
    return sum(param.numel() for param in model.parameters())


def make_run_slug(
    dataset_family: str,
    backbone: str,
    objective_family: str,
    adaptation_regime: str,
    patch_size: int,
    patch_stride: int,
    standardize_scope: str,
    post_proj_norm: str,
) -> str:
    return (
        f"{dataset_family}"
        f"__{backbone}"
        f"__{objective_family}"
        f"__{adaptation_regime}"
        f"__{standardize_scope}"
        f"__{post_proj_norm}"
        f"__p{patch_size}s{patch_stride}"
    )


def format_summary(summary: BenchmarkSummary) -> str:
    lines = [
        f"benchmark_state: {summary.benchmark_state}",
        f"primary_metric_name: {summary.primary_metric_name}",
        f"primary_metric_value: {summary.primary_metric_value:.10f}",
        f"total_seconds: {summary.total_seconds:.1f}",
        f"pretrain_seconds: {summary.pretrain_seconds:.1f}",
        f"probe_seconds: {summary.probe_seconds:.1f}",
        f"device: {summary.device}",
        f"profile: {summary.profile}",
        f"dataset_family: {summary.dataset_family}",
        f"backbone: {summary.backbone}",
        f"objective_family: {summary.objective_family}",
        f"adaptation_regime: {summary.adaptation_regime}",
        f"patch_size: {summary.patch_size}",
        f"patch_stride: {summary.patch_stride}",
        f"standardize_scope: {summary.standardize_scope}",
        f"post_proj_norm: {summary.post_proj_norm}",
        f"num_source_sessions: {summary.num_source_sessions}",
        f"num_target_sessions: {summary.num_target_sessions}",
        f"checkpoint_path: {summary.checkpoint_path}",
    ]
    return "\n".join(lines)

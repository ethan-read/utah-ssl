"""Run Brain2Text24-only future-prediction SSL on Modal.

This launcher now has two responsibilities:

1. Strictly verify whether Hugging Face Mamba can use optimized kernels in the
   actual Modal runtime.
2. Run a tiny stage-1 smoke only after one candidate image passes that
   verification.
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import modal


APP_NAME = "utah-ssl-future-prediction"
DEFAULT_CACHE_VOLUME_NAME = "utah-ssl-cache"
DEFAULT_OUTPUT_VOLUME_NAME = "utah-ssl-outputs"


def _resolve_local_repo_root() -> Path:
    resolved = Path(__file__).resolve()
    if len(resolved.parents) >= 3:
        return resolved.parents[2]
    return resolved.parent


LOCAL_REPO_ROOT = _resolve_local_repo_root()

LOCAL_CACHE_ROOT = Path("/Users/home/thesis/data/cache_v1")

REMOTE_REPO_ROOT = Path("/root/utah-ssl")
REMOTE_CACHE_VOLUME_ROOT = Path("/vol/cache")
REMOTE_OUTPUT_VOLUME_ROOT = Path("/vol/outputs")

DEFAULT_REMOTE_CACHE_SUBDIR = "cache_v1"
DEFAULT_REMOTE_STATS_SUBDIR = "stats"
DEFAULT_OUTPUT_SUBDIR = "ssl_experiments/modal_future_prediction_ssl"

DEFAULT_LOCAL_DATASET_DIR = LOCAL_CACHE_ROOT / "brain2text24"
DEFAULT_REMOTE_STATS_FILE = (
    REMOTE_CACHE_VOLUME_ROOT
    / DEFAULT_REMOTE_STATS_SUBDIR
    / "session_feature_stats"
    / "raw"
    / "tx_sbp"
    / "session"
    / "ssl_pretrain_brain2text24_only_v1.pt"
)


@dataclass(frozen=True)
class CandidateSpec:
    key: str
    description: str
    torch_version: str
    transformers_version: str
    mamba_ssm_version: str
    causal_conv1d_version: str


CANDIDATES: tuple[CandidateSpec, ...] = (
    CandidateSpec(
        key="baseline",
        description="Current baseline pins",
        torch_version="2.8.0",
        transformers_version="4.57.6",
        mamba_ssm_version="2.3.2.post1",
        causal_conv1d_version="1.6.2.post1",
    ),
    CandidateSpec(
        key="tf456",
        description="Transformers 4.56 candidate",
        torch_version="2.8.0",
        transformers_version="4.56.2",
        mamba_ssm_version="2.3.2.post1",
        causal_conv1d_version="1.6.2.post1",
    ),
    CandidateSpec(
        key="mamba22",
        description="Mamba-SSM 2.2 candidate",
        torch_version="2.8.0",
        transformers_version="4.57.6",
        mamba_ssm_version="2.2.4",
        causal_conv1d_version="1.6.2.post1",
    ),
)

CANDIDATE_BY_KEY = {spec.key: spec for spec in CANDIDATES}

cache_volume = modal.Volume.from_name(DEFAULT_CACHE_VOLUME_NAME, create_if_missing=True)
output_volume = modal.Volume.from_name(DEFAULT_OUTPUT_VOLUME_NAME, create_if_missing=True)


def _base_registry_image() -> modal.Image:
    return modal.Image.from_registry(
        "nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04",
        add_python="3.10",
    ).apt_install("git").env({"PYTHONPATH": str(REMOTE_REPO_ROOT)})


def _infra_image() -> modal.Image:
    return (
        _base_registry_image()
        .uv_pip_install(
            "numpy==2.1.2",
            "pandas==2.2.3",
            "torch==2.8.0",
        )
        .add_local_file(
            str(Path(__file__).resolve()),
            remote_path=str(REMOTE_REPO_ROOT / "run_future_prediction_ssl.py"),
            copy=True,
        )
        .add_local_dir(
            str(LOCAL_REPO_ROOT / "analysis"),
            remote_path=str(REMOTE_REPO_ROOT / "analysis"),
            copy=True,
        )
    )


def _build_candidate_image(spec: CandidateSpec) -> modal.Image:
    return (
        _base_registry_image()
        .uv_pip_install(
            "numpy==2.1.2",
            "pandas==2.2.3",
            f"torch=={spec.torch_version}",
            f"transformers=={spec.transformers_version}",
            "wheel",
            "einops",
        )
        .run_commands(
            (
                "python -m pip install "
                f"'causal-conv1d=={spec.causal_conv1d_version}' --no-build-isolation"
            ),
            (
                "python -m pip install "
                f"'mamba-ssm=={spec.mamba_ssm_version}' --no-build-isolation --no-deps"
            ),
        )
        .add_local_file(
            str(Path(__file__).resolve()),
            remote_path=str(REMOTE_REPO_ROOT / "run_future_prediction_ssl.py"),
            copy=True,
        )
        .add_local_dir(
            str(LOCAL_REPO_ROOT / "analysis"),
            remote_path=str(REMOTE_REPO_ROOT / "analysis"),
            copy=True,
        )
    )


app = modal.App(APP_NAME)


def _ensure_import_paths() -> None:
    repo_root = str(REMOTE_REPO_ROOT)
    analysis_root = str(REMOTE_REPO_ROOT / "analysis" / "active" / "ssl_experiments")
    for path in (repo_root, analysis_root):
        if path not in sys.path:
            sys.path.insert(0, path)


def _candidate_payload(spec: CandidateSpec) -> dict[str, object]:
    return asdict(spec)


def _requested_candidate_from_argv() -> str:
    argv = list(sys.argv[1:])
    for index, token in enumerate(argv):
        if token == "--candidate" and index + 1 < len(argv):
            return str(argv[index + 1])
        if token.startswith("--candidate="):
            return str(token.split("=", 1)[1])
    return "auto"


def _format_warning_message(*args: object, **kwargs: object) -> str:
    if not args:
        return ""
    message = str(args[0])
    if len(args) > 1:
        try:
            message = message % tuple(args[1:])
        except Exception:
            message = " ".join(str(part) for part in args)
    if kwargs:
        message = f"{message} | kwargs={kwargs}"
    return message


def _mamba_runtime_diagnostics(spec: CandidateSpec) -> dict[str, object]:
    import torch
    import transformers
    from transformers.models.mamba import modeling_mamba
    from transformers.models.mamba.configuration_mamba import MambaConfig

    payload: dict[str, object] = {
        "candidate": _candidate_payload(spec),
        "torch_version": str(torch.__version__),
        "transformers_version": str(transformers.__version__),
        "cuda_available": bool(torch.cuda.is_available()),
    }
    if torch.cuda.is_available():
        payload["cuda_device_name"] = str(torch.cuda.get_device_name(0))
    try:
        import mamba_ssm  # type: ignore[import-not-found]

        payload["mamba_ssm_version"] = str(getattr(mamba_ssm, "__version__", "unknown"))
    except Exception as exc:  # pragma: no cover - diagnostic only
        payload["mamba_ssm_version"] = {
            "import_ok": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
    try:
        import causal_conv1d  # type: ignore[import-not-found]

        payload["causal_conv1d_module"] = {
            "import_ok": True,
            "file": str(getattr(causal_conv1d, "__file__", "")),
            "version": str(getattr(causal_conv1d, "__version__", "unknown")),
            "has_causal_conv1d_fn": bool(hasattr(causal_conv1d, "causal_conv1d_fn")),
            "has_causal_conv1d_update": bool(hasattr(causal_conv1d, "causal_conv1d_update")),
        }
    except Exception as exc:  # pragma: no cover - diagnostic only
        payload["causal_conv1d_module"] = {
            "import_ok": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }

    op_states = {}
    for name in ("selective_state_update", "selective_scan_fn", "mamba_inner_fn"):
        value = getattr(modeling_mamba, name, None)
        op_states[name] = {
            "present": hasattr(modeling_mamba, name),
            "is_none": value is None,
            "type": None if value is None else type(value).__name__,
        }
    payload["required_ops"] = op_states

    lazy_payload: dict[str, object]
    lazy_update = None
    lazy_fn = None
    try:
        lazy_update, lazy_fn = modeling_mamba._lazy_load_causal_conv1d()
        lazy_payload = {
            "call_ok": True,
            "update_present": lazy_update is not None,
            "fn_present": lazy_fn is not None,
            "update_type": None if lazy_update is None else type(lazy_update).__name__,
            "fn_type": None if lazy_fn is None else type(lazy_fn).__name__,
        }
    except Exception as exc:  # pragma: no cover - diagnostic only
        lazy_payload = {
            "call_ok": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
    payload["lazy_causal_conv1d"] = lazy_payload

    warning_messages: list[str] = []
    mixer_exception: dict[str, str] | None = None
    original_warning_once = modeling_mamba.logger.warning_once

    def _capture_warning(*args: object, **kwargs: object) -> None:
        warning_messages.append(_format_warning_message(*args, **kwargs))

    modeling_mamba.logger.warning_once = _capture_warning
    try:
        test_config = MambaConfig(
            hidden_size=64,
            state_size=16,
            num_hidden_layers=1,
            expand=2,
            conv_kernel=4,
            use_mambapy=False,
        )
        modeling_mamba.MambaMixer(test_config, layer_idx=0)
    except Exception as exc:  # pragma: no cover - diagnostic only
        mixer_exception = {
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
    finally:
        modeling_mamba.logger.warning_once = original_warning_once

    payload["mixer_instantiation"] = {
        "ok": mixer_exception is None,
        "warning_messages": warning_messages,
        "warned_slow_path": bool(warning_messages),
        "exception": mixer_exception,
    }

    reasons: list[str] = []
    if not bool(payload["cuda_available"]):
        reasons.append("cuda_unavailable")
    for name in ("selective_state_update", "selective_scan_fn", "mamba_inner_fn"):
        state = op_states[name]
        if bool(state["is_none"]):
            reasons.append(f"{name}_missing")
    if not bool(lazy_payload.get("call_ok", False)):
        reasons.append("lazy_causal_conv1d_call_failed")
    else:
        if not bool(lazy_payload.get("update_present", False)):
            reasons.append("lazy_causal_conv1d_update_missing")
        if not bool(lazy_payload.get("fn_present", False)):
            reasons.append("lazy_causal_conv1d_fn_missing")
    if mixer_exception is not None:
        reasons.append("mixer_instantiation_failed")
    if warning_messages:
        reasons.append("mixer_warned_slow_path")

    payload["fast_path_ok"] = not reasons
    payload["failure_reasons"] = reasons
    return payload


def _verify_kernels_or_raise(spec: CandidateSpec) -> dict[str, object]:
    payload = _mamba_runtime_diagnostics(spec)
    print("Mamba kernel verification:", json.dumps(payload, sort_keys=True), flush=True)
    if not bool(payload["fast_path_ok"]):
        raise RuntimeError(json.dumps(payload, sort_keys=True))
    return payload


def _resolve_train_result(
    spec: CandidateSpec,
    *,
    run_name: str,
    cache_subdir: str,
    output_subdir: str,
    stats_path: str,
    segment_bins: int,
    temporal_bin_stride: int,
    future_bins: int,
    variance_match_weight: float,
    tx_loss_type: str,
    sbp_loss_type: str,
    ssl_steps: int,
    probe_steps: int,
    run_frozen_probe: bool,
    batch_size: int,
    probe_batch_size: int,
    probe_feature_source: str,
    probe_forecast_horizon_index: int,
    hidden_size: int,
    state_size: int,
    num_layers: int,
    seed: int,
    resume: bool,
    resume_checkpoint_path: str | None,
) -> dict[str, object]:
    _ensure_import_paths()

    from analysis.active.ssl_experiments.future_prediction_ssl import (
        FuturePredictionSSLConfig,
        run_future_prediction_ssl,
    )

    verification = _verify_kernels_or_raise(spec)

    cache_root = REMOTE_CACHE_VOLUME_ROOT / cache_subdir
    output_root = REMOTE_OUTPUT_VOLUME_ROOT / output_subdir
    output_root.mkdir(parents=True, exist_ok=True)

    config = FuturePredictionSSLConfig(
        seed=int(seed),
        backbone_type="mamba",
        dataset="brain2text24",
        pretrain_datasets=("brain2text24",),
        feature_mode="tx_sbp",
        boundary_key_mode="session",
        cache_root=str(cache_root),
        cache_mode="drive_direct",
        local_cache_base="/tmp/utah_ssl_cache",
        use_normalization=True,
        precomputed_session_stats_path=str(stats_path),
        tx_dim=128,
        sbp_dim=128,
        segment_bins=int(segment_bins),
        temporal_bin_stride=int(temporal_bin_stride),
        batch_size=int(batch_size),
        ssl_steps=int(ssl_steps),
        learning_rate=3e-4,
        weight_decay=1e-2,
        max_grad_norm=1.0,
        hidden_size=int(hidden_size),
        state_size=int(state_size),
        num_layers=int(num_layers),
        dropout=0.1,
        direction="causal",
        ffn_multiplier=2.0,
        input_mode="raw_bin",
        future_bins=int(future_bins),
        forecast_loss_delta=1.0,
        variance_match_weight=float(variance_match_weight),
        tx_loss_type=str(tx_loss_type),
        sbp_loss_type=str(sbp_loss_type),
        val_every_steps=100,
        val_batches=4,
        progress_every_steps=25,
        progress_every_seconds=30.0,
        checkpoint_every_steps=1000,
        resume=bool(resume),
        resume_checkpoint_path=resume_checkpoint_path,
        run_frozen_probe=bool(run_frozen_probe),
        probe_feature_source=str(probe_feature_source),
        probe_forecast_horizon_index=int(probe_forecast_horizon_index),
        probe_steps=int(probe_steps),
        probe_batch_size=int(probe_batch_size),
        probe_learning_rate=1e-3,
        probe_weight_decay=0.0,
        output_root=str(output_root),
        run_name=str(run_name),
    )

    print(f"Running in {REMOTE_REPO_ROOT}", flush=True)
    print(f"Candidate: {spec.key}", flush=True)
    print(f"Cache root: {cache_root}", flush=True)
    print(f"Stats path: {stats_path}", flush=True)
    print(f"Output root: {output_root}", flush=True)
    print(f"Run name: {run_name}", flush=True)

    summary = run_future_prediction_ssl(config)
    output_volume.commit()
    return {
        "candidate": _candidate_payload(spec),
        "gpu_request": ["L40S", "RTX-PRO-6000"],
        "cache_root": str(cache_root),
        "stats_path": str(stats_path),
        "output_root": str(output_root),
        "kernel_verification": verification,
        "summary": summary,
    }


infra_image = _infra_image()
VERIFY_FUNCTIONS: dict[str, Callable[..., Any]] = {}
TRAIN_FUNCTIONS: dict[str, Callable[..., Any]] = {}
REQUESTED_CANDIDATE = _requested_candidate_from_argv()
BASELINE_SPEC = CANDIDATE_BY_KEY["baseline"]
TF456_SPEC = CANDIDATE_BY_KEY["tf456"]
MAMBA22_SPEC = CANDIDATE_BY_KEY["mamba22"]


@app.function(
    image=infra_image,
    cpu=2,
    memory=4096,
    timeout=60 * 15,
    volumes={
        str(REMOTE_CACHE_VOLUME_ROOT): cache_volume.with_mount_options(read_only=True),
    },
)
def inspect_remote_cache(
    *,
    cache_subdir: str = DEFAULT_REMOTE_CACHE_SUBDIR,
    stats_subdir: str = DEFAULT_REMOTE_STATS_SUBDIR,
    dataset: str = "brain2text24",
) -> dict[str, object]:
    cache_root = REMOTE_CACHE_VOLUME_ROOT / cache_subdir
    dataset_dir = cache_root / dataset
    stats_path = (
        REMOTE_CACHE_VOLUME_ROOT
        / stats_subdir
        / "session_feature_stats"
        / "raw"
        / "tx_sbp"
        / "session"
        / "ssl_pretrain_brain2text24_only_v1.pt"
    )
    manifest_path = stats_path.with_suffix(".json")
    metadata_path = dataset_dir / "metadata.json"
    return {
        "cache_root": str(cache_root),
        "dataset_dir": str(dataset_dir),
        "dataset_exists": bool(dataset_dir.exists()),
        "dataset_metadata_exists": bool(metadata_path.exists()),
        "stats_path": str(stats_path),
        "stats_exists": bool(stats_path.exists()),
        "stats_manifest_exists": bool(manifest_path.exists()),
    }


@app.function(
    image=infra_image,
    cpu=8,
    memory=16384,
    timeout=60 * 60 * 6,
    volumes={
        str(REMOTE_CACHE_VOLUME_ROOT): cache_volume,
    },
)
def recompute_remote_session_stats(
    *,
    cache_subdir: str = DEFAULT_REMOTE_CACHE_SUBDIR,
    stats_subdir: str = DEFAULT_REMOTE_STATS_SUBDIR,
    feature_mode: str = "tx_sbp",
    boundary_key_mode: str = "session",
    dataset: str = "brain2text24",
    tx_dim: int = 128,
    sbp_dim: int = 128,
    segment_bins: int = 256,
    examples_per_shard: int = 8,
    seed: int = 7,
    overwrite: bool = True,
) -> dict[str, object]:
    _ensure_import_paths()

    from analysis.active.ssl_experiments.recompute_session_feature_stats import (
        recompute_session_feature_stats,
    )

    cache_root = REMOTE_CACHE_VOLUME_ROOT / cache_subdir
    output_path = (
        REMOTE_CACHE_VOLUME_ROOT
        / stats_subdir
        / "session_feature_stats"
        / "raw"
        / feature_mode
        / boundary_key_mode
        / "ssl_pretrain_brain2text24_only_v1.pt"
    )

    result = recompute_session_feature_stats(
        cache_root=cache_root,
        output_path=output_path,
        feature_mode=feature_mode,
        boundary_key_mode=boundary_key_mode,
        datasets=(dataset,),
        tx_dim=int(tx_dim),
        sbp_dim=int(sbp_dim),
        segment_bins=int(segment_bins),
        seed=int(seed),
        examples_per_shard=int(examples_per_shard),
        overwrite=bool(overwrite),
    )
    cache_volume.commit()
    return {
        "cache_root": str(cache_root),
        "output_path": str(result["output_path"]),
        "metadata_path": str(result["metadata_path"]),
        "metadata": result["metadata"],
        "session_count": int(result["session_count"]),
        "dataset_count": int(result["dataset_count"]),
    }


if REQUESTED_CANDIDATE == "baseline":
    @app.function(
        name="verify-mamba-kernels-baseline",
        image=_build_candidate_image(BASELINE_SPEC),
        serialized=True,
        gpu=["L40S", "RTX-PRO-6000"],
        cpu=4,
        memory=16384,
        timeout=60 * 30,
    )
    def verify_mamba_kernels_baseline() -> dict[str, object]:
        return _verify_kernels_or_raise(BASELINE_SPEC)


    @app.function(
        name="train-future-prediction-baseline",
        image=_build_candidate_image(BASELINE_SPEC),
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
    def train_future_prediction_baseline(
        *,
        run_name: str,
        cache_subdir: str = DEFAULT_REMOTE_CACHE_SUBDIR,
        output_subdir: str = DEFAULT_OUTPUT_SUBDIR,
        stats_path: str = str(DEFAULT_REMOTE_STATS_FILE),
        segment_bins: int = 256,
        temporal_bin_stride: int = 1,
        future_bins: int = 3,
        variance_match_weight: float = 0.05,
        tx_loss_type: str = "huber",
        sbp_loss_type: str = "huber",
        ssl_steps: int = 60000,
        probe_steps: int = 12000,
        run_frozen_probe: bool = True,
        batch_size: int = 16,
        probe_batch_size: int = 8,
        probe_feature_source: str = "encoder_hidden",
        probe_forecast_horizon_index: int = 0,
        hidden_size: int = 256,
        state_size: int = 64,
        num_layers: int = 4,
        seed: int = 7,
        resume: bool = False,
        resume_checkpoint_path: str | None = None,
    ) -> dict[str, object]:
        return _resolve_train_result(
            BASELINE_SPEC,
            run_name=run_name,
            cache_subdir=cache_subdir,
            output_subdir=output_subdir,
            stats_path=stats_path,
            segment_bins=segment_bins,
            temporal_bin_stride=temporal_bin_stride,
            future_bins=future_bins,
            variance_match_weight=variance_match_weight,
            tx_loss_type=tx_loss_type,
            sbp_loss_type=sbp_loss_type,
            ssl_steps=ssl_steps,
            probe_steps=probe_steps,
            run_frozen_probe=run_frozen_probe,
            batch_size=batch_size,
            probe_batch_size=probe_batch_size,
            probe_feature_source=probe_feature_source,
            probe_forecast_horizon_index=probe_forecast_horizon_index,
            hidden_size=hidden_size,
            state_size=state_size,
            num_layers=num_layers,
            seed=seed,
            resume=resume,
            resume_checkpoint_path=resume_checkpoint_path,
        )


    VERIFY_FUNCTIONS["baseline"] = verify_mamba_kernels_baseline
    TRAIN_FUNCTIONS["baseline"] = train_future_prediction_baseline

if REQUESTED_CANDIDATE == "tf456":
    @app.function(
        name="verify-mamba-kernels-tf456",
        image=_build_candidate_image(TF456_SPEC),
        serialized=True,
        gpu=["L40S", "RTX-PRO-6000"],
        cpu=4,
        memory=16384,
        timeout=60 * 30,
    )
    def verify_mamba_kernels_tf456() -> dict[str, object]:
        return _verify_kernels_or_raise(TF456_SPEC)


    @app.function(
        name="train-future-prediction-tf456",
        image=_build_candidate_image(TF456_SPEC),
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
    def train_future_prediction_tf456(
        *,
        run_name: str,
        cache_subdir: str = DEFAULT_REMOTE_CACHE_SUBDIR,
        output_subdir: str = DEFAULT_OUTPUT_SUBDIR,
        stats_path: str = str(DEFAULT_REMOTE_STATS_FILE),
        segment_bins: int = 256,
        temporal_bin_stride: int = 1,
        future_bins: int = 3,
        variance_match_weight: float = 0.05,
        tx_loss_type: str = "huber",
        sbp_loss_type: str = "huber",
        ssl_steps: int = 60000,
        probe_steps: int = 12000,
        run_frozen_probe: bool = True,
        batch_size: int = 16,
        probe_batch_size: int = 8,
        probe_feature_source: str = "encoder_hidden",
        probe_forecast_horizon_index: int = 0,
        hidden_size: int = 256,
        state_size: int = 64,
        num_layers: int = 4,
        seed: int = 7,
        resume: bool = False,
        resume_checkpoint_path: str | None = None,
    ) -> dict[str, object]:
        return _resolve_train_result(
            TF456_SPEC,
            run_name=run_name,
            cache_subdir=cache_subdir,
            output_subdir=output_subdir,
            stats_path=stats_path,
            segment_bins=segment_bins,
            temporal_bin_stride=temporal_bin_stride,
            future_bins=future_bins,
            variance_match_weight=variance_match_weight,
            tx_loss_type=tx_loss_type,
            sbp_loss_type=sbp_loss_type,
            ssl_steps=ssl_steps,
            probe_steps=probe_steps,
            run_frozen_probe=run_frozen_probe,
            batch_size=batch_size,
            probe_batch_size=probe_batch_size,
            probe_feature_source=probe_feature_source,
            probe_forecast_horizon_index=probe_forecast_horizon_index,
            hidden_size=hidden_size,
            state_size=state_size,
            num_layers=num_layers,
            seed=seed,
            resume=resume,
            resume_checkpoint_path=resume_checkpoint_path,
        )


    VERIFY_FUNCTIONS["tf456"] = verify_mamba_kernels_tf456
    TRAIN_FUNCTIONS["tf456"] = train_future_prediction_tf456

if REQUESTED_CANDIDATE == "mamba22":
    @app.function(
        name="verify-mamba-kernels-mamba22",
        image=_build_candidate_image(MAMBA22_SPEC),
        serialized=True,
        gpu=["L40S", "RTX-PRO-6000"],
        cpu=4,
        memory=16384,
        timeout=60 * 30,
    )
    def verify_mamba_kernels_mamba22() -> dict[str, object]:
        return _verify_kernels_or_raise(MAMBA22_SPEC)


    @app.function(
        name="train-future-prediction-mamba22",
        image=_build_candidate_image(MAMBA22_SPEC),
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
    def train_future_prediction_mamba22(
        *,
        run_name: str,
        cache_subdir: str = DEFAULT_REMOTE_CACHE_SUBDIR,
        output_subdir: str = DEFAULT_OUTPUT_SUBDIR,
        stats_path: str = str(DEFAULT_REMOTE_STATS_FILE),
        segment_bins: int = 256,
        temporal_bin_stride: int = 1,
        future_bins: int = 3,
        variance_match_weight: float = 0.05,
        tx_loss_type: str = "huber",
        sbp_loss_type: str = "huber",
        ssl_steps: int = 60000,
        probe_steps: int = 12000,
        run_frozen_probe: bool = True,
        batch_size: int = 16,
        probe_batch_size: int = 8,
        probe_feature_source: str = "encoder_hidden",
        probe_forecast_horizon_index: int = 0,
        hidden_size: int = 256,
        state_size: int = 64,
        num_layers: int = 4,
        seed: int = 7,
        resume: bool = False,
        resume_checkpoint_path: str | None = None,
    ) -> dict[str, object]:
        return _resolve_train_result(
            MAMBA22_SPEC,
            run_name=run_name,
            cache_subdir=cache_subdir,
            output_subdir=output_subdir,
            stats_path=stats_path,
            segment_bins=segment_bins,
            temporal_bin_stride=temporal_bin_stride,
            future_bins=future_bins,
            variance_match_weight=variance_match_weight,
            tx_loss_type=tx_loss_type,
            sbp_loss_type=sbp_loss_type,
            ssl_steps=ssl_steps,
            probe_steps=probe_steps,
            run_frozen_probe=run_frozen_probe,
            batch_size=batch_size,
            probe_batch_size=probe_batch_size,
            probe_feature_source=probe_feature_source,
            probe_forecast_horizon_index=probe_forecast_horizon_index,
            hidden_size=hidden_size,
            state_size=state_size,
            num_layers=num_layers,
            seed=seed,
            resume=resume,
            resume_checkpoint_path=resume_checkpoint_path,
        )


    VERIFY_FUNCTIONS["mamba22"] = verify_mamba_kernels_mamba22
    TRAIN_FUNCTIONS["mamba22"] = train_future_prediction_mamba22


def _candidate_order(candidate: str) -> list[CandidateSpec]:
    if candidate == "auto":
        return list(CANDIDATES)
    if candidate not in CANDIDATE_BY_KEY:
        raise KeyError(f"Unknown candidate {candidate!r}. Available: {sorted(CANDIDATE_BY_KEY)}")
    return [CANDIDATE_BY_KEY[candidate]]


def _run_candidate_verification(candidate: CandidateSpec) -> dict[str, object]:
    return VERIFY_FUNCTIONS[candidate.key].remote()


def _bool_flag(name: str, value: bool) -> list[str]:
    return [f"--{name}" if value else f"--no-{name}"]


def _run_candidate_subprocess(args: list[str]) -> subprocess.CompletedProcess[str]:
    script_path = str(Path(__file__).resolve())
    command = ["modal", "run", script_path, *args]
    return subprocess.run(command, text=True, capture_output=True, check=False)


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


def _sync_cache_if_requested(
    *,
    sync_cache: bool,
    remote_cache_subdir: str,
    remote_stats_subdir: str,
) -> None:
    if not sync_cache:
        return
    remote_cache_state = inspect_remote_cache.remote(
        cache_subdir=remote_cache_subdir,
        stats_subdir=remote_stats_subdir,
        dataset="brain2text24",
    )
    print(json.dumps({"remote_cache_before_sync": remote_cache_state}, indent=2, sort_keys=True))
    if not bool(remote_cache_state["dataset_exists"]):
        _upload_directory_to_volume(
            volume=cache_volume,
            local_dir=DEFAULT_LOCAL_DATASET_DIR,
            remote_dir=f"/{remote_cache_subdir}/brain2text24",
        )


@app.local_entrypoint()
def main(
    run_name: str = "future_pred_mamba_b2t24_txsbp_l40s",
    candidate: str = "auto",
    verify_kernels_only: bool = False,
    sync_cache: bool = False,
    refresh_stats: bool = True,
    resume: bool = False,
    resume_checkpoint_path: str = "",
    remote_cache_subdir: str = DEFAULT_REMOTE_CACHE_SUBDIR,
    remote_stats_subdir: str = DEFAULT_REMOTE_STATS_SUBDIR,
    output_subdir: str = DEFAULT_OUTPUT_SUBDIR,
    segment_bins: int = 256,
    temporal_bin_stride: int = 1,
    future_bins: int = 3,
    variance_match_weight: float = 0.05,
    tx_loss_type: str = "huber",
    sbp_loss_type: str = "huber",
    ssl_steps: int = 2,
    probe_steps: int = 2,
    run_frozen_probe: bool = False,
    batch_size: int = 2,
    probe_batch_size: int = 2,
    probe_feature_source: str = "encoder_hidden",
    probe_forecast_horizon_index: int = 0,
    hidden_size: int = 64,
    state_size: int = 16,
    num_layers: int = 1,
    seed: int = 7,
) -> None:
    if candidate == "auto":
        verification_results: list[dict[str, object]] = []
        selected_candidate: CandidateSpec | None = None
        for spec in CANDIDATES:
            proc = _run_candidate_subprocess(
                [
                    "--candidate",
                    spec.key,
                    "--verify-kernels-only",
                ]
            )
            if proc.stdout:
                print(proc.stdout, end="")
            if proc.stderr:
                print(proc.stderr, end="", file=sys.stderr)
            result = {
                "candidate": _candidate_payload(spec),
                "returncode": int(proc.returncode),
            }
            if proc.returncode == 0:
                result["status"] = "passed"
                verification_results.append(result)
                selected_candidate = spec
                break
            result["status"] = "failed"
            verification_results.append(result)
        if selected_candidate is None:
            raise RuntimeError(
                "No kernel candidate passed strict verification.\n"
                + json.dumps({"verification_results": verification_results}, indent=2, sort_keys=True)
            )
        if verify_kernels_only:
            print(
                json.dumps(
                    {
                        "selected_candidate": _candidate_payload(selected_candidate),
                        "verification_results": verification_results,
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return
        train_args = [
            "--candidate",
            selected_candidate.key,
            "--run-name",
            run_name,
            "--remote-cache-subdir",
            remote_cache_subdir,
            "--remote-stats-subdir",
            remote_stats_subdir,
            "--output-subdir",
            output_subdir,
            "--segment-bins",
            str(segment_bins),
            "--temporal-bin-stride",
            str(temporal_bin_stride),
            "--future-bins",
            str(future_bins),
            "--variance-match-weight",
            str(variance_match_weight),
            "--tx-loss-type",
            str(tx_loss_type),
            "--sbp-loss-type",
            str(sbp_loss_type),
            "--ssl-steps",
            str(ssl_steps),
            "--probe-steps",
            str(probe_steps),
            "--batch-size",
            str(batch_size),
            "--probe-batch-size",
            str(probe_batch_size),
            "--probe-feature-source",
            str(probe_feature_source),
            "--probe-forecast-horizon-index",
            str(probe_forecast_horizon_index),
            "--hidden-size",
            str(hidden_size),
            "--state-size",
            str(state_size),
            "--num-layers",
            str(num_layers),
            "--seed",
            str(seed),
            *_bool_flag("sync-cache", sync_cache),
            *_bool_flag("refresh-stats", refresh_stats),
            *_bool_flag("resume", resume),
            *_bool_flag("run-frozen-probe", run_frozen_probe),
        ]
        if resume_checkpoint_path.strip():
            train_args.extend(["--resume-checkpoint-path", resume_checkpoint_path])
        proc = _run_candidate_subprocess(train_args)
        if proc.stdout:
            print(proc.stdout, end="")
        if proc.stderr:
            print(proc.stderr, end="", file=sys.stderr)
        if proc.returncode != 0:
            raise RuntimeError(
                "Smoke training failed after kernel verification.\n"
                + json.dumps(
                    {
                        "selected_candidate": _candidate_payload(selected_candidate),
                        "returncode": int(proc.returncode),
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
        return

    verification_results: list[dict[str, object]] = []
    selected_candidate: CandidateSpec | None = None

    for spec in _candidate_order(candidate):
        try:
            result = _run_candidate_verification(spec)
        except Exception as exc:
            failure = {
                "candidate": _candidate_payload(spec),
                "status": "failed",
                "error": str(exc),
            }
            verification_results.append(failure)
            print(json.dumps({"kernel_verification": failure}, indent=2, sort_keys=True))
            continue
        success = {
            "candidate": _candidate_payload(spec),
            "status": "passed",
            "result": result,
        }
        verification_results.append(success)
        print(json.dumps({"kernel_verification": success}, indent=2, sort_keys=True))
        selected_candidate = spec
        break

    if selected_candidate is None:
        raise RuntimeError(
            "No kernel candidate passed strict verification.\n"
            + json.dumps({"verification_results": verification_results}, indent=2, sort_keys=True)
        )

    if verify_kernels_only:
        print(
            json.dumps(
                {
                    "selected_candidate": _candidate_payload(selected_candidate),
                    "verification_results": verification_results,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return

    _sync_cache_if_requested(
        sync_cache=sync_cache,
        remote_cache_subdir=remote_cache_subdir,
        remote_stats_subdir=remote_stats_subdir,
    )

    stats_path = str(
        REMOTE_CACHE_VOLUME_ROOT
        / remote_stats_subdir
        / "session_feature_stats"
        / "raw"
        / "tx_sbp"
        / "session"
        / "ssl_pretrain_brain2text24_only_v1.pt"
    )

    if refresh_stats:
        stats_result = recompute_remote_session_stats.remote(
            cache_subdir=remote_cache_subdir,
            stats_subdir=remote_stats_subdir,
            feature_mode="tx_sbp",
            boundary_key_mode="session",
            dataset="brain2text24",
            tx_dim=128,
            sbp_dim=128,
            segment_bins=int(segment_bins),
            examples_per_shard=8,
            seed=seed,
            overwrite=True,
        )
        print(json.dumps({"remote_stats_refresh": stats_result}, indent=2, sort_keys=True))

    train_result = TRAIN_FUNCTIONS[selected_candidate.key].remote(
        run_name=run_name,
        cache_subdir=remote_cache_subdir,
        output_subdir=output_subdir,
        stats_path=stats_path,
        segment_bins=segment_bins,
        temporal_bin_stride=temporal_bin_stride,
        future_bins=future_bins,
        variance_match_weight=variance_match_weight,
        tx_loss_type=tx_loss_type,
        sbp_loss_type=sbp_loss_type,
        ssl_steps=ssl_steps,
        probe_steps=probe_steps,
        run_frozen_probe=run_frozen_probe,
        batch_size=batch_size,
        probe_batch_size=probe_batch_size,
        probe_feature_source=probe_feature_source,
        probe_forecast_horizon_index=probe_forecast_horizon_index,
        hidden_size=hidden_size,
        state_size=state_size,
        num_layers=num_layers,
        seed=seed,
        resume=resume,
        resume_checkpoint_path=resume_checkpoint_path.strip() or None,
    )
    print(
        json.dumps(
            {
                "selected_candidate": _candidate_payload(selected_candidate),
                "verification_results": verification_results,
                "train_result": train_result,
            },
            indent=2,
            sort_keys=True,
        )
    )

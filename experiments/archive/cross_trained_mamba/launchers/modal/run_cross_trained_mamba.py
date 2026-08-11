"""Run supervised cross-trained area-6v Mamba training on Modal.

This launcher verifies that Hugging Face Mamba can use optimized kernels in the
actual Modal runtime before starting training. If the fast path is unavailable,
the run fails closed.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import modal


APP_NAME = "utah-ssl-cross-trained-mamba"
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

DEFAULT_CACHE_SUBDIR = "cache_v1"
DEFAULT_OUTPUT_SUBDIR = "ssl_experiments/modal_cross_trained_mamba"
DEFAULT_RUN_NAME = "cross_mamba_b2t24b2t25_txsbp_area6v_native20ms_h512_hctc_fb_affine_seed7_90k"


@dataclass(frozen=True)
class CandidateSpec:
    key: str
    description: str
    torch_version: str
    transformers_version: str
    mamba_ssm_version: str
    causal_conv1d_version: str


BASELINE_SPEC = CandidateSpec(
    key="baseline",
    description="Future-prediction baseline pins",
    torch_version="2.8.0",
    transformers_version="4.57.6",
    mamba_ssm_version="2.3.2.post1",
    causal_conv1d_version="1.6.2.post1",
)

cache_volume = modal.Volume.from_name(DEFAULT_CACHE_VOLUME_NAME, create_if_missing=True)
output_volume = modal.Volume.from_name(DEFAULT_OUTPUT_VOLUME_NAME, create_if_missing=True)


def _base_registry_image() -> modal.Image:
    return modal.Image.from_registry(
        "nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04",
        add_python="3.10",
    ).apt_install("git").env({"PYTHONPATH": str(REMOTE_REPO_ROOT)})


def _build_image(spec: CandidateSpec) -> modal.Image:
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
            remote_path=str(REMOTE_REPO_ROOT / "run_cross_trained_mamba.py"),
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


def _candidate_payload(spec: CandidateSpec) -> dict[str, object]:
    return asdict(spec)


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

    op_states: dict[str, dict[str, object]] = {}
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
            "update_is_none": lazy_update is None,
            "fn_is_none": lazy_fn is None,
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
    mixer_exception: str | None = None
    original_warning_once = modeling_mamba.logger.warning_once

    def _capture_warning(*args: object, **kwargs: object) -> None:
        warning_messages.append(_format_warning_message(*args, **kwargs))

    modeling_mamba.logger.warning_once = _capture_warning
    try:
        test_config = MambaConfig(
            vocab_size=1,
            hidden_size=64,
            state_size=16,
            num_hidden_layers=1,
            conv_kernel=4,
            use_mambapy=False,
        )
        modeling_mamba.MambaMixer(test_config, layer_idx=0)
    except Exception as exc:  # pragma: no cover - diagnostic only
        mixer_exception = f"{type(exc).__name__}: {exc}"
    finally:
        modeling_mamba.logger.warning_once = original_warning_once

    payload["mixer_warning_messages"] = warning_messages
    payload["mixer_exception"] = mixer_exception

    reasons: list[str] = []
    for name in ("selective_state_update", "selective_scan_fn", "mamba_inner_fn"):
        if bool(op_states[name]["is_none"]):
            reasons.append(f"{name}_missing")
    if not lazy_payload.get("call_ok", False):
        reasons.append("lazy_causal_conv1d_load_failed")
    else:
        if lazy_update is None:
            reasons.append("lazy_causal_conv1d_update_missing")
        if lazy_fn is None:
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


@app.function(
    name="verify-mamba-kernels-cross-trained",
    image=_build_image(BASELINE_SPEC),
    serialized=True,
    gpu=["L40S", "RTX-PRO-6000"],
    cpu=4,
    memory=16384,
    timeout=60 * 30,
)
def verify_mamba_kernels() -> dict[str, object]:
    return _verify_kernels_or_raise(BASELINE_SPEC)


@app.function(
    name="train-cross-trained-mamba",
    image=_build_image(BASELINE_SPEC),
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
def train_cross_trained_mamba(
    *,
    run_name: str = DEFAULT_RUN_NAME,
    cache_subdir: str = DEFAULT_CACHE_SUBDIR,
    output_subdir: str = DEFAULT_OUTPUT_SUBDIR,
    max_steps: int = 90000,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    min_learning_rate: float = 1e-5,
    warmup_steps: int = 1000,
    weight_decay: float = 1e-5,
    max_grad_norm: float = 10.0,
    val_every_steps: int = 100,
    checkpoint_every_steps: int = 500,
    progress_every_steps: int = 25,
    hidden_size: int = 512,
    state_size: int = 64,
    stage1_num_layers: int = 2,
    stage2_num_layers: int = 2,
    stage3_num_layers: int = 1,
    dropout: float = 0.1,
    seed: int = 7,
    resume_latest: bool = False,
    resume_checkpoint_path: str | None = None,
) -> dict[str, object]:
    _ensure_import_paths()

    from experiments.archive.cross_trained_mamba import CrossTrainedMambaConfig
    from experiments.archive.cross_trained_mamba.train import run_cross_trained_mamba_with_callbacks

    verification = _verify_kernels_or_raise(BASELINE_SPEC)
    cache_root = REMOTE_CACHE_VOLUME_ROOT / cache_subdir
    output_root = REMOTE_OUTPUT_VOLUME_ROOT / output_subdir
    output_root.mkdir(parents=True, exist_ok=True)

    config = CrossTrainedMambaConfig(
        seed=int(seed),
        datasets=("brain2text24", "brain2text25"),
        feature_mode="tx_sbp",
        area6v_feature_dim=128,
        cache_root=str(cache_root),
        output_root=str(output_root),
        run_name=str(run_name),
        batch_size=int(batch_size),
        max_steps=int(max_steps),
        learning_rate=float(learning_rate),
        min_learning_rate=float(min_learning_rate),
        warmup_steps=int(warmup_steps),
        weight_decay=float(weight_decay),
        max_grad_norm=float(max_grad_norm),
        val_every_steps=int(val_every_steps),
        checkpoint_every_steps=int(checkpoint_every_steps),
        progress_every_steps=int(progress_every_steps),
        hidden_size=int(hidden_size),
        state_size=int(state_size),
        stage1_num_layers=int(stage1_num_layers),
        stage2_num_layers=int(stage2_num_layers),
        stage3_num_layers=int(stage3_num_layers),
        dropout=float(dropout),
        adapter_mode="affine",
        feedback_detach=False,
        resume_latest=bool(resume_latest),
        resume_checkpoint_path=resume_checkpoint_path,
    )

    print(f"Running in {REMOTE_REPO_ROOT}", flush=True)
    print(f"Cache root: {cache_root}", flush=True)
    print(f"Output root: {output_root}", flush=True)
    print(f"Run name: {run_name}", flush=True)

    summary = run_cross_trained_mamba_with_callbacks(
        config,
        commit_callback=output_volume.commit,
    )
    output_volume.commit()
    return {
        "candidate": _candidate_payload(BASELINE_SPEC),
        "gpu_request": ["L40S", "RTX-PRO-6000"],
        "cache_root": str(cache_root),
        "output_root": str(output_root),
        "kernel_verification": verification,
        "summary": summary,
    }


@app.local_entrypoint()
def main(
    run_name: str = DEFAULT_RUN_NAME,
    cache_subdir: str = DEFAULT_CACHE_SUBDIR,
    output_subdir: str = DEFAULT_OUTPUT_SUBDIR,
    max_steps: int = 90000,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    min_learning_rate: float = 1e-5,
    warmup_steps: int = 1000,
    weight_decay: float = 1e-5,
    max_grad_norm: float = 10.0,
    val_every_steps: int = 100,
    checkpoint_every_steps: int = 500,
    progress_every_steps: int = 25,
    hidden_size: int = 512,
    state_size: int = 64,
    stage1_num_layers: int = 2,
    stage2_num_layers: int = 2,
    stage3_num_layers: int = 1,
    dropout: float = 0.1,
    seed: int = 7,
    resume_latest: bool = False,
    resume_checkpoint_path: str = "",
    verify_kernels_only: bool = False,
) -> None:
    if bool(verify_kernels_only):
        result = verify_mamba_kernels.remote()
    else:
        result = train_cross_trained_mamba.remote(
            run_name=run_name,
            cache_subdir=cache_subdir,
            output_subdir=output_subdir,
            max_steps=max_steps,
            batch_size=batch_size,
            learning_rate=learning_rate,
            min_learning_rate=min_learning_rate,
            warmup_steps=warmup_steps,
            weight_decay=weight_decay,
            max_grad_norm=max_grad_norm,
            val_every_steps=val_every_steps,
            checkpoint_every_steps=checkpoint_every_steps,
            progress_every_steps=progress_every_steps,
            hidden_size=hidden_size,
            state_size=state_size,
            stage1_num_layers=stage1_num_layers,
            stage2_num_layers=stage2_num_layers,
            stage3_num_layers=stage3_num_layers,
            dropout=dropout,
            seed=seed,
            resume_latest=resume_latest,
            resume_checkpoint_path=resume_checkpoint_path or None,
        )
    print(json.dumps(result, indent=2, sort_keys=True))

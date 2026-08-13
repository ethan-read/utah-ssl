"""Convert Stanford release TensorFlow RNN checkpoints to local PyTorch format."""

from __future__ import annotations

import json
import tarfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .checkpointing import adapter_keys_from_problem, build_willett_model
from .config import WillettReconstructionConfig
from .data import build_willett_problem

RELEASED_SESSIONS: tuple[str, ...] = (
    "t12.2022.04.28",
    "t12.2022.05.05",
    "t12.2022.05.17",
    "t12.2022.05.19",
    "t12.2022.05.24",
    "t12.2022.05.26",
    "t12.2022.06.02",
    "t12.2022.06.07",
    "t12.2022.06.14",
    "t12.2022.06.16",
    "t12.2022.06.21",
    "t12.2022.06.28",
    "t12.2022.07.05",
    "t12.2022.07.14",
    "t12.2022.07.21",
    "t12.2022.07.27",
    "t12.2022.08.02",
    "t12.2022.08.11",
    "t12.2022.08.13",
)


def ensure_released_archive_extracted(*, archive_path: str | Path, derived_root: str | Path) -> Path:
    """Extract the Stanford `derived.tar.gz` archive if the derived folder is absent."""
    archive_path = Path(archive_path)
    derived_root = Path(derived_root)
    baseline_dir = derived_root / "rnns" / "baselineRelease"
    if (baseline_dir / "checkpoint").exists() and (baseline_dir / "args.yaml").exists():
        return derived_root
    if not archive_path.exists():
        raise FileNotFoundError(f"Released derived archive not found: {archive_path}")
    derived_root.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "r:gz") as archive:
        archive.extractall(derived_root.parent)
    if not (baseline_dir / "checkpoint").exists():
        raise FileNotFoundError(f"Extracted archive did not create expected checkpoint directory: {baseline_dir}")
    return derived_root


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ModuleNotFoundError as exc:  # pragma: no cover - notebook environments install pyyaml
        raise ModuleNotFoundError("pyyaml is required to read the released args.yaml file.") from exc
    payload = yaml.safe_load(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"Expected mapping in {path}")
    return payload


def _latest_checkpoint_prefix(checkpoint_dir: Path) -> Path:
    checkpoint_file = checkpoint_dir / "checkpoint"
    if not checkpoint_file.exists():
        raise FileNotFoundError(f"Missing TensorFlow checkpoint state file: {checkpoint_file}")
    for line in checkpoint_file.read_text().splitlines():
        if line.startswith("model_checkpoint_path:"):
            ckpt_name = line.split('"', 2)[1]
            return checkpoint_dir / ckpt_name
    raise ValueError(f"Could not resolve model_checkpoint_path in {checkpoint_file}")


def _tf_reader(checkpoint_prefix: Path):
    try:
        import tensorflow as tf
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on notebook runtime
        raise ModuleNotFoundError(
            "TensorFlow is required once to convert the released Stanford checkpoint. "
            "Install tensorflow/tensorflow-cpu, rerun conversion, then downstream export is pure PyTorch."
        ) from exc
    return tf.train.load_checkpoint(str(checkpoint_prefix))


def _read_var(reader: Any, name: str) -> np.ndarray:
    shape_map = reader.get_variable_to_shape_map()
    if name not in shape_map:
        raise KeyError(f"TensorFlow checkpoint variable not found: {name}")
    return np.asarray(reader.get_tensor(name))


def _reorder_keras_gru_gates(value: np.ndarray) -> np.ndarray:
    """Map Keras GRU gate order [z, r, h] to PyTorch [r, z, n]."""
    value = np.asarray(value)
    gate_axis = value.ndim - 1
    chunks = np.split(value, 3, axis=gate_axis)
    return np.concatenate([chunks[1], chunks[0], chunks[2]], axis=gate_axis)


def _to_tensor(value: np.ndarray) -> torch.Tensor:
    return torch.as_tensor(np.asarray(value), dtype=torch.float32)


def _assign_linear(state: dict[str, torch.Tensor], prefix: str, kernel: np.ndarray, bias: np.ndarray) -> None:
    state[f"{prefix}.weight"] = _to_tensor(kernel.T)
    state[f"{prefix}.bias"] = _to_tensor(bias)


def _reorder_stanford_logits_to_local_vocab(kernel: np.ndarray, bias: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Map Stanford output order [AA..SIL, BLANK] to local [BLANK, AA..SIL]."""
    kernel = np.asarray(kernel)
    bias = np.asarray(bias)
    if kernel.shape[-1] != 41 or bias.shape[0] != 41:
        raise ValueError(
            "Expected released Stanford classifier to have 41 outputs "
            f"(got kernel {kernel.shape}, bias {bias.shape})."
        )
    order = [40] + list(range(40))
    return kernel[:, order], bias[order]


def _session_for_adapter_key(adapter_key: str, released_sessions: tuple[str, ...]) -> str | None:
    if adapter_key in released_sessions:
        return adapter_key
    for session in released_sessions:
        if session in adapter_key:
            return session
    return None


def released_rnn_config(*, cache_root: str | Path) -> WillettReconstructionConfig:
    """Return the local PyTorch config matching the released Stanford baseline RNN."""
    return WillettReconstructionConfig(
        dataset="brain2text24",
        feature_mode="tx_sbp",
        boundary_key_mode="session",
        split_policy="competition_train_test",
        normalization_mode="none",
        batch_size=64,
        input_projection_size=256,
        input_projection_dropout=0.2,
        decoder_backbone_type="gru",
        gru_hidden_size=1024,
        gru_num_layers=5,
        gru_dropout=0.4,
        patch_size=32,
        patch_stride=4,
        session_adapter_enabled=True,
        input_smoothing_sigma_bins=2.0,
        input_smoothing_kernel_size=100,
        input_smoothing_threshold=0.01,
        white_noise_sd=1.0,
        constant_offset_sd=0.2,
        cache_root=Path(cache_root),
        run_name="stanford_released_baseline_rnn_ckpt9950",
    )


def convert_released_tf_checkpoint_to_pytorch(
    *,
    checkpoint_dir: str | Path,
    output_path: str | Path,
    cache_root: str | Path,
    overwrite: bool = False,
) -> Path:
    """Convert the released TensorFlow GRU checkpoint into an export-compatible `.pt` file."""
    checkpoint_dir = Path(checkpoint_dir)
    output_path = Path(output_path)
    if output_path.exists() and not bool(overwrite):
        return output_path

    args = _load_yaml(checkpoint_dir / "args.yaml")
    released_sessions = tuple(str(session) for session in args["dataset"]["sessions"])
    dataset_to_layer = [int(value) for value in args["dataset"]["datasetToLayerMap"]]
    session_to_layer = dict(zip(released_sessions, dataset_to_layer))

    config = released_rnn_config(cache_root=cache_root)
    problem = build_willett_problem(
        cache_root=Path(config.cache_root),
        dataset=str(config.dataset),
        feature_mode=str(config.feature_mode),
        boundary_key_mode=str(config.boundary_key_mode),
        split_policy=str(config.split_policy),
        cv_num_folds=int(config.cv_num_folds),
        cv_fold_index=int(config.cv_fold_index),
    )
    session_adapter_keys = adapter_keys_from_problem(problem)
    model = build_willett_model(
        config=config,
        input_dim=256,
        vocab_size=int(problem["vocab"]["num_classes"]),
        session_adapter_keys=session_adapter_keys,
    )
    state = model.state_dict()

    reader = _tf_reader(_latest_checkpoint_prefix(checkpoint_dir))
    default_layer = 0
    default_kernel = _read_var(
        reader,
        f"inputLayer_{default_layer}/layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE",
    )
    default_bias = _read_var(
        reader,
        f"inputLayer_{default_layer}/layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE",
    )
    _assign_linear(state, "session_input_adapter.default_layer.linear", default_kernel, default_bias)

    for adapter_key, module_key in model.session_input_adapter._name_map.items():
        session = _session_for_adapter_key(str(adapter_key), released_sessions)
        layer_idx = session_to_layer.get(session, default_layer)
        kernel = _read_var(
            reader,
            f"inputLayer_{layer_idx}/layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE",
        )
        bias = _read_var(
            reader,
            f"inputLayer_{layer_idx}/layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE",
        )
        _assign_linear(state, f"session_input_adapter.layers.{module_key}.linear", kernel, bias)

    state["initial_state"] = _to_tensor(
        _read_var(reader, "net/initStates/.ATTRIBUTES/VARIABLE_VALUE")
    )
    for layer_idx in range(int(config.gru_num_layers)):
        kernel = _reorder_keras_gru_gates(
            _read_var(reader, f"net/rnnLayers/{layer_idx}/cell/kernel/.ATTRIBUTES/VARIABLE_VALUE")
        )
        recurrent_kernel = _reorder_keras_gru_gates(
            _read_var(reader, f"net/rnnLayers/{layer_idx}/cell/recurrent_kernel/.ATTRIBUTES/VARIABLE_VALUE")
        )
        bias = _reorder_keras_gru_gates(
            _read_var(reader, f"net/rnnLayers/{layer_idx}/cell/bias/.ATTRIBUTES/VARIABLE_VALUE")
        )
        state[f"gru.weight_ih_l{layer_idx}"] = _to_tensor(kernel.T)
        state[f"gru.weight_hh_l{layer_idx}"] = _to_tensor(recurrent_kernel.T)
        state[f"gru.bias_ih_l{layer_idx}"] = _to_tensor(bias[0])
        state[f"gru.bias_hh_l{layer_idx}"] = _to_tensor(bias[1])

    classifier_kernel, classifier_bias = _reorder_stanford_logits_to_local_vocab(
        _read_var(reader, "net/dense/kernel/.ATTRIBUTES/VARIABLE_VALUE"),
        _read_var(reader, "net/dense/bias/.ATTRIBUTES/VARIABLE_VALUE"),
    )
    _assign_linear(state, "classifier", classifier_kernel, classifier_bias)
    model.load_state_dict(state, strict=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_step = int(str(_latest_checkpoint_prefix(checkpoint_dir).name).split("-")[-1])
    payload = {
        "model_state": state,
        "config": asdict(config),
        "steps": checkpoint_step,
        "session_adapter_keys": session_adapter_keys,
        "source": {
            "kind": "stanford_speechbci_released_tensorflow_baseline_rnn",
            "checkpoint_dir": str(checkpoint_dir),
            "checkpoint_step": checkpoint_step,
            "feature_mode": "tx_sbp",
            "feature_note": "Released TFRecords use 256 inputFeatures = first 128 TX + first 128 spikePow/SBP.",
            "logit_order_note": "Classifier outputs converted from Stanford [AA..SIL, BLANK] to local [BLANK, AA..SIL].",
            "args": args,
        },
    }
    torch.save(payload, output_path)
    (output_path.with_suffix(".metadata.json")).write_text(json.dumps(payload["source"], indent=2))
    return output_path


__all__ = [
    "RELEASED_SESSIONS",
    "convert_released_tf_checkpoint_to_pytorch",
    "ensure_released_archive_extracted",
    "released_rnn_config",
]

"""Construct and load repository Willett-style decoder checkpoints.

This module centralizes repository checkpoint interpretation for the adapted
Willett-style decoder family. It does not change the uncertain upstream origin
documented in ``PROVENANCE.md`` or identify these checkpoints as official
Stanford artifacts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from .config import WillettReconstructionConfig
from .data import adapter_keys_from_rows
from .model import WillettPhonemeModel


def config_from_checkpoint(
    checkpoint_payload: Mapping[str, Any],
) -> WillettReconstructionConfig:
    """Reconstruct the current config from a repository checkpoint payload."""

    config_payload = dict(checkpoint_payload.get("config") or {})
    if not config_payload:
        raise KeyError("Checkpoint is missing a Willett reconstruction 'config' payload.")
    valid_keys = set(WillettReconstructionConfig.__dataclass_fields__)
    return WillettReconstructionConfig(
        **{key: value for key, value in config_payload.items() if key in valid_keys}
    )


def adapter_keys_from_problem(problem: Mapping[str, Any]) -> tuple[str, ...]:
    """Return the ordered union of training and validation adapter keys."""

    train_adapter_keys = adapter_keys_from_rows(
        problem["train_rows"],
        dataset=str(problem["dataset"]),
        boundary_key_mode=str(problem["boundary_key_mode"]),
    )
    val_adapter_keys = adapter_keys_from_rows(
        problem["val_rows"],
        dataset=str(problem["dataset"]),
        boundary_key_mode=str(problem["boundary_key_mode"]),
    )
    return tuple(dict.fromkeys(train_adapter_keys + val_adapter_keys))


def build_willett_model(
    *,
    config: WillettReconstructionConfig,
    input_dim: int,
    vocab_size: int,
    session_adapter_keys: Sequence[str] = (),
    device: str | torch.device | None = None,
) -> WillettPhonemeModel:
    """Build one decoder using the architecture fields in ``config``."""

    model = WillettPhonemeModel(
        input_dim=int(input_dim),
        vocab_size=int(vocab_size),
        patch_size=int(config.patch_size),
        patch_stride=int(config.patch_stride),
        input_projection_size=int(config.input_projection_size),
        input_projection_dropout=float(config.input_projection_dropout),
        decoder_backbone_type=str(config.decoder_backbone_type),
        gru_hidden_size=int(config.gru_hidden_size),
        gru_num_layers=int(config.gru_num_layers),
        gru_dropout=float(config.gru_dropout),
        s5_hidden_size=int(config.s5_hidden_size),
        s5_state_size=int(config.s5_state_size),
        s5_num_layers=int(config.s5_num_layers),
        s5_dropout=float(config.s5_dropout),
        s5_direction=str(config.s5_direction),
        s5_ffn_multiplier=float(config.s5_ffn_multiplier),
        s4d_hidden_size=int(config.s4d_hidden_size),
        s4d_state_size=int(config.s4d_state_size),
        s4d_num_layers=int(config.s4d_num_layers),
        s4d_dropout=float(config.s4d_dropout),
        s4d_direction=str(config.s4d_direction),
        s4d_ffn_multiplier=float(config.s4d_ffn_multiplier),
        session_adapter_keys=tuple(str(key) for key in session_adapter_keys),
        session_adapter_enabled=bool(config.session_adapter_enabled),
    )
    return model.to(device) if device is not None else model


def load_willett_model_from_checkpoint(
    checkpoint: str | Path | Mapping[str, Any],
    *,
    input_dim: int,
    vocab_size: int,
    config: WillettReconstructionConfig | None = None,
    problem: Mapping[str, Any] | None = None,
    session_adapter_keys: Sequence[str] | None = None,
    device: str | torch.device | None = None,
    strict: bool = True,
) -> tuple[WillettPhonemeModel, WillettReconstructionConfig, dict[str, Any]]:
    """Load a decoder while preserving explicit checkpoint adapter keys first."""

    payload = (
        torch.load(Path(checkpoint), map_location="cpu", weights_only=False)
        if isinstance(checkpoint, (str, Path))
        else dict(checkpoint)
    )
    resolved_config = config or config_from_checkpoint(payload)
    if session_adapter_keys is not None:
        resolved_adapter_keys = tuple(str(key) for key in session_adapter_keys)
    elif payload.get("session_adapter_keys"):
        resolved_adapter_keys = tuple(str(key) for key in payload["session_adapter_keys"])
    elif problem is not None:
        resolved_adapter_keys = adapter_keys_from_problem(problem)
    else:
        resolved_adapter_keys = ()

    model = build_willett_model(
        config=resolved_config,
        input_dim=int(input_dim),
        vocab_size=int(vocab_size),
        session_adapter_keys=resolved_adapter_keys,
        device=device,
    )
    model_state = payload.get("model_state")
    if not isinstance(model_state, Mapping):
        raise KeyError("Checkpoint is missing a valid 'model_state' payload.")
    model.load_state_dict(model_state, strict=bool(strict))
    return model, resolved_config, payload


__all__ = [
    "adapter_keys_from_problem",
    "build_willett_model",
    "config_from_checkpoint",
    "load_willett_model_from_checkpoint",
]

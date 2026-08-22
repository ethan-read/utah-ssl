"""Exact evaluation-time access to intermediate layers of the local GRU.

The decoder is an LLM-assisted Willett-style adaptation with unresolved
upstream provenance; see ``experiments/supervised_baselines/PROVENANCE.md``.
This module adds a repository-specific diagnostic path and is not an upstream
or official Stanford implementation.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from experiments.supervised_baselines.model import WillettPhonemeModel


def clone_gru_as_single_layer_stack(gru: nn.GRU) -> nn.ModuleList:
    """Clone a unidirectional multi-layer GRU into equivalent one-layer GRUs."""

    if bool(gru.bidirectional):
        raise ValueError("Layerwise export currently supports unidirectional GRUs only.")
    if int(gru.proj_size) != 0:
        raise ValueError("Layerwise export does not support projected GRUs.")

    reference = gru.weight_ih_l0
    layers = nn.ModuleList()
    for layer_index in range(int(gru.num_layers)):
        input_size = int(gru.input_size) if layer_index == 0 else int(gru.hidden_size)
        layer = nn.GRU(
            input_size=input_size,
            hidden_size=int(gru.hidden_size),
            num_layers=1,
            bias=bool(gru.bias),
            batch_first=bool(gru.batch_first),
            bidirectional=False,
        ).to(device=reference.device, dtype=reference.dtype)
        with torch.no_grad():
            layer.weight_ih_l0.copy_(getattr(gru, f"weight_ih_l{layer_index}"))
            layer.weight_hh_l0.copy_(getattr(gru, f"weight_hh_l{layer_index}"))
            if bool(gru.bias):
                layer.bias_ih_l0.copy_(getattr(gru, f"bias_ih_l{layer_index}"))
                layer.bias_hh_l0.copy_(getattr(gru, f"bias_hh_l{layer_index}"))
        layer.requires_grad_(False)
        layer.eval()
        layers.append(layer)
    return layers


def forward_gru_layer_stack(
    model: WillettPhonemeModel,
    patched_inputs: torch.Tensor,
    token_lengths: torch.Tensor,
    layers: nn.ModuleList,
) -> tuple[torch.Tensor, ...]:
    """Return one padded time series per recurrent layer in evaluation mode."""

    if str(model.decoder_backbone_type) != "gru":
        raise ValueError("Intermediate GRU states require a GRU decoder checkpoint.")
    if model.training or any(layer.training for layer in layers):
        raise ValueError(
            "Layerwise GRU inference is evaluation-only because training dropout "
            "between recurrent layers is intentionally absent."
        )
    if len(layers) != int(model.gru_num_layers):
        raise ValueError("Layer stack length does not match the checkpoint GRU depth.")
    if patched_inputs.ndim != 3:
        raise ValueError("patched_inputs must have shape [batch, time, features].")

    initial_hidden = model._initial_hidden_state(
        batch_size=int(patched_inputs.shape[0]),
        device=patched_inputs.device,
        dtype=patched_inputs.dtype,
    )
    packed = nn.utils.rnn.pack_padded_sequence(
        patched_inputs,
        token_lengths.cpu(),
        batch_first=True,
        enforce_sorted=False,
    )
    outputs: list[torch.Tensor] = []
    current = packed
    for layer_index, layer in enumerate(layers):
        current, _ = layer(current, initial_hidden[layer_index : layer_index + 1])
        padded, _ = nn.utils.rnn.pad_packed_sequence(
            current,
            batch_first=True,
            total_length=int(patched_inputs.shape[1]),
        )
        outputs.append(padded)
    return tuple(outputs)


def layerwise_equivalence_errors(
    *,
    standard_hidden: torch.Tensor,
    standard_logits: torch.Tensor,
    layer_states: tuple[torch.Tensor, ...],
    classifier: nn.Module,
) -> dict[str, float]:
    """Return maximum absolute top-state and logit reconstruction errors."""

    if not layer_states:
        raise ValueError("At least one layer state is required.")
    reconstructed_hidden = layer_states[-1]
    reconstructed_logits = classifier(reconstructed_hidden)
    return {
        "top_hidden_max_abs_error": float(
            torch.max(torch.abs(reconstructed_hidden - standard_hidden)).item()
        ),
        "logits_max_abs_error": float(
            torch.max(torch.abs(reconstructed_logits - standard_logits)).item()
        ),
    }


__all__ = [
    "clone_gru_as_single_layer_stack",
    "forward_gru_layer_stack",
    "layerwise_equivalence_errors",
]

"""Willett-style GRU phoneme decoder."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import torch
import torch.nn as nn


def patched_length(length: int, *, patch_size: int, patch_stride: int) -> int:
    """Return the number of pre-GRU temporal patches for one example."""
    resolved_length = int(length)
    if resolved_length <= 0:
        return 0
    if resolved_length <= int(patch_size):
        return 1
    return 1 + ((resolved_length - int(patch_size)) // int(patch_stride))


class SessionFeatureAffine(nn.Module):
    """Per-session per-feature affine initialized as identity."""

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.scale = nn.Parameter(torch.ones(self.input_dim))
        self.bias = nn.Parameter(torch.zeros(self.input_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale.view(1, -1) + self.bias.view(1, -1)


class SessionInputAdapterBank(nn.Module):
    """Identity-initialized affine bank keyed by session/day."""

    def __init__(self, session_keys: tuple[str, ...] | list[str], input_dim: int) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        unique_keys = tuple(dict.fromkeys(str(key) for key in session_keys))
        self._name_map = {
            session_key: self._module_key(session_key)
            for session_key in unique_keys
        }
        self.default_layer = SessionFeatureAffine(self.input_dim)
        self.layers = nn.ModuleDict(
            {
                module_key: SessionFeatureAffine(self.input_dim)
                for module_key in self._name_map.values()
            }
        )

    @staticmethod
    def _module_key(session_key: str) -> str:
        digest = hashlib.sha1(str(session_key).encode("utf-8")).hexdigest()
        return f"adapter_{digest}"

    def _layer_for_key(self, session_key: str) -> SessionFeatureAffine:
        module_key = self._name_map.get(str(session_key))
        if module_key is None:
            return self.default_layer
        return self.layers[module_key]

    def forward(
        self,
        x: torch.Tensor,
        session_ids: list[str] | tuple[str, ...] | None,
    ) -> torch.Tensor:
        if session_ids is None:
            raise ValueError("session_ids are required when session adaptation is enabled")
        if len(session_ids) != int(x.shape[0]):
            raise ValueError("session_ids length must match the batch size")
        adapted = [
            self._layer_for_key(str(session_key))(x[row_idx])
            for row_idx, session_key in enumerate(session_ids)
        ]
        return torch.stack(adapted, dim=0)


@dataclass(frozen=True)
class WillettEncoderOutputs:
    adapted_input: torch.Tensor
    patched_inputs: torch.Tensor
    token_lengths: torch.Tensor
    hidden: torch.Tensor
    logits: torch.Tensor


class WillettPhonemeModel(nn.Module):
    """Supervised GRU decoder inspired by the Stanford speech baseline."""

    def __init__(
        self,
        *,
        input_dim: int,
        vocab_size: int,
        patch_size: int = 14,
        patch_stride: int = 4,
        input_projection_size: int = 256,
        input_projection_dropout: float = 0.2,
        gru_hidden_size: int = 768,
        gru_num_layers: int = 5,
        gru_dropout: float = 0.4,
        session_adapter_keys: tuple[str, ...] = (),
        session_adapter_enabled: bool = True,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.vocab_size = int(vocab_size)
        self.patch_size = int(patch_size)
        self.patch_stride = int(patch_stride)
        self.input_projection_size = int(input_projection_size)
        self.gru_hidden_size = int(gru_hidden_size)
        self.gru_num_layers = int(gru_num_layers)
        self.session_adapter_enabled = bool(session_adapter_enabled)
        self.session_input_adapter = SessionInputAdapterBank(
            tuple(session_adapter_keys),
            input_dim=self.input_dim,
        )
        patch_dim = self.input_dim * self.patch_size
        self.input_projection = nn.Sequential(
            nn.Linear(patch_dim, self.input_projection_size),
            nn.Softsign(),
            nn.Dropout(float(input_projection_dropout)),
        )
        effective_gru_dropout = float(gru_dropout) if self.gru_num_layers > 1 else 0.0
        self.gru = nn.GRU(
            input_size=self.input_projection_size,
            hidden_size=self.gru_hidden_size,
            num_layers=self.gru_num_layers,
            dropout=effective_gru_dropout,
            batch_first=True,
            bidirectional=False,
        )
        self.classifier = nn.Linear(self.gru_hidden_size, self.vocab_size)

    def _patch_one(self, sample: torch.Tensor, length: int) -> torch.Tensor:
        valid = sample[:length]
        total_patches = patched_length(
            length,
            patch_size=self.patch_size,
            patch_stride=self.patch_stride,
        )
        if total_patches <= 0:
            return sample.new_zeros((0, self.input_dim * self.patch_size))
        patches: list[torch.Tensor] = []
        for patch_idx in range(total_patches):
            start = patch_idx * self.patch_stride
            patch = valid[start : start + self.patch_size]
            if int(patch.shape[0]) < self.patch_size:
                pad = valid.new_zeros((self.patch_size - int(patch.shape[0]), self.input_dim))
                patch = torch.cat([patch, pad], dim=0)
            patches.append(patch.reshape(-1))
        return torch.stack(patches, dim=0)

    def _patch_batch(
        self,
        x: torch.Tensor,
        input_lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        token_sequences: list[torch.Tensor] = []
        token_lengths: list[int] = []
        for sample, length_tensor in zip(x, input_lengths):
            tokens = self._patch_one(sample, int(length_tensor.item()))
            token_sequences.append(tokens)
            token_lengths.append(int(tokens.shape[0]))
        max_tokens = max(token_lengths, default=0)
        patched = x.new_zeros((int(x.shape[0]), max_tokens, self.input_dim * self.patch_size))
        for batch_idx, tokens in enumerate(token_sequences):
            if int(tokens.shape[0]) > 0:
                patched[batch_idx, : int(tokens.shape[0])] = tokens
        return patched, torch.as_tensor(token_lengths, device=input_lengths.device, dtype=torch.long)

    def forward(
        self,
        x: torch.Tensor,
        input_lengths: torch.Tensor,
        *,
        session_ids: list[str] | tuple[str, ...] | None = None,
    ) -> dict[str, torch.Tensor]:
        adapted_input = (
            self.session_input_adapter(x, session_ids)
            if self.session_adapter_enabled
            else x
        )
        patched_inputs, token_lengths = self._patch_batch(adapted_input, input_lengths)
        projected = self.input_projection(patched_inputs)
        packed = nn.utils.rnn.pack_padded_sequence(
            projected,
            token_lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_hidden, _ = self.gru(packed)
        hidden, _ = nn.utils.rnn.pad_packed_sequence(
            packed_hidden,
            batch_first=True,
            total_length=projected.shape[1],
        )
        logits = self.classifier(hidden)
        return {
            "adapted_input": adapted_input,
            "patched_inputs": patched_inputs,
            "projected_inputs": projected,
            "hidden": hidden,
            "token_lengths": token_lengths,
            "logits": logits,
        }

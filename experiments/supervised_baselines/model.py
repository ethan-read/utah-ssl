"""Willett-style supervised phoneme decoders."""

from __future__ import annotations

import hashlib

import torch
import torch.nn as nn

from utah_ssl.models.s4d import BidirectionalS4DSequenceBackbone, S4DSequenceBackbone
from utah_ssl.models.s5 import BidirectionalS5SequenceBackbone, S5SequenceBackbone
from utah_ssl.patching import patch_batch as _core_patch_batch
from utah_ssl.patching import patched_length as _core_patched_length


def patched_length(length: int, *, patch_size: int, patch_stride: int) -> int:
    """Return the number of pre-GRU temporal patches for one example."""
    return _core_patched_length(
        int(length),
        patch_size=int(patch_size),
        patch_stride=int(patch_stride),
        policy="floor",
    )


class SessionInputNetwork(nn.Module):
    """Per-session framewise input network matching the Stanford recipe."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.linear = nn.Linear(self.input_dim, self.output_dim)
        self.activation = nn.Softsign()
        self.dropout = nn.Dropout(float(dropout))
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        if self.input_dim == self.output_dim:
            with torch.no_grad():
                self.linear.weight.copy_(torch.eye(self.input_dim))
                self.linear.bias.zero_()
            return
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.activation(self.linear(x)))


class SessionInputAdapterBank(nn.Module):
    """Session/day-specific framewise input networks."""

    def __init__(
        self,
        session_keys: tuple[str, ...] | list[str],
        *,
        input_dim: int,
        output_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.dropout = float(dropout)
        unique_keys = tuple(dict.fromkeys(str(key) for key in session_keys))
        self._name_map = {
            session_key: self._module_key(session_key)
            for session_key in unique_keys
        }
        self.default_layer = SessionInputNetwork(
            self.input_dim,
            self.output_dim,
            self.dropout,
        )
        self.layers = nn.ModuleDict(
            {
                module_key: SessionInputNetwork(
                    self.input_dim,
                    self.output_dim,
                    self.dropout,
                )
                for module_key in self._name_map.values()
            }
        )

    @staticmethod
    def _module_key(session_key: str) -> str:
        digest = hashlib.sha1(str(session_key).encode("utf-8")).hexdigest()
        return f"adapter_{digest}"

    def _layer_for_key(self, session_key: str) -> SessionInputNetwork:
        module_key = self._name_map.get(str(session_key))
        if module_key is None:
            return self.default_layer
        return self.layers[module_key]

    def forward(
        self,
        x: torch.Tensor,
        session_ids: list[str] | tuple[str, ...] | None,
        *,
        session_adapter_enabled: bool,
    ) -> torch.Tensor:
        if not bool(session_adapter_enabled):
            return self.default_layer(x)
        if session_ids is None:
            raise ValueError("session_ids are required when session adaptation is enabled")
        if len(session_ids) != int(x.shape[0]):
            raise ValueError("session_ids length must match the batch size")
        adapted = [
            self._layer_for_key(str(session_key))(x[row_idx])
            for row_idx, session_key in enumerate(session_ids)
        ]
        return torch.stack(adapted, dim=0)


class WillettPhonemeModel(nn.Module):
    """Supervised decoder inspired by the Stanford speech baseline."""

    def __init__(
        self,
        *,
        input_dim: int,
        vocab_size: int,
        patch_size: int = 14,
        patch_stride: int = 4,
        input_projection_size: int = 256,
        input_projection_dropout: float = 0.2,
        decoder_backbone_type: str = "gru",
        gru_hidden_size: int = 512,
        gru_num_layers: int = 5,
        gru_dropout: float = 0.4,
        s5_hidden_size: int = 512,
        s5_state_size: int = 128,
        s5_num_layers: int = 5,
        s5_dropout: float = 0.2,
        s5_direction: str = "causal",
        s5_ffn_multiplier: float = 2.0,
        s4d_hidden_size: int = 512,
        s4d_state_size: int = 128,
        s4d_num_layers: int = 5,
        s4d_dropout: float = 0.2,
        s4d_direction: str = "causal",
        s4d_ffn_multiplier: float = 2.0,
        session_adapter_keys: tuple[str, ...] = (),
        session_adapter_enabled: bool = True,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.vocab_size = int(vocab_size)
        self.patch_size = int(patch_size)
        self.patch_stride = int(patch_stride)
        self.input_projection_size = int(input_projection_size)
        self.decoder_backbone_type = str(decoder_backbone_type)
        self.gru_hidden_size = int(gru_hidden_size)
        self.gru_num_layers = int(gru_num_layers)
        self.s5_hidden_size = int(s5_hidden_size)
        self.s5_state_size = int(s5_state_size)
        self.s5_num_layers = int(s5_num_layers)
        self.s5_direction = str(s5_direction)
        self.s4d_hidden_size = int(s4d_hidden_size)
        self.s4d_state_size = int(s4d_state_size)
        self.s4d_num_layers = int(s4d_num_layers)
        self.s4d_direction = str(s4d_direction)
        self.session_adapter_enabled = bool(session_adapter_enabled)
        if self.decoder_backbone_type not in {"gru", "s5", "s4d"}:
            raise ValueError("decoder_backbone_type must be one of {'gru', 's5', 's4d'}")
        if self.s5_direction not in {"causal", "bidirectional"}:
            raise ValueError("s5_direction must be one of {'causal', 'bidirectional'}")
        if self.s4d_direction not in {"causal", "bidirectional"}:
            raise ValueError("s4d_direction must be one of {'causal', 'bidirectional'}")
        self.session_input_adapter = SessionInputAdapterBank(
            tuple(session_adapter_keys),
            input_dim=self.input_dim,
            output_dim=self.input_projection_size,
            dropout=float(input_projection_dropout),
        )
        self.adapter_output_dim = self.input_projection_size
        patch_dim = self.adapter_output_dim * self.patch_size
        if self.decoder_backbone_type == "gru":
            effective_gru_dropout = float(gru_dropout) if self.gru_num_layers > 1 else 0.0
            self.gru = nn.GRU(
                input_size=patch_dim,
                hidden_size=self.gru_hidden_size,
                num_layers=self.gru_num_layers,
                dropout=effective_gru_dropout,
                batch_first=True,
                bidirectional=False,
            )
            self.initial_state = nn.Parameter(torch.empty((1, self.gru_hidden_size)))
            nn.init.xavier_uniform_(self.initial_state)
            self.decoder_output_size = self.gru_hidden_size
        elif self.decoder_backbone_type == "s5":
            self.s5_input_norm = nn.LayerNorm(patch_dim)
            self.s5_input_projection = nn.Linear(patch_dim, self.s5_hidden_size)
            self.s5_hidden_norm = nn.LayerNorm(self.s5_hidden_size)
            s5_backbone_cls = (
                S5SequenceBackbone
                if self.s5_direction == "causal"
                else BidirectionalS5SequenceBackbone
            )
            self.s5 = s5_backbone_cls(
                d_model=self.s5_hidden_size,
                d_state=self.s5_state_size,
                num_layers=self.s5_num_layers,
                dropout=float(s5_dropout),
                ffn_multiplier=float(s5_ffn_multiplier),
            )
            self.decoder_output_size = self.s5_hidden_size
        else:
            self.s4d_input_norm = nn.LayerNorm(patch_dim)
            self.s4d_input_projection = nn.Linear(patch_dim, self.s4d_hidden_size)
            self.s4d_hidden_norm = nn.LayerNorm(self.s4d_hidden_size)
            s4d_backbone_cls = (
                S4DSequenceBackbone
                if self.s4d_direction == "causal"
                else BidirectionalS4DSequenceBackbone
            )
            self.s4d = s4d_backbone_cls(
                d_model=self.s4d_hidden_size,
                d_state=self.s4d_state_size,
                num_layers=self.s4d_num_layers,
                dropout=float(s4d_dropout),
                ffn_multiplier=float(s4d_ffn_multiplier),
            )
            self.decoder_output_size = self.s4d_hidden_size
        self.classifier = nn.Linear(self.decoder_output_size, self.vocab_size)

    def _initial_hidden_state(
        self,
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        # Stanford learns the first-layer recurrent state and leaves deeper
        # layers unset; mirror that by tiling one learned state and keeping the
        # remaining layers zero-initialized.
        hidden = torch.zeros(
            (self.gru_num_layers, int(batch_size), self.gru_hidden_size),
            device=device,
            dtype=dtype,
        )
        hidden[0] = self.initial_state.to(device=device, dtype=dtype).expand(int(batch_size), -1)
        return hidden

    def _patch_batch(
        self,
        x: torch.Tensor,
        input_lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return _core_patch_batch(
            x,
            input_lengths,
            patch_size=self.patch_size,
            patch_stride=self.patch_stride,
            policy="floor",
        )

    def forward(
        self,
        x: torch.Tensor,
        input_lengths: torch.Tensor,
        *,
        session_ids: list[str] | tuple[str, ...] | None = None,
    ) -> dict[str, torch.Tensor]:
        adapted_input = self.session_input_adapter(
            x,
            session_ids,
            session_adapter_enabled=self.session_adapter_enabled,
        )
        patched_inputs, token_lengths = self._patch_batch(adapted_input, input_lengths)
        if self.decoder_backbone_type == "gru":
            packed = nn.utils.rnn.pack_padded_sequence(
                patched_inputs,
                token_lengths.cpu(),
                batch_first=True,
                enforce_sorted=False,
            )
            initial_hidden = self._initial_hidden_state(
                batch_size=int(x.shape[0]),
                device=patched_inputs.device,
                dtype=patched_inputs.dtype,
            )
            packed_hidden, _ = self.gru(packed, initial_hidden)
            hidden, _ = nn.utils.rnn.pad_packed_sequence(
                packed_hidden,
                batch_first=True,
                total_length=patched_inputs.shape[1],
            )
            projected_inputs = patched_inputs
        elif self.decoder_backbone_type == "s5":
            projected_inputs = self.s5_hidden_norm(
                self.s5_input_projection(self.s5_input_norm(patched_inputs))
            )
            hidden = self.s5(projected_inputs, token_lengths)
        else:
            projected_inputs = self.s4d_hidden_norm(
                self.s4d_input_projection(self.s4d_input_norm(patched_inputs))
            )
            hidden = self.s4d(projected_inputs, token_lengths)
        logits = self.classifier(hidden)
        return {
            "adapted_input": adapted_input,
            "patched_inputs": patched_inputs,
            "projected_inputs": projected_inputs,
            "hidden": hidden,
            "decoder_hidden": hidden,
            "token_lengths": token_lengths,
            "logits": logits,
        }

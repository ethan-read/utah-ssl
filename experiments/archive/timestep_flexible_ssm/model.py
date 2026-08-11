"""Timestep-flexible supervised S5 decoder."""

from __future__ import annotations

import hashlib
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from utah_ssl.patching import patch_starts
except ModuleNotFoundError:  # pragma: no cover - repo-root unittest fallback
    from utah_ssl.patching import patch_starts

from .data import resolve_patch_bins


class SessionInputNetwork(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, dropout: float) -> None:
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
        self.default_layer = SessionInputNetwork(self.input_dim, self.output_dim, self.dropout)
        self.layers = nn.ModuleDict(
            {
                module_key: SessionInputNetwork(self.input_dim, self.output_dim, self.dropout)
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


def _sequence_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
    positions = torch.arange(max_len, device=lengths.device).unsqueeze(0)
    return positions < lengths.unsqueeze(1)


def _apply_sequence_mask(x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    mask = _sequence_mask(lengths, int(x.shape[1])).unsqueeze(-1)
    return x * mask.to(x.dtype)


def reverse_padded_sequence(x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    positions = torch.arange(x.shape[1], device=lengths.device).unsqueeze(0)
    valid_mask = positions < lengths.unsqueeze(1)
    reversed_positions = (lengths.unsqueeze(1) - 1 - positions).clamp_min(0)
    gather_positions = torch.where(valid_mask, reversed_positions, positions)
    view_shape = (*gather_positions.shape, *([1] * (x.ndim - 2)))
    gather_index = gather_positions.view(view_shape).expand_as(x)
    return x.gather(dim=1, index=gather_index)


class DiagonalS5SSM(nn.Module):
    def __init__(self, d_model: int, d_state: int):
        super().__init__()
        self.d_model = int(d_model)
        self.d_state = int(d_state)

        real_init = torch.linspace(0.5, 1.5, self.d_state, dtype=torch.float32)
        imag_init = torch.linspace(0.0, math.pi, self.d_state, dtype=torch.float32)
        self.lambda_real_log = nn.Parameter(torch.log(real_init))
        self.lambda_imag = nn.Parameter(imag_init)
        self.log_dt = nn.Parameter(torch.tensor(-2.0, dtype=torch.float32))

        scale_b = 1.0 / math.sqrt(self.d_model)
        scale_c = 1.0 / math.sqrt(self.d_state)
        self.B_re = nn.Parameter(torch.randn(self.d_state, self.d_model) * scale_b)
        self.B_im = nn.Parameter(torch.randn(self.d_state, self.d_model) * scale_b)
        self.C_re = nn.Parameter(torch.randn(self.d_model, self.d_state) * scale_c)
        self.C_im = nn.Parameter(torch.randn(self.d_model, self.d_state) * scale_c)
        self.D = nn.Linear(self.d_model, self.d_model, bias=False)
        with torch.no_grad():
            self.D.weight.zero_()
            self.D.weight += torch.eye(self.d_model)

    def _discretized_params(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dt = F.softplus(self.log_dt) + 1e-4
        lam = torch.complex(-torch.exp(self.lambda_real_log), self.lambda_imag)
        abar = torch.exp(dt * lam)
        b = torch.complex(self.B_re, self.B_im)
        c = torch.complex(self.C_re, self.C_im)
        bbar = ((abar - 1.0) / lam).unsqueeze(-1) * b
        return abar, bbar, c

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        _, seq_len, _ = x.shape
        if seq_len <= 0:
            return x.new_zeros((*x.shape[:-1], self.d_model))
        abar, bbar, c = self._discretized_params()
        input_terms = x.to(torch.complex64) @ bbar.transpose(0, 1)
        input_terms = _apply_sequence_mask(input_terms, lengths)

        powers = torch.arange(seq_len, device=x.device, dtype=torch.float32)
        kernel = abar.unsqueeze(0).pow(powers.unsqueeze(1))
        fft_len = 1 << (2 * seq_len - 1).bit_length()
        input_fft = torch.fft.fft(input_terms, n=fft_len, dim=1)
        kernel_fft = torch.fft.fft(kernel, n=fft_len, dim=0).unsqueeze(0)
        states = torch.fft.ifft(input_fft * kernel_fft, n=fft_len, dim=1)[:, :seq_len, :]
        response = states @ c.transpose(0, 1)
        y = response.real + self.D(x)
        return _apply_sequence_mask(y, lengths)


class S5Block(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_state: int,
        *,
        dropout: float = 0.0,
        ffn_multiplier: float = 2.0,
    ) -> None:
        super().__init__()
        d_ff = max(d_model, int(float(ffn_multiplier) * d_model))
        self.norm1 = nn.LayerNorm(d_model)
        self.ssm = DiagonalS5SSM(d_model=d_model, d_state=d_state)
        self.dropout1 = nn.Dropout(float(dropout))
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(d_ff, d_model),
            nn.Dropout(float(dropout)),
        )

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        x = x + self.dropout1(self.ssm(self.norm1(x), lengths))
        x = _apply_sequence_mask(x, lengths)
        x = x + self.ffn(self.norm2(x))
        return _apply_sequence_mask(x, lengths)


class S5SequenceBackbone(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_state: int,
        num_layers: int,
        *,
        dropout: float = 0.0,
        ffn_multiplier: float = 2.0,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                S5Block(
                    d_model=int(d_model),
                    d_state=int(d_state),
                    dropout=float(dropout),
                    ffn_multiplier=float(ffn_multiplier),
                )
                for _ in range(int(num_layers))
            ]
        )

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x, lengths)
        return _apply_sequence_mask(x, lengths)


class BidirectionalS5SequenceBackbone(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_state: int,
        num_layers: int,
        *,
        dropout: float = 0.0,
        ffn_multiplier: float = 2.0,
    ) -> None:
        super().__init__()
        self.forward_backbone = S5SequenceBackbone(
            d_model=int(d_model),
            d_state=int(d_state),
            num_layers=int(num_layers),
            dropout=float(dropout),
            ffn_multiplier=float(ffn_multiplier),
        )
        self.backward_backbone = S5SequenceBackbone(
            d_model=int(d_model),
            d_state=int(d_state),
            num_layers=int(num_layers),
            dropout=float(dropout),
            ffn_multiplier=float(ffn_multiplier),
        )
        self.fusion = nn.Linear(2 * int(d_model), int(d_model))

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        forward_hidden = self.forward_backbone(x, lengths)
        reversed_x = reverse_padded_sequence(x, lengths)
        backward_hidden_reversed = self.backward_backbone(reversed_x, lengths)
        backward_hidden = reverse_padded_sequence(backward_hidden_reversed, lengths)
        fused = self.fusion(torch.cat([forward_hidden, backward_hidden], dim=-1))
        return _apply_sequence_mask(fused, lengths)


def _resample_patch(patch: torch.Tensor, *, reference_bins: int) -> torch.Tensor:
    if int(patch.shape[0]) == int(reference_bins):
        return patch
    interpolated = F.interpolate(
        patch.transpose(0, 1).unsqueeze(0),
        size=int(reference_bins),
        mode="linear",
        align_corners=False,
    )
    return interpolated.squeeze(0).transpose(0, 1)


def patch_resample_batch(
    x: torch.Tensor,
    lengths: torch.Tensor,
    *,
    active_patch_size_bins: int,
    active_patch_stride_bins: int,
    reference_patch_size_bins: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    token_sequences: list[torch.Tensor] = []
    token_lengths: list[int] = []
    feature_dim = int(x.shape[-1])
    patch_dim = feature_dim * int(reference_patch_size_bins)
    for sample, length_tensor in zip(x, lengths):
        length = max(0, int(length_tensor.item()))
        starts = patch_starts(
            length,
            patch_size=int(active_patch_size_bins),
            patch_stride=int(active_patch_stride_bins),
            policy="floor",
        )
        if not starts:
            tokens = sample.new_zeros((0, patch_dim))
        else:
            valid = sample[:length]
            patches: list[torch.Tensor] = []
            for start in starts:
                patch = valid[start : start + int(active_patch_size_bins)]
                if int(patch.shape[0]) < int(active_patch_size_bins):
                    pad = valid.new_zeros((int(active_patch_size_bins) - int(patch.shape[0]), feature_dim))
                    patch = torch.cat([patch, pad], dim=0)
                patch = _resample_patch(patch, reference_bins=int(reference_patch_size_bins))
                patches.append(patch.reshape(-1))
            tokens = torch.stack(patches, dim=0)
        token_sequences.append(tokens)
        token_lengths.append(int(tokens.shape[0]))
    max_tokens = max(token_lengths, default=0)
    patched = x.new_zeros((int(x.shape[0]), max_tokens, patch_dim))
    for batch_idx, tokens in enumerate(token_sequences):
        if int(tokens.shape[0]) > 0:
            patched[batch_idx, : int(tokens.shape[0])] = tokens
    return patched, torch.as_tensor(token_lengths, device=lengths.device, dtype=torch.long)


class TimestepFlexibleS5Model(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        vocab_size: int,
        train_bin_size_ms: int = 20,
        patch_size_ms: int = 280,
        patch_stride_ms: int = 80,
        input_projection_size: int = 256,
        input_projection_dropout: float = 0.2,
        s5_hidden_size: int = 512,
        s5_state_size: int = 128,
        s5_num_layers: int = 5,
        s5_dropout: float = 0.2,
        s5_direction: str = "causal",
        s5_ffn_multiplier: float = 2.0,
        session_adapter_keys: tuple[str, ...] = (),
        session_adapter_enabled: bool = True,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.vocab_size = int(vocab_size)
        self.train_bin_size_ms = int(train_bin_size_ms)
        self.patch_size_ms = int(patch_size_ms)
        self.patch_stride_ms = int(patch_stride_ms)
        self.input_projection_size = int(input_projection_size)
        self.s5_hidden_size = int(s5_hidden_size)
        self.s5_state_size = int(s5_state_size)
        self.s5_num_layers = int(s5_num_layers)
        self.s5_direction = str(s5_direction)
        self.session_adapter_enabled = bool(session_adapter_enabled)
        if self.s5_direction not in {"causal", "bidirectional"}:
            raise ValueError("s5_direction must be one of {'causal', 'bidirectional'}")
        self.reference_patch_size_bins = resolve_patch_bins(
            int(self.patch_size_ms),
            bin_size_ms=int(self.train_bin_size_ms),
            field_name="patch_size_ms",
        )
        self.reference_patch_stride_bins = resolve_patch_bins(
            int(self.patch_stride_ms),
            bin_size_ms=int(self.train_bin_size_ms),
            field_name="patch_stride_ms",
        )
        self.session_input_adapter = SessionInputAdapterBank(
            tuple(session_adapter_keys),
            input_dim=self.input_dim,
            output_dim=self.input_projection_size,
            dropout=float(input_projection_dropout),
        )
        patch_dim = self.input_projection_size * self.reference_patch_size_bins
        self.s5_input_norm = nn.LayerNorm(patch_dim)
        self.s5_input_projection = nn.Linear(patch_dim, self.s5_hidden_size)
        self.s5_hidden_norm = nn.LayerNorm(self.s5_hidden_size)
        s5_backbone_cls = S5SequenceBackbone if self.s5_direction == "causal" else BidirectionalS5SequenceBackbone
        self.s5 = s5_backbone_cls(
            d_model=self.s5_hidden_size,
            d_state=self.s5_state_size,
            num_layers=self.s5_num_layers,
            dropout=float(s5_dropout),
            ffn_multiplier=float(s5_ffn_multiplier),
        )
        self.classifier = nn.Linear(self.s5_hidden_size, self.vocab_size)

    def forward(
        self,
        x: torch.Tensor,
        input_lengths: torch.Tensor,
        *,
        active_bin_size_ms: int,
        session_ids: list[str] | tuple[str, ...] | None = None,
    ) -> dict[str, torch.Tensor | float | int]:
        adapted_input = self.session_input_adapter(
            x,
            session_ids,
            session_adapter_enabled=self.session_adapter_enabled,
        )
        active_patch_size_bins = resolve_patch_bins(
            int(self.patch_size_ms),
            bin_size_ms=int(active_bin_size_ms),
            field_name="patch_size_ms",
        )
        active_patch_stride_bins = resolve_patch_bins(
            int(self.patch_stride_ms),
            bin_size_ms=int(active_bin_size_ms),
            field_name="patch_stride_ms",
        )
        patched_inputs, token_lengths = patch_resample_batch(
            adapted_input,
            input_lengths,
            active_patch_size_bins=int(active_patch_size_bins),
            active_patch_stride_bins=int(active_patch_stride_bins),
            reference_patch_size_bins=int(self.reference_patch_size_bins),
        )
        projected_inputs = self.s5_hidden_norm(
            self.s5_input_projection(self.s5_input_norm(patched_inputs))
        )
        # The backbone runs on patched tokens, and patch_stride_ms is held fixed
        # across train/eval views, so the recurrent token timebase stays matched.
        dt_scale = 1.0
        hidden = self.s5(projected_inputs, token_lengths)
        logits = self.classifier(hidden)
        return {
            "adapted_input": adapted_input,
            "patched_inputs": patched_inputs,
            "projected_inputs": projected_inputs,
            "hidden": hidden,
            "decoder_hidden": hidden,
            "token_lengths": token_lengths,
            "logits": logits,
            "dt_scale": float(dt_scale),
            "active_patch_size_bins": int(active_patch_size_bins),
            "active_patch_stride_bins": int(active_patch_stride_bins),
        }


__all__ = [
    "DiagonalS5SSM",
    "TimestepFlexibleS5Model",
    "patch_resample_batch",
]

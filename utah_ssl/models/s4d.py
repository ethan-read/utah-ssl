"""Pure-PyTorch S4D reference blocks for sequence experiments."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from utah_ssl.models.s5 import _apply_sequence_mask, reverse_padded_sequence


class DiagonalS4DSSM(nn.Module):
    """Diagonal S4D layer implemented as a causal FFT convolution."""

    def __init__(self, d_model: int, d_state: int):
        super().__init__()
        self.d_model = int(d_model)
        self.d_state = int(d_state)

        real_init = torch.full((self.d_state,), 0.5, dtype=torch.float32)
        imag_init = math.pi * torch.arange(self.d_state, dtype=torch.float32)
        self.lambda_real_log = nn.Parameter(torch.log(real_init))
        self.lambda_imag = nn.Parameter(imag_init)
        self.log_dt = nn.Parameter(torch.full((self.d_model,), -2.0, dtype=torch.float32))

        scale = 1.0 / math.sqrt(self.d_state)
        self.B_re = nn.Parameter(torch.randn(self.d_model, self.d_state) * scale)
        self.B_im = nn.Parameter(torch.randn(self.d_model, self.d_state) * scale)
        self.C_re = nn.Parameter(torch.randn(self.d_model, self.d_state) * scale)
        self.C_im = nn.Parameter(torch.randn(self.d_model, self.d_state) * scale)
        self.D = nn.Parameter(torch.ones(self.d_model))

    def _discretized_params(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dt = F.softplus(self.log_dt) + 1e-4
        lam = torch.complex(-torch.exp(self.lambda_real_log), self.lambda_imag)
        abar = torch.exp(dt.unsqueeze(-1) * lam.unsqueeze(0))

        b = torch.complex(self.B_re, self.B_im)
        c = torch.complex(self.C_re, self.C_im)
        bbar = ((abar - 1.0) / lam.unsqueeze(0)) * b
        return abar, bbar, c

    def _kernel(self, seq_len: int, *, device: torch.device) -> torch.Tensor:
        abar, bbar, c = self._discretized_params()
        powers = torch.arange(seq_len, device=device, dtype=torch.float32)
        vandermonde = abar.unsqueeze(-1).pow(powers.view(1, 1, seq_len))
        kernel = ((bbar * c).unsqueeze(-1) * vandermonde).sum(dim=1)
        return kernel.real

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        _, seq_len, _ = x.shape
        if seq_len <= 0:
            return x.new_zeros((*x.shape[:-1], self.d_model))

        x = _apply_sequence_mask(x, lengths)
        kernel = self._kernel(seq_len, device=x.device).to(dtype=x.dtype)
        fft_len = 1 << (2 * seq_len - 1).bit_length()

        x_fft = torch.fft.rfft(x.transpose(1, 2), n=fft_len, dim=-1)
        kernel_fft = torch.fft.rfft(kernel, n=fft_len, dim=-1).unsqueeze(0)
        y = torch.fft.irfft(x_fft * kernel_fft, n=fft_len, dim=-1)[..., :seq_len]
        y = y.transpose(1, 2) + x * self.D.to(device=x.device, dtype=x.dtype)
        return _apply_sequence_mask(y, lengths)


class S4DBlock(nn.Module):
    """Pre-norm residual S4D block with pointwise channel mixing."""

    def __init__(
        self,
        d_model: int,
        d_state: int,
        *,
        dropout: float = 0.0,
        ffn_multiplier: float = 2.0,
    ):
        super().__init__()
        d_ff = max(int(d_model), int(ffn_multiplier * int(d_model)))
        self.norm1 = nn.LayerNorm(d_model)
        self.ssm = DiagonalS4DSSM(d_model=d_model, d_state=d_state)
        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        x = x + self.dropout1(self.ssm(self.norm1(x), lengths))
        x = _apply_sequence_mask(x, lengths)
        x = x + self.ffn(self.norm2(x))
        return _apply_sequence_mask(x, lengths)


class S4DSequenceBackbone(nn.Module):
    """Stacked causal S4D residual blocks."""

    def __init__(
        self,
        d_model: int,
        d_state: int,
        num_layers: int,
        *,
        dropout: float = 0.0,
        ffn_multiplier: float = 2.0,
    ):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                S4DBlock(
                    d_model=d_model,
                    d_state=d_state,
                    dropout=dropout,
                    ffn_multiplier=ffn_multiplier,
                )
                for _ in range(int(num_layers))
            ]
        )

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x, lengths)
        return _apply_sequence_mask(x, lengths)


class BidirectionalS4DSequenceBackbone(nn.Module):
    """Bidirectional S4D wrapper using forward/backward towers and learned fusion."""

    def __init__(
        self,
        d_model: int,
        d_state: int,
        num_layers: int,
        *,
        dropout: float = 0.0,
        ffn_multiplier: float = 2.0,
    ):
        super().__init__()
        self.forward_backbone = S4DSequenceBackbone(
            d_model=d_model,
            d_state=d_state,
            num_layers=num_layers,
            dropout=dropout,
            ffn_multiplier=ffn_multiplier,
        )
        self.backward_backbone = S4DSequenceBackbone(
            d_model=d_model,
            d_state=d_state,
            num_layers=num_layers,
            dropout=dropout,
            ffn_multiplier=ffn_multiplier,
        )
        self.fusion = nn.Linear(2 * int(d_model), int(d_model))

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        forward_hidden = self.forward_backbone(x, lengths)
        reversed_x = reverse_padded_sequence(x, lengths)
        backward_hidden_reversed = self.backward_backbone(reversed_x, lengths)
        backward_hidden = reverse_padded_sequence(backward_hidden_reversed, lengths)
        fused = self.fusion(torch.cat([forward_hidden, backward_hidden], dim=-1))
        return _apply_sequence_mask(fused, lengths)

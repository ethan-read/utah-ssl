"""Pure-PyTorch S5 reference blocks for the SSL transfer benchmark."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _sequence_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
    positions = torch.arange(max_len, device=lengths.device).unsqueeze(0)
    return positions < lengths.unsqueeze(1)


def _apply_sequence_mask(x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    mask = _sequence_mask(lengths, x.shape[1]).unsqueeze(-1)
    return x * mask.to(x.dtype)


class DiagonalS5SSM(nn.Module):
    """Minimal diagonalized MIMO S5 layer with shared complex state."""

    def __init__(self, d_model: int, d_state: int):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state

        real_init = torch.linspace(0.5, 1.5, d_state, dtype=torch.float32)
        imag_init = torch.linspace(0.0, math.pi, d_state, dtype=torch.float32)

        self.lambda_real_log = nn.Parameter(torch.log(real_init))
        self.lambda_imag = nn.Parameter(imag_init)
        self.log_dt = nn.Parameter(torch.tensor(-2.0, dtype=torch.float32))

        scale_b = 1.0 / math.sqrt(d_model)
        scale_c = 1.0 / math.sqrt(d_state)
        self.B_re = nn.Parameter(torch.randn(d_state, d_model) * scale_b)
        self.B_im = nn.Parameter(torch.randn(d_state, d_model) * scale_b)
        self.C_re = nn.Parameter(torch.randn(d_model, d_state) * scale_c)
        self.C_im = nn.Parameter(torch.randn(d_model, d_state) * scale_c)

        self.D = nn.Linear(d_model, d_model, bias=False)
        with torch.no_grad():
            self.D.weight.zero_()
            self.D.weight += torch.eye(d_model)

    def _discretized_params(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dt = F.softplus(self.log_dt) + 1e-4
        lam = torch.complex(-torch.exp(self.lambda_real_log), self.lambda_imag)
        abar = torch.exp(dt * lam)

        b = torch.complex(self.B_re, self.B_im)
        c = torch.complex(self.C_re, self.C_im)
        bbar = ((abar - 1.0) / lam).unsqueeze(-1) * b
        return abar, bbar, c

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        abar, bbar, c = self._discretized_params()

        state = torch.zeros(
            batch_size,
            self.d_state,
            dtype=torch.complex64,
            device=x.device,
        )
        outputs: list[torch.Tensor] = []
        for step in range(seq_len):
            u_t = x[:, step, :]
            input_term = u_t.to(torch.complex64) @ bbar.transpose(0, 1)
            proposed_state = state * abar.unsqueeze(0) + input_term

            valid = (step < lengths).unsqueeze(-1)
            state = torch.where(valid, proposed_state, state)

            response = state @ c.transpose(0, 1)
            y_t = response.real + self.D(u_t)
            outputs.append(y_t)

        y = torch.stack(outputs, dim=1)
        return _apply_sequence_mask(y, lengths)


class S5Block(nn.Module):
    """Canonical pre-norm residual S5 block with a small FFN."""

    def __init__(
        self,
        d_model: int,
        d_state: int,
        *,
        dropout: float = 0.0,
        ffn_multiplier: float = 2.0,
    ):
        super().__init__()
        d_ff = max(d_model, int(ffn_multiplier * d_model))
        self.norm1 = nn.LayerNorm(d_model)
        self.ssm = DiagonalS5SSM(d_model=d_model, d_state=d_state)
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


class S5SequenceBackbone(nn.Module):
    """Stacked canonical S5 residual blocks."""

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
                S5Block(
                    d_model=d_model,
                    d_state=d_state,
                    dropout=dropout,
                    ffn_multiplier=ffn_multiplier,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x, lengths)
        return _apply_sequence_mask(x, lengths)

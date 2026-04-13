"""Causal S5 masked-reconstruction model definitions for Colab experiments."""

from __future__ import annotations

import torch
import torch.nn as nn

from s5 import S5SequenceBackbone


def sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return (x / rms) * self.scale


def _sequence_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
    positions = torch.arange(max_len, device=lengths.device).unsqueeze(0)
    return positions < lengths.unsqueeze(1)


def masked_mean_pool(hidden: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    mask = _sequence_mask(lengths, hidden.shape[1]).unsqueeze(-1).to(hidden.dtype)
    denom = mask.sum(dim=1).clamp_min(1.0)
    return (hidden * mask).sum(dim=1) / denom


class ReconstructionHead(nn.Module):
    def __init__(self, hidden_size: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, output_dim),
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.net(hidden)


class S5MaskedEncoder(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        hidden_size: int,
        s5_state_size: int,
        num_layers: int,
        dropout: float,
        patch_size: int,
        patch_stride: int,
        post_proj_norm: str,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_size = int(hidden_size)
        self.patch_size = int(patch_size)
        self.patch_stride = int(patch_stride)
        self.token_dim = self.input_dim * self.patch_size
        self.proj = nn.Linear(self.token_dim, self.hidden_size)
        self.post_proj_norm = RMSNorm(self.hidden_size) if post_proj_norm == "rms" else nn.Identity()
        self.backbone = S5SequenceBackbone(
            d_model=self.hidden_size,
            d_state=int(s5_state_size),
            num_layers=int(num_layers),
            dropout=float(dropout),
            ffn_multiplier=2.0,
        )
        self.raw_mask_token = nn.Parameter(torch.zeros(self.token_dim))
        self.hidden_mask_token = nn.Parameter(torch.zeros(self.hidden_size))

    def _patch_starts(self, length: int) -> list[int]:
        if length <= 0:
            return [0]
        if self.patch_size == 1:
            return list(range(length))
        max_start = max(length - self.patch_size, 0)
        starts = list(range(0, max(length - self.patch_size + 1, 1), self.patch_stride))
        if not starts:
            starts = [0]
        if starts[-1] != max_start:
            starts.append(max_start)
        return starts

    def _patch_one(self, sample: torch.Tensor, length: int) -> torch.Tensor:
        valid = sample[:length]
        if self.patch_size == 1:
            return valid

        patches: list[torch.Tensor] = []
        for start in self._patch_starts(length):
            patch = valid[start : start + self.patch_size]
            if patch.shape[0] < self.patch_size:
                pad = valid.new_zeros((self.patch_size - patch.shape[0], valid.shape[1]))
                patch = torch.cat([patch, pad], dim=0)
            patches.append(patch.reshape(-1))
        return torch.stack(patches, dim=0)

    def patch_batch(self, x: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        token_sequences: list[torch.Tensor] = []
        token_lengths: list[int] = []
        for sample, length_tensor in zip(x, lengths):
            length = int(length_tensor.item())
            tokens = self._patch_one(sample, length)
            token_sequences.append(tokens)
            token_lengths.append(int(tokens.shape[0]))

        max_tokens = max(token_lengths)
        token_dim = int(token_sequences[0].shape[1])
        tokens = x.new_zeros((len(token_sequences), max_tokens, token_dim))
        for idx, token_sequence in enumerate(token_sequences):
            tokens[idx, : token_sequence.shape[0]] = token_sequence
        return tokens, torch.tensor(token_lengths, device=lengths.device, dtype=torch.long)

    def encode_patched(
        self,
        tokens: torch.Tensor,
        token_lengths: torch.Tensor,
        *,
        token_mask: torch.Tensor | None = None,
        mask_token_placement: str = "before_projection",
    ) -> dict[str, torch.Tensor]:
        token_mask_bool = (
            token_mask.to(device=tokens.device, dtype=torch.bool)
            if token_mask is not None
            else torch.zeros(tokens.shape[:2], device=tokens.device, dtype=torch.bool)
        )
        if mask_token_placement not in {"before_projection", "after_projection"}:
            raise ValueError(
                "mask_token_placement must be one of {'before_projection', 'after_projection'}"
            )

        if mask_token_placement == "before_projection":
            mask_token = self.raw_mask_token.to(device=tokens.device, dtype=tokens.dtype).view(1, 1, -1)
            masked_tokens = torch.where(token_mask_bool.unsqueeze(-1), mask_token, tokens)
            projected = self.proj(masked_tokens)
            hidden_input = self.post_proj_norm(projected)
        else:
            projected = self.proj(tokens)
            hidden_input = self.post_proj_norm(projected)
            mask_token = self.hidden_mask_token.to(
                device=hidden_input.device,
                dtype=hidden_input.dtype,
            ).view(1, 1, -1)
            hidden_input = torch.where(token_mask_bool.unsqueeze(-1), mask_token, hidden_input)
            masked_tokens = tokens

        hidden = self.backbone(hidden_input, token_lengths)
        return {
            "hidden": hidden,
            "token_lengths": token_lengths,
            "tokens": tokens,
            "masked_tokens": masked_tokens,
            "token_mask": token_mask_bool,
        }

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> dict[str, torch.Tensor]:
        tokens, token_lengths = self.patch_batch(x, lengths)
        return self.encode_patched(tokens, token_lengths, token_mask=None)


class MaskedSSLModel(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        hidden_size: int,
        s5_state_size: int,
        num_layers: int,
        dropout: float,
        patch_size: int,
        patch_stride: int,
        post_proj_norm: str,
    ):
        super().__init__()
        self.encoder = S5MaskedEncoder(
            input_dim=input_dim,
            hidden_size=hidden_size,
            s5_state_size=s5_state_size,
            num_layers=num_layers,
            dropout=dropout,
            patch_size=patch_size,
            patch_stride=patch_stride,
            post_proj_norm=post_proj_norm,
        )
        self.reconstruction_head = ReconstructionHead(
            hidden_size=int(hidden_size),
            output_dim=int(self.encoder.token_dim),
        )

    def encode_sequence(self, x: torch.Tensor, lengths: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.encoder(x, lengths)

    def reconstruct_from_patched_tokens(
        self,
        tokens: torch.Tensor,
        token_lengths: torch.Tensor,
        *,
        token_mask: torch.Tensor | None = None,
        mask_token_placement: str = "before_projection",
    ) -> dict[str, torch.Tensor]:
        outputs = self.encoder.encode_patched(
            tokens,
            token_lengths,
            token_mask=token_mask,
            mask_token_placement=mask_token_placement,
        )
        reconstruction = self.reconstruction_head(outputs["hidden"])
        return {**outputs, "reconstruction": reconstruction}


# Compatibility aliases so the downstream probe helpers can stay nearly unchanged.
S5ContrastiveEncoder = S5MaskedEncoder
ContrastiveSSLModel = MaskedSSLModel


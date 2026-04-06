"""Contrastive S5 model definitions used by the Colab experiments."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

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


class ProjectionHead(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class S5ContrastiveEncoder(nn.Module):
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

    def _patch_one(self, sample: torch.Tensor, length: int) -> torch.Tensor:
        valid = sample[:length]
        if self.patch_size == 1:
            return valid

        starts = list(range(0, max(length - self.patch_size + 1, 1), self.patch_stride))
        patches: list[torch.Tensor] = []
        for start in starts:
            patch = valid[start : start + self.patch_size]
            if patch.shape[0] < self.patch_size:
                pad = valid.new_zeros((self.patch_size - patch.shape[0], valid.shape[1]))
                patch = torch.cat([patch, pad], dim=0)
            patches.append(patch.reshape(-1))
        return torch.stack(patches, dim=0)

    def _patch_batch(self, x: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        token_sequences: list[torch.Tensor] = []
        token_lengths: list[int] = []
        for sample, length_tensor in zip(x, lengths):
            length = int(length_tensor.item())
            tokens = self._patch_one(sample, length)
            token_sequences.append(tokens)
            token_lengths.append(int(tokens.shape[0]))

        max_tokens = max(token_lengths)
        tokens = x.new_zeros((len(token_sequences), max_tokens, self.token_dim))
        for idx, token_sequence in enumerate(token_sequences):
            tokens[idx, : token_sequence.shape[0]] = token_sequence
        return tokens, torch.tensor(token_lengths, device=lengths.device, dtype=torch.long)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> dict[str, torch.Tensor]:
        tokens, token_lengths = self._patch_batch(x, lengths)
        hidden = self.post_proj_norm(self.proj(tokens))
        hidden = self.backbone(hidden, token_lengths)
        return {
            "hidden": hidden,
            "token_lengths": token_lengths,
            "tokens": tokens,
        }


class ContrastiveSSLModel(nn.Module):
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
        self.encoder = S5ContrastiveEncoder(
            input_dim=input_dim,
            hidden_size=hidden_size,
            s5_state_size=s5_state_size,
            num_layers=num_layers,
            dropout=dropout,
            patch_size=patch_size,
            patch_stride=patch_stride,
            post_proj_norm=post_proj_norm,
        )
        self.anchor_head = ProjectionHead(hidden_size)
        self.future_head = ProjectionHead(hidden_size)
        self.segment_head = ProjectionHead(hidden_size)

    def encode_sequence(self, x: torch.Tensor, lengths: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.encoder(x, lengths)

    def encode_pooled(self, x: torch.Tensor, lengths: torch.Tensor) -> dict[str, torch.Tensor]:
        outputs = self.encoder(x, lengths)
        pooled = masked_mean_pool(outputs["hidden"], outputs["token_lengths"])
        z = F.normalize(self.segment_head(pooled), dim=-1)
        return {**outputs, "pooled": pooled, "z": z}

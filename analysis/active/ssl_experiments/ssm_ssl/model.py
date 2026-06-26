"""Generic S5/Mamba encoder models for SSL and CTC experiments."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import torch
import torch.nn as nn

try:
    from ssl_core.imports import ensure_s5_import_path
    from ssl_core.patching import PatchPolicy, causal_conv_lengths, patch_batch
except ModuleNotFoundError:  # pragma: no cover - repo-root unittest fallback
    from analysis.active.ssl_experiments.ssl_core.imports import ensure_s5_import_path
    from analysis.active.ssl_experiments.ssl_core.patching import (
        PatchPolicy,
        causal_conv_lengths,
        patch_batch,
    )

ensure_s5_import_path()
from s5 import BidirectionalS5SequenceBackbone, S5SequenceBackbone


def _sequence_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
    positions = torch.arange(max_len, device=lengths.device).unsqueeze(0)
    return positions < lengths.unsqueeze(1)


def _zero_invalid(x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    if int(x.shape[1]) == 0:
        return x
    mask = _sequence_mask(lengths.to(x.device), int(x.shape[1])).unsqueeze(-1).to(x.dtype)
    return x * mask


class CausalConvStem(nn.Module):
    def __init__(self, *, input_dim: int, hidden_size: int, kernel_size: int, stride: int) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_size = int(hidden_size)
        self.kernel_size = int(kernel_size)
        self.stride = int(stride)
        self.conv = nn.Conv1d(
            self.input_dim,
            self.hidden_size,
            kernel_size=self.kernel_size,
            stride=self.stride,
        )
        self.activation = nn.GELU()
        self.norm = nn.LayerNorm(self.hidden_size)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        conv_input = torch.nn.functional.pad(x.transpose(1, 2), (self.kernel_size - 1, 0))
        tokens = self.conv(conv_input).transpose(1, 2)
        token_lengths = causal_conv_lengths(lengths.to(x.device), stride=self.stride)
        return _zero_invalid(self.norm(self.activation(tokens)), token_lengths), token_lengths


class MambaSequenceBackbone(nn.Module):
    def __init__(self, *, hidden_size: int, state_size: int, num_layers: int, dropout: float) -> None:
        super().__init__()
        try:
            from transformers import MambaConfig, MambaModel
        except ImportError as exc:  # pragma: no cover - optional Colab dependency
            raise ImportError(
                "Mamba backbone requires transformers with MambaModel available. "
                "Install/upgrade transformers in the runtime before using backbone_type='mamba'."
            ) from exc

        config = MambaConfig(
            # This encoder always feeds `inputs_embeds`, so the default 50k token
            # embedding table is dead weight unless we shrink the vocabulary.
            vocab_size=1,
            hidden_size=int(hidden_size),
            state_size=int(state_size),
            num_hidden_layers=int(num_layers),
            intermediate_size=max(int(hidden_size) * 2, 1),
            hidden_dropout_prob=float(dropout),
        )
        self.model = MambaModel(config)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        outputs = self.model(inputs_embeds=x)
        hidden = getattr(outputs, "last_hidden_state", outputs[0])
        return _zero_invalid(hidden, lengths)


def build_sequence_backbone(
    *,
    backbone_type: str,
    hidden_size: int,
    state_size: int,
    num_layers: int,
    dropout: float,
    direction: str,
    ffn_multiplier: float,
) -> nn.Module:
    if backbone_type == "s5":
        backbone_cls = S5SequenceBackbone if direction == "causal" else BidirectionalS5SequenceBackbone
        return backbone_cls(
            d_model=int(hidden_size),
            d_state=int(state_size),
            num_layers=int(num_layers),
            dropout=float(dropout),
            ffn_multiplier=float(ffn_multiplier),
        )
    if backbone_type == "mamba":
        if direction != "causal":
            raise ValueError("Mamba backbone currently supports only direction='causal'")
        return MambaSequenceBackbone(
            hidden_size=int(hidden_size),
            state_size=int(state_size),
            num_layers=int(num_layers),
            dropout=float(dropout),
        )
    raise ValueError("backbone_type must be one of {'s5', 'mamba'}")


class GenericSSMEncoder(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        hidden_size: int,
        backbone_type: str = "s5",
        state_size: int = 64,
        num_layers: int = 4,
        dropout: float = 0.1,
        direction: str = "causal",
        ffn_multiplier: float = 2.0,
        input_mode: str = "temporal_patch",
        patch_size: int = 14,
        patch_stride: int = 4,
        patch_policy: PatchPolicy = "floor",
        conv_kernel_size: int = 14,
        conv_stride: int = 4,
    ) -> None:
        super().__init__()
        if input_mode not in {"raw_bin", "temporal_patch", "causal_conv_stem"}:
            raise ValueError("input_mode must be one of {'raw_bin', 'temporal_patch', 'causal_conv_stem'}")
        self.input_dim = int(input_dim)
        self.hidden_size = int(hidden_size)
        self.backbone_type = str(backbone_type)
        self.state_size = int(state_size)
        self.num_layers = int(num_layers)
        self.input_mode = str(input_mode)
        self.patch_size = int(patch_size)
        self.patch_stride = int(patch_stride)
        self.patch_policy = patch_policy
        self.conv_kernel_size = int(conv_kernel_size)
        self.conv_stride = int(conv_stride)

        if self.input_mode == "temporal_patch":
            self.token_dim = self.input_dim * self.patch_size
            self.stem: CausalConvStem | None = None
        elif self.input_mode == "causal_conv_stem":
            self.token_dim = self.hidden_size
            self.stem = CausalConvStem(
                input_dim=self.input_dim,
                hidden_size=self.hidden_size,
                kernel_size=self.conv_kernel_size,
                stride=self.conv_stride,
            )
        else:
            self.token_dim = self.input_dim
            self.stem = None

        self.input_projection = (
            nn.Identity()
            if self.input_mode == "causal_conv_stem"
            else nn.Sequential(
                nn.LayerNorm(self.token_dim),
                nn.Linear(self.token_dim, self.hidden_size),
                nn.LayerNorm(self.hidden_size),
            )
        )
        self.backbone = build_sequence_backbone(
            backbone_type=str(backbone_type),
            hidden_size=self.hidden_size,
            state_size=self.state_size,
            num_layers=int(num_layers),
            dropout=float(dropout),
            direction=str(direction),
            ffn_multiplier=float(ffn_multiplier),
        )

    def tokenize(self, x: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 3:
            raise ValueError(f"Expected x to have shape [B, T, D], got {tuple(x.shape)}")
        if int(x.shape[-1]) != self.input_dim:
            raise ValueError(f"Expected input_dim={self.input_dim}, got {int(x.shape[-1])}")
        if self.input_mode == "raw_bin":
            return _zero_invalid(x, lengths), lengths.to(device=x.device, dtype=torch.long)
        if self.input_mode == "temporal_patch":
            return patch_batch(
                x,
                lengths.to(x.device),
                patch_size=self.patch_size,
                patch_stride=self.patch_stride,
                policy=self.patch_policy,
            )
        if self.stem is None:
            raise RuntimeError("causal conv stem is not initialized")
        return self.stem(x, lengths.to(x.device))

    def encode_tokens(self, tokens: torch.Tensor, token_lengths: torch.Tensor) -> torch.Tensor:
        hidden_input = self.input_projection(tokens)
        hidden = self.backbone(hidden_input, token_lengths)
        return _zero_invalid(hidden, token_lengths)

    def encode(self, x: torch.Tensor, lengths: torch.Tensor) -> SimpleNamespace:
        tokens, token_lengths = self.tokenize(x, lengths)
        hidden = self.encode_tokens(tokens, token_lengths)
        return SimpleNamespace(tokens=tokens, token_lengths=token_lengths, hidden=hidden)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        return self.encode(x, lengths).hidden


class GenericMaskedSSMModel(nn.Module):
    def __init__(self, encoder: GenericSSMEncoder) -> None:
        super().__init__()
        self.encoder = encoder
        self.raw_mask_token = nn.Parameter(torch.zeros(int(encoder.token_dim)))
        self.reconstruction_head = nn.Linear(int(encoder.hidden_size), int(encoder.token_dim))

    def forward_tokens(
        self,
        tokens: torch.Tensor,
        token_lengths: torch.Tensor,
        *,
        corruption_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if corruption_mask is None:
            corruption_mask = torch.zeros_like(tokens, dtype=torch.bool)
        mask_token = self.raw_mask_token.to(device=tokens.device, dtype=tokens.dtype).view(1, 1, -1)
        corrupted = torch.where(corruption_mask.to(torch.bool), mask_token, tokens)
        hidden = self.encoder.encode_tokens(corrupted, token_lengths)
        reconstruction = self.reconstruction_head(hidden)
        return {
            "tokens": tokens,
            "corrupted_tokens": corrupted,
            "hidden": hidden,
            "reconstruction": reconstruction,
            "token_lengths": token_lengths,
        }

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor,
        *,
        corruption_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        tokens, token_lengths = self.encoder.tokenize(x, lengths)
        return self.forward_tokens(tokens, token_lengths, corruption_mask=corruption_mask)


class GenericSSMCTCModel(nn.Module):
    def __init__(self, *, encoder: GenericSSMEncoder, vocab_size: int) -> None:
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Linear(int(encoder.hidden_size), int(vocab_size))

    def forward(
        self,
        x: torch.Tensor,
        input_lengths: torch.Tensor,
        *,
        session_ids: list[str] | tuple[str, ...] | None = None,
    ) -> dict[str, torch.Tensor]:
        del session_ids
        outputs = self.encoder.encode(x, input_lengths)
        logits = self.classifier(outputs.hidden)
        return {
            "tokens": outputs.tokens,
            "hidden": outputs.hidden,
            "logits": logits,
            "token_lengths": outputs.token_lengths,
        }


def make_encoder_from_config(config: Any, *, input_dim: int | None = None) -> GenericSSMEncoder:
    return GenericSSMEncoder(
        input_dim=int(input_dim if input_dim is not None else config.input_dim),
        hidden_size=int(config.hidden_size),
        backbone_type=str(config.backbone_type),
        state_size=int(config.state_size),
        num_layers=int(config.num_layers),
        dropout=float(config.dropout),
        direction=str(config.direction),
        ffn_multiplier=float(config.ffn_multiplier),
        input_mode=str(config.input_mode),
        patch_size=int(config.patch_size),
        patch_stride=int(config.patch_stride),
        patch_policy=str(config.patch_policy),
        conv_kernel_size=int(config.conv_kernel_size),
        conv_stride=int(config.conv_stride),
    )


__all__ = [
    "CausalConvStem",
    "GenericMaskedSSMModel",
    "GenericSSMCTCModel",
    "GenericSSMEncoder",
    "MambaSequenceBackbone",
    "build_sequence_backbone",
    "make_encoder_from_config",
]

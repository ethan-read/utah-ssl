"""POSSM-style reconstruction and phoneme decoding models."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class EncoderOutputs:
    hidden: torch.Tensor
    token_lengths: torch.Tensor
    tokens: torch.Tensor
    latent_tokens: torch.Tensor


class ValueEncoder(nn.Module):
    def __init__(
        self,
        *,
        output_dim: int,
        encoder_type: str = "linear",
        hidden_dim: int | None = None,
    ) -> None:
        super().__init__()
        if encoder_type not in {"linear", "mlp"}:
            raise ValueError("encoder_type must be one of {'linear', 'mlp'}")
        self.encoder_type = str(encoder_type)
        if self.encoder_type == "linear":
            self.net = nn.Linear(1, int(output_dim))
        else:
            inner_dim = int(hidden_dim) if hidden_dim is not None else int(output_dim)
            self.net = nn.Sequential(
                nn.Linear(1, inner_dim),
                nn.GELU(),
                nn.Linear(inner_dim, int(output_dim)),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResidualFeedForwardBlock(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(int(dim))
        self.net = nn.Sequential(
            nn.Linear(int(dim), int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), int(dim)),
        )
        self.dropout = nn.Dropout(float(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.dropout(self.net(self.norm(x)))


class ResidualCrossAttentionBlock(nn.Module):
    def __init__(self, dim: int, *, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(int(dim))
        self.context_norm = nn.LayerNorm(int(dim))
        self.attn = nn.MultiheadAttention(
            embed_dim=int(dim),
            num_heads=int(num_heads),
            dropout=float(dropout),
            batch_first=True,
        )
        self.dropout = nn.Dropout(float(dropout))

    def forward(self, query: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(
            self.query_norm(query),
            self.context_norm(context),
            self.context_norm(context),
            need_weights=False,
        )
        return query + self.dropout(attn_out)


class ResidualSelfAttentionBlock(nn.Module):
    def __init__(self, dim: int, *, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(int(dim))
        self.attn = nn.MultiheadAttention(
            embed_dim=int(dim),
            num_heads=int(num_heads),
            dropout=float(dropout),
            batch_first=True,
        )
        self.dropout = nn.Dropout(float(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normalized = self.norm(x)
        attn_out, _ = self.attn(normalized, normalized, normalized, need_weights=False)
        return x + self.dropout(attn_out)


class POSSMEncoder(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        model_dim: int = 64,
        latent_count: int = 4,
        value_encoder_type: str = "linear",
        value_mlp_hidden_size: int | None = None,
        ffn_hidden_size: int = 256,
        dropout: float = 0.1,
        feature_mode: str = "tx_sbp",
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.model_dim = int(model_dim)
        self.latent_count = int(latent_count)
        self.hidden_size = int(self.model_dim * self.latent_count)
        self.token_dim = int(self.model_dim)
        self.feature_mode = str(feature_mode)
        self.source_session_keys: tuple[str, ...] = ()

        self.unit_embedding = nn.Embedding(self.input_dim, self.model_dim)
        self.value_encoder = ValueEncoder(
            output_dim=self.model_dim,
            encoder_type=str(value_encoder_type),
            hidden_dim=value_mlp_hidden_size,
        )
        self.token_norm = nn.LayerNorm(self.model_dim)
        self.latents = nn.Parameter(torch.randn(self.latent_count, self.model_dim) * 0.02)

        self.cross_attention = ResidualCrossAttentionBlock(
            self.model_dim,
            num_heads=1,
            dropout=float(dropout),
        )
        self.cross_ffn = ResidualFeedForwardBlock(
            self.model_dim,
            hidden_dim=int(ffn_hidden_size),
            dropout=float(dropout),
        )
        self.self_attention = ResidualSelfAttentionBlock(
            self.model_dim,
            num_heads=2,
            dropout=float(dropout),
        )
        self.self_ffn = ResidualFeedForwardBlock(
            self.model_dim,
            hidden_dim=int(ffn_hidden_size),
            dropout=float(dropout),
        )

    def tokenize(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected x to have shape [B, T, D], got {tuple(x.shape)}")
        if int(x.shape[-1]) != self.input_dim:
            raise ValueError(
                f"Expected last input dimension {self.input_dim}, got {int(x.shape[-1])}"
            )
        value_tokens = self.value_encoder(x.unsqueeze(-1))
        unit_tokens = self.unit_embedding.weight.view(1, 1, self.input_dim, self.model_dim)
        return self.token_norm(unit_tokens + value_tokens)

    def _encode_tokens(
        self,
        tokens: torch.Tensor,
        input_lengths: torch.Tensor,
    ) -> EncoderOutputs:
        batch_size, time_bins, _, _ = tokens.shape
        flat_tokens = tokens.reshape(batch_size * time_bins, self.input_dim, self.model_dim)
        latents = self.latents.unsqueeze(0).expand(batch_size * time_bins, -1, -1)
        latents = self.cross_attention(latents, flat_tokens)
        latents = self.cross_ffn(latents)
        latents = self.self_attention(latents)
        latents = self.self_ffn(latents)
        latent_tokens = latents.reshape(batch_size, time_bins, self.latent_count, self.model_dim)
        hidden = latent_tokens.reshape(batch_size, time_bins, self.hidden_size)

        valid_bins = (
            torch.arange(time_bins, device=input_lengths.device).unsqueeze(0)
            < input_lengths.unsqueeze(1)
        )
        hidden = hidden * valid_bins.unsqueeze(-1).to(hidden.dtype)
        latent_tokens = latent_tokens * valid_bins.unsqueeze(-1).unsqueeze(-1).to(latent_tokens.dtype)

        return EncoderOutputs(
            hidden=hidden,
            token_lengths=input_lengths.clone(),
            tokens=tokens,
            latent_tokens=latent_tokens,
        )

    def encode(
        self,
        x: torch.Tensor,
        input_lengths: torch.Tensor,
        session_ids: list[str] | tuple[str, ...] | None = None,
        *,
        use_source_affines: bool = True,
        target_affines: Any = None,
    ) -> EncoderOutputs:
        del session_ids, use_source_affines, target_affines
        tokens = self.tokenize(x)
        return self._encode_tokens(tokens, input_lengths)

    def forward(
        self,
        x: torch.Tensor,
        input_lengths: torch.Tensor,
        *,
        session_ids: list[str] | tuple[str, ...] | None = None,
        use_source_affines: bool = True,
        target_affines: Any = None,
    ) -> EncoderOutputs:
        return self.encode(
            x,
            input_lengths,
            session_ids,
            use_source_affines=use_source_affines,
            target_affines=target_affines,
        )


class POSSMReconstructionHead(nn.Module):
    def __init__(
        self,
        *,
        input_size: int,
        output_size: int,
        head_type: str = "linear",
        hidden_size: int | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if head_type not in {"linear", "mlp"}:
            raise ValueError("head_type must be one of {'linear', 'mlp'}")
        if head_type == "linear":
            self.net = nn.Linear(int(input_size), int(output_size))
        else:
            hidden = int(hidden_size) if hidden_size is not None else int(input_size)
            self.net = nn.Sequential(
                nn.LayerNorm(int(input_size)),
                nn.Linear(int(input_size), hidden),
                nn.GELU(),
                nn.Dropout(float(dropout)),
                nn.Linear(hidden, int(output_size)),
            )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.net(hidden)


class IdentityTemporalBackbone(nn.Module):
    def __init__(self, *, input_size: int) -> None:
        super().__init__()
        self.input_size = int(input_size)
        self.output_size = int(input_size)

    def forward(self, hidden: torch.Tensor, input_lengths: torch.Tensor) -> torch.Tensor:
        del input_lengths
        return hidden


class GRUTemporalBackbone(nn.Module):
    def __init__(
        self,
        *,
        input_size: int,
        hidden_size: int | None = None,
        num_layers: int = 1,
        dropout: float = 0.0,
        bidirectional: bool = False,
    ) -> None:
        super().__init__()
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size) if hidden_size is not None else int(input_size)
        self.num_layers = int(num_layers)
        self.bidirectional = bool(bidirectional)
        if self.num_layers <= 0:
            raise ValueError("GRUTemporalBackbone.num_layers must be positive")
        if not (0.0 <= float(dropout) < 1.0):
            raise ValueError("GRUTemporalBackbone.dropout must be in [0, 1)")
        effective_dropout = float(dropout) if self.num_layers > 1 else 0.0
        self.gru = nn.GRU(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=effective_dropout,
            batch_first=True,
            bidirectional=self.bidirectional,
        )
        self.output_size = int(self.hidden_size * (2 if self.bidirectional else 1))

    def forward(self, hidden: torch.Tensor, input_lengths: torch.Tensor) -> torch.Tensor:
        packed = nn.utils.rnn.pack_padded_sequence(
            hidden,
            input_lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_hidden, _ = self.gru(packed)
        unpacked_hidden, _ = nn.utils.rnn.pad_packed_sequence(
            packed_hidden,
            batch_first=True,
            total_length=hidden.shape[1],
        )
        return unpacked_hidden


_TEMPORAL_BACKBONE_REGISTRY: dict[str, type[nn.Module]] = {
    "identity": IdentityTemporalBackbone,
    "gru": GRUTemporalBackbone,
}


def register_temporal_backbone(name: str, backbone_cls: type[nn.Module]) -> None:
    resolved_name = str(name)
    if not resolved_name:
        raise ValueError("Temporal backbone registry key must be non-empty.")
    _TEMPORAL_BACKBONE_REGISTRY[resolved_name] = backbone_cls


def list_registered_temporal_backbones() -> tuple[str, ...]:
    return tuple(sorted(_TEMPORAL_BACKBONE_REGISTRY))


def build_temporal_backbone(
    *,
    backbone_type: str,
    input_size: int,
    gru_hidden_size: int | None = None,
    gru_num_layers: int = 1,
    gru_dropout: float = 0.0,
    gru_bidirectional: bool = False,
    backbone_kwargs: dict[str, Any] | None = None,
) -> nn.Module:
    resolved_type = str(backbone_type)
    if resolved_type not in _TEMPORAL_BACKBONE_REGISTRY:
        raise ValueError(
            f"Unknown temporal backbone type {resolved_type!r}. "
            f"Available: {sorted(_TEMPORAL_BACKBONE_REGISTRY)}"
        )
    resolved_kwargs = dict(backbone_kwargs or {})
    backbone_cls = _TEMPORAL_BACKBONE_REGISTRY[resolved_type]
    if backbone_cls is IdentityTemporalBackbone:
        if resolved_kwargs:
            raise ValueError("Identity temporal backbone does not accept custom backbone_kwargs.")
        return backbone_cls(input_size=int(input_size))
    if backbone_cls is GRUTemporalBackbone:
        if resolved_kwargs:
            raise ValueError("GRU temporal backbone uses dedicated GRU config fields, not backbone_kwargs.")
        return backbone_cls(
            input_size=int(input_size),
            hidden_size=gru_hidden_size,
            num_layers=int(gru_num_layers),
            dropout=float(gru_dropout),
            bidirectional=bool(gru_bidirectional),
        )
    backbone = backbone_cls(input_size=int(input_size), **resolved_kwargs)
    if not hasattr(backbone, "output_size"):
        raise ValueError(
            f"Temporal backbone {resolved_type!r} must expose output_size for decoder construction."
        )
    return backbone


class POSSMReconstructionModel(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        model_dim: int = 64,
        latent_count: int = 4,
        value_encoder_type: str = "linear",
        value_mlp_hidden_size: int | None = None,
        ffn_hidden_size: int = 256,
        dropout: float = 0.1,
        temporal_backbone_type: str = "gru",
        temporal_gru_hidden_size: int | None = None,
        temporal_gru_num_layers: int = 1,
        temporal_gru_dropout: float = 0.0,
        temporal_gru_bidirectional: bool = False,
        temporal_backbone_kwargs: dict[str, Any] | None = None,
        reconstruction_head_type: str = "linear",
        reconstruction_mlp_hidden_size: int | None = None,
        feature_mode: str = "tx_sbp",
    ) -> None:
        super().__init__()
        self.feature_mode = str(feature_mode)
        self.input_dim = int(input_dim)
        self.source_session_keys: tuple[str, ...] = ()
        self.encoder = POSSMEncoder(
            input_dim=int(input_dim),
            model_dim=int(model_dim),
            latent_count=int(latent_count),
            value_encoder_type=str(value_encoder_type),
            value_mlp_hidden_size=value_mlp_hidden_size,
            ffn_hidden_size=int(ffn_hidden_size),
            dropout=float(dropout),
            feature_mode=str(feature_mode),
        )
        self.temporal_backbone_type = str(temporal_backbone_type)
        self.temporal_backbone = build_temporal_backbone(
            backbone_type=str(temporal_backbone_type),
            input_size=int(self.encoder.hidden_size),
            gru_hidden_size=temporal_gru_hidden_size,
            gru_num_layers=int(temporal_gru_num_layers),
            gru_dropout=float(temporal_gru_dropout),
            gru_bidirectional=bool(temporal_gru_bidirectional),
            backbone_kwargs=temporal_backbone_kwargs,
        )
        if not hasattr(self.temporal_backbone, "output_size"):
            raise ValueError("temporal backbone must expose output_size")
        self.reconstruction_head_type = str(reconstruction_head_type)
        self.reconstruction_head = POSSMReconstructionHead(
            input_size=int(getattr(self.temporal_backbone, "output_size")),
            output_size=int(input_dim),
            head_type=str(reconstruction_head_type),
            hidden_size=reconstruction_mlp_hidden_size,
            dropout=float(dropout),
        )

    def encode_sequence(
        self,
        x: torch.Tensor,
        input_lengths: torch.Tensor,
        *,
        session_ids: list[str] | tuple[str, ...] | None = None,
        use_source_affines: bool = True,
        target_affines: Any = None,
    ) -> EncoderOutputs:
        return self.encoder.encode(
            x,
            input_lengths,
            session_ids,
            use_source_affines=use_source_affines,
            target_affines=target_affines,
        )

    def forward(
        self,
        x: torch.Tensor,
        input_lengths: torch.Tensor,
        *,
        session_ids: list[str] | tuple[str, ...] | None = None,
    ) -> dict[str, torch.Tensor]:
        outputs = self.encode_sequence(x, input_lengths, session_ids=session_ids)
        temporal_hidden = self.temporal_backbone(outputs.hidden, outputs.token_lengths)
        reconstruction = self.reconstruction_head(temporal_hidden)
        valid_bins = (
            torch.arange(reconstruction.shape[1], device=input_lengths.device).unsqueeze(0)
            < input_lengths.unsqueeze(1)
        )
        reconstruction = reconstruction * valid_bins.unsqueeze(-1).to(reconstruction.dtype)
        return {
            "encoder_hidden": outputs.hidden,
            "hidden": temporal_hidden,
            "token_lengths": outputs.token_lengths,
            "tokens": outputs.tokens,
            "latent_tokens": outputs.latent_tokens,
            "reconstruction": reconstruction,
        }


def causal_conv_output_lengths(lengths: torch.Tensor, stride: int) -> torch.Tensor:
    stride = int(stride)
    if stride <= 0:
        raise ValueError("stride must be positive")
    lengths = lengths.to(dtype=torch.long)
    positive = lengths > 0
    safe = torch.clamp(lengths - 1, min=0)
    output = torch.div(safe, stride, rounding_mode="floor") + 1
    return torch.where(positive, output, torch.zeros_like(output))


class POSSMPhonemeModel(nn.Module):
    def __init__(
        self,
        *,
        base_encoder: POSSMEncoder,
        pre_decoder_backbone: nn.Module | None = None,
        vocab_size: int,
        gru_hidden_size: int = 768,
        gru_num_layers: int = 5,
        gru_dropout: float = 0.4,
        conv_hidden_size: int | None = None,
        conv_kernel_size: int = 14,
        conv_stride: int = 4,
        conv_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.base_encoder = base_encoder
        self.pre_decoder_backbone = pre_decoder_backbone
        self.vocab_size = int(vocab_size)
        self.gru_hidden_size = int(gru_hidden_size)
        self.gru_num_layers = int(gru_num_layers)
        self.conv_hidden_size = (
            int(conv_hidden_size) if conv_hidden_size is not None else int(gru_hidden_size)
        )
        self.conv_kernel_size = int(conv_kernel_size)
        self.conv_stride = int(conv_stride)
        self.conv_dropout_rate = float(conv_dropout)

        encoder_output_size = (
            int(getattr(self.pre_decoder_backbone, "output_size"))
            if self.pre_decoder_backbone is not None
            else int(base_encoder.hidden_size)
        )
        effective_gru_dropout = float(gru_dropout) if int(gru_num_layers) > 1 else 0.0
        self.gru = nn.GRU(
            input_size=encoder_output_size,
            hidden_size=self.gru_hidden_size,
            num_layers=self.gru_num_layers,
            dropout=effective_gru_dropout,
            batch_first=True,
            bidirectional=False,
        )
        self.conv = nn.Conv1d(
            in_channels=self.gru_hidden_size,
            out_channels=self.conv_hidden_size,
            kernel_size=self.conv_kernel_size,
            stride=self.conv_stride,
        )
        self.conv_activation = nn.GELU()
        self.conv_dropout = nn.Dropout(self.conv_dropout_rate)
        self.classifier = nn.Linear(self.conv_hidden_size, self.vocab_size)

    @property
    def input_dim(self) -> int:
        return int(self.base_encoder.input_dim)

    @property
    def feature_mode(self) -> str:
        return str(self.base_encoder.feature_mode)

    def forward(
        self,
        x: torch.Tensor,
        input_lengths: torch.Tensor,
        *,
        session_ids: list[str] | tuple[str, ...] | None = None,
    ) -> dict[str, torch.Tensor]:
        encoder_outputs = self.base_encoder.encode(x, input_lengths, session_ids)
        encoder_hidden = encoder_outputs.hidden
        if self.pre_decoder_backbone is not None:
            encoder_hidden = self.pre_decoder_backbone(
                encoder_hidden,
                encoder_outputs.token_lengths,
            )
        packed = nn.utils.rnn.pack_padded_sequence(
            encoder_hidden,
            encoder_outputs.token_lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_hidden, _ = self.gru(packed)
        gru_hidden, _ = nn.utils.rnn.pad_packed_sequence(
            packed_hidden,
            batch_first=True,
            total_length=encoder_hidden.shape[1],
        )

        conv_input = gru_hidden.transpose(1, 2)
        conv_input = F.pad(conv_input, (self.conv_kernel_size - 1, 0))
        conv_hidden = self.conv(conv_input)
        conv_hidden = self.conv_activation(conv_hidden)
        conv_hidden = self.conv_dropout(conv_hidden)
        conv_hidden = conv_hidden.transpose(1, 2)
        output_lengths = causal_conv_output_lengths(encoder_outputs.token_lengths, self.conv_stride)
        logits = self.classifier(conv_hidden)
        return {
            "encoder_hidden": encoder_outputs.hidden,
            "sequence_hidden": encoder_hidden,
            "encoder_lengths": encoder_outputs.token_lengths,
            "gru_hidden": gru_hidden,
            "conv_hidden": conv_hidden,
            "logits": logits,
            "token_lengths": output_lengths,
        }

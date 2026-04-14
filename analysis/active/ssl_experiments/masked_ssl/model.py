"""Causal S5 masked-reconstruction model definitions for Colab experiments."""

from __future__ import annotations

import torch
import torch.nn as nn

from s5 import BidirectionalS5SequenceBackbone, S5SequenceBackbone


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


class SessionLinearBank(nn.Module):
    def __init__(self, session_keys: tuple[str, ...], dim: int):
        super().__init__()
        self.dim = int(dim)
        unique_keys = tuple(dict.fromkeys(str(key) for key in session_keys))
        self.session_keys = unique_keys
        self._name_map = {session_key: self._module_key(session_key) for session_key in unique_keys}
        self.default_layer = self._identity_affine(self.dim)
        self.layers = nn.ModuleDict(
            {
                module_key: self._identity_affine(self.dim)
                for module_key in self._name_map.values()
            }
        )

    @staticmethod
    def _module_key(session_key: str) -> str:
        return str(session_key).replace(".", "_dot_").replace("/", "_slash_")

    @staticmethod
    def _identity_affine(dim: int) -> nn.Linear:
        layer = nn.Linear(dim, dim)
        with torch.no_grad():
            layer.weight.zero_()
            layer.weight += torch.eye(dim)
            layer.bias.zero_()
        return layer

    def forward(self, x: torch.Tensor, session_keys: list[str] | tuple[str, ...]) -> torch.Tensor:
        if len(session_keys) != x.shape[0]:
            raise ValueError(
                f"SessionLinearBank expected {x.shape[0]} session keys, got {len(session_keys)}."
            )
        transformed: list[torch.Tensor] = []
        for sample, session_key in zip(x, session_keys):
            module_key = self._name_map.get(str(session_key))
            layer = self.default_layer if module_key is None else self.layers[module_key]
            transformed.append(layer(sample))
        return torch.stack(transformed, dim=0)


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
        source_session_keys: tuple[str, ...] = (),
        feature_mode: str = "tx_only",
        backbone_direction: str = "bidirectional",
    ):
        del post_proj_norm  # kept for checkpoint compatibility
        super().__init__()
        if backbone_direction not in {"causal", "bidirectional"}:
            raise ValueError("backbone_direction must be one of {'causal', 'bidirectional'}")
        self.input_dim = int(input_dim)
        self.hidden_size = int(hidden_size)
        self.s5_state_size = int(s5_state_size)
        self.num_layers = int(num_layers)
        self.patch_size = int(patch_size)
        self.patch_stride = int(patch_stride)
        self.token_dim = self.input_dim * self.patch_size
        self.feature_mode = str(feature_mode)
        self.backbone_direction = str(backbone_direction)
        self.source_session_keys = tuple(str(key) for key in source_session_keys)
        self.source_readin = SessionLinearBank(self.source_session_keys, self.token_dim)
        self.input_patch_embedder = nn.Sequential(
            nn.LayerNorm(self.token_dim),
            nn.Linear(self.token_dim, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
        )
        backbone_cls = (
            S5SequenceBackbone
            if self.backbone_direction == "causal"
            else BidirectionalS5SequenceBackbone
        )
        self.backbone = backbone_cls(
            d_model=self.hidden_size,
            d_state=self.s5_state_size,
            num_layers=self.num_layers,
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

    def _apply_token_alignment(
        self,
        tokens: torch.Tensor,
        *,
        session_keys: list[str] | tuple[str, ...] | None,
        use_source_affines: bool,
        target_affines: SessionLinearBank | None,
    ) -> torch.Tensor:
        if target_affines is not None:
            if session_keys is None:
                raise ValueError("target_affines requires session_keys.")
            return target_affines(tokens, session_keys)
        if use_source_affines and session_keys is not None:
            return self.source_readin(tokens, session_keys)
        return tokens

    def encode_patched(
        self,
        tokens: torch.Tensor,
        token_lengths: torch.Tensor,
        *,
        token_mask: torch.Tensor | None = None,
        mask_token_placement: str = "before_projection",
        session_keys: list[str] | tuple[str, ...] | None = None,
        use_source_affines: bool = True,
        target_affines: SessionLinearBank | None = None,
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

        aligned_tokens = self._apply_token_alignment(
            tokens,
            session_keys=session_keys,
            use_source_affines=use_source_affines,
            target_affines=target_affines,
        )
        if mask_token_placement == "before_projection":
            mask_token = self.raw_mask_token.to(device=tokens.device, dtype=tokens.dtype).view(1, 1, -1)
            masked_tokens = torch.where(token_mask_bool.unsqueeze(-1), mask_token, aligned_tokens)
            hidden_input = self.input_patch_embedder(masked_tokens)
        else:
            hidden_input = self.input_patch_embedder(aligned_tokens)
            mask_token = self.hidden_mask_token.to(
                device=hidden_input.device,
                dtype=hidden_input.dtype,
            ).view(1, 1, -1)
            hidden_input = torch.where(token_mask_bool.unsqueeze(-1), mask_token, hidden_input)
            masked_tokens = aligned_tokens

        hidden = self.backbone(hidden_input, token_lengths)
        return {
            "hidden": hidden,
            "token_lengths": token_lengths,
            "tokens": tokens,
            "aligned_tokens": aligned_tokens,
            "masked_tokens": masked_tokens,
            "token_mask": token_mask_bool,
        }

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor,
        *,
        session_keys: list[str] | tuple[str, ...] | None = None,
        use_source_affines: bool = True,
        target_affines: SessionLinearBank | None = None,
    ) -> dict[str, torch.Tensor]:
        tokens, token_lengths = self.patch_batch(x, lengths)
        return self.encode_patched(
            tokens,
            token_lengths,
            token_mask=None,
            session_keys=session_keys,
            use_source_affines=use_source_affines,
            target_affines=target_affines,
        )


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
        source_session_keys: tuple[str, ...] = (),
        feature_mode: str = "tx_only",
        reconstruction_head_mode: str = "with_output_norm",
        reconstruction_head_type: str = "linear",
        backbone_direction: str = "bidirectional",
    ):
        super().__init__()
        self.feature_mode = str(feature_mode)
        self.source_session_keys = tuple(str(key) for key in source_session_keys)
        if reconstruction_head_mode not in {"with_output_norm", "no_output_norm"}:
            raise ValueError(
                "reconstruction_head_mode must be one of {'with_output_norm', 'no_output_norm'}"
            )
        if reconstruction_head_type not in {"linear", "mlp"}:
            raise ValueError("reconstruction_head_type must be one of {'linear', 'mlp'}")
        self.reconstruction_head_mode = str(reconstruction_head_mode)
        self.reconstruction_head_type = str(reconstruction_head_type)
        self.encoder = S5MaskedEncoder(
            input_dim=input_dim,
            hidden_size=hidden_size,
            s5_state_size=s5_state_size,
            num_layers=num_layers,
            dropout=dropout,
            patch_size=patch_size,
            patch_stride=patch_stride,
            post_proj_norm=post_proj_norm,
            source_session_keys=self.source_session_keys,
            feature_mode=self.feature_mode,
            backbone_direction=backbone_direction,
        )
        # Keep the legacy linear projection layers for checkpoint compatibility.
        self.reverse_patch_embedder = nn.Sequential(
            nn.LayerNorm(self.encoder.hidden_size),
            nn.Linear(self.encoder.hidden_size, self.encoder.token_dim),
            nn.LayerNorm(self.encoder.token_dim),
        )
        self.reverse_patch_mlp = nn.Sequential(
            nn.LayerNorm(self.encoder.hidden_size),
            nn.Linear(self.encoder.hidden_size, self.encoder.hidden_size),
            nn.GELU(),
            nn.Linear(self.encoder.hidden_size, self.encoder.token_dim),
        )
        self.source_readout = SessionLinearBank(self.source_session_keys, self.encoder.token_dim)

    def encode_sequence(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor,
        *,
        session_keys: list[str] | tuple[str, ...] | None = None,
        use_source_affines: bool = True,
        target_affines: SessionLinearBank | None = None,
    ) -> dict[str, torch.Tensor]:
        return self.encoder(
            x,
            lengths,
            session_keys=session_keys,
            use_source_affines=use_source_affines,
            target_affines=target_affines,
        )

    def reconstruct_from_patched_tokens(
        self,
        tokens: torch.Tensor,
        token_lengths: torch.Tensor,
        *,
        token_mask: torch.Tensor | None = None,
        mask_token_placement: str = "before_projection",
        session_keys: list[str] | tuple[str, ...] | None = None,
    ) -> dict[str, torch.Tensor]:
        outputs = self.encoder.encode_patched(
            tokens,
            token_lengths,
            token_mask=token_mask,
            mask_token_placement=mask_token_placement,
            session_keys=session_keys,
            use_source_affines=True,
            target_affines=None,
        )
        if self.reconstruction_head_type == "linear":
            reconstruction = self.reverse_patch_embedder[0](outputs["hidden"])
            reconstruction = self.reverse_patch_embedder[1](reconstruction)
        else:
            reconstruction = self.reverse_patch_mlp(outputs["hidden"])
        if self.reconstruction_head_mode == "with_output_norm":
            reconstruction = self.reverse_patch_embedder[2](reconstruction)
        if session_keys is not None:
            reconstruction = self.source_readout(reconstruction, session_keys)
        return {**outputs, "reconstruction": reconstruction}


# Compatibility aliases so the downstream probe helpers can stay nearly unchanged.
S5ContrastiveEncoder = S5MaskedEncoder
ContrastiveSSLModel = MaskedSSLModel

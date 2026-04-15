"""MAE-style masked-reconstruction model definitions for Colab experiments."""

from __future__ import annotations

import torch
import torch.nn as nn

from s5 import BidirectionalS5SequenceBackbone, S5SequenceBackbone


def sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _sequence_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
    positions = torch.arange(max_len, device=lengths.device).unsqueeze(0)
    return positions < lengths.unsqueeze(1)


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


def _patch_token_count(length: int, patch_size: int, patch_stride: int) -> int:
    if length <= 0:
        return 1
    if patch_size == 1:
        return int(length)
    max_start = max(length - patch_size, 0)
    starts = list(range(0, max(length - patch_size + 1, 1), patch_stride))
    if not starts:
        starts = [0]
    if starts[-1] != max_start:
        starts.append(max_start)
    return len(starts)


class S5MaskedEncoder(nn.Module):
    """MAE encoder that only processes visible tokens."""

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
        max_patches: int,
        source_session_keys: tuple[str, ...] = (),
        feature_mode: str = "tx_only",
        backbone_direction: str = "bidirectional",
    ):
        del post_proj_norm  # kept for checkpoint compatibility
        super().__init__()
        if backbone_direction not in {"causal", "bidirectional"}:
            raise ValueError("backbone_direction must be one of {'causal', 'bidirectional'}")
        if int(max_patches) <= 0:
            raise ValueError("max_patches must be positive")

        self.input_dim = int(input_dim)
        self.hidden_size = int(hidden_size)
        self.s5_state_size = int(s5_state_size)
        self.num_layers = int(num_layers)
        self.patch_size = int(patch_size)
        self.patch_stride = int(patch_stride)
        self.token_dim = self.input_dim * self.patch_size
        self.max_patches = int(max_patches)
        self.feature_mode = str(feature_mode)
        self.backbone_direction = str(backbone_direction)
        self.source_session_keys = tuple(str(key) for key in source_session_keys)

        self.source_readin = SessionLinearBank(self.source_session_keys, self.token_dim)
        self.input_patch_embedder = nn.Sequential(
            nn.LayerNorm(self.token_dim),
            nn.Linear(self.token_dim, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
        )
        self.encoder_pos_embed = nn.Parameter(torch.zeros(1, self.max_patches, self.hidden_size))

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
        mask_token_placement: str = "visible_only",
        session_keys: list[str] | tuple[str, ...] | None = None,
        use_source_affines: bool = True,
        target_affines: SessionLinearBank | None = None,
    ) -> dict[str, torch.Tensor]:
        del mask_token_placement
        token_mask_bool = (
            token_mask.to(device=tokens.device, dtype=torch.bool)
            if token_mask is not None
            else torch.zeros(tokens.shape[:2], device=tokens.device, dtype=torch.bool)
        )

        aligned_tokens = self._apply_token_alignment(
            tokens,
            session_keys=session_keys,
            use_source_affines=use_source_affines,
            target_affines=target_affines,
        )

        hidden_all = self.input_patch_embedder(aligned_tokens)
        if hidden_all.shape[1] > self.max_patches:
            raise ValueError(
                f"Token count {hidden_all.shape[1]} exceeds max_patches={self.max_patches}. "
                "Increase max_patches in MAE config."
            )
        hidden_all = hidden_all + self.encoder_pos_embed[:, : hidden_all.shape[1]].to(hidden_all.dtype)

        valid_token_mask = _sequence_mask(token_lengths, hidden_all.shape[1])
        visible_token_mask = valid_token_mask & ~token_mask_bool
        visible_lengths = visible_token_mask.sum(dim=1)
        if torch.any(visible_lengths <= 0):
            raise ValueError(
                "MAE encoder received zero visible tokens for at least one sample. "
                "Reduce mask_ratio or span length."
            )

        max_visible = int(visible_lengths.max().item())
        visible_hidden = hidden_all.new_zeros((hidden_all.shape[0], max_visible, hidden_all.shape[-1]))
        visible_positions = torch.zeros(
            (hidden_all.shape[0], max_visible),
            device=hidden_all.device,
            dtype=torch.long,
        )
        for sample_idx in range(hidden_all.shape[0]):
            visible_idx = torch.nonzero(visible_token_mask[sample_idx], as_tuple=False).squeeze(1)
            count = int(visible_idx.numel())
            visible_hidden[sample_idx, :count] = hidden_all[sample_idx, visible_idx]
            visible_positions[sample_idx, :count] = visible_idx

        encoded_visible = self.backbone(visible_hidden, visible_lengths.to(torch.long))
        return {
            "hidden": encoded_visible,
            "token_lengths": token_lengths,
            "tokens": tokens,
            "aligned_tokens": aligned_tokens,
            "token_mask": token_mask_bool,
            "visible_positions": visible_positions,
            "visible_lengths": visible_lengths.to(torch.long),
            "visible_token_mask": visible_token_mask,
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
    """MAE model with visible-only encoder and full-token decoder."""

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
        max_patches: int = 128,
        decoder_hidden_size: int | None = None,
        decoder_s5_state_size: int | None = None,
        decoder_num_layers: int = 1,
        decoder_dropout: float = 0.0,
        decoder_backbone_direction: str = "bidirectional",
    ):
        super().__init__()
        if reconstruction_head_mode not in {"with_output_norm", "no_output_norm"}:
            raise ValueError(
                "reconstruction_head_mode must be one of {'with_output_norm', 'no_output_norm'}"
            )
        if reconstruction_head_type not in {"linear", "mlp"}:
            raise ValueError("reconstruction_head_type must be one of {'linear', 'mlp'}")
        if decoder_backbone_direction not in {"causal", "bidirectional"}:
            raise ValueError("decoder_backbone_direction must be one of {'causal', 'bidirectional'}")

        self.feature_mode = str(feature_mode)
        self.source_session_keys = tuple(str(key) for key in source_session_keys)
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
            max_patches=max_patches,
            source_session_keys=self.source_session_keys,
            feature_mode=self.feature_mode,
            backbone_direction=backbone_direction,
        )

        self.max_patches = int(max_patches)
        self.decoder_hidden_size = (
            int(hidden_size) if decoder_hidden_size is None else int(decoder_hidden_size)
        )
        decoder_s5_state = (
            int(s5_state_size) if decoder_s5_state_size is None else int(decoder_s5_state_size)
        )
        self.decoder_proj = (
            nn.Identity()
            if self.decoder_hidden_size == self.encoder.hidden_size
            else nn.Linear(self.encoder.hidden_size, self.decoder_hidden_size)
        )
        self.decoder_mask_token = nn.Parameter(torch.zeros(self.decoder_hidden_size))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, self.max_patches, self.decoder_hidden_size))

        decoder_backbone_cls = (
            S5SequenceBackbone
            if decoder_backbone_direction == "causal"
            else BidirectionalS5SequenceBackbone
        )
        self.decoder_backbone = decoder_backbone_cls(
            d_model=self.decoder_hidden_size,
            d_state=decoder_s5_state,
            num_layers=int(decoder_num_layers),
            dropout=float(decoder_dropout),
            ffn_multiplier=2.0,
        )

        self.reverse_patch_embedder = nn.Sequential(
            nn.LayerNorm(self.decoder_hidden_size),
            nn.Linear(self.decoder_hidden_size, self.encoder.token_dim),
            nn.LayerNorm(self.encoder.token_dim),
        )
        self.reverse_patch_mlp = nn.Sequential(
            nn.LayerNorm(self.decoder_hidden_size),
            nn.Linear(self.decoder_hidden_size, self.decoder_hidden_size),
            nn.GELU(),
            nn.Linear(self.decoder_hidden_size, self.encoder.token_dim),
        )
        self.source_readout = SessionLinearBank(self.source_session_keys, self.encoder.token_dim)

    def _build_decoder_input(
        self,
        *,
        encoded_visible: torch.Tensor,
        visible_positions: torch.Tensor,
        visible_lengths: torch.Tensor,
        token_lengths: torch.Tensor,
    ) -> torch.Tensor:
        max_tokens = int(token_lengths.max().item())
        if max_tokens > self.max_patches:
            raise ValueError(
                f"Token count {max_tokens} exceeds max_patches={self.max_patches}. "
                "Increase max_patches in MAE config."
            )
        if max_tokens > self.decoder_pos_embed.shape[1]:
            raise ValueError(
                f"Decoder positional table too short: need {max_tokens}, "
                f"have {self.decoder_pos_embed.shape[1]}"
            )

        decoder_input = self.decoder_mask_token.view(1, 1, -1).expand(
            encoded_visible.shape[0],
            max_tokens,
            -1,
        ).clone()
        decoder_input = decoder_input + self.decoder_pos_embed[:, :max_tokens].to(decoder_input.dtype)

        projected_visible = self.decoder_proj(encoded_visible)
        pos_embed_slice = self.decoder_pos_embed[:, :max_tokens].to(projected_visible.dtype)
        for sample_idx in range(encoded_visible.shape[0]):
            visible_count = int(visible_lengths[sample_idx].item())
            pos = visible_positions[sample_idx, :visible_count]
            decoder_input[sample_idx, pos] = (
                projected_visible[sample_idx, :visible_count]
                + pos_embed_slice[0, pos]
            )

        valid_token_mask = _sequence_mask(token_lengths, max_tokens).unsqueeze(-1).to(decoder_input.dtype)
        return decoder_input * valid_token_mask

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
        mask_token_placement: str = "visible_only",
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
        decoder_input = self._build_decoder_input(
            encoded_visible=outputs["hidden"],
            visible_positions=outputs["visible_positions"],
            visible_lengths=outputs["visible_lengths"],
            token_lengths=token_lengths,
        )
        decoder_hidden = self.decoder_backbone(decoder_input, token_lengths)
        if self.reconstruction_head_type == "linear":
            reconstruction = self.reverse_patch_embedder[0](decoder_hidden)
            reconstruction = self.reverse_patch_embedder[1](reconstruction)
        else:
            reconstruction = self.reverse_patch_mlp(decoder_hidden)
        if self.reconstruction_head_mode == "with_output_norm":
            reconstruction = self.reverse_patch_embedder[2](reconstruction)
        if session_keys is not None:
            reconstruction = self.source_readout(reconstruction, session_keys)
        return {**outputs, "decoder_hidden": decoder_hidden, "reconstruction": reconstruction}


# Compatibility aliases so shared helper code can remain mostly unchanged.
S5ContrastiveEncoder = S5MaskedEncoder
ContrastiveSSLModel = MaskedSSLModel
MAEMaskedSSLModel = MaskedSSLModel
MAX_PATCH_COUNT = _patch_token_count

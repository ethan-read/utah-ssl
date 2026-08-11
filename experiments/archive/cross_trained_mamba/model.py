"""Hierarchical causal Mamba phoneme decoder for area-6v speech data."""

from __future__ import annotations

import hashlib

import torch
import torch.nn as nn

try:
    from experiments.archive.generic_ssm_ssl.model import build_sequence_backbone
    from experiments.supervised_baselines.model import SessionInputAdapterBank
except ModuleNotFoundError:  # pragma: no cover - repo-root unittest fallback
    from experiments.archive.generic_ssm_ssl.model import build_sequence_backbone
    from experiments.supervised_baselines.model import SessionInputAdapterBank


class AffineAdapterBank(nn.Module):
    """Dataset/session/day-specific affine input transforms."""

    def __init__(self, adapter_keys: tuple[str, ...] | list[str], *, input_dim: int) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        unique_keys = tuple(dict.fromkeys(str(key) for key in adapter_keys))
        self._name_map = {key: self._module_key(key) for key in unique_keys}
        self.default_layer = nn.Linear(self.input_dim, self.input_dim)
        self.layers = nn.ModuleDict(
            {module_key: nn.Linear(self.input_dim, self.input_dim) for module_key in self._name_map.values()}
        )
        self._reset_linear(self.default_layer)
        for layer in self.layers.values():
            self._reset_linear(layer)

    @staticmethod
    def _module_key(adapter_key: str) -> str:
        digest = hashlib.sha1(str(adapter_key).encode("utf-8")).hexdigest()
        return f"adapter_{digest}"

    def _reset_linear(self, layer: nn.Linear) -> None:
        with torch.no_grad():
            layer.weight.copy_(torch.eye(self.input_dim, dtype=layer.weight.dtype))
            layer.bias.zero_()

    def _layer_for_key(self, adapter_key: str) -> nn.Linear:
        module_key = self._name_map.get(str(adapter_key))
        if module_key is None:
            return self.default_layer
        return self.layers[module_key]

    def forward(
        self,
        x: torch.Tensor,
        adapter_keys: list[str] | tuple[str, ...] | None,
        *,
        session_adapter_enabled: bool,
    ) -> torch.Tensor:
        if not bool(session_adapter_enabled):
            return self.default_layer(x)
        if adapter_keys is None:
            raise ValueError("adapter_keys are required when session adaptation is enabled")
        if len(adapter_keys) != int(x.shape[0]):
            raise ValueError("adapter_keys length must match the batch size")
        adapted = [self._layer_for_key(str(key))(x[row_idx]) for row_idx, key in enumerate(adapter_keys)]
        return torch.stack(adapted, dim=0)


class CrossTrainedMambaPhonemeModel(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        vocab_size: int,
        hidden_size: int = 512,
        state_size: int = 64,
        stage1_num_layers: int = 2,
        stage2_num_layers: int = 2,
        stage3_num_layers: int = 1,
        dropout: float = 0.1,
        ffn_multiplier: float = 2.0,
        adapter_mode: str = "affine",
        session_adapter_keys: tuple[str, ...] = (),
        session_adapter_enabled: bool = True,
        feedback_detach: bool = False,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.vocab_size = int(vocab_size)
        self.hidden_size = int(hidden_size)
        self.feedback_detach = bool(feedback_detach)
        self.adapter_mode = str(adapter_mode)
        self.session_adapter_enabled = bool(session_adapter_enabled)
        if self.adapter_mode not in {"affine", "stanford_input_net"}:
            raise ValueError("adapter_mode must be one of {'affine', 'stanford_input_net'}")

        if self.adapter_mode == "affine":
            self.session_input_adapter = AffineAdapterBank(tuple(session_adapter_keys), input_dim=self.input_dim)
        else:
            self.session_input_adapter = SessionInputAdapterBank(
                tuple(session_adapter_keys),
                input_dim=self.input_dim,
                output_dim=self.input_dim,
                dropout=float(dropout),
            )

        self.input_norm = nn.LayerNorm(self.input_dim)
        self.input_projection = nn.Linear(self.input_dim, self.hidden_size)
        self.hidden_norm = nn.LayerNorm(self.hidden_size)

        self.stage1 = build_sequence_backbone(
            backbone_type="mamba",
            hidden_size=self.hidden_size,
            state_size=int(state_size),
            num_layers=int(stage1_num_layers),
            dropout=float(dropout),
            direction="causal",
            ffn_multiplier=float(ffn_multiplier),
        )
        self.stage2 = build_sequence_backbone(
            backbone_type="mamba",
            hidden_size=self.hidden_size,
            state_size=int(state_size),
            num_layers=int(stage2_num_layers),
            dropout=float(dropout),
            direction="causal",
            ffn_multiplier=float(ffn_multiplier),
        )
        self.stage3 = build_sequence_backbone(
            backbone_type="mamba",
            hidden_size=self.hidden_size,
            state_size=int(state_size),
            num_layers=int(stage3_num_layers),
            dropout=float(dropout),
            direction="causal",
            ffn_multiplier=float(ffn_multiplier),
        )

        self.head1 = nn.Linear(self.hidden_size, self.vocab_size)
        self.head2 = nn.Linear(self.hidden_size, self.vocab_size)
        self.head3 = nn.Linear(self.hidden_size, self.vocab_size)
        self.feedback1 = nn.Linear(self.vocab_size, self.hidden_size)
        self.feedback2 = nn.Linear(self.vocab_size, self.hidden_size)

    def _apply_adapter(
        self,
        x: torch.Tensor,
        session_ids: list[str] | tuple[str, ...] | None,
    ) -> torch.Tensor:
        return self.session_input_adapter(
            x,
            session_ids,
            session_adapter_enabled=self.session_adapter_enabled,
        )

    def _feedback(self, logits: torch.Tensor, projection: nn.Linear) -> torch.Tensor:
        probs = torch.softmax(logits, dim=-1)
        if self.feedback_detach:
            probs = probs.detach()
        return projection(probs)

    def forward(
        self,
        x: torch.Tensor,
        input_lengths: torch.Tensor,
        *,
        session_ids: list[str] | tuple[str, ...] | None = None,
    ) -> dict[str, torch.Tensor]:
        if x.ndim != 3:
            raise ValueError(f"Expected x to have shape [B, T, D], got {tuple(x.shape)}")
        if int(x.shape[-1]) != int(self.input_dim):
            raise ValueError(f"Expected input feature dim {self.input_dim}, got {int(x.shape[-1])}")

        adapted_input = self._apply_adapter(x, session_ids)
        hidden0 = self.hidden_norm(self.input_projection(self.input_norm(adapted_input)))

        hidden1 = self.stage1(hidden0, input_lengths)
        logits1 = self.head1(hidden1)
        hidden2_in = hidden1 + self._feedback(logits1, self.feedback1)

        hidden2 = self.stage2(hidden2_in, input_lengths)
        logits2 = self.head2(hidden2)
        hidden3_in = hidden2 + self._feedback(logits2, self.feedback2)

        hidden3 = self.stage3(hidden3_in, input_lengths)
        logits3 = self.head3(hidden3)
        return {
            "adapted_input": adapted_input,
            "hidden0": hidden0,
            "hidden1": hidden1,
            "hidden2": hidden2,
            "hidden3": hidden3,
            "l1": logits1,
            "l2": logits2,
            "l3": logits3,
            "logits": logits3,
            "token_lengths": input_lengths,
        }

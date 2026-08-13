from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from experiments.bit_style.config import BITStyleConfig
from experiments.bit_style.model import (
    BITStyleCTCModel,
    BITStyleEncoder,
    BITStylePretrainingModel,
)
from experiments.bit_style.objectives import masked_reconstruction_loss
from experiments.bit_style.training import load_encoder_checkpoint
from utah_ssl.experiment_contract import (
    DatasetPlan,
    SignalSpec,
)


class BITStyleTest(unittest.TestCase):
    def test_config_round_trip(self) -> None:
        config = BITStyleConfig(
            input_mode="temporal_patch",
            hidden_size=8,
            state_size=4,
            num_layers=1,
            dataset_plan=DatasetPlan.from_mapping(
                {
                    "brain2text24": ("competition_train",),
                    "toy": ("train", "val"),
                }
            ),
            signal_spec=SignalSpec.tx_only(tx_dim=256),
        )
        recovered = BITStyleConfig.from_dict(config.to_dict())
        self.assertEqual(recovered.input_mode, "temporal_patch")
        self.assertEqual(recovered.dataset_plan, config.dataset_plan)
        self.assertEqual(recovered.signal_spec, config.signal_spec)

    def test_s5_encoder_forward(self) -> None:
        encoder = BITStyleEncoder(
            input_dim=3,
            hidden_size=8,
            state_size=4,
            num_layers=1,
            dropout=0.0,
            input_mode="temporal_patch",
            patch_size=2,
            patch_stride=1,
        )
        x = torch.randn(2, 5, 3)
        lengths = torch.tensor([5, 3], dtype=torch.long)
        outputs = encoder.encode(x, lengths)
        self.assertEqual(tuple(outputs.hidden.shape), (2, 4, 8))
        self.assertEqual(outputs.token_lengths.tolist(), [4, 2])

    def test_masked_reconstruction_backward(self) -> None:
        model = BITStylePretrainingModel(
            BITStyleEncoder(
                input_dim=3,
                hidden_size=8,
                state_size=4,
                num_layers=1,
                dropout=0.0,
                input_mode="raw_bin",
            )
        )
        batch = {
            "x": torch.randn(2, 6, 3),
            "lengths": torch.tensor([6, 4], dtype=torch.long),
        }
        loss, metrics = masked_reconstruction_loss(
            model,
            batch,
            device=torch.device("cpu"),
            time_mask_ratio=0.25,
            channel_mask_ratio=0.25,
            chunk_size=2,
        )
        loss.backward()
        self.assertGreater(float(metrics["masked_entry_fraction"]), 0.0)
        grad_norm = sum(
            float(param.grad.abs().sum().item())
            for param in model.parameters()
            if param.grad is not None
        )
        self.assertGreater(grad_norm, 0.0)

    def test_encoder_checkpoint_loads_into_ctc_model(self) -> None:
        encoder = BITStyleEncoder(
            input_dim=3,
            hidden_size=8,
            state_size=4,
            num_layers=1,
            dropout=0.0,
            input_mode="raw_bin",
        )
        with tempfile.TemporaryDirectory(prefix="ssm_ssl_checkpoint_test_") as tmp:
            path = Path(tmp) / "checkpoint.pt"
            torch.save({"encoder_state": encoder.state_dict()}, path)
            ctc_model = BITStyleCTCModel(
                encoder=BITStyleEncoder(
                    input_dim=3,
                    hidden_size=8,
                    state_size=4,
                    num_layers=1,
                    dropout=0.0,
                    input_mode="raw_bin",
                ),
                vocab_size=5,
            )
            payload = load_encoder_checkpoint(ctc_model.encoder, path)
            self.assertIn("encoder_state", payload)
            x = torch.randn(2, 5, 3)
            lengths = torch.tensor([5, 3], dtype=torch.long)
            outputs = ctc_model(x, lengths)
            self.assertEqual(tuple(outputs["logits"].shape), (2, 5, 5))


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[5]
EXPERIMENTS_DIR = REPO_ROOT / "analysis" / "active" / "ssl_experiments"
for path in (REPO_ROOT, EXPERIMENTS_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from analysis.active.ssl_experiments.ssm_ssl.config import GenericSSMSSLConfig
from analysis.active.ssl_experiments.ssm_ssl.model import (
    GenericMaskedSSMModel,
    GenericSSMCTCModel,
    GenericSSMEncoder,
)
from analysis.active.ssl_experiments.ssm_ssl.objectives import masked_reconstruction_loss
from analysis.active.ssl_experiments.ssm_ssl.training import load_encoder_checkpoint
from analysis.active.ssl_experiments.ssl_core.experiment_contract import (
    DatasetPlan,
    SignalSpec,
)


class GenericSSMSSLTest(unittest.TestCase):
    def test_config_round_trip(self) -> None:
        config = GenericSSMSSLConfig(
            backbone_type="s5",
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
        recovered = GenericSSMSSLConfig.from_dict(config.to_dict())
        self.assertEqual(recovered.backbone_type, "s5")
        self.assertEqual(recovered.input_mode, "temporal_patch")
        self.assertEqual(recovered.dataset_plan, config.dataset_plan)
        self.assertEqual(recovered.signal_spec, config.signal_spec)

    def test_s5_encoder_forward(self) -> None:
        encoder = GenericSSMEncoder(
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

    def test_mamba_encoder_optional_forward(self) -> None:
        try:
            encoder = GenericSSMEncoder(
                input_dim=3,
                hidden_size=8,
                state_size=4,
                num_layers=1,
                dropout=0.0,
                backbone_type="mamba",
                input_mode="raw_bin",
            )
        except ImportError as exc:
            self.skipTest(str(exc))
        x = torch.randn(2, 5, 3)
        lengths = torch.tensor([5, 4], dtype=torch.long)
        outputs = encoder.encode(x, lengths)
        self.assertEqual(tuple(outputs.hidden.shape), (2, 5, 8))

    def test_masked_reconstruction_backward(self) -> None:
        model = GenericMaskedSSMModel(
            GenericSSMEncoder(
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
        encoder = GenericSSMEncoder(
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
            ctc_model = GenericSSMCTCModel(
                encoder=GenericSSMEncoder(
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

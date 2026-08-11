from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.archive.mamba_future_prediction import (
    FuturePredictionSSLConfig,
    aggregate_time_bins,
    build_future_prediction_targets,
    future_prediction_loss,
    make_future_prediction_model,
)
from experiments.archive.mamba_future_prediction.training import _latest_stage_checkpoint
from experiments.archive.mamba_future_prediction.training import _load_checkpoint_payload
from experiments.archive.mamba_future_prediction.training import run_future_prediction_pretraining
from experiments.archive.generic_ssm_ssl.model import GenericSSMCTCModel


class FuturePredictionSSLTests(unittest.TestCase):
    class _StaticFutureModel(nn.Module):
        def __init__(self, forecast: torch.Tensor) -> None:
            super().__init__()
            self.future_bins = int(forecast.shape[2])
            self.register_buffer("_forecast", forecast)

        def forward(self, x: torch.Tensor, input_lengths: torch.Tensor) -> dict[str, torch.Tensor]:
            return {
                "hidden": x,
                "tokens": x,
                "forecast": self._forecast.to(device=x.device, dtype=x.dtype),
                "token_lengths": input_lengths,
            }

    def test_aggregate_time_bins_mean_pools_and_truncates_lengths(self) -> None:
        x = torch.tensor(
            [
                [[1.0], [3.0], [5.0], [7.0]],
                [[2.0], [4.0], [0.0], [0.0]],
            ]
        )
        lengths = torch.tensor([4, 2], dtype=torch.long)

        pooled, pooled_lengths = aggregate_time_bins(x, lengths, stride=2)

        self.assertTrue(torch.equal(pooled_lengths, torch.tensor([2, 1], dtype=torch.long)))
        self.assertTrue(torch.equal(pooled[0, :, 0], torch.tensor([2.0, 6.0])))
        self.assertTrue(torch.equal(pooled[1, :, 0], torch.tensor([3.0, 0.0])))

    def test_build_future_prediction_targets_respects_lengths(self) -> None:
        x = torch.tensor(
            [
                [[1.0], [2.0], [3.0], [0.0]],
                [[10.0], [20.0], [0.0], [0.0]],
            ]
        )
        lengths = torch.tensor([3, 2], dtype=torch.long)

        targets, valid = build_future_prediction_targets(x, lengths, future_bins=2)

        self.assertEqual(tuple(targets.shape), (2, 4, 2, 1))
        self.assertTrue(torch.equal(targets[0, 0, :, 0], torch.tensor([2.0, 3.0])))
        self.assertTrue(torch.equal(targets[0, 1, :, 0], torch.tensor([3.0, 0.0])))
        self.assertTrue(torch.equal(targets[1, 0, :, 0], torch.tensor([20.0, 0.0])))
        self.assertTrue(torch.equal(valid[0, :, 0], torch.tensor([True, True, False, False])))
        self.assertTrue(torch.equal(valid[0, :, 1], torch.tensor([True, False, False, False])))
        self.assertTrue(torch.equal(valid[1, :, 0], torch.tensor([True, False, False, False])))
        self.assertTrue(torch.equal(valid[1, :, 1], torch.tensor([False, False, False, False])))

    def test_future_prediction_model_outputs_expected_shape(self) -> None:
        config = FuturePredictionSSLConfig(
            backbone_type="s5",
            hidden_size=8,
            state_size=4,
            num_layers=1,
            future_bins=3,
            tx_dim=4,
            sbp_dim=0,
            feature_mode="tx_only",
        )
        model = make_future_prediction_model(config, input_dim=4)
        x = torch.randn(2, 6, 4)
        lengths = torch.tensor([6, 4], dtype=torch.long)

        outputs = model(x, lengths)

        self.assertEqual(tuple(outputs["hidden"].shape), (2, 6, 8))
        self.assertEqual(tuple(outputs["forecast"].shape), (2, 6, 3, 4))
        self.assertTrue(torch.equal(outputs["token_lengths"], lengths))

    def test_future_prediction_loss_returns_horizon_metrics(self) -> None:
        config = FuturePredictionSSLConfig(
            backbone_type="s5",
            hidden_size=8,
            state_size=4,
            num_layers=1,
            future_bins=2,
            tx_dim=3,
            sbp_dim=0,
            feature_mode="tx_only",
        )
        model = make_future_prediction_model(config, input_dim=3)
        batch = {
            "x": torch.randn(2, 5, 3),
            "lengths": torch.tensor([5, 4], dtype=torch.long),
        }

        loss, metrics = future_prediction_loss(
            model,
            batch,
            device=torch.device("cpu"),
            delta=1.0,
            temporal_bin_stride=1,
            variance_match_weight=0.05,
            tx_dim=3,
            sbp_dim=0,
            feature_mode="tx_only",
            use_normalization=False,
        )

        self.assertGreater(float(loss.item()), 0.0)
        self.assertIn("h1_mae", metrics)
        self.assertIn("h2_mae", metrics)
        self.assertIn("zero_baseline_mae", metrics)
        self.assertIn("pred_std", metrics)
        self.assertIn("base_loss", metrics)
        self.assertIn("variance_match_penalty", metrics)
        self.assertIn("tx_loss", metrics)
        self.assertIn("sbp_loss", metrics)

    def test_future_prediction_loss_supports_tx_poisson_and_sbp_huber(self) -> None:
        raw = torch.tensor([[[2.0, 10.0], [4.0, 12.0], [8.0, 14.0]]], dtype=torch.float32)
        mean = torch.tensor([2.0, 10.0], dtype=torch.float32)
        std = torch.tensor([2.0, 2.0], dtype=torch.float32)
        x_norm = (raw - mean) / std
        forecast = torch.tensor([[[[1.0, 0.75]], [[1.5, 1.25]], [[0.0, 0.0]]]], dtype=torch.float32)
        model = self._StaticFutureModel(forecast)
        batch = {
            "x": x_norm,
            "lengths": torch.tensor([3], dtype=torch.long),
            "boundary_keys": ["toy:sess0"],
        }

        loss, metrics = future_prediction_loss(
            model,
            batch,
            device=torch.device("cpu"),
            delta=1.0,
            temporal_bin_stride=1,
            variance_match_weight=0.0,
            tx_dim=1,
            sbp_dim=1,
            feature_mode="tx_sbp",
            use_normalization=True,
            tx_loss_type="poisson_nll",
            sbp_loss_type="huber",
            session_feature_stats={"toy:sess0": (mean, std)},
        )

        expected_tx = F.poisson_nll_loss(
            F.softplus(torch.tensor([1.0, 1.5])),
            torch.tensor([4.0, 8.0]),
            log_input=False,
            reduction="none",
        ).mean()
        expected_sbp = F.huber_loss(
            torch.tensor([0.75, 1.25]),
            torch.tensor([1.0, 2.0]),
            delta=1.0,
            reduction="none",
        ).mean()

        self.assertTrue(torch.isfinite(loss))
        self.assertAlmostEqual(metrics["tx_loss"], float(expected_tx.item()), places=5)
        self.assertAlmostEqual(metrics["sbp_loss"], float(expected_sbp.item()), places=5)
        self.assertIn("h1_mae", metrics)

    def test_frozen_probe_only_learns_classifier(self) -> None:
        config = FuturePredictionSSLConfig(
            backbone_type="s5",
            hidden_size=8,
            state_size=4,
            num_layers=1,
            future_bins=2,
            tx_dim=3,
            sbp_dim=0,
            feature_mode="tx_only",
        )
        encoder = make_future_prediction_model(config, input_dim=3).encoder
        for parameter in encoder.parameters():
            parameter.requires_grad_(False)
        model = GenericSSMCTCModel(encoder=encoder, vocab_size=5)
        trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}

        self.assertTrue(trainable)
        self.assertTrue(all(name.startswith("classifier.") for name in trainable))

    def test_config_round_trips_resume_fields(self) -> None:
        config = FuturePredictionSSLConfig(
            resume=True,
            resume_checkpoint_path="/tmp/future_checkpoint.pt",
        )

        recovered = FuturePredictionSSLConfig.from_dict(config.to_dict())

        self.assertTrue(recovered.resume)
        self.assertEqual(str(recovered.resume_checkpoint_path), "/tmp/future_checkpoint.pt")

    def test_latest_stage_checkpoint_prefers_highest_step(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            stage_dir = Path(tmpdir)
            checkpoints_dir = stage_dir / "checkpoints"
            checkpoints_dir.mkdir()
            best = stage_dir / "checkpoint_best.pt"
            final = stage_dir / "checkpoint_final.pt"
            step_100 = checkpoints_dir / "step_000100.pt"
            step_200 = checkpoints_dir / "step_000200.pt"
            for path, step in ((best, 50), (final, 150), (step_100, 100), (step_200, 200)):
                torch.save({"step": step}, path)

            latest = _latest_stage_checkpoint(
                stage_dir=stage_dir,
                final_checkpoint_path=final,
                best_checkpoint_path=best,
            )

            self.assertEqual(latest, step_200)

    def test_checkpoint_loader_allows_non_tensor_resume_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = Path(tmpdir) / "checkpoint.pt"
            torch.save({"step": 3, "rng_state": {"python": ("non_tensor", 1)}}, checkpoint)

            payload = _load_checkpoint_payload(checkpoint)

            self.assertEqual(payload["step"], 3)
            self.assertEqual(payload["rng_state"]["python"], ("non_tensor", 1))

    def test_explicit_missing_resume_checkpoint_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = FuturePredictionSSLConfig(
                backbone_type="s5",
                hidden_size=8,
                state_size=4,
                num_layers=1,
                tx_dim=3,
                sbp_dim=0,
                feature_mode="tx_only",
                segment_bins=8,
                batch_size=1,
                ssl_steps=1,
                resume=True,
                resume_checkpoint_path=Path(tmpdir) / "missing.pt",
                cache_root=Path(tmpdir),
            )

            with self.assertRaises(FileNotFoundError):
                run_future_prediction_pretraining(config, run_dir=Path(tmpdir) / "run")


if __name__ == "__main__":
    unittest.main()

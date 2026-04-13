from __future__ import annotations

import json
import random
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch


REPO_ROOT = Path(__file__).resolve().parents[5]
EXPERIMENTS_DIR = REPO_ROOT / "analysis" / "active" / "ssl_experiments"
for path in (REPO_ROOT, EXPERIMENTS_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from masked_ssl.model import MaskedSSLModel
from masked_ssl.objectives import (
    build_masked_batch,
    compute_masked_reconstruction_metrics,
    sample_mask_indices,
    summarize_metrics,
)
from masked_ssl.probe import DownstreamProbeConfig, recover_downstream_probe_state
from masked_ssl.training import recover_ssl_run_state_from_checkpoint


def _make_model(*, input_dim: int = 4, patch_size: int = 2, patch_stride: int = 2) -> MaskedSSLModel:
    return MaskedSSLModel(
        input_dim=input_dim,
        hidden_size=8,
        s5_state_size=4,
        num_layers=1,
        dropout=0.0,
        patch_size=patch_size,
        patch_stride=patch_stride,
        post_proj_norm="rms",
    )


def _make_batch(*, batch_size: int = 2, time_bins: int = 6, input_dim: int = 4) -> dict[str, torch.Tensor]:
    x = torch.arange(batch_size * time_bins * input_dim, dtype=torch.float32).reshape(
        batch_size,
        time_bins,
        input_dim,
    )
    feature_mask = torch.tensor(
        [
            [1.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 1.0, 0.0],
        ],
        dtype=torch.float32,
    )[:batch_size]
    lengths = torch.tensor([time_bins, time_bins - 1], dtype=torch.long)[:batch_size]
    return {
        "x": x[:batch_size],
        "feature_mask": feature_mask,
        "lengths": lengths,
    }


def _checkpoint_config(model: MaskedSSLModel) -> dict[str, object]:
    return {
        "segment_bins": 10,
        "patch_size": model.encoder.patch_size,
        "patch_stride": model.encoder.patch_stride,
        "hidden_size": model.encoder.hidden_size,
        "s5_state_size": model.encoder.backbone.blocks[0].ssm.d_state,
        "num_layers": len(model.encoder.backbone.blocks),
        "dropout": 0.0,
        "batch_size": 2,
        "seed": 7,
        "dataset_weight_alpha": 0.25,
        "examples_per_shard": 2,
        "learning_rate": 3e-4,
        "weight_decay": 1e-2,
        "mask_unit": "patch",
        "mask_token_placement": "before_projection",
        "mask_ratio": 0.4,
        "span_length_min": 1,
        "span_length_max": 2,
        "num_spans_mode": "one",
        "allow_bin_fractional_overlap": True,
        "post_proj_norm": "rms",
    }


class MaskedSSLTests(unittest.TestCase):
    def test_sample_mask_indices_one_span_is_contiguous_and_matches_target_count(self) -> None:
        random.seed(0)
        mask = sample_mask_indices(
            length=16,
            mask_ratio=0.4,
            span_length_min=2,
            span_length_max=8,
            num_spans_mode="one",
        )
        true_idx = torch.nonzero(mask, as_tuple=False).squeeze(1)
        self.assertGreater(true_idx.numel(), 0)
        self.assertTrue(
            torch.equal(true_idx, torch.arange(int(true_idx[0]), int(true_idx[-1]) + 1))
        )
        self.assertEqual(int(mask.sum().item()), 6)

    def test_sample_mask_indices_one_span_raises_when_exact_target_is_impossible(self) -> None:
        with self.assertRaisesRegex(ValueError, "num_spans_mode='one'"):
            sample_mask_indices(
                length=20,
                mask_ratio=0.8,
                span_length_min=2,
                span_length_max=8,
                num_spans_mode="one",
            )

    def test_patch_level_mask_only_scores_masked_present_features(self) -> None:
        random.seed(1)
        model = _make_model()
        batch = _make_batch()
        masked = build_masked_batch(
            model,
            batch,
            mask_unit="patch",
            mask_ratio=0.4,
            span_length_min=1,
            span_length_max=2,
            num_spans_mode="one",
            allow_bin_fractional_overlap=True,
        )
        expected = masked["token_feature_mask"] & masked["token_mask"].unsqueeze(-1)
        self.assertTrue(torch.equal(masked["token_loss_mask"], expected))
        self.assertFalse(masked["token_loss_mask"][~masked["token_feature_mask"]].any())

    def test_masked_reconstruction_summary_includes_prediction_and_target_stats(self) -> None:
        random.seed(3)
        model = _make_model()
        batch = _make_batch()
        metrics = compute_masked_reconstruction_metrics(
            model,
            batch,
            mask_unit="patch",
            mask_token_placement="before_projection",
            mask_ratio=0.4,
            span_length_min=1,
            span_length_max=2,
            num_spans_mode="one",
            allow_bin_fractional_overlap=True,
            device=torch.device("cpu"),
        )
        summary = summarize_metrics(metrics)
        for key in (
            "masked_prediction_mean",
            "masked_prediction_std",
            "masked_target_mean",
            "masked_target_std",
        ):
            self.assertIn(key, summary)
        self.assertGreaterEqual(float(summary["masked_prediction_std"]), 0.0)
        self.assertGreaterEqual(float(summary["masked_target_std"]), 0.0)

    def test_bin_level_masking_generates_partial_patch_overlap(self) -> None:
        random.seed(2)
        model = _make_model(input_dim=3, patch_size=5, patch_stride=3)
        batch = {
            "x": torch.randn(1, 10, 3),
            "feature_mask": torch.ones(1, 3),
            "lengths": torch.tensor([10], dtype=torch.long),
        }
        masked = build_masked_batch(
            model,
            batch,
            mask_unit="bin",
            mask_ratio=0.2,
            span_length_min=2,
            span_length_max=2,
            num_spans_mode="one",
            allow_bin_fractional_overlap=True,
        )
        overlap = masked["token_overlap_fraction"][masked["token_mask"]]
        self.assertGreater(overlap.numel(), 0)
        self.assertTrue(torch.any((overlap > 0.0) & (overlap < 1.0)))

    def test_mask_token_placements_preserve_shapes_and_probe_recovery(self) -> None:
        model = _make_model()
        batch = _make_batch()
        tokens, token_lengths = model.encoder.patch_batch(batch["x"], batch["lengths"])
        token_mask = torch.zeros(tokens.shape[:2], dtype=torch.bool)
        token_mask[:, 0] = True

        before = model.reconstruct_from_patched_tokens(
            tokens,
            token_lengths,
            token_mask=token_mask,
            mask_token_placement="before_projection",
        )
        after = model.reconstruct_from_patched_tokens(
            tokens,
            token_lengths,
            token_mask=token_mask,
            mask_token_placement="after_projection",
        )
        self.assertEqual(tuple(before["reconstruction"].shape), tuple(tokens.shape))
        self.assertEqual(tuple(after["reconstruction"].shape), tuple(tokens.shape))

        tmp_path = Path(self._tmp_dir())
        checkpoint_path = tmp_path / "checkpoint_final.pt"
        config = _checkpoint_config(model)
        torch.save({"model_state": model.state_dict(), "config": config}, checkpoint_path)

        probe_state = recover_downstream_probe_state(
            probe_config=DownstreamProbeConfig(explicit_checkpoint_path=str(checkpoint_path)),
            output_root=tmp_path,
            input_dim=model.encoder.input_dim,
            default_checkpoint_config=config,
            current_run_dir=tmp_path,
        )
        outputs = probe_state["encoder"].encode(
            batch["x"],
            batch["lengths"],
            ["s0", "s1"],
            use_source_affines=False,
            target_affines=None,
        )
        self.assertEqual(outputs.hidden.shape[0], batch["x"].shape[0])
        self.assertEqual(outputs.hidden.shape[2], model.encoder.hidden_size)

    def test_encoder_prefix_hidden_and_reconstruction_do_not_depend_on_future_inputs(self) -> None:
        model = _make_model(input_dim=3, patch_size=1, patch_stride=1)
        model.eval()

        prefix_length = 4
        x_a = torch.randn(1, 6, 3)
        x_b = x_a.clone()
        x_b[:, prefix_length:, :] += 10.0
        lengths = torch.tensor([6], dtype=torch.long)

        encoded_a = model.encode_sequence(x_a, lengths)
        encoded_b = model.encode_sequence(x_b, lengths)
        self.assertTrue(
            torch.allclose(
                encoded_a["hidden"][:, :prefix_length],
                encoded_b["hidden"][:, :prefix_length],
                atol=1e-6,
                rtol=1e-6,
            )
        )

        recon_a = model.reconstruct_from_patched_tokens(
            encoded_a["tokens"],
            encoded_a["token_lengths"],
            token_mask=None,
            mask_token_placement="before_projection",
        )
        recon_b = model.reconstruct_from_patched_tokens(
            encoded_b["tokens"],
            encoded_b["token_lengths"],
            token_mask=None,
            mask_token_placement="before_projection",
        )
        self.assertTrue(
            torch.allclose(
                recon_a["reconstruction"][:, :prefix_length],
                recon_b["reconstruction"][:, :prefix_length],
                atol=1e-6,
                rtol=1e-6,
            )
        )

        self.assertFalse(
            torch.allclose(
                encoded_a["hidden"][:, prefix_length:],
                encoded_b["hidden"][:, prefix_length:],
                atol=1e-6,
                rtol=1e-6,
            )
        )

    def test_probe_recovery_auto_discovers_step_checkpoint(self) -> None:
        model = _make_model()
        tmp_path = Path(self._tmp_dir())
        run_dir = tmp_path / "colab_s5_masked_reconstruction_seg80_fake"
        checkpoints_dir = run_dir / "checkpoints"
        checkpoints_dir.mkdir(parents=True, exist_ok=True)
        config = _checkpoint_config(model)
        (run_dir / "config.json").write_text(json.dumps(config))
        checkpoint_path = checkpoints_dir / "step_000010_20260101T000000Z.pt"
        torch.save(
            {
                "model_state": model.state_dict(),
                "config": config,
                "step": 10,
                "best_score": 1.0,
                "best_step": 10,
                "checkpoint_kind": "step",
            },
            checkpoint_path,
        )

        probe_state = recover_downstream_probe_state(
            probe_config=DownstreamProbeConfig(),
            output_root=tmp_path,
            input_dim=model.encoder.input_dim,
            default_checkpoint_config=config,
        )
        self.assertEqual(Path(probe_state["checkpoint_path"]), checkpoint_path)

    def test_recover_ssl_run_state_from_checkpoint_with_stub_sampler(self) -> None:
        model = _make_model()
        config = _checkpoint_config(model)
        tmp_path = Path(self._tmp_dir())
        checkpoint_path = tmp_path / "checkpoint_final.pt"
        (tmp_path / "config.json").write_text(json.dumps(config))
        torch.save(
            {
                "model_state": model.state_dict(),
                "config": config,
                "step": 3,
                "best_score": 1.23,
                "best_step": 2,
                "checkpoint_kind": "final",
                "train_history": [],
                "val_history": [],
            },
            checkpoint_path,
        )

        def _fake_sampler(_cache_context, split_name, **kwargs):
            return {"split": split_name, **kwargs}

        with mock.patch("masked_ssl.training.build_segment_sampler", new=_fake_sampler):
            state = recover_ssl_run_state_from_checkpoint(
                checkpoint_path=checkpoint_path,
                cache_context=SimpleNamespace(full_dim=model.encoder.input_dim),
                device=torch.device("cpu"),
            )
        self.assertEqual(state["checkpoint_step"], 3)
        self.assertEqual(state["train_sampler"]["split"], "train")
        self.assertEqual(state["val_sampler"]["split"], "val")
        self.assertEqual(state["model"].encoder.input_dim, model.encoder.input_dim)

    def _tmp_dir(self) -> str:
        import tempfile

        tmp_dir = tempfile.mkdtemp(prefix="masked_ssl_test_")
        self.addCleanup(lambda: __import__("shutil").rmtree(tmp_dir, ignore_errors=True))
        return tmp_dir


if __name__ == "__main__":
    unittest.main()

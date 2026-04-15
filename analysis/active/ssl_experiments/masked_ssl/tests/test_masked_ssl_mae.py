from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from masked_ssl.model_mae import MaskedSSLModel
from masked_ssl.objectives_mae import compute_masked_reconstruction_metrics
from masked_ssl.phoneme_finetune import _recover_stage1_encoder
from masked_ssl.probe import _recover_encoder_from_notebook_checkpoint
from masked_ssl.training_mae import SSLTrainingConfig


def _make_model() -> MaskedSSLModel:
    return MaskedSSLModel(
        input_dim=4,
        hidden_size=16,
        s5_state_size=8,
        num_layers=1,
        dropout=0.0,
        patch_size=2,
        patch_stride=1,
        post_proj_norm="rms",
        source_session_keys=("s0", "s1"),
        feature_mode="tx_only",
        reconstruction_head_mode="no_output_norm",
        reconstruction_head_type="mlp",
        backbone_direction="bidirectional",
        max_patches=8,
        decoder_hidden_size=16,
        decoder_s5_state_size=8,
        decoder_num_layers=1,
        decoder_dropout=0.0,
        decoder_backbone_direction="bidirectional",
    )


class MaskedSSLMAETests(unittest.TestCase):
    def test_mae_encoder_uses_visible_tokens_only(self) -> None:
        model = _make_model()
        x = torch.randn(2, 6, 4)
        lengths = torch.tensor([6, 5], dtype=torch.long)
        tokens, token_lengths = model.encoder.patch_batch(x, lengths)
        token_mask = torch.zeros(tokens.shape[:2], dtype=torch.bool)
        token_mask[:, :2] = True

        outputs = model.encoder.encode_patched(
            tokens,
            token_lengths,
            token_mask=token_mask,
            mask_token_placement="visible_only",
            session_keys=("s0", "s1"),
        )
        expected_visible = (~token_mask & (torch.arange(tokens.shape[1]).unsqueeze(0) < token_lengths.unsqueeze(1))).sum(dim=1)
        self.assertTrue(torch.equal(outputs["visible_lengths"], expected_visible.to(outputs["visible_lengths"].dtype)))

    def test_mae_reconstruction_preserves_token_shape(self) -> None:
        model = _make_model()
        x = torch.randn(2, 6, 4)
        lengths = torch.tensor([6, 5], dtype=torch.long)
        tokens, token_lengths = model.encoder.patch_batch(x, lengths)
        token_mask = torch.zeros(tokens.shape[:2], dtype=torch.bool)
        token_mask[:, 1] = True
        outputs = model.reconstruct_from_patched_tokens(
            tokens,
            token_lengths,
            token_mask=token_mask,
            session_keys=("s0", "s1"),
        )
        self.assertEqual(tuple(outputs["reconstruction"].shape), tuple(tokens.shape))

    def test_mae_training_config_requires_visible_only_mask_mode(self) -> None:
        with self.assertRaises(ValueError):
            SSLTrainingConfig(mask_token_placement="before_projection")

    def test_mae_checkpoint_loads_in_probe_recovery(self) -> None:
        model = _make_model()
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "checkpoint.pt"
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": {
                        "objective_mode": "masked_reconstruction_mae",
                        "input_dim": 4,
                        "patch_size": 2,
                        "patch_stride": 1,
                        "hidden_size": 16,
                        "s5_state_size": 8,
                        "num_layers": 1,
                        "dropout": 0.0,
                        "post_proj_norm": "rms",
                        "source_session_keys": ["s0", "s1"],
                        "feature_mode": "tx_only",
                        "backbone_direction": "bidirectional",
                        "max_patches": 8,
                    },
                },
                checkpoint_path,
            )
            recovered_encoder, _ = _recover_encoder_from_notebook_checkpoint(
                path=checkpoint_path,
                input_dim=4,
            )
        self.assertTrue(hasattr(recovered_encoder, "encoder_pos_embed"))

    def test_mae_checkpoint_loads_in_phoneme_recovery(self) -> None:
        model = _make_model()
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "checkpoint.pt"
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": {
                        "objective_mode": "masked_reconstruction_mae",
                        "input_dim": 4,
                        "patch_size": 2,
                        "patch_stride": 1,
                        "hidden_size": 16,
                        "s5_state_size": 8,
                        "num_layers": 1,
                        "dropout": 0.0,
                        "post_proj_norm": "rms",
                        "source_session_keys": ["s0", "s1"],
                        "feature_mode": "tx_only",
                        "backbone_direction": "bidirectional",
                        "max_patches": 8,
                    },
                },
                checkpoint_path,
            )
            recovered_encoder, _, _ = _recover_stage1_encoder(
                checkpoint_path=checkpoint_path,
            )
        self.assertTrue(hasattr(recovered_encoder, "encoder_pos_embed"))

    def test_actual_mask_ratio_reflects_effective_post_safeguard_ratio(self) -> None:
        model = _make_model()
        batch = {
            "x": torch.randn(1, 3, 4),
            "lengths": torch.tensor([3], dtype=torch.long),
            "feature_mask": torch.ones(1, 4, dtype=torch.bool),
            "session_keys": ["s0"],
        }
        metrics = compute_masked_reconstruction_metrics(
            model,
            batch,
            mask_unit="patch",
            mask_token_placement="visible_only",
            mask_ratio=1.0,
            span_length_min=1,
            span_length_max=2,
            num_spans_mode="multiple",
            allow_bin_fractional_overlap=True,
            device=torch.device("cpu"),
        )
        self.assertLess(float(metrics["actual_mask_ratio"]), float(metrics["requested_mask_ratio"]))


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import json
import random
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[5]
EXPERIMENTS_DIR = REPO_ROOT / "analysis" / "active" / "ssl_experiments"
for path in (REPO_ROOT, EXPERIMENTS_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from masked_ssl.cache import (
    _apply_gaussian_smoothing,
    _compute_session_feature_stats,
    sample_base_segment,
)
from masked_ssl.model import MaskedSSLModel, SessionLinearBank
from masked_ssl.objectives import (
    build_masked_batch,
    compute_masked_reconstruction_metrics,
    sample_mask_indices,
    summarize_metrics,
)
from masked_ssl.phoneme_finetune import (
    PhonemeFinetuneConfig,
    RawFeatureAdapter,
    run_phoneme_finetuning,
)
from masked_ssl.probe import (
    CanonicalProbeManifestRow,
    CanonicalSequenceDataset,
    DownstreamProbeConfig,
    NotebookProbeEncoderAdapter,
    recover_downstream_probe_state,
)
from masked_ssl.training import recover_ssl_run_state_from_checkpoint
from s5 import BidirectionalS5SequenceBackbone, S5SequenceBackbone, reverse_padded_sequence


def _make_model(
    *,
    input_dim: int = 4,
    patch_size: int = 2,
    patch_stride: int = 2,
    hidden_size: int = 8,
    source_session_keys: tuple[str, ...] = ("s0", "s1"),
    feature_mode: str = "tx_only",
    reconstruction_head_type: str = "linear",
    backbone_direction: str = "causal",
) -> MaskedSSLModel:
    return MaskedSSLModel(
        input_dim=input_dim,
        hidden_size=hidden_size,
        s5_state_size=4,
        num_layers=1,
        dropout=0.0,
        patch_size=patch_size,
        patch_stride=patch_stride,
        post_proj_norm="rms",
        source_session_keys=source_session_keys,
        feature_mode=feature_mode,
        reconstruction_head_type=reconstruction_head_type,
        backbone_direction=backbone_direction,
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
        "feature_mode": model.feature_mode,
        "boundary_key_mode": "session",
        "input_dim": model.encoder.input_dim,
        "source_session_keys": list(model.source_session_keys),
        "segment_bins": 10,
        "patch_size": model.encoder.patch_size,
        "patch_stride": model.encoder.patch_stride,
        "hidden_size": model.encoder.hidden_size,
        "s5_state_size": model.encoder.s5_state_size,
        "num_layers": model.encoder.num_layers,
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
        "reconstruction_head_type": model.reconstruction_head_type,
        "backbone_direction": model.encoder.backbone_direction,
    }


def _write_tiny_canonical_probe_cache(cache_root: Path) -> None:
    dataset_root = cache_root / "brain2text25"
    shard_dir = dataset_root / "toy_shard"
    shard_dir.mkdir(parents=True, exist_ok=True)

    tx = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 2.0, 0.0],
            [0.0, 2.0, 1.0],
            [2.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [2.0, 1.0, 0.0],
            [0.0, 2.0, 2.0],
            [2.0, 2.0, 0.0],
            [1.0, 0.0, 2.0],
            [0.0, 1.0, 2.0],
            [2.0, 1.0, 1.0],
        ],
        dtype=np.float32,
    )
    sbp = np.array(
        [
            [10.0, 0.0],
            [10.0, 1.0],
            [11.0, 0.0],
            [11.0, 1.0],
            [12.0, 0.0],
            [12.0, 1.0],
            [13.0, 0.0],
            [13.0, 1.0],
            [14.0, 0.0],
            [14.0, 1.0],
            [15.0, 0.0],
            [15.0, 1.0],
            [16.0, 0.0],
            [16.0, 1.0],
            [17.0, 0.0],
            [17.0, 1.0],
        ],
        dtype=np.float32,
    )
    np.save(shard_dir / "time_offsets.npy", np.array([0, 4, 8, 12, 16], dtype=np.int64))
    np.save(shard_dir / "tx.npy", tx)
    np.save(shard_dir / "sbp.npy", sbp)
    np.save(shard_dir / "phoneme_offsets.npy", np.array([0, 2, 4, 6, 8], dtype=np.int64))
    np.save(shard_dir / "phoneme_ids.npy", np.array([1, 2, 2, 1, 1, 1, 2, 2], dtype=np.int64))

    manifest_rows = [
        {
            "example_id": "src-train-0",
            "session_id": "t00.2025.01.01",
            "subject_id": "t00",
            "session_date": "2025.01.01",
            "source_split": "train",
            "has_labels": True,
            "shard_relpath": "brain2text25/toy_shard",
            "example_index": 0,
            "n_tx_features": 3,
            "n_sbp_features": 2,
            "target_length": 2,
            "transcript": "AA",
            "has_tx": True,
            "has_sbp": True,
        },
        {
            "example_id": "src-train-1",
            "session_id": "t00.2025.01.01",
            "subject_id": "t00",
            "session_date": "2025.01.01",
            "source_split": "train",
            "has_labels": True,
            "shard_relpath": "brain2text25/toy_shard",
            "example_index": 1,
            "n_tx_features": 3,
            "n_sbp_features": 2,
            "target_length": 2,
            "transcript": "BB",
            "has_tx": True,
            "has_sbp": True,
        },
        {
            "example_id": "target-train-0",
            "session_id": "t00.2025.01.02",
            "subject_id": "t00",
            "session_date": "2025.01.02",
            "source_split": "train",
            "has_labels": True,
            "shard_relpath": "brain2text25/toy_shard",
            "example_index": 2,
            "n_tx_features": 3,
            "n_sbp_features": 2,
            "target_length": 2,
            "transcript": "AB",
            "has_tx": True,
            "has_sbp": True,
        },
        {
            "example_id": "target-val-0",
            "session_id": "t00.2025.01.02",
            "subject_id": "t00",
            "session_date": "2025.01.02",
            "source_split": "val",
            "has_labels": True,
            "shard_relpath": "brain2text25/toy_shard",
            "example_index": 3,
            "n_tx_features": 3,
            "n_sbp_features": 2,
            "target_length": 2,
            "transcript": "BA",
            "has_tx": True,
            "has_sbp": True,
        },
    ]
    with (dataset_root / "manifest.jsonl").open("w") as handle:
        for row in manifest_rows:
            handle.write(json.dumps(row) + "\n")

    metadata = {
        "n_tx_features": 3,
        "n_sbp_features": 2,
        "phoneme_vocabulary": {
            "num_classes": 3,
            "blank_index": 0,
            "index_to_symbol": ["<blk>", "AA", "BB"],
        },
    }
    (dataset_root / "metadata.json").write_text(json.dumps(metadata))


class MaskedSSLTests(unittest.TestCase):
    def test_reverse_padded_sequence_reverses_only_valid_prefix(self) -> None:
        x = torch.tensor(
            [
                [[1.0], [2.0], [3.0], [0.0], [0.0]],
                [[4.0], [5.0], [6.0], [7.0], [8.0]],
            ]
        )
        lengths = torch.tensor([3, 5], dtype=torch.long)
        reversed_x = reverse_padded_sequence(x, lengths)
        expected = torch.tensor(
            [
                [[3.0], [2.0], [1.0], [0.0], [0.0]],
                [[8.0], [7.0], [6.0], [5.0], [4.0]],
            ]
        )
        self.assertTrue(torch.equal(reversed_x, expected))
        self.assertTrue(torch.equal(reverse_padded_sequence(reversed_x, lengths), x))

    def test_bidirectional_backbone_matches_causal_shape_and_masks_padding(self) -> None:
        x = torch.randn(2, 5, 4)
        lengths = torch.tensor([3, 5], dtype=torch.long)
        causal = S5SequenceBackbone(d_model=4, d_state=3, num_layers=2, dropout=0.0)
        bidirectional = BidirectionalS5SequenceBackbone(
            d_model=4,
            d_state=3,
            num_layers=2,
            dropout=0.0,
        )
        causal_out = causal(x, lengths)
        bidirectional_out = bidirectional(x, lengths)
        self.assertEqual(tuple(causal_out.shape), tuple(bidirectional_out.shape))
        self.assertTrue(torch.allclose(bidirectional_out[0, 3:], torch.zeros_like(bidirectional_out[0, 3:])))

    def test_tx_only_sampling_returns_only_tx_channels_and_featurewise_normalization(self) -> None:
        session_key = "toy:sess0"

        class _DummyShardStore:
            def get(self, _shard_relpath):
                tx = np.arange(24, dtype=np.float32).reshape(6, 4)
                sbp = (100.0 + np.arange(12, dtype=np.float32)).reshape(6, 2)
                return {
                    "time_offsets": np.array([0, 6], dtype=np.int64),
                    "tx": tx,
                    "sbp": sbp,
                }

        cache_context = SimpleNamespace(
            full_dim=4,
            tx_dim=4,
            sbp_dim=2,
            feature_mode="tx_only",
            boundary_key_mode="subject_if_available",
            shard_store=_DummyShardStore(),
            normalize_context_bins=2,
            normalize_impl_version="session_featurewise_v1",
            session_feature_stats={
                session_key: (
                    torch.tensor([1.0, 2.0, 3.0, 4.0]),
                    torch.tensor([2.0, 2.0, 2.0, 2.0]),
                )
            },
        )
        example = SimpleNamespace(
            dataset="toy",
            session_id="sess0",
            subject_id="subj0",
            shard_relpath="unused",
            example_index=0,
            has_tx=True,
            has_sbp=True,
        )
        sample = sample_base_segment(cache_context, example, segment_bins=6, py_rng=random.Random(0))
        self.assertEqual(tuple(sample["x"].shape), (6, 4))
        self.assertTrue(torch.equal(sample["feature_mask"], torch.ones(4)))
        self.assertEqual(sample["boundary_key"], "toy:subj0")
        expected = (
            torch.tensor(np.arange(24, dtype=np.float32).reshape(6, 4))
            - torch.tensor([1.0, 2.0, 3.0, 4.0])
        ) / 2.0
        self.assertTrue(torch.allclose(sample["x"], expected))

    def test_sampling_with_smoothing_matches_full_row_smooth_then_crop(self) -> None:
        session_key = "toy:sess0"
        tx_series = np.arange(10, dtype=np.float32).reshape(10, 1)

        class _DummyShardStore:
            def get(self, _shard_relpath):
                return {
                    "time_offsets": np.array([0, 10], dtype=np.int64),
                    "tx": tx_series,
                    "sbp": None,
                }

        cache_context = SimpleNamespace(
            full_dim=1,
            tx_dim=1,
            sbp_dim=1,
            feature_mode="tx_only",
            boundary_key_mode="subject_if_available",
            shard_store=_DummyShardStore(),
            normalize_context_bins=2,
            normalize_impl_version="session_featurewise_v1",
            gaussian_smoothing_sigma_bins=1.0,
            session_feature_stats={
                session_key: (
                    torch.zeros(1, dtype=torch.float32),
                    torch.ones(1, dtype=torch.float32),
                )
            },
        )
        example = SimpleNamespace(
            dataset="toy",
            session_id="sess0",
            subject_id="subj0",
            shard_relpath="unused",
            example_index=0,
            has_tx=True,
            has_sbp=False,
        )

        segment_bins = 4
        rng_seed = 9
        expected_offset = random.Random(rng_seed).randrange(10 - segment_bins + 1)
        expected_smoothed = _apply_gaussian_smoothing(
            torch.from_numpy(tx_series.copy()),
            torch.ones(1, dtype=torch.float32),
            sigma_bins=1.0,
        )[expected_offset : expected_offset + segment_bins]

        sample = sample_base_segment(
            cache_context,
            example,
            segment_bins=segment_bins,
            py_rng=random.Random(rng_seed),
        )
        self.assertEqual(tuple(sample["x"].shape), (segment_bins, 1))
        self.assertTrue(torch.allclose(sample["x"], expected_smoothed, atol=1e-5, rtol=1e-5))

    def test_session_feature_stats_bin_stride_uses_subsampled_time_bins(self) -> None:
        class _DummyShardStore:
            def get(self, _shard_relpath):
                return {
                    "time_offsets": np.array([0, 10], dtype=np.int64),
                    "tx": np.arange(10, dtype=np.float32).reshape(10, 1),
                    "sbp": None,
                }

        row = SimpleNamespace(
            dataset="toy",
            session_id="sess0",
            subject_id="subj0",
            shard_relpath="toy/shard0",
            example_index=0,
            n_time_bins=10,
            has_tx=True,
            has_sbp=False,
            n_tx_features=1,
            n_sbp_features=0,
        )
        config = SimpleNamespace(
            full_dim=1,
            tx_dim=1,
            sbp_dim=1,
            feature_mode="tx_only",
            gaussian_smoothing_sigma_bins=0.0,
            session_stats_bin_stride=2,
        )

        stats = _compute_session_feature_stats(
            _DummyShardStore(),
            rows_by_dataset={"toy": [row]},
            config=config,
        )
        mean, std = stats["toy:sess0"]
        self.assertAlmostEqual(float(mean[0]), 4.0, places=6)
        self.assertAlmostEqual(float(std[0]), float(np.sqrt(8.0)), places=6)

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

    def test_patch_level_multi_patch_span_only_scores_first_masked_token(self) -> None:
        random.seed(4)
        model = _make_model(input_dim=3, patch_size=1, patch_stride=1, backbone_direction="causal")
        batch = {
            "x": torch.randn(1, 6, 3),
            "feature_mask": torch.ones(1, 3),
            "lengths": torch.tensor([6], dtype=torch.long),
        }
        masked = build_masked_batch(
            model,
            batch,
            mask_unit="patch",
            mask_ratio=2.0 / 6.0,
            span_length_min=2,
            span_length_max=2,
            num_spans_mode="one",
            allow_bin_fractional_overlap=True,
        )
        token_mask = masked["token_mask"][0]
        loss_token_mask = masked["token_loss_token_mask"][0]
        masked_idx = torch.nonzero(token_mask, as_tuple=False).squeeze(1)
        loss_idx = torch.nonzero(loss_token_mask, as_tuple=False).squeeze(1)
        self.assertEqual(masked_idx.numel(), 2)
        self.assertTrue(torch.equal(masked_idx, torch.arange(int(masked_idx[0]), int(masked_idx[0]) + 2)))
        self.assertEqual(loss_idx.numel(), 1)
        self.assertEqual(int(loss_idx[0]), int(masked_idx[0]))
        expected_loss_mask = masked["token_feature_mask"] & loss_token_mask.view(1, -1, 1)
        self.assertTrue(torch.equal(masked["token_loss_mask"], expected_loss_mask))

    def test_patch_level_multi_patch_span_scores_all_masked_tokens_when_bidirectional(self) -> None:
        random.seed(4)
        model = _make_model(
            input_dim=3,
            patch_size=1,
            patch_stride=1,
            backbone_direction="bidirectional",
        )
        batch = {
            "x": torch.randn(1, 6, 3),
            "feature_mask": torch.ones(1, 3),
            "lengths": torch.tensor([6], dtype=torch.long),
        }
        masked = build_masked_batch(
            model,
            batch,
            mask_unit="patch",
            mask_ratio=2.0 / 6.0,
            span_length_min=2,
            span_length_max=2,
            num_spans_mode="one",
            allow_bin_fractional_overlap=True,
        )
        token_mask = masked["token_mask"][0]
        loss_token_mask = masked["token_loss_token_mask"][0]
        self.assertTrue(torch.equal(token_mask, loss_token_mask))
        expected_loss_mask = masked["token_feature_mask"] & token_mask.view(1, -1, 1)
        self.assertTrue(torch.equal(masked["token_loss_mask"], expected_loss_mask))

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

    def test_mlp_reconstruction_head_preserves_reconstruction_shape(self) -> None:
        model = _make_model(reconstruction_head_type="mlp")
        batch = _make_batch()
        outputs = model.reconstruct_from_patched_tokens(
            *model.encoder.patch_batch(batch["x"], batch["lengths"]),
            token_mask=None,
            mask_token_placement="before_projection",
        )
        self.assertEqual(tuple(outputs["reconstruction"].shape[0:2]), tuple(outputs["tokens"].shape[0:2]))
        self.assertEqual(int(outputs["reconstruction"].shape[-1]), model.encoder.token_dim)

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
        skip = model.reconstruct_from_patched_tokens(
            tokens,
            token_lengths,
            token_mask=token_mask,
            mask_token_placement="skip",
        )
        self.assertEqual(tuple(before["reconstruction"].shape), tuple(tokens.shape))
        self.assertEqual(tuple(after["reconstruction"].shape), tuple(tokens.shape))
        self.assertEqual(tuple(skip["reconstruction"].shape), tuple(tokens.shape))

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
        model = _make_model(input_dim=3, patch_size=1, patch_stride=1, backbone_direction="causal")
        model.eval()

        prefix_length = 4
        x_a = torch.randn(1, 6, 3)
        x_b = x_a.clone()
        x_b[:, prefix_length:, 0] *= -3.0
        x_b[:, prefix_length:, 1] += 7.5
        x_b[:, prefix_length:, 2] -= 4.25
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

    def test_bidirectional_encoder_hidden_and_reconstruction_can_depend_on_future_inputs(self) -> None:
        model = _make_model(
            input_dim=3,
            patch_size=1,
            patch_stride=1,
            backbone_direction="bidirectional",
        )
        model.eval()

        prefix_length = 4
        x_a = torch.randn(1, 6, 3)
        x_b = x_a.clone()
        x_b[:, prefix_length:, 0] *= -3.0
        x_b[:, prefix_length:, 1] += 7.5
        x_b[:, prefix_length:, 2] -= 4.25
        lengths = torch.tensor([6], dtype=torch.long)

        encoded_a = model.encode_sequence(x_a, lengths)
        encoded_b = model.encode_sequence(x_b, lengths)
        self.assertFalse(
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
        self.assertFalse(
            torch.allclose(
                recon_a["reconstruction"][:, :prefix_length],
                recon_b["reconstruction"][:, :prefix_length],
                atol=1e-6,
                rtol=1e-6,
            )
        )

    def test_session_keyed_readin_and_readout_route_by_session(self) -> None:
        model = _make_model(input_dim=2, patch_size=1, patch_stride=1, hidden_size=2)
        with torch.no_grad():
            for layer in (model.encoder.source_readin.layers["s0"], model.source_readout.layers["s0"]):
                layer.weight.zero_()
                layer.weight += 2.0 * torch.eye(2)
                layer.bias.zero_()
            for layer in (model.encoder.source_readin.layers["s1"], model.source_readout.layers["s1"]):
                layer.weight.zero_()
                layer.weight += 3.0 * torch.eye(2)
                layer.bias.zero_()

        tokens = torch.ones(2, 3, 2)
        token_lengths = torch.tensor([3, 3], dtype=torch.long)
        encoded = model.encoder.encode_patched(
            tokens,
            token_lengths,
            session_keys=["s0", "s1"],
            use_source_affines=True,
        )
        self.assertTrue(torch.allclose(encoded["aligned_tokens"][0], torch.full((3, 2), 2.0)))
        self.assertTrue(torch.allclose(encoded["aligned_tokens"][1], torch.full((3, 2), 3.0)))

        readout = model.source_readout(tokens, ["s0", "s1"])
        self.assertTrue(torch.allclose(readout[0], torch.full((3, 2), 2.0)))
        self.assertTrue(torch.allclose(readout[1], torch.full((3, 2), 3.0)))

    def test_probe_target_session_bank_gets_gradients_with_frozen_and_unfrozen_encoder(self) -> None:
        model = _make_model(input_dim=3, patch_size=1, patch_stride=1, hidden_size=3)
        x = torch.randn(1, 4, 3)
        lengths = torch.tensor([4], dtype=torch.long)
        session_ids = ["target"]

        adapter_frozen = NotebookProbeEncoderAdapter(model.encoder)
        for parameter in adapter_frozen.parameters():
            parameter.requires_grad = False
        frozen_target_affines = SessionLinearBank(("target",), adapter_frozen.token_dim)
        frozen_outputs = adapter_frozen.encode(
            x,
            lengths,
            session_ids,
            use_source_affines=False,
            target_affines=frozen_target_affines,
        )
        frozen_loss = frozen_outputs.hidden.sum()
        frozen_loss.backward()
        self.assertTrue(
            any(param.grad is not None for param in frozen_target_affines.parameters())
        )
        self.assertTrue(
            all(param.grad is None for param in adapter_frozen.parameters())
        )

        adapter_trainable = NotebookProbeEncoderAdapter(model.encoder)
        for parameter in adapter_trainable.parameters():
            parameter.requires_grad = True
        trainable_target_affines = SessionLinearBank(("target",), adapter_trainable.token_dim)
        trainable_outputs = adapter_trainable.encode(
            x,
            lengths,
            session_ids,
            use_source_affines=False,
            target_affines=trainable_target_affines,
        )
        trainable_loss = trainable_outputs.hidden.sum()
        trainable_loss.backward()
        self.assertTrue(
            any(param.grad is not None for param in trainable_target_affines.parameters())
        )
        self.assertTrue(
            any(param.grad is not None for param in adapter_trainable.parameters())
        )

    def test_probe_dataset_tx_only_slicing_matches_encoder_width(self) -> None:
        tmp_dir = Path(self._tmp_dir())
        shard_dir = tmp_dir / "toy_shard"
        shard_dir.mkdir(parents=True, exist_ok=True)
        np.save(shard_dir / "time_offsets.npy", np.array([0, 3], dtype=np.int64))
        np.save(shard_dir / "tx.npy", np.arange(9, dtype=np.float32).reshape(3, 3))
        np.save(shard_dir / "sbp.npy", (100.0 + np.arange(6, dtype=np.float32)).reshape(3, 2))
        np.save(shard_dir / "phoneme_offsets.npy", np.array([0, 0], dtype=np.int64))

        row = CanonicalProbeManifestRow(
            example_id="ex0",
            session_id="sess0",
            subject_id="subj0",
            source_split="train",
            has_labels=False,
            shard_relpath="toy_shard",
            example_index=0,
            n_tx_features=3,
            n_sbp_features=2,
            target_length=None,
            transcript="",
        )
        dataset = CanonicalSequenceDataset(
            [row],
            cache_root=tmp_dir,
            stats=None,
            feature_mode="tx_only",
            boundary_key_mode="subject_if_available",
        )
        item = dataset[0]
        self.assertEqual(int(item["x"].shape[1]), 3)
        self.assertEqual(item["boundary_key"], "brain2text25:subj0")

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
                cache_context=SimpleNamespace(
                    full_dim=model.encoder.input_dim,
                    feature_mode=model.feature_mode,
                ),
                device=torch.device("cpu"),
            )
        self.assertEqual(state["checkpoint_step"], 3)
        self.assertEqual(state["train_sampler"]["split"], "train")
        self.assertEqual(state["val_sampler"]["split"], "val")
        self.assertEqual(state["model"].encoder.input_dim, model.encoder.input_dim)
        self.assertEqual(state["model"].feature_mode, model.feature_mode)
        self.assertEqual(state["model"].encoder.backbone_direction, "causal")

    def test_recover_ssl_run_state_from_checkpoint_preserves_explicit_bidirectional_direction(self) -> None:
        model = _make_model(backbone_direction="bidirectional")
        config = _checkpoint_config(model)
        tmp_path = Path(self._tmp_dir())
        checkpoint_path = tmp_path / "checkpoint_final.pt"
        (tmp_path / "config.json").write_text(json.dumps(config))
        torch.save(
            {
                "model_state": model.state_dict(),
                "config": config,
                "step": 1,
                "best_score": 1.0,
                "best_step": 1,
                "checkpoint_kind": "final",
            },
            checkpoint_path,
        )

        def _fake_sampler(_cache_context, split_name, **kwargs):
            return {"split": split_name, **kwargs}

        with mock.patch("masked_ssl.training.build_segment_sampler", new=_fake_sampler):
            state = recover_ssl_run_state_from_checkpoint(
                checkpoint_path=checkpoint_path,
                cache_context=SimpleNamespace(
                    full_dim=model.encoder.input_dim,
                    feature_mode=model.feature_mode,
                ),
                device=torch.device("cpu"),
            )
        self.assertEqual(state["model"].encoder.backbone_direction, "bidirectional")

    def test_raw_feature_adapter_starts_as_identity_on_tx_block(self) -> None:
        adapter = RawFeatureAdapter(input_dim=5, output_dim=3)
        x = torch.tensor(
            [[[1.0, 2.0, 3.0, 100.0, 200.0]]],
            dtype=torch.float32,
        )
        y = adapter(x)
        self.assertTrue(torch.allclose(y, torch.tensor([[[1.0, 2.0, 3.0]]])))

    def test_run_phoneme_finetuning_with_tx_sbp_adapter_writes_checkpoint(self) -> None:
        model = _make_model(input_dim=3, patch_size=1, patch_stride=1, hidden_size=4)
        config = _checkpoint_config(model)
        tmp_path = Path(self._tmp_dir())
        checkpoint_path = tmp_path / "checkpoint_final.pt"
        torch.save({"model_state": model.state_dict(), "config": config}, checkpoint_path)
        _write_tiny_canonical_probe_cache(tmp_path)

        summary = run_phoneme_finetuning(
            checkpoint_path=checkpoint_path,
            cache_root=tmp_path,
            config=PhonemeFinetuneConfig(
                seed=7,
                mode="probe_frozen",
                feature_mode="tx_sbp",
                session_limit=2,
                target_session_count=1,
                batch_size=1,
                num_steps=2,
                budget_seconds=30,
                learning_rate=1e-3,
                encoder_learning_rate=3e-4,
                checkpoint_every_steps=1,
            ),
            device=torch.device("cpu"),
        )
        self.assertEqual(summary["feature_mode"], "tx_sbp")
        self.assertEqual(summary["adapter_type"], "RawFeatureAdapter")
        self.assertEqual(summary["external_input_dim"], 5)
        self.assertEqual(summary["encoder_input_dim"], 3)
        self.assertTrue(Path(summary["checkpoint_final_path"]).exists())
        self.assertTrue((Path(summary["checkpoints_dir"]) / "step_000001.pt").exists())
        self.assertTrue((Path(summary["checkpoints_dir"]) / "step_000002.pt").exists())
        self.assertEqual(summary["checkpoint_every_steps"], 1)

        payload = torch.load(summary["checkpoint_final_path"], map_location="cpu")
        self.assertEqual(payload["feature_mode"], "tx_sbp")
        self.assertEqual(payload["external_input_dim"], 5)
        self.assertEqual(payload["encoder_input_dim"], 3)

    def test_run_phoneme_finetuning_full_mode_writes_summary(self) -> None:
        model = _make_model(input_dim=3, patch_size=1, patch_stride=1, hidden_size=4)
        config = _checkpoint_config(model)
        tmp_path = Path(self._tmp_dir())
        checkpoint_path = tmp_path / "checkpoint_final.pt"
        torch.save({"model_state": model.state_dict(), "config": config}, checkpoint_path)
        _write_tiny_canonical_probe_cache(tmp_path)

        summary = run_phoneme_finetuning(
            checkpoint_path=checkpoint_path,
            cache_root=tmp_path,
            config=PhonemeFinetuneConfig(
                seed=7,
                mode="finetune_full",
                feature_mode="tx_only",
                session_limit=2,
                target_session_count=1,
                batch_size=1,
                num_steps=2,
                budget_seconds=30,
                learning_rate=1e-3,
                encoder_learning_rate=3e-4,
                checkpoint_every_steps=1,
            ),
            device=torch.device("cpu"),
        )
        self.assertEqual(summary["feature_mode"], "tx_only")
        self.assertEqual(summary["adapter_type"], "IdentityFeatureAdapter")
        self.assertIn("val_ctc_bpphone", summary["metrics"])
        self.assertIn("best_val_ctc_bpphone", summary["metrics"])
        self.assertIn("best_step", summary["metrics"])

    def _tmp_dir(self) -> str:
        import tempfile

        tmp_dir = tempfile.mkdtemp(prefix="masked_ssl_test_")
        self.addCleanup(lambda: __import__("shutil").rmtree(tmp_dir, ignore_errors=True))
        return tmp_dir


if __name__ == "__main__":
    unittest.main()

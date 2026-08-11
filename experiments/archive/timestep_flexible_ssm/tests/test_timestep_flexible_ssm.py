from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from utah_ssl.datasets import CanonicalProbeManifestRow
from experiments.archive.timestep_flexible_ssm.data import (
    RebinnedSequenceDataset,
    make_length_aware_batch_sampler,
    rebin_features,
    resolve_patch_bins,
)
from experiments.archive.timestep_flexible_ssm.experiment_utils import (
    carry_forward_missing_frames,
    interpolate_missing_frames,
    irregular_observation_view,
)
from experiments.archive.timestep_flexible_ssm.future_infonce import (
    FutureInfoNCEConfig,
    run_future_infonce,
)
from experiments.archive.timestep_flexible_ssm.model import TimestepFlexibleS5Model
from experiments.archive.timestep_flexible_ssm.supervised_experiments import (
    SupervisedExperimentConfig,
    run_mixed_bin_gru,
)
from experiments.archive.timestep_flexible_ssm.train import (
    TimestepFlexibleSSMConfig,
    run_timestep_flexible_reconstruction,
)


def _write_tiny_competition_probe_cache(cache_root: Path) -> None:
    dataset_dir = cache_root / "brain2text24"
    shards_dir = dataset_dir / "shards"
    shard_dir = shards_dir / "toy_shard"
    shard_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    examples = [
        ("train-0", "competition_train", "t12.2022.08.10", np.array([[1, 2, 3], [2, 4, 6], [3, 6, 9], [4, 8, 12]], dtype=np.float32), [1, 2]),
        ("train-1", "competition_train", "t12.2022.08.10", np.array([[6, 5, 4], [5, 4, 3], [4, 3, 2], [3, 2, 1]], dtype=np.float32), [2, 1]),
        ("train-2", "competition_train", "t12.2022.08.11", np.array([[1, 1, 2], [2, 2, 3], [3, 3, 4], [4, 4, 5]], dtype=np.float32), [1, 1]),
        ("train-3", "competition_train", "t12.2022.08.11", np.array([[5, 4, 3], [4, 3, 2], [3, 2, 1], [2, 1, 0]], dtype=np.float32), [2, 2]),
        ("test-0", "competition_test", "t12.2022.08.10", np.array([[1, 0, 1], [2, 0, 2], [3, 0, 3], [4, 0, 4]], dtype=np.float32), [1, 2]),
        ("test-1", "competition_test", "t12.2022.08.11", np.array([[0, 1, 1], [0, 2, 2], [0, 3, 3], [0, 4, 4]], dtype=np.float32), [2, 2]),
    ]
    tx_rows = []
    phoneme_ids = []
    time_offsets = [0]
    phoneme_offsets = [0]
    for example_index, (example_id, source_split, session_id, x, labels) in enumerate(examples):
        tx_rows.append(x)
        phoneme_ids.extend(labels)
        time_offsets.append(time_offsets[-1] + int(x.shape[0]))
        phoneme_offsets.append(phoneme_offsets[-1] + len(labels))
        manifest_rows.append(
            {
                "example_id": example_id,
                "session_id": session_id,
                "subject_id": "t12",
                "session_date": session_id.split(".", 1)[1],
                "source_split": source_split,
                "has_labels": True,
                "shard_relpath": "brain2text24/shards/toy_shard",
                "example_index": example_index,
                "block_num": 100 + example_index,
                "normalization_group": session_id,
                "n_tx_features": 3,
                "n_sbp_features": 0,
                "n_time_bins": int(x.shape[0]),
                "target_length": len(labels),
                "transcript": "AA",
                "has_tx": True,
                "has_sbp": False,
            }
        )
    np.save(shard_dir / "tx.npy", np.concatenate(tx_rows, axis=0).astype(np.float32))
    np.save(shard_dir / "time_offsets.npy", np.asarray(time_offsets, dtype=np.int64))
    np.save(shard_dir / "phoneme_offsets.npy", np.asarray(phoneme_offsets, dtype=np.int64))
    np.save(shard_dir / "phoneme_ids.npy", np.asarray(phoneme_ids, dtype=np.int64))
    (dataset_dir / "manifest.jsonl").write_text("".join(json.dumps(row) + "\n" for row in manifest_rows))
    metadata = {
        "dataset_family": "brain2text24",
        "phoneme_vocabulary": {
            "index_to_symbol": ["BLANK", "AA", "AE", "SIL"],
            "num_classes": 4,
            "blank_index": 0,
            "sil_index": 3,
        },
    }
    (dataset_dir / "metadata.json").write_text(json.dumps(metadata))


class TimestepFlexibleSSMTest(unittest.TestCase):
    def _tmp_dir(self) -> str:
        return tempfile.mkdtemp(prefix="timestep_flexible_ssm_test_")

    def test_rebin_features_uses_average(self) -> None:
        x = np.asarray(
            [
                [1.0, 10.0],
                [3.0, 14.0],
                [5.0, 18.0],
                [7.0, 22.0],
            ],
            dtype=np.float32,
        )
        rebinned = rebin_features(x, bin_size_ms=40)
        expected = np.asarray([[2.0, 12.0], [6.0, 20.0]], dtype=np.float32)
        self.assertTrue(np.allclose(rebinned, expected))

    def test_resolve_patch_bins_requires_exact_divisibility(self) -> None:
        self.assertEqual(resolve_patch_bins(280, bin_size_ms=20, field_name="patch_size_ms"), 14)
        self.assertEqual(resolve_patch_bins(280, bin_size_ms=40, field_name="patch_size_ms"), 7)
        with self.assertRaisesRegex(ValueError, "divisible"):
            resolve_patch_bins(280, bin_size_ms=60, field_name="patch_size_ms")

    def test_model_keeps_token_dim_stable_across_bin_sizes(self) -> None:
        model = TimestepFlexibleS5Model(
            input_dim=3,
            vocab_size=4,
            train_bin_size_ms=20,
            patch_size_ms=40,
            patch_stride_ms=40,
            input_projection_size=5,
            s5_hidden_size=8,
            s5_state_size=4,
            s5_num_layers=1,
            s5_dropout=0.0,
            s5_ffn_multiplier=1.0,
            session_adapter_enabled=False,
        )
        x20 = torch.randn(2, 4, 3)
        lengths20 = torch.tensor([4, 3], dtype=torch.long)
        out20 = model(x20, lengths20, active_bin_size_ms=20)
        x40 = torch.randn(2, 2, 3)
        lengths40 = torch.tensor([2, 2], dtype=torch.long)
        out40 = model(x40, lengths40, active_bin_size_ms=40)
        self.assertEqual(int(out20["patched_inputs"].shape[-1]), 10)
        self.assertEqual(int(out40["patched_inputs"].shape[-1]), 10)
        self.assertEqual(float(out20["dt_scale"]), 1.0)
        self.assertEqual(float(out40["dt_scale"]), 1.0)
        self.assertEqual(int(out20["active_patch_size_bins"]), 2)
        self.assertEqual(int(out40["active_patch_size_bins"]), 1)
        self.assertEqual(int(out20["active_patch_stride_bins"]), 2)
        self.assertEqual(int(out40["active_patch_stride_bins"]), 1)

    def test_length_aware_sampler_uses_rebinned_lengths(self) -> None:
        rows = [
            CanonicalProbeManifestRow(
                example_id="a",
                session_id="s1",
                subject_id="p1",
                source_split="competition_train",
                has_labels=True,
                shard_relpath="brain2text24/shards/toy",
                example_index=0,
                n_tx_features=3,
                n_sbp_features=0,
                target_length=2,
                transcript="AA",
                n_time_bins=4,
            ),
            CanonicalProbeManifestRow(
                example_id="b",
                session_id="s1",
                subject_id="p1",
                source_split="competition_train",
                has_labels=True,
                shard_relpath="brain2text24/shards/toy",
                example_index=1,
                n_tx_features=3,
                n_sbp_features=0,
                target_length=2,
                transcript="AA",
                n_time_bins=8,
            ),
        ]
        sampler = make_length_aware_batch_sampler(
            rows,
            batch_size=2,
            shuffle=False,
            seed=7,
            bin_size_ms=40,
        )
        self.assertEqual([row.n_time_bins for row in sampler.rows], [2, 4])

    def test_missing_bin_helpers(self) -> None:
        x = np.asarray(
            [
                [1.0, 10.0],
                [3.0, 14.0],
                [5.0, 18.0],
                [7.0, 22.0],
            ],
            dtype=np.float32,
        )
        keep = np.asarray([True, False, False, True], dtype=bool)
        interpolated = interpolate_missing_frames(x, keep)
        carried = carry_forward_missing_frames(x, keep)
        self.assertTrue(np.allclose(interpolated[:, 0], np.asarray([1.0, 3.0, 5.0, 7.0], dtype=np.float32)))
        self.assertTrue(np.allclose(carried[:, 0], np.asarray([1.0, 1.0, 1.0, 7.0], dtype=np.float32)))
        irregular_x, deltas_ms, keep_mask = irregular_observation_view(
            x,
            example_id="demo",
            seed=7,
            drop_probability=0.25,
        )
        self.assertGreaterEqual(int(irregular_x.shape[0]), 1)
        self.assertEqual(int(irregular_x.shape[0]), int(deltas_ms.shape[0]))
        self.assertEqual(int(keep_mask.shape[0]), int(x.shape[0]))

    def test_rebinned_dataset_and_short_training_run(self) -> None:
        cache_root = Path(self._tmp_dir())
        _write_tiny_competition_probe_cache(cache_root)
        dataset = RebinnedSequenceDataset(
            [],
            cache_root=cache_root,
            stats=None,
            feature_mode="tx_only",
            active_bin_size_ms=40,
        )
        del dataset
        output_root = Path(self._tmp_dir())
        config = TimestepFlexibleSSMConfig(
            cache_root=cache_root,
            output_root=output_root,
            run_name="tiny_timestep_run",
            normalization_mode="global",
            max_steps=1,
            batch_size=2,
            learning_rate=1e-3,
            min_learning_rate=1e-4,
            warmup_steps=0,
            checkpoint_every_steps=1,
            val_every_steps=1,
            progress_every_steps=1,
            input_smoothing_sigma_ms=0.0,
            white_noise_sd=0.0,
            constant_offset_sd=0.0,
            patch_size_ms=40,
            patch_stride_ms=40,
            input_projection_size=8,
            s5_hidden_size=16,
            s5_state_size=4,
            s5_num_layers=1,
            s5_dropout=0.0,
            s5_ffn_multiplier=1.0,
            eval_bin_sizes_ms=(20, 40),
        )
        summary = run_timestep_flexible_reconstruction(config)
        run_dir = output_root / "tiny_timestep_run"
        self.assertTrue((run_dir / "summary.json").exists())
        self.assertTrue((run_dir / "checkpoint_best.pt").exists())
        self.assertTrue((run_dir / "checkpoint_final.pt").exists())
        self.assertIn("val_20ms_ctc_bpphone", summary["metrics"])
        self.assertIn("val_40ms_ctc_bpphone", summary["metrics"])
        self.assertEqual(summary["train_bin_size_ms"], 20)
        self.assertEqual(summary["eval_bin_sizes_ms"], [20, 40])

    def test_mixed_gru_smoke_run(self) -> None:
        cache_root = Path(self._tmp_dir())
        _write_tiny_competition_probe_cache(cache_root)
        output_root = Path(self._tmp_dir())
        config = SupervisedExperimentConfig(
            cache_root=cache_root,
            output_root=output_root,
            run_name="tiny_mixed_gru",
            normalization_mode="global",
            max_steps=1,
            batch_size=2,
            learning_rate=1e-3,
            min_learning_rate=1e-4,
            warmup_steps=0,
            val_every_steps=1,
            progress_every_steps=1,
            input_smoothing_sigma_ms=0.0,
            white_noise_sd=0.0,
            constant_offset_sd=0.0,
            patch_size_ms=40,
            patch_stride_ms=40,
            input_projection_size=8,
            gru_hidden_size=16,
            gru_num_layers=1,
            gru_dropout=0.0,
        )
        summary = run_mixed_bin_gru(config)
        self.assertIn("val_20ms_ctc_bpphone", summary["metrics"])
        self.assertIn("val_40ms_ctc_bpphone", summary["metrics"])

    def test_future_infonce_smoke_run(self) -> None:
        cache_root = Path(self._tmp_dir())
        _write_tiny_competition_probe_cache(cache_root)
        output_root = Path(self._tmp_dir())
        config = FutureInfoNCEConfig(
            cache_root=cache_root,
            output_root=output_root,
            run_name="tiny_future_infonce",
            model_family="s5",
            max_steps=1,
            batch_size=2,
            warmup_steps=0,
            val_every_steps=1,
            progress_every_steps=1,
            input_smoothing_sigma_ms=0.0,
            white_noise_sd=0.0,
            constant_offset_sd=0.0,
            patch_size_ms=40,
            patch_stride_ms=40,
            input_projection_size=8,
            s5_hidden_size=16,
            s5_state_size=4,
            s5_num_layers=1,
            s5_dropout=0.0,
            s5_ffn_multiplier=1.0,
            horizons_ms=(20, 40),
            projection_dim=8,
        )
        summary = run_future_infonce(config)
        self.assertIn("h20_infonce_loss", summary["metrics"])
        self.assertIn("h40_infonce_loss", summary["metrics"])


if __name__ == "__main__":
    unittest.main()

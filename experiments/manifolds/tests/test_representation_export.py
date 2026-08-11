from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from utah_ssl.stats_artifact_test_utils import (
    write_valid_split_stats_artifact as _write_valid_split_stats_artifact,
)
from utah_ssl.stats import (
    resolve_precomputed_split_stats_path,
)
from utah_ssl.experiment_contract import SignalSpec
from experiments.supervised_baselines.data import (
    adapter_keys_from_rows,
    build_willett_problem,
)
from experiments.supervised_baselines.model import (
    WillettPhonemeModel,
)
from experiments.manifolds.representation_export import (
    CONSONANT_CATEGORIES,
    PHONEME_CATEGORY_ORDER,
    RepresentationExportConfig,
    category_for_symbol,
    category_probability_frame,
    export_willett_representations,
    patch_timing_for_token,
)
from experiments.supervised_baselines.released_tf_checkpoint import (
    _reorder_keras_gru_gates,
    _reorder_stanford_logits_to_local_vocab,
)
from experiments.supervised_baselines.tests.test_willett_reconstruction import (
    _write_tiny_competition_probe_cache,
)
from experiments.supervised_baselines.train import (
    WillettReconstructionConfig,
)


class WillettRepresentationExportTest(unittest.TestCase):
    def _tmp_dir(self) -> Path:
        return Path(tempfile.mkdtemp(prefix="willett_representation_export_test_"))

    def test_phoneme_category_mapping_covers_expected_symbols(self) -> None:
        self.assertEqual(category_for_symbol("AA"), "vowel")
        self.assertEqual(category_for_symbol("T"), "stop")
        self.assertEqual(category_for_symbol("SH"), "fricative")
        self.assertEqual(category_for_symbol("NG"), "nasal")
        self.assertEqual(category_for_symbol("SIL"), "silence")
        self.assertEqual(category_for_symbol("BLANK"), "blank")
        self.assertEqual(category_for_symbol("NOT_A_REAL_PHONE"), "other")
        self.assertTrue(CONSONANT_CATEGORIES.issubset(set(PHONEME_CATEGORY_ORDER)))

    def test_category_probability_frame_sums_synthetic_probs(self) -> None:
        vocab = {
            "index_to_symbol": ["BLANK", "AA", "T", "SIL"],
            "num_classes": 4,
            "blank_index": 0,
            "sil_index": 3,
        }
        probs = np.array(
            [
                [0.10, 0.60, 0.20, 0.10],
                [0.05, 0.10, 0.70, 0.15],
            ],
            dtype=np.float32,
        )
        frame = category_probability_frame(probs, vocab=vocab)
        self.assertAlmostEqual(float(frame.loc[0, "vowel_prob"]), 0.60, places=6)
        self.assertAlmostEqual(float(frame.loc[0, "stop_prob"]), 0.20, places=6)
        self.assertAlmostEqual(float(frame.loc[0, "consonant_prob"]), 0.20, places=6)
        self.assertEqual(frame.loc[1, "top_category"], "stop")

    def test_patch_timing_uses_willett_window_and_stride(self) -> None:
        timing = patch_timing_for_token(
            3,
            patch_size=14,
            patch_stride=4,
            bin_size_ms=20,
        )
        self.assertEqual(timing["patch_start_bin"], 12)
        self.assertEqual(timing["patch_end_bin"], 26)
        self.assertEqual(timing["patch_start_ms"], 240)
        self.assertEqual(timing["patch_end_ms"], 520)

    def test_existing_export_without_input_windows_must_be_regenerated(self) -> None:
        checkpoint_path = self._tmp_dir() / "checkpoint_best.pt"
        checkpoint_path.write_bytes(b"placeholder")
        export_root = self._tmp_dir()
        export_dir = export_root / "tiny_gru"
        export_dir.mkdir(parents=True)
        (export_dir / "metadata.json").write_text(json.dumps({"input_window_dim": None}))

        with self.assertRaisesRegex(FileExistsError, "does not contain saved input windows"):
            export_willett_representations(
                RepresentationExportConfig(
                    checkpoint_path=checkpoint_path,
                    export_root=export_root,
                    model_key="tiny_gru",
                    save_input_windows=True,
                    overwrite=False,
                )
            )

    def test_released_checkpoint_gate_reorder_maps_keras_to_torch(self) -> None:
        value = np.arange(12, dtype=np.float32).reshape(2, 6)
        reordered = _reorder_keras_gru_gates(value)
        np.testing.assert_array_equal(
            reordered,
            np.array([[2, 3, 0, 1, 4, 5], [8, 9, 6, 7, 10, 11]], dtype=np.float32),
        )

    def test_released_checkpoint_classifier_reorder_moves_blank_to_zero(self) -> None:
        kernel = np.arange(2 * 41, dtype=np.float32).reshape(2, 41)
        bias = np.arange(41, dtype=np.float32)
        reordered_kernel, reordered_bias = _reorder_stanford_logits_to_local_vocab(kernel, bias)
        np.testing.assert_array_equal(reordered_kernel[:, 0], kernel[:, 40])
        np.testing.assert_array_equal(reordered_kernel[:, 1:], kernel[:, :40])
        np.testing.assert_array_equal(reordered_bias, np.concatenate([bias[40:41], bias[:40]]))

    def test_export_writes_consistent_shards_and_tables(self) -> None:
        cache_root = self._tmp_dir()
        output_root = self._tmp_dir()
        export_root = self._tmp_dir()
        _write_tiny_competition_probe_cache(cache_root)
        stats_path = resolve_precomputed_split_stats_path(
            cache_root=cache_root,
            dataset="brain2text24",
            train_split_name="competition_train",
            signal_spec=SignalSpec.tx_only(tx_dim=3),
            preferred_path=None,
        )
        _write_valid_split_stats_artifact(
            cache_root=cache_root,
            stats_path=stats_path,
            dataset="brain2text24",
            signal_spec=SignalSpec.tx_only(tx_dim=3),
            boundary_key_mode="session",
            split_policy="competition_train_test",
            train_split_name="competition_train",
            val_split_name="competition_test",
        )
        model_config = WillettReconstructionConfig(
            cache_root=cache_root,
            output_root=output_root,
            run_name="tiny_run",
            max_steps=1,
            batch_size=2,
            input_smoothing_sigma_bins=0.0,
            white_noise_sd=0.0,
            constant_offset_sd=0.0,
            patch_size=2,
            patch_stride=1,
            input_projection_size=8,
            gru_hidden_size=16,
            gru_num_layers=1,
            gru_dropout=0.0,
        )
        problem = build_willett_problem(
            cache_root=cache_root,
            dataset="brain2text24",
            feature_mode="tx_only",
            boundary_key_mode="session",
        )
        adapter_keys = tuple(
            dict.fromkeys(
                adapter_keys_from_rows(
                    problem["train_rows"],
                    dataset="brain2text24",
                    boundary_key_mode="session",
                )
                + adapter_keys_from_rows(
                    problem["val_rows"],
                    dataset="brain2text24",
                    boundary_key_mode="session",
                )
            )
        )
        model = WillettPhonemeModel(
            input_dim=3,
            vocab_size=4,
            patch_size=2,
            patch_stride=1,
            input_projection_size=8,
            gru_hidden_size=16,
            gru_num_layers=1,
            gru_dropout=0.0,
            session_adapter_keys=adapter_keys,
            session_adapter_enabled=True,
        )
        checkpoint_path = output_root / "checkpoint_best.pt"
        torch.save(
            {
                "model_family": "willett_reconstruction",
                "config": json.loads(json.dumps(model_config.__dict__, default=str)),
                "model_state": model.state_dict(),
                "step": 1,
                "vocab": problem["vocab"],
                "train_split_name": "competition_train",
                "val_split_name": "competition_test",
            },
            checkpoint_path,
        )

        metadata = export_willett_representations(
            RepresentationExportConfig(
                checkpoint_path=checkpoint_path,
                export_root=export_root,
                model_key="tiny_gru",
                max_examples=2,
                batch_size=2,
                shard_size_tokens=3,
                overwrite=True,
                save_input_windows=True,
            )
        )

        export_dir = export_root / "tiny_gru"
        self.assertTrue((export_dir / "metadata.json").exists())
        self.assertTrue((export_dir / "tokens.csv").exists())
        self.assertTrue((export_dir / "examples.csv").exists())
        self.assertTrue((export_dir / "shards.json").exists())
        self.assertEqual(metadata["example_count"], 2)
        tokens = pd.read_csv(export_dir / "tokens.csv")
        examples = pd.read_csv(export_dir / "examples.csv")
        self.assertEqual(int(tokens.shape[0]), int(metadata["token_count"]))
        self.assertEqual(int(examples.shape[0]), 2)
        self.assertIn("hidden_dim", metadata)
        self.assertEqual(metadata["input_window_dim"], 6)
        self.assertEqual(metadata["adapted_input_window_dim"], 16)
        self.assertIn("transition_type", tokens.columns)
        shard_manifest = json.loads((export_dir / "shards.json").read_text())
        self.assertGreaterEqual(len(shard_manifest), 1)
        shard_token_count = 0
        for shard in shard_manifest:
            arrays = np.load(export_dir / "shards" / shard["shard"])
            self.assertEqual(arrays["hidden"].shape[0], arrays["logits"].shape[0])
            self.assertEqual(arrays["hidden"].shape[0], arrays["input_windows"].shape[0])
            self.assertEqual(arrays["hidden"].shape[0], arrays["adapted_input_windows"].shape[0])
            self.assertEqual(arrays["input_windows"].shape[1], 6)
            self.assertEqual(arrays["adapted_input_windows"].shape[1], 16)
            shard_token_count += int(arrays["hidden"].shape[0])
        self.assertEqual(shard_token_count, int(metadata["token_count"]))


if __name__ == "__main__":
    unittest.main()

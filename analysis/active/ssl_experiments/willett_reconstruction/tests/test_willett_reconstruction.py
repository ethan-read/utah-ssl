from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from analysis.active.ssl_experiments.recompute_split_feature_stats import (
    resolve_precomputed_split_stats_path as resolve_canonical_split_stats_path,
)
from analysis.active.ssl_experiments.stats_artifact_test_utils import (
    write_valid_split_stats_artifact as _write_valid_split_stats_artifact,
)
from analysis.active.ssl_experiments.willett_reconstruction.data import (
    WillettInputTransformConfig,
    adapter_keys_from_rows,
    build_willett_problem,
    compute_willett_normalization_stats,
    group_rows_by_adapter_key,
    normalization_stats_missing_rows,
    prepare_willett_inputs,
)
from analysis.active.ssl_experiments.willett_reconstruction.model import (
    WillettPhonemeModel,
    patched_length,
)
from analysis.active.ssl_experiments.willett_reconstruction.train import (
    WillettReconstructionConfig,
    run_willett_reconstruction,
)


def _write_tiny_competition_probe_cache(cache_root: Path) -> None:
    dataset_dir = cache_root / "brain2text24"
    shards_dir = dataset_dir / "shards"
    shard_dir = shards_dir / "toy_shard"
    shard_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    examples = [
        ("train-0", "competition_train", "t12.2022.08.10", np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6]], dtype=np.float32), [1, 2]),
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

class WillettReconstructionTest(unittest.TestCase):
    def _tmp_dir(self) -> str:
        return tempfile.mkdtemp(prefix="willett_reconstruction_test_")

    def test_build_problem_uses_competition_split(self) -> None:
        cache_root = Path(self._tmp_dir())
        _write_tiny_competition_probe_cache(cache_root)
        problem = build_willett_problem(
            cache_root=cache_root,
            dataset="brain2text24",
            feature_mode="tx_only",
            boundary_key_mode="session",
        )
        self.assertEqual(problem["train_split_name"], "competition_train")
        self.assertEqual(problem["val_split_name"], "competition_test")
        self.assertEqual(len(problem["train_rows"]), 4)
        self.assertEqual(len(problem["val_rows"]), 2)

    def test_build_problem_can_use_competition_train_kfold(self) -> None:
        cache_root = Path(self._tmp_dir())
        _write_tiny_competition_probe_cache(cache_root)
        problem = build_willett_problem(
            cache_root=cache_root,
            dataset="brain2text24",
            feature_mode="tx_only",
            boundary_key_mode="session",
            split_policy="competition_train_kfold",
            cv_num_folds=2,
            cv_fold_index=1,
        )
        self.assertEqual(problem["split_policy"], "competition_train_kfold")
        self.assertEqual(problem["train_split_name"], "competition_train_cv2_fold1_train")
        self.assertEqual(problem["val_split_name"], "competition_train_cv2_fold1_val")
        self.assertEqual(len(problem["train_rows"]), 2)
        self.assertEqual(len(problem["val_rows"]), 2)
        self.assertTrue(all(row.source_split == "competition_train" for row in problem["train_rows"]))
        self.assertTrue(all(row.source_split == "competition_train" for row in problem["val_rows"]))
        self.assertEqual(set(problem["train_session_ids"]), {"t12.2022.08.10", "t12.2022.08.11"})
        self.assertEqual(set(problem["val_session_ids"]), {"t12.2022.08.10", "t12.2022.08.11"})

    def test_adapter_keys_follow_boundary_key_mode(self) -> None:
        cache_root = Path(self._tmp_dir())
        _write_tiny_competition_probe_cache(cache_root)
        problem = build_willett_problem(
            cache_root=cache_root,
            dataset="brain2text24",
            feature_mode="tx_only",
            boundary_key_mode="subject_if_available",
        )
        adapter_keys = adapter_keys_from_rows(
            problem["train_rows"],
            dataset="brain2text24",
            boundary_key_mode="subject_if_available",
        )
        self.assertEqual(adapter_keys, ("brain2text24:t12",))

    def test_group_rows_by_adapter_key_matches_session_partition(self) -> None:
        cache_root = Path(self._tmp_dir())
        _write_tiny_competition_probe_cache(cache_root)
        problem = build_willett_problem(
            cache_root=cache_root,
            dataset="brain2text24",
            feature_mode="tx_only",
            boundary_key_mode="session",
        )
        grouped = group_rows_by_adapter_key(
            problem["train_rows"],
            dataset="brain2text24",
            boundary_key_mode="session",
        )
        self.assertEqual(sorted(grouped), ["brain2text24:t12.2022.08.10", "brain2text24:t12.2022.08.11"])
        self.assertEqual(len(grouped["brain2text24:t12.2022.08.10"]), 2)
        self.assertEqual(len(grouped["brain2text24:t12.2022.08.11"]), 2)

    def test_prepare_inputs_preserves_shape(self) -> None:
        x = torch.arange(24, dtype=torch.float32).view(2, 4, 3)
        lengths = torch.tensor([4, 3], dtype=torch.long)
        transformed = prepare_willett_inputs(
            x,
            lengths,
            config=WillettInputTransformConfig(
                input_smoothing_sigma_bins=2.0,
                input_smoothing_kernel_size=100,
                input_smoothing_threshold=0.01,
                white_noise_sd=0.0,
                constant_offset_sd=0.0,
            ),
            is_training=False,
        )
        self.assertEqual(tuple(transformed.shape), (2, 4, 3))
        self.assertTrue(torch.allclose(transformed[1, 3], torch.zeros_like(transformed[1, 3])))

    def test_train_derived_block_stats_do_not_cover_val_blocks(self) -> None:
        cache_root = Path(self._tmp_dir())
        _write_tiny_competition_probe_cache(cache_root)
        problem = build_willett_problem(
            cache_root=cache_root,
            dataset="brain2text24",
            feature_mode="tx_only",
            boundary_key_mode="session",
        )
        train_stats = compute_willett_normalization_stats(
            problem["train_rows"],
            cache_root=cache_root,
            feature_mode="tx_only",
            mode="block",
        )
        missing = normalization_stats_missing_rows(train_stats, problem["val_rows"])
        self.assertEqual(sorted(missing), ["test-0", "test-1"])

    def test_model_forward_shapes_and_lengths(self) -> None:
        model = WillettPhonemeModel(
            input_dim=3,
            vocab_size=4,
            patch_size=3,
            patch_stride=2,
            input_projection_size=5,
            session_adapter_keys=("t12.2022.08.10",),
            session_adapter_enabled=True,
        )
        x = torch.randn(2, 5, 3)
        lengths = torch.tensor([5, 2], dtype=torch.long)
        outputs = model(
            x,
            lengths,
            session_ids=["t12.2022.08.10", "t12.2022.08.10"],
        )
        self.assertEqual(outputs["token_lengths"].tolist(), [patched_length(5, patch_size=3, patch_stride=2), patched_length(2, patch_size=3, patch_stride=2)])
        self.assertEqual(int(outputs["adapted_input"].shape[-1]), 5)
        self.assertEqual(int(outputs["patched_inputs"].shape[-1]), 15)
        self.assertEqual(int(outputs["logits"].shape[-1]), 4)
        self.assertEqual(tuple(model.initial_state.shape), (1, model.gru_hidden_size))
        self.assertTrue(model.initial_state.requires_grad)

    def test_model_uses_shared_input_network_when_session_adaptation_disabled(self) -> None:
        model = WillettPhonemeModel(
            input_dim=3,
            vocab_size=4,
            patch_size=2,
            patch_stride=1,
            input_projection_size=6,
            session_adapter_keys=(),
            session_adapter_enabled=False,
            gru_hidden_size=8,
            gru_num_layers=1,
            gru_dropout=0.0,
        )
        x = torch.randn(2, 4, 3)
        lengths = torch.tensor([4, 3], dtype=torch.long)
        outputs = model(x, lengths)
        self.assertEqual(int(outputs["adapted_input"].shape[-1]), 6)
        self.assertEqual(int(outputs["patched_inputs"].shape[-1]), 12)
        self.assertEqual(
            outputs["token_lengths"].tolist(),
            [
                patched_length(4, patch_size=2, patch_stride=1),
                patched_length(3, patch_size=2, patch_stride=1),
            ],
        )

    def test_s5_model_forward_shapes_and_lengths(self) -> None:
        model = WillettPhonemeModel(
            input_dim=3,
            vocab_size=4,
            patch_size=3,
            patch_stride=2,
            input_projection_size=5,
            decoder_backbone_type="s5",
            s5_hidden_size=8,
            s5_state_size=4,
            s5_num_layers=1,
            s5_dropout=0.0,
            s5_ffn_multiplier=1.0,
            session_adapter_keys=("t12.2022.08.10",),
            session_adapter_enabled=True,
        )
        x = torch.randn(2, 5, 3)
        lengths = torch.tensor([5, 2], dtype=torch.long)
        outputs = model(
            x,
            lengths,
            session_ids=["t12.2022.08.10", "t12.2022.08.10"],
        )
        self.assertEqual(
            outputs["token_lengths"].tolist(),
            [
                patched_length(5, patch_size=3, patch_stride=2),
                patched_length(2, patch_size=3, patch_stride=2),
            ],
        )
        self.assertEqual(int(outputs["patched_inputs"].shape[-1]), 15)
        self.assertEqual(int(outputs["projected_inputs"].shape[-1]), 8)
        self.assertEqual(int(outputs["hidden"].shape[-1]), 8)
        self.assertEqual(int(outputs["logits"].shape[-1]), 4)
        self.assertFalse(any(name.startswith("gru.") for name, _ in model.named_parameters()))

    def test_s4d_model_forward_shapes_and_lengths(self) -> None:
        model = WillettPhonemeModel(
            input_dim=3,
            vocab_size=4,
            patch_size=3,
            patch_stride=2,
            input_projection_size=5,
            decoder_backbone_type="s4d",
            s4d_hidden_size=8,
            s4d_state_size=4,
            s4d_num_layers=1,
            s4d_dropout=0.0,
            s4d_ffn_multiplier=1.0,
            session_adapter_keys=("t12.2022.08.10",),
            session_adapter_enabled=True,
        )
        x = torch.randn(2, 5, 3)
        lengths = torch.tensor([5, 2], dtype=torch.long)
        outputs = model(
            x,
            lengths,
            session_ids=["t12.2022.08.10", "t12.2022.08.10"],
        )
        self.assertEqual(
            outputs["token_lengths"].tolist(),
            [
                patched_length(5, patch_size=3, patch_stride=2),
                patched_length(2, patch_size=3, patch_stride=2),
            ],
        )
        self.assertEqual(int(outputs["patched_inputs"].shape[-1]), 15)
        self.assertEqual(int(outputs["projected_inputs"].shape[-1]), 8)
        self.assertEqual(int(outputs["hidden"].shape[-1]), 8)
        self.assertEqual(int(outputs["logits"].shape[-1]), 4)
        self.assertTrue(torch.isfinite(outputs["logits"]).all())
        self.assertFalse(any(name.startswith("gru.") for name, _ in model.named_parameters()))

    def test_short_training_run_writes_outputs_and_can_resume(self) -> None:
        cache_root = Path(self._tmp_dir())
        _write_tiny_competition_probe_cache(cache_root)
        output_root = Path(self._tmp_dir())
        stats_path = resolve_canonical_split_stats_path(
            cache_root=cache_root,
            dataset="brain2text24",
            train_split_name="competition_train",
            feature_mode="tx_only",
            preferred_path=None,
        )
        _write_valid_split_stats_artifact(
            cache_root=cache_root,
            stats_path=stats_path,
            dataset="brain2text24",
            feature_mode="tx_only",
            boundary_key_mode="session",
            train_split_name="competition_train",
            val_split_name="competition_test",
            dim=3,
        )
        config = WillettReconstructionConfig(
            cache_root=cache_root,
            output_root=output_root,
            run_name="tiny_run",
            max_steps=2,
            batch_size=2,
            learning_rate=1e-3,
            min_learning_rate=1e-4,
            warmup_steps=0,
            adam_epsilon=1e-1,
            val_every_steps=1,
            checkpoint_every_steps=1,
            progress_every_steps=1,
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
        self.assertEqual(config.normalization_mode, "global")
        summary = run_willett_reconstruction(config)
        run_dir = output_root / "tiny_run"
        self.assertTrue((run_dir / "progress.jsonl").exists())
        self.assertTrue((run_dir / "summary.json").exists())
        self.assertTrue((run_dir / "checkpoint_best.pt").exists())
        self.assertTrue((run_dir / "checkpoint_final.pt").exists())
        self.assertTrue(any((run_dir / "checkpoints").glob("step_*.pt")))
        self.assertIn("collapse_diagnostics", summary["metrics"])
        self.assertIn("predicted_to_reference_token_ratio", summary["metrics"]["collapse_diagnostics"])
        self.assertEqual(summary["train_sampling_mode"], "uniform_single_boundary_key_per_step")

        resumed_summary = run_willett_reconstruction(
            WillettReconstructionConfig(
                **{
                    **config.__dict__,
                    "max_steps": 3,
                    "resume_latest": True,
                }
            )
        )
        self.assertGreaterEqual(int(resumed_summary["steps"]), 3)
        self.assertGreaterEqual(int(resumed_summary["best_step"]), 1)

    def test_short_s5_training_run_writes_outputs(self) -> None:
        cache_root = Path(self._tmp_dir())
        _write_tiny_competition_probe_cache(cache_root)
        output_root = Path(self._tmp_dir())
        stats_path = resolve_canonical_split_stats_path(
            cache_root=cache_root,
            dataset="brain2text24",
            train_split_name="competition_train",
            feature_mode="tx_only",
            preferred_path=None,
        )
        _write_valid_split_stats_artifact(
            cache_root=cache_root,
            stats_path=stats_path,
            dataset="brain2text24",
            feature_mode="tx_only",
            boundary_key_mode="session",
            train_split_name="competition_train",
            val_split_name="competition_test",
            dim=3,
        )
        config = WillettReconstructionConfig(
            cache_root=cache_root,
            output_root=output_root,
            run_name="tiny_s5_run",
            max_steps=1,
            batch_size=2,
            learning_rate=1e-3,
            min_learning_rate=1e-4,
            warmup_steps=0,
            adam_epsilon=1e-1,
            val_every_steps=1,
            checkpoint_every_steps=1,
            progress_every_steps=1,
            input_smoothing_sigma_bins=0.0,
            white_noise_sd=0.0,
            constant_offset_sd=0.0,
            patch_size=2,
            patch_stride=1,
            input_projection_size=8,
            decoder_backbone_type="s5",
            s5_hidden_size=16,
            s5_state_size=4,
            s5_num_layers=1,
            s5_dropout=0.0,
            s5_ffn_multiplier=1.0,
        )
        summary = run_willett_reconstruction(config)
        run_dir = output_root / "tiny_s5_run"
        self.assertTrue((run_dir / "progress.jsonl").exists())
        self.assertTrue((run_dir / "summary.json").exists())
        self.assertTrue((run_dir / "checkpoint_best.pt").exists())
        self.assertTrue((run_dir / "checkpoint_final.pt").exists())
        self.assertEqual(summary["config"]["decoder_backbone_type"], "s5")
        checkpoint_payload = torch.load(run_dir / "checkpoint_final.pt", map_location="cpu", weights_only=False)
        self.assertFalse(any(name.startswith("gru.") for name in checkpoint_payload["model_state"]))

    def test_short_cv_training_run_computes_fold_stats(self) -> None:
        cache_root = Path(self._tmp_dir())
        _write_tiny_competition_probe_cache(cache_root)
        output_root = Path(self._tmp_dir())
        config = WillettReconstructionConfig(
            cache_root=cache_root,
            output_root=output_root,
            run_name="tiny_cv_run",
            split_policy="competition_train_kfold",
            cv_num_folds=2,
            cv_fold_index=0,
            max_steps=1,
            batch_size=2,
            learning_rate=1e-3,
            min_learning_rate=1e-4,
            warmup_steps=0,
            adam_epsilon=1e-1,
            val_every_steps=1,
            checkpoint_every_steps=1,
            progress_every_steps=1,
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
        summary = run_willett_reconstruction(config)
        run_dir = output_root / "tiny_cv_run"
        self.assertTrue((run_dir / "summary.json").exists())
        self.assertEqual(summary["split_policy"], "competition_train_kfold")
        self.assertEqual(summary["cv_num_folds"], 2)
        self.assertEqual(summary["cv_fold_index"], 0)
        self.assertEqual(summary["precomputed_split_stats_path"], None)
        self.assertEqual(summary["train_examples"], 2)
        self.assertEqual(summary["val_examples"], 2)

    def test_short_s4d_training_run_writes_outputs(self) -> None:
        cache_root = Path(self._tmp_dir())
        _write_tiny_competition_probe_cache(cache_root)
        output_root = Path(self._tmp_dir())
        stats_path = resolve_canonical_split_stats_path(
            cache_root=cache_root,
            dataset="brain2text24",
            train_split_name="competition_train",
            feature_mode="tx_only",
            preferred_path=None,
        )
        _write_valid_split_stats_artifact(
            cache_root=cache_root,
            stats_path=stats_path,
            dataset="brain2text24",
            feature_mode="tx_only",
            boundary_key_mode="session",
            train_split_name="competition_train",
            val_split_name="competition_test",
            dim=3,
        )
        config = WillettReconstructionConfig(
            cache_root=cache_root,
            output_root=output_root,
            run_name="tiny_s4d_run",
            max_steps=1,
            batch_size=2,
            learning_rate=1e-3,
            min_learning_rate=1e-4,
            warmup_steps=0,
            adam_epsilon=1e-1,
            val_every_steps=1,
            checkpoint_every_steps=1,
            progress_every_steps=1,
            input_smoothing_sigma_bins=0.0,
            white_noise_sd=0.0,
            constant_offset_sd=0.0,
            patch_size=2,
            patch_stride=1,
            input_projection_size=8,
            decoder_backbone_type="s4d",
            s4d_hidden_size=16,
            s4d_state_size=4,
            s4d_num_layers=1,
            s4d_dropout=0.0,
            s4d_ffn_multiplier=1.0,
        )
        summary = run_willett_reconstruction(config)
        run_dir = output_root / "tiny_s4d_run"
        self.assertTrue((run_dir / "progress.jsonl").exists())
        self.assertTrue((run_dir / "summary.json").exists())
        self.assertTrue((run_dir / "checkpoint_best.pt").exists())
        self.assertTrue((run_dir / "checkpoint_final.pt").exists())
        self.assertEqual(summary["config"]["decoder_backbone_type"], "s4d")
        checkpoint_payload = torch.load(run_dir / "checkpoint_final.pt", map_location="cpu", weights_only=False)
        self.assertFalse(any(name.startswith("gru.") for name in checkpoint_payload["model_state"]))

    def test_willett_global_normalization_fails_without_canonical_split_stats(self) -> None:
        cache_root = Path(self._tmp_dir())
        _write_tiny_competition_probe_cache(cache_root)
        output_root = Path(self._tmp_dir())
        config = WillettReconstructionConfig(
            cache_root=cache_root,
            output_root=output_root,
            run_name="missing_stats",
            max_steps=1,
            batch_size=2,
            learning_rate=1e-3,
            min_learning_rate=1e-4,
            warmup_steps=0,
            adam_epsilon=1e-1,
            val_every_steps=1,
            checkpoint_every_steps=1,
            progress_every_steps=1,
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
        with self.assertRaisesRegex(FileNotFoundError, "recompute_command:"):
            run_willett_reconstruction(config)

    def test_willett_global_normalization_fails_for_stale_split_stats_signature(self) -> None:
        cache_root = Path(self._tmp_dir())
        _write_tiny_competition_probe_cache(cache_root)
        output_root = Path(self._tmp_dir())
        stats_path = resolve_canonical_split_stats_path(
            cache_root=cache_root,
            dataset="brain2text24",
            train_split_name="competition_train",
            feature_mode="tx_only",
            preferred_path=None,
        )
        _write_valid_split_stats_artifact(
            cache_root=cache_root,
            stats_path=stats_path,
            dataset="brain2text24",
            feature_mode="tx_only",
            boundary_key_mode="session",
            train_split_name="competition_train",
            val_split_name="competition_test",
            dim=3,
        )
        payload = torch.load(stats_path, map_location="cpu")
        payload["metadata"]["source_cache_signature"] = "stale"
        torch.save(payload, stats_path)
        stats_path.with_suffix(".json").write_text(json.dumps(payload["metadata"], indent=2) + "\n")

        config = WillettReconstructionConfig(
            cache_root=cache_root,
            output_root=output_root,
            run_name="stale_stats",
            max_steps=1,
            batch_size=2,
            learning_rate=1e-3,
            min_learning_rate=1e-4,
            warmup_steps=0,
            adam_epsilon=1e-1,
            val_every_steps=1,
            checkpoint_every_steps=1,
            progress_every_steps=1,
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
        with self.assertRaisesRegex(ValueError, "source_cache_signature"):
            run_willett_reconstruction(config)


if __name__ == "__main__":
    unittest.main()

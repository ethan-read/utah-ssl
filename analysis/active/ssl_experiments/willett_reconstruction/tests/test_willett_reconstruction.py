from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from analysis.active.ssl_experiments.willett_reconstruction.data import (
    WillettInputTransformConfig,
    build_willett_problem,
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
        self.assertEqual(len(problem["train_rows"]), 3)
        self.assertEqual(len(problem["val_rows"]), 2)

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

    def test_model_forward_shapes_and_lengths(self) -> None:
        model = WillettPhonemeModel(
            input_dim=3,
            vocab_size=4,
            patch_size=3,
            patch_stride=2,
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
        self.assertEqual(int(outputs["logits"].shape[-1]), 4)

    def test_short_training_run_writes_outputs_and_can_resume(self) -> None:
        cache_root = Path(self._tmp_dir())
        _write_tiny_competition_probe_cache(cache_root)
        output_root = Path(self._tmp_dir())
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
        summary = run_willett_reconstruction(config)
        run_dir = output_root / "tiny_run"
        self.assertTrue((run_dir / "progress.jsonl").exists())
        self.assertTrue((run_dir / "summary.json").exists())
        self.assertTrue((run_dir / "checkpoint_best.pt").exists())
        self.assertTrue((run_dir / "checkpoint_final.pt").exists())
        self.assertTrue(any((run_dir / "checkpoints").glob("step_*.pt")))
        self.assertIn("collapse_diagnostics", summary["metrics"])
        self.assertIn("predicted_to_reference_token_ratio", summary["metrics"]["collapse_diagnostics"])

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


if __name__ == "__main__":
    unittest.main()

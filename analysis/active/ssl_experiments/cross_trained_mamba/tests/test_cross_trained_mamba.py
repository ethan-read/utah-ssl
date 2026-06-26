from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from analysis.active.ssl_experiments.cross_trained_mamba.config import CrossTrainedMambaConfig
from analysis.active.ssl_experiments.cross_trained_mamba.data import (
    CrossDatasetSequenceDataset,
    build_cross_dataset_problem,
    compute_dataset_train_stats,
    cross_dataset_adapter_key,
)
from analysis.active.ssl_experiments.cross_trained_mamba.model import CrossTrainedMambaPhonemeModel
from analysis.active.ssl_experiments.cross_trained_mamba.train import (
    compute_hierarchical_ctc_losses,
    run_cross_trained_mamba,
)
from analysis.active.ssl_experiments.masked_ssl.probe import CanonicalProbeManifestRow


VOCAB = {
    "index_to_symbol": ["BLANK", "AA", "AE", "SIL"],
    "num_classes": 4,
    "blank_index": 0,
    "sil_index": 3,
}


def _make_feature_block(rows: int, cols: int, *, base: float, extra_offset: float = 0.0) -> np.ndarray:
    data = np.arange(rows * cols, dtype=np.float32).reshape(rows, cols)
    return data + float(base) + float(extra_offset)


def _write_dataset(
    cache_root: Path,
    *,
    dataset: str,
    train_split: str,
    val_split: str,
    tx_dim: int,
    sbp_dim: int,
    full_width_tail_offset: float = 0.0,
) -> None:
    dataset_dir = cache_root / dataset
    shard_dir = dataset_dir / "shards" / "toy_shard"
    shard_dir.mkdir(parents=True, exist_ok=True)

    example_specs = [
        ("train-0", train_split, "subj0.2025.01.01"),
        ("train-1", train_split, "subj0.2025.01.02"),
        ("val-0", val_split, "subj0.2025.01.03"),
        ("val-1", val_split, "subj0.2025.01.04"),
    ]
    time_offsets = [0]
    phoneme_offsets = [0]
    tx_rows = []
    sbp_rows = []
    phoneme_ids: list[int] = []
    manifest_rows = []

    for example_index, (example_id, split_name, session_id) in enumerate(example_specs):
        tx = _make_feature_block(4, tx_dim, base=10.0 * (example_index + 1))
        sbp = _make_feature_block(4, sbp_dim, base=100.0 * (example_index + 1))
        if full_width_tail_offset > 0.0:
            tx[:, tx_dim // 2 :] += float(full_width_tail_offset)
            sbp[:, sbp_dim // 2 :] += float(full_width_tail_offset)
        tx_rows.append(tx)
        sbp_rows.append(sbp)
        labels = [1, 2]
        phoneme_ids.extend(labels)
        time_offsets.append(time_offsets[-1] + 4)
        phoneme_offsets.append(phoneme_offsets[-1] + len(labels))
        manifest_rows.append(
            {
                "example_id": example_id,
                "session_id": session_id,
                "subject_id": "subj0",
                "session_date": session_id.split(".", 1)[1],
                "source_split": split_name,
                "has_labels": True,
                "shard_relpath": f"{dataset}/shards/toy_shard",
                "example_index": example_index,
                "n_tx_features": tx_dim,
                "n_sbp_features": sbp_dim,
                "n_time_bins": 4,
                "target_length": len(labels),
                "transcript": "AA",
                "has_tx": True,
                "has_sbp": True,
            }
        )

    np.save(shard_dir / "tx.npy", np.concatenate(tx_rows, axis=0).astype(np.float32))
    np.save(shard_dir / "sbp.npy", np.concatenate(sbp_rows, axis=0).astype(np.float32))
    np.save(shard_dir / "time_offsets.npy", np.asarray(time_offsets, dtype=np.int64))
    np.save(shard_dir / "phoneme_offsets.npy", np.asarray(phoneme_offsets, dtype=np.int64))
    np.save(shard_dir / "phoneme_ids.npy", np.asarray(phoneme_ids, dtype=np.int64))
    (dataset_dir / "manifest.jsonl").write_text("".join(json.dumps(row) + "\n" for row in manifest_rows))
    (dataset_dir / "metadata.json").write_text(
        json.dumps({"dataset_family": dataset, "phoneme_vocabulary": VOCAB})
    )


def _write_tiny_cross_dataset_cache(cache_root: Path) -> None:
    _write_dataset(
        cache_root,
        dataset="brain2text24",
        train_split="competition_train",
        val_split="competition_test",
        tx_dim=128,
        sbp_dim=128,
    )
    _write_dataset(
        cache_root,
        dataset="brain2text25",
        train_split="train",
        val_split="val",
        tx_dim=256,
        sbp_dim=256,
        full_width_tail_offset=10000.0,
    )


class CrossTrainedMambaTest(unittest.TestCase):
    def _tmp_dir(self) -> str:
        return tempfile.mkdtemp(prefix="cross_trained_mamba_test_")

    def test_b2t25_runtime_area6v_slice_returns_256_features(self) -> None:
        cache_root = Path(self._tmp_dir())
        _write_tiny_cross_dataset_cache(cache_root)
        problem = build_cross_dataset_problem(
            cache_root=cache_root,
            datasets=("brain2text25",),
            feature_mode="tx_sbp",
        )
        stats_by_dataset = compute_dataset_train_stats(problem=problem, area6v_feature_dim=128)
        dataset = CrossDatasetSequenceDataset(
            problem["rows_by_dataset"]["brain2text25"]["train"],
            cache_root=cache_root,
            dataset="brain2text25",
            stats=stats_by_dataset["brain2text25"],
            feature_mode="tx_sbp",
            area6v_feature_dim=128,
        )
        item = dataset[0]
        self.assertEqual(tuple(item["x"].shape), (4, 256))
        self.assertLess(float(item["x"].abs().max().item()), 1000.0)

    def test_unsliced_512_dim_tensor_is_rejected(self) -> None:
        model = CrossTrainedMambaPhonemeModel(
            input_dim=256,
            vocab_size=4,
            hidden_size=16,
            state_size=8,
            stage1_num_layers=1,
            stage2_num_layers=1,
            stage3_num_layers=1,
            dropout=0.0,
            session_adapter_keys=("brain2text25:subj0:2025.01.01",),
        )
        x = torch.randn(2, 4, 512)
        lengths = torch.tensor([4, 4], dtype=torch.long)
        with self.assertRaisesRegex(ValueError, "Expected input feature dim 256"):
            model(x, lengths, session_ids=["brain2text25:subj0:2025.01.01"] * 2)

    def test_b2t24_rows_load_unchanged_at_area6v_width(self) -> None:
        cache_root = Path(self._tmp_dir())
        _write_tiny_cross_dataset_cache(cache_root)
        problem = build_cross_dataset_problem(
            cache_root=cache_root,
            datasets=("brain2text24",),
            feature_mode="tx_sbp",
        )
        dataset = CrossDatasetSequenceDataset(
            problem["rows_by_dataset"]["brain2text24"]["train"],
            cache_root=cache_root,
            dataset="brain2text24",
            stats=None,
            feature_mode="tx_sbp",
            area6v_feature_dim=128,
        )
        item = dataset[0]
        self.assertEqual(tuple(item["x"].shape), (4, 256))

    def test_per_dataset_stats_have_dim_256_after_slicing(self) -> None:
        cache_root = Path(self._tmp_dir())
        _write_tiny_cross_dataset_cache(cache_root)
        problem = build_cross_dataset_problem(
            cache_root=cache_root,
            datasets=("brain2text24", "brain2text25"),
            feature_mode="tx_sbp",
        )
        stats = compute_dataset_train_stats(problem=problem, area6v_feature_dim=128)
        self.assertEqual(tuple(stats["brain2text24"][0].shape), (256,))
        self.assertEqual(tuple(stats["brain2text25"][0].shape), (256,))

    def test_adapter_keys_are_dataset_qualified(self) -> None:
        row = CanonicalProbeManifestRow(
            example_id="ex0",
            session_id="subj0.2025.01.01",
            subject_id="subj0",
            source_split="train",
            has_labels=True,
            shard_relpath="brain2text24/shards/toy_shard",
            example_index=0,
            n_tx_features=128,
            n_sbp_features=128,
            target_length=2,
            transcript="AA",
        )
        self.assertNotEqual(
            cross_dataset_adapter_key(row, dataset="brain2text24"),
            cross_dataset_adapter_key(row, dataset="brain2text25"),
        )

    def test_model_returns_hierarchical_logits(self) -> None:
        model = CrossTrainedMambaPhonemeModel(
            input_dim=256,
            vocab_size=4,
            hidden_size=16,
            state_size=8,
            stage1_num_layers=1,
            stage2_num_layers=1,
            stage3_num_layers=1,
            dropout=0.0,
            session_adapter_keys=("brain2text24:subj0:2025.01.01",),
        )
        x = torch.randn(2, 5, 256)
        lengths = torch.tensor([5, 4], dtype=torch.long)
        outputs = model(x, lengths, session_ids=["brain2text24:subj0:2025.01.01"] * 2)
        self.assertEqual(tuple(outputs["l1"].shape), (2, 5, 4))
        self.assertEqual(tuple(outputs["l2"].shape), (2, 5, 4))
        self.assertEqual(tuple(outputs["l3"].shape), (2, 5, 4))

    def test_hierarchical_ctc_uses_expected_weights(self) -> None:
        model = CrossTrainedMambaPhonemeModel(
            input_dim=256,
            vocab_size=4,
            hidden_size=16,
            state_size=8,
            stage1_num_layers=1,
            stage2_num_layers=1,
            stage3_num_layers=1,
            dropout=0.0,
            session_adapter_keys=("brain2text24:subj0:2025.01.01",),
        )
        x = torch.randn(2, 5, 256)
        lengths = torch.tensor([5, 4], dtype=torch.long)
        labels = torch.tensor([[1, 2], [2, 1]], dtype=torch.long)
        label_lengths = torch.tensor([2, 2], dtype=torch.long)
        outputs = model(x, lengths, session_ids=["brain2text24:subj0:2025.01.01"] * 2)
        total, parts, target_count = compute_hierarchical_ctc_losses(
            outputs,
            labels,
            label_lengths,
            blank_index=0,
            intermediate_ctc_weight=0.3,
        )
        expected = parts["l1_loss_sum"] + 0.3 * parts["l2_loss_sum"] + parts["l3_loss_sum"]
        self.assertEqual(target_count, 4)
        self.assertAlmostEqual(float(total.item()), float(expected), places=5)

    def test_tiny_train_loop_writes_checkpoints_and_summary(self) -> None:
        cache_root = Path(self._tmp_dir())
        output_root = Path(self._tmp_dir())
        _write_tiny_cross_dataset_cache(cache_root)
        config = CrossTrainedMambaConfig(
            cache_root=cache_root,
            output_root=output_root,
            run_name="tiny_cross_mamba",
            max_steps=1,
            batch_size=2,
            learning_rate=1e-3,
            min_learning_rate=1e-5,
            warmup_steps=0,
            weight_decay=0.0,
            adam_epsilon=1e-8,
            val_every_steps=1,
            checkpoint_every_steps=1,
            checkpoint_keep_last=1,
            progress_every_steps=1,
            progress_every_seconds=0.0,
            hidden_size=16,
            state_size=8,
            stage1_num_layers=1,
            stage2_num_layers=1,
            stage3_num_layers=1,
            dropout=0.0,
            input_smoothing_sigma_bins=0.0,
            white_noise_sd=0.0,
            constant_offset_sd=0.0,
        )
        summary = run_cross_trained_mamba(config)
        run_dir = output_root / "tiny_cross_mamba"
        self.assertTrue((run_dir / "progress.jsonl").exists())
        self.assertTrue((run_dir / "summary.json").exists())
        self.assertTrue((run_dir / "checkpoint_best.pt").exists())
        self.assertTrue((run_dir / "checkpoint_final.pt").exists())
        self.assertIn("by_dataset", summary["metrics"])
        self.assertIn("brain2text24", summary["metrics"]["by_dataset"])
        self.assertIn("brain2text25", summary["metrics"]["by_dataset"])
        payload = torch.load(run_dir / "checkpoint_final.pt", map_location="cpu", weights_only=False)
        self.assertEqual(int(payload["step"]), 1)
        self.assertIn("optimizer_state", payload)
        self.assertIn("scheduler_state", payload)


if __name__ == "__main__":
    unittest.main()

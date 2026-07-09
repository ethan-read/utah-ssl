from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[5]
EXPERIMENTS_DIR = REPO_ROOT / "analysis" / "active" / "ssl_experiments"
for path in (REPO_ROOT, EXPERIMENTS_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from ssl_core.scripts.trim_area6v_cache import trim_area6v_cache
from masked_ssl.probe import CanonicalShardAccessor, build_competition_split_problem


def _write_full_width_cache(cache_root: Path) -> None:
    dataset_root = cache_root / "brain2text24"
    shard_dir = dataset_root / "shards" / "shard_000"
    shard_dir.mkdir(parents=True)
    tx = np.arange(20 * 256, dtype=np.uint8).reshape(20, 256)
    sbp = np.arange(20 * 256, dtype=np.float32).reshape(20, 256)
    np.save(shard_dir / "tx.npy", tx)
    np.save(shard_dir / "sbp.npy", sbp)
    np.save(shard_dir / "time_offsets.npy", np.asarray([0, 10, 20], dtype=np.int64))
    np.save(shard_dir / "phoneme_offsets.npy", np.asarray([0, 1, 2], dtype=np.int64))
    np.save(shard_dir / "phoneme_ids.npy", np.asarray([1, 2], dtype=np.int32))

    rows = [
        {
            "session_id": "t12.2022.01.01",
            "subject_id": "t12",
            "source_split": "competition_train" if idx == 0 else "competition_test",
            "example_id": f"example-{idx}",
            "shard_relpath": "brain2text24/shards/shard_000",
            "example_index": idx,
            "n_time_bins": 10,
            "has_tx": True,
            "has_sbp": True,
            "n_tx_features": 256,
            "n_sbp_features": 256,
            "has_labels": True,
            "target_length": 1,
            "transcript": "test",
            "normalization_group": "t12.2022.01.01",
        }
        for idx in range(2)
    ]
    with (dataset_root / "manifest.jsonl").open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    metadata = {
        "dataset_family": "brain2text24",
        "modalities": ["tx", "sbp"],
        "feature_layout": {
            "n_total_features": 512,
            "n_tx_features": 256,
            "n_sbp_features": 256,
        },
        "shards": [
            {
                "shard_id": "shard_000",
                "shard_relpath": "brain2text24/shards/shard_000",
                "n_tx_features": 256,
                "n_sbp_features": 256,
            }
        ],
    }
    (dataset_root / "metadata.json").write_text(json.dumps(metadata, indent=2))


class Area6vCacheMigrationTests(unittest.TestCase):
    def test_trims_arrays_manifest_and_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_root = Path(tmpdir) / "cache"
            _write_full_width_cache(cache_root)

            summary = trim_area6v_cache(cache_root)

            self.assertEqual(summary.arrays_trimmed, 2)
            tx = np.load(cache_root / "brain2text24/shards/shard_000/tx.npy")
            sbp = np.load(cache_root / "brain2text24/shards/shard_000/sbp.npy")
            self.assertEqual(tx.shape, (20, 128))
            self.assertEqual(sbp.shape, (20, 128))
            np.testing.assert_array_equal(tx, np.arange(20 * 256, dtype=np.uint8).reshape(20, 256)[:, :128])

            rows = [
                json.loads(line)
                for line in (cache_root / "brain2text24/manifest.jsonl").read_text().splitlines()
            ]
            self.assertEqual({row["n_tx_features"] for row in rows}, {128})
            self.assertEqual({row["n_sbp_features"] for row in rows}, {128})

            metadata = json.loads((cache_root / "brain2text24/metadata.json").read_text())
            self.assertEqual(metadata["feature_layout"]["n_total_features"], 256)
            self.assertEqual(metadata["feature_layout"]["n_tx_features"], 128)
            self.assertEqual(metadata["feature_layout"]["n_sbp_features"], 128)
            self.assertTrue(metadata["area6v_migration"]["area6v_only"])
            self.assertEqual(metadata["area6v_migration"]["trimmed_feature_columns"], [0, 128])
            self.assertTrue((cache_root / "brain2text24/metadata.json.pre_area6v_backup").exists())

            problem = build_competition_split_problem(
                cache_root=cache_root,
                dataset="brain2text24",
                feature_mode="tx_sbp",
            )
            accessor = CanonicalShardAccessor(cache_root)
            x_tx = accessor.load_features(problem["train_rows"][0], feature_mode="tx_only")
            x_tx_sbp = accessor.load_features(problem["train_rows"][0], feature_mode="tx_sbp")
            self.assertEqual(x_tx.shape, (10, 128))
            self.assertEqual(x_tx_sbp.shape, (10, 256))

    def test_second_run_is_idempotent_for_arrays(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_root = Path(tmpdir) / "cache"
            _write_full_width_cache(cache_root)

            trim_area6v_cache(cache_root)
            summary = trim_area6v_cache(cache_root)

            self.assertEqual(summary.arrays_trimmed, 0)
            self.assertEqual(summary.arrays_already_trimmed, 2)
            self.assertEqual(summary.manifest_rows_updated, 0)


if __name__ == "__main__":
    unittest.main()

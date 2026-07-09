from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[5]
EXPERIMENTS_DIR = REPO_ROOT / "analysis" / "active" / "ssl_experiments"
for path in (REPO_ROOT, EXPERIMENTS_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from ssl_core.bit_cache_contract import (
    BIT_CANONICAL_FEATURE_POLICY,
    BIT_STAGE1_DEFAULT_EXCLUDED_DATASETS,
    BIT_STAGE1_DEFAULT_INCLUDED_DATASETS,
    BIT_STAGE1_DEFAULT_STATS_STEM,
    BIT_STAGE1_FEATURE_MODE,
    BIT_STAGE1_SBP_DIM,
    BIT_STAGE1_TX_DIM,
)
from ssl_core.scripts.harmonize_bit_cache import harmonize_bit_cache
from masked_ssl.cache import CacheAccessConfig, prepare_cache_context
from ssl_core.scripts.prepare_bit_cache import prepare_bit_cache
from ssl_core.stats_artifact_test_utils import write_valid_session_stats_artifact


def _write_manifest(dataset_root: Path, rows: list[dict[str, object]]) -> None:
    with (dataset_root / "manifest.jsonl").open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _write_metadata(dataset_root: Path, payload: dict[str, object]) -> None:
    (dataset_root / "metadata.json").write_text(json.dumps(payload, indent=2) + "\n")


def _write_tx_sbp_dataset(
    cache_root: Path,
    *,
    dataset: str,
    tx_dim: int,
    sbp_dim: int,
    full_width: bool = False,
    session_id: str = "toy.2025.01.01",
) -> None:
    dataset_root = cache_root / dataset
    shard_dir = dataset_root / "shards" / "shard_000"
    shard_dir.mkdir(parents=True, exist_ok=True)
    tx = np.arange(6 * tx_dim, dtype=np.float32).reshape(6, tx_dim)
    sbp = (100.0 + np.arange(6 * sbp_dim, dtype=np.float32)).reshape(6, sbp_dim)
    np.save(shard_dir / "tx.npy", tx)
    np.save(shard_dir / "sbp.npy", sbp)
    np.save(shard_dir / "time_offsets.npy", np.array([0, 3, 6], dtype=np.int64))
    np.save(shard_dir / "phoneme_offsets.npy", np.array([0, 1, 2], dtype=np.int64))
    np.save(shard_dir / "phoneme_ids.npy", np.array([1, 2], dtype=np.int64))
    _write_manifest(
        dataset_root,
        [
            {
                "example_id": f"{dataset}-0",
                "session_id": session_id,
                "subject_id": session_id.split(".")[0],
                "shard_relpath": f"{dataset}/shards/shard_000",
                "example_index": 0,
                "n_time_bins": 3,
                "has_tx": True,
                "has_sbp": True,
                "n_tx_features": tx_dim,
                "n_sbp_features": sbp_dim,
            },
            {
                "example_id": f"{dataset}-1",
                "session_id": session_id.replace("01", "02"),
                "subject_id": session_id.split(".")[0],
                "shard_relpath": f"{dataset}/shards/shard_000",
                "example_index": 1,
                "n_time_bins": 3,
                "has_tx": True,
                "has_sbp": True,
                "n_tx_features": tx_dim,
                "n_sbp_features": sbp_dim,
            },
        ],
    )
    _write_metadata(
        dataset_root,
        {
            "dataset_family": dataset,
            "modalities": ["tx", "sbp"],
            "feature_layout": {
                "n_total_features": int(tx_dim + sbp_dim),
                "n_tx_features": int(tx_dim),
                "n_sbp_features": int(sbp_dim),
            },
            "shards": [
                {
                    "shard_id": "shard_000",
                    "shard_relpath": f"{dataset}/shards/shard_000",
                    "n_tx_features": int(tx_dim),
                    "n_sbp_features": int(sbp_dim),
                }
            ],
            "full_width": bool(full_width),
        },
    )


def _write_motor_data_dataset(cache_root: Path) -> None:
    dataset_root = cache_root / "motor_data"
    shard_root = dataset_root / "shards"
    shard_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    shards_meta: list[dict[str, object]] = []
    for shard_id, width in (("shard_000", 128), ("shard_001", 256)):
        shard_dir = shard_root / shard_id
        shard_dir.mkdir(parents=True, exist_ok=True)
        tx = np.arange(6 * width, dtype=np.float32).reshape(6, width)
        np.save(shard_dir / "tx.npy", tx)
        np.save(shard_dir / "time_offsets.npy", np.array([0, 3, 6], dtype=np.int64))
        for example_index in range(2):
            rows.append(
                {
                    "example_id": f"motor-{shard_id}-{example_index}",
                    "session_id": f"motor.2025.01.0{example_index + 1}",
                    "subject_id": "motor",
                    "shard_relpath": f"motor_data/shards/{shard_id}",
                    "example_index": example_index,
                    "n_time_bins": 3,
                    "has_tx": True,
                    "has_sbp": False,
                    "n_tx_features": width,
                    "n_sbp_features": 0,
                }
            )
        shards_meta.append(
            {
                "shard_id": shard_id,
                "shard_relpath": f"motor_data/shards/{shard_id}",
                "n_tx_features": width,
                "n_sbp_features": 0,
            }
        )
    _write_manifest(dataset_root, rows)
    _write_metadata(
        dataset_root,
        {
            "dataset_family": "motor_data",
            "modalities": ["tx"],
            "feature_layout": {
                "n_total_features": 256,
                "n_tx_features": 256,
                "n_sbp_features": 0,
            },
            "shards": shards_meta,
        },
    )


def _write_tx_only_dataset(
    cache_root: Path,
    *,
    dataset: str,
    tx_dim: int = 192,
) -> None:
    dataset_root = cache_root / dataset
    shard_dir = dataset_root / "shards" / "shard_000"
    shard_dir.mkdir(parents=True, exist_ok=True)
    tx = np.arange(6 * tx_dim, dtype=np.float32).reshape(6, tx_dim)
    np.save(shard_dir / "tx.npy", tx)
    np.save(shard_dir / "time_offsets.npy", np.array([0, 3, 6], dtype=np.int64))
    _write_manifest(
        dataset_root,
        [
            {
                "example_id": f"{dataset}-0",
                "session_id": f"{dataset}.2025.01.01",
                "subject_id": dataset,
                "shard_relpath": f"{dataset}/shards/shard_000",
                "example_index": 0,
                "n_time_bins": 3,
                "has_tx": True,
                "has_sbp": False,
                "n_tx_features": tx_dim,
                "n_sbp_features": 0,
            },
            {
                "example_id": f"{dataset}-1",
                "session_id": f"{dataset}.2025.01.02",
                "subject_id": dataset,
                "shard_relpath": f"{dataset}/shards/shard_000",
                "example_index": 1,
                "n_time_bins": 3,
                "has_tx": True,
                "has_sbp": False,
                "n_tx_features": tx_dim,
                "n_sbp_features": 0,
            },
        ],
    )
    _write_metadata(
        dataset_root,
        {
            "dataset_family": dataset,
            "modalities": ["tx"],
            "feature_layout": {
                "n_tx_features": tx_dim,
                "n_sbp_features": 0,
                "n_total_features": tx_dim,
            },
            "shards": [
                {
                    "shard_id": "shard_000",
                    "shard_relpath": f"{dataset}/shards/shard_000",
                    "n_tx_features": tx_dim,
                    "n_sbp_features": 0,
                }
            ],
        },
    )


def _write_full_bit_cache(cache_root: Path) -> None:
    _write_tx_sbp_dataset(cache_root, dataset="brain2text24", tx_dim=128, sbp_dim=128)
    _write_tx_sbp_dataset(cache_root, dataset="brain2text25", tx_dim=256, sbp_dim=256, full_width=True)
    _write_motor_data_dataset(cache_root)
    for dataset in (
        "000950",
        "plug_n_play",
        "unsupervised_cursor_recalibration_offline",
        "unsupervised_cursor_recalibration_online",
        "willett_handwriting",
    ):
        _write_tx_only_dataset(cache_root, dataset=dataset)


class BitCacheContractTests(unittest.TestCase):
    def test_harmonize_bit_cache_trims_braintotext25_and_pads_motor_data_tx_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_root = Path(tmpdir) / "cache_v1"
            _write_full_bit_cache(cache_root)

            summary = harmonize_bit_cache(cache_root)

            self.assertEqual(summary["feature_policy"], BIT_CANONICAL_FEATURE_POLICY)
            b2t25_tx = np.load(cache_root / "brain2text25/shards/shard_000/tx.npy")
            b2t25_sbp = np.load(cache_root / "brain2text25/shards/shard_000/sbp.npy")
            self.assertEqual(b2t25_tx.shape[1], 128)
            self.assertEqual(b2t25_sbp.shape[1], 128)

            motor_tx_128 = np.load(cache_root / "motor_data/shards/shard_000/tx.npy")
            motor_tx_256 = np.load(cache_root / "motor_data/shards/shard_001/tx.npy")
            self.assertEqual(motor_tx_128.shape[1], BIT_STAGE1_TX_DIM)
            self.assertEqual(motor_tx_256.shape[1], BIT_STAGE1_TX_DIM)

            motor_rows = [
                json.loads(line)
                for line in (cache_root / "motor_data/manifest.jsonl").read_text().splitlines()
            ]
            self.assertEqual({row["n_tx_features"] for row in motor_rows}, {BIT_STAGE1_TX_DIM})
            self.assertEqual({row["n_sbp_features"] for row in motor_rows}, {0})

            b2t25_metadata = json.loads((cache_root / "brain2text25/metadata.json").read_text())
            self.assertTrue(b2t25_metadata["area6v_migration"]["area6v_only"])
            self.assertFalse(b2t25_metadata["canonical_cache_policy"]["stage1_default_included"])

            included_metadata = json.loads((cache_root / "000950/metadata.json").read_text())
            self.assertTrue(included_metadata["canonical_cache_policy"]["stage1_default_included"])

    def test_prepare_bit_cache_uses_tx_only_defaults_and_special_stats_stem(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_root = Path(tmpdir) / "cache_v1"
            _write_full_bit_cache(cache_root)

            summary = prepare_bit_cache(
                cache_root=cache_root,
                dry_run=True,
            )

            bit_prep = summary["bit_prep"]
            self.assertEqual(bit_prep["recommended_ssl_feature_mode"], BIT_STAGE1_FEATURE_MODE)
            self.assertEqual(bit_prep["recommended_boundary_key_mode"], "session")
            self.assertEqual(bit_prep["recommended_tx_dim"], BIT_STAGE1_TX_DIM)
            self.assertEqual(bit_prep["excluded_dataset_names"], list(BIT_STAGE1_DEFAULT_EXCLUDED_DATASETS))
            self.assertNotIn("brain2text25", bit_prep["dataset_names"])
            self.assertTrue(
                str(bit_prep["recommended_stats_output_path"]).endswith(
                    f"/tx_only/session/{BIT_STAGE1_DEFAULT_STATS_STEM}.pt"
                )
            )

    def test_prepare_cache_context_discovers_bit_stage1_stats_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_root = Path(tmpdir) / "cache_v1_smoothed_sigma2p0"
            _write_full_bit_cache(cache_root)
            harmonize_bit_cache(cache_root)
            stats_path = (
                Path(tmpdir)
                / "stats"
                / "session_feature_stats"
                / "smoothed_sigma2p0"
                / "tx_only"
                / "session"
                / f"{BIT_STAGE1_DEFAULT_STATS_STEM}.pt"
            )
            stats_entries = {
                f"{dataset}:{dataset}.2025.01.01": (
                    torch.zeros(BIT_STAGE1_TX_DIM),
                    torch.ones(BIT_STAGE1_TX_DIM),
                )
                for dataset in BIT_STAGE1_DEFAULT_INCLUDED_DATASETS
            }
            stats_entries.update(
                {
                    f"{dataset}:{dataset}.2025.01.02": (
                        torch.zeros(BIT_STAGE1_TX_DIM),
                        torch.ones(BIT_STAGE1_TX_DIM),
                    )
                    for dataset in BIT_STAGE1_DEFAULT_INCLUDED_DATASETS
                }
            )
            write_valid_session_stats_artifact(
                cache_root=cache_root,
                stats_path=stats_path,
                stats_entries=stats_entries,
                feature_mode="tx_only",
                boundary_key_mode="session",
                tx_dim=BIT_STAGE1_TX_DIM,
                sbp_dim=BIT_STAGE1_SBP_DIM,
                excluded_datasets=BIT_STAGE1_DEFAULT_EXCLUDED_DATASETS,
            )

            context = prepare_cache_context(
                cache_candidates=[cache_root],
                config=CacheAccessConfig(
                    mode="drive_direct",
                    excluded_datasets=BIT_STAGE1_DEFAULT_EXCLUDED_DATASETS,
                    feature_mode="tx_only",
                    tx_dim=BIT_STAGE1_TX_DIM,
                    sbp_dim=BIT_STAGE1_SBP_DIM,
                    use_normalization=True,
                ),
            )

            self.assertEqual(
                sorted(context.pretrain_datasets),
                sorted(BIT_STAGE1_DEFAULT_INCLUDED_DATASETS),
            )
            self.assertEqual(
                sorted(context.session_feature_stats),
                sorted(stats_entries),
            )


if __name__ == "__main__":
    unittest.main()

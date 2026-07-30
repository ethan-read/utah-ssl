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
    BIT_STAGE1_DATASET_SPLITS,
    BIT_STAGE1_DATASETS,
    BIT_STAGE1_TX_DIM,
)
from ssl_core.experiment_contract import DatasetPlan, SignalSpec
from masked_ssl.cache import (
    CacheAccessConfig,
    prepare_cache_context,
    resolve_precomputed_session_stats_path,
)
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
                "source_split": (
                    "competition_train"
                    if dataset == "brain2text24"
                    else "train"
                ),
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
                "source_split": (
                    "competition_train"
                    if dataset == "brain2text24"
                    else "val"
                ),
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
                    "source_split": "none",
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
                "source_split": BIT_STAGE1_DATASET_SPLITS[dataset][0],
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
                "source_split": BIT_STAGE1_DATASET_SPLITS[dataset][0],
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
    def test_prepare_bit_cache_rejects_source_as_destination(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_root = Path(tmpdir) / "cache_v1"
            _write_full_bit_cache(cache_root)
            with self.assertRaisesRegex(ValueError, "destination separate"):
                prepare_bit_cache(
                    cache_root=cache_root,
                    output_root=cache_root,
                    dry_run=True,
                )

    def test_prepare_bit_cache_uses_exact_named_plan_without_mutating_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_root = Path(tmpdir) / "cache_v1"
            _write_full_bit_cache(cache_root)
            b2t25_tx_before = np.load(
                cache_root / "brain2text25/shards/shard_000/tx.npy"
            ).copy()

            summary = prepare_bit_cache(
                cache_root=cache_root,
                dry_run=True,
            )

            bit_prep = summary["bit_prep"]
            self.assertEqual(
                bit_prep["dataset_plan"],
                {
                    dataset: list(source_splits)
                    for dataset, source_splits in BIT_STAGE1_DATASET_SPLITS.items()
                },
            )
            self.assertEqual(bit_prep["signal_spec"]["mode"], "tx_only")
            self.assertEqual(bit_prep["signal_spec"]["tx_dim"], BIT_STAGE1_TX_DIM)
            self.assertEqual(bit_prep["boundary_key_mode"], "session")
            self.assertNotIn("brain2text25", bit_prep["dataset_plan"])
            np.testing.assert_array_equal(
                np.load(cache_root / "brain2text25/shards/shard_000/tx.npy"),
                b2t25_tx_before,
            )
            stats_path = str(bit_prep["stats_output_path"])
            self.assertIn("/tx_only/session/ssl_pretrain_", stats_path)
            self.assertTrue(stats_path.endswith("_v2.pt"))

    def test_prepare_cache_context_discovers_bit_stage1_stats_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_root = Path(tmpdir) / "cache_v1_smoothed_sigma2p0"
            _write_full_bit_cache(cache_root)
            dataset_plan = DatasetPlan.from_mapping(
                BIT_STAGE1_DATASET_SPLITS
            )
            signal_spec = SignalSpec.tx_only(
                tx_dim=BIT_STAGE1_TX_DIM,
                missing_channel_policy="zero_pad",
            )
            stats_path = resolve_precomputed_session_stats_path(
                cache_root=cache_root,
                signal_spec=signal_spec,
                dataset_plan=dataset_plan,
                boundary_key_mode="session",
            )
            stats_entries = {
                f"{dataset}:{dataset}.2025.01.01": (
                    torch.zeros(BIT_STAGE1_TX_DIM),
                    torch.ones(BIT_STAGE1_TX_DIM),
                )
                for dataset in BIT_STAGE1_DATASETS
            }
            stats_entries.update(
                {
                    f"{dataset}:{dataset}.2025.01.02": (
                        torch.zeros(BIT_STAGE1_TX_DIM),
                        torch.ones(BIT_STAGE1_TX_DIM),
                    )
                    for dataset in BIT_STAGE1_DATASETS
                }
            )
            write_valid_session_stats_artifact(
                cache_root=cache_root,
                stats_path=stats_path,
                stats_entries=stats_entries,
                signal_spec=signal_spec,
                dataset_plan=dataset_plan,
                boundary_key_mode="session",
            )

            context = prepare_cache_context(
                cache_candidates=[cache_root],
                config=CacheAccessConfig(
                    dataset_plan={
                        dataset: source_splits
                        for dataset, source_splits in BIT_STAGE1_DATASET_SPLITS.items()
                    },
                    signal_spec=signal_spec,
                    mode="drive_direct",
                    use_normalization=True,
                    precomputed_session_stats_path=stats_path,
                ),
            )

            self.assertEqual(
                sorted(context.pretrain_datasets),
                sorted(BIT_STAGE1_DATASETS),
            )
            self.assertEqual(
                sorted(context.session_feature_stats),
                sorted(stats_entries),
            )


if __name__ == "__main__":
    unittest.main()

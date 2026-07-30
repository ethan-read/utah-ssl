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

from masked_ssl.cache import (  # noqa: E402
    CacheAccessConfig,
    ShardStore,
    _compute_dataset_cache_source_signature,
    prepare_cache_context,
)
from ssl_core.experiment_contract import DatasetPlan, SignalSpec  # noqa: E402
from ssl_core.scripts.prepare_possm_pooled_cache import (  # noqa: E402
    SUMMARY_NAME,
    prepare_possm_pooled_caches,
)
from ssl_core.scripts.recompute_session_feature_stats import (  # noqa: E402
    recompute_session_feature_stats,
)
from ssl_core.scripts.repack_cache_shards import repack_cache_root  # noqa: E402


def _write_dataset(
    root: Path,
    *,
    dataset: str,
    width: int,
    value_offset: float,
) -> None:
    dataset_root = root / dataset
    shard_root = dataset_root / "shards"
    rows = []
    shards = []
    for shard_idx in range(2):
        shard_id = f"shard_{shard_idx:03d}"
        shard_dir = shard_root / shard_id
        shard_dir.mkdir(parents=True, exist_ok=True)
        tx = (
            np.arange(8 * width, dtype=np.float32).reshape(8, width) / 100.0
            + value_offset
            + shard_idx * 3.0
        )
        sbp = tx + 50.0
        np.save(shard_dir / "tx.npy", tx)
        np.save(shard_dir / "sbp.npy", sbp)
        np.save(shard_dir / "time_offsets.npy", np.array([0, 4, 8], dtype=np.int64))
        np.save(
            shard_dir / "phoneme_offsets.npy",
            np.array([0, 2, 4], dtype=np.int64),
        )
        np.save(
            shard_dir / "phoneme_ids.npy",
            np.array([1, 2, 3, 4], dtype=np.int64) + shard_idx,
        )
        shards.append(
            {
                "shard_id": shard_id,
                "shard_relpath": f"{dataset}/shards/{shard_id}",
                "n_tx_features": width,
                "n_sbp_features": width,
            }
        )
        for example_index in range(2):
            source_split = (
                "competition_train"
                if dataset == "brain2text24"
                else ("train" if example_index == 0 else "val")
            )
            rows.append(
                {
                    "example_id": f"{dataset}-{shard_idx}-{example_index}",
                    "session_id": f"{dataset}.2025.01.0{shard_idx + 1}",
                    "subject_id": dataset,
                    "source_split": source_split,
                    "shard_id": shard_id,
                    "shard_relpath": f"{dataset}/shards/{shard_id}",
                    "example_index": example_index,
                    "n_time_bins": 4,
                    "has_tx": True,
                    "has_sbp": True,
                    "n_tx_features": width,
                    "n_sbp_features": width,
                    "n_total_features": width * 2,
                }
            )
    with (dataset_root / "manifest.jsonl").open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    (dataset_root / "metadata.json").write_text(
        json.dumps(
            {
                "dataset_family": dataset,
                "modalities": ["tx", "sbp"],
                "feature_layout": {
                    "n_tx_features": width,
                    "n_sbp_features": width,
                    "n_total_features": width * 2,
                },
                "shards": shards,
            },
            indent=2,
        )
        + "\n"
    )


def _write_source_root(root: Path, *, value_offset: float = 0.0) -> None:
    _write_dataset(
        root,
        dataset="brain2text24",
        width=128,
        value_offset=value_offset,
    )
    _write_dataset(
        root,
        dataset="brain2text25",
        width=256,
        value_offset=value_offset + 0.125,
    )


class POSSMCachePreparationTests(unittest.TestCase):
    def test_versioned_build_projects_and_validates_without_mutating_sources(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw_source = root / "cache_v1"
            smoothed_source = root / "cache_v1_smoothed_sigma2p0"
            raw_destination = root / "cache_v1_possm_area6v"
            smoothed_destination = root / "cache_v1_possm_area6v_sigma2p0"
            _write_source_root(raw_source)
            _write_source_root(smoothed_source, value_offset=0.25)
            source_manifest_before = (
                raw_source / "brain2text25/manifest.jsonl"
            ).read_bytes()
            source_tx_before = np.load(
                raw_source / "brain2text25/shards/shard_000/tx.npy"
            ).copy()

            summary = prepare_possm_pooled_caches(
                raw_source_root=raw_source,
                smoothed_source_root=smoothed_source,
                raw_destination_root=raw_destination,
                smoothed_destination_root=smoothed_destination,
                target_mb=1.0,
            )

            self.assertEqual(summary["raw_result"]["status"], "built")
            self.assertTrue((raw_destination / SUMMARY_NAME).exists())
            self.assertTrue((smoothed_destination / SUMMARY_NAME).exists())
            self.assertFalse((raw_destination / "brain2text24").exists())
            self.assertFalse((smoothed_destination / "brain2text24").exists())
            self.assertFalse(raw_destination.with_name(raw_destination.name + ".partial").exists())
            self.assertEqual(
                source_manifest_before,
                (raw_source / "brain2text25/manifest.jsonl").read_bytes(),
            )
            np.testing.assert_array_equal(
                source_tx_before,
                np.load(raw_source / "brain2text25/shards/shard_000/tx.npy"),
            )

            destination_rows = [
                json.loads(line)
                for line in (raw_destination / "brain2text25/manifest.jsonl")
                .read_text()
                .splitlines()
            ]
            self.assertEqual({row["n_tx_features"] for row in destination_rows}, {128})
            self.assertEqual({row["n_sbp_features"] for row in destination_rows}, {128})
            destination_tx = np.load(
                raw_destination / destination_rows[0]["shard_relpath"] / "tx.npy"
            )
            self.assertEqual(destination_tx.shape[1], 128)
            self.assertEqual(destination_tx.dtype, source_tx_before.dtype)
            np.testing.assert_array_equal(
                source_tx_before[:4, :128],
                destination_tx[:4],
            )
            destination_sbp = np.load(
                raw_destination / destination_rows[0]["shard_relpath"] / "sbp.npy"
            )
            self.assertEqual(destination_sbp.dtype, np.dtype(np.float32))
            np.testing.assert_array_equal(
                np.load(raw_source / "brain2text25/shards/shard_000/sbp.npy")[
                    :4, :128
                ],
                destination_sbp[:4],
            )
            destination_metadata = json.loads(
                (raw_destination / "brain2text25/metadata.json").read_text()
            )
            self.assertEqual(
                destination_metadata["feature_layout"]["tx_slice"],
                [0, 128],
            )
            self.assertEqual(
                destination_metadata["feature_layout"]["sbp_slice"],
                [128, 256],
            )
            completion = json.loads(
                (raw_destination / SUMMARY_NAME).read_text()
            )
            self.assertEqual(
                completion["tx_storage_policy"],
                "preserve_source_dtype_exactly",
            )
            self.assertEqual(
                completion["repack"]["dst_root"],
                str(raw_destination),
            )
            self.assertEqual(
                completion["validation"]["destination_root"],
                str(raw_destination),
            )
            self.assertEqual(
                completion["validation"]["datasets"]["brain2text25"][
                    "storage_policy"
                ],
                "preserve_projected_source_dtypes_and_values",
            )

            with self.assertRaises(FileExistsError):
                prepare_possm_pooled_caches(
                    raw_source_root=raw_source,
                    smoothed_source_root=smoothed_source,
                    raw_destination_root=raw_destination,
                    smoothed_destination_root=smoothed_destination,
                    target_mb=1.0,
                )
            resumed = prepare_possm_pooled_caches(
                raw_source_root=raw_source,
                smoothed_source_root=smoothed_source,
                raw_destination_root=raw_destination,
                smoothed_destination_root=smoothed_destination,
                target_mb=1.0,
                resume_completed=True,
            )
            self.assertEqual(resumed["raw_result"]["status"], "reused_completed")

    def test_dry_run_and_partial_recovery_are_non_destructive(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw_source = root / "raw"
            smoothed_source = root / "smooth"
            raw_destination = root / "raw_dst"
            smoothed_destination = root / "smooth_dst"
            _write_source_root(raw_source)
            _write_source_root(smoothed_source, value_offset=0.5)

            dry_run = prepare_possm_pooled_caches(
                raw_source_root=raw_source,
                smoothed_source_root=smoothed_source,
                raw_destination_root=raw_destination,
                smoothed_destination_root=smoothed_destination,
                dry_run=True,
            )
            self.assertTrue(dry_run["dry_run"])
            self.assertFalse(raw_destination.exists())

            partial = raw_destination.with_name(raw_destination.name + ".partial")
            partial.mkdir()
            with self.assertRaises(FileExistsError):
                prepare_possm_pooled_caches(
                    raw_source_root=raw_source,
                    smoothed_source_root=smoothed_source,
                    raw_destination_root=raw_destination,
                    smoothed_destination_root=smoothed_destination,
                    target_mb=1.0,
                )
            recovered = prepare_possm_pooled_caches(
                raw_source_root=raw_source,
                smoothed_source_root=smoothed_source,
                raw_destination_root=raw_destination,
                smoothed_destination_root=smoothed_destination,
                target_mb=1.0,
                replace_partial=True,
            )
            self.assertEqual(recovered["raw_result"]["status"], "built")

    def test_area6v_projection_rejects_unexpected_width(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "source"
            destination = root / "destination"
            _write_dataset(
                source,
                dataset="brain2text25",
                width=192,
                value_offset=0.0,
            )
            with self.assertRaisesRegex(ValueError, "expected 128 or 256"):
                repack_cache_root(
                    src_root=source,
                    dst_root=destination,
                    repack_datasets=["brain2text25"],
                    copy_datasets=[],
                    target_mb=1.0,
                    area6v_datasets=["brain2text25"],
                )

    def test_float16_tx_conversion_rejects_overflow(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "source"
            destination = root / "destination"
            _write_dataset(
                source,
                dataset="brain2text25",
                width=256,
                value_offset=100000.0,
            )
            with self.assertRaisesRegex(ValueError, "conversion overflowed"):
                repack_cache_root(
                    src_root=source,
                    dst_root=destination,
                    repack_datasets=["brain2text25"],
                    copy_datasets=[],
                    target_mb=1.0,
                    area6v_datasets=["brain2text25"],
                    tx_float16_datasets=["brain2text25"],
                )


class ShardStoreModalityTests(unittest.TestCase):
    def _write_shard(self, root: Path, shard_name: str, offset: float) -> str:
        shard_dir = root / "brain2text25" / "shards" / shard_name
        shard_dir.mkdir(parents=True, exist_ok=True)
        np.save(shard_dir / "time_offsets.npy", np.array([0, 4], dtype=np.int64))
        np.save(shard_dir / "tx.npy", np.ones((4, 128), dtype=np.float32) + offset)
        np.save(shard_dir / "sbp.npy", np.ones((4, 128), dtype=np.float32) + 100 + offset)
        return f"brain2text25/shards/{shard_name}"

    def test_tx_only_store_skips_sbp_and_reports_cache_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            shard = self._write_shard(root, "shard_000", 0.0)
            store = ShardStore(root, ram_cache_gb=0.01, modalities=("tx",))

            loaded = store.get(shard)
            self.assertIsNone(loaded["sbp"])
            first_summary = store.summary()
            self.assertEqual(first_summary["modalities"], ["tx"])
            self.assertEqual(first_summary["cache_misses"], 1)
            self.assertEqual(first_summary["cache_hits"], 0)

            store.get(shard)
            second_summary = store.summary()
            self.assertEqual(second_summary["cache_hits"], 1)
            self.assertAlmostEqual(second_summary["cache_hit_rate"], 0.5)

    def test_tx_sbp_store_loads_sbp_and_tracks_eviction(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            first = self._write_shard(root, "shard_000", 0.0)
            second = self._write_shard(root, "shard_001", 1.0)
            store = ShardStore(root, ram_cache_gb=0.000005, modalities=("tx", "sbp"))

            self.assertIsNotNone(store.get(first)["sbp"])
            self.assertIsNotNone(store.get(second)["sbp"])
            summary = store.summary()
            self.assertEqual(summary["modalities"], ["tx", "sbp"])
            self.assertGreaterEqual(summary["evictions"], 1)
            self.assertGreater(summary["bytes_read"], 0)

    def test_requested_missing_modality_fails_instead_of_returning_none(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            shard = self._write_shard(root, "shard_000", 0.0)
            (root / shard / "sbp.npy").unlink()
            store = ShardStore(root, ram_cache_gb=0.01, modalities=("sbp",))

            with self.assertRaisesRegex(FileNotFoundError, "Requested modality 'sbp'"):
                store.get(shard)

    def test_sbp_only_context_does_not_load_tx(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            primary = root / "canonical"
            _write_source_root(primary, value_offset=0.0)

            context = prepare_cache_context(
                cache_candidates=[primary],
                config=CacheAccessConfig(
                    dataset_plan={"brain2text24": ()},
                    signal_spec=SignalSpec.sbp_only(sbp_dim=128),
                    mode="drive_direct",
                    use_normalization=False,
                ),
            )

            shard = context.shard_store.get(
                context.rows_by_dataset["brain2text24"][0].shard_relpath
            )
            self.assertIsNone(shard["tx"])
            self.assertIsNotNone(shard["sbp"])
            self.assertEqual(context.full_dim, 128)
            self.assertEqual(context.shard_store.summary()["modalities"], ["sbp"])

    def test_mixed_root_context_keeps_primary_b2t24_and_overrides_b2t25(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            primary = root / "canonical"
            optimized = root / "optimized"
            _write_source_root(primary, value_offset=0.0)
            _write_dataset(
                primary,
                dataset="unrelated_dataset",
                width=128,
                value_offset=500.0,
            )
            _write_dataset(
                optimized,
                dataset="brain2text25",
                width=128,
                value_offset=1000.0,
            )

            context = prepare_cache_context(
                cache_candidates=[primary],
                config=CacheAccessConfig(
                    dataset_plan={
                        "brain2text24": (),
                        "brain2text25": (),
                    },
                    signal_spec=SignalSpec.tx_only(tx_dim=128),
                    mode="drive_direct",
                    use_normalization=False,
                    dataset_cache_roots={"brain2text25": optimized},
                ),
            )

            self.assertEqual(context.pretrain_datasets, ["brain2text24", "brain2text25"])
            self.assertIn("unrelated_dataset", context.available_datasets)
            self.assertNotIn("unrelated_dataset", context.rows_by_dataset)
            self.assertEqual(context.drive_dataset_cache_roots["brain2text24"], primary)
            self.assertEqual(context.drive_dataset_cache_roots["brain2text25"], optimized)
            b2t24_shard = context.shard_store.get(
                context.rows_by_dataset["brain2text24"][0].shard_relpath
            )
            b2t25_shard = context.shard_store.get(
                context.rows_by_dataset["brain2text25"][0].shard_relpath
            )
            self.assertLess(float(np.asarray(b2t24_shard["tx"])[0, 0]), 10.0)
            self.assertGreater(float(np.asarray(b2t25_shard["tx"])[0, 0]), 999.0)
            self.assertEqual(
                context.source_cache_signature,
                _compute_dataset_cache_source_signature(
                    {
                        "brain2text24": primary,
                        "brain2text25": optimized,
                    }
                ),
            )

    def test_mixed_root_session_stats_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            primary = root / "canonical"
            optimized = root / "optimized"
            _write_source_root(primary, value_offset=0.0)
            _write_dataset(
                optimized,
                dataset="brain2text25",
                width=128,
                value_offset=1000.0,
            )
            split_policy = {
                "brain2text24": ("competition_train",),
                "brain2text25": ("train", "val"),
            }

            for feature_mode in ("tx_only", "sbp_only"):
                stats_path = root / f"mixed_stats_{feature_mode}.pt"
                recompute_session_feature_stats(
                    cache_root=primary,
                    output_path=stats_path,
                    signal_spec=SignalSpec.from_mode(
                        feature_mode, tx_dim=128, sbp_dim=128
                    ),
                    dataset_plan=DatasetPlan.from_mapping(split_policy),
                    dataset_cache_roots={"brain2text25": optimized},
                )
                stats_metadata = json.loads(stats_path.with_suffix(".json").read_text())
                self.assertEqual(
                    stats_metadata["signal_spec"],
                    SignalSpec.from_mode(
                        feature_mode, tx_dim=128, sbp_dim=128
                    ).to_dict(),
                )
                context = prepare_cache_context(
                    cache_candidates=[primary],
                    config=CacheAccessConfig(
                        dataset_plan=DatasetPlan.from_mapping(split_policy),
                        signal_spec=SignalSpec.from_mode(
                            feature_mode, tx_dim=128, sbp_dim=128
                        ),
                        mode="drive_direct",
                        use_normalization=True,
                        segment_bins=4,
                        examples_per_shard=1,
                        precomputed_session_stats_path=stats_path,
                        dataset_cache_roots={"brain2text25": optimized},
                    ),
                )
                self.assertGreater(len(context.session_feature_stats), 0)
                self.assertEqual(context.full_dim, 128)


if __name__ == "__main__":
    unittest.main()

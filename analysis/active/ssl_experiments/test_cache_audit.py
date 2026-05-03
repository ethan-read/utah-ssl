from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENTS_DIR = REPO_ROOT / "analysis" / "active" / "ssl_experiments"
for path in (REPO_ROOT, EXPERIMENTS_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from audit_cache_roots import run_audit


def _write_dataset(
    cache_root: Path,
    dataset_name: str,
    *,
    lengths_by_session: dict[str, list[int]],
    tx_dim: int = 256,
    sbp_dim: int = 256,
    include_metadata: bool = True,
    smoothing_sigma: float | None = None,
    tx_offset: float = 0.0,
) -> None:
    dataset_root = cache_root / dataset_name
    shard_root = dataset_root / "shards"
    shard_root.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict[str, object]] = []
    metadata_shards: list[dict[str, object]] = []
    global_example_id = 0

    for shard_idx, (session_id, lengths) in enumerate(lengths_by_session.items()):
        shard_id = f"shard_{shard_idx:02d}_{session_id.replace('.', '_')}"
        shard_dir = shard_root / shard_id
        shard_dir.mkdir(parents=True, exist_ok=True)

        time_offsets = [0]
        total_rows = 0
        for length in lengths:
            total_rows += int(length)
            time_offsets.append(total_rows)
        time_offsets_arr = np.asarray(time_offsets, dtype=np.int64)
        tx = (np.arange(total_rows * tx_dim, dtype=np.float32).reshape(total_rows, tx_dim) + tx_offset)
        sbp = (np.arange(total_rows * sbp_dim, dtype=np.float32).reshape(total_rows, sbp_dim) + 10.0 + tx_offset)
        np.save(shard_dir / "time_offsets.npy", time_offsets_arr)
        np.save(shard_dir / "tx.npy", tx)
        np.save(shard_dir / "sbp.npy", sbp)

        for example_index, length in enumerate(lengths):
            manifest_rows.append(
                {
                    "example_id": f"ex-{global_example_id}",
                    "session_id": session_id,
                    "subject_id": session_id.split(".")[0],
                    "session_date": "2025.01.01",
                    "source_split": "train",
                    "shard_relpath": f"{dataset_name}/shards/{shard_id}",
                    "example_index": example_index,
                    "n_time_bins": int(length),
                    "n_tx_features": tx_dim,
                    "n_sbp_features": sbp_dim,
                    "has_tx": True,
                    "has_sbp": True,
                }
            )
            global_example_id += 1

        metadata_shards.append({"shard_id": shard_id, "n_examples": len(lengths)})

    with (dataset_root / "manifest.jsonl").open("w") as handle:
        for row in manifest_rows:
            handle.write(json.dumps(row) + "\n")

    if include_metadata:
        metadata = {
            "dataset": dataset_name,
            "modalities": ["tx", "sbp"],
            "feature_layout": {
                "n_tx_features": tx_dim,
                "n_sbp_features": sbp_dim,
            },
            "shards": metadata_shards,
        }
        if smoothing_sigma is not None:
            metadata["smoothing_provenance"] = {
                "source_cache_root": str(cache_root),
                "sigma_bins": float(smoothing_sigma),
                "implementation": "test",
            }
        (dataset_root / "metadata.json").write_text(json.dumps(metadata, indent=2))


def _write_stats(
    path: Path,
    *,
    session_ids: list[str],
    dim: int,
    sigma: float | None = None,
    per_session_dims: dict[str, int] | None = None,
) -> None:
    payload = {
        "session_feature_stats": {
            f"brain2text24:{session_id}": (
                torch.zeros(per_session_dims.get(session_id, dim) if per_session_dims else dim),
                torch.ones(per_session_dims.get(session_id, dim) if per_session_dims else dim),
            )
            for session_id in session_ids
        },
        "metadata": {
            "gaussian_smoothing_sigma_bins": sigma,
            "session_stats_bin_stride": 2,
        },
    }
    torch.save(payload, path)


class CacheAuditTests(unittest.TestCase):
    def test_matching_roots_compare_equal(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            raw_a = tmp / "cache_a"
            raw_b = tmp / "cache_b"
            _write_dataset(raw_a, "brain2text24", lengths_by_session={"t00.2025.01.01": [120], "t00.2025.01.02": [160]})
            _write_dataset(raw_b, "brain2text24", lengths_by_session={"t00.2025.01.01": [120], "t00.2025.01.02": [160]})

            report = run_audit(cache_roots=[raw_a, raw_b], dataset="brain2text24", segment_bins=80)
            comparison = report["root_comparisons"][0]["datasets"]["brain2text24"]
            self.assertTrue(comparison["manifest_sha256_match"])
            self.assertTrue(comparison["metadata_sha256_match"])

    def test_missing_metadata_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            raw_root = tmp / "cache"
            _write_dataset(
                raw_root,
                "brain2text24",
                lengths_by_session={"t00.2025.01.01": [120], "t00.2025.01.02": [160]},
                include_metadata=False,
            )
            report = run_audit(cache_roots=[raw_root], dataset="brain2text24", segment_bins=80)
            findings = report["root_audits"][0]["datasets"]["brain2text24"]["findings"]
            self.assertIn("missing_metadata_json", findings)

    def test_short_manifest_is_flagged_for_segment_length(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            raw_root = tmp / "cache"
            _write_dataset(raw_root, "brain2text24", lengths_by_session={"t00.2025.01.01": [20], "t00.2025.01.02": [30]})
            report = run_audit(cache_roots=[raw_root], dataset="brain2text24", segment_bins=80)
            ds = report["root_audits"][0]["datasets"]["brain2text24"]
            self.assertEqual(ds["manifest_summary"]["rows_supporting_segment_bins"], 0)
            self.assertIn("no_manifest_rows_support_segment_bins_80", ds["findings"])
            self.assertFalse(ds["feature_mode_audits"]["tx_only"]["train_sampling_ok"])

    def test_256_dim_stats_flagged_incompatible_with_tx_sbp(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            raw_root = tmp / "cache"
            _write_dataset(raw_root, "brain2text24", lengths_by_session={"t00.2025.01.01": [120], "t00.2025.01.02": [160]})
            stats_path = tmp / "stats.pt"
            _write_stats(stats_path, session_ids=["t00.2025.01.01", "t00.2025.01.02"], dim=256)
            report = run_audit(cache_roots=[raw_root], stats_paths=[stats_path], dataset="brain2text24", segment_bins=80)
            stats = report["stats_audits"][0]
            self.assertTrue(stats["compatible_feature_modes"]["tx_only"])
            self.assertFalse(stats["compatible_feature_modes"]["tx_sbp"])
            self.assertIn("tx_only_only_not_tx_sbp", stats["findings"])

    def test_matching_stats_are_marked_usable(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            raw_root = tmp / "cache"
            _write_dataset(raw_root, "brain2text24", lengths_by_session={"t00.2025.01.01": [120], "t00.2025.01.02": [160]})
            stats_path = tmp / "stats.pt"
            _write_stats(stats_path, session_ids=["t00.2025.01.01", "t00.2025.01.02"], dim=512, sigma=2.0)
            report = run_audit(cache_roots=[raw_root], stats_paths=[stats_path], dataset="brain2text24", segment_bins=80)
            stats = report["stats_audits"][0]
            self.assertTrue(stats["compatible_feature_modes"]["tx_only"])
            self.assertTrue(stats["compatible_feature_modes"]["tx_sbp"])

    def test_mixed_dimension_stats_are_not_marked_tx_sbp_compatible(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            raw_root = tmp / "cache"
            _write_dataset(raw_root, "brain2text24", lengths_by_session={"t00.2025.01.01": [120], "t00.2025.01.02": [160]})
            stats_path = tmp / "stats.pt"
            _write_stats(
                stats_path,
                session_ids=["t00.2025.01.01", "t00.2025.01.02"],
                dim=512,
                per_session_dims={"t00.2025.01.01": 512, "t00.2025.01.02": 256},
            )
            report = run_audit(cache_roots=[raw_root], stats_paths=[stats_path], dataset="brain2text24", segment_bins=80)
            stats = report["stats_audits"][0]
            self.assertTrue(stats["compatible_feature_modes"]["tx_only"])
            self.assertFalse(stats["compatible_feature_modes"]["tx_sbp"])

    def test_wrong_session_keys_are_flagged(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            raw_root = tmp / "cache"
            _write_dataset(raw_root, "brain2text24", lengths_by_session={"t00.2025.01.01": [120], "t00.2025.01.02": [160]})
            stats_path = tmp / "stats.pt"
            _write_stats(stats_path, session_ids=["t00.2025.01.01", "t00.2099.12.31"], dim=512)
            report = run_audit(cache_roots=[raw_root], stats_paths=[stats_path], dataset="brain2text24", segment_bins=80)
            comparison = report["stats_root_comparisons"][0]["datasets"]["brain2text24"]
            findings = report["stats_root_comparisons"][0]["findings"]
            self.assertIn("t00.2025.01.02", comparison["missing_session_keys_in_stats"])
            self.assertIn("t00.2099.12.31", comparison["extra_session_keys_in_stats"])
            self.assertIn("brain2text24:session_key_mismatch", findings)

    def test_smoothed_root_reports_structural_match_and_provenance_difference(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            raw_root = tmp / "cache_raw"
            smooth_root = tmp / "cache_smooth"
            lengths = {"t00.2025.01.01": [120], "t00.2025.01.02": [160]}
            _write_dataset(raw_root, "brain2text24", lengths_by_session=lengths)
            _write_dataset(smooth_root, "brain2text24", lengths_by_session=lengths, smoothing_sigma=2.0, tx_offset=5.0)
            report = run_audit(cache_roots=[raw_root, smooth_root], dataset="brain2text24", segment_bins=80)
            comparison = report["root_comparisons"][0]["datasets"]["brain2text24"]
            self.assertTrue(comparison["manifest_sha256_match"])
            self.assertFalse(comparison["metadata_sha256_match"])
            self.assertIn("same_manifest_different_metadata", comparison["findings"])

    def test_json_report_is_parseable(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            raw_root = tmp / "cache"
            other_root = tmp / "cache2"
            _write_dataset(raw_root, "brain2text24", lengths_by_session={"t00.2025.01.01": [120], "t00.2025.01.02": [160]})
            _write_dataset(other_root, "brain2text24", lengths_by_session={"t00.2025.01.01": [120], "t00.2025.01.02": [160]})
            report = run_audit(cache_roots=[raw_root, other_root], dataset="brain2text24", segment_bins=80)
            serialized = json.dumps(report)
            round_tripped = json.loads(serialized)
            self.assertIn("root_audits", round_tripped)
            self.assertEqual(len(round_tripped["root_audits"]), 2)


if __name__ == "__main__":
    unittest.main()

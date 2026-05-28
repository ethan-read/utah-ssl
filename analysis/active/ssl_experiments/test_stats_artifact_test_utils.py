from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch

from analysis.active.ssl_experiments.masked_ssl.cache import (
    CacheAccessConfig,
    resolve_precomputed_session_stats_path,
)
from analysis.active.ssl_experiments.recompute_split_feature_stats import (
    load_precomputed_split_feature_stats,
    resolve_precomputed_split_stats_path,
)
from analysis.active.ssl_experiments.stats_artifact_test_utils import (
    write_valid_session_stats_artifact,
    write_valid_split_stats_artifact,
)


class StatsArtifactTestUtilsTests(unittest.TestCase):
    def _tmp_dir(self) -> str:
        return tempfile.mkdtemp(prefix="stats_artifact_test_utils_")

    def test_write_valid_split_stats_artifact_writes_matching_sidecar(self) -> None:
        cache_root = Path(self._tmp_dir())
        dataset_root = cache_root / "brain2text24"
        dataset_root.mkdir(parents=True, exist_ok=True)
        (dataset_root / "manifest.jsonl").write_text("")
        (dataset_root / "metadata.json").write_text("{}")
        stats_path = resolve_precomputed_split_stats_path(
            cache_root=cache_root,
            dataset="brain2text24",
            train_split_name="competition_train",
            feature_mode="tx_only",
            preferred_path=None,
        )
        write_valid_split_stats_artifact(
            cache_root=cache_root,
            stats_path=stats_path,
            dataset="brain2text24",
            feature_mode="tx_only",
            boundary_key_mode="session",
            train_split_name="competition_train",
            val_split_name="competition_test",
            dim=3,
        )
        payload = torch.load(stats_path, map_location="cpu", weights_only=False)
        metadata = json.loads(stats_path.with_suffix(".json").read_text())
        self.assertEqual(payload["metadata"], metadata)
        self.assertEqual(int(payload["mean"].numel()), 3)

    def test_split_stats_load_across_boundary_key_modes(self) -> None:
        cache_root = Path(self._tmp_dir())
        dataset_root = cache_root / "brain2text24"
        dataset_root.mkdir(parents=True, exist_ok=True)
        (dataset_root / "manifest.jsonl").write_text("")
        (dataset_root / "metadata.json").write_text("{}")
        stats_path = resolve_precomputed_split_stats_path(
            cache_root=cache_root,
            dataset="brain2text24",
            train_split_name="competition_train",
            feature_mode="tx_only",
            preferred_path=None,
        )
        write_valid_split_stats_artifact(
            cache_root=cache_root,
            stats_path=stats_path,
            dataset="brain2text24",
            feature_mode="tx_only",
            boundary_key_mode="session",
            train_split_name="competition_train",
            val_split_name="competition_test",
            dim=3,
        )

        (_, _), metadata, loaded_path = load_precomputed_split_feature_stats(
            stats_path=stats_path,
            cache_root=cache_root,
            dataset="brain2text24",
            feature_mode="tx_only",
            boundary_key_mode="subject_if_available",
            train_split_name="competition_train",
            val_split_name="competition_test",
            expected_dim=3,
        )

        self.assertEqual(loaded_path, stats_path)
        self.assertEqual(metadata["boundary_key_mode"], "session")

    def test_write_valid_session_stats_artifact_writes_matching_sidecar(self) -> None:
        cache_root = Path(self._tmp_dir())
        dataset_root = cache_root / "brain2text25"
        dataset_root.mkdir(parents=True, exist_ok=True)
        (dataset_root / "manifest.jsonl").write_text("")
        (dataset_root / "metadata.json").write_text("{}")
        stats_path = resolve_precomputed_session_stats_path(
            cache_root=cache_root,
            feature_mode="tx_sbp",
            boundary_key_mode="session",
            excluded_datasets=("brain2text25",),
        )
        write_valid_session_stats_artifact(
            cache_root=cache_root,
            stats_path=stats_path,
            stats_entries={"brain2text25:t00.2025.01.01": (torch.zeros(5), torch.ones(5))},
            feature_mode="tx_sbp",
            boundary_key_mode="session",
            tx_dim=3,
            sbp_dim=2,
            excluded_datasets=("brain2text25",),
        )
        payload = torch.load(stats_path, map_location="cpu", weights_only=False)
        metadata = json.loads(stats_path.with_suffix(".json").read_text())
        self.assertEqual(payload["metadata"], metadata)
        self.assertEqual(sorted(payload["session_feature_stats"]), ["brain2text25:t00.2025.01.01"])

    def test_cache_config_deduplicates_excluded_datasets_for_canonical_paths(self) -> None:
        config = CacheAccessConfig(excluded_datasets=("zeta", "brain2text25", "zeta", ""))
        self.assertEqual(config.excluded_datasets, ("brain2text25", "zeta"))


if __name__ == "__main__":
    unittest.main()

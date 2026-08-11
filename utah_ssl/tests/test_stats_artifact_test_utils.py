from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch

from utah_ssl.cache import (
    CacheAccessConfig,
)
from utah_ssl.experiment_contract import (
    DatasetPlan,
    SignalSpec,
)
from experiments.bit_style.config import GenericSSMSSLConfig
from utah_ssl.stats import (
    build_recompute_session_feature_stats_command,
    build_recompute_split_feature_stats_command,
    load_precomputed_split_feature_stats,
    resolve_precomputed_session_stats_path,
    resolve_precomputed_split_stats_path,
)
from utah_ssl.stats_artifact_test_utils import (
    write_valid_session_stats_artifact,
    write_valid_split_stats_artifact,
)


class StatsArtifactTestUtilsTests(unittest.TestCase):
    def _tmp_dir(self) -> str:
        return tempfile.mkdtemp(prefix="stats_artifact_test_utils_")

    def test_recompute_commands_preserve_the_full_signal_contract(self) -> None:
        signal_spec = SignalSpec.tx_only(
            tx_dim=256,
            column_start=4,
            missing_channel_policy="zero_pad",
        )
        session_command = build_recompute_session_feature_stats_command(
            cache_root="/cache",
            output_path="/stats/session.pt",
            signal_spec=signal_spec,
            dataset_plan=DatasetPlan.from_mapping(
                {"brain2text24": ("competition_train",)}
            ),
            boundary_key_mode="session",
        )
        split_command = build_recompute_split_feature_stats_command(
            cache_root="/cache",
            dataset="brain2text24",
            signal_spec=signal_spec,
            boundary_key_mode="session",
            split_policy="competition_train_test",
            output_path="/stats/split.pt",
        )
        for command in (session_command, split_command):
            self.assertIn("--column-start 4", command)
            self.assertIn("--missing-channel-policy zero_pad", command)

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
            signal_spec=SignalSpec.tx_only(tx_dim=3),
            preferred_path=None,
        )
        write_valid_split_stats_artifact(
            cache_root=cache_root,
            stats_path=stats_path,
            dataset="brain2text24",
            signal_spec=SignalSpec.tx_only(tx_dim=3),
            boundary_key_mode="session",
            split_policy="competition_train_test",
            train_split_name="competition_train",
            val_split_name="competition_test",
        )
        payload = torch.load(stats_path, map_location="cpu", weights_only=False)
        metadata = json.loads(stats_path.with_suffix(".json").read_text())
        self.assertEqual(payload["metadata"], metadata)
        self.assertEqual(int(payload["mean"].numel()), 3)

    def test_split_stats_reject_different_boundary_contract(self) -> None:
        cache_root = Path(self._tmp_dir())
        dataset_root = cache_root / "brain2text24"
        dataset_root.mkdir(parents=True, exist_ok=True)
        (dataset_root / "manifest.jsonl").write_text("")
        (dataset_root / "metadata.json").write_text("{}")
        stats_path = resolve_precomputed_split_stats_path(
            cache_root=cache_root,
            dataset="brain2text24",
            train_split_name="competition_train",
            signal_spec=SignalSpec.tx_only(tx_dim=3),
            preferred_path=None,
        )
        write_valid_split_stats_artifact(
            cache_root=cache_root,
            stats_path=stats_path,
            dataset="brain2text24",
            signal_spec=SignalSpec.tx_only(tx_dim=3),
            boundary_key_mode="session",
            split_policy="competition_train_test",
            train_split_name="competition_train",
            val_split_name="competition_test",
        )

        with self.assertRaisesRegex(ValueError, "boundary_key_mode"):
            load_precomputed_split_feature_stats(
                stats_path=stats_path,
                cache_root=cache_root,
                dataset="brain2text24",
                signal_spec=SignalSpec.tx_only(tx_dim=3),
                boundary_key_mode="subject_if_available",
                train_split_name="competition_train",
                val_split_name="competition_test",
                split_policy="competition_train_test",
            )

    def test_write_valid_session_stats_artifact_writes_matching_sidecar(self) -> None:
        cache_root = Path(self._tmp_dir())
        dataset_root = cache_root / "brain2text25"
        dataset_root.mkdir(parents=True, exist_ok=True)
        (dataset_root / "manifest.jsonl").write_text("")
        (dataset_root / "metadata.json").write_text("{}")
        stats_path = resolve_precomputed_session_stats_path(
            cache_root=cache_root,
            signal_spec=SignalSpec.tx_sbp(tx_dim=3, sbp_dim=2),
            dataset_plan=DatasetPlan.from_mapping({"brain2text25": ()}),
            boundary_key_mode="session",
        )
        write_valid_session_stats_artifact(
            cache_root=cache_root,
            stats_path=stats_path,
            stats_entries={"brain2text25:t00.2025.01.01": (torch.zeros(5), torch.ones(5))},
            signal_spec=SignalSpec.tx_sbp(tx_dim=3, sbp_dim=2),
            dataset_plan=DatasetPlan.from_mapping({"brain2text25": ()}),
            boundary_key_mode="session",
        )
        payload = torch.load(stats_path, map_location="cpu", weights_only=False)
        metadata = json.loads(stats_path.with_suffix(".json").read_text())
        self.assertEqual(payload["metadata"], metadata)
        self.assertEqual(sorted(payload["session_feature_stats"]), ["brain2text25:t00.2025.01.01"])

    def test_mixed_session_stats_path_is_data_derived_not_model_named(self) -> None:
        root = Path(self._tmp_dir())
        primary = root / "cache_v1_smoothed_sigma2p0"
        override = root / "cache_v1_b2t25_area6v_sigma2p0_v1"
        for cache_root, dataset in (
            (primary, "brain2text24"),
            (override, "brain2text25"),
        ):
            dataset_root = cache_root / dataset
            dataset_root.mkdir(parents=True)
            (dataset_root / "manifest.jsonl").write_text("")
            (dataset_root / "metadata.json").write_text("{}")

        path = resolve_precomputed_session_stats_path(
            cache_root=primary,
            signal_spec=SignalSpec.sbp_only(sbp_dim=128),
            dataset_plan=DatasetPlan.from_mapping(
                {
                    "brain2text24": ("competition_train",),
                    "brain2text25": ("train", "val"),
                }
            ),
            boundary_key_mode="session",
            dataset_cache_roots={"brain2text25": override},
        )

        self.assertIn("smoothed_sigma2p0_mixed_", str(path))
        self.assertNotIn("possm", str(path).lower())

    def test_cache_config_requires_an_explicit_dataset_plan(self) -> None:
        with self.assertRaises(TypeError):
            CacheAccessConfig(signal_spec=SignalSpec.tx_only(tx_dim=128))

    def test_cache_config_allows_zero_sbp_dim_for_tx_only(self) -> None:
        config = CacheAccessConfig(
            dataset_plan={"toy": ()},
            signal_spec=SignalSpec.tx_only(tx_dim=256),
        )
        self.assertEqual(config.sbp_dim, 0)
        self.assertEqual(config.full_dim, 256)

    def test_cache_config_requires_positive_sbp_dim_for_tx_sbp(self) -> None:
        with self.assertRaisesRegex(ValueError, "sbp_dim must be positive"):
            CacheAccessConfig(
                dataset_plan={"toy": ()},
                signal_spec={
                    "mode": "tx_sbp",
                    "tx_dim": 256,
                    "sbp_dim": 0,
                },
            )

    def test_generic_ssl_config_allows_zero_sbp_dim_for_tx_only(self) -> None:
        config = GenericSSMSSLConfig(
            signal_spec=SignalSpec.tx_only(tx_dim=256),
        )
        self.assertEqual(config.sbp_dim, 0)
        self.assertEqual(config.input_dim, 256)

    def test_generic_ssl_config_requires_positive_sbp_dim_for_tx_sbp(self) -> None:
        with self.assertRaisesRegex(ValueError, "sbp_dim must be positive"):
            GenericSSMSSLConfig(
                signal_spec={
                    "mode": "tx_sbp",
                    "tx_dim": 256,
                    "sbp_dim": 0,
                }
            )


if __name__ == "__main__":
    unittest.main()

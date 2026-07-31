from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch

from analysis.active.ssl_experiments.ssl_core.normalization_stats import (
    FEATURE_STATS_SCHEMA,
    build_feature_stats_payload,
    extract_feature_stats_entries,
    write_feature_stats_artifact,
)


class NormalizationStatsTests(unittest.TestCase):
    def test_session_payload_preserves_brain2text24_legacy_key(self) -> None:
        entries = {"brain2text24:session-1": (torch.zeros(3), torch.ones(3))}
        payload = build_feature_stats_payload(
            scope="session",
            entries=entries,
            metadata={"kind": "session_featurewise_zscore_stats"},
        )

        self.assertEqual(payload["session_feature_stats"], payload["feature_stats"])
        self.assertEqual(payload["metadata"]["stats_schema"], FEATURE_STATS_SCHEMA)
        self.assertEqual(payload["metadata"]["normalization_scope"], "session")

    def test_global_payload_preserves_brain2text24_mean_std_keys(self) -> None:
        mean = torch.arange(3, dtype=torch.float32)
        std = torch.ones(3)
        payload = build_feature_stats_payload(
            scope="global",
            entries={"global": (mean, std)},
            metadata={"kind": "split_feature_stats"},
        )

        torch.testing.assert_close(payload["mean"], mean)
        torch.testing.assert_close(payload["std"], std)
        self.assertEqual(set(payload["feature_stats"]), {"global"})
        self.assertEqual(payload["metadata"]["normalization_scope"], "global")

    def test_reader_accepts_both_established_legacy_shapes(self) -> None:
        session_scope, session_entries = extract_feature_stats_entries(
            {
                "session_feature_stats": {
                    "brain2text24:s1": (torch.zeros(2), torch.ones(2))
                }
            }
        )
        global_scope, global_entries = extract_feature_stats_entries(
            {"mean": torch.zeros(2), "std": torch.ones(2)}
        )

        self.assertEqual(session_scope, "session")
        self.assertEqual(set(session_entries), {"brain2text24:s1"})
        self.assertEqual(global_scope, "global")
        self.assertEqual(set(global_entries), {"global"})

    def test_reader_rejects_empty_or_inconsistent_canonical_payloads(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least one entry"):
            extract_feature_stats_entries({"feature_stats": {}})
        with self.assertRaisesRegex(ValueError, "exactly one 'global'"):
            extract_feature_stats_entries(
                {
                    "feature_stats": {"session-1": (torch.zeros(2), torch.ones(2))},
                    "metadata": {"normalization_scope": "global"},
                }
            )

    def test_writer_keeps_payload_and_sidecar_metadata_identical(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "stats.pt"
            payload = write_feature_stats_artifact(
                output_path=output_path,
                scope="global",
                entries={"global": (torch.zeros(2), torch.ones(2))},
                metadata={"kind": "split_feature_stats"},
            )
            sidecar = json.loads(output_path.with_suffix(".json").read_text())

        self.assertEqual(payload["metadata"], sidecar)


if __name__ == "__main__":
    unittest.main()

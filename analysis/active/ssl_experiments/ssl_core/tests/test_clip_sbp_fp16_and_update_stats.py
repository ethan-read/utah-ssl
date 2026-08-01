from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from analysis.active.ssl_experiments.ssl_core.normalization_stats import (
    extract_feature_stats_entries,
    write_feature_stats_artifact,
)
from analysis.active.ssl_experiments.ssl_core.scripts.clip_sbp_fp16_and_update_stats import (
    CacheMap,
    transform_cache_maps,
    update_stats_artifact_algebraically,
)
from analysis.active.ssl_experiments.masked_ssl.cache import (
    _compute_dataset_cache_source_signature,
)


class ClipSbpFp16AndUpdateStatsTests(unittest.TestCase):
    def _make_cache(self, root: Path, *, include_ineligible_row: bool = False) -> np.ndarray:
        dataset = "brain2text24"
        dataset_root = root / dataset
        shard_root = dataset_root / "shards" / "shard-00000"
        shard_root.mkdir(parents=True)
        values = np.asarray(
            [
                [1.0, 10001.0],
                [2.0, 3.0],
                [20000.0, 4.0],
                [5.0, 6.0],
                [7.0, 8.0],
                [9.0, 10.0],
            ],
            dtype=np.float32,
        )
        np.save(shard_root / "sbp.npy", values)
        np.save(shard_root / "time_offsets.npy", np.asarray([0, 3, 6], dtype=np.int64))
        (dataset_root / "metadata.json").write_text(
            json.dumps(
                {
                    "dataset": dataset,
                    "feature_layout": {"n_sbp_features": 2},
                    "n_sbp_features": 2,
                }
            )
            + "\n"
        )
        manifest_rows = [
            {
                "session_id": "session-1",
                "subject_id": "subject-1",
                "source_split": "competition_train",
                "shard_relpath": f"{dataset}/shards/shard-00000",
                "example_index": 0,
                "n_time_bins": 3,
                "has_labels": True,
                "has_sbp": True,
                "n_sbp_features": 2,
            },
            {
                "session_id": "session-1",
                "subject_id": "subject-1",
                "source_split": "competition_train",
                "shard_relpath": f"{dataset}/shards/shard-00000",
                "example_index": 1,
                "n_time_bins": 3,
                "has_labels": True,
                "has_sbp": True,
                "n_sbp_features": 2,
            },
        ]
        if include_ineligible_row:
            manifest_rows.append(
                {
                    "session_id": "session-without-labels",
                    "subject_id": "subject-1",
                    "source_split": "competition_train",
                    "shard_relpath": f"{dataset}/shards/shard-00000",
                    "example_index": 0,
                    "n_time_bins": 3,
                    "has_labels": False,
                    "has_sbp": False,
                    "n_sbp_features": 0,
                }
            )
        (dataset_root / "manifest.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in manifest_rows)
        )
        return values

    def _write_old_stats(self, source_root: Path, stats_path: Path, values: np.ndarray) -> None:
        selected = values
        mean = selected.mean(axis=0)
        std = np.sqrt(np.maximum(selected.var(axis=0), 1e-6))
        write_feature_stats_artifact(
            output_path=stats_path,
            scope="global",
            entries={"global": (torch.from_numpy(mean), torch.from_numpy(std))},
            metadata={
                "kind": "split_feature_stats",
                "source_cache_root": str(source_root.resolve()),
                "source_cache_signature": _compute_dataset_cache_source_signature(
                    {"brain2text24": source_root}
                ),
            },
        )

    def test_transform_and_global_algebraic_update_match_direct_stats(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            source_root = tmp / "source"
            destination_root = tmp / "destination"
            values = self._make_cache(source_root)
            cache_map = CacheMap("brain2text24", source_root, destination_root)

            transform_result = transform_cache_maps([cache_map], clip_threshold=12_500.0)
            self.assertEqual(transform_result[0]["dataset_summaries"]["brain2text24"]["values_above_clip_threshold"], 1)
            transformed = np.load(
                destination_root / "brain2text24/shards/shard-00000/sbp.npy"
            )
            self.assertEqual(transformed.dtype, np.dtype(np.float16))
            np.testing.assert_array_equal(
                transformed.astype(np.float32),
                np.minimum(values, 12_500.0).astype(np.float16).astype(np.float32),
            )

            old_stats_path = tmp / "old_global.pt"
            self._write_old_stats(source_root, old_stats_path, values)
            new_stats_path = tmp / "new_global.pt"
            update_stats_artifact_algebraically(
                stats_path=old_stats_path,
                output_path=new_stats_path,
                cache_maps={"brain2text24": cache_map},
                source_splits_by_dataset={"brain2text24": ("competition_train",)},
                scope="global",
                sbp_dim=2,
            )
            _, new_entries = extract_feature_stats_entries(
                torch.load(new_stats_path, map_location="cpu", weights_only=False)
            )
            direct = transformed.astype(np.float64)
            expected_mean = direct.mean(axis=0)
            expected_std = np.sqrt(np.maximum(direct.var(axis=0), 1e-6))
            np.testing.assert_allclose(new_entries["global"][0].numpy(), expected_mean, rtol=1e-6, atol=1e-5)
            np.testing.assert_allclose(new_entries["global"][1].numpy(), expected_std, rtol=1e-6, atol=1e-5)

    def test_session_stride_two_is_updated_algebraically(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            source_root = tmp / "source"
            destination_root = tmp / "destination"
            values = self._make_cache(source_root)
            cache_map = CacheMap("brain2text24", source_root, destination_root)
            transform_cache_maps([cache_map], clip_threshold=12_500.0)
            transformed = np.load(
                destination_root / "brain2text24/shards/shard-00000/sbp.npy"
            ).astype(np.float64)

            selected_source = values[[0, 2, 3, 5]]
            old_stats_path = tmp / "old_session.pt"
            write_feature_stats_artifact(
                output_path=old_stats_path,
                scope="session",
                entries={
                    "brain2text24:session-1": (
                        torch.from_numpy(selected_source.mean(axis=0)),
                        torch.from_numpy(
                            np.sqrt(np.maximum(selected_source.var(axis=0), 1e-6))
                        ),
                    )
                },
                metadata={
                    "kind": "session_featurewise_zscore_stats",
                    "source_cache_signature": _compute_dataset_cache_source_signature(
                        {"brain2text24": source_root}
                    ),
                },
            )
            new_stats_path = tmp / "new_session.pt"
            update_stats_artifact_algebraically(
                stats_path=old_stats_path,
                output_path=new_stats_path,
                cache_maps={"brain2text24": cache_map},
                source_splits_by_dataset={"brain2text24": ("competition_train",)},
                scope="session",
                sbp_dim=2,
            )
            _, new_entries = extract_feature_stats_entries(
                torch.load(new_stats_path, map_location="cpu", weights_only=False)
            )
            selected_destination = transformed[[0, 2, 3, 5]]
            expected_mean = selected_destination.mean(axis=0)
            expected_std = np.sqrt(np.maximum(selected_destination.var(axis=0), 1e-6))
            np.testing.assert_allclose(
                new_entries["brain2text24:session-1"][0].numpy(),
                expected_mean,
                rtol=1e-6,
                atol=1e-5,
            )
            np.testing.assert_allclose(
                new_entries["brain2text24:session-1"][1].numpy(),
                expected_std,
                rtol=1e-6,
                atol=1e-5,
            )

    def test_global_update_uses_labeled_compatible_rows_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            source_root = tmp / "source"
            destination_root = tmp / "destination"
            values = self._make_cache(source_root, include_ineligible_row=True)
            cache_map = CacheMap("brain2text24", source_root, destination_root)
            transform_cache_maps([cache_map], clip_threshold=12_500.0)

            old_stats_path = tmp / "old_global.pt"
            self._write_old_stats(source_root, old_stats_path, values)
            new_stats_path = tmp / "new_global.pt"
            result = update_stats_artifact_algebraically(
                stats_path=old_stats_path,
                output_path=new_stats_path,
                cache_maps={"brain2text24": cache_map},
                source_splits_by_dataset={"brain2text24": ("competition_train",)},
                scope="global",
                sbp_dim=2,
            )
            self.assertEqual(
                result["selected_summary"]["datasets"]["brain2text24"]["manifest_rows"],
                2,
            )

    def test_transform_rejects_nested_destination(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            source_root = Path(tmpdir) / "source"
            values = self._make_cache(source_root)
            del values
            with self.assertRaisesRegex(ValueError, "inside the source"):
                transform_cache_maps(
                    [
                        CacheMap(
                            "brain2text24",
                            source_root,
                            source_root / "nested-destination",
                        )
                    ],
                    clip_threshold=12_500.0,
                )

    def test_session_update_rejects_missing_old_session(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            source_root = tmp / "source"
            destination_root = tmp / "destination"
            values = self._make_cache(source_root)
            cache_map = CacheMap("brain2text24", source_root, destination_root)
            transform_cache_maps([cache_map], clip_threshold=12_500.0)
            selected_source = values[[0, 2, 3, 5]]

            old_stats_path = tmp / "old_session.pt"
            write_feature_stats_artifact(
                output_path=old_stats_path,
                scope="session",
                entries={
                    "brain2text24:session-1": (
                        torch.from_numpy(selected_source.mean(axis=0)),
                        torch.from_numpy(np.sqrt(np.maximum(selected_source.var(axis=0), 1e-6))),
                    ),
                    "brain2text24:session-2": (
                        torch.zeros(2),
                        torch.ones(2),
                    ),
                },
                metadata={
                    "kind": "session_featurewise_zscore_stats",
                    "source_cache_signature": _compute_dataset_cache_source_signature(
                        {"brain2text24": source_root}
                    ),
                    "dataset_plan": {"brain2text24": ["competition_train"]},
                    "boundary_key_mode": "session",
                    "session_stats_bin_stride": 2,
                },
            )
            with self.assertRaisesRegex(ValueError, "did not cover stats keys"):
                update_stats_artifact_algebraically(
                    stats_path=old_stats_path,
                    output_path=tmp / "new_session.pt",
                    cache_maps={"brain2text24": cache_map},
                    source_splits_by_dataset={"brain2text24": ("competition_train",)},
                    scope="session",
                    sbp_dim=2,
                )


if __name__ == "__main__":
    unittest.main()

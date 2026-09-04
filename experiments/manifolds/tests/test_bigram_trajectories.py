import tempfile
import unittest
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.manifolds.bigram_trajectories import (
    BigramEventSet,
    BigramTrajectoryConfig,
    _join_cache_rows,
    alignment_confidence,
    analyze_bigram_event_set,
    build_bigram_event_set,
    count_reference_bigrams,
    extract_transition_window,
    fit_equal_bigram_pca,
    normalize_paths_by_session,
    save_bigram_trajectory_result,
    transition_anchor_bin,
)
from utah_ssl.canonical_data import CanonicalProbeManifestRow
from utah_ssl.experiment_contract import SignalSpec


class BigramTrajectoryTest(unittest.TestCase):
    def test_count_reference_bigrams_excludes_blank_and_silence(self):
        counts = count_reference_bigrams(
            ([1, 2, 9, 3], [1, 2, 3]),
            excluded_ids={0, 9},
        )
        self.assertEqual(counts[(1, 2)], 2)
        self.assertEqual(counts[(2, 3)], 1)
        self.assertNotIn((2, 9), counts)
        self.assertNotIn((9, 3), counts)

    def test_transition_anchor_uses_last_and_first_patch_centers(self):
        anchor = transition_anchor_bin(
            (2, 4),
            (5, 7),
            patch_size=14,
            patch_stride=4,
        )
        self.assertEqual(anchor, 23)
        with self.assertRaisesRegex(ValueError, "time ordered"):
            transition_anchor_bin(
                (2, 4),
                (3, 5),
                patch_size=14,
                patch_stride=4,
            )

    def test_alignment_confidence_is_geometric_mean_probability(self):
        logits = np.zeros((4, 3), dtype=np.float64)
        logits[0:2, 1] = np.log(2.0)
        logits[2:4, 2] = np.log(4.0)
        observed = alignment_confidence(
            logits,
            (0, 2),
            (2, 4),
            (1, 2),
        )
        expected = np.sqrt((2.0 / 4.0) * (4.0 / 6.0))
        self.assertAlmostEqual(observed, expected)

    def test_transition_window_never_truncates_boundaries(self):
        raw = np.arange(20, dtype=np.float32).reshape(10, 2)
        observed = extract_transition_window(
            raw,
            anchor_bin=5,
            before_bins=2,
            after_bins=3,
        )
        np.testing.assert_array_equal(observed, raw[3:9])
        self.assertIsNone(
            extract_transition_window(
                raw,
                anchor_bin=1,
                before_bins=2,
                after_bins=3,
            )
        )

    @staticmethod
    def _cache_row(session_id: str = "s0") -> CanonicalProbeManifestRow:
        return CanonicalProbeManifestRow(
            example_id="e0",
            session_id=session_id,
            subject_id="t12",
            source_split="competition_test",
            has_labels=True,
            shard_relpath="brain2text24/shards/s0",
            example_index=0,
            n_tx_features=0,
            n_sbp_features=128,
            target_length=2,
            transcript="test",
        )

    def test_cache_join_rejects_session_mismatch(self):
        examples = pd.DataFrame(
            [
                {
                    "example_id": "e0",
                    "session_id": "s1",
                    "source_split": "competition_test",
                }
            ]
        )
        with self.assertRaisesRegex(ValueError, "session or split mismatch"):
            _join_cache_rows(
                examples,
                [self._cache_row("s0")],
                signal_spec=SignalSpec.sbp_only(sbp_dim=128),
            )

    def test_session_normalization_preserves_missing_jitter_paths(self):
        paths = np.asarray(
            [
                [[1.0, 3.0], [3.0, 5.0]],
                [[np.nan, np.nan], [np.nan, np.nan]],
            ],
            dtype=np.float32,
        )
        normalized = normalize_paths_by_session(
            paths,
            ["s0", "s1"],
            means={"s0": np.asarray([1.0, 1.0]), "s1": np.zeros(2)},
            stds={"s0": np.asarray([2.0, 2.0]), "s1": np.ones(2)},
        )
        np.testing.assert_allclose(normalized[0], [[0.0, 1.0], [1.0, 2.0]])
        self.assertTrue(np.isnan(normalized[1]).all())

    def test_equal_bigram_covariance_matches_explicit_weighting(self):
        paths = np.asarray(
            [
                [[0.0, 0.0], [2.0, 0.0]],
                [[0.0, 0.0], [4.0, 0.0]],
                [[0.0, 0.0], [0.0, 6.0]],
            ]
        )
        labels = ["A-B", "A-B", "C-D"]
        observed = fit_equal_bigram_pca(paths, labels, temporal_center=False)
        mean_a = paths[:2].reshape(-1, 2).mean(axis=0)
        mean_b = paths[2:].reshape(-1, 2).mean(axis=0)
        grand = (mean_a + mean_b) / 2.0
        covariance = (
            (paths[:2].reshape(-1, 2) - grand).T
            @ (paths[:2].reshape(-1, 2) - grand)
            / 4
            + (paths[2:].reshape(-1, 2) - grand).T
            @ (paths[2:].reshape(-1, 2) - grand)
            / 2
        ) / 2
        np.testing.assert_allclose(observed.mean, grand)
        np.testing.assert_allclose(observed.covariance, covariance)

    def test_rank_two_synthetic_trajectory_is_captured_by_two_pcs(self):
        time = np.linspace(-1.0, 1.0, 9)
        paths = []
        labels = []
        for label_index, label in enumerate(("A-B", "C-D")):
            for event_index in range(4):
                path = np.zeros((9, 128))
                path[:, 0] = time * (label_index + 1)
                path[:, 1] = np.square(time) + event_index * 0.01
                paths.append(path)
                labels.append(label)
        pca = fit_equal_bigram_pca(np.asarray(paths), labels, temporal_center=True)
        self.assertGreater(np.cumsum(pca.explained_variance_ratio)[1], 0.999999)
        self.assertTrue(
            np.all(np.diff(np.cumsum(pca.explained_variance_ratio)) >= -1e-12)
        )

    def test_source_join_alignment_and_session_statistics_end_to_end(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model_dir = root / "export"
            shard_dir = model_dir / "shards"
            shard_dir.mkdir(parents=True)
            cache_root = root / "cache_v1_sbpclip12500_fp16_raw"
            cache_shard = cache_root / "brain2text24/shards/s0"
            cache_shard.mkdir(parents=True)
            vocab = {
                "index_to_symbol": ["BLANK", "A", "B", "SIL"],
                "num_classes": 4,
                "blank_index": 0,
                "sil_index": 3,
            }
            metadata = {
                "dataset": "brain2text24",
                "feature_mode": "tx_sbp",
                "checkpoint_step": 18_300,
                "patch_size_bins": 14,
                "patch_stride_bins": 4,
                "bin_size_ms": 20,
                "selected_source_splits": ["competition_test"],
                "vocab": vocab,
            }
            (model_dir / "metadata.json").write_text(json.dumps(metadata))
            (model_dir / "validation.json").write_text(json.dumps({"status": "passed"}))
            (model_dir / "_SUCCESS.json").write_text(json.dumps({"status": "complete"}))
            examples = pd.DataFrame(
                [
                    {
                        "example_export_index": index,
                        "example_id": f"e{index}",
                        "session_id": "s0",
                        "source_split": "competition_test",
                        "reference_ids": "1 2",
                        "input_length_bins": 40,
                    }
                    for index in range(2)
                ]
            )
            examples.to_csv(model_dir / "examples.csv", index=False)
            token_example_index = np.repeat(np.arange(2), 5)
            token_index = np.tile(np.arange(5), 2)
            pd.DataFrame(
                {
                    "example_export_index": token_example_index,
                    "token_index": token_index,
                }
            ).to_csv(model_dir / "tokens.csv", index=False)
            logits = np.zeros((10, 4), dtype=np.float32)
            pattern = [0, 1, 0, 2, 0]
            for example_index in range(2):
                for token, label in enumerate(pattern):
                    logits[example_index * 5 + token, label] = 8.0
            np.savez_compressed(
                shard_dir / "shard_00000.npz",
                logits=logits,
                token_example_index=token_example_index,
                token_index=token_index,
            )
            (model_dir / "shards.json").write_text(
                json.dumps([{"shard": "shard_00000.npz"}])
            )
            cache_metadata = {
                "dataset_family": "brain2text24",
                "bin_size_ms": 20,
                "sbp_storage_dtype": "float16",
                "sbp_clip_threshold": 12_500.0,
                "phoneme_vocabulary": vocab,
            }
            dataset_root = cache_root / "brain2text24"
            (dataset_root / "metadata.json").write_text(json.dumps(cache_metadata))
            manifest_rows = [
                {
                    "example_id": f"e{index}",
                    "session_id": "s0",
                    "subject_id": "t12",
                    "source_split": "competition_test",
                    "has_labels": True,
                    "shard_relpath": "brain2text24/shards/s0",
                    "example_index": index,
                    "n_tx_features": 0,
                    "n_sbp_features": 128,
                    "target_length": 2,
                    "n_time_bins": 40,
                    "transcript": "test",
                }
                for index in range(2)
            ]
            (dataset_root / "manifest.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in manifest_rows)
            )
            raw = np.arange(80 * 128, dtype=np.float32).reshape(80, 128)
            np.save(cache_shard / "sbp.npy", raw.astype(np.float16))
            np.save(cache_shard / "time_offsets.npy", np.asarray([0, 40, 80]))
            np.save(cache_shard / "phoneme_ids.npy", np.asarray([1, 2, 1, 2]))
            np.save(cache_shard / "phoneme_offsets.npy", np.asarray([0, 2, 4]))
            config = BigramTrajectoryConfig(
                minimum_transcript_count=2,
                before_bins=3,
                after_bins=3,
                bootstrap_repetitions=2,
                expected_examples=2,
                expected_sessions=1,
                expected_bigram_count=1,
            )
            event_set = build_bigram_event_set(
                model_dir,
                cache_root,
                config=config,
            )
            self.assertEqual(len(event_set.events), 2)
            self.assertEqual(event_set.events.anchor_bin.tolist(), [15, 15])
            self.assertEqual(event_set.paths_by_jitter[0].shape, (2, 7, 128))
            self.assertTrue(np.isfinite(event_set.paths_by_jitter[0]).all())
            self.assertEqual(event_set.session_statistics.loc[0, "n_bins"], 80)
            self.assertTrue(event_set.exclusions.empty)
            np.save(cache_shard / "phoneme_ids.npy", np.asarray([1, 1, 1, 2]))
            with self.assertRaisesRegex(ValueError, "labels disagree"):
                build_bigram_event_set(
                    model_dir,
                    cache_root,
                    config=config,
                )

    @staticmethod
    def _synthetic_event_set() -> BigramEventSet:
        rng = np.random.default_rng(4)
        bigrams = ("A-B", "C-D", "E-F")
        sessions = ("s0", "s1", "s2", "s3")
        rows = []
        nominal = []
        for bigram_index, bigram in enumerate(bigrams):
            for session_index, session in enumerate(sessions):
                for repetition in range(2):
                    time = np.linspace(-1.0, 1.0, 7)
                    path = rng.normal(scale=0.01, size=(7, 128))
                    path[:, bigram_index] += time
                    path[:, 3] += np.square(time) * 0.3
                    nominal.append(path.astype(np.float32))
                    rows.append(
                        {
                            "event_index": len(rows),
                            "session_id": session,
                            "bigram": bigram,
                            "alignment_confidence": 0.8,
                        }
                    )
        nominal_array = np.stack(nominal)
        counts = pd.DataFrame(
            [
                {
                    "bigram": value,
                    "first_id": index * 2 + 1,
                    "second_id": index * 2 + 2,
                    "transcript_count": 8,
                    "aligned_count": 8,
                    "jitter_-2_valid_count": 8,
                    "jitter_+0_valid_count": 8,
                    "jitter_+2_valid_count": 8,
                }
                for index, value in enumerate(bigrams)
            ]
        )
        return BigramEventSet(
            paths_by_jitter={-2: nominal_array, 0: nominal_array, 2: nominal_array},
            events=pd.DataFrame(rows),
            counts=counts,
            diagnostics=pd.DataFrame([{"example_id": "e0"}]),
            exclusions=pd.DataFrame(columns=["example_id", "bigram", "reason"]),
            session_statistics=pd.DataFrame(
                [{"session_id": value, "n_bins": 100} for value in sessions]
            ),
            session_means={value: np.zeros(128) for value in sessions},
            session_stds={value: np.ones(128) for value in sessions},
            candidate_pairs=((1, 2), (3, 4), (5, 6)),
            symbol_by_id={1: "A", 2: "B", 3: "C", 4: "D", 5: "E", 6: "F"},
            time_ms=np.arange(-3, 4) * 20,
            metadata={"synthetic": True},
        )

    def test_ranking_jitter_and_artifact_round_trip(self):
        config = BigramTrajectoryConfig(
            minimum_transcript_count=1,
            before_bins=3,
            after_bins=3,
            bootstrap_repetitions=10,
            expected_examples=None,
            expected_sessions=None,
            expected_bigram_count=None,
        )
        first = analyze_bigram_event_set(self._synthetic_event_set(), config=config)
        second = analyze_bigram_event_set(self._synthetic_event_set(), config=config)
        pd.testing.assert_frame_equal(first.ranking, second.ranking)
        pd.testing.assert_frame_equal(
            first.jitter_sensitivity,
            second.jitter_sensitivity,
        )
        self.assertEqual(set(first.jitter_sensitivity.jitter_ms), {-40, 0, 40})
        fractions = first.ranking.top6_trajectory_captured_fraction.to_numpy()
        self.assertTrue(np.all((fractions >= 0) & (fractions <= 1)))
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "result"
            saved = save_bigram_trajectory_result(
                first,
                destination,
                git_commit="test",
            )
            self.assertEqual(Path(saved["output_dir"]), destination)
            self.assertTrue((destination / "_SUCCESS.json").exists())
            self.assertTrue((destination / "excluded_events.csv").exists())
            self.assertEqual(len(pd.read_csv(destination / "bigram_ranking.csv")), 3)
            with np.load(destination / "mean_trajectories.npz") as arrays:
                self.assertEqual(arrays["session_change_128"].shape[-1], 128)
            with self.assertRaises(FileExistsError):
                save_bigram_trajectory_result(
                    first,
                    destination,
                    git_commit="test",
                )


if __name__ == "__main__":
    unittest.main()

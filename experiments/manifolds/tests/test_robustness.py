import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from experiments.manifolds.robustness import (
    ReconstructedEventSet,
    RobustnessConfig,
    RobustnessResult,
    TrajectoryEvent,
    _nearest_centroid_predictions,
    _sample_pair_indices,
    build_reconstructed_event_set,
    evaluate_session_fold,
    generate_unique_session_splits,
    reconstruct_overlapping_bins,
    save_robustness_result,
    summarize_loso,
)


class NeuralTrajectoryRobustnessTest(unittest.TestCase):
    def test_config_rejects_overlapping_null_paths(self):
        config = RobustnessConfig(
            before_bins=15,
            after_bins=15,
            null_exclusion_bins=15,
        )
        with self.assertRaisesRegex(ValueError, "cannot overlap"):
            config.validate()

    def test_reconstruct_overlapping_bins_is_exact(self):
        bins = np.arange(18 * 3, dtype=np.float32).reshape(18, 3)
        windows = np.stack(
            [bins[start : start + 6].reshape(-1) for start in (0, 4, 8, 12)]
        )
        reconstructed, max_error = reconstruct_overlapping_bins(
            windows,
            patch_size=6,
            patch_stride=4,
        )
        np.testing.assert_allclose(reconstructed, bins)
        self.assertEqual(max_error, 0.0)

    def test_reconstruct_rejects_inconsistent_overlap(self):
        bins = np.arange(10 * 2, dtype=np.float32).reshape(10, 2)
        windows = np.stack(
            [bins[start : start + 6].reshape(-1) for start in (0, 4)]
        )
        windows[1, 0] += 0.5
        with self.assertRaisesRegex(ValueError, "overlapping saved windows disagree"):
            reconstruct_overlapping_bins(
                windows,
                patch_size=6,
                patch_stride=4,
                overlap_atol=1e-5,
            )

    def test_unique_session_splits_are_deterministic_and_disjoint(self):
        sessions = tuple(f"s{index}" for index in range(8))
        first = generate_unique_session_splits(
            sessions,
            count=20,
            heldout_fraction=0.25,
            seed=7,
        )
        second = generate_unique_session_splits(
            sessions,
            count=20,
            heldout_fraction=0.25,
            seed=7,
        )
        self.assertEqual(first, second)
        self.assertEqual(len(first), len(set(first)))
        self.assertTrue(all(len(split) == 2 for split in first))
        self.assertTrue(all(set(split).issubset(set(sessions)) for split in first))

    def test_sampled_pairs_are_valid_and_reusable(self):
        first, second = _sample_pair_indices(
            100,
            max_pairs=250,
            rng=np.random.default_rng(7),
        )
        self.assertEqual(len(first), 250)
        self.assertTrue(np.all(first != second))
        self.assertTrue(np.all((first >= 0) & (first < 100)))
        self.assertTrue(np.all((second >= 0) & (second < 100)))

    def test_centroid_distances_are_mean_squared_distances(self):
        train_paths = np.asarray([[[0.0, 0.0]], [[2.0, 2.0]]])
        test_paths = np.asarray([[[1.0, 1.0]]])
        _, distances, labels = _nearest_centroid_predictions(
            train_paths,
            np.asarray([1, 2]),
            test_paths,
        )
        self.assertEqual(labels, (1, 2))
        np.testing.assert_allclose(distances, [[1.0, 1.0]])

    def test_event_builder_discards_sequences_without_usable_events(self):
        bins = np.arange(8, dtype=np.float32).reshape(4, 2)
        windows = np.stack(
            [bins[start : start + 2].reshape(-1) for start in range(3)]
        )
        payload = {
            "metadata": {
                "patch_size_bins": 2,
                "patch_stride_bins": 1,
                "bin_size_ms": 20,
                "vocab": {
                    "blank_index": 0,
                    "id_to_symbol": {"0": "BLANK", "1": "AA"},
                },
            },
            "examples": pd.DataFrame(
                [
                    {
                        "example_export_index": 0,
                        "session_id": "s0",
                        "reference_ids": "1",
                        "input_length_bins": 4,
                    }
                ]
            ),
            "example_indices": np.zeros(3, dtype=np.int64),
            "values": windows,
            "logits": np.asarray(
                [
                    [0.0, 5.0],
                    [5.0, 0.0],
                    [5.0, 0.0],
                ],
                dtype=np.float32,
            ),
        }
        config = RobustnessConfig(
            before_bins=1,
            after_bins=1,
            null_exclusion_bins=10,
        )
        with patch(
            "experiments.manifolds.robustness."
            "load_representation_export",
            return_value=payload,
        ):
            event_set = build_reconstructed_event_set(
                "unused",
                representation="input_windows",
                config=config,
            )
        self.assertFalse(event_set.sequences)
        self.assertFalse(event_set.events)
        self.assertEqual(event_set.diagnostics.loc[0, "status"], "no_usable_events")

    def test_event_builder_keeps_null_paths_disjoint_from_real_path(self):
        bins = np.arange(40, dtype=np.float32).reshape(20, 2)
        windows = np.stack(
            [bins[start : start + 2].reshape(-1) for start in range(19)]
        )
        payload = {
            "metadata": {
                "patch_size_bins": 2,
                "patch_stride_bins": 1,
                "bin_size_ms": 20,
                "vocab": {
                    "blank_index": 0,
                    "id_to_symbol": {"0": "BLANK", "1": "AA"},
                },
            },
            "examples": pd.DataFrame(
                [
                    {
                        "example_export_index": 0,
                        "session_id": "s0",
                        "reference_ids": "1",
                        "input_length_bins": 20,
                    }
                ]
            ),
            "example_indices": np.zeros(19, dtype=np.int64),
            "values": windows,
            "logits": np.zeros((19, 2), dtype=np.float32),
        }
        config = RobustnessConfig(
            before_bins=2,
            after_bins=2,
            null_centers_per_event=5,
            null_exclusion_bins=4,
        )
        with (
            patch(
                "experiments.manifolds.robustness."
                "load_representation_export",
                return_value=payload,
            ),
            patch(
                "experiments.manifolds.robustness."
                "ctc_forced_align",
                return_value=[(9, 10)],
            ),
        ):
            event_set = build_reconstructed_event_set(
                "unused",
                representation="input_windows",
                config=config,
            )
        self.assertEqual(len(event_set.events), 1)
        event = event_set.events[0]
        for null_center in event.null_center_bins:
            self.assertGreater(
                abs(null_center - event.real_center_bin),
                config.before_bins + config.after_bins,
            )

    @staticmethod
    def _synthetic_event_set() -> ReconstructedEventSet:
        rng = np.random.default_rng(11)
        sequences = {}
        events = []
        event_index = 0
        for session_index, session_id in enumerate(("s0", "s1", "s2", "s3")):
            session_offset = np.asarray([3.0 * session_index, -2.0 * session_index])
            for label_id in (1, 2):
                for repetition in range(6):
                    sequence = rng.normal(scale=0.03, size=(15, 2))
                    sequence += session_offset
                    relative_time = np.linspace(-1.0, 1.0, 5)
                    if label_id == 1:
                        pattern = np.stack((relative_time, relative_time**2), axis=1)
                    else:
                        pattern = np.stack((-relative_time**2, relative_time), axis=1)
                    sequence[5:10] += 2.0 * pattern
                    example_index = event_index
                    sequences[example_index] = sequence.astype(np.float32)
                    events.append(
                        TrajectoryEvent(
                            event_index=event_index,
                            label_id=label_id,
                            example_index=example_index,
                            session_id=session_id,
                            real_center_bin=7,
                            null_center_bins=(2, 12),
                            alignment_confidence=0.9,
                        )
                    )
                    event_index += 1
        return ReconstructedEventSet(
            sequences=sequences,
            events=tuple(events),
            metadata={
                "reconstructed_bin_size_ms": 20,
                "vocab": {
                    "id_to_symbol": {"1": "AA", "2": "T"},
                },
            },
            diagnostics=pd.DataFrame(),
            symbol_by_id={1: "AA", 2: "T"},
            before_bins=2,
            after_bins=2,
        )

    def test_fold_recovers_shape_and_rejects_matched_null(self):
        config = RobustnessConfig(
            before_bins=2,
            after_bins=2,
            null_centers_per_event=2,
            null_exclusion_bins=4,
            min_train_events=4,
            min_test_events=4,
            max_train_events_per_phoneme=20,
            max_test_events_per_phoneme=20,
            primary_pca_components=2,
            sensitivity_pca_components=(),
            repeated_split_count=3,
            permutation_repetitions=20,
            bootstrap_repetitions=50,
            seed=7,
        )
        row, details = evaluate_session_fold(
            self._synthetic_event_set(),
            test_sessions=("s3",),
            config=config,
            components=2,
            seed=7,
            return_details=True,
        )
        self.assertIsNotNone(row)
        self.assertIsNotNone(details)
        self.assertGreater(row["real_separation"], row["null_separation"])
        self.assertGreater(row["real_phoneme_balanced_accuracy"], 0.9)
        self.assertLess(row["null_phoneme_balanced_accuracy"], 0.8)
        self.assertEqual(row["test_session_count"], 1)

    def test_loso_summary_uses_session_level_effects(self):
        loso = pd.DataFrame(
            {
                "real_minus_null_separation": [0.2, 0.1, 0.3, -0.05],
                "real_phoneme_balanced_accuracy": [0.5, 0.4, 0.6, 0.3],
                "null_phoneme_balanced_accuracy": [0.2, 0.2, 0.2, 0.2],
                "real_category_balanced_accuracy": [0.7, 0.6, 0.8, 0.5],
                "null_category_balanced_accuracy": [0.4, 0.4, 0.4, 0.4],
            }
        )
        summary = summarize_loso(
            loso,
            bootstrap_repetitions=100,
            seed=7,
        )
        self.assertEqual(set(summary["metric"]), {
            "separation",
            "phoneme_balanced_accuracy",
            "category_balanced_accuracy",
        })
        self.assertTrue((summary["fraction_positive"] >= 0.75).all())

    def test_result_writer_creates_tables_and_figures(self):
        config = RobustnessConfig(
            before_bins=2,
            after_bins=2,
            null_centers_per_event=2,
            min_train_events=4,
            min_test_events=4,
            primary_pca_components=2,
            sensitivity_pca_components=(),
            repeated_split_count=2,
            permutation_repetitions=10,
            bootstrap_repetitions=20,
        )
        event_set = self._synthetic_event_set()
        row, details = evaluate_session_fold(
            event_set,
            test_sessions=("s3",),
            config=config,
            components=2,
            seed=7,
            return_details=True,
        )
        row["heldout_session"] = "s3"
        loso = pd.DataFrame([row])
        summary = summarize_loso(
            loso,
            bootstrap_repetitions=20,
            seed=7,
        )
        summary.insert(0, "representation", "input_windows")
        result = RobustnessResult(
            config=config,
            representation="input_windows",
            metadata=event_set.metadata,
            diagnostics=pd.DataFrame([{
                "example_index": 0,
                "session_id": "s0",
                "status": "included",
                "event_count": 1,
            }]),
            loso=loso,
            repeated_splits=loso.copy(),
            sensitivity=pd.DataFrame(),
            summary=summary,
            reference_details=details,
        )
        with tempfile.TemporaryDirectory() as temporary_dir:
            paths = save_robustness_result(result, Path(temporary_dir))
            for key in (
                "diagnostics",
                "loso",
                "repeated",
                "summary",
                "metadata",
                "loso_real_vs_null.png",
                "timecourses",
                "distance_matrix",
            ):
                self.assertIn(key, paths)
                self.assertTrue(Path(paths[key]).exists(), key)


if __name__ == "__main__":
    unittest.main()

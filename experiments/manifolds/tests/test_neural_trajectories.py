import unittest
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from experiments.manifolds.analysis import (
    ctc_forced_align,
    split_half_reliability,
    trajectory_separation,
)
from experiments.manifolds.run_export_analysis import run


class NeuralTrajectoryTest(unittest.TestCase):
    def test_ctc_forced_alignment_recovers_repeated_label_spans(self):
        # blank=0, target=[1, 1, 2]; the repeated 1 requires an intervening blank.
        best_path = [0, 1, 1, 0, 1, 0, 2, 2, 0]
        logits = np.full((len(best_path), 3), -8.0, dtype=np.float32)
        logits[np.arange(len(best_path)), best_path] = 8.0
        spans = ctc_forced_align(logits, [1, 1, 2], blank_index=0)
        self.assertEqual(spans, [(1, 3), (4, 5), (6, 8)])

    def test_repeatability_and_separation_detect_repeated_paths(self):
        rng = np.random.default_rng(4)
        base_a = np.stack([np.linspace(0, 1, 7), np.zeros(7)], axis=1)
        base_b = np.stack([np.zeros(7), np.linspace(0, 1, 7)], axis=1)
        paths = []
        labels = []
        for label, base in (("a", base_a), ("b", base_b)):
            for _ in range(12):
                paths.append(base + rng.normal(scale=0.02, size=base.shape))
                labels.append(label)
        reliability = split_half_reliability(paths, labels, repetitions=30)
        self.assertGreater(reliability["a"], 0.9)
        self.assertGreater(reliability["b"], 0.9)
        separation = trajectory_separation(paths, labels, permutations=99)
        self.assertGreater(separation["between_minus_within"], 0.2)
        self.assertLessEqual(separation["permutation_p_value"], 0.05)

    def test_end_to_end_export_analysis_writes_outputs(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            model_dir = root / "model"
            shard_dir = model_dir / "shards"
            shard_dir.mkdir(parents=True)
            best_path = np.asarray([0, 1, 1, 0, 2, 2, 0])
            example_count = 8
            token_count = len(best_path) * example_count
            logits = np.full((token_count, 3), -8.0, dtype=np.float32)
            repeated_path = np.tile(best_path, example_count)
            logits[np.arange(token_count), repeated_path] = 8.0
            example_indices = np.repeat(np.arange(example_count), len(best_path))
            token_indices = np.tile(np.arange(len(best_path)), example_count)
            time = np.tile(np.linspace(-1, 1, len(best_path)), example_count)
            hidden = np.stack([time, time**2, repeated_path], axis=1).astype(np.float32)
            np.savez_compressed(
                shard_dir / "part-00000.npz",
                hidden=hidden,
                logits=logits,
                token_example_index=example_indices,
                token_index=token_indices,
            )
            (model_dir / "metadata.json").write_text(json.dumps({
                "vocab": {"blank_index": 0, "id_to_symbol": {"0": "BLANK", "1": "A", "2": "B"}},
                "patch_stride_ms": 80,
            }))
            (model_dir / "shards.json").write_text(json.dumps([{"shard": "part-00000.npz"}]))
            pd.DataFrame({
                "example_export_index": example_indices,
                "token_index": token_indices,
            }).to_csv(model_dir / "tokens.csv", index=False)
            pd.DataFrame({
                "example_export_index": np.arange(example_count),
                "reference_ids": ["1 2"] * example_count,
            }).to_csv(model_dir / "examples.csv", index=False)
            output_dir = root / "output"
            summary = run(SimpleNamespace(
                model_dir=model_dir,
                output_dir=output_dir,
                representation="hidden",
                before=1,
                after=1,
                min_trials=4,
                max_events_per_phoneme=5,
                components=2,
                reliability_repetitions=5,
                permutations=9,
                plot_phonemes=2,
                max_plot_trials=5,
                seed=7,
            ))
            self.assertEqual(summary["retained_event_count"], 10)
            self.assertEqual(summary["eligible_event_count"], 16)
            self.assertTrue((output_dir / "phoneme_trajectories.png").exists())
            self.assertTrue((output_dir / "phoneme_repeatability.csv").exists())
            self.assertTrue((output_dir / "summary.json").exists())


if __name__ == "__main__":
    unittest.main()

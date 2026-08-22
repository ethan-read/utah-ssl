from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.manifolds.articulator_probe import (
    build_aligned_consonant_events,
    fit_session_heldout_articulator_probes,
    load_articulatory_taxonomy,
    load_representation_arrays,
)


class ArticulatorProbeTest(unittest.TestCase):
    def test_representation_loader_validates_manifest_and_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / "shards").mkdir()
            hidden = np.arange(12, dtype=np.float32).reshape(4, 3)
            logits = np.arange(20, dtype=np.float32).reshape(4, 5)
            example_indices = np.asarray([0, 0, 1, 1], dtype=np.int64)
            np.savez_compressed(
                root / "shards" / "part-00000.npz",
                hidden=hidden,
                logits=logits,
                token_example_index=example_indices,
            )
            manifest = [
                {
                    "shard": "part-00000.npz",
                    "token_count": 4,
                    "hidden_dim": 3,
                    "vocab_size": 5,
                }
            ]
            metadata = {
                "token_count": 4,
                "hidden_dim": 3,
                "example_count": 2,
                "vocab": {"num_classes": 5},
            }
            (root / "shards.json").write_text(json.dumps(manifest))
            (root / "metadata.json").write_text(json.dumps(metadata))

            loaded_hidden, loaded_logits, loaded_indices = (
                load_representation_arrays(root)
            )
            np.testing.assert_array_equal(loaded_hidden, hidden)
            np.testing.assert_array_equal(loaded_logits, logits)
            np.testing.assert_array_equal(loaded_indices, example_indices)

            manifest[0]["token_count"] = 5
            (root / "shards.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "shape does not match"):
                load_representation_arrays(root)

    def test_alignment_mean_pools_reference_consonants_and_excludes_hh(self) -> None:
        paths = ([0, 27, 27, 0, 7, 7, 0], [0, 16, 0, 36, 36, 0])
        logits_parts = []
        hidden_parts = []
        example_index_parts = []
        for example_index, path in enumerate(paths):
            logits = np.full((len(path), 41), -8.0, dtype=np.float32)
            logits[np.arange(len(path)), path] = 8.0
            logits_parts.append(logits)
            hidden_parts.append(
                np.column_stack(
                    [
                        np.full(len(path), example_index, dtype=np.float32),
                        np.arange(len(path), dtype=np.float32),
                    ]
                )
            )
            example_index_parts.append(
                np.full(len(path), example_index, dtype=np.int64)
            )
        examples = pd.DataFrame(
            {
                "example_export_index": [0, 1],
                "example_id": ["example-0", "example-1"],
                "session_id": ["train", "test"],
                "reference_ids": ["27 7", "16 36"],
            }
        )
        features, events = build_aligned_consonant_events(
            hidden=np.concatenate(hidden_parts),
            logits=np.concatenate(logits_parts),
            token_example_indices=np.concatenate(example_index_parts),
            examples=examples,
            taxonomy=load_articulatory_taxonomy(),
            blank_index=0,
        )
        self.assertEqual(events["symbol"].tolist(), ["P", "B", "W"])
        self.assertEqual(events[["lips", "tongue_front", "tongue_body"]].values.tolist(), [[1, 0, 0], [1, 0, 0], [1, 0, 1]])
        np.testing.assert_allclose(features[:, 1], [1.5, 4.5, 3.5])

    def test_session_heldout_probes_recover_synthetic_linear_targets(self) -> None:
        rng = np.random.default_rng(11)
        sessions = ("train-a", "train-b", "test-a", "test-b")
        features = []
        rows = []
        event_index = 0
        for session_index, session_id in enumerate(sessions):
            for index in range(80):
                labels = np.asarray(
                    [index % 2, (index // 2) % 2, (index // 4) % 2],
                    dtype=np.int64,
                )
                signal = 2.5 * (2 * labels - 1)
                features.append(
                    np.concatenate(
                        [signal, rng.normal(scale=0.25, size=3)]
                    ).astype(np.float32)
                )
                rows.append(
                    {
                        "event_index": event_index,
                        "example_id": f"{session_index}-{index}",
                        "session_id": session_id,
                        "symbol": "synthetic",
                        "lips": int(labels[0]),
                        "tongue_front": int(labels[1]),
                        "tongue_body": int(labels[2]),
                    }
                )
                event_index += 1
        result = fit_session_heldout_articulator_probes(
            features=np.stack(features),
            events=pd.DataFrame(rows),
            train_session_ids=sessions[:2],
            test_session_ids=sessions[2:],
            permutations=19,
            seed=7,
        )
        pooled = result["pooled_metrics"].set_index("target")
        self.assertTrue((pooled["balanced_accuracy"] > 0.99).all())
        self.assertTrue(pooled["converged"].all())
        self.assertTrue((pooled["iterations"] < pooled["max_iterations"]).all())
        self.assertTrue(
            (pooled["within_session_shuffle_mean_balanced_accuracy"] < 0.6).all()
        )
        self.assertEqual(len(result["session_metrics"]), 6)
        self.assertEqual(len(result["predictions"]), 3 * 160)


if __name__ == "__main__":
    unittest.main()

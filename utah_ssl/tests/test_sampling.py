from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np
import torch

from utah_ssl.cache import ExampleRow
from utah_ssl.sampling import get_sampling_plan, normalize_segment


class SamplingTests(unittest.TestCase):
    def test_normalize_segment_only_changes_present_features(self) -> None:
        x = torch.tensor([[3.0, 100.0], [5.0, 100.0]])
        normalized = normalize_segment(
            x,
            torch.tensor([1.0, 0.0]),
            session_feature_stats={
                "toy:s1": (torch.tensor([1.0, 0.0]), torch.tensor([2.0, 1.0]))
            },
            session_key="toy:s1",
        )
        torch.testing.assert_close(
            normalized,
            torch.tensor([[1.0, 100.0], [2.0, 100.0]]),
        )
        torch.testing.assert_close(x, torch.tensor([[3.0, 100.0], [5.0, 100.0]]))

    def test_sampling_plan_uses_valid_window_mass_and_is_cached(self) -> None:
        rows = [
            ExampleRow("toy", "s1", None, "toy/shards/a", 0, 10, True, False, 1, 0),
            ExampleRow("toy", "s2", None, "toy/shards/b", 0, 20, True, False, 1, 0),
        ]
        context = SimpleNamespace(
            sampling_plan_cache={},
            pretrain_datasets=["toy"],
            split_rows_by_dataset={"train": {"toy": rows}},
        )
        plan = get_sampling_plan(context, "train", 5, 1.0)
        self.assertEqual(plan.dataset_names, ("toy",))
        np.testing.assert_allclose(plan.dataset_probs, np.array([1.0]))
        np.testing.assert_allclose(plan.shard_probs_by_dataset["toy"], np.array([6 / 22, 16 / 22]))
        self.assertIs(get_sampling_plan(context, "train", 5, 1.0), plan)


if __name__ == "__main__":
    unittest.main()

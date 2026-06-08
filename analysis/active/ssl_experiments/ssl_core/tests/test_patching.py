from __future__ import annotations

import unittest

import torch

from analysis.active.ssl_experiments.ssl_core.patching import (
    causal_conv_lengths,
    patch_batch,
    patch_starts,
    patched_length,
    patched_lengths,
)


class PatchingTest(unittest.TestCase):
    def test_floor_policy_matches_willett_lengths(self) -> None:
        self.assertEqual(patch_starts(0, patch_size=3, patch_stride=2, policy="floor"), [])
        self.assertEqual(patch_starts(1, patch_size=3, patch_stride=2, policy="floor"), [0])
        self.assertEqual(patch_starts(3, patch_size=3, patch_stride=2, policy="floor"), [0])
        self.assertEqual(patch_starts(4, patch_size=3, patch_stride=2, policy="floor"), [0])
        self.assertEqual(patch_starts(5, patch_size=3, patch_stride=2, policy="floor"), [0, 2])
        self.assertEqual(patch_starts(7, patch_size=3, patch_stride=2, policy="floor"), [0, 2, 4])
        self.assertEqual(patched_length(7, patch_size=3, patch_stride=2, policy="floor"), 3)

    def test_cover_tail_policy_keeps_legacy_tail_patch(self) -> None:
        self.assertEqual(patch_starts(6, patch_size=4, patch_stride=3, policy="cover_tail"), [0, 2])
        self.assertEqual(patch_starts(8, patch_size=4, patch_stride=3, policy="cover_tail"), [0, 3, 4])

    def test_patch_batch_pads_short_floor_patch(self) -> None:
        x = torch.arange(2 * 5 * 2, dtype=torch.float32).reshape(2, 5, 2)
        lengths = torch.tensor([5, 2], dtype=torch.long)
        patched, token_lengths = patch_batch(
            x,
            lengths,
            patch_size=3,
            patch_stride=2,
            policy="floor",
        )
        self.assertEqual(tuple(patched.shape), (2, 2, 6))
        self.assertEqual(token_lengths.tolist(), [2, 1])
        self.assertTrue(torch.equal(patched[0, 0], x[0, 0:3].reshape(-1)))
        self.assertTrue(torch.equal(patched[0, 1], x[0, 2:5].reshape(-1)))
        expected_short = torch.cat([x[1, 0:2], x.new_zeros((1, 2))], dim=0).reshape(-1)
        self.assertTrue(torch.equal(patched[1, 0], expected_short))

    def test_vectorized_lengths_and_causal_conv_lengths(self) -> None:
        lengths = torch.tensor([0, 1, 5, 7], dtype=torch.long)
        self.assertEqual(
            patched_lengths(lengths, patch_size=3, patch_stride=2, policy="floor").tolist(),
            [0, 1, 2, 3],
        )
        self.assertEqual(causal_conv_lengths(lengths, stride=4).tolist(), [0, 1, 2, 2])


if __name__ == "__main__":
    unittest.main()

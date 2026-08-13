from __future__ import annotations

import unittest

import torch

from utah_ssl.signal_processing import apply_gaussian_smoothing, gaussian_kernel_1d


class SignalProcessingTests(unittest.TestCase):
    def test_gaussian_kernel_is_odd_symmetric_and_normalized(self) -> None:
        kernel = gaussian_kernel_1d(
            2.0,
            device=torch.device("cpu"),
            dtype=torch.float32,
            radius=3,
        )
        self.assertEqual(kernel.numel(), 7)
        torch.testing.assert_close(kernel, kernel.flip(0))
        torch.testing.assert_close(kernel.sum(), torch.tensor(1.0))

    def test_smoothing_preserves_constant_present_and_absent_features(self) -> None:
        x = torch.stack(
            [torch.ones(8), torch.arange(8, dtype=torch.float32)],
            dim=1,
        )
        smoothed = apply_gaussian_smoothing(
            x,
            torch.tensor([1.0, 0.0]),
            sigma_bins=2.0,
        )
        torch.testing.assert_close(smoothed[:, 0], x[:, 0])
        torch.testing.assert_close(smoothed[:, 1], x[:, 1])


if __name__ == "__main__":
    unittest.main()

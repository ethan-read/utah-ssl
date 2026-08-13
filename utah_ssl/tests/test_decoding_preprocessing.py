from __future__ import annotations

import unittest

import torch

from utah_ssl.decoding_preprocessing import (
    WillettInputTransformConfig,
    prepare_willett_inputs,
    willett_gaussian_kernel_1d,
)


class WillettStylePreprocessingTests(unittest.TestCase):
    def test_sigma_two_kernel_matches_established_thresholded_width(self) -> None:
        kernel = willett_gaussian_kernel_1d(
            sigma_bins=2.0,
            kernel_size=100,
            threshold=0.01,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )

        self.assertEqual(int(kernel.numel()), 9)
        self.assertAlmostEqual(float(kernel.sum().item()), 1.0, places=6)
        self.assertTrue(torch.allclose(kernel, torch.flip(kernel, dims=[0])))

    def test_transform_masks_padding_after_smoothing(self) -> None:
        x = torch.arange(24, dtype=torch.float32).view(2, 4, 3)
        lengths = torch.tensor([4, 3], dtype=torch.long)

        transformed = prepare_willett_inputs(
            x,
            lengths,
            config=WillettInputTransformConfig(
                white_noise_sd=0.0,
                constant_offset_sd=0.0,
            ),
            is_training=False,
        )

        self.assertEqual(tuple(transformed.shape), tuple(x.shape))
        self.assertTrue(torch.allclose(transformed[1, 3], torch.zeros(3)))

    def test_augmentation_is_training_only(self) -> None:
        x = torch.zeros(2, 8, 3)
        lengths = torch.tensor([8, 5], dtype=torch.long)
        config = WillettInputTransformConfig()

        torch.manual_seed(7)
        train_x = prepare_willett_inputs(x, lengths, config=config, is_training=True)
        eval_x = prepare_willett_inputs(x, lengths, config=config, is_training=False)

        self.assertFalse(torch.allclose(train_x, torch.zeros_like(train_x)))
        self.assertTrue(torch.allclose(eval_x, torch.zeros_like(eval_x)))
        self.assertTrue(torch.allclose(train_x[1, 5:], torch.zeros_like(train_x[1, 5:])))


if __name__ == "__main__":
    unittest.main()

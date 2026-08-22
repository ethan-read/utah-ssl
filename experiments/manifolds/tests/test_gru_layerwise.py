from __future__ import annotations

import unittest

import torch

from experiments.manifolds.gru_layerwise import (
    clone_gru_as_single_layer_stack,
    forward_gru_layer_stack,
    layerwise_equivalence_errors,
)
from experiments.supervised_baselines.model import WillettPhonemeModel


class GRULayerwiseTest(unittest.TestCase):
    def test_layerwise_eval_reproduces_multilayer_gru(self) -> None:
        torch.manual_seed(17)
        model = WillettPhonemeModel(
            input_dim=3,
            vocab_size=5,
            patch_size=2,
            patch_stride=1,
            input_projection_size=4,
            input_projection_dropout=0.3,
            gru_hidden_size=7,
            gru_num_layers=3,
            gru_dropout=0.4,
            session_adapter_keys=("session-a", "session-b"),
            session_adapter_enabled=True,
        ).eval()
        inputs = torch.randn(3, 8, 3)
        lengths = torch.tensor([8, 5, 7], dtype=torch.long)
        outputs = model(
            inputs,
            lengths,
            session_ids=["session-a", "session-b", "session-a"],
        )
        layers = clone_gru_as_single_layer_stack(model.gru)
        layer_states = forward_gru_layer_stack(
            model,
            outputs["patched_inputs"],
            outputs["token_lengths"],
            layers,
        )

        self.assertEqual(len(layer_states), 3)
        self.assertEqual(layer_states[0].shape, outputs["hidden"].shape)
        torch.testing.assert_close(layer_states[-1], outputs["hidden"])
        torch.testing.assert_close(model.classifier(layer_states[-1]), outputs["logits"])
        errors = layerwise_equivalence_errors(
            standard_hidden=outputs["hidden"],
            standard_logits=outputs["logits"],
            layer_states=layer_states,
            classifier=model.classifier,
        )
        self.assertLess(errors["top_hidden_max_abs_error"], 1e-6)
        self.assertLess(errors["logits_max_abs_error"], 1e-6)

    def test_layerwise_path_rejects_training_mode(self) -> None:
        model = WillettPhonemeModel(
            input_dim=3,
            vocab_size=4,
            patch_size=2,
            patch_stride=1,
            input_projection_size=4,
            gru_hidden_size=6,
            gru_num_layers=2,
            gru_dropout=0.2,
            session_adapter_enabled=False,
        )
        layers = clone_gru_as_single_layer_stack(model.gru)
        patched = torch.randn(1, 3, 8)
        with self.assertRaisesRegex(ValueError, "evaluation-only"):
            forward_gru_layer_stack(
                model,
                patched,
                torch.tensor([3]),
                layers,
            )


if __name__ == "__main__":
    unittest.main()

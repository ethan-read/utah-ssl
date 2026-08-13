from __future__ import annotations

import unittest
from dataclasses import asdict
from types import SimpleNamespace

import torch

from experiments.supervised_baselines.checkpointing import (
    adapter_keys_from_problem,
    build_willett_model,
    config_from_checkpoint,
    load_willett_model_from_checkpoint,
)
from experiments.supervised_baselines.config import WillettReconstructionConfig


def _tiny_config() -> WillettReconstructionConfig:
    return WillettReconstructionConfig(
        normalization_mode="none",
        input_projection_size=4,
        input_projection_dropout=0.0,
        decoder_backbone_type="gru",
        gru_hidden_size=5,
        gru_num_layers=1,
        gru_dropout=0.0,
        patch_size=3,
        patch_stride=1,
    )


class WillettCheckpointingTests(unittest.TestCase):
    def test_config_loader_ignores_retired_fields(self) -> None:
        payload = {"config": {**asdict(_tiny_config()), "retired_field": "ignored"}}

        recovered = config_from_checkpoint(payload)

        self.assertEqual(recovered.gru_hidden_size, 5)
        self.assertFalse(hasattr(recovered, "retired_field"))

    def test_adapter_keys_preserve_train_then_validation_order(self) -> None:
        problem = {
            "dataset": "brain2text24",
            "boundary_key_mode": "session",
            "train_rows": [SimpleNamespace(session_id="day-a", subject_id=None)],
            "val_rows": [
                SimpleNamespace(session_id="day-a", subject_id=None),
                SimpleNamespace(session_id="day-b", subject_id=None),
            ],
        }

        self.assertEqual(
            adapter_keys_from_problem(problem),
            ("brain2text24:day-a", "brain2text24:day-b"),
        )

    def test_explicit_checkpoint_adapter_keys_take_precedence_over_problem(self) -> None:
        config = _tiny_config()
        source = build_willett_model(
            config=config,
            input_dim=3,
            vocab_size=4,
            session_adapter_keys=("brain2text24:checkpoint-day",),
        )
        payload = {
            "config": asdict(config),
            "model_state": source.state_dict(),
            "session_adapter_keys": ("brain2text24:checkpoint-day",),
        }
        problem = {
            "dataset": "brain2text24",
            "boundary_key_mode": "session",
            "train_rows": [SimpleNamespace(session_id="problem-day", subject_id=None)],
            "val_rows": [],
        }

        loaded, recovered_config, _ = load_willett_model_from_checkpoint(
            payload,
            input_dim=3,
            vocab_size=4,
            problem=problem,
        )

        self.assertEqual(recovered_config.gru_hidden_size, 5)
        self.assertEqual(
            tuple(loaded.session_input_adapter._name_map),
            ("brain2text24:checkpoint-day",),
        )
        for key, value in source.state_dict().items():
            self.assertTrue(torch.equal(value, loaded.state_dict()[key]))


if __name__ == "__main__":
    unittest.main()

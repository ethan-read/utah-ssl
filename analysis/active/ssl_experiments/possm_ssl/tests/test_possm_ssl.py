from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[5]
EXPERIMENTS_DIR = REPO_ROOT / "analysis" / "active" / "ssl_experiments"
for path in (REPO_ROOT, EXPERIMENTS_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from possm_ssl.model import (
    POSSMPhonemeModel,
    POSSMReconstructionModel,
    causal_conv_output_lengths,
    register_temporal_backbone,
)
from possm_ssl.phoneme_finetune import (
    POSSMFinetuneConfig,
    _set_train_mode,
    recover_possm_stage1_encoder,
    recover_possm_stage1_sequence_components,
    run_possm_phoneme_finetuning,
)
from possm_ssl.stage1_objectives import (
    MaskedReconstructionObjective,
    PlainReconstructionObjective,
    build_stage1_objective,
)
from possm_ssl.training import (
    POSSMTrainingConfig,
    build_possm_segment_sampler,
    compute_reconstruction_metrics,
    recover_possm_run_state_from_checkpoint,
    run_possm_training,
)


class _DummyShardStore:
    def __init__(self, shards: dict[str, dict[str, np.ndarray | None]]) -> None:
        self.shards = shards

    def get(self, shard_relpath: str) -> dict[str, np.ndarray | None]:
        return self.shards[str(shard_relpath)]


def _make_sampling_cache_context() -> SimpleNamespace:
    tx = np.array(
        [
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 1.0, 0.0],
            [4.0, 1.0, 0.0],
            [5.0, 2.0, 0.0],
            [6.0, 2.0, 0.0],
            [0.0, 1.0, 1.0],
            [0.0, 2.0, 1.0],
            [0.0, 3.0, 2.0],
            [1.0, 3.0, 2.0],
            [1.0, 4.0, 2.0],
            [1.0, 5.0, 3.0],
        ],
        dtype=np.float32,
    )
    sbp = np.array(
        [
            [10.0, 0.0],
            [11.0, 0.0],
            [12.0, 1.0],
            [13.0, 1.0],
            [14.0, 2.0],
            [15.0, 2.0],
            [20.0, 0.0],
            [21.0, 0.0],
            [22.0, 1.0],
            [23.0, 1.0],
            [24.0, 2.0],
            [25.0, 2.0],
        ],
        dtype=np.float32,
    )
    shard_key = "brain2text24/toy_shard"
    row_train = SimpleNamespace(
        dataset="brain2text24",
        session_id="t00.2025.01.01",
        subject_id="t00",
        shard_relpath=shard_key,
        example_index=0,
        n_time_bins=6,
        has_tx=True,
        has_sbp=True,
    )
    row_val = SimpleNamespace(
        dataset="brain2text24",
        session_id="t00.2025.01.02",
        subject_id="t00",
        shard_relpath=shard_key,
        example_index=1,
        n_time_bins=6,
        has_tx=True,
        has_sbp=True,
    )
    return SimpleNamespace(
        full_dim=5,
        tx_dim=3,
        sbp_dim=2,
        feature_mode="tx_sbp",
        boundary_key_mode="session",
        use_normalization=True,
        gaussian_smoothing_sigma_bins=0.0,
        session_feature_stats={
            "brain2text24:t00.2025.01.01": (torch.zeros(5), torch.ones(5)),
            "brain2text24:t00.2025.01.02": (torch.zeros(5), torch.ones(5)),
        },
        pretrain_datasets=["brain2text24"],
        split_rows_by_dataset={
            "train": {"brain2text24": [row_train]},
            "val": {"brain2text24": [row_val]},
        },
        sampling_plan_cache={},
        has_val_datasets=True,
        shard_store=_DummyShardStore(
            {
                shard_key: {
                    "time_offsets": np.array([0, 6, 12], dtype=np.int64),
                    "tx": tx,
                    "sbp": sbp,
                }
            }
        ),
    )


def _write_tiny_canonical_probe_cache(cache_root: Path) -> None:
    dataset_root = cache_root / "brain2text24"
    shard_dir = dataset_root / "toy_shard"
    shard_dir.mkdir(parents=True, exist_ok=True)

    tx = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 2.0, 0.0],
            [0.0, 2.0, 1.0],
            [2.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [2.0, 1.0, 0.0],
            [0.0, 2.0, 2.0],
            [2.0, 2.0, 0.0],
            [1.0, 0.0, 2.0],
            [0.0, 1.0, 2.0],
            [2.0, 1.0, 1.0],
        ],
        dtype=np.float32,
    )
    sbp = np.array(
        [
            [10.0, 0.0],
            [10.0, 1.0],
            [11.0, 0.0],
            [11.0, 1.0],
            [12.0, 0.0],
            [12.0, 1.0],
            [13.0, 0.0],
            [13.0, 1.0],
            [14.0, 0.0],
            [14.0, 1.0],
            [15.0, 0.0],
            [15.0, 1.0],
            [16.0, 0.0],
            [16.0, 1.0],
            [17.0, 0.0],
            [17.0, 1.0],
        ],
        dtype=np.float32,
    )
    np.save(shard_dir / "time_offsets.npy", np.array([0, 4, 8, 12, 16], dtype=np.int64))
    np.save(shard_dir / "tx.npy", tx)
    np.save(shard_dir / "sbp.npy", sbp)
    np.save(shard_dir / "phoneme_offsets.npy", np.array([0, 2, 4, 6, 8], dtype=np.int64))
    np.save(shard_dir / "phoneme_ids.npy", np.array([1, 2, 2, 1, 1, 1, 2, 2], dtype=np.int64))

    manifest_rows = [
        {
            "example_id": "src-train-0",
            "session_id": "t00.2025.01.01",
            "subject_id": "t00",
            "session_date": "2025.01.01",
            "source_split": "train",
            "has_labels": True,
            "shard_relpath": "brain2text24/toy_shard",
            "example_index": 0,
            "n_tx_features": 3,
            "n_sbp_features": 2,
            "target_length": 2,
            "transcript": "AA",
            "has_tx": True,
            "has_sbp": True,
        },
        {
            "example_id": "src-train-1",
            "session_id": "t00.2025.01.01",
            "subject_id": "t00",
            "session_date": "2025.01.01",
            "source_split": "train",
            "has_labels": True,
            "shard_relpath": "brain2text24/toy_shard",
            "example_index": 1,
            "n_tx_features": 3,
            "n_sbp_features": 2,
            "target_length": 2,
            "transcript": "BB",
            "has_tx": True,
            "has_sbp": True,
        },
        {
            "example_id": "target-train-0",
            "session_id": "t00.2025.01.02",
            "subject_id": "t00",
            "session_date": "2025.01.02",
            "source_split": "train",
            "has_labels": True,
            "shard_relpath": "brain2text24/toy_shard",
            "example_index": 2,
            "n_tx_features": 3,
            "n_sbp_features": 2,
            "target_length": 2,
            "transcript": "AB",
            "has_tx": True,
            "has_sbp": True,
        },
        {
            "example_id": "target-val-0",
            "session_id": "t00.2025.01.02",
            "subject_id": "t00",
            "session_date": "2025.01.02",
            "source_split": "val",
            "has_labels": True,
            "shard_relpath": "brain2text24/toy_shard",
            "example_index": 3,
            "n_tx_features": 3,
            "n_sbp_features": 2,
            "target_length": 2,
            "transcript": "BA",
            "has_tx": True,
            "has_sbp": True,
        },
    ]
    with (dataset_root / "manifest.jsonl").open("w") as handle:
        for row in manifest_rows:
            handle.write(json.dumps(row) + "\n")

    metadata = {
        "n_tx_features": 3,
        "n_sbp_features": 2,
        "phoneme_vocabulary": {
            "num_classes": 3,
            "blank_index": 0,
            "index_to_symbol": ["<blk>", "AA", "BB"],
        },
    }
    (dataset_root / "metadata.json").write_text(json.dumps(metadata))


def _make_stage1_checkpoint(
    tmp_path: Path,
    *,
    temporal_backbone_type: str = "gru",
    temporal_gru_hidden_size: int | None = None,
    temporal_backbone_kwargs: dict[str, object] | None = None,
) -> Path:
    model = POSSMReconstructionModel(
        input_dim=5,
        model_dim=4,
        latent_count=4,
        ffn_hidden_size=16,
        dropout=0.0,
        temporal_backbone_type=temporal_backbone_type,
        temporal_gru_hidden_size=temporal_gru_hidden_size,
        temporal_backbone_kwargs=temporal_backbone_kwargs,
        reconstruction_head_type="linear",
        feature_mode="tx_sbp",
    )
    checkpoint_path = tmp_path / "checkpoint_final.pt"
    torch.save(
        {
            "model_family": "possm",
            "stage": "stage1_reconstruction",
            "model_state": model.state_dict(),
            "config": {
                "model_family": "possm",
                "stage": "stage1_reconstruction",
                "data_mode": "normalized",
                "feature_mode": "tx_sbp",
                "boundary_key_mode": "session",
                "input_dim": 5,
                "model_dim": 4,
                "latent_count": 4,
                "value_encoder_type": "linear",
                "value_mlp_hidden_size": None,
                "ffn_hidden_size": 16,
                "dropout": 0.0,
                "temporal_backbone_type": temporal_backbone_type,
                "temporal_gru_hidden_size": temporal_gru_hidden_size,
                "temporal_gru_num_layers": 1,
                "temporal_gru_dropout": 0.0,
                "temporal_gru_bidirectional": False,
                "temporal_backbone_kwargs": dict(temporal_backbone_kwargs or {}),
                "stage1_objective_type": "plain_mse",
                "masking_type": "none",
                "mask_prob": 0.0,
                "mask_span_bins": 8,
                "mask_replace_mode": "zero",
                "reconstruction_head_type": "linear",
                "reconstruction_mlp_hidden_size": None,
                "batch_size": 1,
                "seed": 7,
                "segment_bins": 4,
                "dataset_weight_alpha": 0.25,
                "examples_per_shard": 1,
                "learning_rate": 1e-3,
                "weight_decay": 1e-2,
                "log_every": 1,
                "val_every": 1,
                "val_batches": 1,
                "checkpoint_every_steps": 1,
            },
        },
        checkpoint_path,
    )
    return checkpoint_path


def _make_legacy_stage1_checkpoint_without_objective_fields(tmp_path: Path) -> Path:
    model = POSSMReconstructionModel(
        input_dim=5,
        model_dim=4,
        latent_count=4,
        ffn_hidden_size=16,
        dropout=0.0,
        reconstruction_head_type="linear",
        feature_mode="tx_sbp",
    )
    checkpoint_path = tmp_path / "checkpoint_legacy.pt"
    torch.save(
        {
            "model_family": "possm",
            "stage": "stage1_reconstruction",
            "model_state": model.state_dict(),
            "config": {
                "model_family": "possm",
                "stage": "stage1_reconstruction",
                "data_mode": "normalized",
                "feature_mode": "tx_sbp",
                "boundary_key_mode": "session",
                "input_dim": 5,
                "model_dim": 4,
                "latent_count": 4,
                "value_encoder_type": "linear",
                "value_mlp_hidden_size": None,
                "ffn_hidden_size": 16,
                "dropout": 0.0,
                "temporal_backbone_type": "gru",
                "temporal_gru_hidden_size": None,
                "temporal_gru_num_layers": 1,
                "temporal_gru_dropout": 0.0,
                "temporal_gru_bidirectional": False,
                "reconstruction_head_type": "linear",
                "reconstruction_mlp_hidden_size": None,
                "batch_size": 1,
                "seed": 7,
                "segment_bins": 4,
                "dataset_weight_alpha": 0.25,
                "examples_per_shard": 1,
                "learning_rate": 1e-3,
                "weight_decay": 1e-2,
                "log_every": 1,
                "val_every": 1,
                "val_batches": 1,
                "checkpoint_every_steps": 1,
            },
        },
        checkpoint_path,
    )
    return checkpoint_path


def _make_legacy_stage1_checkpoint_without_temporal_backbone(tmp_path: Path) -> Path:
    model = POSSMReconstructionModel(
        input_dim=5,
        model_dim=4,
        latent_count=4,
        ffn_hidden_size=16,
        dropout=0.0,
        temporal_backbone_type="identity",
        reconstruction_head_type="linear",
        feature_mode="tx_sbp",
    )
    checkpoint_path = tmp_path / "checkpoint_legacy_no_temporal.pt"
    model_state = {
        key: value for key, value in model.state_dict().items() if not key.startswith("temporal_backbone.")
    }
    torch.save(
        {
            "model_family": "possm",
            "stage": "stage1_reconstruction",
            "model_state": model_state,
            "config": {
                "model_family": "possm",
                "stage": "stage1_reconstruction",
                "data_mode": "normalized",
                "feature_mode": "tx_sbp",
                "boundary_key_mode": "session",
                "input_dim": 5,
                "model_dim": 4,
                "latent_count": 4,
                "value_encoder_type": "linear",
                "value_mlp_hidden_size": None,
                "ffn_hidden_size": 16,
                "dropout": 0.0,
                "reconstruction_head_type": "linear",
                "reconstruction_mlp_hidden_size": None,
                "batch_size": 1,
                "seed": 7,
                "segment_bins": 4,
                "dataset_weight_alpha": 0.25,
                "examples_per_shard": 1,
                "learning_rate": 1e-3,
                "weight_decay": 1e-2,
                "log_every": 1,
                "val_every": 1,
                "val_batches": 1,
                "checkpoint_every_steps": 1,
            },
        },
        checkpoint_path,
    )
    return checkpoint_path


def _make_inconsistent_stage1_checkpoint_missing_temporal_weights(tmp_path: Path) -> Path:
    checkpoint_path = _make_stage1_checkpoint(tmp_path, temporal_backbone_type="gru")
    payload = torch.load(checkpoint_path, map_location="cpu")
    payload["model_state"] = {
        key: value
        for key, value in payload["model_state"].items()
        if not key.startswith("temporal_backbone.")
    }
    torch.save(payload, checkpoint_path)
    return checkpoint_path


class POSSMSSLTests(unittest.TestCase):
    def test_stage1_reconstruction_shapes_match_input(self) -> None:
        model = POSSMReconstructionModel(
            input_dim=5,
            model_dim=4,
            latent_count=4,
            dropout=0.0,
            feature_mode="tx_sbp",
        )
        x = torch.randn(2, 6, 5)
        lengths = torch.tensor([6, 4], dtype=torch.long)
        outputs = model(x, lengths)
        self.assertEqual(tuple(outputs["reconstruction"].shape), (2, 6, 5))
        self.assertEqual(tuple(outputs["hidden"].shape), (2, 6, 16))
        self.assertEqual(tuple(outputs["encoder_hidden"].shape), (2, 6, 16))
        self.assertTrue(torch.allclose(outputs["reconstruction"][1, 4:], torch.zeros_like(outputs["reconstruction"][1, 4:])))

    def test_stage1_identity_backbone_is_supported(self) -> None:
        model = POSSMReconstructionModel(
            input_dim=5,
            model_dim=4,
            latent_count=4,
            dropout=0.0,
            temporal_backbone_type="identity",
            feature_mode="tx_sbp",
        )
        x = torch.randn(1, 6, 5)
        lengths = torch.tensor([6], dtype=torch.long)
        outputs = model(x, lengths)
        self.assertEqual(tuple(outputs["encoder_hidden"].shape), (1, 6, 16))
        self.assertEqual(tuple(outputs["hidden"].shape), (1, 6, 16))
        self.assertEqual(tuple(outputs["reconstruction"].shape), (1, 6, 5))

    def test_stage1_gru_backbone_bidirectional_changes_hidden_width(self) -> None:
        model = POSSMReconstructionModel(
            input_dim=5,
            model_dim=4,
            latent_count=4,
            dropout=0.0,
            temporal_backbone_type="gru",
            temporal_gru_hidden_size=8,
            temporal_gru_num_layers=1,
            temporal_gru_dropout=0.0,
            temporal_gru_bidirectional=True,
            feature_mode="tx_sbp",
        )
        x = torch.randn(2, 6, 5)
        lengths = torch.tensor([6, 4], dtype=torch.long)
        outputs = model(x, lengths)
        self.assertEqual(tuple(outputs["encoder_hidden"].shape), (2, 6, 16))
        self.assertEqual(tuple(outputs["hidden"].shape), (2, 6, 16))
        self.assertEqual(tuple(outputs["reconstruction"].shape), (2, 6, 5))

    def test_plain_objective_matches_manual_mse(self) -> None:
        torch.manual_seed(7)
        model = POSSMReconstructionModel(
            input_dim=5,
            model_dim=4,
            latent_count=4,
            dropout=0.0,
            feature_mode="tx_sbp",
        )
        batch = {
            "x": torch.randn(2, 6, 5),
            "lengths": torch.tensor([6, 4], dtype=torch.long),
            "feature_mask": torch.tensor(
                [[1.0, 1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0, 0.0]], dtype=torch.float32
            ),
            "session_keys": ["a", "b"],
        }
        objective = PlainReconstructionObjective()
        metrics = compute_reconstruction_metrics(
            model,
            batch,
            objective,
            {"stage1_objective_type": "plain_mse"},
            device=torch.device("cpu"),
        )

        outputs = model(batch["x"], batch["lengths"], session_ids=batch["session_keys"])
        reconstruction = outputs["reconstruction"]
        valid_time = torch.arange(6).unsqueeze(0) < batch["lengths"].unsqueeze(1)
        valid_features = batch["feature_mask"].bool().unsqueeze(1)
        valid = valid_time.unsqueeze(-1) & valid_features
        manual_loss = (reconstruction - batch["x"]).pow(2).masked_select(valid).mean()
        self.assertAlmostEqual(float(metrics["mse"]), float(manual_loss.item()), places=6)

    def test_masked_objective_prepare_batch_is_reproducible(self) -> None:
        raw_batch = {
            "x": torch.randn(2, 6, 5),
            "lengths": torch.tensor([6, 4], dtype=torch.long),
            "feature_mask": torch.ones(2, 5, dtype=torch.float32),
            "session_keys": ["a", "b"],
        }
        objective_a = MaskedReconstructionObjective(
            masking_type="random",
            mask_prob=0.3,
            mask_span_bins=2,
            mask_replace_mode="zero",
            seed=42,
        )
        objective_b = MaskedReconstructionObjective(
            masking_type="random",
            mask_prob=0.3,
            mask_span_bins=2,
            mask_replace_mode="zero",
            seed=42,
        )
        batch_a = objective_a.prepare_batch(raw_batch, device=torch.device("cpu"), config={})
        batch_b = objective_b.prepare_batch(raw_batch, device=torch.device("cpu"), config={})
        self.assertTrue(torch.equal(batch_a.loss_mask, batch_b.loss_mask))
        self.assertTrue(torch.equal(batch_a.x_target, raw_batch["x"]))
        self.assertTrue(torch.equal(batch_b.x_target, raw_batch["x"]))
        self.assertTrue(torch.all(batch_a.x_input[batch_a.loss_mask] == 0.0).item())

    def test_masked_objective_mean_replace_runs_with_batched_feature_masks(self) -> None:
        raw_batch = {
            "x": torch.randn(2, 6, 5),
            "lengths": torch.tensor([6, 4], dtype=torch.long),
            "feature_mask": torch.tensor(
                [[1.0, 1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0, 0.0]],
                dtype=torch.float32,
            ),
            "session_keys": ["a", "b"],
        }
        objective = MaskedReconstructionObjective(
            masking_type="random",
            mask_prob=0.3,
            mask_span_bins=2,
            mask_replace_mode="mean",
            seed=7,
        )
        batch = objective.prepare_batch(raw_batch, device=torch.device("cpu"), config={})
        self.assertEqual(tuple(batch.x_input.shape), (2, 6, 5))
        self.assertFalse(torch.isnan(batch.x_input).any().item())

    def test_masked_objective_loss_only_uses_masked_positions(self) -> None:
        objective = MaskedReconstructionObjective(
            masking_type="none",
            mask_prob=0.0,
            mask_span_bins=2,
            mask_replace_mode="zero",
            seed=7,
        )
        x_target = torch.tensor([[[1.0], [2.0], [3.0], [4.0]]], dtype=torch.float32)
        lengths = torch.tensor([4], dtype=torch.long)
        feature_mask = torch.tensor([[1.0]], dtype=torch.float32)
        loss_mask = torch.tensor([[[True], [False], [False], [False]]])
        stage1_batch = objective.prepare_batch(
            {
                "x": x_target,
                "lengths": lengths,
                "feature_mask": feature_mask,
                "session_keys": ["a"],
            },
            device=torch.device("cpu"),
            config={},
        )
        stage1_batch = stage1_batch.__class__(
            x_input=stage1_batch.x_input,
            x_target=stage1_batch.x_target,
            lengths=stage1_batch.lengths,
            feature_mask=stage1_batch.feature_mask,
            loss_mask=loss_mask,
            mask_metadata=stage1_batch.mask_metadata,
            session_ids=stage1_batch.session_ids,
        )
        model_outputs = {"reconstruction": torch.tensor([[[0.0], [2.0], [3.0], [4.0]]])}
        metrics = objective.compute_loss(model_outputs, stage1_batch)
        self.assertAlmostEqual(float(metrics["mse"]), 1.0, places=6)

    def test_build_stage1_masked_objective_smoke(self) -> None:
        objective = build_stage1_objective(
            config={
                "stage1_objective_type": "masked_mse",
                "masking_type": "random",
                "mask_prob": 0.2,
                "mask_span_bins": 4,
                "mask_replace_mode": "zero",
            },
            seed=7,
        )
        self.assertIsInstance(objective, MaskedReconstructionObjective)

    def test_masked_training_config_rejects_zero_mask_setup(self) -> None:
        with self.assertRaises(ValueError):
            POSSMTrainingConfig(
                stage1_objective_type="masked_mse",
                masking_type="none",
                mask_prob=0.0,
            )

    def test_custom_temporal_backbone_registration_flows_through_config_and_model(self) -> None:
        class ToyBackbone(torch.nn.Module):
            def __init__(self, *, input_size: int, scale: float = 1.0) -> None:
                super().__init__()
                self.output_size = int(input_size)
                self.scale = float(scale)

            def forward(self, hidden: torch.Tensor, input_lengths: torch.Tensor) -> torch.Tensor:
                del input_lengths
                return hidden * self.scale

        register_temporal_backbone("toy_scale", ToyBackbone)
        config = POSSMTrainingConfig(
            temporal_backbone_type="toy_scale",
            temporal_backbone_kwargs={"scale": 2.0},
        )
        self.assertEqual(config.temporal_backbone_type, "toy_scale")
        model = POSSMReconstructionModel(
            input_dim=5,
            model_dim=4,
            latent_count=4,
            dropout=0.0,
            temporal_backbone_type="toy_scale",
            temporal_backbone_kwargs={"scale": 2.0},
            feature_mode="tx_sbp",
        )
        outputs = model(torch.randn(1, 4, 5), torch.tensor([4], dtype=torch.long))
        self.assertEqual(tuple(outputs["reconstruction"].shape), (1, 4, 5))

    def test_dense_tokenization_keeps_all_unit_positions(self) -> None:
        model = POSSMReconstructionModel(
            input_dim=5,
            model_dim=4,
            latent_count=4,
            dropout=0.0,
            feature_mode="tx_sbp",
        )
        x = torch.zeros(1, 3, 5)
        lengths = torch.tensor([3], dtype=torch.long)
        outputs = model.encode_sequence(x, lengths)
        self.assertEqual(tuple(outputs.tokens.shape), (1, 3, 5, 4))
        self.assertEqual(int(outputs.tokens.shape[2]), 5)
        self.assertFalse(torch.isnan(outputs.tokens).any().item())

    def test_causal_conv_output_lengths_match_stride_rule(self) -> None:
        lengths = torch.tensor([1, 2, 3, 4, 5, 8], dtype=torch.long)
        expected = torch.tensor([1, 1, 1, 1, 2, 2], dtype=torch.long)
        self.assertTrue(torch.equal(causal_conv_output_lengths(lengths, stride=4), expected))

    def test_recover_stage1_encoder_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = _make_stage1_checkpoint(Path(tmpdir))
            encoder, checkpoint_cfg, _ = recover_possm_stage1_encoder(checkpoint_path=checkpoint_path)
        self.assertEqual(encoder.input_dim, 5)
        self.assertEqual(encoder.hidden_size, 16)
        self.assertEqual(checkpoint_cfg["feature_mode"], "tx_sbp")

    def test_recover_stage1_sequence_components_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = _make_stage1_checkpoint(
                Path(tmpdir),
                temporal_backbone_type="gru",
                temporal_gru_hidden_size=7,
            )
            encoder, temporal_backbone, checkpoint_cfg, _ = recover_possm_stage1_sequence_components(
                checkpoint_path=checkpoint_path
            )
        self.assertEqual(encoder.hidden_size, 16)
        self.assertEqual(int(temporal_backbone.output_size), 7)
        self.assertEqual(checkpoint_cfg["temporal_backbone_type"], "gru")

    def test_recover_stage1_sequence_components_legacy_checkpoint_uses_identity_backbone(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = _make_legacy_stage1_checkpoint_without_temporal_backbone(Path(tmpdir))
            _, temporal_backbone, checkpoint_cfg, _ = recover_possm_stage1_sequence_components(
                checkpoint_path=checkpoint_path
            )
        self.assertEqual(int(temporal_backbone.output_size), 16)
        self.assertNotIn("temporal_backbone_type", checkpoint_cfg)

    def test_recover_stage1_sequence_components_raises_on_missing_declared_temporal_weights(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = _make_inconsistent_stage1_checkpoint_missing_temporal_weights(Path(tmpdir))
            with self.assertRaises(KeyError):
                recover_possm_stage1_sequence_components(checkpoint_path=checkpoint_path)

    def test_run_possm_training_normalized_smoke_and_recovery(self) -> None:
        cache_context = _make_sampling_cache_context()
        with tempfile.TemporaryDirectory() as tmpdir:
            run_state = run_possm_training(
                cache_context=cache_context,
                config=POSSMTrainingConfig(
                    feature_mode="tx_sbp",
                    data_mode="normalized",
                    segment_bins=4,
                    model_dim=4,
                    latent_count=4,
                    ffn_hidden_size=16,
                    dropout=0.0,
                    batch_size=1,
                    num_steps=2,
                    val_every=1,
                    val_batches=1,
                    checkpoint_every_steps=1,
                    log_every=1,
                ),
                output_root=Path(tmpdir),
                device=torch.device("cpu"),
            )
            self.assertTrue(Path(run_state["checkpoint_path"]).exists())
            self.assertTrue(any(Path(run_state["checkpoints_dir"]).glob("step_*.pt")))
            recovered = recover_possm_run_state_from_checkpoint(
                cache_context=cache_context,
                checkpoint_path=run_state["checkpoint_path"],
                device=torch.device("cpu"),
            )
        self.assertEqual(recovered["checkpoint_step"], 2)
        self.assertEqual(recovered["model"].feature_mode, "tx_sbp")
        self.assertEqual(recovered["train_sampler"].split_name, "train")

    def test_recover_legacy_checkpoint_injects_objective_defaults(self) -> None:
        cache_context = _make_sampling_cache_context()
        with tempfile.TemporaryDirectory() as tmpdir:
            recovered = recover_possm_run_state_from_checkpoint(
                cache_context=cache_context,
                checkpoint_path=_make_legacy_stage1_checkpoint_without_objective_fields(Path(tmpdir)),
                device=torch.device("cpu"),
            )
        self.assertEqual(recovered["config"]["stage1_objective_type"], "plain_mse")
        self.assertEqual(recovered["config"]["masking_type"], "none")
        self.assertAlmostEqual(float(recovered["config"]["mask_prob"]), 0.0, places=8)

    def test_run_possm_training_raw_smoke(self) -> None:
        cache_context = _make_sampling_cache_context()
        with tempfile.TemporaryDirectory() as tmpdir:
            run_state = run_possm_training(
                cache_context=cache_context,
                config=POSSMTrainingConfig(
                    feature_mode="tx_sbp",
                    data_mode="raw",
                    segment_bins=4,
                    model_dim=4,
                    latent_count=4,
                    ffn_hidden_size=16,
                    dropout=0.0,
                    batch_size=1,
                    num_steps=1,
                    val_every=1,
                    val_batches=1,
                    checkpoint_every_steps=1,
                    log_every=1,
                ),
                output_root=Path(tmpdir),
                device=torch.device("cpu"),
            )
            self.assertTrue(Path(run_state["checkpoint_path"]).exists())
            self.assertEqual(run_state["config"]["data_mode"], "raw")

    def test_raw_possm_sampler_ignores_cache_sigma(self) -> None:
        cache_context = _make_sampling_cache_context()
        cache_context.gaussian_smoothing_sigma_bins = 2.0
        sampler = build_possm_segment_sampler(
            cache_context,
            "train",
            batch_size=1,
            seed=7,
            segment_bins=4,
            dataset_weight_alpha=0.25,
            examples_per_shard=1,
            data_mode="raw",
        )
        batch = sampler.sample_batch()
        expected_offset = 1
        expected_x = torch.tensor(
            [
                [2.0, 0.0, 0.0, 11.0, 0.0],
                [3.0, 1.0, 0.0, 12.0, 1.0],
                [4.0, 1.0, 0.0, 13.0, 1.0],
                [5.0, 2.0, 0.0, 14.0, 2.0],
            ],
            dtype=torch.float32,
        )
        self.assertEqual(expected_offset, 1)
        self.assertTrue(torch.allclose(batch["x"][0], expected_x))

    def test_probe_frozen_keeps_conv_dropout_in_train_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base_encoder = recover_possm_stage1_encoder(
                checkpoint_path=_make_stage1_checkpoint(Path(tmpdir)),
            )[0]
            model = POSSMPhonemeModel(
                base_encoder=base_encoder,
                vocab_size=3,
                gru_hidden_size=8,
                gru_num_layers=2,
                gru_dropout=0.0,
                conv_hidden_size=8,
                conv_kernel_size=3,
                conv_stride=2,
                conv_dropout=0.2,
            )
            _set_train_mode(model, train_encoder=False)
            self.assertFalse(model.base_encoder.training)
            self.assertTrue(model.gru.training)
            self.assertTrue(model.conv.training)
            self.assertTrue(model.conv_dropout.training)
            self.assertTrue(model.classifier.training)

    def test_run_possm_phoneme_finetuning_writes_checkpoints(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            checkpoint_path = _make_stage1_checkpoint(tmp_path, temporal_gru_hidden_size=7)
            _write_tiny_canonical_probe_cache(tmp_path)
            summary = run_possm_phoneme_finetuning(
                checkpoint_path=checkpoint_path,
                cache_root=tmp_path,
                config=POSSMFinetuneConfig(
                    seed=7,
                    mode="probe_frozen",
                    dataset="brain2text24",
                    feature_mode="tx_sbp",
                    data_mode="normalized",
                    session_limit=2,
                    target_session_count=1,
                    batch_size=1,
                    num_steps=2,
                    budget_seconds=30,
                    learning_rate=1e-3,
                    encoder_learning_rate=3e-4,
                    checkpoint_every_steps=1,
                    gru_hidden_size=8,
                    gru_num_layers=2,
                    gru_dropout=0.0,
                    conv_hidden_size=8,
                    conv_kernel_size=3,
                    conv_stride=2,
                    conv_dropout=0.0,
                ),
                device=torch.device("cpu"),
            )
            self.assertEqual(summary["dataset"], "brain2text24")
            self.assertEqual(summary["feature_mode"], "tx_sbp")
            self.assertTrue(Path(summary["checkpoint_final_path"]).exists())
            self.assertTrue((Path(summary["checkpoints_dir"]) / "step_000001.pt").exists())
            self.assertIn("val_ctc_bpphone", summary["metrics"])
            self.assertIn("best_val_ctc_bpphone", summary["metrics"])
            payload = torch.load(summary["checkpoint_final_path"], map_location="cpu")
            self.assertTrue(
                any(key.startswith("pre_decoder_backbone.") for key in payload["model_state"].keys())
            )
            self.assertEqual(int(payload["model_state"]["gru.weight_ih_l0"].shape[1]), 7)


if __name__ == "__main__":
    unittest.main()

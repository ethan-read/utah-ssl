from __future__ import annotations

import json
import sys
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[5]
EXPERIMENTS_DIR = REPO_ROOT / "analysis" / "active" / "ssl_experiments"
POSSM_DIR = REPO_ROOT / "analysis" / "reference" / "possm"
for path in (REPO_ROOT, EXPERIMENTS_DIR, POSSM_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from masked_ssl.cache import _compute_cache_source_signature
from ssl_core.experiment_contract import DatasetPlan, SignalSpec
from ssl_core.scripts.recompute_split_feature_stats import (
    resolve_precomputed_split_stats_path as resolve_canonical_split_stats_path,
)
from analysis.active.ssl_experiments.ssl_core.stats_artifact_test_utils import (
    write_valid_split_stats_artifact as _write_valid_split_stats_artifact,
)

from possm_ssl.model import (
    POSSMPhonemeModel,
    POSSMReconstructionModel,
    SessionInputAdapterBank,
    causal_conv_output_lengths,
    patch_temporal_sequence,
    register_temporal_backbone,
    temporal_patch_output_lengths,
)
from possm_ssl.phoneme_finetune import (
    POSSMFinetuneConfig,
    _prepare_stage2_inputs,
    _set_train_mode,
    _stage2_decoder_train_modules,
    _willett_gaussian_kernel_1d,
    find_latest_possm_stage2_run_dir,
    recover_possm_stage1_encoder,
    recover_possm_stage1_sequence_components,
    recover_possm_stage2_summary,
    run_possm_phoneme_finetuning,
)
from possm_ssl.reporting import display_possm_stage2_summary, summarize_possm_stage2_progress
from possm_ssl.stage1_objectives import (
    MaskedReconstructionObjective,
    PlainReconstructionObjective,
    build_stage1_objective,
)
from possm_ssl.training import (
    POSSMTrainingConfig,
    build_possm_segment_sampler,
    compute_reconstruction_metrics,
    find_latest_possm_step_checkpoint,
    prune_possm_resumable_checkpoints,
    recover_possm_run_state_from_checkpoint,
    resolve_latest_possm_checkpoint_path,
    run_possm_training,
)
from s5 import BidirectionalS5SequenceBackbone, DiagonalS5SSM, S5SequenceBackbone


class _DummyShardStore:
    def __init__(self, shards: dict[str, dict[str, np.ndarray | None]]) -> None:
        self.shards = shards

    def get(self, shard_relpath: str) -> dict[str, np.ndarray | None]:
        return self.shards[str(shard_relpath)]


def _make_sampling_cache_context() -> SimpleNamespace:
    cache_root = Path(tempfile.mkdtemp(prefix="possm_test_cache_"))
    dataset_root = cache_root / "brain2text24"
    dataset_root.mkdir()
    (dataset_root / "metadata.json").write_text("{}\n")
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
        cache_root=cache_root,
        source_cache_signature="synthetic-test-signature",
        config=SimpleNamespace(
            signal_spec=SignalSpec.tx_sbp(tx_dim=3, sbp_dim=2),
            dataset_plan=DatasetPlan.from_mapping({"brain2text24": ()}),
        ),
        signal_spec=SignalSpec.tx_sbp(tx_dim=3, sbp_dim=2),
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
            "example_id": "train-0",
            "session_id": "t00.2025.01.01",
            "subject_id": "t00",
            "session_date": "2025.01.01",
            "source_split": "competition_train",
            "has_labels": True,
            "shard_relpath": "brain2text24/toy_shard",
            "example_index": 0,
            "n_tx_features": 3,
            "n_sbp_features": 2,
            "n_time_bins": 4,
            "target_length": 2,
            "transcript": "AA",
            "has_tx": True,
            "has_sbp": True,
        },
        {
            "example_id": "train-1",
            "session_id": "t00.2025.01.01",
            "subject_id": "t00",
            "session_date": "2025.01.01",
            "source_split": "competition_train",
            "has_labels": True,
            "shard_relpath": "brain2text24/toy_shard",
            "example_index": 1,
            "n_tx_features": 3,
            "n_sbp_features": 2,
            "n_time_bins": 4,
            "target_length": 2,
            "transcript": "BB",
            "has_tx": True,
            "has_sbp": True,
        },
        {
            "example_id": "test-0",
            "session_id": "t00.2025.01.01",
            "subject_id": "t00",
            "session_date": "2025.01.01",
            "source_split": "competition_test",
            "has_labels": True,
            "shard_relpath": "brain2text24/toy_shard",
            "example_index": 2,
            "n_tx_features": 3,
            "n_sbp_features": 2,
            "n_time_bins": 4,
            "target_length": 2,
            "transcript": "AB",
            "has_tx": True,
            "has_sbp": True,
        },
        {
            "example_id": "holdout-0",
            "session_id": "t00.2025.01.01",
            "subject_id": "t00",
            "session_date": "2025.01.01",
            "source_split": "competition_holdout",
            "has_labels": True,
            "shard_relpath": "brain2text24/toy_shard",
            "example_index": 3,
            "n_tx_features": 3,
            "n_sbp_features": 2,
            "n_time_bins": 4,
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
    for signal_spec in (
        SignalSpec.tx_sbp(tx_dim=3, sbp_dim=2),
        SignalSpec.tx_only(tx_dim=3),
        SignalSpec.sbp_only(sbp_dim=2),
    ):
        _write_valid_split_stats_artifact(
            cache_root=cache_root,
            stats_path=resolve_canonical_split_stats_path(
                cache_root=cache_root,
                dataset="brain2text24",
                train_split_name="competition_train",
                signal_spec=signal_spec,
                preferred_path=None,
            ),
            dataset="brain2text24",
            signal_spec=signal_spec,
            boundary_key_mode="session",
            split_policy="competition_train_test",
            train_split_name="competition_train",
            val_split_name="competition_test",
        )


def _make_stage1_checkpoint(
    tmp_path: Path,
    *,
    temporal_backbone_type: str = "gru",
    temporal_gru_hidden_size: int | None = None,
    temporal_backbone_kwargs: dict[str, object] | None = None,
    use_token_norm: bool = True,
    feature_mode: str = "tx_sbp",
) -> Path:
    input_dim = {
        "tx_only": 3,
        "sbp_only": 2,
        "tx_sbp": 5,
    }[feature_mode]
    signal_spec = {
        "tx_only": SignalSpec.tx_only(tx_dim=3),
        "sbp_only": SignalSpec.sbp_only(sbp_dim=2),
        "tx_sbp": SignalSpec.tx_sbp(tx_dim=3, sbp_dim=2),
    }[feature_mode]
    model = POSSMReconstructionModel(
        input_dim=input_dim,
        model_dim=4,
        latent_count=4,
        ffn_hidden_size=16,
        dropout=0.0,
        use_token_norm=use_token_norm,
        temporal_backbone_type=temporal_backbone_type,
        temporal_gru_hidden_size=temporal_gru_hidden_size,
        temporal_backbone_kwargs=temporal_backbone_kwargs,
        reconstruction_head_type="linear",
        signal_spec=signal_spec,
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
                "feature_mode": feature_mode,
                "signal_spec": signal_spec.to_dict(),
                "dataset_plan": {"brain2text24": []},
                "boundary_key_mode": "session",
                "input_dim": input_dim,
                "model_dim": 4,
                "latent_count": 4,
                "value_encoder_type": "linear",
                "value_mlp_hidden_size": None,
                "ffn_hidden_size": 16,
                "dropout": 0.0,
                "use_token_norm": use_token_norm,
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
        signal_spec=SignalSpec.tx_sbp(tx_dim=3, sbp_dim=2),
        model_dim=4,
        latent_count=4,
        ffn_hidden_size=16,
        dropout=0.0,
        reconstruction_head_type="linear",
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
                "signal_spec": SignalSpec.tx_sbp(tx_dim=3, sbp_dim=2).to_dict(),
                "dataset_plan": {"brain2text24": []},
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
        signal_spec=SignalSpec.tx_sbp(tx_dim=3, sbp_dim=2),
        model_dim=4,
        latent_count=4,
        ffn_hidden_size=16,
        dropout=0.0,
        temporal_backbone_type="identity",
        reconstruction_head_type="linear",
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
                "signal_spec": SignalSpec.tx_sbp(tx_dim=3, sbp_dim=2).to_dict(),
                "dataset_plan": {"brain2text24": []},
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
    def test_find_latest_possm_step_checkpoint_prefers_highest_step(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoints_dir = Path(tmpdir) / "checkpoints"
            checkpoints_dir.mkdir()
            older = checkpoints_dir / "step_000003_20260519T000000Z.pt"
            latest = checkpoints_dir / "checkpoint_final_step_000020_20260519T000000Z.pt"
            step = checkpoints_dir / "step_000010_20260519T000000Z.pt"
            malformed = checkpoints_dir / "step_latest.pt"
            for path in (older, step, latest, malformed):
                path.touch()

            self.assertEqual(find_latest_possm_step_checkpoint(checkpoints_dir), latest)

    def test_resolve_latest_possm_checkpoint_path_prefers_step_over_best_and_final(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "stage1_run"
            checkpoints_dir = run_dir / "checkpoints"
            checkpoints_dir.mkdir(parents=True)
            best = run_dir / "checkpoint_best.pt"
            final = run_dir / "checkpoint_final.pt"
            step = checkpoints_dir / "step_000005_20260519T000000Z.pt"
            for path in (best, final, step):
                path.touch()

            self.assertEqual(resolve_latest_possm_checkpoint_path(run_dir=run_dir), step)

    def test_resolve_latest_possm_checkpoint_path_prefers_final_when_it_is_ahead(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "stage1_run"
            checkpoints_dir = run_dir / "checkpoints"
            checkpoints_dir.mkdir(parents=True)
            step = checkpoints_dir / "step_004000_20260519T000000Z.pt"
            final = run_dir / "checkpoint_final.pt"
            torch.save({"step": 4000}, step)
            torch.save({"step": 4200}, final)

            self.assertEqual(resolve_latest_possm_checkpoint_path(run_dir=run_dir), final)

    def test_prune_possm_resumable_checkpoints_keeps_latest_n(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoints_dir = Path(tmpdir) / "checkpoints"
            checkpoints_dir.mkdir()
            old_step = checkpoints_dir / "step_000001_20260519T000000Z.pt"
            latest_step = checkpoints_dir / "step_000003_20260519T000000Z.pt"
            old_final_archive = checkpoints_dir / "checkpoint_final_step_000002_20260519T000000Z.pt"
            for path in (old_step, old_final_archive, latest_step):
                path.touch()

            deleted = prune_possm_resumable_checkpoints(checkpoints_dir, keep_last=1)

            self.assertEqual({path.name for path in deleted}, {old_step.name, old_final_archive.name})
            self.assertFalse(old_step.exists())
            self.assertFalse(old_final_archive.exists())
            self.assertTrue(latest_step.exists())

    def test_run_possm_training_retains_only_configured_step_checkpoints(self) -> None:
        cache_context = _make_sampling_cache_context()
        with tempfile.TemporaryDirectory() as tmpdir:
            run_state = run_possm_training(
                cache_context=cache_context,
                config=POSSMTrainingConfig(
                    signal_spec=SignalSpec.tx_sbp(tx_dim=3, sbp_dim=2),
                    data_mode="normalized",
                    segment_bins=4,
                    model_dim=4,
                    latent_count=4,
                    ffn_hidden_size=16,
                    dropout=0.0,
                    batch_size=1,
                    num_steps=3,
                    val_every=1,
                    val_batches=1,
                    checkpoint_every_steps=1,
                    checkpoint_keep_last=1,
                    log_every=1,
                ),
                output_root=Path(tmpdir),
                device=torch.device("cpu"),
            )

            checkpoint_names = sorted(path.name for path in Path(run_state["checkpoints_dir"]).glob("*.pt"))
            self.assertEqual(len(checkpoint_names), 1)
            self.assertTrue(checkpoint_names[0].startswith("step_000003_"))
            self.assertTrue(Path(run_state["checkpoint_path"]).exists())
            self.assertTrue(Path(run_state["best_checkpoint_path"]).exists())

    def test_stage1_reconstruction_shapes_match_input(self) -> None:
        model = POSSMReconstructionModel(
            input_dim=5,
            model_dim=4,
            latent_count=4,
            dropout=0.0,
            signal_spec=SignalSpec.tx_sbp(tx_dim=3, sbp_dim=2),
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
            signal_spec=SignalSpec.tx_sbp(tx_dim=3, sbp_dim=2),
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
            signal_spec=SignalSpec.tx_sbp(tx_dim=3, sbp_dim=2),
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
            signal_spec=SignalSpec.tx_sbp(tx_dim=3, sbp_dim=2),
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
                signal_spec=SignalSpec.tx_sbp(tx_dim=3, sbp_dim=2),
                stage1_objective_type="masked_mse",
                masking_type="none",
                mask_prob=0.0,
            )

    def test_sbp_only_configs_are_supported(self) -> None:
        stage1 = POSSMTrainingConfig(signal_spec=SignalSpec.sbp_only(sbp_dim=2))
        stage2 = POSSMFinetuneConfig(signal_spec=SignalSpec.sbp_only(sbp_dim=2))
        self.assertEqual(stage1.feature_mode, "sbp_only")
        self.assertEqual(stage2.feature_mode, "sbp_only")

    def test_stage1_signal_is_explicit(self) -> None:
        with self.assertRaises(TypeError):
            POSSMTrainingConfig()

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
            signal_spec=SignalSpec.tx_sbp(tx_dim=3, sbp_dim=2),
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
            signal_spec=SignalSpec.tx_sbp(tx_dim=3, sbp_dim=2),
        )
        outputs = model(torch.randn(1, 4, 5), torch.tensor([4], dtype=torch.long))
        self.assertEqual(tuple(outputs["reconstruction"].shape), (1, 4, 5))

    def test_dense_tokenization_keeps_all_unit_positions(self) -> None:
        model = POSSMReconstructionModel(
            input_dim=5,
            model_dim=4,
            latent_count=4,
            dropout=0.0,
            signal_spec=SignalSpec.tx_sbp(tx_dim=3, sbp_dim=2),
        )
        x = torch.zeros(1, 3, 5)
        lengths = torch.tensor([3], dtype=torch.long)
        outputs = model.encode_sequence(x, lengths)
        self.assertEqual(tuple(outputs.tokens.shape), (1, 3, 5, 4))
        self.assertEqual(int(outputs.tokens.shape[2]), 5)
        self.assertFalse(torch.isnan(outputs.tokens).any().item())

    def test_causal_conv_output_lengths_match_left_padded_stride_rule(self) -> None:
        lengths = torch.tensor([0, 1, 13, 14, 15, 18, 80], dtype=torch.long)
        expected = torch.tensor([0, 1, 4, 4, 4, 5, 20], dtype=torch.long)
        actual = causal_conv_output_lengths(lengths, stride=4)
        self.assertTrue(torch.equal(actual, expected))

    def test_possm_finetune_config_has_no_s5_implementation_switch(self) -> None:
        payload = asdict(POSSMFinetuneConfig())
        self.assertNotIn("s5_implementation", payload)

    def test_session_input_adapter_bank_initializes_to_identity_feature_affine(self) -> None:
        adapter = SessionInputAdapterBank(("day.1", "day/2"), input_dim=3)
        reversed_adapter = SessionInputAdapterBank(("day/2", "day.1"), input_dim=3)
        self.assertEqual(set(adapter.layers.keys()), set(reversed_adapter.layers.keys()))
        for layer in [adapter.default_layer, *adapter.layers.values()]:
            self.assertTrue(torch.allclose(layer.scale, torch.ones(3)))
            self.assertTrue(torch.allclose(layer.bias, torch.zeros(3)))
        x = torch.tensor(
            [
                [[-2.0, 0.0, 2.0], [1.0, -1.0, 0.5]],
                [[3.0, -3.0, 1.0], [0.0, 2.0, -2.0]],
            ],
            dtype=torch.float32,
        )
        actual = adapter(x, ["day.1", "unknown-day"])
        self.assertTrue(torch.allclose(actual, x))
        with self.assertRaisesRegex(ValueError, "length must match batch size"):
            adapter(x, ["day.1"])

    def test_willett_style_stage2_input_transform_smooths_after_augmentation(self) -> None:
        torch.manual_seed(7)
        x = torch.zeros(2, 8, 3)
        lengths = torch.tensor([8, 5], dtype=torch.long)
        config = POSSMFinetuneConfig(
            input_smoothing_sigma_bins=2.0,
            input_smoothing_kernel_size=100,
            input_smoothing_threshold=0.01,
            white_noise_sd=1.0,
            constant_offset_sd=0.2,
        )
        train_x = _prepare_stage2_inputs(x, lengths, config=config, is_training=True)
        eval_x = _prepare_stage2_inputs(x, lengths, config=config, is_training=False)
        self.assertEqual(tuple(train_x.shape), tuple(x.shape))
        self.assertEqual(tuple(eval_x.shape), tuple(x.shape))
        self.assertFalse(torch.allclose(train_x, torch.zeros_like(train_x)))
        self.assertTrue(torch.allclose(eval_x, torch.zeros_like(eval_x)))
        self.assertTrue(torch.allclose(train_x[1, 5:], torch.zeros_like(train_x[1, 5:])))

    def test_willett_gaussian_kernel_matches_sigma2_threshold_width(self) -> None:
        kernel = _willett_gaussian_kernel_1d(
            sigma_bins=2.0,
            kernel_size=100,
            threshold=0.01,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        self.assertEqual(int(kernel.numel()), 9)
        self.assertAlmostEqual(float(kernel.sum().item()), 1.0, places=6)
        self.assertTrue(torch.allclose(kernel, torch.flip(kernel, dims=[0])))

    def test_recover_stage1_encoder_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = _make_stage1_checkpoint(Path(tmpdir))
            encoder, checkpoint_cfg, _ = recover_possm_stage1_encoder(checkpoint_path=checkpoint_path)
        self.assertEqual(encoder.input_dim, 5)
        self.assertEqual(encoder.hidden_size, 16)
        self.assertEqual(checkpoint_cfg["feature_mode"], "tx_sbp")

    def test_recover_stage1_encoder_respects_saved_token_norm_flag(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = _make_stage1_checkpoint(Path(tmpdir), use_token_norm=False)
            encoder, checkpoint_cfg, _ = recover_possm_stage1_encoder(checkpoint_path=checkpoint_path)
        self.assertFalse(encoder.use_token_norm)
        self.assertFalse(bool(checkpoint_cfg["use_token_norm"]))

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

    def test_recover_stage1_sequence_components_rejects_incomplete_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = _make_legacy_stage1_checkpoint_without_temporal_backbone(Path(tmpdir))
            with self.assertRaisesRegex(ValueError, "missing required fields"):
                recover_possm_stage1_sequence_components(checkpoint_path=checkpoint_path)

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
                    signal_spec=SignalSpec.tx_sbp(tx_dim=3, sbp_dim=2),
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
            checkpoint_payload = torch.load(
                run_state["checkpoint_path"],
                map_location="cpu",
                weights_only=False,
            )
            self.assertIn("rng_state", checkpoint_payload)
            self.assertIn("sampler_state", checkpoint_payload)
            self.assertIn("objective_state", checkpoint_payload)
            recovered = recover_possm_run_state_from_checkpoint(
                cache_context=cache_context,
                checkpoint_path=run_state["checkpoint_path"],
                device=torch.device("cpu"),
            )
            expected_next_batch = run_state["train_sampler"].sample_batch()
            recovered_next_batch = recovered["train_sampler"].sample_batch()
        self.assertEqual(recovered["checkpoint_step"], 2)
        self.assertEqual(recovered["model"].feature_mode, "tx_sbp")
        self.assertEqual(recovered["train_sampler"].split_name, "train")
        self.assertTrue(recovered["resume_state_complete"])
        torch.testing.assert_close(expected_next_batch["x"], recovered_next_batch["x"])

    def test_resume_rejects_incomplete_stage1_checkpoint(self) -> None:
        cache_context = _make_sampling_cache_context()
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaisesRegex(ValueError, "missing required fields"):
                recover_possm_run_state_from_checkpoint(
                    cache_context=cache_context,
                    checkpoint_path=_make_legacy_stage1_checkpoint_without_objective_fields(Path(tmpdir)),
                    device=torch.device("cpu"),
                )

    def test_run_possm_training_raw_smoke(self) -> None:
        cache_context = _make_sampling_cache_context()
        with tempfile.TemporaryDirectory() as tmpdir:
            run_state = run_possm_training(
                cache_context=cache_context,
                config=POSSMTrainingConfig(
                    signal_spec=SignalSpec.tx_sbp(tx_dim=3, sbp_dim=2),
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

    def test_possm_phoneme_model_post_gru_conv_shapes(self) -> None:
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
                conv_kernel_size=3,
                conv_stride=2,
            )
            x = torch.randn(2, 8, 5)
            lengths = torch.tensor([8, 7], dtype=torch.long)
            outputs = model(x, lengths, session_ids=["a", "b"])
            self.assertEqual(int(model.gru.weight_ih_l0.shape[1]), base_encoder.hidden_size)
            self.assertEqual(tuple(outputs["gru_hidden"].shape), (2, 8, 8))
            self.assertEqual(tuple(outputs["conv_hidden"].shape), (2, 4, 8))
            self.assertEqual(tuple(outputs["logits"].shape), (2, 4, 3))
            self.assertTrue(torch.equal(outputs["token_lengths"], torch.tensor([4, 4])))
            self.assertEqual(tuple(outputs["adapted_input"].shape), (2, 8, 5))

    def test_diagonal_s5_outputs_are_masked(self) -> None:
        torch.manual_seed(0)
        ssm = DiagonalS5SSM(d_model=5, d_state=3)
        ssm.eval()

        x = torch.randn(3, 7, 5)
        lengths = torch.tensor([7, 4, 2], dtype=torch.long)
        x[1, 4:] = 3.0
        x[2, 2:] = -2.0

        output = ssm(x, lengths)

        self.assertEqual(tuple(output.shape), (3, 7, 5))
        self.assertTrue(torch.equal(output[1, 4:], torch.zeros_like(output[1, 4:])))
        self.assertTrue(torch.equal(output[2, 2:], torch.zeros_like(output[2, 2:])))
        self.assertTrue(torch.isfinite(output).all())

    def test_diagonal_s5_ignores_padded_suffix_inputs(self) -> None:
        torch.manual_seed(1)
        ssm = DiagonalS5SSM(d_model=4, d_state=3)
        ssm.eval()

        x = torch.randn(2, 6, 4)
        lengths = torch.tensor([6, 3], dtype=torch.long)
        changed_suffix = x.clone()
        changed_suffix[1, 3:] = 100.0

        base_output = ssm(x, lengths)
        changed_output = ssm(changed_suffix, lengths)

        torch.testing.assert_close(changed_output[1, :3], base_output[1, :3], atol=1e-5, rtol=1e-4)
        self.assertTrue(torch.equal(changed_output[1, 3:], torch.zeros_like(changed_output[1, 3:])))

    def test_s5_sequence_backbone_outputs_are_masked(self) -> None:
        torch.manual_seed(10)
        backbone = S5SequenceBackbone(
            d_model=5,
            d_state=3,
            num_layers=2,
            dropout=0.0,
            ffn_multiplier=1.0,
        )
        backbone.eval()

        x = torch.randn(2, 6, 5)
        lengths = torch.tensor([6, 4], dtype=torch.long)
        x[1, 4:] = 7.0
        output = backbone(x, lengths)

        self.assertEqual(tuple(output.shape), (2, 6, 5))
        self.assertTrue(torch.equal(output[1, 4:], torch.zeros_like(output[1, 4:])))
        self.assertTrue(torch.isfinite(output).all())

    def test_bidirectional_s5_sequence_backbone_shapes_and_masks(self) -> None:
        torch.manual_seed(12)
        backbone = BidirectionalS5SequenceBackbone(
            d_model=4,
            d_state=3,
            num_layers=1,
            dropout=0.0,
            ffn_multiplier=1.0,
        )
        backbone.eval()

        x = torch.randn(2, 6, 4)
        lengths = torch.tensor([6, 3], dtype=torch.long)
        x[1, 3:] = -5.0
        output = backbone(x, lengths)

        self.assertEqual(tuple(output.shape), (2, 6, 4))
        self.assertTrue(torch.equal(output[1, 3:], torch.zeros_like(output[1, 3:])))
        self.assertTrue(torch.isfinite(output).all())

    def test_possm_phoneme_model_s5_decoder_shapes_and_lengths(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base_encoder = recover_possm_stage1_encoder(
                checkpoint_path=_make_stage1_checkpoint(Path(tmpdir)),
            )[0]
            model = POSSMPhonemeModel(
                base_encoder=base_encoder,
                vocab_size=3,
                decoder_backbone_type="s5",
                s5_hidden_size=8,
                s5_state_size=4,
                s5_num_layers=2,
                s5_dropout=0.0,
                s5_direction="causal",
                conv_kernel_size=3,
                conv_stride=2,
            )
            x = torch.randn(2, 8, 5)
            lengths = torch.tensor([8, 5], dtype=torch.long)
            outputs = model(x, lengths, session_ids=["a", "b"])
            self.assertEqual(tuple(outputs["decoder_hidden"].shape), (2, 8, 8))
            self.assertEqual(tuple(outputs["gru_hidden"].shape), (2, 8, 8))
            self.assertTrue(torch.allclose(outputs["decoder_hidden"][1, 5:], torch.zeros(3, 8)))
            self.assertEqual(tuple(outputs["conv_hidden"].shape), (2, 4, 8))
            self.assertEqual(tuple(outputs["logits"].shape), (2, 4, 3))
            self.assertTrue(torch.equal(outputs["token_lengths"], torch.tensor([4, 3])))

    def test_temporal_patch_output_lengths_match_willett_rule(self) -> None:
        lengths = torch.tensor([0, 1, 3, 4, 5, 8, 9], dtype=torch.long)
        actual = temporal_patch_output_lengths(lengths, patch_size=4, patch_stride=2)
        expected = torch.tensor([0, 1, 1, 1, 1, 3, 3], dtype=torch.long)
        self.assertTrue(torch.equal(actual, expected))

    def test_patch_temporal_sequence_pads_short_final_windows(self) -> None:
        hidden = torch.arange(1 * 5 * 2, dtype=torch.float32).reshape(1, 5, 2)
        patched, token_lengths = patch_temporal_sequence(
            hidden,
            torch.tensor([3], dtype=torch.long),
            patch_size=4,
            patch_stride=2,
        )
        self.assertTrue(torch.equal(token_lengths, torch.tensor([1])))
        self.assertEqual(tuple(patched.shape), (1, 1, 8))
        expected = torch.cat([hidden[0, :3], torch.zeros(1, 2)], dim=0).reshape(-1)
        self.assertTrue(torch.equal(patched[0, 0], expected))

    def test_possm_phoneme_model_pre_decoder_patch_s5_shapes_and_lengths(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base_encoder = recover_possm_stage1_encoder(
                checkpoint_path=_make_stage1_checkpoint(Path(tmpdir)),
            )[0]
            model = POSSMPhonemeModel(
                base_encoder=base_encoder,
                vocab_size=3,
                decoder_backbone_type="s5",
                s5_hidden_size=8,
                s5_state_size=4,
                s5_num_layers=1,
                s5_dropout=0.0,
                s5_direction="causal",
                emission_mode="pre_decoder_patch",
                pre_decoder_patch_size=3,
                pre_decoder_patch_stride=2,
            )
            x = torch.randn(2, 8, 5)
            lengths = torch.tensor([8, 5], dtype=torch.long)
            outputs = model(x, lengths, session_ids=["a", "b"])
            self.assertEqual(int(model.s5_sequence_decoder.input_size), base_encoder.hidden_size * 3)
            self.assertEqual(tuple(outputs["sequence_hidden"].shape), (2, 8, base_encoder.hidden_size))
            self.assertEqual(tuple(outputs["decoder_input"].shape), (2, 3, base_encoder.hidden_size * 3))
            self.assertTrue(torch.equal(outputs["decoder_input_lengths"], torch.tensor([3, 2])))
            self.assertEqual(tuple(outputs["decoder_hidden"].shape), (2, 3, 8))
            self.assertEqual(tuple(outputs["conv_hidden"].shape), (2, 3, 8))
            self.assertEqual(tuple(outputs["logits"].shape), (2, 3, 3))
            self.assertTrue(torch.equal(outputs["token_lengths"], torch.tensor([3, 2])))

    def test_possm_phoneme_model_bidirectional_s5_decoder_shapes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base_encoder = recover_possm_stage1_encoder(
                checkpoint_path=_make_stage1_checkpoint(Path(tmpdir)),
            )[0]
            model = POSSMPhonemeModel(
                base_encoder=base_encoder,
                vocab_size=3,
                decoder_backbone_type="s5",
                s5_hidden_size=8,
                s5_state_size=4,
                s5_num_layers=1,
                s5_dropout=0.0,
                s5_direction="bidirectional",
                conv_kernel_size=3,
                conv_stride=2,
            )
            outputs = model(
                torch.randn(2, 8, 5),
                torch.tensor([8, 6], dtype=torch.long),
                session_ids=["a", "b"],
            )
            self.assertEqual(tuple(outputs["decoder_hidden"].shape), (2, 8, 8))
            self.assertTrue(torch.allclose(outputs["decoder_hidden"][1, 6:], torch.zeros(2, 8)))
            self.assertTrue(torch.equal(outputs["token_lengths"], torch.tensor([4, 3])))

    def test_possm_phoneme_model_requires_session_ids_when_adapter_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base_encoder = recover_possm_stage1_encoder(
                checkpoint_path=_make_stage1_checkpoint(Path(tmpdir)),
            )[0]
            model = POSSMPhonemeModel(
                base_encoder=base_encoder,
                vocab_size=3,
                gru_hidden_size=8,
                gru_num_layers=1,
                gru_dropout=0.0,
                conv_kernel_size=3,
                conv_stride=1,
                session_adapter_enabled=True,
            )
            with self.assertRaisesRegex(ValueError, "required when session adapter is enabled"):
                model(torch.randn(1, 5, 5), torch.tensor([5], dtype=torch.long))

    def test_possm_phoneme_model_can_disable_session_adapter(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base_encoder = recover_possm_stage1_encoder(
                checkpoint_path=_make_stage1_checkpoint(Path(tmpdir)),
            )[0]
            model = POSSMPhonemeModel(
                base_encoder=base_encoder,
                vocab_size=3,
                gru_hidden_size=8,
                gru_num_layers=1,
                gru_dropout=0.0,
                conv_kernel_size=3,
                conv_stride=1,
                session_adapter_enabled=False,
            )
            x = torch.randn(1, 5, 5)
            outputs = model(x, torch.tensor([5], dtype=torch.long))
            self.assertTrue(torch.equal(outputs["adapted_input"], x))

    def test_probe_frozen_keeps_decoder_in_train_mode(self) -> None:
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
                conv_kernel_size=3,
                conv_stride=2,
            )
            _set_train_mode(model, train_encoder=False)
            self.assertFalse(model.base_encoder.training)
            self.assertTrue(model.session_input_adapter.training)
            self.assertTrue(model.gru.training)
            self.assertTrue(model.sequence_decoder.training)
            self.assertTrue(model.conv.training)
            self.assertTrue(model.conv_dropout.training)
            self.assertTrue(model.classifier.training)

    def test_stage2_decoder_train_modules_include_post_gru_conv(self) -> None:
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
                conv_kernel_size=3,
                conv_stride=2,
                session_adapter_enabled=True,
            )
            modules = _stage2_decoder_train_modules(model, session_adapter_enabled=True)
            module_param_ids = {
                id(param)
                for module in modules
                for param in module.parameters()
            }
            conv_param_ids = {id(param) for param in model.conv.parameters()}
            decoder_param_ids = {id(param) for param in model.sequence_decoder.parameters()}
            self.assertTrue(conv_param_ids)
            self.assertTrue(conv_param_ids.issubset(module_param_ids))
            self.assertTrue(decoder_param_ids)
            self.assertTrue(decoder_param_ids.issubset(module_param_ids))

    def test_stage2_decoder_train_modules_include_s5_decoder(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base_encoder = recover_possm_stage1_encoder(
                checkpoint_path=_make_stage1_checkpoint(Path(tmpdir)),
            )[0]
            model = POSSMPhonemeModel(
                base_encoder=base_encoder,
                vocab_size=3,
                decoder_backbone_type="s5",
                s5_hidden_size=8,
                s5_state_size=4,
                s5_num_layers=1,
                s5_dropout=0.0,
                conv_kernel_size=3,
                conv_stride=2,
                session_adapter_enabled=True,
            )
            modules = _stage2_decoder_train_modules(model, session_adapter_enabled=True)
            module_param_ids = {
                id(param)
                for module in modules
                for param in module.parameters()
            }
            decoder_param_ids = {id(param) for param in model.sequence_decoder.parameters()}
            self.assertTrue(decoder_param_ids)
            self.assertTrue(decoder_param_ids.issubset(module_param_ids))

    def test_stage2_decoder_train_modules_include_pre_patch_classifier(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base_encoder = recover_possm_stage1_encoder(
                checkpoint_path=_make_stage1_checkpoint(Path(tmpdir)),
            )[0]
            model = POSSMPhonemeModel(
                base_encoder=base_encoder,
                vocab_size=3,
                decoder_backbone_type="s5",
                s5_hidden_size=8,
                s5_state_size=4,
                s5_num_layers=1,
                s5_dropout=0.0,
                emission_mode="pre_decoder_patch",
                pre_decoder_patch_size=3,
                pre_decoder_patch_stride=2,
                session_adapter_enabled=True,
            )
            modules = _stage2_decoder_train_modules(model, session_adapter_enabled=True)
            module_param_ids = {
                id(param)
                for module in modules
                for param in module.parameters()
            }
            classifier_param_ids = {id(param) for param in model.classifier.parameters()}
            decoder_param_ids = {id(param) for param in model.sequence_decoder.parameters()}
            self.assertTrue(classifier_param_ids.issubset(module_param_ids))
            self.assertTrue(decoder_param_ids.issubset(module_param_ids))

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
                    signal_spec=SignalSpec.tx_sbp(tx_dim=3, sbp_dim=2),
                    data_mode="normalized",
                    batch_size=1,
                    num_steps=2,
                    learning_rate=1e-3,
                    encoder_learning_rate=3e-4,
                    checkpoint_every_steps=1,
                    input_smoothing_sigma_bins=2.0,
                    white_noise_sd=1.0,
                    constant_offset_sd=0.2,
                    gru_hidden_size=8,
                    gru_num_layers=2,
                    gru_dropout=0.0,
                    conv_kernel_size=3,
                    conv_stride=1,
                ),
                device=torch.device("cpu"),
            )
            self.assertEqual(summary["dataset"], "brain2text24")
            self.assertEqual(summary["feature_mode"], "tx_sbp")
            self.assertEqual(summary["split_policy"], "competition_train_test")
            self.assertEqual(summary["train_split_name"], "competition_train")
            self.assertEqual(summary["val_split_name"], "competition_test")
            self.assertEqual(summary["train_session_ids"], ["t00.2025.01.01"])
            self.assertEqual(summary["val_session_ids"], ["t00.2025.01.01"])
            self.assertEqual(int(summary["train_examples"]), 2)
            self.assertEqual(int(summary["val_examples"]), 1)
            self.assertTrue(bool(summary["dynamic_batching_enabled"]))
            self.assertEqual(int(summary["p95_train_input_length"]), 4)
            self.assertEqual(int(summary["max_padded_time_per_microbatch"]), 4)
            self.assertEqual(summary["train_microbatch_examples_range"], {"min": 1, "max": 1})
            self.assertEqual(summary["train_microbatch_max_input_length_range"], {"min": 4, "max": 4})
            self.assertTrue(Path(summary["checkpoint_final_path"]).exists())
            self.assertTrue((Path(summary["checkpoints_dir"]) / "step_000001.pt").exists())
            self.assertIn("val_ctc_bpphone", summary["metrics"])
            self.assertIn("best_val_ctc_bpphone", summary["metrics"])
            payload = torch.load(summary["checkpoint_final_path"], map_location="cpu")
            self.assertTrue(
                any(key.startswith("pre_decoder_backbone.") for key in payload["model_state"].keys())
            )
            self.assertEqual(int(payload["config"]["conv_kernel_size"]), 3)
            self.assertEqual(int(payload["config"]["conv_stride"]), 1)
            self.assertAlmostEqual(float(payload["config"]["input_smoothing_sigma_bins"]), 2.0)
            self.assertAlmostEqual(float(payload["config"]["white_noise_sd"]), 1.0)
            self.assertAlmostEqual(float(payload["config"]["constant_offset_sd"]), 0.2)
            self.assertEqual(str(payload["cache_root"]), str(tmp_path))
            self.assertEqual(str(payload["split_policy"]), "competition_train_test")
            self.assertEqual(str(payload["train_split_name"]), "competition_train")
            self.assertEqual(str(payload["val_split_name"]), "competition_test")
            self.assertEqual(int(payload["train_examples"]), 2)
            self.assertEqual(int(payload["val_examples"]), 1)
            self.assertTrue(bool(payload["dynamic_batching_enabled"]))
            self.assertEqual(int(payload["p95_train_input_length"]), 4)
            self.assertEqual(int(payload["max_padded_time_per_microbatch"]), 4)
            self.assertEqual(payload["train_microbatch_examples_range"], {"min": 1, "max": 1})
            self.assertEqual(payload["train_microbatch_max_input_length_range"], {"min": 4, "max": 4})
            self.assertTrue(bool(payload["config"]["session_adapter_enabled"]))
            self.assertTrue(bool(payload["session_adapter_keys"]))
            self.assertTrue(
                any(key.startswith("session_input_adapter.") for key in payload["model_state"].keys())
            )
            self.assertEqual(int(payload["model_state"]["gru.weight_ih_l0"].shape[1]), 7)
            self.assertTrue(any(key.startswith("conv.") for key in payload["model_state"].keys()))

    def test_run_possm_phoneme_finetuning_s5_decoder_smoke(self) -> None:
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
                    signal_spec=SignalSpec.tx_sbp(tx_dim=3, sbp_dim=2),
                    data_mode="normalized",
                    batch_size=1,
                    num_steps=1,
                    learning_rate=1e-3,
                    encoder_learning_rate=3e-4,
                    checkpoint_every_steps=1,
                    input_smoothing_sigma_bins=0.0,
                    decoder_backbone_type="s5",
                    s5_hidden_size=8,
                    s5_state_size=4,
                    s5_num_layers=1,
                    s5_dropout=0.0,
                    s5_direction="causal",
                    conv_kernel_size=3,
                    conv_stride=1,
                ),
                device=torch.device("cpu"),
            )
            self.assertEqual(summary["decoder_backbone_type"], "s5")
            self.assertEqual(summary["s5_direction"], "causal")
            self.assertTrue(Path(summary["checkpoint_final_path"]).exists())
            payload = torch.load(summary["checkpoint_final_path"], map_location="cpu")
            self.assertEqual(str(payload["config"]["decoder_backbone_type"]), "s5")
            self.assertNotIn("s5_implementation", payload["config"])
            self.assertTrue(
                any(key.startswith("s5_sequence_decoder.") for key in payload["model_state"].keys())
            )
            self.assertFalse(any(key.startswith("gru.") for key in payload["model_state"].keys()))

    def test_run_possm_phoneme_finetuning_auto_discovers_canonical_split_stats(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            cache_root = tmp_path / "cache_v1"
            checkpoint_path = _make_stage1_checkpoint(tmp_path, temporal_gru_hidden_size=7)
            _write_tiny_canonical_probe_cache(cache_root)
            stats_path = (
                tmp_path
                / "stats"
                / "split_feature_stats"
                / "raw"
                / "brain2text24"
                / "competition_train"
                / "tx_sbp"
                / "global_v1.pt"
            )
            _write_valid_split_stats_artifact(
                cache_root=cache_root,
                stats_path=stats_path,
                dataset="brain2text24",
                signal_spec=SignalSpec.tx_sbp(tx_dim=3, sbp_dim=2),
                boundary_key_mode="session",
                split_policy="competition_train_test",
                train_split_name="competition_train",
                val_split_name="competition_test",
            )

            with mock.patch(
                "possm_ssl.phoneme_finetune.compute_feature_stats",
                side_effect=AssertionError("should not recompute split stats"),
            ):
                summary = run_possm_phoneme_finetuning(
                    checkpoint_path=checkpoint_path,
                    cache_root=cache_root,
                    config=POSSMFinetuneConfig(
                        seed=7,
                        mode="probe_frozen",
                        dataset="brain2text24",
                        signal_spec=SignalSpec.tx_sbp(tx_dim=3, sbp_dim=2),
                        data_mode="normalized",
                        batch_size=1,
                        num_steps=1,
                        learning_rate=1e-3,
                        encoder_learning_rate=3e-4,
                        checkpoint_every_steps=1,
                        input_smoothing_sigma_bins=0.0,
                        gru_hidden_size=8,
                        gru_num_layers=2,
                        gru_dropout=0.0,
                        conv_kernel_size=3,
                        conv_stride=1,
                    ),
                    device=torch.device("cpu"),
                )

            self.assertEqual(summary["precomputed_split_stats_path"], str(stats_path))
            self.assertEqual(
                summary["precomputed_split_stats_metadata"]["feature_mode"],
                "tx_sbp",
            )
            self.assertEqual(
                int(summary["precomputed_split_stats_metadata"]["feature_dim"]),
                5,
            )

    def test_run_possm_phoneme_finetuning_fails_for_missing_split_stats_sidecar(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            cache_root = tmp_path / "cache_v1"
            checkpoint_path = _make_stage1_checkpoint(tmp_path, temporal_gru_hidden_size=7)
            _write_tiny_canonical_probe_cache(cache_root)
            stats_path = (
                tmp_path
                / "stats"
                / "split_feature_stats"
                / "raw"
                / "brain2text24"
                / "competition_train"
                / "tx_sbp"
                / "global_v1.pt"
            )
            _write_valid_split_stats_artifact(
                cache_root=cache_root,
                stats_path=stats_path,
                dataset="brain2text24",
                signal_spec=SignalSpec.tx_sbp(tx_dim=3, sbp_dim=2),
                boundary_key_mode="session",
                split_policy="competition_train_test",
                train_split_name="competition_train",
                val_split_name="competition_test",
            )
            stats_path.with_suffix(".json").unlink()

            with self.assertRaisesRegex(FileNotFoundError, "missing_sidecar"):
                run_possm_phoneme_finetuning(
                    checkpoint_path=checkpoint_path,
                    cache_root=cache_root,
                    config=POSSMFinetuneConfig(
                        seed=7,
                        mode="probe_frozen",
                        dataset="brain2text24",
                        signal_spec=SignalSpec.tx_sbp(tx_dim=3, sbp_dim=2),
                        data_mode="normalized",
                        batch_size=1,
                        num_steps=1,
                        learning_rate=1e-3,
                        encoder_learning_rate=3e-4,
                        checkpoint_every_steps=1,
                        input_smoothing_sigma_bins=0.0,
                        gru_hidden_size=8,
                        gru_num_layers=2,
                        gru_dropout=0.0,
                        conv_kernel_size=3,
                        conv_stride=1,
                    ),
                    device=torch.device("cpu"),
                )

    def test_run_possm_phoneme_finetuning_fails_for_wrong_split_stats_kind(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            cache_root = tmp_path / "cache_v1"
            checkpoint_path = _make_stage1_checkpoint(tmp_path, temporal_gru_hidden_size=7)
            _write_tiny_canonical_probe_cache(cache_root)
            stats_path = (
                tmp_path
                / "stats"
                / "split_feature_stats"
                / "raw"
                / "brain2text24"
                / "competition_train"
                / "tx_sbp"
                / "global_v1.pt"
            )
            payload = {
                "session_feature_stats": {
                    "brain2text24:t00.2025.01.01": (torch.zeros(5), torch.ones(5)),
                },
                "metadata": {
                    "kind": "session_featurewise_zscore_stats",
                    "source_cache_root": str(cache_root.resolve()),
                    "source_cache_name": cache_root.name,
                    "source_cache_variant": "raw",
                    "source_cache_signature": _compute_cache_source_signature(cache_root),
                    "feature_mode": "tx_sbp",
                    "boundary_key_mode": "session",
                    "tx_dim": 3,
                    "sbp_dim": 2,
                    "full_dim": 5,
                    "feature_policy": "area6v_v1",
                },
            }
            stats_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(payload, stats_path)
            stats_path.with_suffix(".json").write_text(json.dumps(payload["metadata"], indent=2) + "\n")

            with self.assertRaisesRegex(ValueError, "wrong kind"):
                run_possm_phoneme_finetuning(
                    checkpoint_path=checkpoint_path,
                    cache_root=cache_root,
                    config=POSSMFinetuneConfig(
                        seed=7,
                        mode="probe_frozen",
                        dataset="brain2text24",
                        signal_spec=SignalSpec.tx_sbp(tx_dim=3, sbp_dim=2),
                        data_mode="normalized",
                        batch_size=1,
                        num_steps=1,
                        learning_rate=1e-3,
                        encoder_learning_rate=3e-4,
                        checkpoint_every_steps=1,
                        input_smoothing_sigma_bins=0.0,
                        gru_hidden_size=8,
                        gru_num_layers=2,
                        gru_dropout=0.0,
                        conv_kernel_size=3,
                        conv_stride=1,
                    ),
                    device=torch.device("cpu"),
                )

    def test_run_possm_phoneme_finetuning_fails_for_stale_split_stats_signature(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            cache_root = tmp_path / "cache_v1"
            checkpoint_path = _make_stage1_checkpoint(tmp_path, temporal_gru_hidden_size=7)
            _write_tiny_canonical_probe_cache(cache_root)
            stats_path = (
                tmp_path
                / "stats"
                / "split_feature_stats"
                / "raw"
                / "brain2text24"
                / "competition_train"
                / "tx_sbp"
                / "global_v1.pt"
            )
            _write_valid_split_stats_artifact(
                cache_root=cache_root,
                stats_path=stats_path,
                dataset="brain2text24",
                signal_spec=SignalSpec.tx_sbp(tx_dim=3, sbp_dim=2),
                boundary_key_mode="session",
                split_policy="competition_train_test",
                train_split_name="competition_train",
                val_split_name="competition_test",
            )
            payload = torch.load(stats_path, map_location="cpu")
            payload["metadata"]["source_cache_signature"] = "stale"
            torch.save(payload, stats_path)
            stats_path.with_suffix(".json").write_text(json.dumps(payload["metadata"], indent=2) + "\n")

            with self.assertRaisesRegex(ValueError, "source_cache_signature"):
                run_possm_phoneme_finetuning(
                    checkpoint_path=checkpoint_path,
                    cache_root=cache_root,
                    config=POSSMFinetuneConfig(
                        seed=7,
                        mode="probe_frozen",
                        dataset="brain2text24",
                        signal_spec=SignalSpec.tx_sbp(tx_dim=3, sbp_dim=2),
                        data_mode="normalized",
                        batch_size=1,
                        num_steps=1,
                        learning_rate=1e-3,
                        encoder_learning_rate=3e-4,
                        checkpoint_every_steps=1,
                        input_smoothing_sigma_bins=0.0,
                        gru_hidden_size=8,
                        gru_num_layers=2,
                        gru_dropout=0.0,
                        conv_kernel_size=3,
                        conv_stride=1,
                    ),
                    device=torch.device("cpu"),
                )

    def test_recover_possm_stage2_summary_prefers_latest_step_for_interrupted_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            checkpoint_path = _make_stage1_checkpoint(tmp_path, temporal_gru_hidden_size=7)
            _write_tiny_canonical_probe_cache(tmp_path)
            output_root = tmp_path / "stage2_runs"
            summary = run_possm_phoneme_finetuning(
                checkpoint_path=checkpoint_path,
                cache_root=tmp_path,
                output_root=output_root,
                run_name="interrupted_run",
                config=POSSMFinetuneConfig(
                    seed=7,
                    mode="probe_frozen",
                    dataset="brain2text24",
                    signal_spec=SignalSpec.tx_sbp(tx_dim=3, sbp_dim=2),
                    data_mode="normalized",
                    batch_size=1,
                    num_steps=2,
                    learning_rate=1e-3,
                    encoder_learning_rate=3e-4,
                    checkpoint_every_steps=1,
                    input_smoothing_sigma_bins=2.0,
                    gru_hidden_size=8,
                    gru_num_layers=2,
                    gru_dropout=0.0,
                    conv_kernel_size=3,
                    conv_stride=1,
                ),
                device=torch.device("cpu"),
            )
            Path(summary["checkpoint_final_path"]).unlink()

            recovered = recover_possm_stage2_summary(output_root)
            latest_run_dir = find_latest_possm_stage2_run_dir(output_root)

            self.assertEqual(latest_run_dir.name, "interrupted_run")
            self.assertIsNone(recovered["checkpoint_final_path"])
            self.assertTrue(str(recovered["resume_checkpoint_path"]).endswith("step_000002.pt"))
            self.assertEqual(recovered["run_name"], "interrupted_run")
            self.assertEqual(recovered["stage1_checkpoint_path"], str(checkpoint_path))
            self.assertEqual(recovered["cache_root"], str(tmp_path))
            self.assertEqual(int(recovered["steps"]), 2)
            self.assertEqual(int(recovered["config"]["num_steps"]), 2)
            self.assertIn("val_ctc_bpphone", recovered["metrics"])

    def test_recover_possm_stage2_summary_prefers_final_when_it_is_ahead(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_root = Path(tmpdir) / "stage2_runs"
            run_dir = output_root / "stage2_run"
            checkpoints_dir = run_dir / "checkpoints"
            checkpoints_dir.mkdir(parents=True)
            config = asdict(POSSMFinetuneConfig(num_steps=4200))
            common_payload = {
                "stage": "stage2_phoneme_finetune",
                "stage1_checkpoint_path": "stage1.pt",
                "cache_root": str(Path(tmpdir) / "cache"),
                "config": config,
                "metrics": {"val_ctc_bpphone": 1.0},
            }
            torch.save({**common_payload, "steps": 4000}, checkpoints_dir / "step_004000.pt")
            torch.save({**common_payload, "steps": 4200}, run_dir / "checkpoint_final.pt")
            torch.save({**common_payload, "steps": 3900}, run_dir / "checkpoint_best.pt")

            recovered = recover_possm_stage2_summary(output_root)

            self.assertEqual(recovered["resume_checkpoint_path"], str(run_dir / "checkpoint_final.pt"))
            self.assertEqual(int(recovered["steps"]), 4200)

    def test_run_possm_phoneme_finetuning_resume_from_latest_loads_step_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            checkpoint_path = _make_stage1_checkpoint(tmp_path, temporal_gru_hidden_size=7)
            _write_tiny_canonical_probe_cache(tmp_path)
            output_root = tmp_path / "stage2_runs"
            config = POSSMFinetuneConfig(
                seed=7,
                mode="probe_frozen",
                dataset="brain2text24",
                signal_spec=SignalSpec.tx_sbp(tx_dim=3, sbp_dim=2),
                data_mode="normalized",
                batch_size=1,
                num_steps=1,
                learning_rate=1e-3,
                encoder_learning_rate=3e-4,
                checkpoint_every_steps=1,
                input_smoothing_sigma_bins=2.0,
                gru_hidden_size=8,
                gru_num_layers=2,
                gru_dropout=0.0,
                conv_kernel_size=3,
                conv_stride=1,
            )
            first_summary = run_possm_phoneme_finetuning(
                checkpoint_path=checkpoint_path,
                cache_root=tmp_path,
                output_root=output_root,
                run_name="resume_run",
                config=config,
                device=torch.device("cpu"),
            )
            step_path = Path(first_summary["checkpoints_dir"]) / "step_000001.pt"
            self.assertTrue(step_path.exists())

            resumed_summary = run_possm_phoneme_finetuning(
                checkpoint_path=checkpoint_path,
                cache_root=tmp_path,
                output_root=output_root,
                run_name="resume_run",
                config=config,
                device=torch.device("cpu"),
                resume_from_latest=True,
            )

            self.assertEqual(resumed_summary["resumed_from_checkpoint"], str(step_path))
            self.assertTrue(bool(resumed_summary["dynamic_batching_enabled"]))
            self.assertEqual(int(resumed_summary["steps"]), 1)
            final_payload = torch.load(resumed_summary["checkpoint_final_path"], map_location="cpu")
            self.assertIn("optimizer_state", final_payload)
            self.assertIn("rng_state", final_payload)
            self.assertIn("train_batch_position", final_payload)
            self.assertTrue(bool(final_payload["dynamic_batching_enabled"]))

    def test_stage2_resume_reproduces_uninterrupted_next_step(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            checkpoint_path = _make_stage1_checkpoint(
                tmp_path,
                temporal_gru_hidden_size=7,
            )
            _write_tiny_canonical_probe_cache(tmp_path)
            output_root = tmp_path / "stage2_runs"
            config = POSSMFinetuneConfig(
                seed=7,
                mode="probe_frozen",
                dataset="brain2text24",
                signal_spec=SignalSpec.tx_sbp(tx_dim=3, sbp_dim=2),
                data_mode="normalized",
                batch_size=1,
                num_steps=2,
                learning_rate=1e-3,
                encoder_learning_rate=3e-4,
                checkpoint_every_steps=1,
                input_smoothing_sigma_bins=2.0,
                white_noise_sd=0.1,
                constant_offset_sd=0.05,
                gru_hidden_size=8,
                gru_num_layers=2,
                gru_dropout=0.0,
                conv_kernel_size=3,
                conv_stride=1,
            )
            first_summary = run_possm_phoneme_finetuning(
                checkpoint_path=checkpoint_path,
                cache_root=tmp_path,
                output_root=output_root,
                run_name="exact_resume_run",
                config=config,
                device=torch.device("cpu"),
            )
            final_path = Path(first_summary["checkpoint_final_path"])
            uninterrupted_payload = torch.load(
                final_path,
                map_location="cpu",
                weights_only=False,
            )
            step_two_path = Path(first_summary["checkpoints_dir"]) / "step_000002.pt"
            step_two_path.unlink()
            final_path.unlink()

            resumed_summary = run_possm_phoneme_finetuning(
                checkpoint_path=checkpoint_path,
                cache_root=tmp_path,
                output_root=output_root,
                run_name="exact_resume_run",
                config=config,
                device=torch.device("cpu"),
                resume_from_latest=True,
            )
            resumed_payload = torch.load(
                resumed_summary["checkpoint_final_path"],
                map_location="cpu",
                weights_only=False,
            )

            self.assertEqual(int(resumed_payload["steps"]), 2)
            for name, expected in uninterrupted_payload["model_state"].items():
                torch.testing.assert_close(
                    resumed_payload["model_state"][name],
                    expected,
                    rtol=0.0,
                    atol=0.0,
                )

    def test_sbp_only_stage1_checkpoint_finetunes_and_resumes_stage2(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            checkpoint_path = _make_stage1_checkpoint(
                tmp_path,
                temporal_gru_hidden_size=7,
                feature_mode="sbp_only",
            )
            _write_tiny_canonical_probe_cache(tmp_path)
            output_root = tmp_path / "sbp_stage2_runs"
            config = POSSMFinetuneConfig(
                seed=7,
                mode="finetune_full",
                dataset="brain2text24",
                signal_spec=SignalSpec.sbp_only(sbp_dim=2),
                data_mode="normalized",
                batch_size=1,
                num_steps=1,
                learning_rate=1e-3,
                encoder_learning_rate=3e-4,
                checkpoint_every_steps=1,
                input_smoothing_sigma_bins=0.0,
                gru_hidden_size=8,
                gru_num_layers=1,
                gru_dropout=0.0,
                conv_kernel_size=3,
                conv_stride=1,
            )

            first_summary = run_possm_phoneme_finetuning(
                checkpoint_path=checkpoint_path,
                cache_root=tmp_path,
                output_root=output_root,
                run_name="sbp_resume_run",
                config=config,
                device=torch.device("cpu"),
            )
            step_path = Path(first_summary["checkpoints_dir"]) / "step_000001.pt"
            self.assertTrue(step_path.exists())
            self.assertEqual(first_summary["feature_mode"], "sbp_only")
            first_payload = torch.load(
                first_summary["checkpoint_final_path"],
                map_location="cpu",
            )
            self.assertEqual(first_payload["config"]["signal_spec"]["mode"], "sbp_only")
            self.assertEqual(
                int(first_payload["model_state"]["base_encoder.unit_embedding.weight"].shape[0]),
                2,
            )

            resumed_summary = run_possm_phoneme_finetuning(
                checkpoint_path=checkpoint_path,
                cache_root=tmp_path,
                output_root=output_root,
                run_name="sbp_resume_run",
                config=config,
                device=torch.device("cpu"),
                resume_from_latest=True,
            )
            self.assertEqual(
                resumed_summary["resumed_from_checkpoint"],
                str(step_path),
            )
            self.assertEqual(resumed_summary["feature_mode"], "sbp_only")
            resumed_payload = torch.load(
                resumed_summary["checkpoint_final_path"],
                map_location="cpu",
            )
            self.assertEqual(resumed_payload["config"]["signal_spec"]["mode"], "sbp_only")
            self.assertIn("optimizer_state", resumed_payload)

    def test_run_possm_phoneme_finetuning_resume_tolerates_older_config_payload(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            checkpoint_path = _make_stage1_checkpoint(tmp_path, temporal_gru_hidden_size=7)
            _write_tiny_canonical_probe_cache(tmp_path)
            output_root = tmp_path / "stage2_runs"
            config = POSSMFinetuneConfig(
                seed=7,
                mode="probe_frozen",
                dataset="brain2text24",
                signal_spec=SignalSpec.tx_sbp(tx_dim=3, sbp_dim=2),
                data_mode="normalized",
                batch_size=1,
                num_steps=1,
                learning_rate=1e-3,
                encoder_learning_rate=3e-4,
                checkpoint_every_steps=1,
                input_smoothing_sigma_bins=2.0,
                gru_hidden_size=8,
                gru_num_layers=2,
                gru_dropout=0.0,
                conv_kernel_size=3,
                conv_stride=1,
            )
            first_summary = run_possm_phoneme_finetuning(
                checkpoint_path=checkpoint_path,
                cache_root=tmp_path,
                output_root=output_root,
                run_name="resume_old_config_run",
                config=config,
                device=torch.device("cpu"),
            )
            step_path = Path(first_summary["checkpoints_dir"]) / "step_000001.pt"
            payload = torch.load(step_path, map_location="cpu")
            assert isinstance(payload["config"], dict)
            payload["config"].pop("precomputed_split_stats_path", None)
            torch.save(payload, step_path)

            resumed_summary = run_possm_phoneme_finetuning(
                checkpoint_path=checkpoint_path,
                cache_root=tmp_path,
                output_root=output_root,
                run_name="resume_old_config_run",
                config=config,
                device=torch.device("cpu"),
                resume_from_latest=True,
            )

            self.assertEqual(resumed_summary["resumed_from_checkpoint"], str(step_path))

    def test_run_possm_phoneme_finetuning_decouples_validation_from_checkpointing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            checkpoint_path = _make_stage1_checkpoint(tmp_path, temporal_gru_hidden_size=7)
            _write_tiny_canonical_probe_cache(tmp_path)
            output_root = tmp_path / "stage2_runs"

            summary = run_possm_phoneme_finetuning(
                checkpoint_path=checkpoint_path,
                cache_root=tmp_path,
                output_root=output_root,
                run_name="decoupled_eval_run",
                config=POSSMFinetuneConfig(
                    seed=7,
                    mode="probe_frozen",
                    dataset="brain2text24",
                    signal_spec=SignalSpec.tx_sbp(tx_dim=3, sbp_dim=2),
                    data_mode="normalized",
                    batch_size=1,
                    num_steps=2,
                    learning_rate=1e-3,
                    encoder_learning_rate=3e-4,
                    val_every_steps=1,
                    checkpoint_every_steps=2,
                    checkpoint_keep_last=1,
                    input_smoothing_sigma_bins=2.0,
                    gru_hidden_size=8,
                    gru_num_layers=2,
                    gru_dropout=0.0,
                    conv_kernel_size=3,
                    conv_stride=1,
                ),
                device=torch.device("cpu"),
            )

            progress_records = [
                json.loads(line)
                for line in Path(summary["progress_log_path"]).read_text().splitlines()
                if line.strip()
            ]
            val_steps = [int(record["step"]) for record in progress_records if record.get("event") == "phoneme_val_report"]
            self.assertEqual(val_steps, [1, 2])
            checkpoint_paths = sorted(Path(summary["checkpoints_dir"]).glob("step_*.pt"))
            self.assertEqual([path.name for path in checkpoint_paths], ["step_000002.pt"])

    def test_stage2_progress_logs_include_per_and_blank_rate(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            checkpoint_path = _make_stage1_checkpoint(tmp_path, temporal_gru_hidden_size=7)
            _write_tiny_canonical_probe_cache(tmp_path)

            summary = run_possm_phoneme_finetuning(
                checkpoint_path=checkpoint_path,
                cache_root=tmp_path,
                output_root=tmp_path / "stage2_runs",
                run_name="val_logging_run",
                config=POSSMFinetuneConfig(
                    seed=7,
                    mode="probe_frozen",
                    dataset="brain2text24",
                    signal_spec=SignalSpec.tx_sbp(tx_dim=3, sbp_dim=2),
                    data_mode="normalized",
                    batch_size=1,
                    num_steps=1,
                    learning_rate=1e-3,
                    encoder_learning_rate=3e-4,
                    val_every_steps=1,
                    checkpoint_every_steps=10,
                    input_smoothing_sigma_bins=2.0,
                    gru_hidden_size=8,
                    gru_num_layers=2,
                    gru_dropout=0.0,
                    conv_kernel_size=3,
                    conv_stride=1,
                ),
                device=torch.device("cpu"),
            )

            progress_records = [
                json.loads(line)
                for line in Path(summary["progress_log_path"]).read_text().splitlines()
                if line.strip()
            ]
            train_record = next(record for record in progress_records if record.get("event") == "phoneme_train_report")
            val_record = next(record for record in progress_records if record.get("event") == "phoneme_val_report")
            self.assertIn("sample_seconds", train_record)
            self.assertIn("model_seconds", train_record)
            self.assertGreaterEqual(float(train_record["sample_seconds"]), 0.0)
            self.assertGreaterEqual(float(train_record["model_seconds"]), 0.0)
            self.assertIn("val_phoneme_error_rate", val_record)
            self.assertIn("blank_frame_rate", val_record)

    def test_possm_reporting_helpers_tolerate_missing_progress(self) -> None:
        summary = {
            "run_name": "missing_progress",
            "steps": 0,
            "progress_log_path": "/tmp/does-not-exist-possm-progress.jsonl",
            "resume_checkpoint_path": None,
            "checkpoint_best_path": None,
            "checkpoint_final_path": None,
            "metrics": {},
        }

        self.assertTrue(summarize_possm_stage2_progress(summary).empty)
        frames = display_possm_stage2_summary(summary)
        self.assertEqual(len(frames["summary"]), 1)
        self.assertTrue(frames["collapse"].empty)

    def test_run_possm_phoneme_finetuning_flushes_partial_accumulation_at_epoch_end(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            checkpoint_path = _make_stage1_checkpoint(tmp_path, temporal_gru_hidden_size=7)
            _write_tiny_canonical_probe_cache(tmp_path)
            manifest_path = tmp_path / "brain2text24" / "manifest.jsonl"
            rows = [json.loads(line) for line in manifest_path.read_text().splitlines()]
            for row in rows:
                if row["example_id"] == "holdout-0":
                    row["source_split"] = "competition_train"
            manifest_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            _write_valid_split_stats_artifact(
                cache_root=tmp_path,
                stats_path=resolve_canonical_split_stats_path(
                    cache_root=tmp_path,
                    dataset="brain2text24",
                    train_split_name="competition_train",
                    signal_spec=SignalSpec.tx_sbp(tx_dim=3, sbp_dim=2),
                    preferred_path=None,
                ),
                dataset="brain2text24",
                signal_spec=SignalSpec.tx_sbp(tx_dim=3, sbp_dim=2),
                boundary_key_mode="session",
                split_policy="competition_train_test",
                train_split_name="competition_train",
                val_split_name="competition_test",
            )

            summary = run_possm_phoneme_finetuning(
                checkpoint_path=checkpoint_path,
                cache_root=tmp_path,
                config=POSSMFinetuneConfig(
                    seed=7,
                    mode="probe_frozen",
                    dataset="brain2text24",
                    signal_spec=SignalSpec.tx_sbp(tx_dim=3, sbp_dim=2),
                    data_mode="normalized",
                    batch_size=2,
                    num_steps=2,
                    learning_rate=1e-3,
                    encoder_learning_rate=3e-4,
                    checkpoint_every_steps=1,
                    input_smoothing_sigma_bins=2.0,
                    gru_hidden_size=8,
                    gru_num_layers=2,
                    gru_dropout=0.0,
                    conv_kernel_size=3,
                    conv_stride=1,
                ),
                device=torch.device("cpu"),
            )

            self.assertEqual(int(summary["train_examples"]), 3)
            self.assertEqual(int(summary["steps"]), 2)
            self.assertEqual(summary["train_microbatch_examples_range"], {"min": 1, "max": 2})


if __name__ == "__main__":
    unittest.main()

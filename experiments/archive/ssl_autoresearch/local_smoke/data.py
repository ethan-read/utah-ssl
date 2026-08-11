"""Fixed data loading for the local SSL autoresearch smoke test."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass(frozen=True)
class SmokeDataBundle:
    train_x: torch.Tensor
    train_session_idx: torch.Tensor
    val_x: torch.Tensor
    val_session_idx: torch.Tensor
    session_names: tuple[str, ...]
    train_session_names: tuple[str, ...]
    val_session_names: tuple[str, ...]
    input_dim: int
    seq_len: int
    num_sessions: int


def _common_session_bases(tx_cache_dir: Path, sbp_cache_dir: Path) -> list[str]:
    tx_bases = {path.name.replace("_64.npy", "") for path in tx_cache_dir.glob("*_64.npy")}
    sbp_bases = {path.name.replace("_sbp_64.npy", "") for path in sbp_cache_dir.glob("*_sbp_64.npy")}
    return sorted(tx_bases & sbp_bases)


def _to_sequence_features(tx_windows: np.ndarray, sbp_windows: np.ndarray) -> np.ndarray:
    tx_seq = np.transpose(tx_windows, (0, 2, 1))
    sbp_seq = np.transpose(sbp_windows, (0, 2, 1))
    return np.concatenate([tx_seq, sbp_seq], axis=-1).astype(np.float32)


def _standardize_subject_level(
    train_by_session: list[np.ndarray],
    val_by_session: list[np.ndarray],
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    train_all = np.concatenate(train_by_session, axis=0)
    mean = train_all.mean(axis=(0, 1), keepdims=True)
    std = train_all.std(axis=(0, 1), keepdims=True) + 1e-6
    train_out = [(session - mean) / std for session in train_by_session]
    val_out = [(session - mean) / std for session in val_by_session]
    return train_out, val_out


def _standardize_session_level(
    train_by_session: list[np.ndarray],
    val_by_session: list[np.ndarray],
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    train_out: list[np.ndarray] = []
    val_out: list[np.ndarray] = []
    for train_session in train_by_session:
        mean = train_session.mean(axis=(0, 1), keepdims=True)
        std = train_session.std(axis=(0, 1), keepdims=True) + 1e-6
        train_out.append((train_session - mean) / std)
    for val_session in val_by_session:
        mean = val_session.mean(axis=(0, 1), keepdims=True)
        std = val_session.std(axis=(0, 1), keepdims=True) + 1e-6
        val_out.append((val_session - mean) / std)
    return train_out, val_out


def load_local_smoke_bundle(
    tx_cache_dir: Path,
    sbp_cache_dir: Path,
    *,
    session_limit: int,
    session_selection: str,
    val_session_count: int,
    train_windows_per_session: int,
    val_windows_per_session: int,
    standardize_scope: str,
    seed: int,
) -> SmokeDataBundle:
    common_sessions = _common_session_bases(tx_cache_dir, sbp_cache_dir)
    if session_selection == "latest":
        selected_sessions = common_sessions[-session_limit:]
    elif session_selection == "earliest":
        selected_sessions = common_sessions[:session_limit]
    else:
        raise ValueError(f"Unsupported session_selection: {session_selection}")
    if not selected_sessions:
        raise FileNotFoundError("No matched TX/SBP cache sessions were found for the local smoke benchmark.")
    if val_session_count < 1 or val_session_count >= len(selected_sessions):
        raise ValueError("val_session_count must be at least 1 and smaller than the number of selected sessions.")

    train_session_names = selected_sessions[:-val_session_count]
    val_session_names = selected_sessions[-val_session_count:]
    session_names = train_session_names + val_session_names

    train_by_session: list[np.ndarray] = []
    val_by_session: list[np.ndarray] = []

    for session_offset, session_base in enumerate(train_session_names):
        tx_path = tx_cache_dir / f"{session_base}_64.npy"
        sbp_path = sbp_cache_dir / f"{session_base}_sbp_64.npy"

        tx = np.load(tx_path, mmap_mode="r")
        sbp = np.load(sbp_path, mmap_mode="r")
        n_windows = min(tx.shape[0], sbp.shape[0])

        rng = np.random.default_rng(seed + session_offset)
        indices = rng.permutation(n_windows)
        train_count = min(train_windows_per_session, max(1, n_windows))
        train_indices = indices[:train_count]
        train_by_session.append(_to_sequence_features(tx[train_indices], sbp[train_indices]))

    for session_offset, session_base in enumerate(val_session_names, start=len(train_session_names)):
        tx_path = tx_cache_dir / f"{session_base}_64.npy"
        sbp_path = sbp_cache_dir / f"{session_base}_sbp_64.npy"

        tx = np.load(tx_path, mmap_mode="r")
        sbp = np.load(sbp_path, mmap_mode="r")
        n_windows = min(tx.shape[0], sbp.shape[0])

        rng = np.random.default_rng(seed + session_offset)
        indices = rng.permutation(n_windows)
        val_count = min(val_windows_per_session, max(1, n_windows))
        val_indices = indices[:val_count]
        val_by_session.append(_to_sequence_features(tx[val_indices], sbp[val_indices]))

    if standardize_scope == "subject":
        train_by_session, val_by_session = _standardize_subject_level(train_by_session, val_by_session)
    elif standardize_scope == "session":
        train_by_session, val_by_session = _standardize_session_level(train_by_session, val_by_session)
    else:
        raise ValueError(f"Unsupported standardize_scope: {standardize_scope}")

    train_session_idx = []
    val_session_idx = []
    for session_idx, train_session in enumerate(train_by_session):
        train_session_idx.append(np.full(train_session.shape[0], session_idx, dtype=np.int64))
    val_offset = len(train_by_session)
    for session_idx, val_session in enumerate(val_by_session, start=val_offset):
        val_session_idx.append(np.full(val_session.shape[0], session_idx, dtype=np.int64))

    train_x = torch.from_numpy(np.concatenate(train_by_session, axis=0))
    val_x = torch.from_numpy(np.concatenate(val_by_session, axis=0))
    train_idx = torch.from_numpy(np.concatenate(train_session_idx, axis=0))
    val_idx = torch.from_numpy(np.concatenate(val_session_idx, axis=0))

    return SmokeDataBundle(
        train_x=train_x,
        train_session_idx=train_idx,
        val_x=val_x,
        val_session_idx=val_idx,
        session_names=tuple(session_names),
        train_session_names=tuple(train_session_names),
        val_session_names=tuple(val_session_names),
        input_dim=int(train_x.shape[-1]),
        seq_len=int(train_x.shape[1]),
        num_sessions=len(session_names),
    )

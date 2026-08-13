"""Canonical keys for session- and subject-scoped neural data."""

from __future__ import annotations


def resolve_boundary_key(
    *,
    dataset: str,
    session_id: str,
    subject_id: str | None,
    boundary_key_mode: str,
) -> str:
    """Resolve the key used by normalization and session-specific adapters."""

    if boundary_key_mode == "session":
        return f"{dataset}:{session_id}"
    if boundary_key_mode == "subject_if_available":
        return f"{dataset}:{subject_id or session_id}"
    raise ValueError(f"Unsupported boundary_key_mode: {boundary_key_mode}")


__all__ = ["resolve_boundary_key"]

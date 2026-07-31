"""Canonical, backward-compatible normalization-stat artifact helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import torch


FEATURE_STATS_SCHEMA = "feature_stats_v1"
SUPPORTED_NORMALIZATION_SCOPES = ("session", "global")


def _as_stats_pair(value: Any, *, key: str) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(value, Mapping):
        mean = value.get("mean")
        std = value.get("std")
    elif isinstance(value, (tuple, list)) and len(value) == 2:
        mean, std = value
    else:
        raise ValueError(
            f"Feature stats entry {key!r} must be a mean/std mapping or 2-item pair."
        )
    if mean is None or std is None:
        raise ValueError(f"Feature stats entry {key!r} is missing mean or std.")
    return torch.as_tensor(mean).float().cpu(), torch.as_tensor(std).float().cpu()


def normalize_feature_stats_entries(
    entries: Mapping[str, Any],
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """Normalize canonical or legacy entries to ``key -> (mean, std)``."""
    normalized = {
        str(key): _as_stats_pair(value, key=str(key))
        for key, value in entries.items()
    }
    if not normalized:
        raise ValueError("Feature stats must contain at least one entry.")
    return normalized


def build_feature_stats_payload(
    *,
    scope: str,
    entries: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """Build one canonical payload while retaining established B2T24 keys."""
    resolved_scope = str(scope)
    if resolved_scope not in SUPPORTED_NORMALIZATION_SCOPES:
        raise ValueError(
            f"scope must be one of {SUPPORTED_NORMALIZATION_SCOPES}; got {resolved_scope!r}"
        )
    normalized_entries = normalize_feature_stats_entries(entries)
    resolved_metadata = {
        **dict(metadata),
        "stats_schema": FEATURE_STATS_SCHEMA,
        "normalization_scope": resolved_scope,
    }
    payload: dict[str, Any] = {
        "feature_stats": normalized_entries,
        "metadata": resolved_metadata,
    }
    if resolved_scope == "session":
        payload["session_feature_stats"] = normalized_entries
    else:
        if set(normalized_entries) != {"global"}:
            raise ValueError("Global feature stats must contain exactly one 'global' entry.")
        payload["mean"], payload["std"] = normalized_entries["global"]
    return payload


def extract_feature_stats_entries(
    payload: Mapping[str, Any],
) -> tuple[str, dict[str, tuple[torch.Tensor, torch.Tensor]]]:
    """Read canonical payloads and both established B2T24 legacy shapes."""
    metadata = payload.get("metadata")
    metadata_scope = (
        str(metadata.get("normalization_scope"))
        if isinstance(metadata, Mapping) and metadata.get("normalization_scope")
        else None
    )
    canonical = payload.get("feature_stats")
    if isinstance(canonical, Mapping):
        scope = metadata_scope or (
            "global" if set(canonical) == {"global"} else "session"
        )
        if scope not in SUPPORTED_NORMALIZATION_SCOPES:
            raise ValueError(f"Unsupported normalization scope: {scope!r}")
        normalized = normalize_feature_stats_entries(canonical)
        if scope == "global" and set(normalized) != {"global"}:
            raise ValueError("Global feature stats must contain exactly one 'global' entry.")
        return scope, normalized

    legacy_session = payload.get("session_feature_stats")
    if isinstance(legacy_session, Mapping):
        return "session", normalize_feature_stats_entries(legacy_session)
    if payload.get("mean") is not None and payload.get("std") is not None:
        return "global", {
            "global": _as_stats_pair(
                {"mean": payload["mean"], "std": payload["std"]},
                key="global",
            )
        }
    raise ValueError(
        "Stats payload must contain feature_stats, session_feature_stats, or mean/std."
    )


def write_feature_stats_artifact(
    *,
    output_path: str | Path,
    scope: str,
    entries: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    output_path = Path(output_path)
    payload = build_feature_stats_payload(
        scope=scope,
        entries=entries,
        metadata=metadata,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)
    metadata_path = output_path.with_suffix(".json")
    metadata_path.write_text(json.dumps(payload["metadata"], indent=2) + "\n")
    return payload


__all__ = [
    "FEATURE_STATS_SCHEMA",
    "SUPPORTED_NORMALIZATION_SCOPES",
    "build_feature_stats_payload",
    "extract_feature_stats_entries",
    "normalize_feature_stats_entries",
    "write_feature_stats_artifact",
]

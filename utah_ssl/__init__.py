"""Reusable Utah-array data, modeling, and evaluation utilities."""

from .imports import repo_root
from .patching import PatchPolicy, patch_batch, patch_starts, patched_length, patched_lengths

__all__ = [
    "PatchPolicy",
    "patch_batch",
    "patch_starts",
    "patched_length",
    "patched_lengths",
    "repo_root",
]

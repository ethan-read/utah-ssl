"""Import-path helpers shared by experiment packages.

The notebooks usually add the repo and S5 benchmark directories manually. This
module gives scripts and tests one small place to do the same thing.
"""

from __future__ import annotations

import sys
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def ssl_experiments_dir() -> Path:
    return repo_root() / "analysis" / "active" / "ssl_experiments"


def s5_source_dir() -> Path:
    return repo_root() / "analysis" / "active" / "transfer_benchmark" / "ssl_autoresearch"


def _prepend_sys_path(path: Path) -> None:
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


def ensure_experiment_import_paths() -> None:
    for path in (repo_root(), ssl_experiments_dir(), s5_source_dir()):
        _prepend_sys_path(path)


def ensure_s5_import_path() -> None:
    _prepend_sys_path(s5_source_dir())


__all__ = [
    "ensure_experiment_import_paths",
    "ensure_s5_import_path",
    "repo_root",
    "s5_source_dir",
    "ssl_experiments_dir",
]

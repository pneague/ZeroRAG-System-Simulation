"""Environment-driven default paths for local and cluster runs."""

from __future__ import annotations

import os
from pathlib import Path


def _env_value(*names: str) -> str | None:
    for name in names:
        value = os.environ.get(name, "").strip()
        if value:
            return value
    return None


def _env_path(*names: str) -> Path | None:
    value = _env_value(*names)
    if value is None:
        return None
    return Path(os.path.expandvars(value)).expanduser()


def default_scratch_root() -> Path:
    override = _env_path("ZERORAG_SCRATCH_ROOT", "LOW_COORD_DHT_SCRATCH_ROOT")
    if override is not None:
        return override
    username = _env_value("USER", "LOGNAME") or "user"
    return Path("/var/scratch") / username / "ZeroRAG"


def default_data_root() -> Path:
    return default_scratch_root() / "data"


def default_results_root() -> Path:
    return default_scratch_root() / "results"


def default_experiments_registry() -> Path:
    return default_scratch_root() / "experiments_registry.json"


def default_project_root(repo_root: Path | None = None) -> Path:
    override = _env_path("ZERORAG_PROJECT_ROOT", "LOW_COORD_DHT_ROOT")
    if override is not None:
        return override
    if repo_root is not None:
        return repo_root
    return Path.cwd()


def default_remote_host() -> str:
    return _env_value("ZERORAG_REMOTE_HOST", "LOW_COORD_DHT_REMOTE_HOST") or "user@cluster.example.edu"


def default_remote_root() -> str:
    return str(
        _env_path("ZERORAG_REMOTE_ROOT", "LOW_COORD_DHT_REMOTE_ROOT")
        or Path("/path/to/ZeroRAG")
    )


def default_remote_scratch() -> str:
    return str(_env_path("ZERORAG_REMOTE_SCRATCH", "LOW_COORD_DHT_REMOTE_SCRATCH") or default_scratch_root())

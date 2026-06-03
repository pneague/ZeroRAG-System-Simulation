"""Resolve dataset directories with a three-tier precedence rule.

Precedence:

1. ``LOW_COORD_DHT_DATA_ROOT`` env var (wins for every dataset, points at the
   parent directory that holds ``aol4foltr/``, ``amazon/``, ``mind/``,
   ``synthetic/``, and ``pubmedqa/``).
2. Local marker-file presence under ``<repo>/data/<dataset>[/<category>]/``.
   Each dataset declares its own marker files; if any of them exist on disk
   we treat the local copy as authoritative.
3. Hardcoded scratch fallback at
   ``/var/scratch/pmneague/Low_Coord_Semantic_DHT/data/<dataset>[/<category>]/``.

The same helper is used by all dataset callers so large local artifacts can be
migrated to scratch on remote without changing code call sites.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal


DATA_ROOT_ENV_VAR = "LOW_COORD_DHT_DATA_ROOT"
DEFAULT_SCRATCH_DATA_ROOT = Path("/var/scratch/pmneague/Low_Coord_Semantic_DHT/data")
LOCAL_DATA_ROOT = Path(__file__).resolve().parent.parent / "data"


# Marker files used to decide that a local copy of the dataset exists.
# The first marker that is found wins; if none exist locally we fall back
# to scratch. Markers are evaluated in order and the first hit short-circuits.
DATASET_MARKERS: dict[str, tuple[str, ...]] = {
    "aol4foltr": ("metadata.csv",),
    "amazon": ("events.csv", "benchmark/query_text.txt.gz"),
    "mind": ("events.csv", "raw/train/news.tsv"),
    "synthetic": ("events.csv",),
    "pubmedqa": ("events.csv",),
}


AMAZON_CATEGORIES: tuple[str, ...] = (
    "cell_phones",
    "electronics",
    "kindle",
    "cds",
    "all",
)


ResolutionSource = Literal["env_override", "local", "scratch"]


def _dataset_relative_path(dataset: str, category: str | None) -> Path:
    if dataset == "amazon":
        if category is None:
            raise ValueError("amazon dataset requires a category")
        if category not in AMAZON_CATEGORIES:
            raise ValueError(
                f"unknown amazon category {category!r}; expected one of {AMAZON_CATEGORIES}"
            )
        return Path("amazon") / category
    if dataset == "aol4foltr":
        if category is not None:
            raise ValueError("aol4foltr does not accept a category argument")
        return Path("aol4foltr")
    if dataset == "mind":
        if category is not None:
            raise ValueError("mind does not accept a category argument")
        return Path("mind")
    if dataset == "synthetic":
        if category is not None:
            raise ValueError("synthetic does not accept a category argument")
        return Path("synthetic")
    if dataset == "pubmedqa":
        if category is not None:
            raise ValueError("pubmedqa does not accept a category argument")
        return Path("pubmedqa")
    raise ValueError(f"unknown dataset {dataset!r}")


def resolve_dataset_dir(
    dataset: str,
    category: str | None = None,
    *,
    return_source: bool = False,
) -> Path | tuple[Path, ResolutionSource]:
    """Return the directory that should hold the dataset's data files.

    The path is returned even when it does not exist yet, so preprocessing
    scripts can use this helper to decide where to write fresh artifacts.
    """

    rel = _dataset_relative_path(dataset, category)

    env_override = os.environ.get(DATA_ROOT_ENV_VAR, "").strip()
    if env_override:
        resolved = Path(env_override) / rel
        source: ResolutionSource = "env_override"
    else:
        local = LOCAL_DATA_ROOT / rel
        markers = DATASET_MARKERS.get(dataset, ())
        if markers and any((local / marker).exists() for marker in markers):
            resolved = local
            source = "local"
        else:
            resolved = DEFAULT_SCRATCH_DATA_ROOT / rel
            source = "scratch"

    if return_source:
        return resolved, source
    return resolved


def describe_resolution(dataset: str, category: str | None = None) -> str:
    """Return a one-line description suitable for logging."""

    resolved, source = resolve_dataset_dir(dataset, category, return_source=True)
    label = dataset if category is None else f"{dataset}/{category}"
    return f"dataset={label} resolved_dir={resolved} source={source}"

"""Helpers for the Amazon Product Search benchmark.

The Amazon scenario is built from:

* QingyaoAi/Amazon-Product-Search-Datasets benchmark files
  (``query_text.txt.gz``, ``train.qrels.gz``, ``test.qrels.gz``,
  ``train_review_id.txt.gz``); and
* the matching per-category UCSD Amazon Review Data 2018 metadata file
  (``meta_<Category>.json.gz``).

The benchmark mixes real user-item purchases/reviews with synthetic
generated queries; the joined product attributes come from UCSD metadata
keyed by ``asin``. We never describe Amazon events as "user searched for
product" - that phrasing misrepresents how the queries are produced.

This module owns the per-category constants, qrels parsing, query-text
loading, UCSD metadata normalization, and the artifact filenames written
by :mod:`precompute_amazon`. The simulation itself reuses the AOL
``QueryEvent`` dataclass and CSV loaders, so Amazon events.csv must
respect the AOL column order.
"""

from __future__ import annotations

import ast
import csv
import gzip
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Mapping

from src.scenarios import AMAZON_SUPPORTED_SCENARIOS as AMAZON_SUPPORTED_SCENARIOS  # noqa: F401


def _bump_csv_field_size_limit() -> None:
    """Raise the csv module's per-field size cap to the largest value the
    underlying C long accepts.

    Some UCSD product descriptions exceed Python's default 131072-char limit,
    which makes ``csv.DictReader`` raise ``_csv.Error: field larger than field
    limit (131072)`` while reading ``doc_metadata.csv.gz``. We bump it once at
    module import time with a backoff loop because ``sys.maxsize`` can overflow
    a C long on 32-bit builds.
    """

    candidate = sys.maxsize
    while True:
        try:
            csv.field_size_limit(candidate)
            return
        except OverflowError:
            candidate = candidate // 10
            if candidate <= 0:
                return


_bump_csv_field_size_limit()


@dataclass(frozen=True, slots=True)
class AmazonCategorySpec:
    key: str
    benchmark_dir: str
    ucsd_meta_filename: str
    display_name: str


AMAZON_CATEGORY_SPECS: dict[str, AmazonCategorySpec] = {
    "cell_phones": AmazonCategorySpec(
        key="cell_phones",
        benchmark_dir="reviews_Cell_Phones_and_Accessories_5.json.gz",
        ucsd_meta_filename="meta_Cell_Phones_and_Accessories.json.gz",
        display_name="Cell Phones & Accessories",
    ),
    "electronics": AmazonCategorySpec(
        key="electronics",
        benchmark_dir="reviews_Electronics_5.json.gz",
        ucsd_meta_filename="meta_Electronics.json.gz",
        display_name="Electronics",
    ),
    "kindle": AmazonCategorySpec(
        key="kindle",
        benchmark_dir="reviews_Kindle_Store_5.json.gz",
        ucsd_meta_filename="meta_Kindle_Store.json.gz",
        display_name="Kindle Store",
    ),
    "cds": AmazonCategorySpec(
        key="cds",
        benchmark_dir="reviews_CDs_and_Vinyl_5.json.gz",
        ucsd_meta_filename="meta_CDs_and_Vinyl.json.gz",
        display_name="CDs & Vinyl",
    ),
    # Synthetic key representing the union of all four categories. The combined
    # dataset is built by precompute_amazon_all.py, not by download_amazon.py /
    # precompute_amazon.py (those reject --category=all explicitly). The empty
    # benchmark_dir / ucsd_meta_filename fields are intentional: they have no
    # meaning for the combined dataset and any consumer that looks them up
    # should be using the per-category specs instead.
    "all": AmazonCategorySpec(
        key="all",
        benchmark_dir="",
        ucsd_meta_filename="",
        display_name="All Categories Combined",
    ),
}


# Real per-category Amazon keys, excluding the synthetic "all" combined key.
# Used by combiner / aggregation tools that need to iterate over the source
# categories without including the union as one of them.
AMAZON_REAL_CATEGORIES: tuple[str, ...] = (
    "cell_phones",
    "electronics",
    "kindle",
    "cds",
)


AMAZON_BENCHMARK_FILENAMES: tuple[str, ...] = (
    "query_text.txt.gz",
    "train.qrels.gz",
    "test.qrels.gz",
    "train_review_id.txt.gz",
)


# AMAZON_SUPPORTED_SCENARIOS is re-exported at the top of this module from
# :mod:`src.scenarios`, which is the single source of truth shared with the
# LSH-DHT entrypoint. Keep this comment for grep continuity.


# Filenames written under <data>/amazon/<category>/ by precompute_amazon.py.
AMAZON_EVENTS_FILENAME = "events.csv"
AMAZON_RANDOMIZED_EVENTS_FILENAME = "metadata_seeded_randomized.csv.gz"
AMAZON_SEEDED_SEARCHABLE_FILENAME = "seeded_searchable_queries.csv.gz"
AMAZON_PEER_FUTURE_DOCS_FILENAME = "peer_future_doc_ids.json.gz"
AMAZON_PEER_FUTURE_DOCS_SUMMARY_FILENAME = "peer_future_doc_ids_summary.json"
AMAZON_PEER_ID_MAP_FILENAME = "peer_id_map.json.gz"
AMAZON_DOC_METADATA_FILENAME = "doc_metadata.csv.gz"
AMAZON_PRECOMPUTE_SUMMARY_FILENAME = "amazon_precompute_summary.json"


def category_spec(category: str) -> AmazonCategorySpec:
    if category not in AMAZON_CATEGORY_SPECS:
        raise ValueError(
            f"unknown amazon category {category!r}; expected one of "
            f"{sorted(AMAZON_CATEGORY_SPECS)}"
        )
    return AMAZON_CATEGORY_SPECS[category]


def parse_amazon_qrels_key(key: str) -> tuple[str, int]:
    """Split a qrels user-query key on the **last** ``_`` separator.

    The benchmark composes the first column as ``<reviewerID>_<query_line_number>``
    and reviewerIDs themselves can contain underscores, so we split from the right.
    """

    last_sep = key.rfind("_")
    if last_sep <= 0 or last_sep == len(key) - 1:
        raise ValueError(f"invalid amazon qrels key {key!r}")
    reviewer_id = key[:last_sep]
    try:
        query_line_number = int(key[last_sep + 1 :])
    except ValueError as exc:
        raise ValueError(f"invalid amazon qrels key {key!r}") from exc
    return reviewer_id, query_line_number


def iter_qrels(path: str | Path) -> Iterator[dict[str, object]]:
    """Yield parsed rows from a ``train.qrels.gz`` / ``test.qrels.gz`` file.

    Each yielded record contains ``reviewer_id``, ``query_line_number``,
    ``asin`` and ``relevance_label`` (int).
    """

    qrels_path = Path(path)
    with gzip.open(qrels_path, "rt", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 4:
                raise ValueError(
                    f"invalid qrels line in {qrels_path}: {raw_line.rstrip()!r}"
                )
            user_query_key, _ignored, asin, relevance = parts
            reviewer_id, query_line_number = parse_amazon_qrels_key(user_query_key)
            yield {
                "reviewer_id": reviewer_id,
                "query_line_number": query_line_number,
                "asin": asin,
                "relevance_label": int(relevance),
            }


def load_query_texts(path: str | Path) -> list[str]:
    """Load ``query_text.txt.gz`` as a list indexed by 0-based line number."""

    with gzip.open(Path(path), "rt", encoding="utf-8") as handle:
        return [line.rstrip("\n") for line in handle]


def iter_ucsd_metadata_records(path: str | Path) -> Iterator[Mapping[str, object]]:
    """Stream UCSD per-category metadata (``meta_<Category>.json.gz``).

    The files contain one JSON-like object per line. Some older releases use
    Python-literal syntax instead of strict JSON; we fall back to
    :func:`ast.literal_eval` for those lines so we can parse both formats.
    Malformed lines are skipped silently after a single warning per call site.
    """

    with gzip.open(Path(path), "rt", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                try:
                    yield ast.literal_eval(line)
                except (SyntaxError, ValueError):
                    continue


def normalize_ucsd_record(record: Mapping[str, object]) -> dict[str, str]:
    """Return a flat ``{asin,title,description,categories,brand,price}`` row.

    Empty/missing values are coerced to empty strings. Lists are joined into
    human-readable text so downstream embedders can use a single text field
    per asin.
    """

    asin = str(record.get("asin", "")).strip()
    title = str(record.get("title", "") or "").strip()

    description_raw = record.get("description", "") or ""
    if isinstance(description_raw, list):
        description = "\n".join(
            str(item).strip() for item in description_raw if str(item).strip()
        )
    else:
        description = str(description_raw).strip()

    # UCSD 2018 uses both ``category`` (newer dump) and ``categories`` (older).
    categories_raw = record.get("category", None)
    if categories_raw in (None, "", []):
        categories_raw = record.get("categories", "")
    if isinstance(categories_raw, list):
        if categories_raw and isinstance(categories_raw[0], list):
            flat: list[str] = []
            for sublist in categories_raw:
                flat.extend(
                    str(item).strip() for item in sublist if str(item).strip()
                )
            categories = " > ".join(flat)
        else:
            categories = " > ".join(
                str(item).strip() for item in categories_raw if str(item).strip()
            )
    else:
        categories = str(categories_raw or "").strip()

    brand = str(record.get("brand", "") or "").strip()

    price_raw = record.get("price", "")
    if price_raw is None:
        price = ""
    else:
        price = str(price_raw).strip()

    return {
        "asin": asin,
        "title": title,
        "description": description,
        "categories": categories,
        "brand": brand,
        "price": price,
    }


def compose_doc_text(record: Mapping[str, str]) -> str:
    """Build the embed-ready text for one product from its normalized fields."""

    parts: list[str] = []
    if record.get("title"):
        parts.append(record["title"])
    if record.get("description"):
        parts.append(record["description"])
    if record.get("categories"):
        parts.append(record["categories"])
    if record.get("brand"):
        parts.append(f"Brand: {record['brand']}")
    if record.get("price"):
        parts.append(f"Price: {record['price']}")
    return "\n\n".join(parts)

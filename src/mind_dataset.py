"""Helpers for the MIND news recommendation dataset.

MIND (ACL 2020) ships impression logs in ``behaviors.tsv`` and article text in
``news.tsv``. This module keeps the download locations, raw TSV parsing, and
artifact filenames in one place. Downstream simulations reuse the AOL-shaped
``QueryEvent`` CSV contract, so MIND precompute writes the same column names as
the existing Amazon pipeline.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterator


MIND_BLOB_BASE_URL = "https://mind201910small.blob.core.windows.net/release"
MIND_DOWNLOAD_FILENAMES: tuple[str, ...] = (
    "MINDlarge_train.zip",
    "MINDlarge_dev.zip",
)
MIND_SPLITS: tuple[str, ...] = ("train", "dev")
MIND_TIME_FORMAT = "%m/%d/%Y %I:%M:%S %p"
ARTICLE_TEXT_DELIMITER = "\n\n"


MIND_EVENTS_FILENAME = "events.csv"
MIND_RANDOMIZED_EVENTS_FILENAME = "metadata_seeded_randomized.csv.gz"
MIND_SEEDED_SEARCHABLE_FILENAME = "seeded_searchable_queries.csv.gz"
MIND_PEER_FUTURE_DOCS_FILENAME = "peer_future_doc_ids.json.gz"
MIND_PEER_FUTURE_DOCS_SUMMARY_FILENAME = "peer_future_doc_ids_summary.json"
MIND_PEER_ID_MAP_FILENAME = "peer_id_map.json.gz"
MIND_QUERY_ID_MAP_FILENAME = "query_id_map.json.gz"
MIND_DOC_METADATA_FILENAME = "doc_metadata.csv.gz"
MIND_PRECOMPUTE_SUMMARY_FILENAME = "mind_precompute_summary.json"


MIND_DOC_METADATA_FIELDNAMES: tuple[str, ...] = (
    "news_id",
    "category",
    "subcategory",
    "title",
    "abstract",
    "url",
    "text",
)


@dataclass(frozen=True, slots=True)
class MindArticle:
    news_id: str
    category: str
    subcategory: str
    title: str
    abstract: str
    url: str

    @property
    def text(self) -> str:
        return compose_article_text(self.title, self.abstract)


@dataclass(frozen=True, slots=True)
class MindPositiveClick:
    split: str
    impression_id: str
    user_id: str
    timestamp: datetime
    clicked_news_id: str
    impression_position: int


def compose_article_text(title: str, abstract: str) -> str:
    """Return the exact text embedded for MIND queries and documents."""

    parts = [title.strip(), abstract.strip()]
    return ARTICLE_TEXT_DELIMITER.join(part for part in parts if part)


def parse_mind_timestamp(raw_value: str) -> datetime:
    return datetime.strptime(raw_value.strip(), MIND_TIME_FORMAT)


def iter_news(path: str | Path) -> Iterator[MindArticle]:
    """Yield article records from a MIND ``news.tsv`` file."""

    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle, delimiter="\t")
        for row in reader:
            if not row:
                continue
            if len(row) < 6:
                raise ValueError(f"invalid MIND news row in {path}: expected at least 6 columns, got {len(row)}")
            yield MindArticle(
                news_id=row[0].strip(),
                category=row[1].strip(),
                subcategory=row[2].strip(),
                title=row[3].strip(),
                abstract=row[4].strip(),
                url=row[5].strip(),
            )


def iter_positive_clicks(path: str | Path, *, split: str) -> Iterator[MindPositiveClick]:
    """Yield clicked impression items from a MIND ``behaviors.tsv`` file."""

    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle, delimiter="\t")
        for row in reader:
            if not row:
                continue
            if len(row) != 5:
                raise ValueError(
                    f"invalid MIND behavior row in {path}: expected 5 columns, got {len(row)}"
                )
            impression_id, user_id, raw_time, _history, impressions = row
            timestamp = parse_mind_timestamp(raw_time)
            for position, raw_item in enumerate(impressions.split()):
                news_id, separator, label = raw_item.rpartition("-")
                if not separator or not news_id:
                    raise ValueError(f"invalid MIND impression item {raw_item!r} in {path}")
                if label == "1":
                    yield MindPositiveClick(
                        split=split,
                        impression_id=impression_id.strip(),
                        user_id=user_id.strip(),
                        timestamp=timestamp,
                        clicked_news_id=news_id.strip(),
                        impression_position=position,
                    )

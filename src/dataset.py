from __future__ import annotations

import ast
import csv
import gzip
import json
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Iterable, Iterator, Mapping, TextIO


METADATA_FILENAME = "metadata.csv"
SEEDED_RANDOMIZED_METADATA_FILENAME = "metadata_seeded_randomized.csv.gz"
CHRONOLOGICAL_SEARCHABLE_QUERIES_FILENAME = "chronological_searchable_queries.csv.gz"
SEEDED_SEARCHABLE_QUERIES_FILENAME = "seeded_searchable_queries.csv.gz"
TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"
SEARCHABLE_QUERY_FIELDNAMES = ("query_id", "searchable")
RELEVANT_DOC_IDS_BY_QID_FILENAME = "relevant_doc_ids_by_qid.json.gz"
AMAZON_QUERY_AVAILABILITY_FILENAME = "amazon_query_availability.json.gz"
# Synthetic-timestamp epoch for non-chronological datasets (e.g. Amazon).
# Concrete value is irrelevant; it just has to round-trip through TIMESTAMP_FORMAT.
EPOCH_FOR_SYNTHETIC_TIMESTAMPS = datetime(2000, 1, 1, 0, 0, 0)
AMAZON_EVENTS_FIELDNAMES = (
    "timestamp",
    "qid",
    "user_id",
    "query",
    "target_doc_id",
    "candidate_doc_ids",
)


@dataclass(slots=True)
class QueryEvent:
    event_index: int
    qid: int
    user_id: int
    timestamp: datetime
    day: date
    query_text: str
    clicked_doc_id: str
    candidate_doc_ids: tuple[str, ...]
    searchable: bool


def parse_candidate_doc_ids(raw_value: str) -> tuple[str, ...]:
    if not raw_value:
        return ()
    parsed = ast.literal_eval(raw_value)
    if not isinstance(parsed, list):
        raise ValueError(f"Expected candidate_doc_ids to parse to list, got {type(parsed)!r}")
    return tuple(str(item) for item in parsed)


def parse_bool(raw_value: object) -> bool:
    return str(raw_value).strip().lower() in {"1", "true", "yes"}


def _open_csv_text(path: Path) -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(path, "rt", newline="", encoding="utf-8")
    return path.open("r", newline="", encoding="utf-8")


def open_csv_text(path: str | Path) -> TextIO:
    """Public alias for :func:`_open_csv_text` used by sibling modules."""

    return _open_csv_text(Path(path))


def synthetic_timestamp_for_index(event_index: int) -> str:
    """Return a TIMESTAMP_FORMAT string ``EPOCH + event_index`` seconds.

    Used by datasets that do not ship real timestamps (e.g. Amazon) so they can
    still flow through the AOL-format CSV loaders without breaking ``datetime``
    parsing. The value carries no real-world meaning and must never be reported
    as such.
    """

    from datetime import timedelta

    moment = EPOCH_FOR_SYNTHETIC_TIMESTAMPS + timedelta(seconds=event_index)
    return moment.strftime(TIMESTAMP_FORMAT)


def iter_metadata_rows(path: str | Path) -> Iterator[dict[str, str]]:
    metadata_path = Path(path)
    with _open_csv_text(metadata_path) as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            yield row


def load_searchable_query_annotations(path: str | Path) -> dict[int, bool]:
    searchable_path = Path(path)
    annotations: dict[int, bool] = {}
    with _open_csv_text(searchable_path) as handle:
        reader = csv.DictReader(handle)
        expected_fields = set(SEARCHABLE_QUERY_FIELDNAMES)
        actual_fields = set(reader.fieldnames or [])
        if actual_fields != expected_fields:
            raise ValueError(
                f"{searchable_path} must contain exactly {sorted(expected_fields)}, "
                f"found {sorted(actual_fields)}"
            )
        for row in reader:
            qid = int(row["query_id"])
            if qid in annotations:
                raise ValueError(f"Duplicate query_id {qid} in {searchable_path}")
            annotations[qid] = parse_bool(row["searchable"])
    return annotations


def load_relevant_doc_ids_by_qid(path: str | Path) -> dict[int, frozenset[str]]:
    """Load qid -> relevant doc ids from a JSON(.gz) side artifact."""

    relevant_path = Path(path)
    opener = gzip.open if relevant_path.suffix == ".gz" else open
    with opener(relevant_path, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    return {
        int(qid): frozenset(str(doc_id) for doc_id in doc_ids)
        for qid, doc_ids in payload.items()
    }


def row_to_event(
    row: Mapping[str, str],
    event_index: int,
    *,
    searchable_lookup: Mapping[int, bool] | None = None,
) -> QueryEvent:
    qid = int(row["qid"])
    timestamp = datetime.strptime(row["timestamp"], TIMESTAMP_FORMAT)
    if searchable_lookup is not None:
        if qid not in searchable_lookup:
            raise ValueError(f"Missing searchable annotation for query_id={qid}")
        searchable = searchable_lookup[qid]
    else:
        searchable = parse_bool(row.get("searchable", ""))
    return QueryEvent(
        event_index=event_index,
        qid=qid,
        user_id=int(row["user_id"]),
        timestamp=timestamp,
        day=timestamp.date(),
        query_text=row["query"],
        clicked_doc_id=row["target_doc_id"],
        candidate_doc_ids=parse_candidate_doc_ids(row.get("candidate_doc_ids", "")),
        searchable=searchable,
    )


def iter_query_events(
    path: str | Path,
    *,
    searchable_only: bool = False,
    require_searchable_column: bool = False,
    searchable_queries_path: str | Path | None = None,
) -> Iterator[QueryEvent]:
    searchable_lookup = (
        load_searchable_query_annotations(searchable_queries_path)
        if searchable_queries_path is not None
        else None
    )
    for event_index, row in enumerate(iter_metadata_rows(path)):
        if require_searchable_column and searchable_lookup is None and "searchable" not in row:
            raise ValueError(
                f"{path} is missing searchable annotations. "
                "Provide --searchable-queries or generate inline searchable data."
            )
        event = row_to_event(row, event_index=event_index, searchable_lookup=searchable_lookup)
        if searchable_only and not event.searchable:
            continue
        yield event


PROGRESS_LOG_INTERVAL = 5_000


def load_query_events(
    path: str | Path,
    *,
    searchable_only: bool = False,
    max_events: int | None = None,
    require_searchable_column: bool = False,
    searchable_queries_path: str | Path | None = None,
    progress_interval: int = PROGRESS_LOG_INTERVAL,
) -> list[QueryEvent]:
    import sys

    events: list[QueryEvent] = []
    seen_user_ids: set[int] = set()
    for event in iter_query_events(
        path,
        searchable_only=searchable_only,
        require_searchable_column=require_searchable_column,
        searchable_queries_path=searchable_queries_path,
    ):
        events.append(event)
        if event.user_id not in seen_user_ids:
            seen_user_ids.add(event.user_id)
            if progress_interval > 0 and len(seen_user_ids) % progress_interval == 0:
                print(
                    f"[load_query_events] loaded {len(seen_user_ids):,} unique user_ids "
                    f"({len(events):,} events so far) from {path}",
                    flush=True,
                )
        if max_events is not None and len(events) >= max_events:
            break
    if progress_interval > 0:
        print(
            f"[load_query_events] done: {len(seen_user_ids):,} unique user_ids "
            f"across {len(events):,} events from {path}",
            flush=True,
        )
        sys.stdout.flush()
    return events


def collect_unique_queries_and_doc_ids(
    path: str | Path,
    *,
    searchable_only: bool = False,
    include_candidates: bool = True,
    require_searchable_column: bool = False,
    searchable_queries_path: str | Path | None = None,
) -> tuple[list[str], list[str]]:
    query_texts: set[str] = set()
    doc_ids: set[str] = set()
    for event in iter_query_events(
        path,
        searchable_only=searchable_only,
        require_searchable_column=require_searchable_column,
        searchable_queries_path=searchable_queries_path,
    ):
        query_texts.add(event.query_text)
        doc_ids.add(event.clicked_doc_id)
        if include_candidates:
            doc_ids.update(event.candidate_doc_ids)
    return sorted(query_texts), sorted(doc_ids)


def day_range(path: str | Path) -> tuple[date, date]:
    start_day: date | None = None
    end_day: date | None = None
    for row in iter_metadata_rows(path):
        timestamp = datetime.strptime(row["timestamp"], TIMESTAMP_FORMAT)
        current_day = timestamp.date()
        if start_day is None:
            start_day = current_day
        end_day = current_day
    if start_day is None or end_day is None:
        raise ValueError(f"No rows found in {path}")
    return start_day, end_day


def ensure_parent_dir(path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def count_rows(path: str | Path) -> int:
    return sum(1 for _ in iter_metadata_rows(path))


def filter_events(events: Iterable[QueryEvent], *, searchable_only: bool) -> Iterator[QueryEvent]:
    for event in events:
        if searchable_only and not event.searchable:
            continue
        yield event


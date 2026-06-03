#!/usr/bin/env python3
"""Build simulator-facing artifacts for MIND-large.

The pipeline combines MIND train+dev because this benchmark uses the event
stream as a distributed retrieval trace rather than as a supervised train/dev
recommendation split. Each positive clicked article becomes an AOL-compatible
event whose query text is exactly that article's title+abstract text.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from precompute_searchable import build_chronological_searchable_queries
from src.data_root import resolve_dataset_dir
from src.dataset import (
    AMAZON_EVENTS_FIELDNAMES,
    CHRONOLOGICAL_SEARCHABLE_QUERIES_FILENAME,
    SEARCHABLE_QUERY_FIELDNAMES,
    synthetic_timestamp_for_index,
)
from src.mind_dataset import (
    MIND_DOC_METADATA_FIELDNAMES,
    MIND_DOC_METADATA_FILENAME,
    MIND_EVENTS_FILENAME,
    MIND_PEER_FUTURE_DOCS_FILENAME,
    MIND_PEER_FUTURE_DOCS_SUMMARY_FILENAME,
    MIND_PEER_ID_MAP_FILENAME,
    MIND_PRECOMPUTE_SUMMARY_FILENAME,
    MIND_QUERY_ID_MAP_FILENAME,
    MIND_RANDOMIZED_EVENTS_FILENAME,
    MIND_SEEDED_SEARCHABLE_FILENAME,
    MIND_SPLITS,
    MindArticle,
    MindPositiveClick,
    iter_news,
    iter_positive_clicks,
)


@dataclass(frozen=True, slots=True)
class MindEventRecord:
    qid: int
    peer_id: int
    timestamp: str
    query_text: str
    target_doc_id: str
    source_split: str
    impression_id: str
    impression_position: int


def _open_text_writer(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "wt", newline="", encoding="utf-8")
    return path.open("w", newline="", encoding="utf-8")


def _write_json_gz(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle)


def _impression_sort_key(click: MindPositiveClick) -> tuple[object, ...]:
    try:
        impression_key: object = int(click.impression_id)
    except ValueError:
        impression_key = click.impression_id
    return (
        click.timestamp,
        MIND_SPLITS.index(click.split),
        impression_key,
        click.impression_position,
        click.user_id,
        click.clicked_news_id,
    )


def _load_articles(data_dir: Path) -> dict[str, MindArticle]:
    articles: dict[str, MindArticle] = {}
    for split in MIND_SPLITS:
        path = data_dir / "raw" / split / "news.tsv"
        if not path.exists():
            raise FileNotFoundError(f"missing required input {path}; run download_mind.py first")
        for article in iter_news(path):
            if not article.news_id:
                continue
            existing = articles.get(article.news_id)
            if existing is None or (not existing.text and article.text):
                articles[article.news_id] = article
    return articles


def _load_positive_clicks(data_dir: Path) -> list[MindPositiveClick]:
    clicks: list[MindPositiveClick] = []
    for split in MIND_SPLITS:
        path = data_dir / "raw" / split / "behaviors.tsv"
        if not path.exists():
            raise FileNotFoundError(f"missing required input {path}; run download_mind.py first")
        clicks.extend(iter_positive_clicks(path, split=split))
    clicks.sort(key=_impression_sort_key)
    return clicks


def _build_peer_id_map(clicks: list[MindPositiveClick]) -> dict[str, int]:
    unique_user_ids = sorted({click.user_id for click in clicks})
    return {user_id: idx for idx, user_id in enumerate(unique_user_ids)}


def _write_peer_id_map(peer_id_map: dict[str, int], output_path: Path) -> None:
    _write_json_gz(output_path, peer_id_map)


def _write_query_id_map(records: list[MindEventRecord], output_path: Path) -> None:
    payload = {
        str(record.qid): {
            "news_id": record.target_doc_id,
            "source_split": record.source_split,
            "impression_id": record.impression_id,
            "impression_position": record.impression_position,
        }
        for record in records
    }
    _write_json_gz(output_path, payload)


def _write_doc_metadata(articles: dict[str, MindArticle], output_path: Path) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with _open_text_writer(output_path) as handle:
        writer = csv.DictWriter(handle, fieldnames=MIND_DOC_METADATA_FIELDNAMES)
        writer.writeheader()
        for news_id in sorted(articles):
            article = articles[news_id]
            writer.writerow(
                {
                    "news_id": article.news_id,
                    "category": article.category,
                    "subcategory": article.subcategory,
                    "title": article.title,
                    "abstract": article.abstract,
                    "url": article.url,
                    "text": article.text,
                }
            )
    return len(articles)


def _build_event_records(
    clicks: list[MindPositiveClick],
    *,
    articles: dict[str, MindArticle],
    peer_id_map: dict[str, int],
) -> tuple[list[MindEventRecord], int]:
    records: list[MindEventRecord] = []
    missing_clicked_articles = 0
    for click in clicks:
        article = articles.get(click.clicked_news_id)
        if article is None:
            missing_clicked_articles += 1
            continue
        qid = len(records)
        records.append(
            MindEventRecord(
                qid=qid,
                peer_id=peer_id_map[click.user_id],
                timestamp=click.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                query_text=article.text,
                target_doc_id=click.clicked_news_id,
                source_split=click.split,
                impression_id=click.impression_id,
                impression_position=click.impression_position,
            )
        )
    return records, missing_clicked_articles


def _write_events_csv(output_path: Path, records: list[MindEventRecord]) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=AMAZON_EVENTS_FIELDNAMES)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "timestamp": record.timestamp,
                    "qid": record.qid,
                    "user_id": record.peer_id,
                    "query": record.query_text,
                    "target_doc_id": record.target_doc_id,
                    "candidate_doc_ids": "[]",
                }
            )
    return len(records)


def _write_randomized_events(
    source_path: Path,
    output_path: Path,
    *,
    random_seed: int,
) -> int:
    with source_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or AMAZON_EVENTS_FIELDNAMES)
        original_rows = list(reader)

    rng = np.random.default_rng(random_seed)
    rows_by_user: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in original_rows:
        rows_by_user[row["user_id"]].append(row)

    shuffled: list[dict[str, str]] = []
    for user_id in sorted(rows_by_user, key=lambda value: int(value)):
        user_rows = list(rows_by_user[user_id])
        if len(user_rows) > 1:
            order = rng.permutation(len(user_rows))
            user_rows = [user_rows[i] for i in order]
        shuffled.extend(user_rows)

    global_order = rng.permutation(len(shuffled)).tolist()
    ordered_rows = [shuffled[i] for i in global_order]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with _open_text_writer(output_path) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for new_index, row in enumerate(ordered_rows):
            row = dict(row)
            row["timestamp"] = synthetic_timestamp_for_index(new_index)
            writer.writerow(row)
    return len(ordered_rows)


def _write_seeded_searchable_sidecar(
    records: list[MindEventRecord],
    *,
    output_path: Path,
) -> tuple[int, int]:
    owners_by_doc: dict[str, set[int]] = defaultdict(set)
    for record in records:
        owners_by_doc[record.target_doc_id].add(record.peer_id)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    searchable_count = 0
    with _open_text_writer(output_path) as handle:
        writer = csv.DictWriter(handle, fieldnames=SEARCHABLE_QUERY_FIELDNAMES)
        writer.writeheader()
        for record in records:
            other_owners = owners_by_doc[record.target_doc_id] - {record.peer_id}
            searchable = bool(other_owners)
            writer.writerow(
                {
                    "query_id": record.qid,
                    "searchable": "true" if searchable else "false",
                }
            )
            searchable_count += int(searchable)
    return len(records), searchable_count


def _write_peer_future_docs(
    records: list[MindEventRecord],
    *,
    output_path: Path,
    summary_path: Path,
) -> dict[str, int | float | str]:
    docs_by_peer: dict[int, set[str]] = defaultdict(set)
    for record in records:
        docs_by_peer[record.peer_id].add(record.target_doc_id)

    payload = {
        str(peer_id): sorted(doc_ids)
        for peer_id, doc_ids in sorted(docs_by_peer.items())
    }
    _write_json_gz(output_path, payload)

    total_pairs = sum(len(doc_ids) for doc_ids in payload.values())
    summary = {
        "output_path": str(output_path),
        "peer_count": len(payload),
        "total_unique_peer_doc_pairs": total_pairs,
        "avg_docs_per_peer": total_pairs / len(payload) if payload else 0.0,
        "max_docs_per_peer": max((len(v) for v in payload.values()), default=0),
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def precompute_mind_artifacts(
    *,
    data_dir: Path | None = None,
    random_seed: int = 42,
) -> dict[str, object]:
    if data_dir is None:
        data_dir = resolve_dataset_dir("mind")
    data_dir = Path(data_dir)

    print(f"[mind-precompute] data_dir={data_dir}")
    articles = _load_articles(data_dir)
    if not articles:
        raise RuntimeError("no MIND articles found in raw train/dev news.tsv files")
    print(f"[mind-precompute] loaded {len(articles):,} unique articles from train+dev")

    clicks = _load_positive_clicks(data_dir)
    if not clicks:
        raise RuntimeError("no positive MIND clicks found in raw train/dev behaviors.tsv files")
    print(f"[mind-precompute] loaded {len(clicks):,} positive clicks from train+dev")

    peer_id_map = _build_peer_id_map(clicks)
    peer_id_map_path = data_dir / MIND_PEER_ID_MAP_FILENAME
    _write_peer_id_map(peer_id_map, peer_id_map_path)
    print(f"[mind-precompute] mapped {len(peer_id_map):,} MIND users to int peer_ids")

    doc_metadata_path = data_dir / MIND_DOC_METADATA_FILENAME
    doc_count = _write_doc_metadata(articles, doc_metadata_path)
    print(f"[mind-precompute] wrote {doc_count:,} article metadata rows to {doc_metadata_path}")

    records, missing_clicked_articles = _build_event_records(
        clicks,
        articles=articles,
        peer_id_map=peer_id_map,
    )
    if not records:
        raise RuntimeError("no MIND events remain after joining clicks to article metadata")
    if missing_clicked_articles:
        print(
            f"[mind-precompute] skipped {missing_clicked_articles:,} clicks whose articles "
            "were missing from news.tsv"
        )

    events_path = data_dir / MIND_EVENTS_FILENAME
    written = _write_events_csv(events_path, records)
    print(f"[mind-precompute] wrote {written:,} events to {events_path}")

    randomized_path = data_dir / MIND_RANDOMIZED_EVENTS_FILENAME
    randomized_count = _write_randomized_events(
        events_path,
        randomized_path,
        random_seed=random_seed,
    )
    print(f"[mind-precompute] wrote {randomized_count:,} randomized events to {randomized_path}")

    searchable_path = data_dir / MIND_SEEDED_SEARCHABLE_FILENAME
    sidecar_total, sidecar_searchable = _write_seeded_searchable_sidecar(
        records,
        output_path=searchable_path,
    )
    print(
        f"[mind-precompute] wrote {sidecar_total:,} searchable labels to "
        f"{searchable_path} ({sidecar_searchable:,} searchable)"
    )

    chronological_searchable_path = data_dir / CHRONOLOGICAL_SEARCHABLE_QUERIES_FILENAME
    chronological_summary_path = data_dir / "chronological_searchable_queries_summary.json"
    build_chronological_searchable_queries(
        events_path,
        chronological_searchable_path,
        chronological_summary_path,
    )
    print(
        f"[mind-precompute] wrote chronological searchable labels to "
        f"{chronological_searchable_path}"
    )

    peer_docs_path = data_dir / MIND_PEER_FUTURE_DOCS_FILENAME
    peer_docs_summary_path = data_dir / MIND_PEER_FUTURE_DOCS_SUMMARY_FILENAME
    peer_docs_summary = _write_peer_future_docs(
        records,
        output_path=peer_docs_path,
        summary_path=peer_docs_summary_path,
    )
    print(
        f"[mind-precompute] wrote seeded peer-doc map to {peer_docs_path} "
        f"(peer_count={peer_docs_summary['peer_count']}, "
        f"pairs={peer_docs_summary['total_unique_peer_doc_pairs']})"
    )

    query_id_map_path = data_dir / MIND_QUERY_ID_MAP_FILENAME
    _write_query_id_map(records, query_id_map_path)

    clicked_doc_ids = {record.target_doc_id for record in records}
    summary = {
        "data_dir": str(data_dir),
        "random_seed": random_seed,
        "article_count": len(articles),
        "positive_click_count_pre_join": len(clicks),
        "positive_event_count": len(records),
        "missing_clicked_article_count": missing_clicked_articles,
        "unique_user_count": len(peer_id_map),
        "unique_clicked_doc_count": len(clicked_doc_ids),
        "searchable_label_count": sidecar_total,
        "searchable_true_count": sidecar_searchable,
        "events_path": str(events_path),
        "randomized_events_path": str(randomized_path),
        "searchable_path": str(searchable_path),
        "chronological_searchable_path": str(chronological_searchable_path),
        "chronological_searchable_summary_path": str(chronological_summary_path),
        "peer_future_docs_path": str(peer_docs_path),
        "peer_id_map_path": str(peer_id_map_path),
        "query_id_map_path": str(query_id_map_path),
        "doc_metadata_path": str(doc_metadata_path),
    }
    summary_path = data_dir / MIND_PRECOMPUTE_SUMMARY_FILENAME
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[mind-precompute] summary written to {summary_path}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Precompute MIND-large train+dev into simulation-ready artifacts. "
            "Each positive click becomes a query whose text is the clicked "
            "article's title+abstract."
        )
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Override the resolved data directory (default: <data_root>/mind/)",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Seed for randomized-timeline shuffling",
    )
    args = parser.parse_args()
    precompute_mind_artifacts(
        data_dir=Path(args.data_dir) if args.data_dir else None,
        random_seed=args.random_seed,
    )


if __name__ == "__main__":
    main()

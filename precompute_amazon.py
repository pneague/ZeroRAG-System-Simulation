#!/usr/bin/env python3
"""Build all Amazon Product Search artifacts the simulation consumes.

Inputs (per category, under ``<data_root>/amazon/<category>/``):

* ``benchmark/query_text.txt.gz`` - one synthetic generated query per line
* ``benchmark/train.qrels.gz``    - qrels in ``<reviewerID>_<line> 0 <asin> <rel>`` form
* ``benchmark/test.qrels.gz``     - same format
* ``metadata/meta_<Category>.json.gz`` - UCSD per-category product metadata
  (joined by ``asin``)

Outputs (written next to the inputs):

* ``events.csv``                       - AOL-compatible columns; positive only
* ``metadata_seeded_randomized.csv.gz`` - randomized-timeline copy
* ``seeded_searchable_queries.csv.gz`` - sidecar marking qids ``searchable``
                                          iff >=2 distinct reviewers have the
                                          target asin as a positive
* ``peer_future_doc_ids.json.gz``      - peer_id -> sorted list of asins
* ``peer_future_doc_ids_summary.json``
* ``peer_id_map.json.gz``              - reviewerID(str) -> peer_id(int)
* ``doc_metadata.csv.gz``              - asin -> normalized product fields
* ``amazon_precompute_summary.json``

The event stream merges train+test qrels, keeps only ``relevance_label==1``
(real user-item purchases/reviews), and joins each row to the synthetic
generated query indexed by ``query_line_number`` and to the UCSD product
metadata by ``asin``. We never describe Amazon events as 'user searched
for product'.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from src.amazon_dataset import (
    AMAZON_CATEGORY_SPECS,
    AMAZON_DOC_METADATA_FILENAME,
    AMAZON_EVENTS_FILENAME,
    AMAZON_PEER_FUTURE_DOCS_FILENAME,
    AMAZON_PEER_FUTURE_DOCS_SUMMARY_FILENAME,
    AMAZON_PEER_ID_MAP_FILENAME,
    AMAZON_PRECOMPUTE_SUMMARY_FILENAME,
    AMAZON_RANDOMIZED_EVENTS_FILENAME,
    AMAZON_SEEDED_SEARCHABLE_FILENAME,
    category_spec,
    iter_qrels,
    iter_ucsd_metadata_records,
    load_query_texts,
    normalize_ucsd_record,
)
from src.data_root import resolve_dataset_dir
from src.dataset import (
    AMAZON_EVENTS_FIELDNAMES,
    AMAZON_QUERY_AVAILABILITY_FILENAME,
    RELEVANT_DOC_IDS_BY_QID_FILENAME,
    SEARCHABLE_QUERY_FIELDNAMES,
    synthetic_timestamp_for_index,
)


DOC_METADATA_FIELDNAMES = (
    "asin",
    "title",
    "description",
    "categories",
    "brand",
    "price",
)


def _open_text_writer(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "wt", newline="", encoding="utf-8")
    return path.open("w", newline="", encoding="utf-8")


def _write_json_gz(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle)


def _collect_positive_rows(
    qrels_paths: list[Path],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in qrels_paths:
        for record in iter_qrels(path):
            if record["relevance_label"] != 1:
                continue
            rows.append(record)
    return rows


def _build_peer_id_map(rows: list[dict[str, object]]) -> dict[str, int]:
    unique_reviewer_ids = sorted({str(row["reviewer_id"]) for row in rows})
    return {reviewer_id: idx for idx, reviewer_id in enumerate(unique_reviewer_ids)}


def _build_relevant_docs_by_qid(rows: list[dict[str, object]]) -> dict[int, set[str]]:
    relevant: dict[int, set[str]] = defaultdict(set)
    for row in rows:
        relevant[int(row["query_line_number"])].add(str(row["asin"]))
    return relevant


def _sample_count(relevant_count: int) -> int:
    if relevant_count <= 0:
        return 0
    return max(1, min(int(round(0.1 * relevant_count)), 10))


def _build_sampled_peer_docs(
    rows: list[dict[str, object]],
    *,
    peer_id_map: dict[str, int],
    relevant_docs_by_qid: dict[int, set[str]],
    random_seed: int,
) -> dict[int, set[str]]:
    user_qids = sorted(
        {
            (str(row["reviewer_id"]), int(row["query_line_number"]))
            for row in rows
        },
        key=lambda item: (item[0], item[1]),
    )
    rng = np.random.default_rng(random_seed)
    docs_by_peer: dict[int, set[str]] = defaultdict(set)
    for reviewer_id, qid in user_qids:
        relevant_docs = sorted(relevant_docs_by_qid[qid])
        sample_size = _sample_count(len(relevant_docs))
        if sample_size == 0:
            continue
        sampled = rng.choice(relevant_docs, size=sample_size, replace=True)
        docs_by_peer[peer_id_map[reviewer_id]].update(str(doc_id) for doc_id in sampled.tolist())
    return docs_by_peer


def _build_user_query_events(
    rows: list[dict[str, object]],
    *,
    relevant_docs_by_qid: dict[int, set[str]],
) -> list[dict[str, object]]:
    events: list[dict[str, object]] = []
    for reviewer_id, qid in sorted(
        {
            (str(row["reviewer_id"]), int(row["query_line_number"]))
            for row in rows
        },
        key=lambda item: (item[0], item[1]),
    ):
        representative = sorted(relevant_docs_by_qid[qid])[0]
        events.append(
            {
                "reviewer_id": reviewer_id,
                "query_line_number": qid,
                "asin": representative,
            }
        )
    return events


def _write_relevant_docs_by_qid(
    relevant_docs_by_qid: dict[int, set[str]],
    output_path: Path,
) -> None:
    payload = {
        str(qid): sorted(doc_ids)
        for qid, doc_ids in sorted(relevant_docs_by_qid.items())
    }
    _write_json_gz(output_path, payload)


def _write_availability(
    relevant_docs_by_qid: dict[int, set[str]],
    docs_by_peer: dict[int, set[str]],
    output_path: Path,
) -> dict[str, dict[str, int]]:
    owners_by_doc: dict[str, set[int]] = defaultdict(set)
    for peer_id, doc_ids in docs_by_peer.items():
        for doc_id in doc_ids:
            owners_by_doc[doc_id].add(peer_id)
    payload: dict[str, dict[str, int]] = {}
    for qid, relevant_docs in sorted(relevant_docs_by_qid.items()):
        available_docs = [doc_id for doc_id in relevant_docs if owners_by_doc.get(doc_id)]
        owner_assignments = sum(len(owners_by_doc[doc_id]) for doc_id in available_docs)
        payload[str(qid)] = {
            "relevant_doc_count": len(relevant_docs),
            "available_relevant_doc_count": len(available_docs),
            "available_relevant_owner_assignments": owner_assignments,
        }
    _write_json_gz(output_path, payload)
    return payload


def _stream_doc_metadata(
    metadata_path: Path,
    needed_asins: set[str],
    output_path: Path,
) -> tuple[int, int]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    found: set[str] = set()
    written = 0
    with _open_text_writer(output_path) as handle:
        writer = csv.DictWriter(handle, fieldnames=DOC_METADATA_FIELDNAMES)
        writer.writeheader()
        for record in iter_ucsd_metadata_records(metadata_path):
            normalized = normalize_ucsd_record(record)
            asin = normalized["asin"]
            if not asin or asin not in needed_asins or asin in found:
                continue
            found.add(asin)
            writer.writerow({field: normalized.get(field, "") for field in DOC_METADATA_FIELDNAMES})
            written += 1
            if len(found) == len(needed_asins):
                break
    missing = len(needed_asins) - len(found)
    return written, missing


def _write_events_csv(
    output_path: Path,
    rows: list[dict[str, object]],
    *,
    query_texts: list[str],
    peer_id_map: dict[str, int],
) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=AMAZON_EVENTS_FIELDNAMES)
        writer.writeheader()
        for event_index, row in enumerate(rows):
            line_number = int(row["query_line_number"])
            if line_number < 0 or line_number >= len(query_texts):
                raise ValueError(
                    f"qrels row references missing query line {line_number}; "
                    f"query_text has {len(query_texts)} lines"
                )
            writer.writerow(
                {
                    "timestamp": synthetic_timestamp_for_index(event_index),
                    "qid": line_number,
                    "user_id": peer_id_map[str(row["reviewer_id"])],
                    "query": query_texts[line_number],
                    "target_doc_id": str(row["asin"]),
                    "candidate_doc_ids": "[]",
                }
            )
            written += 1
    return written


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
    relevant_docs_by_qid: dict[int, set[str]],
    *,
    docs_by_peer: dict[int, set[str]],
    output_path: Path,
) -> tuple[int, int]:
    owned_docs = set().union(*docs_by_peer.values()) if docs_by_peer else set()
    searchable_by_qid: dict[int, bool] = {}
    for qid, relevant_docs in relevant_docs_by_qid.items():
        searchable_by_qid[qid] = bool(relevant_docs & owned_docs)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    searchable_count = 0
    with _open_text_writer(output_path) as handle:
        writer = csv.DictWriter(handle, fieldnames=SEARCHABLE_QUERY_FIELDNAMES)
        writer.writeheader()
        for qid in sorted(searchable_by_qid):
            value = searchable_by_qid[qid]
            writer.writerow(
                {
                    "query_id": qid,
                    "searchable": "true" if value else "false",
                }
            )
            searchable_count += int(value)
    return len(searchable_by_qid), searchable_count


def _write_peer_future_docs(
    docs_by_peer: dict[int, set[str]],
    *,
    output_path: Path,
    summary_path: Path,
) -> dict[str, int]:
    payload = {
        str(peer_id): sorted(asins)
        for peer_id, asins in sorted(docs_by_peer.items())
    }
    _write_json_gz(output_path, payload)

    total_pairs = sum(len(asins) for asins in payload.values())
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


def _write_peer_id_map(peer_id_map: dict[str, int], output_path: Path) -> None:
    payload = {reviewer_id: peer_id for reviewer_id, peer_id in peer_id_map.items()}
    _write_json_gz(output_path, payload)


def precompute_amazon_artifacts(
    category: str,
    *,
    data_dir: Path | None = None,
    random_seed: int = 42,
) -> dict[str, object]:
    if category == "all":
        raise SystemExit(
            "--category=all is not valid for precompute_amazon.py; use "
            "precompute_amazon_all.py to build the combined dataset from the "
            "four per-category raw inputs."
        )
    spec = category_spec(category)
    if data_dir is None:
        data_dir = resolve_dataset_dir("amazon", category)
    data_dir = Path(data_dir)

    benchmark_dir = data_dir / "benchmark"
    metadata_path = data_dir / "metadata" / spec.ucsd_meta_filename
    query_text_path = benchmark_dir / "query_text.txt.gz"
    train_qrels_path = benchmark_dir / "train.qrels.gz"
    test_qrels_path = benchmark_dir / "test.qrels.gz"

    for required in (query_text_path, train_qrels_path, test_qrels_path, metadata_path):
        if not required.exists():
            raise FileNotFoundError(
                f"missing required input {required}; run download_amazon.py first"
            )

    print(f"[amazon-precompute] category={category} data_dir={data_dir}")

    query_texts = load_query_texts(query_text_path)
    print(f"[amazon-precompute] loaded {len(query_texts):,} generated queries")

    positive_rows = _collect_positive_rows([train_qrels_path, test_qrels_path])
    if not positive_rows:
        raise RuntimeError("no positive (relevance==1) rows found in qrels")
    print(
        f"[amazon-precompute] kept {len(positive_rows):,} positive (reviewerID,query,asin) rows"
    )

    peer_id_map = _build_peer_id_map(positive_rows)
    peer_id_map_path = data_dir / AMAZON_PEER_ID_MAP_FILENAME
    _write_peer_id_map(peer_id_map, peer_id_map_path)
    print(
        f"[amazon-precompute] mapped {len(peer_id_map):,} reviewerIDs to int peer_ids"
    )

    needed_asins = {str(row["asin"]) for row in positive_rows}
    doc_metadata_path = data_dir / AMAZON_DOC_METADATA_FILENAME
    found, missing = _stream_doc_metadata(metadata_path, needed_asins, doc_metadata_path)
    print(
        f"[amazon-precompute] joined {found:,}/{len(needed_asins):,} asins from "
        f"UCSD metadata ({missing:,} missing)"
    )

    relevant_docs_by_qid = _build_relevant_docs_by_qid(positive_rows)
    relevant_docs_path = data_dir / RELEVANT_DOC_IDS_BY_QID_FILENAME
    _write_relevant_docs_by_qid(relevant_docs_by_qid, relevant_docs_path)
    print(
        f"[amazon-precompute] wrote {len(relevant_docs_by_qid):,} qid relevant-doc sets to "
        f"{relevant_docs_path}"
    )

    sampled_docs_by_peer = _build_sampled_peer_docs(
        positive_rows,
        peer_id_map=peer_id_map,
        relevant_docs_by_qid=relevant_docs_by_qid,
        random_seed=random_seed,
    )
    availability_path = data_dir / AMAZON_QUERY_AVAILABILITY_FILENAME
    availability = _write_availability(
        relevant_docs_by_qid, sampled_docs_by_peer, availability_path
    )
    print(f"[amazon-precompute] wrote query availability to {availability_path}")

    user_query_events = _build_user_query_events(
        positive_rows, relevant_docs_by_qid=relevant_docs_by_qid
    )

    events_path = data_dir / AMAZON_EVENTS_FILENAME
    written = _write_events_csv(
        events_path,
        user_query_events,
        query_texts=query_texts,
        peer_id_map=peer_id_map,
    )
    print(f"[amazon-precompute] wrote {written:,} events to {events_path}")

    randomized_path = data_dir / AMAZON_RANDOMIZED_EVENTS_FILENAME
    randomized_count = _write_randomized_events(
        events_path, randomized_path, random_seed=random_seed
    )
    print(
        f"[amazon-precompute] wrote {randomized_count:,} randomized events to "
        f"{randomized_path}"
    )

    searchable_path = data_dir / AMAZON_SEEDED_SEARCHABLE_FILENAME
    sidecar_total, sidecar_searchable = _write_seeded_searchable_sidecar(
        relevant_docs_by_qid,
        docs_by_peer=sampled_docs_by_peer,
        output_path=searchable_path,
    )
    print(
        f"[amazon-precompute] wrote {sidecar_total:,} searchable labels to "
        f"{searchable_path} ({sidecar_searchable:,} searchable)"
    )

    peer_docs_path = data_dir / AMAZON_PEER_FUTURE_DOCS_FILENAME
    peer_docs_summary_path = data_dir / AMAZON_PEER_FUTURE_DOCS_SUMMARY_FILENAME
    peer_docs_summary = _write_peer_future_docs(
        sampled_docs_by_peer,
        output_path=peer_docs_path,
        summary_path=peer_docs_summary_path,
    )
    print(
        f"[amazon-precompute] wrote seeded peer-doc map to {peer_docs_path} "
        f"(peer_count={peer_docs_summary['peer_count']}, "
        f"pairs={peer_docs_summary['total_unique_peer_doc_pairs']})"
    )

    summary = {
        "category": category,
        "category_display_name": spec.display_name,
        "data_dir": str(data_dir),
        "random_seed": random_seed,
        "query_text_lines": len(query_texts),
        "positive_event_count": len(positive_rows),
        "user_query_event_count": len(user_query_events),
        "relevant_qid_count": len(relevant_docs_by_qid),
        "available_relevant_qid_count": sum(
            1 for item in availability.values() if item["available_relevant_doc_count"] > 0
        ),
        "unique_reviewer_count": len(peer_id_map),
        "unique_asin_count": len(needed_asins),
        "doc_metadata_found": found,
        "doc_metadata_missing": missing,
        "searchable_label_count": sidecar_total,
        "searchable_true_count": sidecar_searchable,
        "events_path": str(events_path),
        "randomized_events_path": str(randomized_path),
        "searchable_path": str(searchable_path),
        "relevant_docs_by_qid_path": str(relevant_docs_path),
        "query_availability_path": str(availability_path),
        "peer_future_docs_path": str(peer_docs_path),
        "peer_id_map_path": str(peer_id_map_path),
        "doc_metadata_path": str(doc_metadata_path),
    }
    summary_path = data_dir / AMAZON_PRECOMPUTE_SUMMARY_FILENAME
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[amazon-precompute] summary written to {summary_path}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Precompute Amazon Product Search artifacts. Combines train+test "
            "qrels (relevance==1), joins to UCSD per-category metadata by asin, "
            "and writes simulation-ready inputs."
        )
    )
    parser.add_argument(
        "--category",
        choices=sorted(AMAZON_CATEGORY_SPECS),
        default="cell_phones",
        help="Amazon benchmark category (default: cell_phones)",
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help=(
            "Override the resolved data directory (default: "
            "<data_root>/amazon/<category>/)"
        ),
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Seed for randomized-timeline shuffling",
    )
    args = parser.parse_args()
    precompute_amazon_artifacts(
        args.category,
        data_dir=Path(args.data_dir) if args.data_dir else None,
        random_seed=args.random_seed,
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Build a single combined Amazon dataset from all 4 categories.

Reads RAW per-category inputs (no per-category precompute needed) from
``<data_root>/amazon/<cat>/{benchmark,metadata}/`` for each of
``cell_phones``, ``electronics``, ``kindle``, ``cds`` and writes the merged
artifacts under ``<data_root>/amazon/all/``.

Identity rules (intentional, locked):

* reviewer identity is **global**: the same ``reviewerID`` string across
  categories collapses to one ``peer_id`` and that peer can own asins from
  any category.
* qids are renumbered globally so each ``(category, original_query_line_number)``
  becomes a fresh sequential int. The original mapping is preserved in
  ``qid_map.json.gz`` so analysis can recover provenance.
* asins are globally unique; the combined ``doc_metadata.csv.gz`` is the
  union over all 4 categories' UCSD ``meta_<Category>.json.gz`` files.
* events whose target asin has **no UCSD metadata** are expunged from the
  dataset, and qids / peers that are left with no remaining events are
  dropped from the qid map and peer map. With the 2014 SNAP per-category
  metadata source (the matching companion to the QingyaoAi qrels), every
  qrels asin resolves and ``expunged_event_count`` is expected to be 0;
  the expunge path is retained as a safety net so an accidental switch
  to a different metadata source does not silently produce events that
  the simulator would later flag as ``non_searchable``.

Outputs (written under ``<data_root>/amazon/all/``):

* ``events.csv``                       - AOL-compatible columns
* ``metadata_seeded_randomized.csv.gz`` - randomized-timeline copy
* ``seeded_searchable_queries.csv.gz`` - sidecar
* ``peer_future_doc_ids.json.gz`` (+ summary)
* ``peer_id_map.json.gz``              - reviewerID(str) -> peer_id(int)
* ``qid_map.json.gz``                  - new_qid(int) -> {category, original_query_line_number, query}
* ``doc_metadata.csv.gz``              - union over all categories
* ``amazon_precompute_summary.json``   - per-category breakdown + totals
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.amazon_dataset import (
    AMAZON_DOC_METADATA_FILENAME,
    AMAZON_EVENTS_FILENAME,
    AMAZON_PEER_FUTURE_DOCS_FILENAME,
    AMAZON_PEER_FUTURE_DOCS_SUMMARY_FILENAME,
    AMAZON_PEER_ID_MAP_FILENAME,
    AMAZON_PRECOMPUTE_SUMMARY_FILENAME,
    AMAZON_RANDOMIZED_EVENTS_FILENAME,
    AMAZON_REAL_CATEGORIES,
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

from precompute_amazon import (
    DOC_METADATA_FIELDNAMES,
    _build_relevant_docs_by_qid,
    _build_sampled_peer_docs,
    _build_user_query_events,
    _open_text_writer,
    _write_availability,
    _write_json_gz,
    _write_peer_future_docs,
    _write_peer_id_map,
    _write_randomized_events,
    _write_relevant_docs_by_qid,
    _write_seeded_searchable_sidecar,
)


COMBINED_CATEGORY_KEY = "all"
QID_MAP_FILENAME = "qid_map.json.gz"


def _resolve_category_dir(category: str, data_root_override: Path | None) -> Path:
    if data_root_override is not None:
        return data_root_override / "amazon" / category
    return resolve_dataset_dir("amazon", category)


def _validate_inputs(category_dirs: dict[str, Path]) -> None:
    missing: list[Path] = []
    for category, base in category_dirs.items():
        spec = category_spec(category)
        for required in (
            base / "benchmark" / "query_text.txt.gz",
            base / "benchmark" / "train.qrels.gz",
            base / "benchmark" / "test.qrels.gz",
            base / "metadata" / spec.ucsd_meta_filename,
        ):
            if not required.exists():
                missing.append(required)
    if missing:
        formatted = "\n  - ".join(str(p) for p in missing)
        raise FileNotFoundError(
            "missing required raw Amazon inputs (run download_amazon.py for "
            f"each missing category first):\n  - {formatted}"
        )


def _collect_positive_rows_for_category(
    category: str,
    base_dir: Path,
) -> tuple[list[dict[str, object]], list[str]]:
    """Return (positive_rows_with_category_tag, query_texts_for_this_category)."""

    query_texts = load_query_texts(base_dir / "benchmark" / "query_text.txt.gz")
    rows: list[dict[str, object]] = []
    for qrels_name in ("train.qrels.gz", "test.qrels.gz"):
        for record in iter_qrels(base_dir / "benchmark" / qrels_name):
            if record["relevance_label"] != 1:
                continue
            rows.append(
                {
                    "category": category,
                    "reviewer_id": str(record["reviewer_id"]),
                    "query_line_number": int(record["query_line_number"]),
                    "asin": str(record["asin"]),
                    "relevance_label": 1,
                }
            )
    return rows, query_texts


def _build_global_qid_map(
    rows: list[dict[str, object]],
    *,
    query_texts_by_category: dict[str, list[str]],
) -> tuple[dict[tuple[str, int], int], dict[str, dict[str, object]]]:
    """Assign each (category, original_query_line_number) pair a unique global qid.

    Returns:
      key_to_qid: (category, original_qid) -> new_qid
      qid_map: str(new_qid) -> {category, original_query_line_number, query}
    """

    seen_keys: set[tuple[str, int]] = set()
    ordered_keys: list[tuple[str, int]] = []
    for row in rows:
        key = (str(row["category"]), int(row["query_line_number"]))
        if key in seen_keys:
            continue
        seen_keys.add(key)
        ordered_keys.append(key)
    ordered_keys.sort(key=lambda key: (AMAZON_REAL_CATEGORIES.index(key[0]), key[1]))

    key_to_qid: dict[tuple[str, int], int] = {}
    qid_map: dict[str, dict[str, object]] = {}
    for new_qid, key in enumerate(ordered_keys):
        category, original_qid = key
        query_texts = query_texts_by_category[category]
        if original_qid < 0 or original_qid >= len(query_texts):
            raise ValueError(
                f"qrels in category={category} reference query line "
                f"{original_qid}; query_text has {len(query_texts)} lines"
            )
        key_to_qid[key] = new_qid
        qid_map[str(new_qid)] = {
            "category": category,
            "original_query_line_number": original_qid,
            "query": query_texts[original_qid],
        }
    return key_to_qid, qid_map


def _build_peer_id_map(rows: list[dict[str, object]]) -> dict[str, int]:
    unique_reviewer_ids = sorted({str(row["reviewer_id"]) for row in rows})
    return {reviewer_id: idx for idx, reviewer_id in enumerate(unique_reviewer_ids)}


def _stream_combined_doc_metadata(
    category_dirs: dict[str, Path],
    needed_asins: set[str],
    output_path: Path,
) -> tuple[int, int, dict[str, int], set[str]]:
    """Stream every category's UCSD metadata, write a deduped union CSV.

    Returns (written, missing, found_per_category, found_asins).
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)
    found: set[str] = set()
    found_per_category: dict[str, int] = {category: 0 for category in category_dirs}
    written = 0
    with _open_text_writer(output_path) as handle:
        writer = csv.DictWriter(handle, fieldnames=DOC_METADATA_FIELDNAMES)
        writer.writeheader()
        for category, base in category_dirs.items():
            spec = category_spec(category)
            metadata_path = base / "metadata" / spec.ucsd_meta_filename
            print(f"[amazon-all] streaming {metadata_path}")
            for record in iter_ucsd_metadata_records(metadata_path):
                normalized = normalize_ucsd_record(record)
                asin = normalized["asin"]
                if not asin or asin not in needed_asins or asin in found:
                    continue
                found.add(asin)
                found_per_category[category] += 1
                writer.writerow(
                    {field: normalized.get(field, "") for field in DOC_METADATA_FIELDNAMES}
                )
                written += 1
                if len(found) == len(needed_asins):
                    break
            if len(found) == len(needed_asins):
                break
    missing = len(needed_asins) - len(found)
    return written, missing, found_per_category, found


def _write_combined_events_csv(
    output_path: Path,
    rows: list[dict[str, object]],
    *,
    qid_map: dict[str, dict[str, object]],
    peer_id_map: dict[str, int],
) -> int:
    """Write events.csv where each row uses the global qid + original query string."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=AMAZON_EVENTS_FIELDNAMES)
        writer.writeheader()
        for event_index, row in enumerate(rows):
            new_qid = int(row["query_line_number"])
            query_text = qid_map[str(new_qid)]["query"]
            writer.writerow(
                {
                    "timestamp": synthetic_timestamp_for_index(event_index),
                    "qid": new_qid,
                    "user_id": peer_id_map[str(row["reviewer_id"])],
                    "query": query_text,
                    "target_doc_id": str(row["asin"]),
                    "candidate_doc_ids": "[]",
                }
            )
            written += 1
    return written


def _write_combined_seeded_searchable_sidecar(
    relevant_docs_by_qid: dict[int, set[str]],
    *,
    docs_by_peer: dict[int, set[str]],
    output_path: Path,
) -> tuple[int, int]:
    """Same logic as the per-category sidecar but keyed by the global qid."""
    return _write_seeded_searchable_sidecar(
        relevant_docs_by_qid, docs_by_peer=docs_by_peer, output_path=output_path
    )


def precompute_amazon_all_artifacts(
    *,
    data_root_override: Path | None = None,
    output_dir: Path | None = None,
    random_seed: int = 42,
) -> dict[str, object]:
    category_dirs = {
        category: _resolve_category_dir(category, data_root_override)
        for category in AMAZON_REAL_CATEGORIES
    }
    _validate_inputs(category_dirs)

    if output_dir is None:
        output_dir = _resolve_category_dir(COMBINED_CATEGORY_KEY, data_root_override)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[amazon-all] output_dir={output_dir}")

    all_rows: list[dict[str, object]] = []
    query_texts_by_category: dict[str, list[str]] = {}
    per_category_event_counts_pre: dict[str, int] = {}
    per_category_query_text_counts: dict[str, int] = {}
    for category, base in category_dirs.items():
        print(f"[amazon-all] reading category={category} from {base}")
        rows, query_texts = _collect_positive_rows_for_category(category, base)
        query_texts_by_category[category] = query_texts
        per_category_query_text_counts[category] = len(query_texts)
        per_category_event_counts_pre[category] = len(rows)
        all_rows.extend(rows)
        print(
            f"[amazon-all]   loaded {len(query_texts):,} query texts, "
            f"{len(rows):,} positive rows"
        )

    if not all_rows:
        raise RuntimeError("no positive rows collected across categories")

    pre_filter_event_count = len(all_rows)
    pre_filter_unique_asins = {str(row["asin"]) for row in all_rows}
    pre_filter_unique_reviewers = {str(row["reviewer_id"]) for row in all_rows}
    pre_filter_unique_qid_keys = {
        (str(row["category"]), int(row["query_line_number"])) for row in all_rows
    }

    # Stream UCSD metadata FIRST so we know which asins are reachable (have
    # metadata text we can embed). Anything else is unreachable for routing
    # because no peer can ever own a doc that has no embedding; we expunge
    # those events from the dataset rather than letting the simulator skip
    # them at runtime as `non_searchable`.
    doc_metadata_path = output_dir / AMAZON_DOC_METADATA_FILENAME
    found, missing, found_per_category, found_asins = _stream_combined_doc_metadata(
        category_dirs, pre_filter_unique_asins, doc_metadata_path
    )
    print(
        f"[amazon-all] joined {found:,}/{len(pre_filter_unique_asins):,} asins "
        f"from UCSD metadata across categories ({missing:,} missing - those "
        f"events will be expunged)"
    )
    if missing > 0:
        # With the 2014 SNAP per-category metadata source we expect every qrels
        # asin to resolve. A non-zero miss count means either a stale 2018-era
        # cached meta file is still on disk, or the metadata source URL was
        # changed away from snap.stanford.edu. The expunge path below still
        # runs (so the dataset stays embeddable), but emit a loud warning so
        # the operator notices.
        print(
            f"[amazon-all] WARNING: {missing:,} asins still missing after "
            f"the metadata join; expected 0 with the 2014 SNAP metadata "
            f"source. Check that meta_*.json.gz under each category's "
            f"metadata/ dir was downloaded from "
            f"http://snap.stanford.edu/data/amazon/productGraph/categoryFiles "
            f"and not the older 2018 amazon_v2 host."
        )

    expunged_per_category: dict[str, int] = {category: 0 for category in category_dirs}
    filtered_rows: list[dict[str, object]] = []
    for row in all_rows:
        if str(row["asin"]) in found_asins:
            filtered_rows.append(row)
        else:
            expunged_per_category[str(row["category"])] += 1
    expunged_event_count = pre_filter_event_count - len(filtered_rows)
    print(
        f"[amazon-all] expunged {expunged_event_count:,}/{pre_filter_event_count:,} "
        f"events whose target asin has no UCSD metadata "
        f"({len(filtered_rows):,} survive)"
    )
    if not filtered_rows:
        raise RuntimeError(
            "every event was expunged because no asin had UCSD metadata; "
            "check the per-category meta_*.json.gz files exist and are non-empty"
        )

    # Free the un-filtered list immediately; the rest of the pipeline only
    # uses filtered_rows so we don't want a stale duplicate sitting in memory
    # for the multi-million-row case.
    all_rows.clear()
    del all_rows

    per_category_event_counts_post: dict[str, int] = {
        category: 0 for category in category_dirs
    }
    for row in filtered_rows:
        per_category_event_counts_post[str(row["category"])] += 1

    key_to_qid, qid_map = _build_global_qid_map(
        filtered_rows, query_texts_by_category=query_texts_by_category
    )
    print(
        f"[amazon-all] assigned {len(key_to_qid):,} unique global qids "
        f"(pre-expunge {len(pre_filter_unique_qid_keys):,} unique "
        f"(category, original_qid) pairs)"
    )

    qid_map_path = output_dir / QID_MAP_FILENAME
    _write_json_gz(qid_map_path, qid_map)
    print(f"[amazon-all] wrote qid map to {qid_map_path}")

    peer_id_map = _build_peer_id_map(filtered_rows)
    peer_id_map_path = output_dir / AMAZON_PEER_ID_MAP_FILENAME
    _write_peer_id_map(peer_id_map, peer_id_map_path)
    print(
        f"[amazon-all] mapped {len(peer_id_map):,} unique reviewerIDs to int "
        f"peer_ids (pre-expunge {len(pre_filter_unique_reviewers):,})"
    )

    global_rows: list[dict[str, object]] = []
    for row in filtered_rows:
        key = (str(row["category"]), int(row["query_line_number"]))
        global_rows.append(
            {
                "reviewer_id": str(row["reviewer_id"]),
                "query_line_number": key_to_qid[key],
                "asin": str(row["asin"]),
            }
        )

    relevant_docs_by_qid = _build_relevant_docs_by_qid(global_rows)
    relevant_docs_path = output_dir / RELEVANT_DOC_IDS_BY_QID_FILENAME
    _write_relevant_docs_by_qid(relevant_docs_by_qid, relevant_docs_path)
    print(
        f"[amazon-all] wrote {len(relevant_docs_by_qid):,} qid relevant-doc sets to "
        f"{relevant_docs_path}"
    )

    sampled_docs_by_peer = _build_sampled_peer_docs(
        global_rows,
        peer_id_map=peer_id_map,
        relevant_docs_by_qid=relevant_docs_by_qid,
        random_seed=random_seed,
    )
    availability_path = output_dir / AMAZON_QUERY_AVAILABILITY_FILENAME
    availability = _write_availability(
        relevant_docs_by_qid, sampled_docs_by_peer, availability_path
    )
    print(f"[amazon-all] wrote query availability to {availability_path}")

    user_query_events = _build_user_query_events(
        global_rows, relevant_docs_by_qid=relevant_docs_by_qid
    )

    events_path = output_dir / AMAZON_EVENTS_FILENAME
    written = _write_combined_events_csv(
        events_path,
        user_query_events,
        qid_map=qid_map,
        peer_id_map=peer_id_map,
    )
    print(f"[amazon-all] wrote {written:,} events to {events_path}")

    randomized_path = output_dir / AMAZON_RANDOMIZED_EVENTS_FILENAME
    randomized_count = _write_randomized_events(
        events_path, randomized_path, random_seed=random_seed
    )
    print(
        f"[amazon-all] wrote {randomized_count:,} randomized events to "
        f"{randomized_path}"
    )

    searchable_path = output_dir / AMAZON_SEEDED_SEARCHABLE_FILENAME
    sidecar_total, sidecar_searchable = _write_combined_seeded_searchable_sidecar(
        relevant_docs_by_qid,
        docs_by_peer=sampled_docs_by_peer,
        output_path=searchable_path,
    )
    print(
        f"[amazon-all] wrote {sidecar_total:,} searchable labels to "
        f"{searchable_path} ({sidecar_searchable:,} searchable)"
    )

    peer_docs_path = output_dir / AMAZON_PEER_FUTURE_DOCS_FILENAME
    peer_docs_summary_path = output_dir / AMAZON_PEER_FUTURE_DOCS_SUMMARY_FILENAME
    peer_docs_summary = _write_peer_future_docs(
        sampled_docs_by_peer,
        output_path=peer_docs_path,
        summary_path=peer_docs_summary_path,
    )
    print(
        f"[amazon-all] wrote seeded peer-doc map to {peer_docs_path} "
        f"(peer_count={peer_docs_summary['peer_count']}, "
        f"pairs={peer_docs_summary['total_unique_peer_doc_pairs']})"
    )

    summary = {
        "category": COMBINED_CATEGORY_KEY,
        "category_display_name": "All Categories Combined",
        "data_dir": str(output_dir),
        "random_seed": random_seed,
        "real_categories": list(AMAZON_REAL_CATEGORIES),
        "per_category_query_text_lines": per_category_query_text_counts,
        "per_category_event_count_pre_expunge": per_category_event_counts_pre,
        "per_category_event_count": per_category_event_counts_post,
        "per_category_events_expunged": expunged_per_category,
        "per_category_doc_metadata_found": found_per_category,
        "global_qid_count": len(key_to_qid),
        "global_qid_count_pre_expunge": len(pre_filter_unique_qid_keys),
        "total_event_count": len(filtered_rows),
        "total_user_query_event_count": len(user_query_events),
        "total_event_count_pre_expunge": pre_filter_event_count,
        "expunged_event_count": expunged_event_count,
        "available_relevant_qid_count": sum(
            1 for item in availability.values() if item["available_relevant_doc_count"] > 0
        ),
        "unique_reviewer_count": len(peer_id_map),
        "unique_reviewer_count_pre_expunge": len(pre_filter_unique_reviewers),
        "unique_asin_count": len(found_asins),
        "unique_asin_count_pre_expunge": len(pre_filter_unique_asins),
        "doc_metadata_found": found,
        "doc_metadata_missing": missing,
        "searchable_label_count": sidecar_total,
        "searchable_true_count": sidecar_searchable,
        "events_path": str(events_path),
        "randomized_events_path": str(randomized_path),
        "searchable_path": str(searchable_path),
        "peer_future_docs_path": str(peer_docs_path),
        "peer_id_map_path": str(peer_id_map_path),
        "qid_map_path": str(qid_map_path),
        "doc_metadata_path": str(doc_metadata_path),
        "relevant_docs_by_qid_path": str(relevant_docs_path),
        "query_availability_path": str(availability_path),
    }
    summary_path = output_dir / AMAZON_PRECOMPUTE_SUMMARY_FILENAME
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[amazon-all] summary written to {summary_path}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build the combined Amazon dataset from all 4 categories' raw "
            "inputs (cell_phones, electronics, kindle, cds). Writes simulator-"
            "ready artifacts under <data_root>/amazon/all/."
        )
    )
    parser.add_argument(
        "--data-root",
        default=None,
        help=(
            "Override the parent dir that holds amazon/<cat>/ for every "
            "category (default: resolved per-category via "
            "src.data_root.resolve_dataset_dir, which honors "
            "LOW_COORD_DHT_DATA_ROOT)"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Override the combined output directory (default: "
            "<data_root>/amazon/all/)"
        ),
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Seed for randomized-timeline shuffling",
    )
    args = parser.parse_args()
    precompute_amazon_all_artifacts(
        data_root_override=Path(args.data_root) if args.data_root else None,
        output_dir=Path(args.output_dir) if args.output_dir else None,
        random_seed=args.random_seed,
    )


if __name__ == "__main__":
    main()

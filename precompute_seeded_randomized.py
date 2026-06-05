#!/usr/bin/env python3
"""Build seeded searchable labels and randomized seeded metadata artifacts."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from src.dataset import (
    METADATA_FILENAME,
    SEARCHABLE_QUERY_FIELDNAMES,
    SEEDED_RANDOMIZED_METADATA_FILENAME,
    SEEDED_SEARCHABLE_QUERIES_FILENAME,
    TIMESTAMP_FORMAT,
    iter_metadata_rows,
)


def _open_output(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "wt", newline="", encoding="utf-8")
    return path.open("w", newline="", encoding="utf-8")


def _load_seeded_doc_mapping(path: Path) -> dict[int, set[str]]:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            payload = json.load(handle)
    else:
        payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        int(peer_id): {str(doc_id) for doc_id in doc_ids}
        for peer_id, doc_ids in payload.items()
    }


def _write_metadata_rows(
    output_path: Path,
    *,
    fieldnames: list[str],
    rows: list[dict[str, str]],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with _open_output(output_path) as output_handle:
        writer = csv.DictWriter(output_handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_searchable_rows(output_path: Path, searchable_by_qid: dict[int, bool]) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    searchable_rows = 0
    with _open_output(output_path) as output_handle:
        writer = csv.DictWriter(output_handle, fieldnames=SEARCHABLE_QUERY_FIELDNAMES)
        writer.writeheader()
        for qid in sorted(searchable_by_qid):
            searchable = searchable_by_qid[qid]
            writer.writerow(
                {
                    "query_id": qid,
                    "searchable": "true" if searchable else "false",
                }
            )
            searchable_rows += int(searchable)
    return searchable_rows


def build_seeded_randomized_artifacts(
    input_path: Path,
    seeded_peer_docs_path: Path,
    randomized_metadata_path: Path,
    searchable_output_path: Path,
    summary_path: Path | None = None,
    *,
    random_seed: int = 42,
) -> None:
    rows_by_user: dict[int, list[dict[str, str]]] = defaultdict(list)
    fieldnames: list[str] | None = None
    source_rows: list[dict[str, str]] = []
    for row in iter_metadata_rows(input_path):
        clean_row = dict(row)
        clean_row.pop("searchable", None)
        if fieldnames is None:
            fieldnames = [name for name in clean_row.keys()]
        rows_by_user[int(clean_row["user_id"])].append(clean_row)
        source_rows.append(clean_row)
    if fieldnames is None:
        raise ValueError(f"No rows found in {input_path}")

    rng = np.random.default_rng(random_seed)
    shuffled_rows: list[dict[str, str]] = []
    for user_id in sorted(rows_by_user):
        user_rows = list(rows_by_user[user_id])
        if len(user_rows) > 1:
            user_rows = [user_rows[idx] for idx in rng.permutation(len(user_rows))]
        shuffled_rows.extend(user_rows)

    global_order = rng.permutation(len(shuffled_rows)).tolist()
    ordered_rows = [shuffled_rows[idx] for idx in global_order]
    synthetic_start = datetime(2000, 1, 1, 0, 0, 0)
    randomized_rows: list[dict[str, str]] = []
    for rank, row in enumerate(ordered_rows):
        randomized_row = dict(row)
        randomized_row["timestamp"] = (
            synthetic_start + timedelta(seconds=rank)
        ).strftime(TIMESTAMP_FORMAT)
        randomized_rows.append(randomized_row)

    seeded_mapping = _load_seeded_doc_mapping(seeded_peer_docs_path)
    seeded_doc_owners: dict[str, set[int]] = defaultdict(set)
    for peer_id, doc_ids in seeded_mapping.items():
        for doc_id in doc_ids:
            seeded_doc_owners[doc_id].add(peer_id)

    searchable_by_qid: dict[int, bool] = {}
    for row in source_rows:
        user_id = int(row["user_id"])
        qid = int(row["qid"])
        other_owners = seeded_doc_owners[row["target_doc_id"]] - {user_id}
        searchable_by_qid[qid] = bool(other_owners)

    _write_metadata_rows(randomized_metadata_path, fieldnames=fieldnames, rows=randomized_rows)
    searchable_rows = _write_searchable_rows(searchable_output_path, searchable_by_qid)

    print(
        f"Wrote {len(randomized_rows):,} randomized rows to {randomized_metadata_path} "
        f"and {len(searchable_by_qid):,} seeded searchable labels to {searchable_output_path}"
    )
    if summary_path is not None:
        summary = {
            "input_path": str(input_path),
            "seeded_peer_docs_path": str(seeded_peer_docs_path),
            "randomized_metadata_path": str(randomized_metadata_path),
            "searchable_output_path": str(searchable_output_path),
            "random_seed": random_seed,
            "total_rows": len(randomized_rows),
            "user_count": len(rows_by_user),
            "searchable_rows": searchable_rows,
            "searchable_fraction": searchable_rows / len(searchable_by_qid) if searchable_by_qid else 0.0,
        }
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Summary saved to {summary_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build randomized seeded metadata and seeded searchable labels"
    )
    parser.add_argument(
        "--input",
        default=str(Path("data") / "aol" / METADATA_FILENAME),
        help="Path to source metadata CSV(.gz)",
    )
    parser.add_argument(
        "--seeded-peer-docs",
        default=str(Path("data") / "aol" / "peer_future_doc_ids.json.gz"),
        help="Path to seeded peer-doc JSON(.gz)",
    )
    parser.add_argument(
        "--randomized-output",
        default=str(Path("data") / "aol" / SEEDED_RANDOMIZED_METADATA_FILENAME),
        help="Path for randomized seeded metadata CSV(.gz)",
    )
    parser.add_argument(
        "--searchable-output",
        default=str(Path("data") / "aol" / SEEDED_SEARCHABLE_QUERIES_FILENAME),
        help="Path for seeded searchable sidecar CSV(.gz)",
    )
    parser.add_argument(
        "--summary-json",
        default=str(Path("data") / "aol" / "metadata_seeded_randomized_summary.json"),
        help="Optional summary JSON path",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for per-user shuffling and synthetic timestamps",
    )
    args = parser.parse_args()
    build_seeded_randomized_artifacts(
        Path(args.input),
        Path(args.seeded_peer_docs),
        Path(args.randomized_output),
        Path(args.searchable_output),
        Path(args.summary_json),
        random_seed=args.random_seed,
    )


if __name__ == "__main__":
    main()

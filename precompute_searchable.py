#!/usr/bin/env python3
"""Compute chronological searchable labels in a compact sidecar artifact."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from src.dataset import (
    CHRONOLOGICAL_SEARCHABLE_QUERIES_FILENAME,
    METADATA_FILENAME,
    SEARCHABLE_QUERY_FIELDNAMES,
    TIMESTAMP_FORMAT,
    iter_metadata_rows,
)


def _open_output(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "wt", newline="", encoding="utf-8")
    return path.open("w", newline="", encoding="utf-8")


def _flush_day(
    day_rows: list[dict[str, str]],
    *,
    writer: csv.DictWriter,
    doc_owners: dict[str, set[int]],
) -> tuple[int, int]:
    for row in day_rows:
        doc_owners[row["target_doc_id"]].add(int(row["user_id"]))

    searchable_count = 0
    for row in day_rows:
        other_owners = doc_owners[row["target_doc_id"]] - {int(row["user_id"])}
        writer.writerow(
            {
                "query_id": int(row["qid"]),
                "searchable": "true" if other_owners else "false",
            }
        )
        searchable_count += int(bool(other_owners))
    return len(day_rows), searchable_count


def build_chronological_searchable_queries(
    input_path: Path,
    output_path: Path,
    summary_path: Path | None = None,
) -> None:
    doc_owners: dict[str, set[int]] = defaultdict(set)
    total_rows = 0
    searchable_rows = 0
    processed_days = 0
    current_day = None
    day_rows: list[dict[str, str]] = []
    last_timestamp = None

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with _open_output(output_path) as output_handle:
        writer = csv.DictWriter(output_handle, fieldnames=SEARCHABLE_QUERY_FIELDNAMES)
        writer.writeheader()

        for row in iter_metadata_rows(input_path):
            timestamp = datetime.strptime(row["timestamp"], TIMESTAMP_FORMAT)
            if last_timestamp is not None and timestamp < last_timestamp:
                raise ValueError(
                    f"Input metadata is not ordered by timestamp: {timestamp} < {last_timestamp}"
                )
            last_timestamp = timestamp
            row_day = timestamp.date()

            if current_day is None:
                current_day = row_day

            if row_day != current_day:
                written, searchable = _flush_day(day_rows, writer=writer, doc_owners=doc_owners)
                total_rows += written
                searchable_rows += searchable
                processed_days += 1
                day_rows = []
                current_day = row_day

            day_rows.append(row)

        if day_rows:
            written, searchable = _flush_day(day_rows, writer=writer, doc_owners=doc_owners)
            total_rows += written
            searchable_rows += searchable
            processed_days += 1

    print(
        f"Wrote {total_rows} query labels to {output_path} "
        f"({searchable_rows} searchable across {processed_days} days)"
    )
    if summary_path is not None:
        summary = {
            "input_path": str(input_path),
            "output_path": str(output_path),
            "total_rows": total_rows,
            "searchable_rows": searchable_rows,
            "processed_days": processed_days,
            "searchable_fraction": searchable_rows / total_rows if total_rows else 0.0,
        }
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Summary saved to {summary_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute chronological searchable labels for AOL metadata"
    )
    parser.add_argument(
        "--input",
        default=str(Path("data") / "aol" / METADATA_FILENAME),
        help="Path to metadata CSV(.gz)",
    )
    parser.add_argument(
        "--output",
        default=str(Path("data") / "aol" / CHRONOLOGICAL_SEARCHABLE_QUERIES_FILENAME),
        help="Path for chronological searchable sidecar CSV(.gz)",
    )
    parser.add_argument(
        "--summary-json",
        default=str(Path("data") / "aol" / "chronological_searchable_queries_summary.json"),
        help="Optional summary JSON path",
    )
    args = parser.parse_args()
    build_chronological_searchable_queries(
        Path(args.input),
        Path(args.output),
        Path(args.summary_json),
    )


if __name__ == "__main__":
    main()


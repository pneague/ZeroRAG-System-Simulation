#!/usr/bin/env python3
"""Compute each peer's eventual clicked-document set for seeded experiments."""

from __future__ import annotations

import argparse
import gzip
import json
from collections import defaultdict
from pathlib import Path

from src.dataset import METADATA_FILENAME, iter_metadata_rows


def _open_output(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "wt", encoding="utf-8")
    return path.open("w", encoding="utf-8")


def compute_peer_future_docs(input_path: Path, output_path: Path, summary_path: Path | None = None) -> None:
    doc_ids_by_user: dict[int, set[str]] = defaultdict(set)
    total_rows = 0

    for row in iter_metadata_rows(input_path):
        user_id = int(row["user_id"])
        doc_ids_by_user[user_id].add(str(row["target_doc_id"]))
        total_rows += 1

    payload = {
        str(user_id): sorted(doc_ids)
        for user_id, doc_ids in sorted(doc_ids_by_user.items())
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with _open_output(output_path) as handle:
        json.dump(payload, handle)

    total_unique_pairs = sum(len(doc_ids) for doc_ids in payload.values())
    print(
        f"Wrote future-doc mapping for {len(payload):,} peers to {output_path} "
        f"({total_unique_pairs:,} unique peer-doc pairs from {total_rows:,} rows)"
    )

    if summary_path is not None:
        summary = {
            "input_path": str(input_path),
            "output_path": str(output_path),
            "total_rows": total_rows,
            "peer_count": len(payload),
            "total_unique_peer_doc_pairs": total_unique_pairs,
            "avg_docs_per_peer": total_unique_pairs / len(payload) if payload else 0.0,
            "max_docs_per_peer": max((len(doc_ids) for doc_ids in payload.values()), default=0),
        }
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Summary saved to {summary_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute eventual clicked-doc sets per peer")
    parser.add_argument(
        "--input",
        default=str(Path("data") / "aol" / METADATA_FILENAME),
        help="Path to metadata CSV(.gz)",
    )
    parser.add_argument(
        "--output",
        default=str(Path("data") / "aol" / "peer_future_doc_ids.json.gz"),
        help="Path to JSON(.gz) mapping of user_id -> eventual clicked doc ids",
    )
    parser.add_argument(
        "--summary-json",
        default=str(Path("data") / "aol" / "peer_future_doc_ids_summary.json"),
        help="Optional summary JSON path",
    )
    args = parser.parse_args()
    compute_peer_future_docs(Path(args.input), Path(args.output), Path(args.summary_json))


if __name__ == "__main__":
    main()

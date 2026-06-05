#!/usr/bin/env python3
"""Aggregate Amazon multi-relevant retrieval by relevant-set-size bins."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.path_defaults import default_results_root  # noqa: E402


DEFAULT_METHODS: tuple[tuple[str, str], ...] = (
    ("ZeroRAG", "amazon_electronics_multi_relevant_lowcoord_query_memory_forming_run"),
    (
        "ZeroRAG no query memory",
        "amazon_electronics_multi_relevant_lowcoord_replica_only_forming_run",
    ),
    (
        "ZeroRAG no new connections",
        "amazon_electronics_multi_relevant_lowcoord_query_memory_static_run",
    ),
    (
        "ZeroRAG no query memory, no new connections",
        "amazon_electronics_multi_relevant_lowcoord_replica_only_static_run",
    ),
    ("LSH w=3.0 k=10 L=50", "amazon_electronics_multi_relevant_lsh_w3.0_k10_L50_run"),
    ("LSH w=3.0 k=10 L=20", "amazon_electronics_multi_relevant_lsh_w3.0_k10_L20_run"),
    ("LSH w=3.0 k=10 L=10", "amazon_electronics_multi_relevant_lsh_w3.0_k10_L10_run"),
    ("Semantica", "amazon_electronics_multi_relevant_semantica_run"),
)

BIN_LABELS = ("1-10", "11-50", "51-100", "100-150", "150-200")


def _bin_for(relevant_doc_count: int) -> str | None:
    if 1 <= relevant_doc_count <= 10:
        return "1-10"
    if 11 <= relevant_doc_count <= 50:
        return "11-50"
    if 51 <= relevant_doc_count <= 100:
        return "51-100"
    if 101 <= relevant_doc_count <= 150:
        return "100-150"
    if 151 <= relevant_doc_count <= 200:
        return "150-200"
    return None


def _slug(label: str) -> str:
    return (
        label.lower()
        .replace(" ", "_")
        .replace(",", "")
        .replace("=", "")
        .replace(".", "p")
        .replace("-", "_")
    )


def _new_stats() -> dict[str, float | int]:
    return {
        "event_count": 0,
        "available": 0,
        "retrieved": 0,
        "successes": 0,
        "query_retrieval_sum": 0.0,
    }


def _aggregate_query_log(path: Path) -> dict[str, dict[str, float | int]]:
    stats: dict[str, dict[str, float | int]] = defaultdict(_new_stats)
    saw_relevance_fields = False
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            if not raw_line.strip():
                continue
            event = json.loads(raw_line)
            if "relevant_doc_count" not in event:
                continue
            bucket = _bin_for(int(event.get("relevant_doc_count") or 0))
            if bucket is None:
                continue
            saw_relevance_fields = True
            bucket_stats = stats[bucket]
            bucket_stats["event_count"] = int(bucket_stats["event_count"]) + 1
            bucket_stats["available"] = int(bucket_stats["available"]) + int(
                event.get("available_relevant_doc_count") or 0
            )
            bucket_stats["retrieved"] = int(bucket_stats["retrieved"]) + int(
                event.get("retrieved_relevant_doc_count") or 0
            )
            bucket_stats["successes"] = int(bucket_stats["successes"]) + int(
                event.get("result") == "+"
            )
            bucket_stats["query_retrieval_sum"] = float(
                bucket_stats["query_retrieval_sum"]
            ) + float(event.get("query_retrieval_recall") or 0.0)
    if not saw_relevance_fields:
        raise ValueError(f"{path} has no multi-relevant fields in the requested bins")
    return {label: stats[label] for label in BIN_LABELS}


def _parse_method(spec: str) -> tuple[str, str]:
    if "::" not in spec:
        raise argparse.ArgumentTypeError("methods must be LABEL::RUN_DIR")
    label, run_dir = spec.split("::", 1)
    return label, run_dir


def build_rows(
    *,
    results_root: Path,
    methods: list[tuple[str, str]],
) -> tuple[list[dict[str, Any]], list[str]]:
    method_stats: dict[str, dict[str, dict[str, float | int]]] = {}
    missing: list[str] = []
    for label, run_dir_name in methods:
        run_dir = Path(run_dir_name)
        if not run_dir.is_absolute():
            run_dir = results_root / run_dir
        query_log = run_dir / "query_log.jsonl"
        if not query_log.exists():
            missing.append(str(query_log))
            continue
        method_stats[label] = _aggregate_query_log(query_log)

    if not method_stats:
        raise RuntimeError("No usable query logs found")

    first_label = next(iter(method_stats))
    base_stats = method_stats[first_label]
    total_included = sum(int(base_stats[label]["event_count"]) for label in BIN_LABELS)

    rows: list[dict[str, Any]] = []
    for bucket in BIN_LABELS:
        event_count = int(base_stats[bucket]["event_count"])
        available = int(base_stats[bucket]["available"])
        row: dict[str, Any] = {
            "relevant_docs_bin": bucket,
            "event_count": event_count,
            "event_share_pct": (event_count / total_included * 100.0)
            if total_included
            else 0.0,
            "avg_available_relevant_docs": (available / event_count)
            if event_count
            else 0.0,
        }
        for label, _ in methods:
            key = _slug(label)
            if label not in method_stats:
                row[f"{key}_retrieved_relevant_docs"] = ""
                row[f"{key}_retrieval_pct"] = ""
                row[f"{key}_success_pct"] = ""
                row[f"{key}_mean_query_retrieval_recall"] = ""
                continue
            stats = method_stats[label].get(bucket, _new_stats())
            n = int(stats["event_count"])
            method_available = int(stats["available"])
            retrieved = int(stats["retrieved"])
            row[f"{key}_retrieved_relevant_docs"] = retrieved
            row[f"{key}_retrieval_pct"] = (
                retrieved / method_available * 100.0 if method_available else ""
            )
            row[f"{key}_success_pct"] = int(stats["successes"]) / n * 100.0 if n else ""
            row[f"{key}_mean_query_retrieval_recall"] = (
                float(stats["query_retrieval_sum"]) / n if n else ""
            )
        rows.append(row)
    return rows, missing


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root",
        type=Path,
        default=default_results_root(),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--method",
        action="append",
        type=_parse_method,
        default=[],
        help="Optional LABEL::RUN_DIR override. May be repeated.",
    )
    args = parser.parse_args()

    methods = args.method or list(DEFAULT_METHODS)
    rows, missing = build_rows(results_root=args.results_root, methods=methods)

    header = list(rows[0].keys())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)

    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(
            {
                "csv_path": str(args.output),
                "results_root": str(args.results_root),
                "methods_order": [label for label, _ in methods],
                "bins": list(BIN_LABELS),
                "boundary_note": (
                    "Bins are non-overlapping: 100 is counted in 51-100; "
                    "150 is counted in 100-150."
                ),
                "missing_query_logs": missing,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    if missing:
        print(f"[amazon-bins] missing {len(missing)} query logs; see {summary_path}")
    print(f"[amazon-bins] wrote {args.output}")


if __name__ == "__main__":
    main()

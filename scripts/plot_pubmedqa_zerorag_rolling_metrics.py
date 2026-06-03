#!/usr/bin/env python3
"""Plot rolling ZeroRAG request count and search latency in 1000-query windows."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def _load_bin_stats(jsonl_path: Path, *, bin_size: int) -> list[dict[str, float | int]]:
    totals: dict[int, dict[str, float | int]] = defaultdict(
        lambda: {"count": 0, "cycles_sum": 0.0, "search_seconds_sum": 0.0, "timed_count": 0}
    )
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("skipped_reason"):
                continue
            event_index = int(row["event_index"])
            bin_id = event_index // bin_size
            bucket = totals[bin_id]
            bucket["count"] = int(bucket["count"]) + 1
            bucket["cycles_sum"] = float(bucket["cycles_sum"]) + float(row["cycles_used"])
            if row.get("search_seconds") is not None:
                bucket["search_seconds_sum"] = float(bucket["search_seconds_sum"]) + float(
                    row["search_seconds"]
                )
                bucket["timed_count"] = int(bucket["timed_count"]) + 1

    rows: list[dict[str, float | int]] = []
    for bin_id in sorted(totals):
        bucket = totals[bin_id]
        count = int(bucket["count"])
        timed_count = int(bucket["timed_count"])
        rows.append(
            {
                "bin_id": bin_id,
                "event_start": bin_id * bin_size,
                "event_end": (bin_id + 1) * bin_size - 1,
                "query_count": count,
                "avg_requests": float(bucket["cycles_sum"]) / count if count else 0.0,
                "avg_search_ms": (
                    1000.0 * float(bucket["search_seconds_sum"]) / timed_count
                    if timed_count
                    else 0.0
                ),
            }
        )
    return rows


def write_rolling_plot(
    *,
    jsonl_path: Path,
    output_prefix: Path,
    bin_size: int,
    title: str,
) -> dict[str, object]:
    rows = _load_bin_stats(jsonl_path, bin_size=bin_size)
    if not rows:
        raise ValueError(f"No executed rows found in {jsonl_path}")

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_path = output_prefix.with_suffix(".csv")
    png_path = output_prefix.with_suffix(".png")
    pdf_path = output_prefix.with_suffix(".pdf")

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "bin_id",
                "event_start",
                "event_end",
                "query_count",
                "avg_requests",
                "avg_search_ms",
            ),
        )
        writer.writeheader()
        writer.writerows(rows)

    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 11, "axes.titlesize": 13, "axes.labelsize": 11})
    x = [int(row["event_start"]) + bin_size / 2 for row in rows]
    requests = [float(row["avg_requests"]) for row in rows]
    search_ms = [float(row["avg_search_ms"]) for row in rows]

    fig, ax_left = plt.subplots(figsize=(12, 5), dpi=220)
    ax_right = ax_left.twinx()

    ax_left.plot(x, requests, color="#4C78A8", linewidth=1.8, label="Avg requests")
    ax_right.plot(x, search_ms, color="#F58518", linewidth=1.8, label="Avg search time (ms)")

    ax_left.set_title(title)
    ax_left.set_xlabel(f"Event index (center of each {bin_size:,}-query window)")
    ax_left.set_ylabel("Average requests per query")
    ax_right.set_ylabel("Average search time (ms)")
    ax_left.grid(axis="y", alpha=0.25)

    lines = ax_left.get_lines() + ax_right.get_lines()
    labels = [line.get_label() for line in lines]
    ax_left.legend(lines, labels, loc="upper right", framealpha=0.9)

    fig.tight_layout()
    fig.savefig(png_path)
    fig.savefig(pdf_path)
    plt.close(fig)

    summary = {
        "jsonl_path": str(jsonl_path),
        "bin_size": bin_size,
        "bin_count": len(rows),
        "csv_path": str(csv_path),
        "png_path": str(png_path),
        "pdf_path": str(pdf_path),
        "avg_requests_overall": sum(requests) / len(requests),
        "avg_search_ms_overall": sum(search_ms) / len(search_ms),
    }
    summary_path = output_prefix.with_name(output_prefix.name + "_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot ZeroRAG rolling average requests and search latency"
    )
    parser.add_argument(
        "--jsonl",
        type=Path,
        required=True,
        help="Path to pubmedqa_top10_low_coord.jsonl",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        required=True,
        help="Output path prefix without extension",
    )
    parser.add_argument("--bin-size", type=int, default=1000)
    parser.add_argument(
        "--title",
        default="ZeroRAG PubMedQA: rolling average requests and search time",
    )
    args = parser.parse_args()
    if args.bin_size <= 0:
        raise SystemExit("--bin-size must be positive")
    summary = write_rolling_plot(
        jsonl_path=args.jsonl,
        output_prefix=args.output_prefix,
        bin_size=args.bin_size,
        title=args.title,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

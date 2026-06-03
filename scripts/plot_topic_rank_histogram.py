#!/usr/bin/env python3
"""Build a per-peer topic-interest rank histogram from query_topics.csv(.gz)."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import TextIO


def _open_text(path: Path) -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(path, "rt", newline="", encoding="utf-8")
    return path.open("r", newline="", encoding="utf-8")


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def build_topic_rank_rows(query_topics_path: Path) -> tuple[list[dict[str, float | int]], dict]:
    peer_topics: dict[int, Counter[int]] = defaultdict(Counter)
    row_count = 0
    with _open_text(query_topics_path) as handle:
        reader = csv.DictReader(handle)
        if "user_id" not in (reader.fieldnames or ()) or "generated_topic" not in (reader.fieldnames or ()):
            raise ValueError(f"{query_topics_path} must contain user_id and generated_topic columns")
        for row in reader:
            peer_topics[int(row["user_id"])][int(row["generated_topic"])] += 1
            row_count += 1

    peer_count = len(peer_topics)
    max_rank = max((len(counter) for counter in peer_topics.values()), default=0)
    rank_values: list[list[int]] = [[] for _ in range(max_rank)]
    for counter in peer_topics.values():
        ranked_counts = sorted(counter.values(), reverse=True)
        for rank_index, count in enumerate(ranked_counts):
            rank_values[rank_index].append(int(count))

    rows: list[dict[str, float | int]] = []
    for rank_index, nonzero_values in enumerate(rank_values, start=1):
        total = sum(nonzero_values)
        zero_included_mean = total / peer_count if peer_count else 0.0
        zero_square_sum = sum((value - zero_included_mean) ** 2 for value in nonzero_values)
        zero_square_sum += (peer_count - len(nonzero_values)) * (zero_included_mean**2)
        zero_std = math.sqrt(zero_square_sum / peer_count) if peer_count else 0.0
        rows.append(
            {
                "topic_rank": rank_index,
                "avg_queries_per_peer_zero_included": zero_included_mean,
                "std_queries_per_peer_zero_included": zero_std,
                "lower_stddev_zero_included": min(zero_included_mean, zero_std),
                "upper_stddev_zero_included": zero_std,
                "avg_queries_among_peers_with_rank": (
                    total / len(nonzero_values) if nonzero_values else 0.0
                ),
                "peers_with_rank": len(nonzero_values),
                "peer_share_with_rank": len(nonzero_values) / peer_count if peer_count else 0.0,
            }
        )

    summary = {
        "query_topics_path": str(query_topics_path),
        "query_count": row_count,
        "peer_count": peer_count,
        "max_rank": max_rank,
        "avg_nonempty_topics_per_peer": (
            sum(len(counter) for counter in peer_topics.values()) / peer_count if peer_count else 0.0
        ),
    }
    return rows, summary


def write_topic_rank_histogram(
    *,
    query_topics_path: Path,
    output_prefix: Path,
    title: str,
) -> dict:
    rows, summary = build_topic_rank_rows(query_topics_path)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_path = output_prefix.with_suffix(".csv")
    json_path = output_prefix.with_name(output_prefix.name + "_summary.json")
    png_path = output_prefix.with_suffix(".png")
    pdf_path = output_prefix.with_suffix(".pdf")

    fieldnames = (
        "topic_rank",
        "avg_queries_per_peer_zero_included",
        "std_queries_per_peer_zero_included",
        "lower_stddev_zero_included",
        "upper_stddev_zero_included",
        "avg_queries_among_peers_with_rank",
        "peers_with_rank",
        "peer_share_with_rank",
    )
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        **summary,
        "csv_path": str(csv_path),
        "png_path": str(png_path),
        "pdf_path": str(pdf_path),
        "ranks": rows,
    }
    _write_json(json_path, summary)

    if rows:
        import matplotlib.pyplot as plt

        plt.rcParams.update({"font.size": 11, "axes.titlesize": 13, "axes.labelsize": 11})
        ranks = [int(row["topic_rank"]) for row in rows]
        means = [float(row["avg_queries_per_peer_zero_included"]) for row in rows]
        lower = [float(row["lower_stddev_zero_included"]) for row in rows]
        upper = [float(row["upper_stddev_zero_included"]) for row in rows]
        fig, ax = plt.subplots(figsize=(10, 5), dpi=220)
        ax.bar(
            ranks,
            means,
            width=0.75,
            color="#4C78A8",
            yerr=[lower, upper],
            ecolor="#222222",
            capsize=3,
            label="Mean",
        )
        ax.errorbar(
            [],
            [],
            yerr=[[1.0], [1.0]],
            fmt="none",
            ecolor="#222222",
            capsize=3,
            label="Lower / upper std deviation",
        )
        if title:
            ax.set_title(title)
        ax.set_xlabel("Topic rank within peer (1 = most sampled topic)")
        ax.set_ylabel("Average queries per peer")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(loc="upper right", framealpha=0.9)
        fig.tight_layout()
        fig.savefig(png_path)
        fig.savefig(pdf_path)
        plt.close(fig)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot topic-interest rank histogram")
    parser.add_argument("--query-topics", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument(
        "--title",
        default="",
        help="Optional plot title (omit for no title)",
    )
    args = parser.parse_args()
    summary = write_topic_rank_histogram(
        query_topics_path=args.query_topics,
        output_prefix=args.output_prefix,
        title=args.title,
    )
    print(json.dumps({key: summary[key] for key in ("csv_path", "png_path", "pdf_path")}, indent=2))


if __name__ == "__main__":
    main()


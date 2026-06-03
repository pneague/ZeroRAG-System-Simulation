#!/usr/bin/env python3
"""Histogram of distinct seeded documents per peer from peer_future_doc_ids.json.gz."""

from __future__ import annotations

import argparse
import gzip
import json
from collections import Counter
from pathlib import Path


def load_docs_per_peer(peer_future_docs_path: Path) -> list[int]:
    with gzip.open(peer_future_docs_path, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    return [len(doc_ids) for doc_ids in payload.values()]


def write_histogram(
    *,
    peer_future_docs_path: Path,
    output_prefix: Path,
    title: str,
    bins: int,
) -> dict:
    counts = load_docs_per_peer(peer_future_docs_path)
    if not counts:
        raise RuntimeError(f"no peers found in {peer_future_docs_path}")

    peer_count = len(counts)
    distribution = Counter(counts)
    summary = {
        "peer_future_docs_path": str(peer_future_docs_path),
        "peer_count": peer_count,
        "min_docs_per_peer": min(counts),
        "max_docs_per_peer": max(counts),
        "avg_docs_per_peer": sum(counts) / peer_count,
        "median_docs_per_peer": sorted(counts)[peer_count // 2],
        "total_peer_doc_pairs": sum(counts),
    }

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_path = output_prefix.with_suffix(".csv")
    json_path = output_prefix.with_name(output_prefix.name + "_summary.json")
    png_path = output_prefix.with_suffix(".png")
    pdf_path = output_prefix.with_suffix(".pdf")

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        handle.write("docs_per_peer,peer_count\n")
        for docs_per_peer in sorted(distribution):
            handle.write(f"{docs_per_peer},{distribution[docs_per_peer]}\n")

    summary.update(
        {
            "csv_path": str(csv_path),
            "json_path": str(json_path),
            "png_path": str(png_path),
            "pdf_path": str(pdf_path),
            "histogram_bins": bins,
        }
    )
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams.update({"font.size": 11, "axes.titlesize": 13, "axes.labelsize": 11})
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=220)
    ax.hist(
        counts,
        bins=bins,
        color="#4C78A8",
        edgecolor="white",
        linewidth=0.4,
    )
    if title:
        ax.set_title(title)
    ax.set_xlabel("Distinct documents owned per user (peer)")
    ax.set_ylabel("Number of users")
    ax.grid(axis="y", alpha=0.25)
    median = float(np.median(counts))
    mean = summary["avg_docs_per_peer"]
    ax.axvline(median, color="#E45756", linestyle="--", linewidth=1.5, label=f"Median: {median:.0f}")
    ax.axvline(mean, color="#F58518", linestyle=":", linewidth=1.5, label=f"Mean: {mean:.1f}")
    ax.legend(loc="upper right", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(png_path)
    fig.savefig(pdf_path)
    plt.close(fig)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot docs-per-peer histogram for Amazon peer_future_doc_ids")
    parser.add_argument("--peer-future-docs", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--bins", type=int, default=60)
    parser.add_argument(
        "--title",
        default="Amazon Electronics: distinct documents per user",
    )
    args = parser.parse_args()
    summary = write_histogram(
        peer_future_docs_path=args.peer_future_docs,
        output_prefix=args.output_prefix,
        title=args.title,
        bins=args.bins,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

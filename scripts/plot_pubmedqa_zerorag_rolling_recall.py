#!/usr/bin/env python3
"""Plot rolling recall@1 and recall@10 for ZeroRAG PubMedQA top-k jsonl output."""

from __future__ import annotations

import argparse
import csv
import json
from collections import deque
from pathlib import Path


def _hit_at_k(row: dict, k: int) -> int:
    if row.get("skipped_reason"):
        return 0
    top_doc_ids = row.get("top10_doc_ids") or []
    target_doc_id = row.get("target_doc_id")
    if not top_doc_ids or target_doc_id is None:
        return 0
    if k == 1:
        return int(str(top_doc_ids[0]) == str(target_doc_id))
    if k == 10:
        return int(bool(row.get("target_in_top10")))
    limit = min(k, len(top_doc_ids))
    return int(str(target_doc_id) in {str(doc_id) for doc_id in top_doc_ids[:limit]})


def _rolling_recall_points(jsonl_path: Path, *, window: int) -> list[dict[str, float | int]]:
    hit1 = deque()
    hit10 = deque()
    sum1 = 0
    sum10 = 0
    points: list[dict[str, float | int]] = []

    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            event_index = int(row["event_index"])
            val1 = _hit_at_k(row, 1)
            val10 = _hit_at_k(row, 10)
            hit1.append(val1)
            hit10.append(val10)
            sum1 += val1
            sum10 += val10
            if len(hit1) > window:
                sum1 -= hit1.popleft()
                sum10 -= hit10.popleft()
            if len(hit1) == window:
                points.append(
                    {
                        "event_index": event_index,
                        "window_start": event_index - window + 1,
                        "window_end": event_index,
                        "recall_at_1": sum1 / window,
                        "recall_at_10": sum10 / window,
                    }
                )
    return points


def write_rolling_recall_plot(
    *,
    jsonl_path: Path,
    output_prefix: Path,
    window: int,
    title: str,
) -> dict[str, object]:
    points = _rolling_recall_points(jsonl_path, window=window)
    if not points:
        raise ValueError(f"Need at least {window} rows in {jsonl_path}")

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_path = output_prefix.with_suffix(".csv")
    png_path = output_prefix.with_suffix(".png")
    pdf_path = output_prefix.with_suffix(".pdf")

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "event_index",
                "window_start",
                "window_end",
                "recall_at_1",
                "recall_at_10",
            ),
        )
        writer.writeheader()
        writer.writerows(points)

    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 11, "axes.titlesize": 13, "axes.labelsize": 11})
    x = [int(point["event_index"]) for point in points]
    recall1 = [float(point["recall_at_1"]) for point in points]
    recall10 = [float(point["recall_at_10"]) for point in points]

    fig, ax = plt.subplots(figsize=(12, 5), dpi=220)
    ax.plot(x, recall1, color="#4C78A8", linewidth=1.6, label="Recall@1")
    ax.plot(x, recall10, color="#F58518", linewidth=1.6, label="Recall@10")
    ax.set_title(title)
    ax.set_xlabel(f"Event index (right edge of {window:,}-query rolling window)")
    ax.set_ylabel("Rolling recall")
    ax.set_ylim(0.0, 1.0)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="lower right", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(png_path)
    fig.savefig(pdf_path)
    plt.close(fig)

    summary = {
        "jsonl_path": str(jsonl_path),
        "window": window,
        "point_count": len(points),
        "csv_path": str(csv_path),
        "png_path": str(png_path),
        "pdf_path": str(pdf_path),
        "final_recall_at_1": recall1[-1],
        "final_recall_at_10": recall10[-1],
        "mean_recall_at_1": sum(recall1) / len(recall1),
        "mean_recall_at_10": sum(recall10) / len(recall10),
    }
    summary_path = output_prefix.with_name(output_prefix.name + "_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot ZeroRAG rolling recall@1 and recall@10 from PubMedQA jsonl output"
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
    parser.add_argument("--window", type=int, default=1000)
    parser.add_argument(
        "--title",
        default="ZeroRAG PubMedQA: rolling recall@1 and recall@10",
    )
    args = parser.parse_args()
    if args.window <= 0:
        raise SystemExit("--window must be positive")
    summary = write_rolling_recall_plot(
        jsonl_path=args.jsonl,
        output_prefix=args.output_prefix,
        window=args.window,
        title=args.title,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

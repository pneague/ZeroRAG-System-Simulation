#!/usr/bin/env python3
"""Overlay rolling recall@10 for PubMedQA ZeroRAG ablation runs."""

from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path


def _rolling_r10(jsonl_path: Path, *, window_size: int, max_event_index: int | None) -> list[tuple[int, float]]:
    hits: deque[int] = deque()
    total = 0
    points: list[tuple[int, float]] = []
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            event_index = int(row["event_index"])
            if max_event_index is not None and event_index > max_event_index:
                break
            if row.get("skipped_reason"):
                continue
            value = 1 if row.get("target_in_top10") else 0
            hits.append(value)
            total += value
            while len(hits) > window_size:
                total -= hits.popleft()
            if len(hits) == window_size:
                points.append((event_index, total / window_size))
    return points


def main() -> None:
    parser = argparse.ArgumentParser(description="Overlay ZeroRAG ablation rolling recall@10")
    parser.add_argument("--run", action="append", required=True, help="label|pubmedqa_top10_zerorag.jsonl")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--window", type=int, default=1000)
    parser.add_argument("--total-events", type=int, default=1_500_000)
    parser.add_argument("--title", default=None)
    args = parser.parse_args()

    series: list[tuple[str, Path, list[tuple[int, float]]]] = []
    for item in args.run:
        label, path_str = item.split("|", 1)
        series.append((label, Path(path_str), _rolling_r10(Path(path_str), window_size=args.window, max_event_index=None)))

    common_max_event = min(points[-1][0] for _, _, points in series if points)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = ["#4C78A8", "#F58518", "#E45756", "#54A24B"]
    plt.rcParams.update({"font.size": 11, "axes.titlesize": 13, "axes.labelsize": 11})
    fig, ax = plt.subplots(figsize=(12, 5.5), dpi=220)

    summary_runs = []
    for index, (label, path, points) in enumerate(series):
        clipped = [(event_index, recall) for event_index, recall in points if event_index <= common_max_event]
        ax.plot(
            [event_index for event_index, _ in clipped],
            [recall for _, recall in clipped],
            color=colors[index % len(colors)],
            linewidth=1.5,
            label=label,
        )
        summary_runs.append(
            {
                "label": label,
                "jsonl_path": str(path),
                "point_count": len(clipped),
                "max_event_index": clipped[-1][0] if clipped else None,
                "final_recall_at_10": clipped[-1][1] if clipped else None,
            }
        )

    ax.set_title(
        args.title
        or f"ZeroRAG PubMedQA ablations: rolling recall@10 until event {common_max_event:,} (window={args.window:,})"
    )
    ax.set_xlabel(f"Event index (right edge of {args.window:,}-query rolling window)")
    ax.set_ylabel("Rolling recall@10")
    ax.set_xlim(0, args.total_events)
    ax.set_ylim(0.0, 1.02)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="lower right", framealpha=0.9)
    fig.tight_layout()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output)
    fig.savefig(args.output.with_suffix(".pdf"))
    plt.close(fig)

    summary = {
        "window_size": args.window,
        "total_events": args.total_events,
        "common_max_event_index": common_max_event,
        "runs": summary_runs,
        "output_png": str(args.output),
        "output_pdf": str(args.output.with_suffix(".pdf")),
    }
    summary_path = args.output.with_name(args.output.stem + "_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

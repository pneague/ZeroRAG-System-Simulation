#!/usr/bin/env python3
"""Overlay rolling recall for multiple concept-drift owner-count runs."""

from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path

from plot_pubmedqa_concept_drift_rolling_recall import RecallPoint, _rolling_recall_by_event_index


def _rolling_all_queries(query_log_path: Path, *, window_size: int, max_event_index: int | None) -> list[RecallPoint]:
    points = _rolling_recall_by_event_index(
        query_log_path,
        window_size=window_size,
        include_event=lambda _event: True,
    )
    if max_event_index is None:
        return points
    return [point for point in points if point.event_index <= max_event_index]


def main() -> None:
    parser = argparse.ArgumentParser(description="Overlay concept-drift owner-sweep rolling recall")
    parser.add_argument("--run", action="append", required=True, help="label|query_log.jsonl")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--window", type=int, default=1000)
    parser.add_argument("--total-events", type=int, default=1_500_000)
    parser.add_argument("--drift-start-fraction", type=float, default=0.4)
    parser.add_argument("--drift-end-fraction", type=float, default=1.0)
    parser.add_argument("--title", default=None)
    args = parser.parse_args()

    series: list[tuple[str, Path, list[RecallPoint]]] = []
    for item in args.run:
        label, path_str = item.split("|", 1)
        path = Path(path_str)
        series.append((label, path, _rolling_all_queries(path, window_size=args.window, max_event_index=None)))

    common_max_event = min(points[-1].event_index for _, _, points in series if points)
    drift_start = int(args.total_events * args.drift_start_fraction)
    drift_end = int(args.total_events * args.drift_end_fraction)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = ["#4C78A8", "#F58518", "#E45756", "#72B7B2", "#54A24B"]
    plt.rcParams.update({"font.size": 11, "axes.titlesize": 13, "axes.labelsize": 11})
    fig, ax = plt.subplots(figsize=(12, 5.5), dpi=220)

    summary_runs = []
    for index, (label, path, points) in enumerate(series):
        clipped = [point for point in points if point.event_index <= common_max_event]
        color = colors[index % len(colors)]
        ax.plot(
            [point.event_index for point in clipped],
            [point.recall for point in clipped],
            color=color,
            linewidth=1.5,
            label=label,
        )
        summary_runs.append(
            {
                "label": label,
                "query_log": str(path),
                "point_count": len(clipped),
                "max_event_index": clipped[-1].event_index if clipped else None,
                "final_recall": clipped[-1].recall if clipped else None,
            }
        )

    ax.axvline(drift_start, color="#E45756", linestyle=":", linewidth=1.5)
    if drift_end < args.total_events:
        ax.axvline(drift_end, color="#E45756", linestyle=":", linewidth=1.5)
    ax.text(
        drift_start,
        0.02,
        "concept appears (40%)",
        rotation=90,
        va="bottom",
        ha="right",
        color="#E45756",
        transform=ax.get_xaxis_transform(),
    )
    ax.set_title(
        args.title
        or f"Concept drift owner sweep — rolling recall until event {common_max_event:,} (window={args.window:,})"
    )
    ax.set_xlabel(f"Event index (right edge of {args.window:,}-event rolling window)")
    ax.set_ylabel("Rolling recall (distributed hit, result=+)")
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
        "drift_start_event_index": drift_start,
        "drift_end_event_index": drift_end,
        "runs": summary_runs,
        "output_png": str(args.output),
        "output_pdf": str(args.output.with_suffix(".pdf")),
    }
    summary_path = args.output.with_name(args.output.stem + "_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

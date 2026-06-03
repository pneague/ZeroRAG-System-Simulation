#!/usr/bin/env python3
"""Plot rolling recall curves from PubMedQA ablation metrics windows."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def _iter_windows(path: Path):
    if path.suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    yield from payload.get("windows") or []


def _binned_recall_points(
    metrics_path: Path,
    *,
    bin_size: int,
) -> list[dict[str, float | int]]:
    totals: dict[int, dict[str, float | int]] = {}
    for window in _iter_windows(metrics_path):
        event_end = int(window["event_end"])
        executed = int(window.get("executed_queries") or 0)
        if executed <= 0:
            continue
        recall = float(window["recall"])
        bin_id = event_end // bin_size
        bucket = totals.setdefault(bin_id, {"executed": 0, "successes": 0.0})
        bucket["executed"] = int(bucket["executed"]) + executed
        bucket["successes"] = float(bucket["successes"]) + recall * executed

    points: list[dict[str, float | int]] = []
    for bin_id in sorted(totals):
        bucket = totals[bin_id]
        executed = int(bucket["executed"])
        if executed <= 0:
            continue
        event_start = bin_id * bin_size
        event_end = (bin_id + 1) * bin_size - 1
        points.append(
            {
                "bin_id": bin_id,
                "event_start": event_start,
                "event_end": event_end,
                "event_center": event_start + bin_size // 2,
                "executed_queries": executed,
                "recall": float(bucket["successes"]) / executed,
            }
        )
    return points


def write_rolling_recall_plot(
    *,
    series: list[tuple[str, Path]],
    output_prefix: Path,
    bin_size: int,
    title: str,
) -> dict[str, object]:
    if not series:
        raise ValueError("Need at least one metrics path")

    all_points: dict[str, list[dict[str, float | int]]] = {}
    for label, metrics_path in series:
        points = _binned_recall_points(metrics_path, bin_size=bin_size)
        if not points:
            raise ValueError(f"No recall points produced from {metrics_path}")
        all_points[label] = points

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_path = output_prefix.with_suffix(".csv")
    png_path = output_prefix.with_suffix(".png")
    pdf_path = output_prefix.with_suffix(".pdf")

    fieldnames = (
        "label",
        "bin_id",
        "event_start",
        "event_end",
        "event_center",
        "executed_queries",
        "recall",
    )
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for label, points in all_points.items():
            for point in points:
                writer.writerow({"label": label, **point})

    import matplotlib.pyplot as plt

    colors = ("#4C78A8", "#F58518", "#54A24B", "#E45756", "#B279A2")
    plt.rcParams.update({"font.size": 11, "axes.titlesize": 13, "axes.labelsize": 11})
    fig, ax = plt.subplots(figsize=(12, 5), dpi=220)
    for index, (label, points) in enumerate(all_points.items()):
        x = [int(point["event_center"]) for point in points]
        recall = [float(point["recall"]) for point in points]
        ax.plot(
            x,
            recall,
            color=colors[index % len(colors)],
            linewidth=1.8,
            label=label,
        )
    ax.set_title(title)
    ax.set_xlabel(f"Event index (center of each {bin_size:,}-event bin)")
    ax.set_ylabel("Rolling recall")
    ax.set_ylim(0.0, 1.0)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="lower right", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(png_path)
    fig.savefig(pdf_path)
    plt.close(fig)

    summary = {
        "bin_size": bin_size,
        "csv_path": str(csv_path),
        "png_path": str(png_path),
        "pdf_path": str(pdf_path),
        "series": {
            label: {
                "metrics_path": str(metrics_path),
                "bin_count": len(points),
                "final_recall": float(points[-1]["recall"]),
                "mean_recall": sum(float(point["recall"]) for point in points) / len(points),
            }
            for (label, metrics_path), points in zip(series, all_points.values(), strict=True)
        },
    }
    summary_path = output_prefix.with_name(output_prefix.name + "_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metrics",
        action="append",
        nargs=2,
        metavar=("LABEL", "PATH"),
        required=True,
        help="Series label and metrics_windows.jsonl or metrics.json path",
    )
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--bin-size", type=int, default=25000)
    parser.add_argument(
        "--title",
        default="PubMedQA ablation: rolling recall",
    )
    args = parser.parse_args()
    if args.bin_size <= 0:
        raise SystemExit("--bin-size must be positive")
    series = [(label, Path(path)) for label, path in args.metrics]
    summary = write_rolling_recall_plot(
        series=series,
        output_prefix=args.output_prefix,
        bin_size=args.bin_size,
        title=args.title,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

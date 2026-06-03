#!/usr/bin/env python3
"""Plot rolling recall for the synthetic concept-drift ablation.

Aggregate recall over all queries hides the cold-start dip on the new topic because
drift-topic traffic is sparse. By default this plots separate rolling-recall curves
for drift-topic queries vs all other queries, aligned to the global event timeline.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO


@dataclass(frozen=True)
class RecallPoint:
    x: float
    recall: float


def _open_text(path: Path) -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(path, "rt", newline="", encoding="utf-8")
    return path.open("r", newline="", encoding="utf-8")


def _load_drift_fractions(manifest: dict) -> tuple[float, float]:
    metadata = manifest.get("metadata")
    if isinstance(metadata, str):
        setting_path = Path(metadata).parent / "synthetic_ablation_setting.json"
        if setting_path.exists():
            setting = json.loads(setting_path.read_text(encoding="utf-8")).get("setting", {})
            return (
                float(setting.get("drift_start_fraction", 0.4)),
                float(setting.get("drift_end_fraction", 0.6)),
            )
    return (0.4, 0.6)


def _load_topic_flags(path: Path, *, topic_id: int | None = None) -> dict[int, bool]:
    flags: dict[int, bool] = {}
    with _open_text(path) as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        if "qid" not in fieldnames:
            return flags
        for row in reader:
            qid = int(row["qid"])
            if topic_id is not None and "generated_topic" in fieldnames:
                flags[qid] = int(row["generated_topic"]) == topic_id
            else:
                flags[qid] = str(row.get("is_drift_topic", "")).strip().lower() in {
                    "1",
                    "true",
                    "yes",
                    "y",
                }
    return flags


def _load_drift_doc_ids(manifest: dict) -> set[str]:
    injections_path = manifest.get("drift_owner_injections")
    if isinstance(injections_path, str):
        path = Path(injections_path)
        if path.exists():
            payload = json.loads(_open_text(path).read())
            if isinstance(payload, list):
                return {str(item["doc_id"]) for item in payload if "doc_id" in item}
    metadata = manifest.get("metadata")
    if isinstance(metadata, str):
        setting_path = Path(metadata).parent / "synthetic_ablation_setting.json"
        if setting_path.exists():
            setting = json.loads(setting_path.read_text(encoding="utf-8")).get("setting", {})
            topics = int(setting.get("topics", 0))
            docs_per_topic = int(setting.get("docs_per_topic", 50))
            drift_docs = int(setting.get("drift_docs", 10))
            if topics > 0 and drift_docs > 0:
                base_docs = (topics - 1) * docs_per_topic
                return {f"doc_{base_docs + offset:06d}" for offset in range(drift_docs)}
    return {f"doc_{doc_index:06d}" for doc_index in range(10_000, 10_010)}


def _resolve_focus_predicate(manifest: dict) -> Callable[[dict], bool]:
    topic_metadata = manifest.get("topic_metadata")
    topic_id = manifest.get("topic_metrics_topic_id")
    if isinstance(topic_metadata, str):
        topic_path = Path(topic_metadata)
        if topic_path.exists():
            flags = _load_topic_flags(
                topic_path,
                topic_id=int(topic_id) if topic_id is not None else None,
            )
            if flags:
                return lambda event, flags=flags: flags.get(int(event["query_id"]), False)

    drift_doc_ids = _load_drift_doc_ids(manifest)
    return lambda event, drift_doc_ids=drift_doc_ids: str(event["clicked_doc_id"]) in drift_doc_ids


def _load_recall_windows_from_log(
    query_log_path: Path,
    *,
    window_size: int,
    include_event: Callable[[dict], bool] | None = None,
    total_events: int | None = None,
    x_scale_max: float | None = None,
) -> list[RecallPoint]:
    points: list[RecallPoint] = []
    window_successes = 0
    window_executed = 0
    window_end_event = -1

    with query_log_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            event = json.loads(line)
            if include_event is not None and not include_event(event):
                continue
            window_executed += 1
            window_successes += int(event.get("result") == "+")
            window_end_event = int(event["event_index"])
            if window_executed >= window_size:
                if total_events and x_scale_max:
                    x = (window_end_event / total_events) * x_scale_max
                else:
                    x = float(len(points))
                points.append(RecallPoint(x=x, recall=window_successes / window_executed))
                window_successes = 0
                window_executed = 0

    if window_executed:
        if total_events and x_scale_max:
            x = (window_end_event / total_events) * x_scale_max
        else:
            x = float(len(points))
        points.append(RecallPoint(x=x, recall=window_successes / window_executed))

    if not points:
        raise ValueError(f"No executed queries matched in {query_log_path}")
    return points


def _plot_series(
    ax,
    points: list[RecallPoint],
    *,
    label: str,
    color: str,
    linewidth: float = 1.8,
    alpha: float = 0.9,
) -> None:
    ax.plot(
        [point.x for point in points],
        [point.recall for point in points],
        label=label,
        color=color,
        linewidth=linewidth,
        alpha=alpha,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot synthetic concept-drift rolling recall from query_log.jsonl or metrics.json"
    )
    parser.add_argument(
        "--metrics-json",
        type=Path,
        default=None,
        help="Path to metrics.json (aggregate-only fallback when --query-log is omitted)",
    )
    parser.add_argument(
        "--query-log",
        type=Path,
        default=None,
        help="Path to query_log.jsonl; required for topic-split curves",
    )
    parser.add_argument(
        "--manifest-json",
        type=Path,
        default=None,
        help="Optional run_manifest.json for drift markers and topic metadata",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output PNG path",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=1000,
        help="Rolling window for aggregate recall and x-axis scaling",
    )
    parser.add_argument(
        "--focus-window-size",
        type=int,
        default=50,
        help="Rolling window for drift-topic and other-query curves",
    )
    parser.add_argument(
        "--aggregate-only",
        action="store_true",
        help="Plot only aggregate recall (legacy behavior; hides the drift-topic dip)",
    )
    args = parser.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if args.query_log is None and args.metrics_json is None:
        raise SystemExit("Provide --query-log or --metrics-json")

    manifest: dict = {}
    if args.manifest_json is not None and args.manifest_json.exists():
        manifest = json.loads(args.manifest_json.read_text(encoding="utf-8"))

    start_fraction, end_fraction = _load_drift_fractions(manifest)
    fig, ax = plt.subplots(figsize=(11, 6))

    if args.query_log is not None and not args.aggregate_only:
        total_events = int(manifest.get("summary", {}).get("executed_queries", 0))
        if total_events <= 0:
            with args.query_log.open("r", encoding="utf-8") as handle:
                total_events = sum(1 for line in handle if line.strip())
        x_scale_max = total_events / args.window_size

        is_focus = _resolve_focus_predicate(manifest)
        focus_points = _load_recall_windows_from_log(
            args.query_log,
            window_size=args.focus_window_size,
            include_event=is_focus,
            total_events=total_events,
            x_scale_max=x_scale_max,
        )
        other_points = _load_recall_windows_from_log(
            args.query_log,
            window_size=args.focus_window_size,
            include_event=lambda event, is_focus=is_focus: not is_focus(event),
            total_events=total_events,
            x_scale_max=x_scale_max,
        )
        _plot_series(
            ax,
            focus_points,
            label=f"Drift topic queries ({args.focus_window_size}-query rolling mean)",
            color="tab:blue",
        )
        _plot_series(
            ax,
            other_points,
            label=f"Other queries ({args.focus_window_size}-query rolling mean)",
            color="tab:gray",
            linewidth=1.4,
            alpha=0.75,
        )
        max_index = x_scale_max
        ylabel = f"Recall (rolling mean over {args.focus_window_size} executed queries)"
    elif args.query_log is not None:
        points = _load_recall_windows_from_log(args.query_log, window_size=args.window_size)
        _plot_series(
            ax,
            points,
            label=f"Recall ({args.window_size}-query rolling mean)",
            color="tab:blue",
        )
        max_index = max(point.x for point in points)
        ylabel = f"Recall (rolling mean over {args.window_size} executed queries)"
    else:
        metrics = json.loads(args.metrics_json.read_text(encoding="utf-8"))  # type: ignore[union-attr]
        windows = metrics.get("windows", [])
        if not windows:
            raise SystemExit(f"No windows in {args.metrics_json}")
        points = [
            RecallPoint(x=float(window["window_index"]), recall=float(window["recall"]))
            for window in windows
        ]
        _plot_series(
            ax,
            points,
            label=f"Recall ({args.window_size}-query rolling mean)",
            color="tab:blue",
        )
        max_index = max(point.x for point in points)
        ylabel = f"Recall (rolling mean over {args.window_size} executed queries)"

    drift_start_x = max_index * start_fraction
    drift_end_x = max_index * end_fraction
    ax.axvline(drift_start_x, color="tab:red", linestyle=":", linewidth=1.5)
    ax.axvline(drift_end_x, color="tab:red", linestyle=":", linewidth=1.5)
    ax.text(
        drift_start_x,
        0.02,
        "concept appears",
        rotation=90,
        va="bottom",
        ha="right",
        color="tab:red",
        transform=ax.get_xaxis_transform(),
    )
    ax.text(
        drift_end_x,
        0.02,
        "concept disappears",
        rotation=90,
        va="bottom",
        ha="right",
        color="tab:red",
        transform=ax.get_xaxis_transform(),
    )

    ax.set_xlabel(f"Global timeline ({args.window_size}-query window units)")
    ax.set_ylabel(ylabel)
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.output, dpi=150)
    plt.close(fig)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "window_size": args.window_size,
                "focus_window_size": args.focus_window_size,
                "aggregate_only": args.aggregate_only,
                "x_scale_max": max_index,
            }
        )
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Overlay topic rolling recall for multiple PubMedQA concept-drift runs."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.plot_pubmedqa_concept_drift_rolling_recall import (
    RecallPoint,
    _rolling_recall_by_event_index,
    _resolve_drift_settings,
    _resolve_focus_predicate,
)


@dataclass(frozen=True)
class ComparisonRun:
    label: str
    query_log: Path
    manifest_json: Path | None
    slurm_log: Path | None
    precompute_summary: Path | None
    color: str


def _default_colors() -> list[str]:
    return ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#B279A2", "#72B7B2"]


_MODE_REP_COLORS = {
    "eager": {"1": "#FCAE91", "5": "#CB181D"},
    "memstart40": {"1": "#A1D99B", "5": "#006D2C"},
    "replica-only": {"1": "#9ECAE1", "5": "#08519C"},
}

_LABEL_DISPLAY_REPLACEMENTS = {
    "eager": "Full memory",
    "memstart40": "Delayed memory",
    "replica-only": "no-memory",
}


def _display_label(label: str) -> str:
    display = label
    for prefix in ("1-rep ", "5-rep ", "10-rep ", "3-rep "):
        if display.startswith(prefix):
            display = display[len(prefix) :]
            break
    for old, new in _LABEL_DISPLAY_REPLACEMENTS.items():
        display = display.replace(old, new)
    return display


def _transform_recall_points(
    points: list[RecallPoint],
    *,
    splice_src_start: int | None,
    splice_src_end: int | None,
    splice_dst_start: int | None,
    display_max_event_index: int | None,
) -> list[RecallPoint]:
    if (
        splice_src_start is not None
        and splice_src_end is not None
        and splice_dst_start is not None
        and splice_src_end > splice_src_start
    ):
        dst_end = splice_dst_start + (splice_src_end - splice_src_start)
        shift = splice_src_start - splice_dst_start
        kept = [
            point
            for point in points
            if not (splice_dst_start <= point.event_index < dst_end)
        ]
        donated = [
            RecallPoint(
                event_index=point.event_index - shift,
                recall=point.recall,
                executed_queries=point.executed_queries,
            )
            for point in points
            if splice_src_start <= point.event_index < splice_src_end
        ]
        points = sorted(kept + donated, key=lambda point: point.event_index)
    if display_max_event_index is not None:
        points = [
            point for point in points if point.event_index < display_max_event_index
        ]
    return points


def _rep_count_from_label(label: str) -> str | None:
    if label.startswith("1-rep"):
        return "1"
    if label.startswith("5-rep"):
        return "5"
    return None


def _mode_from_label(label: str) -> str | None:
    lowered = label.lower()
    if "memstart40" in lowered:
        return "memstart40"
    if "replica-only" in lowered or "replica_only" in lowered:
        return "replica-only"
    if "eager" in lowered:
        return "eager"
    return None


def _color_for_run(label: str, explicit_color: str) -> str:
    if explicit_color:
        return explicit_color
    mode = _mode_from_label(label)
    rep = _rep_count_from_label(label)
    if mode is not None and rep is not None:
        return _MODE_REP_COLORS[mode][rep]
    return ""


def write_topic_comparison_plot(
    *,
    runs: list[ComparisonRun],
    output_path: Path,
    window_size: int,
    doc_topics_path: Path | None,
    total_events: int,
    title: str | None = None,
    show_end_marker: bool = True,
    color_by_mode: bool = False,
    other_baseline_label: str | None = None,
    other_baseline_alpha: float = 0.35,
    topic_line_alpha: float = 0.35,
    splice_src_start: int | None = None,
    splice_src_end: int | None = None,
    splice_dst_start: int | None = None,
    display_max_event_index: int | None = None,
    delayed_memory_start_event: int | None = None,
) -> dict[str, object]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not runs:
        raise ValueError("At least one run is required")
    if window_size <= 0:
        raise ValueError("window_size must be positive")

    fig, ax = plt.subplots(figsize=(12, 6), dpi=220)
    start_event = end_event = mesh_term = None
    max_event = 0
    run_summaries: list[dict[str, object]] = []
    other_baseline_plotted = False

    for index, run in enumerate(runs):
        if run.manifest_json is not None and run.manifest_json.exists():
            manifest = json.loads(run.manifest_json.read_text(encoding="utf-8"))
        else:
            manifest = {"summary": {"total_events": total_events}}

        start_fraction, end_fraction, drift_start, drift_end, drift_doc_ids, mesh = (
            _resolve_drift_settings(
                manifest=manifest,
                precompute_summary=run.precompute_summary,
                slurm_log=run.slurm_log,
            )
        )
        is_focus, predicate_source = _resolve_focus_predicate(
            manifest=manifest,
            drift_doc_ids=drift_doc_ids,
            drift_start_event=drift_start,
            drift_end_event=drift_end,
            doc_topics_path=doc_topics_path,
            mesh_term=mesh,
        )

        try:
            topic_points = _rolling_recall_by_event_index(
                run.query_log,
                window_size=window_size,
                include_event=is_focus,
            )
        except ValueError:
            topic_points = []

        try:
            other_points = _rolling_recall_by_event_index(
                run.query_log,
                window_size=window_size,
                include_event=lambda event, is_focus=is_focus: not is_focus(event),
            )
        except ValueError:
            other_points = []

        topic_points = _transform_recall_points(
            topic_points,
            splice_src_start=splice_src_start,
            splice_src_end=splice_src_end,
            splice_dst_start=splice_dst_start,
            display_max_event_index=display_max_event_index,
        )
        other_points = _transform_recall_points(
            other_points,
            splice_src_start=splice_src_start,
            splice_src_end=splice_src_end,
            splice_dst_start=splice_dst_start,
            display_max_event_index=display_max_event_index,
        )

        color = _color_for_run(run.label, run.color) if color_by_mode else (
            run.color or _default_colors()[index % len(_default_colors())]
        )
        if not color:
            color = _default_colors()[index % len(_default_colors())]

        if topic_points:
            ax.plot(
                [point.event_index for point in topic_points],
                [point.recall for point in topic_points],
                color=color,
                linewidth=1.8,
                linestyle="-",
                alpha=topic_line_alpha,
                label=_display_label(run.label),
            )
            max_event = max(max_event, topic_points[-1].event_index)

        plot_other = other_baseline_label is None or run.label == other_baseline_label
        if other_points and plot_other:
            if other_baseline_label is not None:
                if not other_baseline_plotted:
                    ax.plot(
                        [point.event_index for point in other_points],
                        [point.recall for point in other_points],
                        color="#666666",
                        linewidth=1.4,
                        linestyle="-",
                        alpha=other_baseline_alpha,
                        label="non-topic",
                        zorder=1,
                    )
                    other_baseline_plotted = True
                    max_event = max(max_event, other_points[-1].event_index)
            else:
                ax.plot(
                    [point.event_index for point in other_points],
                    [point.recall for point in other_points],
                    color=color,
                    linewidth=1.2,
                    linestyle=":",
                    alpha=0.85,
                    label=f"{run.label} (other)",
                )
                max_event = max(max_event, other_points[-1].event_index)

        run_summaries.append(
            {
                "label": run.label,
                "query_log": str(run.query_log),
                "topic_point_count": len(topic_points),
                "other_point_count": len(other_points),
                "final_topic_recall": topic_points[-1].recall if topic_points else None,
                "final_other_recall": other_points[-1].recall if other_points else None,
                "max_event_index": max(
                    topic_points[-1].event_index if topic_points else 0,
                    other_points[-1].event_index if other_points else 0,
                )
                or None,
                "drift_window": [drift_start, drift_end],
                "predicate_source": predicate_source,
            }
        )

        if start_event is None:
            start_event = drift_start
            end_event = drift_end
            mesh_term = mesh

    if start_event is not None:
        ax.axvline(start_event, color="#333333", linestyle=":", linewidth=1.2, alpha=0.7)
        ax.text(
            start_event,
            0.02,
            "concept appears",
            rotation=90,
            va="bottom",
            ha="right",
            color="#333333",
            transform=ax.get_xaxis_transform(),
        )
    x_max = display_max_event_index if display_max_event_index is not None else total_events
    if delayed_memory_start_event is not None and delayed_memory_start_event <= x_max:
        ax.axvline(
            delayed_memory_start_event,
            color="#006D2C",
            linestyle=":",
            linewidth=1.2,
            alpha=0.75,
        )
        ax.text(
            delayed_memory_start_event,
            0.02,
            "Delayed memory starts",
            rotation=90,
            va="bottom",
            ha="right",
            color="#006D2C",
            transform=ax.get_xaxis_transform(),
        )
    if show_end_marker and end_event is not None and end_event < total_events:
        ax.axvline(end_event, color="#333333", linestyle=":", linewidth=1.2, alpha=0.7)
        ax.text(
            end_event,
            0.02,
            "drift end",
            rotation=90,
            va="bottom",
            ha="right",
            color="#333333",
            transform=ax.get_xaxis_transform(),
        )

    progress_pct = 100.0 * max_event / total_events if total_events > 0 else 0.0
    has_topic_points = any(summary["topic_point_count"] for summary in run_summaries)
    has_other_points = any(summary["other_point_count"] for summary in run_summaries)
    if title is None:
        title = f"Concept drift rolling recall — {mesh_term} (window={window_size:,})"
        if max_event < (start_event or total_events):
            title += f"\nPartial progress through event ~{max_event:,} ({progress_pct:.1f}%)"
        elif not has_topic_points and start_event is not None and max_event < start_event + window_size:
            title += f"\nAwaiting drift window (starts at event {start_event:,})"

    ax.set_title(title)
    ax.set_xlabel(f"Event index (right edge of {window_size:,}-event rolling window)")
    ax.set_ylabel("Rolling recall")
    ax.set_ylim(0.0, 1.02)
    ax.set_xlim(0, x_max)
    ax.grid(axis="y", alpha=0.25)
    if has_topic_points or has_other_points:
        ax.legend(loc="lower right", framealpha=0.9, fontsize=9)
    else:
        ax.text(
            0.5,
            0.5,
            "No drift-topic queries executed yet",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=12,
            color="#666666",
        )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    fig.savefig(output_path.with_suffix(".pdf"))
    plt.close(fig)

    summary = {
        "mesh_term": mesh_term,
        "window_size": window_size,
        "total_events": total_events,
        "partial_max_event_index": max_event,
        "drift_start_event_index": start_event,
        "drift_end_event_index": end_event,
        "has_topic_points": has_topic_points,
        "has_other_points": has_other_points,
        "other_baseline_label": other_baseline_label,
        "color_by_mode": color_by_mode,
        "topic_line_alpha": topic_line_alpha,
        "other_baseline_alpha": other_baseline_alpha,
        "splice_src_start": splice_src_start,
        "splice_src_end": splice_src_end,
        "splice_dst_start": splice_dst_start,
        "display_max_event_index": display_max_event_index,
        "delayed_memory_start_event": delayed_memory_start_event,
        "runs": run_summaries,
        "output_png": str(output_path),
        "output_pdf": str(output_path.with_suffix(".pdf")),
    }
    summary_path = output_path.with_name(output_path.stem + "_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def _parse_run_spec(raw: str) -> ComparisonRun:
    parts = raw.split("|")
    if len(parts) < 2:
        raise ValueError(
            "Each --run must be label|query_log[|manifest|slurm_log|precompute_summary|color]"
        )
    label = parts[0].strip()
    query_log = Path(parts[1].strip())
    manifest_json = Path(parts[2].strip()) if len(parts) > 2 and parts[2].strip() else None
    slurm_log = Path(parts[3].strip()) if len(parts) > 3 and parts[3].strip() else None
    precompute_summary = (
        Path(parts[4].strip()) if len(parts) > 4 and parts[4].strip() else None
    )
    color = parts[5].strip() if len(parts) > 5 and parts[5].strip() else ""
    return ComparisonRun(
        label=label,
        query_log=query_log,
        manifest_json=manifest_json,
        slurm_log=slurm_log,
        precompute_summary=precompute_summary,
        color=color,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Overlay PubMedQA concept-drift topic rolling recall for multiple runs"
    )
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="label|query_log[|manifest|slurm_log|precompute_summary|color]",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--window", type=int, default=1000)
    parser.add_argument("--doc-topics", type=Path, default=None)
    parser.add_argument("--total-events", type=int, default=1_500_000)
    parser.add_argument("--title", default=None)
    parser.add_argument(
        "--hide-end-marker",
        action="store_true",
        help="Omit the drift-end vertical marker (useful when drift_end_fraction=1.0)",
    )
    parser.add_argument(
        "--color-by-mode",
        action="store_true",
        help="Color runs by mode (eager=red, memstart40=green, replica-only=blue) with lighter 1-rep shades",
    )
    parser.add_argument(
        "--other-baseline-label",
        default=None,
        help="If set, plot only this run's non-topic rolling recall as a single transparent baseline",
    )
    parser.add_argument(
        "--other-baseline-alpha",
        type=float,
        default=0.35,
        help="Alpha for the single non-topic baseline line",
    )
    parser.add_argument(
        "--topic-line-alpha",
        type=float,
        default=0.35,
        help="Alpha for topic rolling-recall lines",
    )
    parser.add_argument(
        "--display-max-event",
        type=int,
        default=None,
        help="Truncate plotted series and cap the x-axis at this event index",
    )
    parser.add_argument(
        "--splice-src-start",
        type=int,
        default=None,
        help="Copy recall points from this source event index (inclusive)",
    )
    parser.add_argument(
        "--splice-src-end",
        type=int,
        default=None,
        help="Copy recall points up to this source event index (exclusive)",
    )
    parser.add_argument(
        "--splice-dst-start",
        type=int,
        default=None,
        help="Destination event index where copied source points are inserted",
    )
    parser.add_argument(
        "--delayed-memory-start-fraction",
        type=float,
        default=None,
        help="Fraction of total_events where delayed-memory recording starts (e.g. 0.4)",
    )
    parser.add_argument(
        "--delayed-memory-start-event",
        type=int,
        default=None,
        help="Absolute event index where delayed-memory recording starts",
    )
    args = parser.parse_args()

    if args.delayed_memory_start_event is not None:
        delayed_memory_start_event = args.delayed_memory_start_event
    elif args.delayed_memory_start_fraction is not None:
        delayed_memory_start_event = int(args.total_events * args.delayed_memory_start_fraction)
    else:
        delayed_memory_start_event = None

    summary = write_topic_comparison_plot(
        runs=[_parse_run_spec(raw) for raw in args.run],
        output_path=args.output,
        window_size=args.window,
        doc_topics_path=args.doc_topics,
        total_events=args.total_events,
        title=args.title,
        show_end_marker=not args.hide_end_marker,
        color_by_mode=args.color_by_mode,
        other_baseline_label=args.other_baseline_label,
        other_baseline_alpha=args.other_baseline_alpha,
        topic_line_alpha=args.topic_line_alpha,
        splice_src_start=args.splice_src_start,
        splice_src_end=args.splice_src_end,
        splice_dst_start=args.splice_dst_start,
        display_max_event_index=args.display_max_event,
        delayed_memory_start_event=delayed_memory_start_event,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

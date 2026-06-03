#!/usr/bin/env python3
"""Compare Amazon methods using rolling relevance metrics from query_log.jsonl."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from plot_recall_comparison_suite import (
    RunRecord,
    _default_label,
    _find_primary_record,
    _has_query_log,
    discover_runs,
)


@dataclass(frozen=True, slots=True)
class RollingPoint:
    window_index: int
    event_start: int
    event_end: int
    executed_queries: int
    available_relevant_doc_count: int
    retrieved_relevant_doc_count: int
    relevant_retrieval_pct: float
    mean_retrieved_relevant_doc_count: float


def load_rolling_points(
    query_log_path: Path,
    *,
    window_size: int,
) -> list[RollingPoint]:
    if window_size <= 0:
        raise ValueError("window_size must be positive")
    points: list[RollingPoint] = []
    executed = 0
    available_sum = 0
    retrieved_sum = 0
    window_start_event: int | None = None
    window_end_event = -1

    def flush() -> None:
        nonlocal executed, available_sum, retrieved_sum
        nonlocal window_start_event, window_end_event
        if executed == 0:
            return
        relevant_retrieval_pct = (
            (retrieved_sum / available_sum) * 100.0 if available_sum else 0.0
        )
        points.append(
            RollingPoint(
                window_index=len(points),
                event_start=window_start_event if window_start_event is not None else -1,
                event_end=window_end_event,
                executed_queries=executed,
                available_relevant_doc_count=available_sum,
                retrieved_relevant_doc_count=retrieved_sum,
                relevant_retrieval_pct=relevant_retrieval_pct,
                mean_retrieved_relevant_doc_count=retrieved_sum / executed,
            )
        )
        executed = 0
        available_sum = 0
        retrieved_sum = 0
        window_start_event = None

    with query_log_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            event = json.loads(line)
            event_index = int(event["event_index"])
            if window_start_event is None:
                window_start_event = event_index
            window_end_event = event_index
            executed += 1
            if (
                "available_relevant_doc_count" not in event
                or "retrieved_relevant_doc_count" not in event
            ):
                raise ValueError(
                    f"{query_log_path} does not contain relevance-count fields; "
                    "use a relevance-aware run"
                )
            available_sum += int(event["available_relevant_doc_count"])
            retrieved_sum += int(event["retrieved_relevant_doc_count"])

            if executed >= window_size:
                flush()

    flush()
    if not points:
        raise ValueError(f"No executed queries found in {query_log_path}")
    return points


def _record_notes(record: RunRecord) -> list[str]:
    notes: list[str] = []
    if record.method == "random_walk" and record.routing_mode != "semantic_query_memory":
        notes.append(
            f"Random baseline uses routing_mode={record.routing_mode} (only matching run on disk)."
        )
    return notes


def _amazon_category_ok(record: RunRecord, category: str | None) -> bool:
    if category is None:
        return True
    cat = record.manifest.get("amazon_category")
    return isinstance(cat, str) and cat == category


def _filtered_records(
    records: list[RunRecord],
    *,
    category: str | None,
    include_smoke: bool,
) -> list[RunRecord]:
    out: list[RunRecord] = []
    for record in records:
        if record.dataset != "amazon":
            continue
        if not include_smoke and "smoke" in record.run_dir.name.lower():
            continue
        if not _amazon_category_ok(record, category):
            continue
        out.append(record)
    return out


def _explicit_runs(specs: list[str]) -> tuple[list[tuple[str, Path]], list[str]]:
    runs: list[tuple[str, Path]] = []
    errors: list[str] = []
    for spec in specs:
        parts = spec.split("::", 2)
        if len(parts) != 3:
            errors.append(f"invalid explicit run {spec!r}; expected method::label::run_dir")
            continue
        method, label, run_dir = parts
        if method not in {"low_coord", "random_walk", "lsh", "semantica"}:
            errors.append(f"invalid method {method!r} in explicit run {spec!r}")
            continue
        path = Path(run_dir)
        if not (path / "query_log.jsonl").exists():
            errors.append(f"{label}: missing query_log.jsonl in {path}")
            continue
        runs.append((label, path))
    return runs, errors


def _series_from_summary(path: Path) -> tuple[list[tuple[str, list[RollingPoint]]], dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    series: list[tuple[str, list[RollingPoint]]] = []
    for item in payload.get("series", []):
        label = str(item["label"])
        points = [
            RollingPoint(
                window_index=int(point["window_index"]),
                event_start=int(point["event_start"]),
                event_end=int(point["event_end"]),
                executed_queries=int(point["executed_queries"]),
                available_relevant_doc_count=int(point["available_relevant_doc_count"]),
                retrieved_relevant_doc_count=int(point["retrieved_relevant_doc_count"]),
                relevant_retrieval_pct=float(point["relevant_retrieval_pct"]),
                mean_retrieved_relevant_doc_count=float(
                    point["avg_retrieved_relevant_doc_count"]
                ),
            )
            for point in item.get("points", [])
        ]
        if points:
            series.append((label, points))
    return series, payload


def _plot_series(
    series: list[tuple[str, list[RollingPoint]]],
    *,
    output_path: Path,
    title: str,
    y_label: str,
    value_fn: Callable[[RollingPoint], float],
    y_lim: tuple[float, float] | None = None,
    notes: list[str] | None = None,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 6))
    for label, points in series:
        ax.plot(
            [point.window_index for point in points],
            [value_fn(point) for point in points],
            label=label,
            linewidth=1.8,
            alpha=0.55,
        )
    ax.set_title(title)
    ax.set_xlabel("Window index (rolling over executed queries)")
    ax.set_ylabel(y_label)
    if y_lim is not None:
        ax.set_ylim(*y_lim)
    ax.grid(True, alpha=0.3)
    ax.legend()
    if notes:
        fig.text(0.01, 0.01, " | ".join(notes), fontsize=8)
    fig.tight_layout(rect=(0, 0.04 if notes else 0, 1, 1))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("results"),
        help="Directory tree containing run directories with run_manifest.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results") / "amazon_method_comparison_plots",
        help="Where to write PNG files",
    )
    parser.add_argument("--window-size", type=int, default=100)
    parser.add_argument(
        "--scenario",
        default="seeded_randomized_timeline",
        help="Scenario name as recorded in run_manifest scenario/requested_scenario",
    )
    parser.add_argument(
        "--amazon-category",
        default=None,
        help="If set, keep only runs whose manifest amazon_category matches (e.g. electronics)",
    )
    parser.add_argument(
        "--include-smoke",
        action="store_true",
        help="Include *smoke* run directories",
    )
    parser.add_argument(
        "--run",
        action="append",
        default=[],
        metavar="METHOD::LABEL::RUN_DIR",
        help=(
            "Explicit partial run to plot, bypassing manifest discovery. "
            "METHOD is one of low_coord, random_walk, lsh, semantica. "
            "Can be repeated."
        ),
    )
    parser.add_argument(
        "--rolling-summary-json",
        type=Path,
        default=None,
        help="Precomputed rolling relevance summary JSON from partial remote logs.",
    )
    args = parser.parse_args()

    scenario = args.scenario
    series_relevance_pct: list[tuple[str, list[RollingPoint]]] = []
    series_relevant_count: list[tuple[str, list[RollingPoint]]] = []
    footnotes: list[str] = []
    missing: list[str] = []

    if args.rolling_summary_json is not None:
        summary_series, summary_payload = _series_from_summary(args.rolling_summary_json)
        series_relevance_pct.extend(summary_series)
        series_relevant_count.extend(summary_series)
        missing.extend(str(item) for item in summary_payload.get("missing", []))
        scenario = str(summary_payload.get("scenario") or scenario)
        if not args.amazon_category:
            category = summary_payload.get("amazon_category")
            if isinstance(category, str):
                args.amazon_category = category
    else:
        explicit_runs, explicit_errors = _explicit_runs(args.run)
        missing.extend(explicit_errors)

        if explicit_runs:
            selected_runs = explicit_runs
        else:
            all_records = discover_runs(args.results_root, include_smoke=args.include_smoke)
            records = _filtered_records(
                all_records, category=args.amazon_category, include_smoke=args.include_smoke
            )
            selected_runs = []
            for method in ["low_coord", "random_walk", "lsh", "semantica"]:
                record = _find_primary_record(
                    records, dataset="amazon", scenario=scenario, method=method
                )
                if record is None:
                    missing.append(f"{method}")
                    continue
                if not _has_query_log(record):
                    missing.append(f"{method} (no query_log.jsonl)")
                    continue
                selected_runs.append((_default_label(record), record.run_dir))
                footnotes.extend(_record_notes(record))

        for label, run_dir in selected_runs:
            try:
                points = load_rolling_points(
                    run_dir / "query_log.jsonl", window_size=args.window_size
                )
            except (OSError, json.JSONDecodeError, ValueError) as exc:
                missing.append(f"{label}: {exc}")
                continue
            series_relevance_pct.append((label, points))
            series_relevant_count.append((label, points))

    if footnotes:
        footnotes = list(dict.fromkeys(footnotes))

    cat_tag = f"_{args.amazon_category}" if args.amazon_category else ""
    base = f"amazon{cat_tag}_{scenario}_w{args.window_size}"

    plot_notes = list(footnotes)
    if missing:
        plot_notes.append(f"missing: {', '.join(missing)}")

    if series_relevance_pct:
        _plot_series(
            series_relevance_pct,
            output_path=args.output_dir / f"{base}_relevant_docs_retrieved_pct.png",
            title=f"Amazon{(' / ' + args.amazon_category) if args.amazon_category else ''}: "
            f"{scenario} — relevant documents retrieved (rolling {args.window_size})",
            y_label="Relevant documents retrieved (%)",
            value_fn=lambda point: point.relevant_retrieval_pct,
            y_lim=(0.0, 100.0),
            notes=plot_notes or None,
        )
        _plot_series(
            series_relevant_count,
            output_path=args.output_dir / f"{base}_avg_retrieved_relevant_docs.png",
            title=f"Amazon{(' / ' + args.amazon_category) if args.amazon_category else ''}: "
            f"{scenario} — mean retrieved relevant docs (rolling {args.window_size})",
            y_label="Mean retrieved relevant documents per query",
            value_fn=lambda point: point.mean_retrieved_relevant_doc_count,
            notes=plot_notes or None,
        )
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "plotted_methods": [label for label, _ in series_relevance_pct],
                "missing": missing,
                "footnotes": footnotes,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

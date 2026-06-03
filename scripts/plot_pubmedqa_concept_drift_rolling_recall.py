#!/usr/bin/env python3
"""Plot PubMedQA concept-drift rolling recall split by drift topic vs other queries."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, TextIO


@dataclass(frozen=True)
class RecallPoint:
    event_index: int
    recall: float
    executed_queries: int


def _open_text(path: Path) -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(path, "rt", newline="", encoding="utf-8")
    return path.open("r", newline="", encoding="utf-8")


def _load_topic_flags(path: Path) -> dict[int, bool]:
    flags: dict[int, bool] = {}
    with _open_text(path) as handle:
        reader = csv.DictReader(handle)
        if "qid" not in (reader.fieldnames or []):
            return flags
        for row in reader:
            qid = int(row["qid"])
            flags[qid] = str(row.get("is_drift_topic", "")).strip().lower() in {
                "1",
                "true",
                "yes",
                "y",
            }
    return flags


def _parse_concept_drift_from_log(log_path: Path) -> dict:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    marker = '"concept_drift":'
    index = text.rfind(marker)
    if index < 0:
        raise ValueError(f"No concept_drift block found in {log_path}")
    brace_start = text.find("{", index + len(marker))
    depth = 0
    for offset, char in enumerate(text[brace_start:], start=brace_start):
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[brace_start : offset + 1])
    raise ValueError(f"Could not parse concept_drift JSON from {log_path}")


def _load_mesh_doc_ids(doc_topics_path: Path, *, mesh_term: str) -> set[str]:
    doc_ids: set[str] = set()
    with _open_text(doc_topics_path) as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if str(row.get("mesh_term", "")).strip() == mesh_term:
                doc_ids.add(str(row["doc_id"]))
    if not doc_ids:
        raise ValueError(f"No docs found for mesh term {mesh_term!r} in {doc_topics_path}")
    return doc_ids


def _resolve_drift_settings(
    *,
    manifest: dict,
    precompute_summary: Path | None,
    slurm_log: Path | None,
) -> tuple[float, float, int, int, set[str], str]:
    concept_drift: dict | None = None
    if precompute_summary is not None and precompute_summary.exists():
        payload = json.loads(precompute_summary.read_text(encoding="utf-8"))
        concept_drift = payload.get("concept_drift")
    if concept_drift is None and slurm_log is not None and slurm_log.exists():
        concept_drift = _parse_concept_drift_from_log(slurm_log)

    start_fraction = float(concept_drift.get("drift_start_fraction", 0.4)) if concept_drift else 0.4
    end_fraction = float(concept_drift.get("drift_end_fraction", 0.6)) if concept_drift else 0.6
    total_events = int(manifest.get("summary", {}).get("total_events", 1_500_000))
    if total_events <= 0:
        total_events = 1_500_000

    if concept_drift and "drift_start_event_index" in concept_drift:
        start_event = int(concept_drift["drift_start_event_index"])
        end_event = int(concept_drift["drift_end_event_index"])
    else:
        start_event = int(total_events * start_fraction)
        end_event = int(total_events * end_fraction)

    drift_doc_ids: set[str] = set()
    if concept_drift:
        for key in ("heldout_doc_ids", "injected_doc_ids", "drift_doc_ids"):
            values = concept_drift.get(key)
            if isinstance(values, list):
                drift_doc_ids.update(str(value) for value in values)
        injections = concept_drift.get("drift_owner_injections")
        if isinstance(injections, list):
            drift_doc_ids.update(str(item["doc_id"]) for item in injections if "doc_id" in item)

    mesh_term = str(concept_drift.get("mesh_term", "drift topic")) if concept_drift else "drift topic"
    return start_fraction, end_fraction, start_event, end_event, drift_doc_ids, mesh_term


def _resolve_focus_predicate(
    *,
    manifest: dict,
    drift_doc_ids: set[str],
    drift_start_event: int,
    drift_end_event: int,
    doc_topics_path: Path | None,
    mesh_term: str,
) -> tuple[Callable[[dict], bool], str]:
    topic_metadata = manifest.get("topic_metadata")
    if isinstance(topic_metadata, str):
        topic_path = Path(topic_metadata)
        if topic_path.exists():
            flags = _load_topic_flags(topic_path)
            if flags:
                return (
                    lambda event, flags=flags: flags.get(int(event["query_id"]), False),
                    "topic_metadata.is_drift_topic",
                )

    if doc_topics_path is not None and doc_topics_path.exists():
        mesh_doc_ids = _load_mesh_doc_ids(doc_topics_path, mesh_term=mesh_term)
        return (
            lambda event, mesh_doc_ids=mesh_doc_ids, start=drift_start_event, end=drift_end_event: (
                str(event["clicked_doc_id"]) in mesh_doc_ids
                and start <= int(event["event_index"]) < end
            ),
            f"doc_topics mesh={mesh_term!r} within events {drift_start_event}-{drift_end_event}",
        )

    if drift_doc_ids:
        return (
            lambda event, drift_doc_ids=drift_doc_ids, start=drift_start_event, end=drift_end_event: (
                str(event["clicked_doc_id"]) in drift_doc_ids
                and start <= int(event["event_index"]) < end
            ),
            "concept_drift held-out doc ids within drift event window",
        )

    raise ValueError(
        "Could not resolve drift-topic predicate; provide --topic-metadata, "
        "--doc-topics, --precompute-summary, or --slurm-log with held-out doc ids"
    )


def _rolling_recall_by_event_index(
    query_log_path: Path,
    *,
    window_size: int,
    include_event: Callable[[dict], bool],
) -> list[RecallPoint]:
    events: list[tuple[int, int, bool]] = []
    with query_log_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            record = json.loads(line)
            is_match = bool(include_event(record))
            events.append(
                (
                    int(record["event_index"]),
                    int(is_match and record.get("result") == "+"),
                    is_match,
                )
            )

    if not events:
        raise ValueError(f"No executed queries found in {query_log_path}")

    points: list[RecallPoint] = []
    left = 0
    matched_successes = 0
    matched_executed = 0

    for right, (event_index, success, is_match) in enumerate(events):
        if is_match:
            matched_executed += 1
            matched_successes += success

        window_start = event_index - window_size + 1
        while left < right and events[left][0] < window_start:
            left_event_index, left_success, left_match = events[left]
            if left_match:
                matched_executed -= 1
                matched_successes -= left_success
            left += 1

        if event_index + 1 >= window_size and matched_executed > 0:
            points.append(
                RecallPoint(
                    event_index=event_index,
                    recall=matched_successes / matched_executed,
                    executed_queries=matched_executed,
                )
            )

    if not points:
        raise ValueError(f"No rolling recall points produced from {query_log_path}")
    return points


def write_concept_drift_plot(
    *,
    query_log_path: Path,
    manifest_path: Path | None,
    output_path: Path,
    window_size: int,
    precompute_summary: Path | None = None,
    slurm_log: Path | None = None,
    doc_topics_path: Path | None = None,
    title: str | None = None,
    total_events: int | None = None,
) -> dict[str, object]:
    if manifest_path is not None and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        manifest = {"summary": {"total_events": total_events or 1_500_000}}
    start_fraction, end_fraction, start_event, end_event, drift_doc_ids, mesh_term = _resolve_drift_settings(
        manifest=manifest,
        precompute_summary=precompute_summary,
        slurm_log=slurm_log,
    )
    is_focus, predicate_source = _resolve_focus_predicate(
        manifest=manifest,
        drift_doc_ids=drift_doc_ids,
        drift_start_event=start_event,
        drift_end_event=end_event,
        doc_topics_path=doc_topics_path,
        mesh_term=mesh_term,
    )

    focus_points = _rolling_recall_by_event_index(
        query_log_path,
        window_size=window_size,
        include_event=is_focus,
    )
    other_points = _rolling_recall_by_event_index(
        query_log_path,
        window_size=window_size,
        include_event=lambda event, is_focus=is_focus: not is_focus(event),
    )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 11, "axes.titlesize": 13, "axes.labelsize": 11})
    fig, ax = plt.subplots(figsize=(12, 5.5), dpi=220)

    ax.plot(
        [point.event_index for point in focus_points],
        [point.recall for point in focus_points],
        color="#4C78A8",
        linewidth=1.6,
        label=f"Drift topic ({mesh_term})",
    )
    ax.plot(
        [point.event_index for point in other_points],
        [point.recall for point in other_points],
        color="#9D9D9D",
        linewidth=1.4,
        alpha=0.85,
        label="Other queries",
    )

    ax.axvline(start_event, color="#E45756", linestyle=":", linewidth=1.5)
    ax.axvline(end_event, color="#E45756", linestyle=":", linewidth=1.5)
    ax.text(
        start_event,
        0.02,
        "concept appears",
        rotation=90,
        va="bottom",
        ha="right",
        color="#E45756",
        transform=ax.get_xaxis_transform(),
    )
    ax.text(
        end_event,
        0.02,
        "concept disappears",
        rotation=90,
        va="bottom",
        ha="right",
        color="#E45756",
        transform=ax.get_xaxis_transform(),
    )

    ax.set_title(
        title
        or f"PubMedQA concept drift: {window_size:,}-event rolling recall ({mesh_term})"
    )
    ax.set_xlabel(f"Event index (right edge of {window_size:,}-event rolling window)")
    ax.set_ylabel("Rolling recall")
    ax.set_ylim(0.0, 1.02)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="lower right", framealpha=0.9)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    pdf_path = output_path.with_suffix(".pdf")
    fig.savefig(pdf_path)
    plt.close(fig)

    summary = {
        "query_log_path": str(query_log_path),
        "manifest_path": str(manifest_path),
        "output_png": str(output_path),
        "output_pdf": str(pdf_path),
        "window_size": window_size,
        "mesh_term": mesh_term,
        "drift_start_event_index": start_event,
        "drift_end_event_index": end_event,
        "drift_start_fraction": start_fraction,
        "drift_end_fraction": end_fraction,
        "predicate_source": predicate_source,
        "focus_point_count": len(focus_points),
        "other_point_count": len(other_points),
        "focus_final_recall": focus_points[-1].recall,
        "other_final_recall": other_points[-1].recall,
    }
    summary_path = output_path.with_name(output_path.stem + "_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot PubMedQA concept-drift rolling recall with drift-topic split"
    )
    parser.add_argument("--query-log", type=Path, required=True)
    parser.add_argument("--manifest-json", type=Path, default=None)
    parser.add_argument(
        "--total-events",
        type=int,
        default=1_500_000,
        help="Used when --manifest-json is missing (in-progress runs)",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--window", type=int, default=1000)
    parser.add_argument("--precompute-summary", type=Path, default=None)
    parser.add_argument("--slurm-log", type=Path, default=None)
    parser.add_argument(
        "--doc-topics",
        type=Path,
        default=None,
        help="doc_topics.csv.gz used to resolve held-out MeSH doc ids when topic metadata was cleaned up",
    )
    parser.add_argument("--title", default=None)
    args = parser.parse_args()
    if args.window <= 0:
        raise SystemExit("--window must be positive")
    summary = write_concept_drift_plot(
        query_log_path=args.query_log,
        manifest_path=args.manifest_json,
        output_path=args.output,
        window_size=args.window,
        precompute_summary=args.precompute_summary,
        slurm_log=args.slurm_log,
        doc_topics_path=args.doc_topics,
        title=args.title,
        total_events=args.total_events,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

from __future__ import annotations

import csv
import gzip
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

from src.dataset import QueryEvent, parse_bool
from src.metrics import MetricsTracker, QueryLog


def _open_text(path: Path) -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(path, "rt", newline="", encoding="utf-8")
    return path.open("r", newline="", encoding="utf-8")


def load_topic_flags(
    path: str | Path,
    *,
    topic_id: int | None = None,
    flag_column: str = "is_drift_topic",
) -> dict[int, bool]:
    """Load qid-level membership flags from synthetic ``query_topics.csv.gz``."""

    flags: dict[int, bool] = {}
    metadata_path = Path(path)
    with _open_text(metadata_path) as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        if "qid" not in fieldnames:
            raise ValueError(f"{metadata_path} must contain a qid column")
        if topic_id is None and flag_column not in fieldnames:
            raise ValueError(
                f"{metadata_path} must contain {flag_column!r}, or pass a topic_id"
            )
        if topic_id is not None and "generated_topic" not in fieldnames:
            raise ValueError(f"{metadata_path} must contain generated_topic for topic_id matching")
        for row in reader:
            qid = int(row["qid"])
            if topic_id is not None:
                flags[qid] = int(row["generated_topic"]) == topic_id
            else:
                flags[qid] = parse_bool(row.get(flag_column, ""))
    return flags


@dataclass(slots=True)
class _Bucket:
    total_events: int = 0
    executed_queries: int = 0
    skipped_queries: int = 0
    successes: int = 0
    cycles: int = 0

    def note_executed(self, query_log: QueryLog) -> None:
        self.total_events += 1
        self.executed_queries += 1
        self.successes += int(query_log.result == "+")
        self.cycles += query_log.cycles_used

    def note_skipped(self) -> None:
        self.total_events += 1
        self.skipped_queries += 1

    def as_dict(self) -> dict[str, float | int]:
        return {
            "total_events": self.total_events,
            "executed_queries": self.executed_queries,
            "skipped_queries": self.skipped_queries,
            "successes": self.successes,
            "recall": self.successes / self.executed_queries if self.executed_queries else 0.0,
            "avg_cycles": self.cycles / self.executed_queries if self.executed_queries else 0.0,
            "skip_rate": self.skipped_queries / self.total_events if self.total_events else 0.0,
        }


class TopicMetricsTracker:
    def __init__(self, *, output_dir: str | Path, topic_flags: dict[int, bool]) -> None:
        self.output_path = Path(output_dir) / "topic_metrics.json"
        self.topic_flags = topic_flags
        self.overall = _Bucket()
        self.focus = _Bucket()
        self.non_focus = _Bucket()

    def note_after_event(
        self,
        event: QueryEvent,
        *,
        metrics: MetricsTracker,
        previous_executed: int,
        previous_skipped: int,
    ) -> None:
        is_focus = self.topic_flags.get(event.qid, False)
        executed = metrics.executed_queries > previous_executed
        skipped = metrics.skipped_queries > previous_skipped
        if not executed and not skipped:
            return
        if executed:
            query_log = metrics.last_query_log
            if query_log is None or query_log.query_id != event.qid:
                return
            self.overall.note_executed(query_log)
            (self.focus if is_focus else self.non_focus).note_executed(query_log)
            return
        self.overall.note_skipped()
        (self.focus if is_focus else self.non_focus).note_skipped()

    def finalize(self) -> dict[str, object]:
        payload = {
            "output_path": str(self.output_path),
            "overall": self.overall.as_dict(),
            "focus_topic": self.focus.as_dict(),
            "non_focus_topic": self.non_focus.as_dict(),
        }
        self.output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return payload

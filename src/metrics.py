from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TextIO


@dataclass(slots=True)
class QueryLog:
    event_index: int
    query_id: int
    querier_peer_id: int
    clicked_doc_id: str
    cycles_used: int
    result: str
    termination_reason: str
    doc_vector_comparisons: int = 0
    local_doc_vector_comparisons: int = 0
    neighbor_doc_vector_comparisons: int = 0
    routed_query_memory_comparisons: int = 0
    total_vector_comparisons: int = 0
    relevant_doc_count: int = 0
    available_relevant_doc_count: int = 0
    retrieved_relevant_doc_count: int = 0
    query_retrieval_recall: float = 0.0
    requests_until_first_relevant: int | None = None


class MetricsTracker:
    QUERY_LOG_MODES = {"full", "window", "none"}

    def __init__(
        self,
        *,
        output_dir: str | Path,
        window_size: int,
        query_log_mode: str = "full",
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.window_size = window_size
        if query_log_mode not in self.QUERY_LOG_MODES:
            raise ValueError(
                f"query_log_mode must be one of {sorted(self.QUERY_LOG_MODES)}, "
                f"got {query_log_mode!r}"
            )
        self.query_log_mode = query_log_mode
        self.query_log_path = self.output_dir / "query_log.jsonl"
        self.window_log_path = self.output_dir / "metrics_windows.jsonl"
        self.metrics_json_path = self.output_dir / "metrics.json"

        self.executed_queries = 0
        self.skipped_queries = 0
        self.total_cycles = 0
        self.total_successes = 0
        self.total_doc_vector_comparisons = 0
        self.total_local_doc_vector_comparisons = 0
        self.total_neighbor_doc_vector_comparisons = 0
        self.total_routed_query_memory_comparisons = 0
        self.total_vector_comparisons = 0
        self.total_available_relevant_docs = 0
        self.total_retrieved_relevant_docs = 0
        self.total_requests_until_first_relevant = 0
        self.first_relevant_found_count = 0
        self.skipped_by_reason: dict[str, int] = {}

        self.window_executed = 0
        self.window_skipped = 0
        self.window_cycles = 0
        self.window_successes = 0
        self.window_doc_vector_comparisons = 0
        self.window_local_doc_vector_comparisons = 0
        self.window_neighbor_doc_vector_comparisons = 0
        self.window_routed_query_memory_comparisons = 0
        self.window_total_vector_comparisons = 0
        self.window_available_relevant_docs = 0
        self.window_retrieved_relevant_docs = 0
        self.window_requests_until_first_relevant = 0
        self.window_first_relevant_found_count = 0
        self.window_skipped_by_reason: dict[str, int] = {}
        self.windows: list[dict[str, float | int]] = []
        self.last_query_log: QueryLog | None = None
        self.window_start_event_index: int | None = None
        self.window_end_event_index: int | None = None

        self._log_handle: TextIO | None = (
            self.query_log_path.open("w", encoding="utf-8")
            if self.query_log_mode == "full"
            else None
        )
        self._window_log_handle: TextIO | None = (
            self.window_log_path.open("w", encoding="utf-8")
            if self.query_log_mode == "window"
            else None
        )

    def log_query(self, query_log: QueryLog) -> None:
        self.last_query_log = query_log
        self.executed_queries += 1
        self.total_cycles += query_log.cycles_used
        self.total_successes += int(query_log.result == "+")
        self.total_doc_vector_comparisons += query_log.doc_vector_comparisons
        self.total_local_doc_vector_comparisons += query_log.local_doc_vector_comparisons
        self.total_neighbor_doc_vector_comparisons += query_log.neighbor_doc_vector_comparisons
        self.total_routed_query_memory_comparisons += (
            query_log.routed_query_memory_comparisons
        )
        self.total_vector_comparisons += query_log.total_vector_comparisons
        self.total_available_relevant_docs += query_log.available_relevant_doc_count
        self.total_retrieved_relevant_docs += query_log.retrieved_relevant_doc_count
        if query_log.requests_until_first_relevant is not None:
            self.total_requests_until_first_relevant += query_log.requests_until_first_relevant
            self.first_relevant_found_count += 1

        self.window_executed += 1
        if self.window_start_event_index is None:
            self.window_start_event_index = query_log.event_index
        self.window_end_event_index = query_log.event_index
        self.window_cycles += query_log.cycles_used
        self.window_successes += int(query_log.result == "+")
        self.window_doc_vector_comparisons += query_log.doc_vector_comparisons
        self.window_local_doc_vector_comparisons += query_log.local_doc_vector_comparisons
        self.window_neighbor_doc_vector_comparisons += (
            query_log.neighbor_doc_vector_comparisons
        )
        self.window_routed_query_memory_comparisons += (
            query_log.routed_query_memory_comparisons
        )
        self.window_total_vector_comparisons += query_log.total_vector_comparisons
        self.window_available_relevant_docs += query_log.available_relevant_doc_count
        self.window_retrieved_relevant_docs += query_log.retrieved_relevant_doc_count
        if query_log.requests_until_first_relevant is not None:
            self.window_requests_until_first_relevant += query_log.requests_until_first_relevant
            self.window_first_relevant_found_count += 1

        if self._log_handle is not None:
            self._log_handle.write(json.dumps(asdict(query_log)) + "\n")

        if self.window_executed >= self.window_size:
            self._flush_window()

    def note_skipped_query(self, reason: str) -> None:
        self.skipped_queries += 1
        self.window_skipped += 1
        self.skipped_by_reason[reason] = self.skipped_by_reason.get(reason, 0) + 1
        self.window_skipped_by_reason[reason] = self.window_skipped_by_reason.get(reason, 0) + 1

    def _flush_window(self) -> None:
        if self.window_executed == 0 and self.window_skipped == 0:
            return
        recall = self.window_successes / self.window_executed if self.window_executed else 0.0
        avg_cycles = self.window_cycles / self.window_executed if self.window_executed else 0.0
        avg_doc_vector_comparisons = (
            self.window_doc_vector_comparisons / self.window_executed
            if self.window_executed
            else 0.0
        )
        avg_local_doc_vector_comparisons = (
            self.window_local_doc_vector_comparisons / self.window_executed
            if self.window_executed
            else 0.0
        )
        avg_neighbor_doc_vector_comparisons = (
            self.window_neighbor_doc_vector_comparisons / self.window_executed
            if self.window_executed
            else 0.0
        )
        avg_routed_query_memory_comparisons = (
            self.window_routed_query_memory_comparisons / self.window_executed
            if self.window_executed
            else 0.0
        )
        avg_total_vector_comparisons = (
            self.window_total_vector_comparisons / self.window_executed
            if self.window_executed
            else 0.0
        )
        skip_total = self.window_executed + self.window_skipped
        skip_rate = self.window_skipped / skip_total if skip_total else 0.0
        retrieval_recall = (
            self.window_retrieved_relevant_docs / self.window_available_relevant_docs
            if self.window_available_relevant_docs
            else 0.0
        )
        avg_retrieved_relevant_doc_count = (
            self.window_retrieved_relevant_docs / self.window_executed
            if self.window_executed
            else 0.0
        )
        avg_requests_until_first_relevant = (
            self.window_requests_until_first_relevant / self.window_first_relevant_found_count
            if self.window_first_relevant_found_count
            else 0.0
        )
        window = {
            "window_index": len(self.windows),
            "event_start": self.window_start_event_index if self.window_start_event_index is not None else -1,
            "event_end": self.window_end_event_index if self.window_end_event_index is not None else -1,
            "executed_queries": self.window_executed,
            "skipped_queries": self.window_skipped,
            "skipped_by_reason": dict(sorted(self.window_skipped_by_reason.items())),
            "recall": recall,
            "avg_cycles": avg_cycles,
            "avg_doc_vector_comparisons": avg_doc_vector_comparisons,
            "avg_local_doc_vector_comparisons": avg_local_doc_vector_comparisons,
            "avg_neighbor_doc_vector_comparisons": avg_neighbor_doc_vector_comparisons,
            "avg_routed_query_memory_comparisons": avg_routed_query_memory_comparisons,
            "avg_total_vector_comparisons": avg_total_vector_comparisons,
            "retrieval_recall": retrieval_recall,
            "available_relevant_doc_count": self.window_available_relevant_docs,
            "retrieved_relevant_doc_count": self.window_retrieved_relevant_docs,
            "avg_retrieved_relevant_doc_count": avg_retrieved_relevant_doc_count,
            "avg_requests_until_first_relevant": avg_requests_until_first_relevant,
            "first_relevant_found_count": self.window_first_relevant_found_count,
            "skip_rate": skip_rate,
        }
        self.windows.append(window)
        if self._window_log_handle is not None:
            self._window_log_handle.write(json.dumps(window, sort_keys=True) + "\n")
            self._window_log_handle.flush()
            print(
                "[metrics.window] "
                f"index={window['window_index']} "
                f"executed={window['executed_queries']} "
                f"skipped={window['skipped_queries']} "
                f"recall={window['recall']:.6f} "
                f"avg_cycles={window['avg_cycles']:.3f} "
                f"skip_rate={window['skip_rate']:.6f}",
                flush=True,
            )
        self.window_executed = 0
        self.window_skipped = 0
        self.window_cycles = 0
        self.window_successes = 0
        self.window_doc_vector_comparisons = 0
        self.window_local_doc_vector_comparisons = 0
        self.window_neighbor_doc_vector_comparisons = 0
        self.window_routed_query_memory_comparisons = 0
        self.window_total_vector_comparisons = 0
        self.window_available_relevant_docs = 0
        self.window_retrieved_relevant_docs = 0
        self.window_requests_until_first_relevant = 0
        self.window_first_relevant_found_count = 0
        self.window_skipped_by_reason = {}
        self.window_start_event_index = None
        self.window_end_event_index = None

    def finalize(self) -> dict[str, object]:
        self._flush_window()
        if self._log_handle is not None:
            self._log_handle.close()
        if self._window_log_handle is not None:
            self._window_log_handle.close()

        overall_recall = self.total_successes / self.executed_queries if self.executed_queries else 0.0
        overall_avg_cycles = self.total_cycles / self.executed_queries if self.executed_queries else 0.0
        overall_avg_doc_vector_comparisons = (
            self.total_doc_vector_comparisons / self.executed_queries
            if self.executed_queries
            else 0.0
        )
        overall_avg_local_doc_vector_comparisons = (
            self.total_local_doc_vector_comparisons / self.executed_queries
            if self.executed_queries
            else 0.0
        )
        overall_avg_neighbor_doc_vector_comparisons = (
            self.total_neighbor_doc_vector_comparisons / self.executed_queries
            if self.executed_queries
            else 0.0
        )
        overall_avg_routed_query_memory_comparisons = (
            self.total_routed_query_memory_comparisons / self.executed_queries
            if self.executed_queries
            else 0.0
        )
        overall_avg_total_vector_comparisons = (
            self.total_vector_comparisons / self.executed_queries
            if self.executed_queries
            else 0.0
        )
        total_events = self.executed_queries + self.skipped_queries
        overall_skip_rate = self.skipped_queries / total_events if total_events else 0.0
        overall_retrieval_recall = (
            self.total_retrieved_relevant_docs / self.total_available_relevant_docs
            if self.total_available_relevant_docs
            else 0.0
        )
        overall_avg_retrieved_relevant_doc_count = (
            self.total_retrieved_relevant_docs / self.executed_queries
            if self.executed_queries
            else 0.0
        )
        overall_avg_requests_until_first_relevant = (
            self.total_requests_until_first_relevant / self.first_relevant_found_count
            if self.first_relevant_found_count
            else 0.0
        )

        payload: dict[str, object] = {
            "query_log_mode": self.query_log_mode,
            "query_log_path": str(self.query_log_path) if self.query_log_mode == "full" else None,
            "window_log_path": str(self.window_log_path) if self.query_log_mode == "window" else None,
            "executed_queries": self.executed_queries,
            "skipped_queries": self.skipped_queries,
            "skipped_by_reason": dict(sorted(self.skipped_by_reason.items())),
            "overall_recall": overall_recall,
            "overall_avg_cycles": overall_avg_cycles,
            "overall_avg_doc_vector_comparisons": overall_avg_doc_vector_comparisons,
            "overall_avg_local_doc_vector_comparisons": (
                overall_avg_local_doc_vector_comparisons
            ),
            "overall_avg_neighbor_doc_vector_comparisons": (
                overall_avg_neighbor_doc_vector_comparisons
            ),
            "overall_avg_routed_query_memory_comparisons": (
                overall_avg_routed_query_memory_comparisons
            ),
            "overall_avg_total_vector_comparisons": overall_avg_total_vector_comparisons,
            "overall_retrieval_recall": overall_retrieval_recall,
            "total_available_relevant_doc_count": self.total_available_relevant_docs,
            "total_retrieved_relevant_doc_count": self.total_retrieved_relevant_docs,
            "overall_avg_retrieved_relevant_doc_count": (
                overall_avg_retrieved_relevant_doc_count
            ),
            "overall_avg_requests_until_first_relevant": (
                overall_avg_requests_until_first_relevant
            ),
            "first_relevant_found_count": self.first_relevant_found_count,
            "overall_skip_rate": overall_skip_rate,
            "windows": self.windows,
        }
        self.metrics_json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return payload


from __future__ import annotations

import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from typing import Sequence

from src.dataset import QueryEvent
from src.metrics import MetricsTracker, QueryLog
from src.network import RUNTIME_SEARCHABLE_SCENARIOS, ZeroRAGNetwork
from src.peer import PeerResponse, RoutedQuery, VectorComparisonCounts
from src.routing import (
    SearchResult,
    run_best_first_multi_relevant_search,
    run_best_first_search,
    run_fanout_multi_relevant_search,
    run_fanout_search,
)


@dataclass(slots=True)
class RoutedQueryDelta:
    peer_id: int
    routed_query: RoutedQuery


@dataclass(slots=True)
class EventDelta:
    event_index: int
    event: QueryEvent
    query_log: QueryLog | None
    skip_reason: str | None
    routed_query_deltas: list[RoutedQueryDelta] = field(default_factory=list)
    connect_peer_id: int | None = None
    search_seconds: float = 0.0


@dataclass(slots=True)
class EpochParallelSummary:
    epochs_processed: int = 0
    events_processed: int = 0
    worker_chunks: int = 0
    total_worker_search_seconds: float = 0.0
    max_worker_search_seconds: float = 0.0


_WORKER_NETWORK: ZeroRAGNetwork | None = None


def _set_worker_network(network: ZeroRAGNetwork) -> None:
    global _WORKER_NETWORK
    _WORKER_NETWORK = network


def _worker_query_peer(
    network: ZeroRAGNetwork,
    peer_id: int,
    *,
    query_embedding,
    query_index: int,
    target_doc_id: str,
    original_peer_id: int,
    comparison_counts: VectorComparisonCounts,
    routed_query_deltas: list[RoutedQueryDelta],
    relevant_doc_ids: frozenset[str] | None = None,
) -> PeerResponse:
    replica_referrals = network._best_replica_referrals(
        peer_id,
        query_embedding,
        comparison_counts=comparison_counts,
    )
    computation = network.peers[peer_id].compute_query_response(
        query_embedding=query_embedding,
        query_index=query_index,
        target_doc_id=target_doc_id,
        original_peer_id=original_peer_id,
        referral_limit=network.config.referrals_per_response,
        similarity_threshold=network.config.similarity_threshold,
        resolve_query_embedding=network.query_lookup.get_by_index,
        replica_referrals=replica_referrals,
        allow_local_hit=peer_id != original_peer_id,
        use_routed_query_memory=network.config.routing_mode == "semantic_query_memory",
        comparison_counts=comparison_counts,
        relevant_doc_ids=relevant_doc_ids,
    )
    if computation.routed_query is not None:
        routed_query_deltas.append(
            RoutedQueryDelta(peer_id=peer_id, routed_query=computation.routed_query)
        )
    return computation.response


def _skip_reason(network: ZeroRAGNetwork, event: QueryEvent) -> str | None:
    if network.config.scenario in RUNTIME_SEARCHABLE_SCENARIOS:
        raise RuntimeError("epoch_parallel does not support runtime-searchable scenarios yet")
    if network.config.retrieval_mode == "multi_relevant":
        relevant_doc_ids = network._relevant_doc_ids_for_event(event)
        available_doc_ids = network._available_relevant_doc_ids(relevant_doc_ids)
        if not available_doc_ids:
            return "non_searchable"
    elif not event.searchable:
        return "non_searchable"
    if event.query_text not in network.query_lookup:
        return "missing_query_embedding"
    return None


def _compute_event_delta(network: ZeroRAGNetwork, event: QueryEvent) -> EventDelta:
    reason = _skip_reason(network, event)
    if reason is not None:
        return EventDelta(
            event_index=event.event_index,
            event=event,
            query_log=None,
            skip_reason=reason,
        )

    query_index = network.query_lookup.index_of(event.query_text)
    query_embedding = network.query_lookup.get_by_index(query_index)
    comparison_counts = VectorComparisonCounts()
    routed_query_deltas: list[RoutedQueryDelta] = []
    is_multi_relevant = network.config.retrieval_mode == "multi_relevant"
    relevant_doc_ids = network._relevant_doc_ids_for_event(event)
    available_relevant_doc_ids = network._available_relevant_doc_ids(relevant_doc_ids)

    def query_peer(peer_id: int) -> PeerResponse:
        return _worker_query_peer(
            network,
            peer_id,
            query_embedding=query_embedding,
            query_index=query_index,
            target_doc_id=event.clicked_doc_id,
            original_peer_id=event.user_id,
            comparison_counts=comparison_counts,
            routed_query_deltas=routed_query_deltas,
            relevant_doc_ids=available_relevant_doc_ids if is_multi_relevant else None,
        )

    started_at = time.perf_counter()
    if is_multi_relevant and network.config.search_mode == "fanout":
        search_result = run_fanout_multi_relevant_search(
            querier_peer_id=event.user_id,
            max_cycles=network.config.max_cycles,
            fanout=network.config.fanout,
            query_peer=query_peer,
        )
    elif is_multi_relevant:
        search_result = run_best_first_multi_relevant_search(
            querier_peer_id=event.user_id,
            max_cycles=network.config.max_cycles,
            query_peer=query_peer,
        )
    elif network.config.search_mode == "fanout":
        search_result = run_fanout_search(
            querier_peer_id=event.user_id,
            target_doc_id=event.clicked_doc_id,
            max_cycles=network.config.max_cycles,
            fanout=network.config.fanout,
            query_peer=query_peer,
        )
    else:
        search_result = run_best_first_search(
            querier_peer_id=event.user_id,
            target_doc_id=event.clicked_doc_id,
            max_cycles=network.config.max_cycles,
            query_peer=query_peer,
        )
    search_seconds = time.perf_counter() - started_at
    if search_result.result_symbol != "+":
        routed_query_deltas = []

    connect_peer_id = None
    if network.config.selection_mode != "random_walk":
        if is_multi_relevant:
            connect_peer_id = search_result.best_contributing_peer_id
        else:
            connect_peer_id = (
                search_result.successful_peer_id or search_result.best_known_peer_id
            )

    return EventDelta(
        event_index=event.event_index,
        event=event,
        query_log=QueryLog(
            event_index=event.event_index,
            query_id=event.qid,
            querier_peer_id=event.user_id,
            clicked_doc_id=event.clicked_doc_id,
            cycles_used=search_result.cycles_used,
            result=search_result.result_symbol,
            termination_reason=search_result.termination_reason,
            doc_vector_comparisons=comparison_counts.doc_vector_comparisons,
            local_doc_vector_comparisons=comparison_counts.local_doc_vector_comparisons,
            neighbor_doc_vector_comparisons=comparison_counts.neighbor_doc_vector_comparisons,
            routed_query_memory_comparisons=(
                comparison_counts.routed_query_memory_comparisons
            ),
            total_vector_comparisons=comparison_counts.total_vector_comparisons,
            relevant_doc_count=len(relevant_doc_ids) if is_multi_relevant else 0,
            available_relevant_doc_count=(
                len(available_relevant_doc_ids) if is_multi_relevant else 0
            ),
            retrieved_relevant_doc_count=(
                len(search_result.retrieved_relevant_doc_ids) if is_multi_relevant else 0
            ),
            query_retrieval_recall=(
                len(search_result.retrieved_relevant_doc_ids) / len(available_relevant_doc_ids)
                if is_multi_relevant and available_relevant_doc_ids
                else 0.0
            ),
            requests_until_first_relevant=(
                search_result.requests_until_first_relevant if is_multi_relevant else None
            ),
        ),
        skip_reason=None,
        routed_query_deltas=routed_query_deltas,
        connect_peer_id=connect_peer_id,
        search_seconds=search_seconds,
    )


def _worker_event_chunk(chunk: list[QueryEvent]) -> list[EventDelta]:
    network = _WORKER_NETWORK
    if network is None:
        raise RuntimeError("epoch-parallel worker network was not initialized")
    return [_compute_event_delta(network, event) for event in chunk]


class ZeroRAGEpochParallelDriver:
    def __init__(
        self,
        *,
        network: ZeroRAGNetwork,
        parallelism: int,
        chunk_size: int,
        epoch_count: int = 0,
    ) -> None:
        self.network = network
        self.parallelism = parallelism
        self.chunk_size = chunk_size
        self.epoch_count = epoch_count
        self.summary = EpochParallelSummary()

    @property
    def epoch_size(self) -> int:
        return max(self.parallelism, 1) * max(self.chunk_size, 1)

    def run(self, events: Sequence[QueryEvent], metrics: MetricsTracker) -> None:
        if self.parallelism <= 1:
            for event in events:
                self.network.process_event(event, metrics)
            return
        if self.network.config.scenario in RUNTIME_SEARCHABLE_SCENARIOS:
            raise RuntimeError("epoch_parallel does not support runtime-searchable scenarios yet")
        if self.network.config.selection_mode != "similarity":
            raise RuntimeError("epoch_parallel currently supports selection_mode='similarity' only")

        try:
            ctx = mp.get_context("fork")
        except ValueError as exc:  # pragma: no cover - non-fork platforms
            raise RuntimeError(
                "epoch_parallel requires the 'fork' multiprocessing start method. "
                "Use sequential mode on platforms without fork."
            ) from exc

        max_epochs = self.epoch_count if self.epoch_count > 0 else None
        i = 0
        epoch_idx = 0
        n = len(events)
        while i < n and (max_epochs is None or epoch_idx < max_epochs):
            epoch_events = list(events[i : i + self.epoch_size])
            self._run_epoch(epoch_events, metrics, ctx)
            i += len(epoch_events)
            epoch_idx += 1
            self.summary.epochs_processed = epoch_idx
            self.summary.events_processed += len(epoch_events)

    def _run_epoch(
        self,
        epoch_events: list[QueryEvent],
        metrics: MetricsTracker,
        ctx: mp.context.BaseContext,
    ) -> None:
        _set_worker_network(self.network)
        chunks = [
            epoch_events[start : start + self.chunk_size]
            for start in range(0, len(epoch_events), self.chunk_size)
        ]
        self.summary.worker_chunks += len(chunks)
        with ProcessPoolExecutor(
            max_workers=self.parallelism,
            mp_context=ctx,
        ) as pool:
            futures = [pool.submit(_worker_event_chunk, chunk) for chunk in chunks]
            deltas: list[EventDelta] = []
            for fut in futures:
                deltas.extend(fut.result())

        deltas.sort(key=lambda delta: delta.event_index)
        for delta in deltas:
            self._apply_delta(delta, metrics)

    def _apply_delta(self, delta: EventDelta, metrics: MetricsTracker) -> None:
        if delta.skip_reason is not None:
            metrics.note_skipped_query(delta.skip_reason)
        else:
            if delta.query_log is None:
                raise RuntimeError(f"Missing query log for event_index={delta.event_index}")
            metrics.log_query(delta.query_log)
            for routed_delta in delta.routed_query_deltas:
                self.network.peers[routed_delta.peer_id].record_routed_query(
                    routed_delta.routed_query
                )
            if (
                self.network.config.topology_mode == "forming_new_connections"
                and delta.connect_peer_id is not None
                and delta.connect_peer_id != delta.event.user_id
            ):
                self.network.add_connection(delta.event.user_id, delta.connect_peer_id)

        if self.network.config.retrieval_mode != "multi_relevant":
            self.network._store_clicked_doc(delta.event.user_id, delta.event.clicked_doc_id)

        self.summary.total_worker_search_seconds += delta.search_seconds
        self.summary.max_worker_search_seconds = max(
            self.summary.max_worker_search_seconds,
            delta.search_seconds,
        )
        if self.network.config.profile_search and delta.search_seconds > 0:
            self.network.search_profile["executed_queries"] += 1
            self.network.search_profile["total_search_seconds"] += delta.search_seconds
            self.network.search_profile["max_search_seconds"] = max(
                self.network.search_profile["max_search_seconds"],
                delta.search_seconds,
            )


# Backward-compatible import name for existing Slurm scripts, tests, and notebooks.
LowCoordEpochParallelDriver = ZeroRAGEpochParallelDriver

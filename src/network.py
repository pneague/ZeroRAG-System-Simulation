from __future__ import annotations

import json
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np

from src.dataset import QueryEvent
from src.metrics import MetricsTracker, QueryLog
from src.peer import (
    REFERRAL_TIE_TOLERANCE,
    Peer,
    PeerResponse,
    Referral,
    RoutedQuery,
    VectorComparisonCounts,
)
from src.routing import (
    SearchResult,
    run_best_first_multi_relevant_search,
    run_best_first_search,
    run_fanout_multi_relevant_search,
    run_fanout_search,
)


@dataclass(slots=True)
class SimulationConfig:
    scenario: str = "empty_start_chronological"
    routing_mode: str = "semantic_query_memory"
    topology_mode: str = "forming_new_connections"
    selection_mode: str = "similarity"
    search_mode: str = "best_first"
    fanout: int = 1
    execution_mode: str = "sequential"
    epoch_parallelism: int = 1
    epoch_chunk_size: int = 500
    epoch_count: int = 0
    max_neighbors: int = 60
    max_cycles: int = 50
    referrals_per_response: int = 10
    similarity_threshold: float = 0.5
    metrics_window: int = 1000
    random_seed: int = 42
    max_routed_queries: int = 1000
    max_owned_docs: int = 0
    profile_search: bool = False
    churn_mode: str = "none"
    retrieval_mode: str = "single_target"
    external_acquisition_probability: float = 0.0
    relevant_doc_ids_by_qid: dict[int, frozenset[str]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.search_mode not in {"best_first", "fanout"}:
            raise ValueError(f"Unsupported search_mode={self.search_mode!r}")
        if self.execution_mode not in {"sequential", "epoch_parallel"}:
            raise ValueError(f"Unsupported execution_mode={self.execution_mode!r}")
        if self.fanout < 1:
            raise ValueError("fanout must be >= 1")
        if self.epoch_parallelism < 1:
            raise ValueError("epoch_parallelism must be >= 1")
        if self.epoch_chunk_size < 1:
            raise ValueError("epoch_chunk_size must be >= 1")
        if self.epoch_count < 0:
            raise ValueError("epoch_count must be >= 0")
        if self.search_mode == "fanout" and self.selection_mode != "similarity":
            raise ValueError("fanout search currently requires selection_mode='similarity'")
        if self.churn_mode not in {"none", "drop", "replace"}:
            raise ValueError(f"Unsupported churn_mode={self.churn_mode!r}")
        if self.retrieval_mode not in {"single_target", "multi_relevant"}:
            raise ValueError(f"Unsupported retrieval_mode={self.retrieval_mode!r}")
        if not 0.0 <= self.external_acquisition_probability <= 1.0:
            raise ValueError("external_acquisition_probability must be between 0 and 1")


@dataclass(frozen=True, slots=True)
class OwnershipInjection:
    event_index: int
    peer_id: int
    doc_id: str


RUNTIME_SEARCHABLE_SCENARIOS = {
    "fifo_chronological",
    "fifo_randomized_timeline",
    "empty_start_randomized_timeline",
}


class EmbeddingLookup:
    def __init__(
        self,
        *,
        keys_path: str | Path,
        embeddings_path: str | Path,
        load_into_memory: bool = False,
    ) -> None:
        mmap_mode = None if load_into_memory else "r"
        self.keys = np.load(Path(keys_path), mmap_mode=mmap_mode)
        self.embeddings = np.load(Path(embeddings_path), mmap_mode=mmap_mode)
        self.index = {str(key): idx for idx, key in enumerate(self.keys.tolist())}

    def get(self, key: str) -> np.ndarray:
        idx = self.index[key]
        return np.asarray(self.embeddings[idx], dtype=np.float32)

    def index_of(self, key: str) -> int:
        """Return the row index of ``key`` in the underlying embeddings array.

        Used by the routed-query memory path so peers can store an ``int``
        reference instead of copying the full embedding vector. The return
        value is stable for the lifetime of this lookup and can be passed to
        :meth:`get_by_index` to resolve the embedding lazily.
        """

        return self.index[key]

    def get_by_index(self, idx: int) -> np.ndarray:
        return np.asarray(self.embeddings[idx], dtype=np.float32)

    def __contains__(self, key: str) -> bool:
        return key in self.index


class ZeroRAGNetwork:
    PROGRESS_LOG_INTERVAL = 5_000

    def __init__(
        self,
        *,
        peer_ids: Iterable[int],
        query_lookup: EmbeddingLookup,
        doc_lookup: EmbeddingLookup,
        config: SimulationConfig,
    ) -> None:
        self.query_lookup = query_lookup
        self.doc_lookup = doc_lookup
        self.config = config
        self.rng = np.random.default_rng(config.random_seed)
        self.search_profile = {
            "enabled": config.profile_search,
            "executed_queries": 0,
            "total_search_seconds": 0.0,
            "max_search_seconds": 0.0,
        }
        unique_peer_ids = sorted(set(peer_ids))
        self.peers: dict[int, Peer] = {}
        for created, peer_id in enumerate(unique_peer_ids, start=1):
            self.peers[peer_id] = Peer(
                peer_id=peer_id,
                max_neighbors=config.max_neighbors,
                max_routed_queries=config.max_routed_queries,
                max_owned_docs=config.max_owned_docs,
            )
            if (
                self.PROGRESS_LOG_INTERVAL > 0
                and created % self.PROGRESS_LOG_INTERVAL == 0
            ):
                print(
                    f"[network.init] created {created:,}/{len(unique_peer_ids):,} peers",
                    flush=True,
                )
        if self.PROGRESS_LOG_INTERVAL > 0:
            print(
                f"[network.init] done: {len(self.peers):,} peers initialized",
                flush=True,
            )
        self.active_peer_ids: set[int] = set(unique_peer_ids)
        self.incoming_connections: dict[int, set[int]] = defaultdict(set)
        self.doc_owners: dict[str, set[int]] = defaultdict(set)
        self._active_peer_ids_array_cache: np.ndarray | None = None
        self._active_peer_index_cache: dict[int, int] | None = None
        self._available_relevant_doc_ids_cache: dict[int, frozenset[str]] = {}
        self._ownership_injections: list[OwnershipInjection] = []
        self._next_ownership_injection = 0

    def _invalidate_active_peer_cache(self) -> None:
        self._active_peer_ids_array_cache = None
        self._active_peer_index_cache = None

    def _invalidate_availability_cache(self) -> None:
        self._available_relevant_doc_ids_cache.clear()

    def _active_peer_ids_array(self) -> tuple[np.ndarray, dict[int, int]]:
        if self._active_peer_ids_array_cache is None or self._active_peer_index_cache is None:
            peer_ids = np.asarray(sorted(self.active_peer_ids), dtype=np.int64)
            self._active_peer_ids_array_cache = peer_ids
            self._active_peer_index_cache = {
                int(peer_id): idx for idx, peer_id in enumerate(peer_ids.tolist())
            }
        return self._active_peer_ids_array_cache, self._active_peer_index_cache

    @classmethod
    def from_embedding_manifest(
        cls,
        *,
        peer_ids: Iterable[int],
        manifest_path: str | Path,
        config: SimulationConfig,
        load_query_embeddings_into_memory: bool = False,
        load_doc_embeddings_into_memory: bool = False,
    ) -> "ZeroRAGNetwork":
        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        query_lookup = EmbeddingLookup(
            keys_path=manifest["query_keys_path"],
            embeddings_path=manifest["query_embeddings_path"],
            load_into_memory=load_query_embeddings_into_memory,
        )
        doc_lookup = EmbeddingLookup(
            keys_path=manifest["doc_keys_path"],
            embeddings_path=manifest["doc_embeddings_path"],
            load_into_memory=load_doc_embeddings_into_memory,
        )
        return cls(
            peer_ids=peer_ids,
            query_lookup=query_lookup,
            doc_lookup=doc_lookup,
            config=config,
        )

    def bootstrap_random_connections(self) -> None:
        peer_ids = sorted(self.active_peer_ids)
        total_peers = len(peer_ids)
        for processed, peer_id in enumerate(peer_ids, start=1):
            candidates = [candidate for candidate in peer_ids if candidate != peer_id]
            if not candidates:
                continue
            sample_size = min(self.config.max_neighbors, len(candidates))
            sampled = self.rng.choice(candidates, size=sample_size, replace=False)
            for neighbor_id in sampled.tolist():
                self.add_connection(peer_id, int(neighbor_id))
            if (
                self.PROGRESS_LOG_INTERVAL > 0
                and processed % self.PROGRESS_LOG_INTERVAL == 0
            ):
                print(
                    f"[network.bootstrap] connected {processed:,}/{total_peers:,} peers",
                    flush=True,
                )
        if self.PROGRESS_LOG_INTERVAL > 0:
            print(
                f"[network.bootstrap] done: {total_peers:,} peers bootstrapped",
                flush=True,
            )

    def add_connection(self, source_peer_id: int, target_peer_id: int) -> None:
        if source_peer_id == target_peer_id:
            return
        if source_peer_id not in self.active_peer_ids or target_peer_id not in self.active_peer_ids:
            return
        source_peer = self.peers[source_peer_id]
        evicted = source_peer.add_connection(target_peer_id)
        if evicted is not None:
            self.incoming_connections[evicted].discard(source_peer_id)
        if target_peer_id not in source_peer.connections:
            return
        self.incoming_connections[target_peer_id].add(source_peer_id)

    def add_peer(self, peer_id: int) -> Peer:
        peer = self.peers.get(peer_id)
        if peer is None:
            peer = Peer(
                peer_id=peer_id,
                max_neighbors=self.config.max_neighbors,
                max_routed_queries=self.config.max_routed_queries,
                max_owned_docs=self.config.max_owned_docs,
            )
            self.peers[peer_id] = peer
        self.active_peer_ids.add(peer_id)
        self._invalidate_active_peer_cache()
        return peer

    def _connect_new_active_peer(self, peer_id: int) -> None:
        candidates = [candidate for candidate in self.active_peer_ids if candidate != peer_id]
        if not candidates:
            return
        sample_size = min(self.config.max_neighbors, len(candidates))
        sampled = self.rng.choice(candidates, size=sample_size, replace=False)
        for neighbor_id in sampled.tolist():
            self.add_connection(peer_id, int(neighbor_id))

    def active_single_owner_doc_count(self) -> int:
        return sum(1 for owners in self.doc_owners.values() if len(owners & self.active_peer_ids) == 1)

    def _random_active_peer(self, *, exclude: set[int] | None = None) -> int | None:
        excluded = exclude or set()
        candidates = [peer_id for peer_id in self.active_peer_ids if peer_id not in excluded]
        if not candidates:
            return None
        return int(self.rng.choice(candidates))

    def _rewire_connection_away_from(self, source_peer_id: int, dropped_peer_id: int) -> None:
        source_peer = self.peers.get(source_peer_id)
        if source_peer is None:
            return
        source_peer.remove_connection(dropped_peer_id)
        replacement = self._random_active_peer(exclude={source_peer_id, dropped_peer_id})
        if replacement is not None:
            self.add_connection(source_peer_id, replacement)

    def deactivate_peer(self, peer_id: int) -> None:
        if peer_id not in self.active_peer_ids:
            return
        peer = self.peers[peer_id]
        self.active_peer_ids.discard(peer_id)
        self._invalidate_active_peer_cache()
        self._invalidate_availability_cache()

        for doc_id in list(peer.owned_docs):
            owners = self.doc_owners.get(doc_id)
            if owners is None:
                continue
            owners.discard(peer_id)
            if not owners:
                del self.doc_owners[doc_id]

        for neighbor_id in list(peer.connections):
            self.incoming_connections[neighbor_id].discard(peer_id)
        peer.connections.clear()

        incoming_sources = list(self.incoming_connections.pop(peer_id, set()))
        for source_peer_id in incoming_sources:
            if source_peer_id in self.active_peer_ids:
                self._rewire_connection_away_from(source_peer_id, peer_id)

    def activate_peer_with_docs(self, peer_id: int, doc_ids: list[str] | tuple[str, ...] | set[str] | None = None) -> None:
        peer = self.add_peer(peer_id)
        self._invalidate_availability_cache()
        ids_to_activate = list(doc_ids) if doc_ids is not None else list(peer.owned_docs)
        for doc_id in ids_to_activate:
            if doc_id not in self.doc_lookup:
                continue
            if doc_id not in peer.owned_docs:
                peer.add_owned_doc(doc_id, self.doc_lookup.get(doc_id))
            self.doc_owners[doc_id].add(peer_id)
        self._connect_new_active_peer(peer_id)

    def replace_peer(self, dropped_peer_id: int, replacement_peer_id: int) -> None:
        if dropped_peer_id not in self.active_peer_ids:
            return
        dropped_peer = self.peers[dropped_peer_id]
        replacement_peer = self.add_peer(replacement_peer_id)
        self._invalidate_availability_cache()
        replacement_peer.connections.clear()
        replacement_peer.owned_docs.clear()
        replacement_peer.owned_doc_catalog = type(replacement_peer.owned_doc_catalog)()
        replacement_peer.fifo_order.clear()

        for doc_id, embedding in list(dropped_peer.owned_docs.items()):
            replacement_peer.add_owned_doc(doc_id, embedding)
            owners = self.doc_owners.get(doc_id)
            if owners is None:
                self.doc_owners[doc_id].add(replacement_peer_id)
                continue
            owners.discard(dropped_peer_id)
            owners.add(replacement_peer_id)

        self.deactivate_peer(dropped_peer_id)
        self._connect_new_active_peer(replacement_peer_id)

    def _best_replica_referrals(
        self,
        peer_id: int,
        query_embedding: np.ndarray,
        *,
        comparison_counts: VectorComparisonCounts | None = None,
    ) -> list[Referral]:
        if self.config.selection_mode == "random_walk":
            return []

        referrals: list[Referral] = []
        for neighbor_id in self.peers[peer_id].connections:
            if neighbor_id not in self.active_peer_ids:
                continue
            referral = self.peers[neighbor_id].best_owned_doc_referral(
                query_embedding,
                source="replica",
                comparison_counts=comparison_counts,
            )
            if referral is not None:
                referrals.append(referral)
        return referrals

    def seed_peer_docs(self, mapping: dict[int, list[str] | tuple[str, ...] | set[str]]) -> dict[str, int]:
        seeded_peer_count = 0
        seeded_doc_assignments = 0
        missing_doc_ids: set[str] = set()
        id_only_seed = self.config.selection_mode == "random_walk"
        empty_embedding = np.empty(0, dtype=np.float32)

        for peer_id, doc_ids in mapping.items():
            peer = self.peers.get(peer_id)
            if peer is None:
                continue
            seeded_any = False
            for doc_id in doc_ids:
                if doc_id not in self.doc_lookup:
                    missing_doc_ids.add(doc_id)
                    continue
                if id_only_seed and peer.max_owned_docs <= 0:
                    # Random-walk probes only inspect owned-doc ids, never doc vectors.
                    # Avoid millions of memmap vector reads for sampled Amazon inventory.
                    peer.owned_docs[doc_id] = empty_embedding
                    peer.fifo_order[doc_id] = None
                    peer.fifo_order.move_to_end(doc_id)
                else:
                    peer.add_owned_doc(doc_id, self.doc_lookup.get(doc_id))
                self.doc_owners[doc_id].add(peer_id)
                seeded_doc_assignments += 1
                seeded_any = True
            if seeded_any:
                seeded_peer_count += 1

        self._invalidate_availability_cache()
        return {
            "seeded_peer_count": seeded_peer_count,
            "seeded_doc_assignments": seeded_doc_assignments,
            "seeded_missing_doc_count": len(missing_doc_ids),
        }

    def set_ownership_injection_schedule(
        self,
        injections: Iterable[OwnershipInjection | dict[str, object]],
    ) -> None:
        parsed: list[OwnershipInjection] = []
        for injection in injections:
            if isinstance(injection, OwnershipInjection):
                parsed.append(injection)
                continue
            parsed.append(
                OwnershipInjection(
                    event_index=int(injection["event_index"]),
                    peer_id=int(injection["peer_id"]),
                    doc_id=str(injection["doc_id"]),
                )
            )
        self._ownership_injections = sorted(
            parsed,
            key=lambda item: (item.event_index, item.peer_id, item.doc_id),
        )
        self._next_ownership_injection = 0

    def _apply_pending_ownership_injections(self, event_index: int) -> None:
        applied = False
        while self._next_ownership_injection < len(self._ownership_injections):
            injection = self._ownership_injections[self._next_ownership_injection]
            if injection.event_index > event_index:
                break
            self._next_ownership_injection += 1
            if injection.peer_id not in self.peers or injection.doc_id not in self.doc_lookup:
                continue
            peer = self.peers[injection.peer_id]
            evicted_doc_id = peer.add_owned_doc(injection.doc_id, self.doc_lookup.get(injection.doc_id))
            self.doc_owners[injection.doc_id].add(injection.peer_id)
            if evicted_doc_id is not None:
                remaining_owners = self.doc_owners.get(evicted_doc_id)
                if remaining_owners is not None:
                    remaining_owners.discard(injection.peer_id)
                    if not remaining_owners:
                        del self.doc_owners[evicted_doc_id]
            applied = True
        if applied:
            self._invalidate_availability_cache()

    def _store_clicked_doc(self, peer_id: int, doc_id: str) -> None:
        if doc_id not in self.doc_lookup:
            return
        embedding = self.doc_lookup.get(doc_id)
        owner_peer = self.peers[peer_id]
        evicted_doc_id = owner_peer.add_owned_doc(doc_id, embedding)
        self.doc_owners[doc_id].add(peer_id)
        self._invalidate_availability_cache()
        if evicted_doc_id is None:
            return
        remaining_owners = self.doc_owners.get(evicted_doc_id)
        if remaining_owners is None:
            return
        remaining_owners.discard(peer_id)
        if not remaining_owners:
            del self.doc_owners[evicted_doc_id]

    def _is_runtime_searchable(self, querier_id: int, doc_id: str) -> bool:
        owners = self.doc_owners.get(doc_id)
        return bool(
            owners
            and any(
                owner_id != querier_id and owner_id in self.active_peer_ids
                for owner_id in owners
            )
        )

    def _relevant_doc_ids_for_event(self, event: QueryEvent) -> frozenset[str]:
        if self.config.retrieval_mode != "multi_relevant":
            return frozenset({event.clicked_doc_id})
        return self.config.relevant_doc_ids_by_qid.get(event.qid, frozenset())

    def _available_relevant_doc_ids(self, relevant_doc_ids: frozenset[str]) -> frozenset[str]:
        return frozenset(
            doc_id
            for doc_id in relevant_doc_ids
            if any(owner in self.active_peer_ids for owner in self.doc_owners.get(doc_id, set()))
        )

    def _available_relevant_doc_ids_for_event(
        self,
        event: QueryEvent,
        relevant_doc_ids: frozenset[str],
    ) -> frozenset[str]:
        if self.config.retrieval_mode != "multi_relevant":
            return self._available_relevant_doc_ids(relevant_doc_ids)
        cached = self._available_relevant_doc_ids_cache.get(event.qid)
        if cached is not None:
            return cached
        available = self._available_relevant_doc_ids(relevant_doc_ids)
        self._available_relevant_doc_ids_cache[event.qid] = available
        return available

    def _query_peer(
        self,
        peer_id: int,
        *,
        query_embedding: np.ndarray,
        query_index: int,
        target_doc_id: str,
        original_peer_id: int,
        comparison_counts: VectorComparisonCounts,
        relevant_doc_ids: frozenset[str] | None = None,
        pending_routed_queries: list[tuple[int, RoutedQuery]] | None = None,
    ) -> PeerResponse:
        if peer_id not in self.active_peer_ids:
            return PeerResponse(
                queried_peer_id=peer_id,
                found=False,
                found_doc_id=None,
                referrals=[],
            )
        replica_referrals = self._best_replica_referrals(
            peer_id,
            query_embedding,
            comparison_counts=comparison_counts,
        )
        computation = self.peers[peer_id].compute_query_response(
            query_embedding=query_embedding,
            query_index=query_index,
            target_doc_id=target_doc_id,
            original_peer_id=original_peer_id,
            referral_limit=self.config.referrals_per_response,
            similarity_threshold=self.config.similarity_threshold,
            resolve_query_embedding=self.query_lookup.get_by_index,
            replica_referrals=replica_referrals,
            allow_local_hit=peer_id != original_peer_id,
            use_routed_query_memory=self.config.routing_mode == "semantic_query_memory",
            comparison_counts=comparison_counts,
            relevant_doc_ids=relevant_doc_ids,
            referral_rng=self.rng,
        )
        if computation.routed_query is not None:
            if pending_routed_queries is None:
                self.peers[peer_id].record_routed_query(computation.routed_query)
            else:
                pending_routed_queries.append((peer_id, computation.routed_query))
        return computation.response

    def _select_random_best_referral(self, unqueried: list[Referral]) -> Referral:
        best_similarity = max(referral.similarity for referral in unqueried)
        tied = [
            referral
            for referral in unqueried
            if abs(referral.similarity - best_similarity) <= REFERRAL_TIE_TOLERANCE
        ]
        return tied[int(self.rng.integers(0, len(tied)))]

    def _run_random_user_probe(
        self,
        *,
        querier_peer_id: int,
        target_doc_id: str,
        relevant_doc_ids: frozenset[str] | None = None,
    ) -> SearchResult:
        active_peer_ids, peer_index = self._active_peer_ids_array()
        total_active = int(active_peer_ids.size)
        if total_active <= 1 or self.config.max_cycles <= 0:
            return SearchResult(
                cycles_used=0,
                result_symbol="-",
                termination_reason="exhausted",
                queried_peer_ids=[],
                successful_peer_id=None,
                best_known_peer_id=None,
            )

        querier_idx = peer_index.get(querier_peer_id)
        if querier_idx is None:
            return SearchResult(
                cycles_used=0,
                result_symbol="-",
                termination_reason="exhausted",
                queried_peer_ids=[],
                successful_peer_id=None,
                best_known_peer_id=None,
            )

        sample_size = min(self.config.max_cycles, total_active - 1)
        sampled_indices = self.rng.choice(total_active - 1, size=sample_size, replace=False)
        sampled_indices = np.where(sampled_indices >= querier_idx, sampled_indices + 1, sampled_indices)
        sampled = active_peer_ids[sampled_indices]
        queried_peer_ids: list[int] = []
        retrieved: set[str] = set()
        contributing: set[int] = set()
        requests_until_first: int | None = None
        for peer_id in (int(value) for value in sampled.tolist()):
            queried_peer_ids.append(peer_id)
            if relevant_doc_ids is not None:
                found_docs = relevant_doc_ids.intersection(self.peers[peer_id].owned_docs)
                if found_docs:
                    if requests_until_first is None:
                        requests_until_first = len(queried_peer_ids)
                    retrieved.update(found_docs)
                    contributing.add(peer_id)
                continue
            if self.peers[peer_id].has_doc(target_doc_id):
                return SearchResult(
                    cycles_used=len(queried_peer_ids),
                    result_symbol="+",
                    termination_reason="found",
                    queried_peer_ids=queried_peer_ids,
                    successful_peer_id=peer_id,
                    best_known_peer_id=peer_id,
                )

        termination_reason = "cap" if total_active - 1 > sample_size else "exhausted"
        if relevant_doc_ids is not None:
            return SearchResult(
                cycles_used=len(queried_peer_ids),
                result_symbol="+" if retrieved else "-",
                termination_reason=termination_reason,
                queried_peer_ids=queried_peer_ids,
                successful_peer_id=None,
                best_known_peer_id=None,
                retrieved_relevant_doc_ids=tuple(sorted(retrieved)),
                contributing_peer_ids=tuple(sorted(contributing)),
                requests_until_first_relevant=requests_until_first,
            )
        return SearchResult(
            cycles_used=len(queried_peer_ids),
            result_symbol="-",
            termination_reason=termination_reason,
            queried_peer_ids=queried_peer_ids,
            successful_peer_id=None,
            best_known_peer_id=None,
        )

    def process_event(self, event: QueryEvent, metrics: MetricsTracker) -> None:
        self._apply_pending_ownership_injections(event.event_index)
        if event.user_id not in self.active_peer_ids:
            metrics.note_skipped_query("inactive_peer")
            return

        requires_query_embedding = self.config.selection_mode != "random_walk"
        is_multi_relevant = self.config.retrieval_mode == "multi_relevant"
        relevant_doc_ids = self._relevant_doc_ids_for_event(event)
        available_relevant_doc_ids = self._available_relevant_doc_ids_for_event(
            event, relevant_doc_ids
        )
        if is_multi_relevant and not available_relevant_doc_ids:
            metrics.note_skipped_query("non_searchable")
            return
        use_runtime_searchability = (
            self.config.scenario in RUNTIME_SEARCHABLE_SCENARIOS
            or self.config.churn_mode != "none"
            or is_multi_relevant
        )
        if use_runtime_searchability:
            if (
                not is_multi_relevant
                and self.config.scenario in RUNTIME_SEARCHABLE_SCENARIOS
                and self.peers[event.user_id].has_doc(event.clicked_doc_id)
            ):
                metrics.note_skipped_query("local_held")
                self._store_clicked_doc(event.user_id, event.clicked_doc_id)
                return

            if requires_query_embedding and event.query_text not in self.query_lookup:
                metrics.note_skipped_query("missing_query_embedding")
                return

            if not is_multi_relevant and not self._is_runtime_searchable(
                event.user_id, event.clicked_doc_id
            ):
                metrics.note_skipped_query("non_searchable")
                return
        elif not event.searchable:
            metrics.note_skipped_query("non_searchable")
            return

        if requires_query_embedding and event.query_text not in self.query_lookup:
            metrics.note_skipped_query("missing_query_embedding")
            return

        if (
            self.config.scenario == "empty_start_chronological"
            and self.peers[event.user_id].has_doc(event.clicked_doc_id)
        ):
            metrics.note_skipped_query("local_held")
            self._store_clicked_doc(event.user_id, event.clicked_doc_id)
            return

        comparison_counts = VectorComparisonCounts()
        search_started_at = time.perf_counter() if self.config.profile_search else 0.0
        if self.config.selection_mode == "random_walk":
            search_result = self._run_random_user_probe(
                querier_peer_id=event.user_id,
                target_doc_id=event.clicked_doc_id,
                relevant_doc_ids=available_relevant_doc_ids if is_multi_relevant else None,
            )
        else:
            query_index = self.query_lookup.index_of(event.query_text)
            query_embedding = self.query_lookup.get_by_index(query_index)
            pending_routed_queries: list[tuple[int, RoutedQuery]] = []
            query_peer = lambda peer_id: self._query_peer(
                peer_id,
                query_embedding=query_embedding,
                query_index=query_index,
                target_doc_id=event.clicked_doc_id,
                original_peer_id=event.user_id,
                comparison_counts=comparison_counts,
                relevant_doc_ids=available_relevant_doc_ids if is_multi_relevant else None,
                pending_routed_queries=pending_routed_queries,
            )
            if is_multi_relevant and self.config.search_mode == "fanout":
                search_result = run_fanout_multi_relevant_search(
                    querier_peer_id=event.user_id,
                    max_cycles=self.config.max_cycles,
                    fanout=self.config.fanout,
                    query_peer=query_peer,
                )
            elif is_multi_relevant:
                search_result = run_best_first_multi_relevant_search(
                    querier_peer_id=event.user_id,
                    max_cycles=self.config.max_cycles,
                    query_peer=query_peer,
                    select_next=self._select_random_best_referral,
                )
            elif self.config.search_mode == "fanout":
                search_result = run_fanout_search(
                    querier_peer_id=event.user_id,
                    target_doc_id=event.clicked_doc_id,
                    max_cycles=self.config.max_cycles,
                    fanout=self.config.fanout,
                    query_peer=query_peer,
                )
            else:
                search_result = run_best_first_search(
                    querier_peer_id=event.user_id,
                    target_doc_id=event.clicked_doc_id,
                    max_cycles=self.config.max_cycles,
                    query_peer=query_peer,
                    select_next=self._select_random_best_referral,
                )
            if search_result.result_symbol == "+":
                for peer_id, routed_query in pending_routed_queries:
                    self.peers[peer_id].record_routed_query(routed_query)
        if self.config.profile_search:
            elapsed = time.perf_counter() - search_started_at
            self.search_profile["executed_queries"] += 1
            self.search_profile["total_search_seconds"] += elapsed
            self.search_profile["max_search_seconds"] = max(
                self.search_profile["max_search_seconds"],
                elapsed,
            )
        metrics.log_query(
            QueryLog(
                event_index=event.event_index,
                query_id=event.qid,
                querier_peer_id=event.user_id,
                clicked_doc_id=event.clicked_doc_id,
                cycles_used=search_result.cycles_used,
                result=search_result.result_symbol,
                termination_reason=search_result.termination_reason,
                doc_vector_comparisons=comparison_counts.doc_vector_comparisons,
                local_doc_vector_comparisons=comparison_counts.local_doc_vector_comparisons,
                neighbor_doc_vector_comparisons=(
                    comparison_counts.neighbor_doc_vector_comparisons
                ),
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
            )
        )
        connect_peer_id = None
        if self.config.selection_mode != "random_walk":
            connect_peer_id = (
                search_result.best_contributing_peer_id
                if is_multi_relevant
                else search_result.successful_peer_id
            )
            if not is_multi_relevant:
                connect_peer_id = connect_peer_id or search_result.best_known_peer_id
        if (
            self.config.topology_mode == "forming_new_connections"
            and connect_peer_id is not None
            and connect_peer_id != event.user_id
        ):
            self.add_connection(event.user_id, connect_peer_id)

        if not is_multi_relevant and search_result.result_symbol == "+":
            # Successful retrieval is the only path by which a queried doc spreads.
            self._store_clicked_doc(event.user_id, event.clicked_doc_id)


# Backward-compatible import name for existing Slurm scripts, tests, and notebooks.
LowCoordSemanticDHTNetwork = ZeroRAGNetwork

"""Distributed E2LSH-DHT baseline.

Sibling of :mod:`src.network`. Same dataset/event surface, same skip
reasons, same ``cycles_used`` semantics (count the querier as cycle 1,
then add the number of distinct non-querier bucket-owner peers contacted),
but the routing layer is replaced by a paper-faithful E2LSH index sharded
across peers via deterministic consistent hashing on sparse dataset peer
ids.

The hash family math lives in :mod:`src.lsh`; this module only handles
peer state, bucket ownership, doc publication and query resolution. Data
classes and skip-reason wording are kept aligned with
:class:`src.network.ZeroRAGNetwork` so the resulting
``query_log.jsonl`` and ``metrics.json`` are plot-compatible with the
semantic baseline.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np

from src.dataset import QueryEvent
from src.lsh import BucketId, E2LSHFamily
from src.metrics import MetricsTracker, QueryLog
from src.network import EmbeddingLookup
from src.peer import Peer
from src.scenarios import (
    FIFO_SCENARIOS,
    RUNTIME_SEARCHABLE_SCENARIOS,
    SEEDED_SCENARIOS,
)


@dataclass(slots=True)
class LSHSimulationConfig:
    """All knobs needed to construct the LSH-DHT.

    ``w`` is the paper bucket-width symbol; ``r`` is accepted as an alias
    by :class:`src.lsh.E2LSHFamily.build` to match the reference
    ``CharlesLiu7/p-stable-lsh-python`` parameter spelling.
    """

    scenario: str = "empty_start_chronological"
    k: int = 10
    L: int = 20
    w: float = 0.75
    bucket_replication: int = 1
    candidate_rerank_limit: int = 0
    metrics_window: int = 1000
    random_seed: int = 42
    max_owned_docs: int = 0
    distinct_candidate_sample_size: int = 5_000
    retrieval_mode: str = "single_target"
    relevant_doc_ids_by_qid: dict[int, frozenset[str]] = field(default_factory=dict)


@dataclass(slots=True)
class LSHSearchResult:
    """LSH analog of :class:`src.routing.SearchResult`.

    ``cycles_used`` is intentionally aligned with the semantic baseline:
    the querier counts as the first cycle (it does the local hash and
    candidate aggregation), then each distinct non-querier bucket-owner
    peer contacted adds one more cycle. ``result_symbol`` and
    ``termination_reason`` are emitted into ``query_log.jsonl`` exactly
    like the semantic baseline does.
    """

    cycles_used: int
    result_symbol: str
    termination_reason: str
    query_contacts: int
    candidate_count: int
    distinct_candidate_doc_count: int | None
    contacted_peer_ids: list[int]
    retrieved_relevant_doc_ids: tuple[str, ...] = ()


@dataclass(slots=True)
class LSHCoordinationCounters:
    """Aggregate coordination counters reported in ``lsh_metrics.json``.

    Kept separate from :class:`src.metrics.MetricsTracker` so that the
    existing ``metrics.json`` schema (recall / avg_cycles / skip_rate)
    stays byte-shape compatible with the semantic baseline's plotting.
    """

    publish_messages: int = 0
    query_contacts: int = 0
    candidate_total: int = 0
    distinct_candidate_doc_total: int = 0
    distinct_candidate_doc_samples: int = 0
    queries_executed: int = 0
    queries_skipped: int = 0
    docs_published: int = 0
    bucket_count_distinct: int = 0


class LSHDHTNetwork:
    """Distributed E2LSH index sharded across simulated peers.

    Bucket ownership is computed by a deterministic SHA1 of
    ``(table_id, bucket_id)`` modulo the number of unique peer ids;
    the resulting index is mapped back through ``sorted_peer_ids`` so
    sparse dataset user ids work correctly. Replication writes the next
    ``bucket_replication - 1`` adjacent indices (wrap-around).
    """

    PROGRESS_LOG_INTERVAL = 5_000

    def __init__(
        self,
        *,
        peer_ids: Iterable[int],
        query_lookup: EmbeddingLookup,
        doc_lookup: EmbeddingLookup,
        config: LSHSimulationConfig,
    ) -> None:
        self.query_lookup = query_lookup
        self.doc_lookup = doc_lookup
        self.config = config

        unique_peer_ids = sorted(set(peer_ids))
        if not unique_peer_ids:
            raise ValueError("LSHDHTNetwork requires at least one peer")
        self.sorted_peer_ids: list[int] = unique_peer_ids

        # Reuse the Peer doc-store + FIFO machinery so eviction matches the
        # semantic baseline byte-for-byte. We never use Peer.connections.
        self.peers: dict[int, Peer] = {}
        for created, peer_id in enumerate(unique_peer_ids, start=1):
            self.peers[peer_id] = Peer(
                peer_id=peer_id,
                max_neighbors=1,  # connections unused; keep deque non-zero to satisfy maxlen
                max_routed_queries=0,
                max_owned_docs=config.max_owned_docs,
            )
            if (
                self.PROGRESS_LOG_INTERVAL > 0
                and created % self.PROGRESS_LOG_INTERVAL == 0
            ):
                print(
                    f"[lsh_network.init] created {created:,}/{len(unique_peer_ids):,} peers",
                    flush=True,
                )
        if self.PROGRESS_LOG_INTERVAL > 0:
            print(
                f"[lsh_network.init] done: {len(self.peers):,} peers initialized",
                flush=True,
            )

        # Doc-id ownership (parallels semantic baseline so the
        # querier-never-counts-local rule can be enforced uniformly).
        self.doc_owners: dict[str, set[int]] = defaultdict(set)

        # E2LSH family parameters are sourced from the embedding manifest
        # dimension and the simulation seed. Both must be locked at network
        # construction so two runs with the same manifest+seed produce
        # identical hash projections.
        embedding_dim = int(query_lookup.embeddings.shape[1])
        if int(doc_lookup.embeddings.shape[1]) != embedding_dim:
            raise ValueError(
                "Query and doc embedding manifests have mismatched dims: "
                f"queries={query_lookup.embeddings.shape[1]} "
                f"docs={doc_lookup.embeddings.shape[1]}"
            )
        self.embedding_dim = embedding_dim
        self.lsh = E2LSHFamily.build(
            d=embedding_dim,
            k=config.k,
            L=config.L,
            w=config.w,
            seed=config.random_seed,
        )

        # Distributed LSH index. Keyed by (table_id, bucket_id) to keep the
        # L tables independent. Inner mapping is doc_id -> set[owner_peer_id]
        # so a query can answer "is target_doc_id present in this bucket
        # with at least one non-querier owner?" in O(1) per bucket without
        # materializing a candidate set.
        self.buckets: dict[tuple[int, BucketId], dict[str, set[int]]] = {}
        self._published_doc_ids: set[str] = set()
        self._distinct_candidate_sample_event_indices: set[int] | None = None
        # Coordination counters aggregated across the whole simulation.
        self.coord = LSHCoordinationCounters()

        if config.bucket_replication < 1:
            raise ValueError(
                f"bucket_replication must be >= 1, got {config.bucket_replication}"
            )

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_embedding_manifest(
        cls,
        *,
        peer_ids: Iterable[int],
        manifest_path: str | Path,
        config: LSHSimulationConfig,
    ) -> "LSHDHTNetwork":
        """Mirror :meth:`ZeroRAGNetwork.from_embedding_manifest`.

        Reuses :class:`EmbeddingLookup` from :mod:`src.network` so the LSH
        baseline reads the exact same memmapped embedding files as the
        semantic baseline. No new embedding model is loaded; no
        ``sentence_transformers`` import in this code path.
        """

        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        query_lookup = EmbeddingLookup(
            keys_path=manifest["query_keys_path"],
            embeddings_path=manifest["query_embeddings_path"],
        )
        doc_lookup = EmbeddingLookup(
            keys_path=manifest["doc_keys_path"],
            embeddings_path=manifest["doc_embeddings_path"],
        )
        return cls(
            peer_ids=peer_ids,
            query_lookup=query_lookup,
            doc_lookup=doc_lookup,
            config=config,
        )

    # ------------------------------------------------------------------
    # Bucket ownership (sparse-peer-id safe consistent hashing)
    # ------------------------------------------------------------------

    def bucket_owners(self, table_id: int, bucket_id: BucketId) -> list[int]:
        """Return the peer ids that own ``(table_id, bucket_id)``.

        ``owner_idx = sha1(table_id || repr(bucket_id)) % len(sorted_peer_ids)``,
        then resolved through ``sorted_peer_ids`` to handle sparse dataset
        user ids. Replication takes the next ``bucket_replication - 1``
        adjacent indices with wrap-around.
        """

        digest = hashlib.sha1()
        digest.update(table_id.to_bytes(4, "big", signed=False))
        # repr() of a tuple of ints is deterministic across runs.
        digest.update(repr(bucket_id).encode("ascii"))
        primary = int.from_bytes(digest.digest(), "big") % len(self.sorted_peer_ids)
        owners: list[int] = []
        for offset in range(self.config.bucket_replication):
            idx = (primary + offset) % len(self.sorted_peer_ids)
            owners.append(self.sorted_peer_ids[idx])
        return owners

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def seed_peer_docs(self, mapping: dict[int, list[str] | tuple[str, ...] | set[str]]) -> dict[str, int]:
        """Assign starting docs to peers and publish them into the index."""

        seeded_peer_count = 0
        seeded_doc_assignments = 0
        missing_doc_ids: set[str] = set()

        for peer_id, doc_ids in mapping.items():
            peer = self.peers.get(peer_id)
            if peer is None:
                continue
            seeded_any = False
            for doc_id in doc_ids:
                if doc_id not in self.doc_lookup:
                    missing_doc_ids.add(doc_id)
                    continue
                self._store_clicked_doc(peer_id, doc_id)
                seeded_doc_assignments += 1
                seeded_any = True
            if seeded_any:
                seeded_peer_count += 1

        return {
            "seeded_peer_count": seeded_peer_count,
            "seeded_doc_assignments": seeded_doc_assignments,
            "seeded_missing_doc_count": len(missing_doc_ids),
        }

    def _publish_doc(self, peer_id: int, doc_id: str, embedding: np.ndarray) -> None:
        """Publish ``doc_id`` owned by ``peer_id`` into all L LSH tables."""

        bucket_ids = self.lsh.hash_tables(embedding)
        replication = self.config.bucket_replication
        for table_id, bucket_id in enumerate(bucket_ids):
            # In a real distributed system each replica owner would receive
            # a message; we account for all R owner-side messages for the
            # coordination metric even though the in-memory bucket is a
            # single shared reference.
            self.coord.publish_messages += replication
            key = (table_id, bucket_id)
            bucket = self.buckets.get(key)
            if bucket is None:
                bucket = {}
                self.buckets[key] = bucket
            owner_set = bucket.get(doc_id)
            if owner_set is None:
                bucket[doc_id] = {peer_id}
            else:
                owner_set.add(peer_id)
        self.coord.docs_published += 1
        self._published_doc_ids.add(doc_id)
        self.coord.bucket_count_distinct = len(self.buckets)

    def _withdraw_doc(self, peer_id: int, doc_id: str, embedding: np.ndarray) -> None:
        """Remove a (peer_id, doc_id) entry after FIFO eviction.

        We re-hash the doc embedding to find which buckets to scrub. This
        keeps bucket bookkeeping faithful even though the published
        ownership entries themselves do not record their L bucket ids.
        """

        bucket_ids = self.lsh.hash_tables(embedding)
        for table_id, bucket_id in enumerate(bucket_ids):
            key = (table_id, bucket_id)
            bucket = self.buckets.get(key)
            if bucket is None:
                continue
            owners = bucket.get(doc_id)
            if owners is None:
                continue
            owners.discard(peer_id)
            if not owners:
                del bucket[doc_id]
                if not bucket:
                    del self.buckets[key]
        self.coord.bucket_count_distinct = len(self.buckets)

    def _store_clicked_doc(self, peer_id: int, doc_id: str) -> None:
        """Mirror :meth:`ZeroRAGNetwork._store_clicked_doc`.

        Adds the doc to the peer (with FIFO eviction if capped), publishes
        it into L LSH tables, and withdraws any evicted doc from its
        published buckets so the candidate set stays consistent.
        """

        if doc_id not in self.doc_lookup:
            return
        embedding = self.doc_lookup.get(doc_id)
        owner_peer = self.peers[peer_id]
        # Capture the about-to-be-evicted embedding while we still have it.
        existing_evicted_embedding: np.ndarray | None = None
        if (
            self.config.max_owned_docs > 0
            and doc_id not in owner_peer.owned_docs
            and len(owner_peer.owned_docs) >= self.config.max_owned_docs
        ):
            # Peek at FIFO head without popping it; Peer.add_owned_doc will pop.
            evicted_doc_id_preview = next(iter(owner_peer.fifo_order))
            existing_evicted_embedding = owner_peer.owned_docs[evicted_doc_id_preview].copy()
            evicted_preview_id = evicted_doc_id_preview
        else:
            evicted_preview_id = None

        evicted_doc_id = owner_peer.add_owned_doc(doc_id, embedding)
        self.doc_owners[doc_id].add(peer_id)
        self._publish_doc(peer_id, doc_id, embedding)

        if evicted_doc_id is None:
            return
        # Defensive: ensure the eviction we predicted actually happened.
        if evicted_preview_id is not None and evicted_preview_id != evicted_doc_id:
            existing_evicted_embedding = None
        if existing_evicted_embedding is not None:
            self._withdraw_doc(peer_id, evicted_doc_id, existing_evicted_embedding)
        remaining_owners = self.doc_owners.get(evicted_doc_id)
        if remaining_owners is None:
            return
        remaining_owners.discard(peer_id)
        if not remaining_owners:
            del self.doc_owners[evicted_doc_id]

    # ------------------------------------------------------------------
    # Query resolution
    # ------------------------------------------------------------------

    def _is_runtime_searchable(self, querier_id: int, doc_id: str) -> bool:
        owners = self.doc_owners.get(doc_id)
        return bool(owners and any(owner_id != querier_id for owner_id in owners))

    def _relevant_doc_ids_for_event(self, event: QueryEvent) -> frozenset[str]:
        if self.config.retrieval_mode != "multi_relevant":
            return frozenset({event.clicked_doc_id})
        return self.config.relevant_doc_ids_by_qid.get(event.qid, frozenset())

    def _available_relevant_doc_ids(self, relevant_doc_ids: frozenset[str]) -> frozenset[str]:
        return frozenset(
            doc_id
            for doc_id in relevant_doc_ids
            if any(owner in self.peers for owner in self.doc_owners.get(doc_id, set()))
        )

    @property
    def distinct_docs_published(self) -> int:
        """Number of unique document ids ever published into the LSH index."""

        return len(self._published_doc_ids)

    def set_distinct_candidate_sample_event_indices(self, event_indices: Iterable[int]) -> None:
        """Restrict exact distinct-candidate counting to sampled event ids."""

        self._distinct_candidate_sample_event_indices = {int(event_id) for event_id in event_indices}

    def _should_count_distinct_candidates(self, event_index: int) -> bool:
        sample = self._distinct_candidate_sample_event_indices
        if sample is None:
            return True
        return event_index in sample

    def _resolve_query(
        self,
        *,
        querier_id: int,
        target_doc_id: str,
        query_embedding: np.ndarray,
        count_distinct_candidates: bool,
    ) -> LSHSearchResult:
        """Hash query, gather candidates, and decide success.

        Cycles convention (matches :mod:`src.routing`):
        - cycle 1 = the querier itself (it computes ``g_j(q)`` locally and
          aggregates candidates);
        - then add 1 per distinct non-querier bucket-owner peer contacted.
        """

        bucket_ids = self.lsh.hash_tables(query_embedding)
        contacted_owners: set[int] = set()
        # Track presence-of-target without materializing the full candidate
        # union: for each bucket, look up the target doc_id directly and
        # check whether at least one non-querier owner is registered.
        target_found = False
        any_non_local_seen = False
        any_bucket_nonempty = False
        candidate_size_sum = 0
        distinct_candidate_doc_ids: set[str] | None = set() if count_distinct_candidates else None
        for table_id, bucket_id in enumerate(bucket_ids):
            for owner_id in self.bucket_owners(table_id, bucket_id):
                contacted_owners.add(owner_id)
            bucket = self.buckets.get((table_id, bucket_id))
            if not bucket:
                continue
            any_bucket_nonempty = True
            # Approximate candidate count as the sum of bucket sizes
            # (overcounting docs that appear in multiple matched buckets,
            # which is a well-known LSH bookkeeping convention).
            candidate_size_sum += len(bucket)
            if distinct_candidate_doc_ids is not None:
                distinct_candidate_doc_ids.update(bucket)
            if not target_found:
                doc_owners = bucket.get(target_doc_id)
                if doc_owners and (
                    len(doc_owners) > 1 or next(iter(doc_owners)) != querier_id
                ):
                    target_found = True
            if not any_non_local_seen:
                # Cheap-ish "did any bucket return non-local content?" proxy
                # used only to distinguish termination_reason=only_querier_owns
                # from not_in_buckets when the target isn't found.
                for owners in bucket.values():
                    if any(owner_id != querier_id for owner_id in owners):
                        any_non_local_seen = True
                        break

        non_self_contacts = sorted(owner for owner in contacted_owners if owner != querier_id)
        cycles_used = 1 + len(non_self_contacts)

        self.coord.queries_executed += 1
        self.coord.query_contacts += len(non_self_contacts)
        self.coord.candidate_total += candidate_size_sum
        distinct_candidate_doc_count = None
        if distinct_candidate_doc_ids is not None:
            distinct_candidate_doc_count = len(distinct_candidate_doc_ids)
            self.coord.distinct_candidate_doc_total += distinct_candidate_doc_count
            self.coord.distinct_candidate_doc_samples += 1

        if target_found:
            return LSHSearchResult(
                cycles_used=cycles_used,
                result_symbol="+",
                termination_reason="found",
                query_contacts=len(non_self_contacts),
                candidate_count=candidate_size_sum,
                distinct_candidate_doc_count=distinct_candidate_doc_count,
                contacted_peer_ids=non_self_contacts,
            )

        if any_bucket_nonempty and not any_non_local_seen:
            termination_reason = "only_querier_owns"
        else:
            termination_reason = "not_in_buckets"
        return LSHSearchResult(
            cycles_used=cycles_used,
            result_symbol="-",
            termination_reason=termination_reason,
            query_contacts=len(non_self_contacts),
            candidate_count=candidate_size_sum,
            distinct_candidate_doc_count=distinct_candidate_doc_count,
            contacted_peer_ids=non_self_contacts,
        )

    def _resolve_multi_relevant_query(
        self,
        *,
        querier_id: int,
        relevant_doc_ids: frozenset[str],
        query_embedding: np.ndarray,
        count_distinct_candidates: bool,
    ) -> LSHSearchResult:
        bucket_ids = self.lsh.hash_tables(query_embedding)
        contacted_owners: set[int] = set()
        any_bucket_nonempty = False
        candidate_size_sum = 0
        retrieved: set[str] = set()
        distinct_candidate_doc_ids: set[str] | None = set() if count_distinct_candidates else None
        for table_id, bucket_id in enumerate(bucket_ids):
            for owner_id in self.bucket_owners(table_id, bucket_id):
                contacted_owners.add(owner_id)
            bucket = self.buckets.get((table_id, bucket_id))
            if not bucket:
                continue
            any_bucket_nonempty = True
            candidate_size_sum += len(bucket)
            if distinct_candidate_doc_ids is not None:
                distinct_candidate_doc_ids.update(bucket)
            for doc_id in relevant_doc_ids.intersection(bucket):
                owners = bucket.get(doc_id)
                if owners and any(owner in self.peers for owner in owners):
                    retrieved.add(doc_id)

        non_self_contacts = sorted(owner for owner in contacted_owners if owner != querier_id)
        cycles_used = 1 + len(non_self_contacts)
        self.coord.queries_executed += 1
        self.coord.query_contacts += len(non_self_contacts)
        self.coord.candidate_total += candidate_size_sum
        distinct_candidate_doc_count = None
        if distinct_candidate_doc_ids is not None:
            distinct_candidate_doc_count = len(distinct_candidate_doc_ids)
            self.coord.distinct_candidate_doc_total += distinct_candidate_doc_count
            self.coord.distinct_candidate_doc_samples += 1

        if retrieved:
            return LSHSearchResult(
                cycles_used=cycles_used,
                result_symbol="+",
                termination_reason="found",
                query_contacts=len(non_self_contacts),
                candidate_count=candidate_size_sum,
                distinct_candidate_doc_count=distinct_candidate_doc_count,
                contacted_peer_ids=non_self_contacts,
                retrieved_relevant_doc_ids=tuple(sorted(retrieved)),
            )
        return LSHSearchResult(
            cycles_used=cycles_used,
            result_symbol="-",
            termination_reason="not_in_buckets" if any_bucket_nonempty else "not_in_buckets",
            query_contacts=len(non_self_contacts),
            candidate_count=candidate_size_sum,
            distinct_candidate_doc_count=distinct_candidate_doc_count,
            contacted_peer_ids=non_self_contacts,
            retrieved_relevant_doc_ids=(),
        )

    # ------------------------------------------------------------------
    # Event loop (mirrors ZeroRAGNetwork.process_event)
    # ------------------------------------------------------------------

    def process_event(self, event: QueryEvent, metrics: MetricsTracker) -> None:
        """Apply the same skip-reason and FIFO-refresh rules as semantic.

        The branching here is intentionally a near-copy of
        :meth:`src.network.ZeroRAGNetwork.process_event` so the
        ``skipped_by_reason`` distributions of the two algorithms are
        directly comparable.
        """

        scenario = self.config.scenario
        is_multi_relevant = self.config.retrieval_mode == "multi_relevant"
        relevant_doc_ids = self._relevant_doc_ids_for_event(event)
        available_relevant_doc_ids = self._available_relevant_doc_ids(relevant_doc_ids)
        if is_multi_relevant and not available_relevant_doc_ids:
            metrics.note_skipped_query("non_searchable")
            self.coord.queries_skipped += 1
            return
        if scenario in RUNTIME_SEARCHABLE_SCENARIOS:
            if self.peers[event.user_id].has_doc(event.clicked_doc_id):
                metrics.note_skipped_query("local_held")
                self.coord.queries_skipped += 1
                self._store_clicked_doc(event.user_id, event.clicked_doc_id)
                return

            if event.query_text not in self.query_lookup:
                metrics.note_skipped_query("missing_query_embedding")
                self.coord.queries_skipped += 1
                self._store_clicked_doc(event.user_id, event.clicked_doc_id)
                return

            if not self._is_runtime_searchable(event.user_id, event.clicked_doc_id):
                metrics.note_skipped_query("non_searchable")
                self.coord.queries_skipped += 1
                self._store_clicked_doc(event.user_id, event.clicked_doc_id)
                return
        elif not event.searchable and not is_multi_relevant:
            metrics.note_skipped_query("non_searchable")
            self.coord.queries_skipped += 1
            self._store_clicked_doc(event.user_id, event.clicked_doc_id)
            return

        if event.query_text not in self.query_lookup:
            metrics.note_skipped_query("missing_query_embedding")
            self.coord.queries_skipped += 1
            if not is_multi_relevant:
                self._store_clicked_doc(event.user_id, event.clicked_doc_id)
            return

        if (
            not is_multi_relevant
            and
            scenario == "empty_start_chronological"
            and self.peers[event.user_id].has_doc(event.clicked_doc_id)
        ):
            metrics.note_skipped_query("local_held")
            self.coord.queries_skipped += 1
            self._store_clicked_doc(event.user_id, event.clicked_doc_id)
            return

        query_embedding = self.query_lookup.get(event.query_text)
        if is_multi_relevant:
            search_result = self._resolve_multi_relevant_query(
                querier_id=event.user_id,
                relevant_doc_ids=available_relevant_doc_ids,
                query_embedding=query_embedding,
                count_distinct_candidates=self._should_count_distinct_candidates(event.event_index),
            )
        else:
            search_result = self._resolve_query(
                querier_id=event.user_id,
                target_doc_id=event.clicked_doc_id,
                query_embedding=query_embedding,
                count_distinct_candidates=self._should_count_distinct_candidates(event.event_index),
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
                requests_until_first_relevant=None,
            )
        )

        if not is_multi_relevant:
            # Acquire the clicked doc only after the search attempt completes,
            # exactly as in the semantic baseline.
            self._store_clicked_doc(event.user_id, event.clicked_doc_id)


# Convenience: tests want to reach into FIFO_SCENARIOS / SEEDED_SCENARIOS
# without depending on ``src.scenarios`` directly.
__all__ = [
    "FIFO_SCENARIOS",
    "LSHCoordinationCounters",
    "LSHDHTNetwork",
    "LSHSearchResult",
    "LSHSimulationConfig",
    "SEEDED_SCENARIOS",
]

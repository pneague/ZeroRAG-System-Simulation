from __future__ import annotations

from collections import OrderedDict, deque
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

REFERRAL_TIE_TOLERANCE = 1e-12


@dataclass(slots=True)
class DocumentCatalog:
    doc_ids: list[str] = field(default_factory=list)
    row_by_doc_id: dict[str, int] = field(default_factory=dict)
    vectors: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=np.float32))
    size: int = 0

    def upsert(self, doc_id: str, embedding: np.ndarray) -> None:
        vector = np.asarray(embedding, dtype=np.float32)
        existing_idx = self.row_by_doc_id.get(doc_id)
        if existing_idx is not None:
            self.vectors[existing_idx] = vector
            return

        self._ensure_capacity(vector.shape[0])
        row_idx = self.size
        self.doc_ids.append(doc_id)
        self.row_by_doc_id[doc_id] = row_idx
        self.vectors[row_idx] = vector
        self.size += 1

    def best_referral(self, *, peer_id: int, query_embedding: np.ndarray, source: str) -> Referral | None:
        if self.size == 0:
            return None
        similarities = self.vectors[: self.size] @ query_embedding
        best_idx = int(np.argmax(similarities))
        return Referral(
            peer_id=peer_id,
            closest_doc_id=self.doc_ids[best_idx],
            similarity=float(similarities[best_idx]),
            source=source,
        )

    def remove(self, doc_id: str) -> None:
        row_idx = self.row_by_doc_id.pop(doc_id, None)
        if row_idx is None:
            return

        last_idx = self.size - 1
        last_doc_id = self.doc_ids[last_idx]
        if row_idx != last_idx:
            self.doc_ids[row_idx] = last_doc_id
            self.vectors[row_idx] = self.vectors[last_idx]
            self.row_by_doc_id[last_doc_id] = row_idx
        self.doc_ids.pop()
        self.size -= 1

    def _ensure_capacity(self, embedding_dim: int) -> None:
        if self.vectors.size == 0:
            initial_capacity = max(4, self.size + 1)
            self.vectors = np.empty((initial_capacity, embedding_dim), dtype=np.float32)
            return
        if self.size < self.vectors.shape[0]:
            return
        new_capacity = max(self.vectors.shape[0] * 2, self.size + 1)
        expanded = np.empty((new_capacity, self.vectors.shape[1]), dtype=np.float32)
        if self.size:
            expanded[: self.size] = self.vectors[: self.size]
        self.vectors = expanded


@dataclass(slots=True)
class VectorComparisonCounts:
    local_doc_vector_comparisons: int = 0
    neighbor_doc_vector_comparisons: int = 0
    routed_query_memory_comparisons: int = 0

    @property
    def doc_vector_comparisons(self) -> int:
        return self.local_doc_vector_comparisons + self.neighbor_doc_vector_comparisons

    @property
    def total_vector_comparisons(self) -> int:
        return self.doc_vector_comparisons + self.routed_query_memory_comparisons


@dataclass(slots=True)
class RoutedQuery:
    """Compact routed-query memory entry.

    Stores only an integer index into a global query embedding table rather than
    a copied embedding vector. With ``sentence-transformers/all-mpnet-base-v2``
    that is roughly ``28 bytes`` (CPython int) versus ``~3 KB`` (768-d float32
    + ndarray overhead) per entry, which dominates Peer memory at the
    ``hundreds of thousands of peers x max_routed_queries`` scale we hit on the
    Amazon ``all`` dataset. The actual embedding is fetched on demand via the
    resolver passed to :meth:`Peer._best_routed_query_referrals`.
    """

    query_index: int
    original_peer_id: int
    forwarded_to_peer_id: int | None
    clicked_doc_id: str


@dataclass(slots=True)
class Referral:
    peer_id: int
    closest_doc_id: str
    similarity: float
    source: str


@dataclass(slots=True)
class PeerResponse:
    queried_peer_id: int
    found: bool
    found_doc_id: str | None
    referrals: list[Referral]
    found_relevant_doc_ids: tuple[str, ...] = ()


@dataclass(slots=True)
class PeerQueryComputation:
    response: PeerResponse
    routed_query: RoutedQuery | None


def _similarity_groups(referrals: list[Referral]) -> list[list[Referral]]:
    groups: list[list[Referral]] = []
    for referral in referrals:
        if not groups or (
            abs(groups[-1][0].similarity - referral.similarity) > REFERRAL_TIE_TOLERANCE
        ):
            groups.append([referral])
            continue
        groups[-1].append(referral)
    return groups


def _shuffled_group(
    referrals: list[Referral],
    rng: np.random.Generator,
) -> list[Referral]:
    order = rng.permutation(len(referrals))
    return [referrals[int(index)] for index in order.tolist()]


@dataclass(slots=True)
class Peer:
    peer_id: int
    max_neighbors: int = 60
    max_routed_queries: int = 1000
    max_owned_docs: int = 0
    owned_docs: dict[str, np.ndarray] = field(default_factory=dict)
    owned_doc_catalog: DocumentCatalog = field(default_factory=DocumentCatalog)
    fifo_order: OrderedDict[str, None] = field(default_factory=OrderedDict)
    connections: deque[int] = field(default_factory=deque)
    routed_queries: list[RoutedQuery] = field(default_factory=list)
    query_counter: int = 0

    def __post_init__(self) -> None:
        self.connections = deque(self.connections, maxlen=self.max_neighbors)

    def has_doc(self, doc_id: str) -> bool:
        return doc_id in self.owned_docs

    def add_owned_doc(self, doc_id: str, embedding: np.ndarray) -> str | None:
        vector = np.asarray(embedding, dtype=np.float32)
        if doc_id in self.owned_docs:
            self.owned_docs[doc_id] = vector
            self.owned_doc_catalog.upsert(doc_id, vector)
            self.fifo_order[doc_id] = None
            self.fifo_order.move_to_end(doc_id)
            return None

        self.owned_docs[doc_id] = vector
        self.owned_doc_catalog.upsert(doc_id, vector)
        self.fifo_order[doc_id] = None
        if self.max_owned_docs > 0 and len(self.owned_docs) > self.max_owned_docs:
            evicted_doc_id, _ = self.fifo_order.popitem(last=False)
            del self.owned_docs[evicted_doc_id]
            self.owned_doc_catalog.remove(evicted_doc_id)
            return evicted_doc_id
        return None

    def add_connection(self, neighbor_id: int) -> int | None:
        if neighbor_id == self.peer_id or neighbor_id in self.connections:
            return None
        evicted: int | None = None
        if len(self.connections) >= self.max_neighbors:
            evicted = self.connections.popleft()
        self.connections.append(neighbor_id)
        self.query_counter += 1
        return evicted

    def remove_connection(self, neighbor_id: int) -> None:
        if neighbor_id not in self.connections:
            return
        filtered = [peer_id for peer_id in self.connections if peer_id != neighbor_id]
        self.connections = deque(filtered, maxlen=self.max_neighbors)

    def best_owned_doc_referral(
        self,
        query_embedding: np.ndarray,
        *,
        source: str,
        comparison_counts: VectorComparisonCounts | None = None,
    ) -> Referral | None:
        if comparison_counts is not None:
            comparison_counts.neighbor_doc_vector_comparisons += self.owned_doc_catalog.size
        return self.owned_doc_catalog.best_referral(
            peer_id=self.peer_id,
            query_embedding=query_embedding,
            source=source,
        )

    def _best_routed_query_referrals(
        self,
        query_embedding: np.ndarray,
        *,
        similarity_threshold: float,
        resolve_query_embedding: Callable[[int], np.ndarray],
        comparison_counts: VectorComparisonCounts | None = None,
    ) -> list[Referral]:
        referrals: list[Referral] = []
        for routed_query in self.routed_queries:
            if comparison_counts is not None:
                comparison_counts.routed_query_memory_comparisons += 1
            stored_embedding = resolve_query_embedding(routed_query.query_index)
            similarity = float(np.dot(stored_embedding, query_embedding))
            if similarity < similarity_threshold:
                continue
            referrals.append(
                Referral(
                    peer_id=routed_query.original_peer_id,
                    closest_doc_id=routed_query.clicked_doc_id,
                    similarity=similarity,
                    source="routed_query",
                )
            )
        return referrals

    def _merge_referrals(
        self,
        referrals: list[Referral],
        *,
        limit: int,
        exclude_peer_id: int | None = None,
        rng: np.random.Generator | None = None,
    ) -> list[Referral]:
        best_by_peer: dict[int, Referral] = {}
        for referral in referrals:
            if exclude_peer_id is not None and referral.peer_id == exclude_peer_id:
                continue
            previous = best_by_peer.get(referral.peer_id)
            if previous is None or referral.similarity > previous.similarity:
                best_by_peer[referral.peer_id] = referral
        ranked = sorted(best_by_peer.values(), key=lambda item: (-item.similarity, item.peer_id))
        selected: list[Referral] = []
        for group in _similarity_groups(ranked):
            remaining = limit - len(selected)
            if remaining <= 0:
                break
            if len(group) <= remaining:
                if rng is not None and len(group) > 1:
                    group = _shuffled_group(group, rng)
                selected.extend(group)
                continue
            if rng is None:
                selected.extend(group[:remaining])
            else:
                indices = rng.choice(len(group), size=remaining, replace=False)
                selected.extend(group[int(index)] for index in indices.tolist())
            break
        return selected

    def respond_to_query(
        self,
        *,
        query_embedding: np.ndarray,
        query_index: int,
        target_doc_id: str,
        original_peer_id: int,
        referral_limit: int,
        similarity_threshold: float,
        resolve_query_embedding: Callable[[int], np.ndarray],
        replica_referrals: list[Referral] | None = None,
        allow_local_hit: bool = True,
        use_routed_query_memory: bool = True,
        comparison_counts: VectorComparisonCounts | None = None,
        relevant_doc_ids: frozenset[str] | set[str] | None = None,
        referral_rng: np.random.Generator | None = None,
    ) -> PeerResponse:
        computation = self.compute_query_response(
            query_embedding=query_embedding,
            query_index=query_index,
            target_doc_id=target_doc_id,
            original_peer_id=original_peer_id,
            referral_limit=referral_limit,
            similarity_threshold=similarity_threshold,
            resolve_query_embedding=resolve_query_embedding,
            replica_referrals=replica_referrals,
            allow_local_hit=allow_local_hit,
            use_routed_query_memory=use_routed_query_memory,
            comparison_counts=comparison_counts,
            relevant_doc_ids=relevant_doc_ids,
            referral_rng=referral_rng,
        )
        if computation.routed_query is not None:
            self.record_routed_query(computation.routed_query)
        return computation.response

    def compute_query_response(
        self,
        *,
        query_embedding: np.ndarray,
        query_index: int,
        target_doc_id: str,
        original_peer_id: int,
        referral_limit: int,
        similarity_threshold: float,
        resolve_query_embedding: Callable[[int], np.ndarray],
        replica_referrals: list[Referral] | None = None,
        allow_local_hit: bool = True,
        use_routed_query_memory: bool = True,
        comparison_counts: VectorComparisonCounts | None = None,
        relevant_doc_ids: frozenset[str] | set[str] | None = None,
        referral_rng: np.random.Generator | None = None,
    ) -> PeerQueryComputation:
        if allow_local_hit and comparison_counts is not None:
            comparison_counts.local_doc_vector_comparisons += self.owned_doc_catalog.size
        found = allow_local_hit and self.has_doc(target_doc_id)
        found_relevant_doc_ids: tuple[str, ...] = ()
        if allow_local_hit and relevant_doc_ids:
            found_relevant_doc_ids = tuple(
                sorted(doc_id for doc_id in relevant_doc_ids if doc_id in self.owned_docs)
            )
        replica_referrals = replica_referrals or []
        effective_use_routed_query_memory = use_routed_query_memory and self.max_routed_queries > 0
        routed_referrals: list[Referral] = []
        if effective_use_routed_query_memory:
            routed_referrals = self._best_routed_query_referrals(
                query_embedding,
                similarity_threshold=similarity_threshold,
                resolve_query_embedding=resolve_query_embedding,
                comparison_counts=comparison_counts,
            )
        referrals = self._merge_referrals(
            replica_referrals + routed_referrals,
            limit=referral_limit,
            exclude_peer_id=original_peer_id,
            rng=referral_rng,
        )
        suggested_peer_id = referrals[0].peer_id if referrals else None
        routed_query = None
        if effective_use_routed_query_memory:
            routed_query = RoutedQuery(
                query_index=query_index,
                original_peer_id=original_peer_id,
                forwarded_to_peer_id=suggested_peer_id,
                clicked_doc_id=target_doc_id,
            )
        return PeerQueryComputation(
            response=PeerResponse(
                queried_peer_id=self.peer_id,
                found=found,
                found_doc_id=target_doc_id if found else None,
                referrals=referrals,
                found_relevant_doc_ids=found_relevant_doc_ids,
            ),
            routed_query=routed_query,
        )

    def record_routed_query(self, routed_query: RoutedQuery) -> None:
        if self.max_routed_queries <= 0:
            return
        self.routed_queries.append(routed_query)
        if len(self.routed_queries) > self.max_routed_queries:
            self.routed_queries = self.routed_queries[-self.max_routed_queries :]


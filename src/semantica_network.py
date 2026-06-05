"""Semantica (arXiv:2502.10151) baseline as a third sibling network class.

Sibling of :mod:`src.network` and :mod:`src.lsh_network`. Same dataset/event
surface, same skip reasons, same ``cycles_used`` semantics (querier counted as
cycle 1, then add the number of distinct non-querier peers contacted), but the
routing layer is replaced by the paper-faithful **Closest-User-Chain** walk
over a binary K-means trie of per-peer average doc embeddings.

Build phase (sequential, runs once at construction):

* Compute the L2-normalized average doc embedding per peer from the seeded
  inventory (the ``df_avgemb`` of the upstream
  ``Construct_avg_embedding.py``).
* Build a binary K-means trie per ``Construct_tree_verified_version.py``:
  split a leaf when it exceeds ``record_limit_per_leafnode``; duplicate
  boundary records when ``|d_left - d_right| < DELTA``.
* For each peer derive a static ``neighbors`` set = union of peers in every
  leaf the peer was placed in (clones from boundary duplication included),
  minus the peer itself (the "co-leaves union" rule of plan-of-record).

Query phase (per event; pure function over the static state):

* Closest-User-Chain: starting at the querier, hop to the unvisited neighbor
  whose average embedding has the highest cosine similarity to the query
  embedding. Stop on ``found`` (a non-querier neighbor owns the target),
  ``exhausted`` (no unvisited neighbor) or ``cap`` (``cycles_used == max_cycles``).

Cycle / result / termination conventions are kept byte-shape-compatible with
:mod:`src.network` and :mod:`src.lsh_network` so cross-algorithm overlays work
without modifying ``MetricsTracker`` or ``query_log.jsonl``.

Scope is restricted to the ``seeded_randomized_timeline`` scenario: peer
ownership is fixed at startup, the trie is built once, and the per-event
chain walk is a pure read-only function of the snapshot. That immutability
is what unlocks the 60-way parallel querying in
:mod:`experiments.run_semantica_simulation`.

The k=2 K-means used for the trie splits is implemented inline (Lloyd's
algorithm with deterministic init) so this module does not introduce a
``scikit-learn`` dependency on the existing virtualenvs. The math is
identical to ``sklearn.cluster.KMeans(n_clusters=2, random_state=...)`` for
the inputs we feed it, with deterministic tie-breaking documented in
:meth:`SemanticaTrie._kmeans2`.
"""

from __future__ import annotations

import json
import multiprocessing as mp
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from src.dataset import QueryEvent
from src.metrics import MetricsTracker, QueryLog
from src.network import EmbeddingLookup
from src.peer import Peer
from src.scenarios import SEEDED_SCENARIOS


SUPPORTED_STRATEGIES: frozenset[str] = frozenset({"closest_user_chain"})
DEFAULT_STRATEGY = "closest_user_chain"
SUPPORTED_SCENARIOS: frozenset[str] = SEEDED_SCENARIOS | frozenset({"empty_start_chronological"})
# Pathological seeded inventories could in principle build a very deep trie;
# guard with a hard cap to fail loud rather than recurse forever.
MAX_TRIE_DEPTH = 64


@dataclass(slots=True)
class SemanticaSimulationConfig:
    """All knobs needed to construct the Semantica trie + chain walk.

    Defaults mirror the chain-hop retrieval experiment of the paper
    (Section VI.B of arXiv:2502.10151):

    * ``record_limit_per_leafnode = 50`` (paper symbol ``L``).
    * ``delta = 0.003`` (paper symbol ``\\Delta``; Sec VI.B explicitly
      uses ``\\Delta = 0.003``).
    * ``ncu = 60`` (size of the per-peer ``closest-users`` list used by
      the chain-hop walk; paper symbol ``n_{cu}``).
    * ``expansion_rounds = 10`` (paper symbol ``r_{max}``; Sec VI.B
      explicitly uses 10 rounds).
    * ``max_cycles = 50`` is the same hop budget exposed by
      :mod:`src.network` and :mod:`src.lsh_network` (``50`` = querier
      counted as cycle 1 + at most 49 forwards).

    ``strategy`` is currently restricted to ``closest_user_chain``;
    the field exists so the future ``query_as_node`` / ``closest_users``
    ablations slot in without breaking the ``run_manifest.json`` schema.
    """

    scenario: str = "seeded_randomized_timeline"
    record_limit_per_leafnode: int = 50
    delta: float = 0.003
    ncu: int = 60
    expansion_rounds: int = 10
    max_cycles: int = 50
    strategy: str = DEFAULT_STRATEGY
    metrics_window: int = 1000
    random_seed: int = 42
    max_owned_docs: int = 0
    retrieval_mode: str = "single_target"
    relevant_doc_ids_by_qid: dict[int, frozenset[str]] = field(default_factory=dict)


@dataclass(slots=True)
class SemanticaSearchResult:
    """Semantica analog of :class:`src.routing.SearchResult`.

    ``cycles_used`` counts the querier as cycle 1 and adds 1 per distinct
    non-querier peer contacted, identical to low-coord and LSH. ``result_symbol``
    and ``termination_reason`` use the low-coord vocabulary (``found``,
    ``exhausted``, ``cap``) so the existing plotters do not need a new branch.
    """

    cycles_used: int
    result_symbol: str
    termination_reason: str
    contacted_peer_ids: list[int]
    retrieved_relevant_doc_ids: tuple[str, ...] = ()
    requests_until_first_relevant: int | None = None


@dataclass(slots=True)
class SemanticaCoordinationCounters:
    """Aggregate algorithm-only counters reported in ``semantica_metrics.json``.

    Kept separate from :class:`src.metrics.MetricsTracker` so that the
    existing ``metrics.json`` schema (recall / avg_cycles / skip_rate)
    stays byte-shape compatible with the other two baselines' plotting.
    """

    queries_executed: int = 0
    queries_skipped: int = 0
    chain_hops_total: int = 0  # Sum of forwards (cycles_used - 1) across executed queries.
    chain_found_count: int = 0
    chain_exhausted_count: int = 0
    chain_cap_count: int = 0
    dead_end_at_querier_count: int = 0  # Querier had an empty neighbor list.
    querier_not_in_trie_count: int = 0  # Querier had no seeded docs and was not placed.
    docs_published: int = 0  # Seeded doc-peer assignments stored.
    # Build-phase counters (filled by build_trie):
    expansion_round_updates_total: int = 0  # Accepted updates across all expansion rounds.
    initial_known_users_avg: float = 0.0  # Mean co-leaves-union list size (pre-expansion).
    final_closest_users_avg: float = 0.0  # Mean closest-users (top-ncu) list size (post-expansion).


@dataclass(slots=True)
class _LeafRecord:
    """Per-peer entry in a trie leaf node.

    Boundary duplication during a split copies the record into both child
    leaves; ``leaf_ids`` therefore tracks every leaf this peer ended up in
    so the co-leaves-union neighbor rule can be derived after the build is
    done.
    """

    peer_id: int
    embedding: np.ndarray
    leaf_ids: list[int] = field(default_factory=list)


class SemanticaTrie:
    """Binary K-means trie over per-peer L2-normalized average embeddings.

    The build mirrors ``Construct_tree_verified_version.py`` in spirit:

    * Start with one leaf containing every record.
    * If a leaf exceeds ``record_limit_per_leafnode`` after a record is
      inserted, split it via 2-means; records with
      ``|d_left - d_right| < delta`` are duplicated to BOTH child leaves.
    * The resulting forest of leaves induces, for every peer, a "neighbor
      set" = union of co-leaves across every leaf the peer was placed in
      (clones included), minus the peer itself.

    The internal node ids are dense ``int`` (0..n_nodes); leaf ids are a
    subset of node ids. Stored shape is intentionally minimal (numpy
    arrays only) so the parallel driver can fork the parent without
    duplicating Python object graphs across workers.
    """

    def __init__(
        self,
        *,
        peer_ids: np.ndarray,
        peer_avg_emb: np.ndarray,
        record_limit_per_leafnode: int,
        delta: float,
        random_seed: int,
    ) -> None:
        if peer_ids.shape[0] != peer_avg_emb.shape[0]:
            raise ValueError(
                f"peer_ids ({peer_ids.shape[0]}) and peer_avg_emb "
                f"({peer_avg_emb.shape[0]}) must have matching first dim"
            )
        if record_limit_per_leafnode <= 1:
            raise ValueError(
                f"record_limit_per_leafnode must be >= 2, got {record_limit_per_leafnode}"
            )
        self.peer_ids = peer_ids
        self.peer_avg_emb = peer_avg_emb
        self.record_limit_per_leafnode = int(record_limit_per_leafnode)
        self.delta = float(delta)
        self.random_seed = int(random_seed)

        # Outputs of build():
        self.leaf_id_for_peer: dict[int, list[int]] = {}
        # leaf_id -> sorted list[peer_id] of peers in that leaf
        self.leaves: dict[int, list[int]] = {}
        self.max_depth: int = 0
        self.clone_count: int = 0
        self._next_node_id: int = 0

    @classmethod
    def build(
        cls,
        *,
        peer_ids: np.ndarray,
        peer_avg_emb: np.ndarray,
        record_limit_per_leafnode: int,
        delta: float,
        random_seed: int,
    ) -> "SemanticaTrie":
        trie = cls(
            peer_ids=peer_ids,
            peer_avg_emb=peer_avg_emb,
            record_limit_per_leafnode=record_limit_per_leafnode,
            delta=delta,
            random_seed=random_seed,
        )
        trie._build()
        return trie

    def _alloc_node(self) -> int:
        node_id = self._next_node_id
        self._next_node_id += 1
        return node_id

    def _build(self) -> None:
        n = self.peer_ids.shape[0]
        if n == 0:
            return

        # Each record is (peer_id_index, embedding_view, leaf_ids_list).
        # We reference rows by index so the leaf membership lists store ints.
        records: list[_LeafRecord] = [
            _LeafRecord(peer_id=int(self.peer_ids[i]), embedding=self.peer_avg_emb[i])
            for i in range(n)
        ]

        root_id = self._alloc_node()
        for rec in records:
            rec.leaf_ids.append(root_id)

        # Stack of (leaf_id, [records], depth) to split.
        # We push the root with all records and split iteratively.
        stack: list[tuple[int, list[_LeafRecord], int]] = [(root_id, records, 0)]
        # leaf_records is the authoritative leaf -> records map being built.
        leaf_records: dict[int, list[_LeafRecord]] = {}

        rng = np.random.default_rng(self.random_seed)

        while stack:
            leaf_id, leaf_recs, depth = stack.pop()
            if depth > self.max_depth:
                self.max_depth = depth
            if depth > MAX_TRIE_DEPTH:
                raise RuntimeError(
                    f"Semantica trie depth {depth} exceeded MAX_TRIE_DEPTH={MAX_TRIE_DEPTH}; "
                    "seeded inventory is pathological for the given record_limit_per_leafnode/delta."
                )

            if len(leaf_recs) <= self.record_limit_per_leafnode:
                leaf_records[leaf_id] = leaf_recs
                continue

            # Try to split. If 2-means can't actually separate the points
            # (e.g. all embeddings identical), keep the leaf as is to avoid
            # an infinite loop.
            embeddings = np.stack([rec.embedding for rec in leaf_recs], axis=0)
            labels, centroid_left, centroid_right = self._kmeans2(embeddings, rng=rng)

            left_records: list[_LeafRecord] = []
            right_records: list[_LeafRecord] = []
            left_id = self._alloc_node()
            right_id = self._alloc_node()
            for rec, lbl in zip(leaf_recs, labels):
                dist_left = float(np.linalg.norm(rec.embedding - centroid_left))
                dist_right = float(np.linalg.norm(rec.embedding - centroid_right))
                # Replace the parent leaf id with the child leaf id this
                # record now belongs to. Boundary duplication adds a SECOND
                # leaf id so the record lives in both children.
                rec.leaf_ids.remove(leaf_id)
                if abs(dist_left - dist_right) < self.delta:
                    # Clone record to both children; track for stats.
                    rec_left = _LeafRecord(
                        peer_id=rec.peer_id, embedding=rec.embedding, leaf_ids=list(rec.leaf_ids)
                    )
                    rec_right = _LeafRecord(
                        peer_id=rec.peer_id, embedding=rec.embedding, leaf_ids=list(rec.leaf_ids)
                    )
                    rec_left.leaf_ids.append(left_id)
                    rec_right.leaf_ids.append(right_id)
                    left_records.append(rec_left)
                    right_records.append(rec_right)
                    self.clone_count += 1
                else:
                    if lbl == 0:
                        rec.leaf_ids.append(left_id)
                        left_records.append(rec)
                    else:
                        rec.leaf_ids.append(right_id)
                        right_records.append(rec)

            # Edge case: 2-means failed to actually partition (all points in
            # one cluster). Stop splitting this leaf to avoid recursing
            # forever on identical embeddings. Records still keep the
            # parent-leaf id since we removed it above; restore it.
            if not left_records or not right_records:
                # Reattach to the original leaf; revert clone count too.
                merged = left_records + right_records
                # Undo clone bookkeeping that didn't end in a real split.
                undo_clones = sum(1 for rec in merged if leaf_id in rec.leaf_ids)
                # (Defensive; in practice the merged path means clone was
                # double-counted because both halves still point to the same leaf id.)
                for rec in merged:
                    if left_id in rec.leaf_ids:
                        rec.leaf_ids.remove(left_id)
                    if right_id in rec.leaf_ids:
                        rec.leaf_ids.remove(right_id)
                    if leaf_id not in rec.leaf_ids:
                        rec.leaf_ids.append(leaf_id)
                self.clone_count -= undo_clones if undo_clones > 0 else 0
                # Discard the freshly-allocated child node ids; deduping the
                # node-id space isn't worth the bookkeeping. The tree's
                # leaf_records map stays small either way.
                leaf_records[leaf_id] = merged
                continue

            stack.append((left_id, left_records, depth + 1))
            stack.append((right_id, right_records, depth + 1))

        # Materialize public outputs.
        # Sort each leaf's peer_id list so cosine-tie-breaking can rely on
        # ascending peer_id ordering at chain-walk time.
        for leaf_id, leaf_recs in leaf_records.items():
            sorted_peer_ids = sorted({rec.peer_id for rec in leaf_recs})
            self.leaves[leaf_id] = sorted_peer_ids
            for peer_id in sorted_peer_ids:
                self.leaf_id_for_peer.setdefault(peer_id, []).append(leaf_id)

    @staticmethod
    def _kmeans2(
        embeddings: np.ndarray, *, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Lloyd's algorithm with k=2.

        Deterministic given ``rng`` and the input. Initial centroids are
        drawn at random from the input rows (k-means style). Iterates until
        labels stabilize or 50 iterations elapse. Returns (labels, centroid0,
        centroid1) where ``labels[i] in {0, 1}``.

        If the two initial centroids are identical (degenerate input), we
        return all-zero labels with both centroids equal to the mean; the
        caller detects this as a no-op split and stops splitting that leaf.
        """

        n = embeddings.shape[0]
        if n < 2:
            mean = embeddings.mean(axis=0) if n > 0 else np.zeros(embeddings.shape[1:])
            return np.zeros(n, dtype=np.int8), mean.astype(np.float32), mean.astype(np.float32)

        # Random initial seeds without replacement.
        idx = rng.choice(n, size=2, replace=False)
        c0 = embeddings[idx[0]].astype(np.float32, copy=True)
        c1 = embeddings[idx[1]].astype(np.float32, copy=True)

        labels = np.zeros(n, dtype=np.int8)
        for _ in range(50):
            d0 = np.linalg.norm(embeddings - c0, axis=1)
            d1 = np.linalg.norm(embeddings - c1, axis=1)
            new_labels = (d1 < d0).astype(np.int8)
            if np.array_equal(new_labels, labels):
                break
            labels = new_labels
            mask0 = labels == 0
            mask1 = ~mask0
            if mask0.any():
                c0 = embeddings[mask0].mean(axis=0).astype(np.float32)
            if mask1.any():
                c1 = embeddings[mask1].mean(axis=0).astype(np.float32)
        return labels, c0, c1


@dataclass(slots=True)
class SemanticaSnapshot:
    """Immutable read-only view of the trie + ownership state.

    Designed to be picklable and forkable so :class:`SemanticaParallelDriver`
    can ship one snapshot to every worker process without rebuilding the
    trie. Workers run :func:`run_closest_user_chain` against this snapshot
    only and never mutate it.
    """

    peer_index: dict[int, int]
    """``peer_id -> row index`` into ``peer_avg_emb``. Querier and visited
    peers are looked up by id and resolved through this index."""

    peer_avg_emb: np.ndarray
    """``(P, d)`` float32 matrix; row order matches ``peer_index``."""

    neighbors: dict[int, np.ndarray]
    """``peer_id -> sorted np.int32 array of co-leaves (excluding self)``."""

    doc_owners: dict[str, frozenset[int]]
    """``doc_id -> frozenset of peer_id owners`` (seeded inventory)."""

    max_cycles: int
    """Hop budget: ``cycles_used <= max_cycles``."""


def run_closest_user_chain(
    snapshot: SemanticaSnapshot,
    *,
    querier_id: int,
    target_doc_id: str,
    query_embedding: np.ndarray,
    relevant_doc_ids: frozenset[str] | None = None,
) -> SemanticaSearchResult:
    """Pure-function Closest-User-Chain walk.

    Reads only from ``snapshot`` and does not import :mod:`src.metrics`,
    making it safe to call from a forked worker. ``cycles_used`` follows the
    "querier counts as cycle 1, + distinct non-querier peers contacted" rule
    used by both :mod:`src.routing` and :mod:`src.lsh_network`.

    Tie-breaking: cosine ties (rare with float32 dot products on normalized
    inputs but possible with quantized embeddings) are broken by lowest
    ``peer_id``. Achieved by sorting each leaf's peer-id list ascending at
    build time and using ``np.argmax`` on the score array, which returns
    the first occurrence in the case of ties.
    """

    contacted: set[int] = {querier_id}
    non_self: list[int] = []
    current = querier_id
    max_cycles = snapshot.max_cycles
    retrieved: set[str] = set()
    requests_until_first: int | None = None

    def make_result(cycles_used: int, termination_reason: str) -> SemanticaSearchResult:
        return SemanticaSearchResult(
            cycles_used=cycles_used,
            result_symbol="+" if retrieved else "-",
            termination_reason=termination_reason,
            contacted_peer_ids=list(non_self),
            retrieved_relevant_doc_ids=tuple(sorted(retrieved)),
            requests_until_first_relevant=requests_until_first,
        )

    while True:
        # cycles_used after one more forward = 1 + len(non_self) + 1.
        # Stop when that would exceed the budget.
        if 1 + len(non_self) >= max_cycles:
            return make_result(max_cycles, "cap")

        candidates_arr = snapshot.neighbors.get(current)
        if candidates_arr is None or candidates_arr.size == 0:
            return make_result(1 + len(non_self), "exhausted")

        # Filter out already-contacted peers. This is hot path; do it with
        # a numpy mask rather than a python loop where possible.
        if contacted:
            mask = np.array(
                [pid not in contacted for pid in candidates_arr.tolist()], dtype=bool
            )
            unvisited = candidates_arr[mask]
        else:
            unvisited = candidates_arr
        if unvisited.size == 0:
            return make_result(1 + len(non_self), "exhausted")

        # Cosine = dot for L2-normalized vectors.
        rows = np.array(
            [snapshot.peer_index[int(pid)] for pid in unvisited.tolist()], dtype=np.int64
        )
        scores = snapshot.peer_avg_emb[rows] @ query_embedding
        best = int(np.argmax(scores))
        next_id = int(unvisited[best])

        # Querier-never-counts-local check: only count owners that are not
        # the querier (matches src/network.py and src/lsh_network.py).
        if next_id != querier_id:
            if relevant_doc_ids is not None:
                non_self.append(next_id)
                found_docs = {
                    doc_id
                    for doc_id in relevant_doc_ids
                    if next_id in snapshot.doc_owners.get(doc_id, frozenset())
                }
                if found_docs:
                    if requests_until_first is None:
                        requests_until_first = len(non_self)
                    retrieved.update(found_docs)
                contacted.add(next_id)
                current = next_id
                continue
            owners = snapshot.doc_owners.get(target_doc_id)
            if owners is not None and next_id in owners:
                non_self.append(next_id)
                retrieved.add(target_doc_id)
                requests_until_first = len(non_self)
                return make_result(1 + len(non_self), "found")

        non_self.append(next_id)
        contacted.add(next_id)
        current = next_id


class SemanticaNetwork:
    """Closest-User-Chain over a binary K-means trie of per-peer avg embeddings.

    Mirrors the surface of :class:`src.lsh_network.LSHDHTNetwork`:
    ``from_embedding_manifest``, ``seed_peer_docs``, ``process_event``, and a
    ``coord`` attribute. The build phase is intentionally split into
    ``__init__`` (creates the empty Peer registry) and
    :meth:`build_trie` (called AFTER ``seed_peer_docs`` populates the
    ownership state). This ordering matches how the runner sequences calls.
    """

    PROGRESS_LOG_INTERVAL = 5_000

    def __init__(
        self,
        *,
        peer_ids: Iterable[int],
        query_lookup: EmbeddingLookup,
        doc_lookup: EmbeddingLookup,
        config: SemanticaSimulationConfig,
    ) -> None:
        if config.scenario not in SUPPORTED_SCENARIOS:
            raise ValueError(
                f"SemanticaNetwork currently only supports scenarios "
                f"{sorted(SUPPORTED_SCENARIOS)}; got scenario={config.scenario!r}"
            )
        if config.strategy not in SUPPORTED_STRATEGIES:
            raise ValueError(
                f"Unsupported Semantica strategy {config.strategy!r}; "
                f"supported = {sorted(SUPPORTED_STRATEGIES)}"
            )
        if config.max_cycles < 1:
            raise ValueError(f"max_cycles must be >= 1, got {config.max_cycles}")

        self.query_lookup = query_lookup
        self.doc_lookup = doc_lookup
        self.config = config

        unique_peer_ids = sorted(set(peer_ids))
        if not unique_peer_ids:
            raise ValueError("SemanticaNetwork requires at least one peer")
        self.sorted_peer_ids: list[int] = unique_peer_ids

        # Reuse Peer for symmetry with the other baselines (mostly for
        # owned_docs); chain walks never read Peer.connections or Peer.routed_queries.
        self.peers: dict[int, Peer] = {}
        for created, peer_id in enumerate(unique_peer_ids, start=1):
            self.peers[peer_id] = Peer(
                peer_id=peer_id,
                max_neighbors=1,  # connections unused; non-zero deque maxlen.
                max_routed_queries=0,
                max_owned_docs=config.max_owned_docs,
            )
            if (
                self.PROGRESS_LOG_INTERVAL > 0
                and created % self.PROGRESS_LOG_INTERVAL == 0
            ):
                print(
                    f"[semantica_network.init] created {created:,}/{len(unique_peer_ids):,} peers",
                    flush=True,
                )
        if self.PROGRESS_LOG_INTERVAL > 0:
            print(
                f"[semantica_network.init] done: {len(self.peers):,} peers initialized",
                flush=True,
            )

        # doc_owners parallels the LSH/semantic baselines so the
        # querier-never-counts-local rule can be enforced uniformly.
        self.doc_owners: dict[str, set[int]] = defaultdict(set)
        # Embedding dim sanity check (lazy at first use).
        self._embedding_dim = int(query_lookup.embeddings.shape[1])
        if int(doc_lookup.embeddings.shape[1]) != self._embedding_dim:
            raise ValueError(
                "Query and doc embedding manifests have mismatched dims: "
                f"queries={query_lookup.embeddings.shape[1]} "
                f"docs={doc_lookup.embeddings.shape[1]}"
            )

        # Built lazily by build_trie() once seeded ownership is known.
        self.trie: SemanticaTrie | None = None
        self._snapshot: SemanticaSnapshot | None = None
        self._peer_index: dict[int, int] = {}
        self._peer_avg_emb: np.ndarray | None = None
        self._neighbors: dict[int, np.ndarray] = {}

        self.coord = SemanticaCoordinationCounters()

    @classmethod
    def from_embedding_manifest(
        cls,
        *,
        peer_ids: Iterable[int],
        manifest_path: str | Path,
        config: SemanticaSimulationConfig,
    ) -> "SemanticaNetwork":
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

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    # ------------------------------------------------------------------
    # Indexing (mirrors LSHDHTNetwork; no LSH publish, just doc ownership)
    # ------------------------------------------------------------------

    def seed_peer_docs(self, mapping: dict[int, list[str] | tuple[str, ...] | set[str]]) -> dict[str, int]:
        """Assign starting docs to peers; tracks ownership in ``doc_owners``."""

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

    def _store_clicked_doc(self, peer_id: int, doc_id: str) -> None:
        """Mirror :meth:`ZeroRAGNetwork._store_clicked_doc`.

        For seeded scenarios this is idempotent on re-clicks because the
        clicked doc is already in the seeded inventory.
        """

        if doc_id not in self.doc_lookup:
            return
        embedding = self.doc_lookup.get(doc_id)
        owner_peer = self.peers[peer_id]
        evicted_doc_id = owner_peer.add_owned_doc(doc_id, embedding)
        was_new = doc_id not in self.doc_owners or peer_id not in self.doc_owners[doc_id]
        self.doc_owners[doc_id].add(peer_id)
        if was_new:
            self.coord.docs_published += 1
        if evicted_doc_id is None:
            return
        remaining_owners = self.doc_owners.get(evicted_doc_id)
        if remaining_owners is None:
            return
        remaining_owners.discard(peer_id)
        if not remaining_owners:
            del self.doc_owners[evicted_doc_id]

    def reset_ownership_for_empty_start(self) -> None:
        """Clear all owned docs after using seeded docs to build the topology.

        Semantica cannot construct a trie from a literally empty network. For
        the AOL ``empty_start_chronological`` run, the runner first loads the
        seeded peer-doc map to build the static Semantica topology, then calls
        this method so document availability follows the true empty-start
        chronological scenario: peers acquire clicked docs only after their
        events are processed. The snapshot keeps a live reference to
        ``self.doc_owners`` so sequential chain walks see current ownership.
        """

        for peer in self.peers.values():
            peer.owned_docs.clear()
            peer.owned_doc_catalog.doc_ids.clear()
            peer.owned_doc_catalog.row_by_doc_id.clear()
            peer.owned_doc_catalog.vectors = np.zeros((0, 0), dtype=np.float32)
            peer.owned_doc_catalog.size = 0
            peer.fifo_order.clear()
        self.doc_owners = defaultdict(set)
        self.coord.docs_published = 0
        if self._snapshot is not None:
            self._snapshot.doc_owners = self.doc_owners  # live mutable view for sequential empty-start

    # ------------------------------------------------------------------
    # Trie build (must be called after seed_peer_docs)
    # ------------------------------------------------------------------

    def build_trie(self) -> None:
        """Compute per-peer avg embeddings, build the K-means trie, run
        expansion rounds, and materialize the per-peer closest-users list.

        The per-peer ``neighbors`` exposed in the snapshot is the
        **closest-users list** (top ``ncu`` by cosine similarity to the
        peer's own avg embedding), refined over ``expansion_rounds``
        rounds of Algorithm 2 in the paper. With ``expansion_rounds=0``
        the closest-users list is just the top-``ncu`` of the co-leaves
        union. Only peers with at least one seeded doc are placed in
        the trie; peers with empty seeded inventories are recorded as
        ``querier_not_in_trie_count`` at query time and surface as
        ``cycles_used=1, exhausted`` (per plan-of-record hole #3).
        """

        # Pull avg embeddings off the Peer.owned_docs catalogs (already
        # populated by _store_clicked_doc above). Skip peers with no
        # seeded docs.
        peer_ids_with_docs: list[int] = []
        avg_rows: list[np.ndarray] = []
        for peer_id in self.sorted_peer_ids:
            peer = self.peers[peer_id]
            if not peer.owned_docs:
                continue
            stacked = np.stack(list(peer.owned_docs.values()), axis=0)
            mean = stacked.mean(axis=0)
            norm = float(np.linalg.norm(mean))
            if norm <= 0.0:
                # Degenerate: skip this peer rather than introduce NaNs.
                continue
            avg_rows.append((mean / norm).astype(np.float32))
            peer_ids_with_docs.append(peer_id)

        if not peer_ids_with_docs:
            print(
                "[semantica_network.build_trie] no peers have seeded docs; "
                "trie is empty and every chain walk will terminate at the querier",
                flush=True,
            )
            self.trie = SemanticaTrie(
                peer_ids=np.zeros(0, dtype=np.int64),
                peer_avg_emb=np.zeros((0, self._embedding_dim), dtype=np.float32),
                record_limit_per_leafnode=self.config.record_limit_per_leafnode,
                delta=self.config.delta,
                random_seed=self.config.random_seed,
            )
            self._peer_index = {}
            self._peer_avg_emb = np.zeros((0, self._embedding_dim), dtype=np.float32)
            self._neighbors = {}
            self._snapshot = SemanticaSnapshot(
                peer_index={},
                peer_avg_emb=self._peer_avg_emb,
                neighbors={},
                doc_owners={k: frozenset(v) for k, v in self.doc_owners.items()},
                max_cycles=self.config.max_cycles,
            )
            return

        peer_ids_arr = np.asarray(peer_ids_with_docs, dtype=np.int64)
        peer_avg_emb = np.stack(avg_rows, axis=0).astype(np.float32)

        # Sanity assertion: rows must be approximately L2-normalized so
        # dot product = cosine. Crash early if upstream preprocessing
        # changes silently.
        sample_norms = np.linalg.norm(peer_avg_emb[: min(8, peer_avg_emb.shape[0])], axis=1)
        if not np.all(np.abs(sample_norms - 1.0) < 1e-3):
            raise RuntimeError(
                "Semantica per-peer avg embeddings are not L2-normalized "
                f"(sample norms={sample_norms.tolist()}). Build cannot proceed."
            )

        print(
            f"[semantica_network.build_trie] building trie over "
            f"{peer_ids_arr.shape[0]:,} peers with seeded docs "
            f"(record_limit_per_leafnode={self.config.record_limit_per_leafnode}, "
            f"delta={self.config.delta}, seed={self.config.random_seed})",
            flush=True,
        )
        trie = SemanticaTrie.build(
            peer_ids=peer_ids_arr,
            peer_avg_emb=peer_avg_emb,
            record_limit_per_leafnode=self.config.record_limit_per_leafnode,
            delta=self.config.delta,
            random_seed=self.config.random_seed,
        )
        print(
            f"[semantica_network.build_trie] trie done: {len(trie.leaves):,} leaves, "
            f"max_depth={trie.max_depth}, clones={trie.clone_count}",
            flush=True,
        )

        peer_index = {int(pid): idx for idx, pid in enumerate(peer_ids_arr.tolist())}

        # ---- Initial known-users (co-leaves union) ----
        known_users: dict[int, set[int]] = {}
        for leaf_peers in trie.leaves.values():
            for peer_id in leaf_peers:
                bucket = known_users.setdefault(peer_id, set())
                bucket.update(leaf_peers)
        for peer_id in list(known_users):
            known_users[peer_id].discard(peer_id)

        if known_users:
            initial_avg = sum(len(s) for s in known_users.values()) / len(known_users)
        else:
            initial_avg = 0.0
        self.coord.initial_known_users_avg = float(initial_avg)
        print(
            f"[semantica_network.build_trie] initial known-users (co-leaves union): "
            f"avg={initial_avg:.1f} per peer over {len(known_users):,} peers",
            flush=True,
        )

        # ---- Expansion rounds (paper Algorithm 2) ----
        if self.config.expansion_rounds > 0:
            self._run_expansion_rounds(
                known_users=known_users,
                peer_index=peer_index,
                peer_avg_emb=peer_avg_emb,
            )

        # ---- Final closest-users: top-ncu of known-users by cos(self, *) ----
        ncu = max(1, int(self.config.ncu))
        neighbors: dict[int, np.ndarray] = {}
        for peer_id, candidates in known_users.items():
            if not candidates:
                neighbors[peer_id] = np.zeros(0, dtype=np.int32)
                continue
            self_emb = peer_avg_emb[peer_index[peer_id]]
            cand_arr = np.fromiter(
                (peer_index[c] for c in candidates), count=len(candidates), dtype=np.int64
            )
            cand_ids = np.fromiter(
                candidates, count=len(candidates), dtype=np.int32
            )
            sims = peer_avg_emb[cand_arr] @ self_emb
            if len(candidates) > ncu:
                top_idx = np.argpartition(-sims, ncu - 1)[:ncu]
                # Sort the top-ncu in descending similarity (stable on cosine ties via peer_id ascending below).
                top_sims = sims[top_idx]
                # Final ordering: descending sim, ties broken by ascending peer_id.
                order = np.lexsort((cand_ids[top_idx], -top_sims))
                neighbors[peer_id] = cand_ids[top_idx][order]
            else:
                order = np.lexsort((cand_ids, -sims))
                neighbors[peer_id] = cand_ids[order]

        if neighbors:
            final_avg = sum(arr.size for arr in neighbors.values()) / len(neighbors)
        else:
            final_avg = 0.0
        self.coord.final_closest_users_avg = float(final_avg)
        print(
            f"[semantica_network.build_trie] closest-users (top {ncu}): "
            f"avg={final_avg:.1f} per peer; expansion_rounds={self.config.expansion_rounds}, "
            f"updates_total={self.coord.expansion_round_updates_total:,}",
            flush=True,
        )

        self.trie = trie
        self._peer_index = peer_index
        self._peer_avg_emb = peer_avg_emb
        self._neighbors = neighbors
        self._snapshot = SemanticaSnapshot(
            peer_index=peer_index,
            peer_avg_emb=peer_avg_emb,
            neighbors=neighbors,
            doc_owners={k: frozenset(v) for k, v in self.doc_owners.items()},
            max_cycles=self.config.max_cycles,
        )

    def _run_expansion_rounds(
        self,
        *,
        known_users: dict[int, set[int]],
        peer_index: dict[int, int],
        peer_avg_emb: np.ndarray,
    ) -> None:
        """Algorithm 2 (paper): rmax rounds of querying a random closest-user
        for a closer peer.

        At the start of each round we snapshot a per-peer top-ncu
        ``closest-users`` list (sorted descending by cosine similarity to
        self). Inside the round each peer:

        1. Picks a random peer ``uj`` from its (snapshotted) closest-users.
        2. Inspects ``uj``'s known-users and finds the one most similar to
           the requester (excluding self, peers already in the requester's
           known-users, and any peer no closer than the requester's current
           ``ncu``-th-closest).
        3. If found, adds it to the requester's known-users.

        This is intentionally close to the reference implementation in
        ``Construct_tree_verified_version.py`` of the upstream Semantica
        repository: pick one random known peer, get the best closer peer
        from its known-users, accept iff strictly closer than the current
        cutoff.
        """

        ncu = max(1, int(self.config.ncu))
        rmax = max(0, int(self.config.expansion_rounds))
        rng = np.random.default_rng(self.config.random_seed)

        # Stable peer_id ordering for the round loop so two runs with the
        # same seed produce identical accepted-update sequences.
        peer_id_order = sorted(known_users.keys())

        for round_idx in range(rmax):
            # Snapshot top-ncu closest-users at the start of the round
            # so mid-round known-users updates do not cascade into the
            # random peer-pick distribution.
            snapshot_closest: dict[int, list[int]] = {}
            for peer_id in peer_id_order:
                cands = known_users[peer_id]
                if not cands:
                    snapshot_closest[peer_id] = []
                    continue
                self_emb = peer_avg_emb[peer_index[peer_id]]
                cand_ids = np.fromiter(cands, count=len(cands), dtype=np.int32)
                cand_rows = np.fromiter(
                    (peer_index[int(c)] for c in cand_ids),
                    count=cand_ids.shape[0],
                    dtype=np.int64,
                )
                sims = peer_avg_emb[cand_rows] @ self_emb
                if cand_ids.shape[0] > ncu:
                    top_idx = np.argpartition(-sims, ncu - 1)[:ncu]
                    top_sims = sims[top_idx]
                    order = np.lexsort((cand_ids[top_idx], -top_sims))
                    snapshot_closest[peer_id] = cand_ids[top_idx][order].tolist()
                else:
                    order = np.lexsort((cand_ids, -sims))
                    snapshot_closest[peer_id] = cand_ids[order].tolist()

            accepted_this_round = 0
            for peer_id in peer_id_order:
                cu = snapshot_closest[peer_id]
                if not cu:
                    continue
                random_peer = int(cu[rng.integers(len(cu))])
                random_known = known_users.get(random_peer)
                if not random_known:
                    continue

                self_emb = peer_avg_emb[peer_index[peer_id]]
                # Threshold = cosine sim to current ncu-th closest (or last
                # if fewer than ncu); only accept candidates strictly closer.
                if len(cu) >= ncu:
                    cutoff_peer = cu[ncu - 1]
                else:
                    cutoff_peer = cu[-1]
                threshold_sim = float(self_emb @ peer_avg_emb[peer_index[cutoff_peer]])

                own_known = known_users[peer_id]
                # Filter candidates uj knows about that we don't.
                novel = [
                    c for c in random_known if c != peer_id and c not in own_known
                ]
                if not novel:
                    continue
                novel_arr = np.fromiter(novel, count=len(novel), dtype=np.int32)
                rows = np.fromiter(
                    (peer_index[int(c)] for c in novel_arr),
                    count=novel_arr.shape[0],
                    dtype=np.int64,
                )
                sims = peer_avg_emb[rows] @ self_emb
                best = int(np.argmax(sims))
                best_sim = float(sims[best])
                if best_sim > threshold_sim:
                    own_known.add(int(novel_arr[best]))
                    accepted_this_round += 1

            self.coord.expansion_round_updates_total += accepted_this_round
            print(
                f"[semantica_network.expansion_rounds] round {round_idx + 1}/{rmax}: "
                f"{accepted_this_round:,} accepted updates",
                flush=True,
            )

    # ------------------------------------------------------------------
    # Snapshot accessors for the parallel driver
    # ------------------------------------------------------------------

    @property
    def snapshot(self) -> SemanticaSnapshot:
        if self._snapshot is None:
            raise RuntimeError(
                "SemanticaNetwork.build_trie() must be called before accessing snapshot"
            )
        return self._snapshot

    def avg_neighbor_list_size(self) -> float:
        if not self._neighbors:
            return 0.0
        sizes = [arr.shape[0] for arr in self._neighbors.values()]
        return float(sum(sizes)) / float(len(sizes))

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

    # ------------------------------------------------------------------
    # Sequential event loop (used by --parallelism=1 and pytest)
    # ------------------------------------------------------------------

    def process_event(self, event: QueryEvent, metrics: MetricsTracker) -> None:
        """Apply skip rules then run a single chain walk inline.

        Skip-rule branching is intentionally a near-copy of
        :meth:`src.lsh_network.LSHDHTNetwork.process_event` for SEEDED
        scenarios so the ``skipped_by_reason`` distributions match exactly.
        """

        scenario = self.config.scenario
        is_multi_relevant = self.config.retrieval_mode == "multi_relevant"
        relevant_doc_ids = self._relevant_doc_ids_for_event(event)
        available_relevant_doc_ids = self._available_relevant_doc_ids(relevant_doc_ids)
        if is_multi_relevant and not available_relevant_doc_ids:
            metrics.note_skipped_query("non_searchable")
            self.coord.queries_skipped += 1
            return

        # For seeded scenarios the only skips are non_searchable (sidecar)
        # and missing_query_embedding. For empty_start_chronological we also
        # mirror low-coord's local_held skip after the query-embedding check.
        if not event.searchable and not is_multi_relevant:
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

        if not is_multi_relevant and scenario == "empty_start_chronological" and self.peers[event.user_id].has_doc(
            event.clicked_doc_id
        ):
            metrics.note_skipped_query("local_held")
            self.coord.queries_skipped += 1
            self._store_clicked_doc(event.user_id, event.clicked_doc_id)
            return

        # Run the chain walk on the current snapshot.
        snapshot = self.snapshot
        querier_id = event.user_id
        target_doc_id = event.clicked_doc_id
        query_embedding = self.query_lookup.get(event.query_text)

        if querier_id not in snapshot.peer_index:
            self.coord.querier_not_in_trie_count += 1
            search_result = SemanticaSearchResult(
                cycles_used=1,
                result_symbol="-",
                termination_reason="exhausted",
                contacted_peer_ids=[],
            )
        else:
            neighbors_arr = snapshot.neighbors.get(querier_id)
            if neighbors_arr is None or neighbors_arr.size == 0:
                self.coord.dead_end_at_querier_count += 1
            search_result = run_closest_user_chain(
                snapshot,
                querier_id=querier_id,
                target_doc_id=target_doc_id,
                query_embedding=query_embedding,
                relevant_doc_ids=available_relevant_doc_ids if is_multi_relevant else None,
            )

        self._record_search_result(search_result)
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
                requests_until_first_relevant=(
                    search_result.requests_until_first_relevant if is_multi_relevant else None
                ),
            )
        )

        # Acquire the clicked doc only after the search attempt completes,
        # exactly as in the other baselines. For seeded scenarios this is
        # idempotent.
        if not is_multi_relevant:
            self._store_clicked_doc(event.user_id, event.clicked_doc_id)

    def _record_search_result(self, search_result: SemanticaSearchResult) -> None:
        """Update algorithm-only counters from a single chain-walk result."""

        self.coord.queries_executed += 1
        self.coord.chain_hops_total += max(0, search_result.cycles_used - 1)
        if search_result.termination_reason == "found":
            self.coord.chain_found_count += 1
        elif search_result.termination_reason == "exhausted":
            self.coord.chain_exhausted_count += 1
        elif search_result.termination_reason == "cap":
            self.coord.chain_cap_count += 1


# ----------------------------------------------------------------------
# Parallel driver (used by experiments/run_semantica_simulation.py)
# ----------------------------------------------------------------------


# Module-level worker globals. Populated in the parent process BEFORE the
# fork-based ProcessPoolExecutor is created so workers inherit them via
# copy-on-write. NEVER assign to these from a worker.
_WORKER_SNAPSHOT: SemanticaSnapshot | None = None
_WORKER_QUERY_LOOKUP: EmbeddingLookup | None = None


def _set_worker_globals(snapshot: SemanticaSnapshot, query_lookup: EmbeddingLookup) -> None:
    """Assign the worker-visible globals in the parent before forking."""

    global _WORKER_SNAPSHOT, _WORKER_QUERY_LOOKUP
    _WORKER_SNAPSHOT = snapshot
    _WORKER_QUERY_LOOKUP = query_lookup


def _worker_chain_chunk(
    chunk: list[tuple[int, int, str, int, tuple[str, ...]]],
) -> list[tuple[int, int, str, str, bool, bool, tuple[str, ...], int | None]]:
    """Run Closest-User-Chain on a chunk of events inside a worker process.

    Each chunk item is ``(event_index, querier_id, target_doc_id,
    query_embedding_index)``. The query embedding is resolved through the
    inherited memmap-backed :class:`EmbeddingLookup` so the IPC payload is
    one ``int`` per event instead of a full ``(d,)`` float32 array.

    Returns one tuple per event: ``(event_index, cycles_used, result_symbol,
    termination_reason, querier_not_in_trie, dead_end_at_querier)``. The
    last two flags let the main thread keep the algorithm-only counters in
    sync with sequential execution without having to inspect the snapshot
    again.
    """

    snapshot = _WORKER_SNAPSHOT
    query_lookup = _WORKER_QUERY_LOOKUP
    if snapshot is None or query_lookup is None:
        raise RuntimeError(
            "Semantica worker globals not initialized; "
            "_set_worker_globals must run in the parent before fork."
        )

    out: list[tuple[int, int, str, str, bool, bool, tuple[str, ...], int | None]] = []
    for event_index, querier_id, target_doc_id, query_index, relevant_doc_ids_tuple in chunk:
        if querier_id not in snapshot.peer_index:
            out.append((event_index, 1, "-", "exhausted", True, False, (), None))
            continue

        neighbors_arr = snapshot.neighbors.get(querier_id)
        dead_end = neighbors_arr is None or neighbors_arr.size == 0

        query_emb = query_lookup.get_by_index(query_index)
        result = run_closest_user_chain(
            snapshot,
            querier_id=querier_id,
            target_doc_id=target_doc_id,
            query_embedding=query_emb,
            relevant_doc_ids=(
                frozenset(relevant_doc_ids_tuple) if relevant_doc_ids_tuple else None
            ),
        )
        out.append(
            (
                event_index,
                result.cycles_used,
                result.result_symbol,
                result.termination_reason,
                False,
                bool(dead_end),
                result.retrieved_relevant_doc_ids,
                result.requests_until_first_relevant,
            )
        )
    return out


@dataclass(slots=True)
class SemanticaParallelDriver:
    """Stream events through skip rules sequentially, dispatch chain walks
    to a fork-based ``ProcessPoolExecutor`` of ``parallelism`` workers,
    and drain results back into ``MetricsTracker`` in event-index order.

    The skip-rule branching is intentionally a near-copy of
    :meth:`SemanticaNetwork.process_event` so any byte-shape difference is
    a bug. Workers receive a copy-on-write snapshot of the trie state via
    fork; the IPC payload per event is just four ints/strings.

    With ``parallelism <= 1`` this falls back to inline
    :meth:`SemanticaNetwork.process_event` so smoke tests, pytest, and
    spawn-only platforms (macOS) keep working.

    The batch model (``batch_size = parallelism * chunk_size``) trades a
    small amount of pipelining for deterministic ordering: each batch is
    classified, dispatched, awaited, and drained before the next one
    begins. With reasonable defaults (60 workers x 256 events = ~15k
    events/batch) the per-batch main-thread overhead is well under 1% of
    wall time on AOL/Amazon at the inputs we care about.
    """

    network: "SemanticaNetwork"
    parallelism: int = 60
    chunk_size: int = 256
    progress_interval_events: int = 50_000
    _processed_events: int = 0

    def run(self, events: Sequence[QueryEvent], metrics: MetricsTracker) -> None:
        if self.parallelism <= 1:
            for event in events:
                self.network.process_event(event, metrics)
            return

        snapshot = self.network.snapshot
        _set_worker_globals(snapshot, self.network.query_lookup)

        # Force fork so workers inherit the parent's snapshot via COW;
        # macOS spawn would re-pickle the snapshot per worker, defeating
        # the parallelism benefit. The runner is documented to require
        # parallelism=1 on macOS.
        try:
            ctx = mp.get_context("fork")
        except ValueError as exc:  # pragma: no cover - non-fork platforms
            raise RuntimeError(
                "SemanticaParallelDriver with parallelism > 1 requires "
                "the 'fork' multiprocessing start method (Linux). "
                "Use --parallelism=1 on platforms without fork."
            ) from exc

        batch_size = max(self.parallelism * self.chunk_size, 1)
        with ProcessPoolExecutor(
            max_workers=self.parallelism,
            mp_context=ctx,
        ) as pool:
            i = 0
            n = len(events)
            while i < n:
                batch = events[i : i + batch_size]
                self._process_batch(batch, metrics, pool)
                i += batch_size

    def _skip_reason(self, event: QueryEvent) -> str | None:
        """Mirror the SEEDED skip-rule order from
        :meth:`SemanticaNetwork.process_event`."""

        if self.network.config.retrieval_mode != "multi_relevant" and not event.searchable:
            return "non_searchable"
        if event.query_text not in self.network.query_lookup:
            return "missing_query_embedding"
        return None

    def _process_batch(
        self,
        batch: Sequence[QueryEvent],
        metrics: MetricsTracker,
        pool: ProcessPoolExecutor,
    ) -> None:
        """Classify -> dispatch chunks -> await -> drain in event-index order."""

        classified: list[
            tuple[int, QueryEvent, str, object]
        ] = []  # (event_index, event, kind, payload)
        walk_tasks: list[tuple[int, int, str, int, tuple[str, ...]]] = []
        available_by_event_index: dict[int, frozenset[str]] = {}
        relevant_by_event_index: dict[int, frozenset[str]] = {}
        for event in batch:
            if self.network.config.retrieval_mode == "multi_relevant":
                relevant_doc_ids = self.network._relevant_doc_ids_for_event(event)
                available_doc_ids = self.network._available_relevant_doc_ids(relevant_doc_ids)
                relevant_by_event_index[event.event_index] = relevant_doc_ids
                available_by_event_index[event.event_index] = available_doc_ids
                if not available_doc_ids:
                    classified.append((event.event_index, event, "skip", "non_searchable"))
                    continue
            reason = self._skip_reason(event)
            if reason is not None:
                classified.append((event.event_index, event, "skip", reason))
                continue
            qidx = self.network.query_lookup.index_of(event.query_text)
            classified.append((event.event_index, event, "walk", qidx))
            walk_tasks.append(
                (
                    event.event_index,
                    event.user_id,
                    event.clicked_doc_id,
                    qidx,
                    tuple(sorted(available_by_event_index.get(event.event_index, ()))),
                )
            )

        # Dispatch chunks. We pre-build the chunks rather than streaming so
        # the worker function receives a list (cheaper unpickle than tuple-of-tuples).
        chunks = [
            walk_tasks[start : start + self.chunk_size]
            for start in range(0, len(walk_tasks), self.chunk_size)
        ]
        futures = [pool.submit(_worker_chain_chunk, chunk) for chunk in chunks]

        walk_results: dict[
            int, tuple[int, int, str, str, bool, bool, tuple[str, ...], int | None]
        ] = {}
        for fut in futures:
            for record in fut.result():
                walk_results[record[0]] = record

        # Drain in event-index order. classified is already in receive order
        # which equals event-index order for in-order event streams; sort
        # defensively to keep the contract explicit.
        classified.sort(key=lambda item: item[0])
        for event_index, event, kind, payload in classified:
            if kind == "skip":
                self.network.coord.queries_skipped += 1
                metrics.note_skipped_query(payload)  # type: ignore[arg-type]
                if self.network.config.retrieval_mode != "multi_relevant":
                    self.network._store_clicked_doc(event.user_id, event.clicked_doc_id)
                continue

            walk_record = walk_results.get(event_index)
            if walk_record is None:
                raise RuntimeError(
                    f"Missing chain-walk result for event_index={event_index}; "
                    "worker likely crashed."
                )
            (
                _,
                cycles_used,
                result_symbol,
                termination_reason,
                not_in_trie,
                dead_end,
                retrieved_relevant_doc_ids,
                requests_until_first_relevant,
            ) = walk_record
            if not_in_trie:
                self.network.coord.querier_not_in_trie_count += 1
            elif dead_end:
                self.network.coord.dead_end_at_querier_count += 1

            self.network.coord.queries_executed += 1
            self.network.coord.chain_hops_total += max(0, cycles_used - 1)
            if termination_reason == "found":
                self.network.coord.chain_found_count += 1
            elif termination_reason == "exhausted":
                self.network.coord.chain_exhausted_count += 1
            elif termination_reason == "cap":
                self.network.coord.chain_cap_count += 1

            metrics.log_query(
                QueryLog(
                    event_index=event.event_index,
                    query_id=event.qid,
                    querier_peer_id=event.user_id,
                    clicked_doc_id=event.clicked_doc_id,
                    cycles_used=cycles_used,
                    result=result_symbol,
                    termination_reason=termination_reason,
                    relevant_doc_count=len(relevant_by_event_index.get(event.event_index, ())),
                    available_relevant_doc_count=len(
                        available_by_event_index.get(event.event_index, ())
                    ),
                    retrieved_relevant_doc_count=len(retrieved_relevant_doc_ids),
                    query_retrieval_recall=(
                        len(retrieved_relevant_doc_ids)
                        / len(available_by_event_index[event.event_index])
                        if available_by_event_index.get(event.event_index)
                        else 0.0
                    ),
                    requests_until_first_relevant=requests_until_first_relevant,
                )
            )
            if self.network.config.retrieval_mode != "multi_relevant":
                self.network._store_clicked_doc(event.user_id, event.clicked_doc_id)

        self._processed_events += len(batch)
        if (
            self.progress_interval_events > 0
            and self._processed_events % self.progress_interval_events < len(batch)
        ):
            print(
                f"[semantica_driver] processed {self._processed_events:,} events",
                flush=True,
            )


__all__ = [
    "DEFAULT_STRATEGY",
    "MAX_TRIE_DEPTH",
    "SUPPORTED_STRATEGIES",
    "SUPPORTED_SCENARIOS",
    "SemanticaCoordinationCounters",
    "SemanticaNetwork",
    "SemanticaParallelDriver",
    "SemanticaSearchResult",
    "SemanticaSimulationConfig",
    "SemanticaSnapshot",
    "SemanticaTrie",
    "run_closest_user_chain",
]

# ZeroRAG

This project evaluates ZeroRAG, a peer-to-peer retrieval method over event
streams where each query belongs to a user peer and each target document is a clicked
or positively bound item. The system is intentionally local: peers do not build a
global index, do not broadcast queries, and do not require centralized routing state.
Instead, each peer keeps a bounded neighborhood, a local document inventory, and
optional memory of previous routed queries.

The main simulator is `experiments/run_simulation.py`. The core method lives in
`src/network.py`, `src/peer.py`, and `src/routing.py`.

## Method Overview

ZeroRAG models retrieval as repeated local decisions on a directed
peer graph.

1. Every peer starts with at most `max_neighbors` outgoing connections. The simulator
  bootstraps these connections randomly from the known peer population.
2. Each peer owns a local set of document embeddings. Depending on the scenario, this
  inventory starts empty, starts with the peer's future clicked documents, or is capped
   with FIFO eviction.
3. A query event starts at the querier peer. The target document is the event's clicked
  or positive item, but the querier's own local copy is never allowed to count as a
   distributed success.
4. The current peer answers with a success if it owns the target document and is not the
  original querier. Otherwise it returns referrals to other peers.
5. The querier accumulates referrals and chooses the next unqueried peer by the highest
  similarity score. Search stops when the target is found, no candidates remain, or
   `max_cycles` is reached.
6. In adaptive topology mode, the querier adds a directed connection to the successful
  peer. If the target was not found, it can connect to the best-known referred peer.
7. After the search attempt, the querier acquires the clicked document for future events.

This creates a feedback loop: successful or near-successful retrievals reshape the
querier's neighborhood, and future peers can route semantically similar queries toward
peers that previously acquired relevant documents.

## Peer State

Each `Peer` stores four local structures:

- `connections`: a bounded FIFO deque of outgoing peer IDs. `max_neighbors` defaults to
`60`. New adaptive connections evict the oldest connection when the deque is full.
- `owned_docs`: the peer's current local document inventory. Each document also appears
in a compact `DocumentCatalog` matrix so nearest-owned-document referrals can be
computed with vector dot products.
- `fifo_order`: insertion/refresh order for bounded-inventory scenarios. When
`max_owned_docs > 0`, adding a new document beyond the cap evicts the oldest local
document.
- `routed_queries`: optional semantic query memory. Entries store a compact query
embedding index, the original querier, and the clicked document. The default cap is
`max_routed_queries=1000`.

The network also keeps `doc_owners`, a global simulator-side map from document ID to
current owner peer IDs. This is not used as a routing oracle. It is used to maintain
ownership bookkeeping and to decide whether runtime-searchable scenarios should skip an
event that currently has no non-querier owner.

## Referral Generation

When peer `P` is queried, it produces candidate referrals from two sources.

### Replica Referrals

Replica referrals inspect `P`'s outgoing neighbors. For each neighbor that owns at least
one document, the neighbor's `DocumentCatalog` returns the owned document with the
highest dot-product similarity to the current query embedding. The referral points to
that neighbor and carries the best document ID, the similarity score, and source
`replica`.

This is the only referral source in `--routing-mode replica_only`.

### Routed-Query Memory

In the default `--routing-mode semantic_query_memory`, a peer also compares the current
query embedding to query embeddings stored in its `routed_queries` memory. If a stored
query is above `similarity_threshold` (default `0.5`), the peer emits a `routed_query`
referral to the original querier of that earlier event.

That referral is useful because the original querier acquired the clicked document after
the earlier search attempt. In other words, routed-query memory remembers where similar
demand previously ended up, not just where similar replicas were already visible through
the current neighborhood.

Replica and routed-query referrals are merged by peer ID. If multiple referrals target
the same peer, the highest-similarity one wins. The response is sorted by similarity and
truncated to `referrals_per_response` (default `10`). The original querier is excluded
from referrals so the system does not route back to local cache hits.

## Search Procedure

`src/routing.py` implements the query walk as best-first search over discovered
referrals:

1. Start with `current_peer_id = querier_peer_id`.
2. Query the current peer and record one cycle.
3. Add all returned referrals to a candidate list, keeping the best referral per target
  peer.
4. If the current peer owns the target document and is not the querier, return success.
5. Otherwise, choose the highest-similarity unqueried candidate as the next peer.
6. Return `exhausted` if no unqueried candidates remain, or `cap` if `max_cycles`
  queries have already been issued.

The default `max_cycles` is `50`, and the querier itself counts as the first cycle.

`--selection-mode random_walk` is an ablation rather than the ZeroRAG semantic
method. In that mode the simulator samples up to `max_cycles` active peers uniformly
at random, excluding the querier, and checks whether any sampled peer owns the target
document. It does not use the peer graph, referrals, query embeddings, routed-query
memory, or adaptive topology updates.

## Topology Modes

The simulator separates routing from graph evolution:

- `forming_new_connections` is the adaptive mode. After each routed event, the querier
adds a directed edge to the successful peer. If the search did not find the target but
did discover referrals, it connects to the best-known referred peer instead.
- `static_initial_connections` freezes the random bootstrap graph. This is the matched
baseline for measuring the value of adaptive edge formation.

For fair forming-vs-static comparisons, use the same `LOW_COORD_DHT_RANDOM_SEED` so both
runs start from the same bootstrap graph.

## Inventory Scenarios

The method is evaluated under several ownership regimes:

- `empty_start_chronological`: peers start empty and acquire clicked documents over the
real chronological event stream. If the querier already holds the target, the event is
skipped as `local_held`.
- `seeded_fixed_inventory`: peers start with the full set of documents they will
eventually click in the trace. This is an oracle-style inventory ablation.
- `seeded_randomized_timeline`: peers use the same seeded inventory as
`seeded_fixed_inventory`, but events are consumed in randomized order.
- `fifo_chronological`: peers start empty and acquire documents chronologically, but keep
at most `max_owned_docs` local documents. Re-clicking a held document refreshes its FIFO
position and skips routing as `local_held`.
- `empty_start_randomized_timeline`: peers start empty and consume the randomized event
stream with runtime searchability checks.
- `fifo_randomized_timeline`: peers combine randomized event order with FIFO-bounded
ownership.

For runtime-searchable scenarios, an event is routable only if some peer other than the
querier currently owns the target document. For sidecar-searchable scenarios, routability
comes from the precomputed searchable-query file.

Across all scenarios, the querier's own local document never counts as a distributed
retrieval success.

## Datasets

The simulator currently supports:

- `aol`: chronological AOL query/click events.
- `amazon`: real user-item purchases/reviews + synthetic generated queries + UCSD
metadata joined by `asin`. Amazon has no real timestamps, so only non-chronological
scenarios are supported. On Amazon, `seeded_fixed_inventory` is canonicalized to
`seeded_randomized_timeline`.
- `mind`: news recommendation events using the same simulator-facing event shape.

Dataset paths are resolved by `src.data_root.resolve_dataset_dir`. Embeddings are loaded
through a manifest produced by `compute_embeddings.py`.

## Important Knobs

Common CLI flags and matching environment variables are exposed through
`experiments/run_simulation.py` and the Slurm wrappers:

- `--routing-mode {semantic_query_memory,replica_only}` controls whether routed-query
memory is enabled.
- `--topology-mode {forming_new_connections,static_initial_connections}` controls
whether adaptive edges are added after search.
- `--selection-mode {similarity,random_walk}` chooses semantic best-first routing or the
global random-peer probe ablation.
- `--max-neighbors` bounds each peer's outgoing connection list.
- `--max-cycles` bounds the number of queried peers per event.
- `--referrals-per-response` bounds each peer response.
- `--similarity-threshold` filters routed-query memory referrals.
- `--max-routed-queries` bounds per-peer routed-query memory.
- `--max-owned-docs` bounds local inventory in FIFO scenarios.
- `--random-seed` controls random bootstrap connections and randomized choices.

Each run writes `run_manifest.json`, `query_log.jsonl`, and `metrics.json` to the output
directory. The main metrics are recall, average cycles, skip rate, and skip reasons.

## Minimal Local Command

For a small local smoke run, point the command at a prepared dataset directory and
embedding manifest:

```bash
python experiments/run_simulation.py \
  --dataset aol \
  --scenario empty_start_chronological \
  --routing-mode semantic_query_memory \
  --topology-mode forming_new_connections \
  --max-events 5000 \
  --output-dir results/smoke_zerorag
```

On DAS-6, use the Slurm scripts in `scripts/` so long simulations run on compute nodes
and write heavy outputs to scratch.

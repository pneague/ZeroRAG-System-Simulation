#!/usr/bin/env python3
"""Run the Semantica (arXiv:2502.10151) Closest-User-Chain baseline.

Sibling of :mod:`experiments.run_simulation` and
:mod:`experiments.run_lsh_simulation`. Reuses the same dataset loaders,
embedding manifests, scenario rules, and :class:`MetricsTracker`, but
routes queries through :class:`src.semantica_network.SemanticaNetwork`
(closest-user-chain over a binary K-means trie of per-peer average doc
embeddings) and dispatches the per-event chain walks to a 60-way
``ProcessPoolExecutor`` via :class:`SemanticaParallelDriver`.

Scope locked to ``--scenario seeded_randomized_timeline``: the upstream
algorithm is a one-shot bulk evaluator; under that scenario peer
ownership is fixed at startup, the trie is built once, and every chain
walk is a pure read-only function of the snapshot, which is what unlocks
the 60-way parallel querying.

Output layout (compatible with both other baselines' tooling):

* ``query_log.jsonl`` -- per-query
  ``(event_index, query_id, querier_peer_id, clicked_doc_id, cycles_used,
  result, termination_reason)``; identical schema to the LSH and semantic
  runs.
* ``metrics.json`` -- rolling-window ``recall``, ``avg_cycles``,
  ``skip_rate``; identical schema.
* ``run_manifest.json`` -- adds ``algorithm``, ``semantica_config``, and
  ``metric_semantics`` blocks so analysis can disambiguate algorithms.
* ``semantica_metrics.json`` -- algorithm-only coordination counters
  (chain hop totals, found / exhausted / cap counts, dead-end-at-querier
  count, querier-not-in-trie count, build-time seconds, query-phase wall
  seconds, average neighbor-list size, trie depth, clone count).
"""

from __future__ import annotations

import argparse
import fcntl
import gzip
import json
import os
import socket
import subprocess
import sys
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.path_defaults import default_experiments_registry  # noqa: E402

DEFAULT_EXPERIMENTS_REGISTRY = str(default_experiments_registry())
REGISTRY_SCHEMA_VERSION = 1
ALGORITHM_NAME = "semantica_chain"
ALLOWED_SCENARIOS: frozenset[str] = frozenset(
    {
        "seeded_randomized_timeline",
        "seeded_fixed_inventory",
        "empty_start_chronological",
    }
)


from src.scenarios import (  # noqa: E402
    AMAZON_SUPPORTED_SCENARIOS,
    PUBMEDQA_SUPPORTED_SCENARIOS,
    RANDOMIZED_TIMELINE_SCENARIOS,
    SEEDED_SCENARIOS,
)


def _default_metadata_path(
    scenario: str,
    *,
    dataset: str,
    amazon_category: str | None,
) -> Path:
    """Resolve the metadata CSV for the requested Semantica scenario."""

    from src.amazon_dataset import AMAZON_RANDOMIZED_EVENTS_FILENAME
    from src.data_root import resolve_dataset_dir
    from src.dataset import METADATA_FILENAME, SEEDED_RANDOMIZED_METADATA_FILENAME

    if dataset == "amazon":
        return resolve_dataset_dir("amazon", amazon_category) / AMAZON_RANDOMIZED_EVENTS_FILENAME
    if dataset == "mind":
        from src.mind_dataset import MIND_EVENTS_FILENAME, MIND_RANDOMIZED_EVENTS_FILENAME

        mind_dir = resolve_dataset_dir("mind")
        if scenario == "seeded_randomized_timeline":
            return mind_dir / MIND_RANDOMIZED_EVENTS_FILENAME
        if scenario in {"seeded_fixed_inventory", "empty_start_chronological"}:
            return mind_dir / MIND_EVENTS_FILENAME
    if dataset == "synthetic":
        from src.synthetic_dataset import (
            SYNTHETIC_EVENTS_FILENAME,
            SYNTHETIC_RANDOMIZED_EVENTS_FILENAME,
        )

        synthetic_dir = resolve_dataset_dir("synthetic")
        if scenario == "seeded_randomized_timeline":
            return synthetic_dir / SYNTHETIC_RANDOMIZED_EVENTS_FILENAME
        if scenario == "seeded_fixed_inventory":
            return synthetic_dir / SYNTHETIC_EVENTS_FILENAME
    if dataset == "pubmedqa":
        from src.pubmedqa_dataset import (
            PUBMEDQA_EVENTS_FILENAME,
            PUBMEDQA_RANDOMIZED_EVENTS_FILENAME,
        )

        pubmedqa_dir = resolve_dataset_dir("pubmedqa")
        if scenario == "seeded_randomized_timeline":
            return pubmedqa_dir / PUBMEDQA_RANDOMIZED_EVENTS_FILENAME
        if scenario == "seeded_fixed_inventory":
            return pubmedqa_dir / PUBMEDQA_EVENTS_FILENAME
    if scenario == "seeded_randomized_timeline":
        return resolve_dataset_dir("aol") / SEEDED_RANDOMIZED_METADATA_FILENAME
    if scenario in {"seeded_fixed_inventory", "empty_start_chronological"}:
        return resolve_dataset_dir("aol") / METADATA_FILENAME
    raise ValueError(f"Unsupported Semantica scenario for metadata: {scenario}")


def _default_searchable_queries_path(
    scenario: str,
    *,
    dataset: str,
    amazon_category: str | None,
) -> Path:
    """Resolve the searchable-query sidecar for the requested scenario."""

    from src.amazon_dataset import AMAZON_SEEDED_SEARCHABLE_FILENAME
    from src.data_root import resolve_dataset_dir
    from src.dataset import (
        CHRONOLOGICAL_SEARCHABLE_QUERIES_FILENAME,
        SEEDED_SEARCHABLE_QUERIES_FILENAME,
    )

    if dataset == "amazon":
        return resolve_dataset_dir("amazon", amazon_category) / AMAZON_SEEDED_SEARCHABLE_FILENAME
    if dataset == "mind":
        from src.mind_dataset import MIND_SEEDED_SEARCHABLE_FILENAME

        mind_dir = resolve_dataset_dir("mind")
        if scenario == "empty_start_chronological":
            return mind_dir / CHRONOLOGICAL_SEARCHABLE_QUERIES_FILENAME
        if scenario in SEEDED_SCENARIOS:
            return mind_dir / MIND_SEEDED_SEARCHABLE_FILENAME
    if dataset == "synthetic":
        from src.synthetic_dataset import SYNTHETIC_SEEDED_SEARCHABLE_FILENAME

        if scenario in SEEDED_SCENARIOS:
            return resolve_dataset_dir("synthetic") / SYNTHETIC_SEEDED_SEARCHABLE_FILENAME
    if dataset == "pubmedqa":
        from src.pubmedqa_dataset import PUBMEDQA_SEEDED_SEARCHABLE_FILENAME

        if scenario in SEEDED_SCENARIOS:
            return resolve_dataset_dir("pubmedqa") / PUBMEDQA_SEEDED_SEARCHABLE_FILENAME
    if scenario == "empty_start_chronological":
        return resolve_dataset_dir("aol") / CHRONOLOGICAL_SEARCHABLE_QUERIES_FILENAME
    if scenario in SEEDED_SCENARIOS:
        return resolve_dataset_dir("aol") / SEEDED_SEARCHABLE_QUERIES_FILENAME
    raise ValueError(f"Unsupported Semantica scenario for searchable queries: {scenario}")


def _default_embedding_manifest_path(*, dataset: str, amazon_category: str | None) -> Path:
    from src.data_root import resolve_dataset_dir

    if dataset == "amazon":
        return resolve_dataset_dir("amazon", amazon_category) / "embeddings" / "manifest.json"
    if dataset == "mind":
        return resolve_dataset_dir("mind") / "embeddings" / "manifest.json"
    if dataset == "synthetic":
        return resolve_dataset_dir("synthetic") / "embeddings" / "manifest.json"
    if dataset == "pubmedqa":
        return resolve_dataset_dir("pubmedqa") / "embeddings" / "manifest.json"
    return resolve_dataset_dir("aol") / "embeddings" / "manifest.json"


def _default_seeded_peer_docs_path(*, dataset: str, amazon_category: str | None) -> Path:
    from src.amazon_dataset import AMAZON_PEER_FUTURE_DOCS_FILENAME
    from src.data_root import resolve_dataset_dir

    if dataset == "amazon":
        return resolve_dataset_dir("amazon", amazon_category) / AMAZON_PEER_FUTURE_DOCS_FILENAME
    if dataset == "mind":
        from src.mind_dataset import MIND_PEER_FUTURE_DOCS_FILENAME

        return resolve_dataset_dir("mind") / MIND_PEER_FUTURE_DOCS_FILENAME
    if dataset == "synthetic":
        from src.synthetic_dataset import SYNTHETIC_PEER_FUTURE_DOCS_FILENAME

        return resolve_dataset_dir("synthetic") / SYNTHETIC_PEER_FUTURE_DOCS_FILENAME
    if dataset == "pubmedqa":
        from src.pubmedqa_dataset import PUBMEDQA_PEER_FUTURE_DOCS_FILENAME

        return resolve_dataset_dir("pubmedqa") / PUBMEDQA_PEER_FUTURE_DOCS_FILENAME
    return resolve_dataset_dir("aol") / "peer_future_doc_ids.json.gz"


def _default_relevant_docs_by_qid_path(*, dataset: str, amazon_category: str | None) -> Path | None:
    if dataset != "amazon":
        return None
    from src.data_root import resolve_dataset_dir
    from src.dataset import RELEVANT_DOC_IDS_BY_QID_FILENAME

    return resolve_dataset_dir("amazon", amazon_category) / RELEVANT_DOC_IDS_BY_QID_FILENAME


def _git_commit_or_none() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent.parent,
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    commit = result.stdout.strip()
    return commit or None


def _append_to_experiments_registry(registry_path: Path, record: dict) -> None:
    """Identical body to the LSH/semantic runners so all three algorithms
    share the central registry envelope."""

    try:
        registry_path.parent.mkdir(parents=True, exist_ok=True)
        with open(registry_path, "a+", encoding="utf-8") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            try:
                handle.seek(0)
                raw = handle.read().strip()
                payload: dict
                if raw:
                    try:
                        payload = json.loads(raw)
                        if not isinstance(payload, dict):
                            raise ValueError("registry root is not an object")
                    except (json.JSONDecodeError, ValueError) as exc:
                        backup_path = registry_path.with_suffix(
                            registry_path.suffix + f".bak.{int(time.time())}"
                        )
                        try:
                            backup_path.write_text(raw, encoding="utf-8")
                        except OSError:
                            pass
                        print(
                            f"[registry] corrupted registry quarantined to {backup_path}: {exc}",
                            file=sys.stderr,
                        )
                        payload = {"schema_version": REGISTRY_SCHEMA_VERSION, "runs": []}
                else:
                    payload = {"schema_version": REGISTRY_SCHEMA_VERSION, "runs": []}

                payload.setdefault("schema_version", REGISTRY_SCHEMA_VERSION)
                runs = payload.setdefault("runs", [])
                if not isinstance(runs, list):
                    runs = []
                    payload["runs"] = runs
                runs.append(record)

                handle.seek(0)
                handle.truncate()
                json.dump(payload, handle, indent=2, default=str)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)
    except Exception as exc:
        print(
            f"[registry] failed to append run record to {registry_path}: {exc}",
            file=sys.stderr,
        )
        traceback.print_exc(file=sys.stderr)


def _load_seeded_doc_mapping(path: str | Path) -> dict[int, list[str]]:
    mapping_path = Path(path)
    if mapping_path.suffix == ".gz":
        with gzip.open(mapping_path, "rt", encoding="utf-8") as handle:
            payload = json.load(handle)
    else:
        payload = json.loads(mapping_path.read_text(encoding="utf-8"))
    return {int(peer_id): [str(doc_id) for doc_id in doc_ids] for peer_id, doc_ids in payload.items()}


def main() -> None:
    from src.amazon_dataset import AMAZON_CATEGORY_SPECS
    from src.semantica_network import (
        DEFAULT_STRATEGY,
        SUPPORTED_STRATEGIES,
    )

    dataset_choices = ("aol", "amazon", "mind", "synthetic", "pubmedqa")
    amazon_category_choices = tuple(sorted(AMAZON_CATEGORY_SPECS))

    parser = argparse.ArgumentParser(
        description="Run the Semantica Closest-User-Chain baseline (arXiv:2502.10151)"
    )
    parser.add_argument(
        "--dataset",
        choices=dataset_choices,
        default=os.environ.get("LOW_COORD_DHT_DATASET", "aol"),
    )
    parser.add_argument(
        "--amazon-category",
        choices=amazon_category_choices,
        default=os.environ.get("LOW_COORD_DHT_AMAZON_CATEGORY", "cell_phones"),
    )
    parser.add_argument("--metadata", default=None)
    parser.add_argument("--searchable-queries", default=None)
    parser.add_argument("--embedding-manifest", default=None)
    parser.add_argument("--output-dir", default=str(Path("results") / "semantica_chain"))
    parser.add_argument("--max-events", type=int, default=0, help="0 means all events")
    parser.add_argument("--metrics-window", type=int, default=1000)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument(
        "--semantica-peer-order-seed",
        type=int,
        default=(
            int(os.environ["LOW_COORD_DHT_SEMANTICA_PEER_ORDER_SEED"])
            if os.environ.get("LOW_COORD_DHT_SEMANTICA_PEER_ORDER_SEED")
            else None
        ),
        help=(
            "Optional seed for permuting peer rows before building the "
            "Semantica trie. Omit to preserve historical ascending peer-id "
            "order. This is independent of --random-seed."
        ),
    )
    parser.add_argument(
        "--scenario",
        choices=tuple(sorted(ALLOWED_SCENARIOS)),
        default="seeded_randomized_timeline",
        help=(
            "Semantica scenarios. Chronological scenarios require real timestamps. "
            "empty_start_chronological uses chronological document "
            "availability with a static Semantica topology built from the "
            "seeded peer-doc map, then clears ownership before the event loop."
        ),
    )
    parser.add_argument(
        "--max-cycles",
        type=int,
        default=int(os.environ.get("LOW_COORD_DHT_SEMANTICA_MAX_CYCLES", "50")),
        help=(
            "Hop budget; cycles_used = 1 (querier) + |distinct non-querier "
            "peers contacted|, capped at this value. Default 50 matches the "
            "low-coord and LSH baselines."
        ),
    )
    parser.add_argument("--seeded-peer-docs", default=None)
    parser.add_argument(
        "--retrieval-mode",
        choices=("single_target", "multi_relevant"),
        default=os.environ.get("LOW_COORD_DHT_RETRIEVAL_MODE"),
    )
    parser.add_argument(
        "--relevant-docs-by-qid",
        default=os.environ.get("LOW_COORD_DHT_RELEVANT_DOCS_BY_QID"),
    )

    # Semantica-specific knobs (paper-canonical defaults from Sec VI.B
    # chain-hop retrieval experiment of arXiv:2502.10151).
    parser.add_argument(
        "--semantica-record-limit-per-leafnode",
        type=int,
        default=int(os.environ.get("LOW_COORD_DHT_SEMANTICA_RECORD_LIMIT_PER_LEAFNODE", "50")),
        help="Trie split threshold L (paper Sec V.B; default 50).",
    )
    parser.add_argument(
        "--semantica-delta",
        type=float,
        default=float(os.environ.get("LOW_COORD_DHT_SEMANTICA_DELTA", "0.003")),
        help=(
            "Boundary-clone threshold Delta for K-means splits. "
            "Default 0.003 matches the chain-hop retrieval experiment in "
            "Section VI.B of the paper."
        ),
    )
    parser.add_argument(
        "--semantica-ncu",
        type=int,
        default=int(os.environ.get("LOW_COORD_DHT_SEMANTICA_NCU", "60")),
        help=(
            "Size of the per-peer closest-users list n_cu used by the "
            "chain-hop walk (paper symbol n_cu; default 60)."
        ),
    )
    parser.add_argument(
        "--semantica-expansion-rounds",
        type=int,
        default=int(os.environ.get("LOW_COORD_DHT_SEMANTICA_EXPANSION_ROUNDS", "10")),
        help=(
            "Number of expansion rounds r_max applied to refine each "
            "peer's known-users / closest-users lists (paper Algorithm 2; "
            "default 10 to match the chain-hop retrieval experiment)."
        ),
    )
    parser.add_argument(
        "--semantica-strategy",
        default=os.environ.get("LOW_COORD_DHT_SEMANTICA_STRATEGY", DEFAULT_STRATEGY),
        choices=tuple(sorted(SUPPORTED_STRATEGIES)),
        help="Semantica routing strategy (only closest_user_chain implemented).",
    )

    # Parallelism
    parser.add_argument(
        "--parallelism",
        type=int,
        default=int(os.environ.get("LOW_COORD_DHT_SEMANTICA_PARALLELISM", "60")),
        help=(
            "Number of fork-based worker processes used for the chain-walk "
            "phase. Skip-rule evaluation and metrics flushing stay sequential. "
            "Use 1 to disable the pool (smoke tests, pytest, macOS dev)."
        ),
    )
    parser.add_argument(
        "--parallel-chunk-size",
        type=int,
        default=int(os.environ.get("LOW_COORD_DHT_SEMANTICA_CHUNK_SIZE", "256")),
        help=(
            "Events per worker task. Larger chunks amortize IPC; smaller "
            "chunks balance load when chain-walk cost varies."
        ),
    )

    parser.add_argument(
        "--experiments-registry",
        default=os.environ.get("LOW_COORD_DHT_EXPERIMENTS_REGISTRY", DEFAULT_EXPERIMENTS_REGISTRY),
    )
    args = parser.parse_args()

    # Scenario guard: be loud if anyone tries to mis-use this entrypoint.
    if args.scenario not in ALLOWED_SCENARIOS:
        raise SystemExit(
            f"Semantica baseline only supports scenarios {sorted(ALLOWED_SCENARIOS)}; "
            f"got --scenario={args.scenario}. Use the LSH or semantic runner for "
            "other scenarios."
        )
    if args.dataset == "amazon" and args.scenario not in AMAZON_SUPPORTED_SCENARIOS:
        raise SystemExit(
            f"Amazon dataset only supports scenarios {sorted(AMAZON_SUPPORTED_SCENARIOS)}; "
            f"got --scenario={args.scenario}."
        )
    if args.dataset == "amazon" and args.scenario != "seeded_randomized_timeline":
        raise SystemExit(
            "Semantica chronological scenarios are AOL-only; Amazon qrels carry no real "
            f"timestamps, got dataset=amazon scenario={args.scenario}."
        )
    if args.dataset == "synthetic" and args.scenario != "seeded_randomized_timeline":
        raise SystemExit(
            "Semantica synthetic runs require scenario=seeded_randomized_timeline; "
            f"synthetic events have no real chronology, got scenario={args.scenario}."
        )
    if args.dataset == "pubmedqa" and args.scenario not in PUBMEDQA_SUPPORTED_SCENARIOS:
        raise SystemExit(
            f"PubMedQA dataset only supports scenarios {sorted(PUBMEDQA_SUPPORTED_SCENARIOS)}; "
            f"got --scenario={args.scenario}."
        )
    if args.dataset == "pubmedqa" and args.scenario != "seeded_randomized_timeline":
        raise SystemExit(
            "Semantica PubMedQA runs require scenario=seeded_randomized_timeline; "
            f"PubMedQA events use synthetic timestamps, got scenario={args.scenario}."
        )

    # Amazon does not have a real timeline -> seeded_randomized_timeline IS
    # the canonical seeded scenario; nothing to alias here.
    requested_scenario = args.scenario

    amazon_category = args.amazon_category if args.dataset == "amazon" else None
    retrieval_mode = args.retrieval_mode
    if retrieval_mode is None:
        retrieval_mode = (
            "multi_relevant"
            if args.dataset == "amazon" and args.scenario in SEEDED_SCENARIOS
            else "single_target"
        )
    if retrieval_mode == "multi_relevant" and not (
        args.dataset == "amazon" and args.scenario in SEEDED_SCENARIOS
    ):
        raise SystemExit(
            "retrieval_mode=multi_relevant is only supported for Amazon seeded runs"
        )

    run_id = str(uuid.uuid4())
    started_at = datetime.now(timezone.utc)
    started_at_iso = started_at.isoformat()

    from tqdm import tqdm

    from src.data_root import describe_resolution
    from src.dataset import load_query_events, load_relevant_doc_ids_by_qid
    from src.metrics import MetricsTracker
    from src.semantica_network import (
        SemanticaNetwork,
        SemanticaParallelDriver,
        SemanticaSimulationConfig,
    )

    metadata_path = args.metadata or str(
        _default_metadata_path(
            args.scenario, dataset=args.dataset, amazon_category=amazon_category
        )
    )
    searchable_queries_path = args.searchable_queries or str(
        _default_searchable_queries_path(
            args.scenario, dataset=args.dataset, amazon_category=amazon_category
        )
    )
    embedding_manifest_path = args.embedding_manifest or str(
        _default_embedding_manifest_path(dataset=args.dataset, amazon_category=amazon_category)
    )
    seeded_peer_docs_path = args.seeded_peer_docs or str(
        _default_seeded_peer_docs_path(dataset=args.dataset, amazon_category=amazon_category)
    )
    relevant_docs_by_qid_path = args.relevant_docs_by_qid or (
        str(_default_relevant_docs_by_qid_path(dataset=args.dataset, amazon_category=amazon_category))
        if retrieval_mode == "multi_relevant"
        else None
    )
    print(describe_resolution(args.dataset, amazon_category))

    relevant_doc_ids_by_qid = (
        load_relevant_doc_ids_by_qid(relevant_docs_by_qid_path)
        if retrieval_mode == "multi_relevant" and relevant_docs_by_qid_path
        else {}
    )

    config = SemanticaSimulationConfig(
        scenario=args.scenario,
        record_limit_per_leafnode=args.semantica_record_limit_per_leafnode,
        delta=args.semantica_delta,
        ncu=args.semantica_ncu,
        expansion_rounds=args.semantica_expansion_rounds,
        max_cycles=args.max_cycles,
        strategy=args.semantica_strategy,
        metrics_window=args.metrics_window,
        random_seed=args.random_seed,
        peer_order_seed=args.semantica_peer_order_seed,
        max_owned_docs=0,  # Seeded scenarios never evict; field kept for symmetry.
        retrieval_mode=retrieval_mode,
        relevant_doc_ids_by_qid=relevant_doc_ids_by_qid,
    )

    events = load_query_events(
        metadata_path,
        searchable_only=False,
        max_events=args.max_events or None,
        require_searchable_column=retrieval_mode != "multi_relevant",
        searchable_queries_path=(
            searchable_queries_path if retrieval_mode != "multi_relevant" else None
        ),
    )
    print(
        f"Loaded {len(events):,} events from {metadata_path} "
        f"with searchable annotations from {searchable_queries_path}"
    )

    peer_ids = [event.user_id for event in events]
    network = SemanticaNetwork.from_embedding_manifest(
        peer_ids=peer_ids,
        manifest_path=embedding_manifest_path,
        config=config,
    )

    # The seeded peer-doc map is needed for all current Semantica scenarios:
    # for seeded scenarios it is both topology + initial ownership; for
    # empty_start_chronological it is only the topology seed, and ownership is
    # cleared before the event loop.
    seeded_mapping = _load_seeded_doc_mapping(seeded_peer_docs_path)
    seeded_stats = network.seed_peer_docs(seeded_mapping)

    # Build trie AFTER seeding so per-peer avg embeddings are populated.
    build_started = time.perf_counter()
    network.build_trie()
    topology_seeded_from_peer_future_docs = False
    if args.scenario == "empty_start_chronological":
        topology_seeded_from_peer_future_docs = True
        network.reset_ownership_for_empty_start()
    build_seconds = time.perf_counter() - build_started

    metrics = MetricsTracker(output_dir=args.output_dir, window_size=config.metrics_window)
    effective_parallelism = max(1, args.parallelism)
    if args.scenario == "empty_start_chronological" and effective_parallelism > 1:
        print(
            "[semantica] forcing --parallelism=1 for empty_start_chronological "
            "because document ownership changes after each event",
            flush=True,
        )
        effective_parallelism = 1
    driver = SemanticaParallelDriver(
        network=network,
        parallelism=effective_parallelism,
        chunk_size=max(1, args.parallel_chunk_size),
    )

    query_started = time.perf_counter()
    if driver.parallelism <= 1:
        # Inline path: tqdm-friendly. The parallel path emits its own
        # progress prints from the driver (per-batch).
        for event in tqdm(events, desc="Semantica events"):
            network.process_event(event, metrics)
    else:
        print(
            f"[semantica] running {len(events):,} events through "
            f"{driver.parallelism}-way pool (chunk_size={driver.chunk_size})",
            flush=True,
        )
        driver.run(events, metrics)
    query_seconds = time.perf_counter() - query_started
    payload = metrics.finalize()

    finished_at = datetime.now(timezone.utc)
    finished_at_iso = finished_at.isoformat()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    registry_path_str = args.experiments_registry.strip() if args.experiments_registry else ""
    registry_enabled = bool(registry_path_str)
    registry_path = Path(registry_path_str) if registry_enabled else None

    metric_semantics = {
        "algorithm": ALGORITHM_NAME,
        "cycles_used": (
            "1 (querier; computes the chain start and ranks its neighbors) "
            "+ count of distinct non-querier peers contacted by the "
            "Closest-User-Chain walk. Cap = max_cycles. Aligned with the "
            "low-coord baseline's len(queried_peer_ids) and the LSH "
            "baseline's 1 + len(non_self_contacts)."
        ),
        "result_symbol": (
            "'+' iff a non-querier peer contacted by the chain owns the "
            "clicked target doc; '-' otherwise. Querier-never-counts-local "
            "rule matches the other two baselines."
        ),
        "termination_reason": (
            "'found' if a non-querier neighbor owned the target; "
            "'exhausted' if the chain ran out of unvisited neighbors "
            "(matches src/routing.py); 'cap' if cycles_used reached "
            "max_cycles (matches src/routing.py)."
        ),
        "skip_reasons": (
            "Seeded scenarios: non_searchable (sidecar) and "
            "missing_query_embedding. empty_start_chronological additionally "
            "uses local_held, matching the low-coord chronological branch."
        ),
        "parallelism_caveat": (
            "Cycles count peer contacts (units of coordination), not "
            "sequential round-trips. Closest-User-Chain hops are inherently "
            "sequential per query; the 60-way parallelism here parallelizes "
            "across queries, not within a query. Per-query latency "
            "comparisons require separate analysis."
        ),
    }

    semantica_config_block = {
        "scenario": config.scenario,
        "record_limit_per_leafnode": config.record_limit_per_leafnode,
        "delta": config.delta,
        "ncu": config.ncu,
        "expansion_rounds": config.expansion_rounds,
        "max_cycles": config.max_cycles,
        "strategy": config.strategy,
        "neighbor_rule": "closest_users_top_ncu_after_expansion",
        "paper_section": "VI.B chain-hop retrieval",
        "topology_seeded_from_peer_future_docs": topology_seeded_from_peer_future_docs,
        "peer_order_seed": config.peer_order_seed,
        "trie_input_order_sha256": network.trie_input_order_sha256,
        "trie_input_peer_id_preview": network.trie_input_peer_id_preview,
        "trie_max_depth": network.trie.max_depth if network.trie is not None else 0,
        "trie_leaf_count": len(network.trie.leaves) if network.trie is not None else 0,
        "trie_clone_count": network.trie.clone_count if network.trie is not None else 0,
        "embedding_dim": network.embedding_dim,
        "parallelism": driver.parallelism,
        "parallel_chunk_size": driver.chunk_size,
    }

    notes = {
        "static_trie_justification": (
            "Semantica's topology is built once from the seeded peer-doc "
            "map and reused for every chain walk. For seeded scenarios this "
            "same map is also initial ownership. For empty_start_chronological "
            "ownership is cleared after topology construction and then evolves "
            "with the chronological event stream; this makes that run an "
            "oracle-topology empty-start Semantica variant."
        ),
        "neighbor_rule_justification": (
            "Per-peer neighbors = closest-users top-ncu of the "
            "co-leaves-union, refined by `expansion_rounds` rounds of "
            "Algorithm 2 in the paper (each peer asks a random "
            "closest-user for a closer peer, accepts iff strictly closer "
            "than the current ncu-th closest). Tie-breaking on cosine "
            "ties is by ascending peer_id. With expansion_rounds=0 this "
            "collapses to top-ncu of the static co-leaves union."
        ),
        "semantica_hyperparameters": (
            "Current defaults for the chain-hop retrieval run: L=50 "
            "(record_limit_per_leafnode), Delta=0.003, n_cu=60, "
            "r_max=10, max_hops=50."
        ),
        "macos_caveat": (
            "Parallel querying requires the 'fork' multiprocessing "
            "start method. Run with --parallelism=1 on platforms "
            "without fork (e.g. macOS dev boxes)."
        ),
    }

    run_manifest = {
        "run_id": run_id,
        "algorithm": ALGORITHM_NAME,
        "started_at": started_at_iso,
        "finished_at": finished_at_iso,
        "experiments_registry_path": str(registry_path) if registry_path is not None else None,
        "dataset": args.dataset,
        "amazon_category": amazon_category,
        "metadata": metadata_path,
        "searchable_queries": searchable_queries_path,
        "searchable_mode": "runtime" if retrieval_mode == "multi_relevant" else "sidecar",
        "retrieval_mode": retrieval_mode,
        "relevant_docs_by_qid": relevant_docs_by_qid_path,
        "embedding_manifest": embedding_manifest_path,
        "output_dir": args.output_dir,
        "max_events": args.max_events,
        "scenario": args.scenario,
        "requested_scenario": requested_scenario,
        "max_owned_docs": 0,
        "seeded_peer_docs": seeded_peer_docs_path,
        "seeded_stats": seeded_stats,
        "semantica_config": semantica_config_block,
        "metric_semantics": metric_semantics,
        "notes": notes,
        "config": {
            "scenario": config.scenario,
            "record_limit_per_leafnode": config.record_limit_per_leafnode,
            "delta": config.delta,
            "ncu": config.ncu,
            "expansion_rounds": config.expansion_rounds,
            "max_cycles": config.max_cycles,
            "strategy": config.strategy,
            "retrieval_mode": config.retrieval_mode,
            "metrics_window": config.metrics_window,
            "random_seed": config.random_seed,
            "peer_order_seed": config.peer_order_seed,
            "trie_input_order_sha256": network.trie_input_order_sha256,
            "max_owned_docs": config.max_owned_docs,
            "parallelism": driver.parallelism,
            "parallel_chunk_size": driver.chunk_size,
            "topology_seeded_from_peer_future_docs": topology_seeded_from_peer_future_docs,
        },
        "summary": {
            "executed_queries": payload["executed_queries"],
            "skipped_queries": payload["skipped_queries"],
            "skipped_by_reason": payload["skipped_by_reason"],
            "overall_recall": payload["overall_recall"],
            "overall_retrieval_recall": payload["overall_retrieval_recall"],
            "total_retrieved_relevant_doc_count": payload[
                "total_retrieved_relevant_doc_count"
            ],
            "overall_avg_cycles": payload["overall_avg_cycles"],
            "overall_skip_rate": payload["overall_skip_rate"],
        },
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(run_manifest, indent=2, default=str), encoding="utf-8"
    )

    coord = network.coord
    avg_chain_hops = (
        coord.chain_hops_total / coord.queries_executed if coord.queries_executed else 0.0
    )
    semantica_metrics_payload = {
        "algorithm": ALGORITHM_NAME,
        "queries_executed": coord.queries_executed,
        "queries_skipped": coord.queries_skipped,
        "chain_hops_total": coord.chain_hops_total,
        "avg_chain_hops": avg_chain_hops,
        "chain_found_count": coord.chain_found_count,
        "chain_exhausted_count": coord.chain_exhausted_count,
        "chain_cap_count": coord.chain_cap_count,
        "dead_end_at_querier_count": coord.dead_end_at_querier_count,
        "querier_not_in_trie_count": coord.querier_not_in_trie_count,
        "docs_published": coord.docs_published,
        "avg_neighbor_list_size": network.avg_neighbor_list_size(),
        "initial_known_users_avg": coord.initial_known_users_avg,
        "final_closest_users_avg": coord.final_closest_users_avg,
        "expansion_round_updates_total": coord.expansion_round_updates_total,
        "build_seconds": build_seconds,
        "query_seconds": query_seconds,
        "semantica_config": semantica_config_block,
    }
    (output_dir / "semantica_metrics.json").write_text(
        json.dumps(semantica_metrics_payload, indent=2, default=str), encoding="utf-8"
    )
    print(json.dumps(run_manifest["summary"], indent=2))
    print("Semantica coordination summary:")
    print(json.dumps(semantica_metrics_payload, indent=2, default=str))

    if registry_path is not None:
        registry_record = {
            "run_id": run_id,
            "algorithm": ALGORITHM_NAME,
            "started_at": started_at_iso,
            "finished_at": finished_at_iso,
            "host": socket.gethostname(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "git_commit": _git_commit_or_none(),
            "dataset": args.dataset,
            "amazon_category": amazon_category,
            "scenario": args.scenario,
            "output_dir": str(output_dir.resolve()),
            "run_manifest_path": str((output_dir / "run_manifest.json").resolve()),
            "manifest": run_manifest,
            "semantica_metrics": semantica_metrics_payload,
        }
        _append_to_experiments_registry(registry_path, registry_record)


if __name__ == "__main__":
    main()

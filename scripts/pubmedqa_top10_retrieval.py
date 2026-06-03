#!/usr/bin/env python3
"""Write PubMedQA per-query top-k retrieval outputs for all retrieval baselines."""

from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data_root import resolve_dataset_dir
from src.dataset import load_query_events
from src.lsh_network import LSHDHTNetwork, LSHSimulationConfig
from src.network import EmbeddingLookup, LowCoordSemanticDHTNetwork, SimulationConfig
from src.peer import VectorComparisonCounts
from src.pubmedqa_dataset import (
    PUBMEDQA_ANSWER_TARGETS_FILENAME,
    PUBMEDQA_PEER_FUTURE_DOCS_FILENAME,
    PUBMEDQA_RANDOMIZED_EVENTS_FILENAME,
    PUBMEDQA_SEEDED_SEARCHABLE_FILENAME,
)
from src.routing import run_best_first_search, run_fanout_search
from src.semantica_network import (
    SemanticaNetwork,
    SemanticaSimulationConfig,
    run_closest_user_chain,
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_seeded_mapping(path: Path) -> dict[int, list[str]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    return {int(peer_id): [str(doc_id) for doc_id in doc_ids] for peer_id, doc_ids in payload.items()}


def _load_ownership_injections(path: Path) -> list[dict[str, int | str]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError(f"ownership injection file must contain a list: {path}")
    return [
        {
            "event_index": int(item["event_index"]),
            "peer_id": int(item["peer_id"]),
            "doc_id": str(item["doc_id"]),
        }
        for item in payload
    ]


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _load_answer_targets(path: Path) -> dict[int, dict[str, Any]]:
    return {int(row["qid"]): row for row in _iter_jsonl(path)}


def _rank_doc_ids(
    *,
    doc_ids: Iterable[str],
    query_embedding: np.ndarray,
    doc_lookup: EmbeddingLookup,
    top_k: int,
) -> tuple[list[str], list[float]]:
    unique_doc_ids = sorted({str(doc_id) for doc_id in doc_ids if str(doc_id) in doc_lookup})
    if not unique_doc_ids:
        return [], []
    vectors = np.stack([doc_lookup.get(doc_id) for doc_id in unique_doc_ids], axis=0)
    scores = vectors @ query_embedding
    k = min(top_k, len(unique_doc_ids))
    if k == len(unique_doc_ids):
        best = np.argsort(scores)[::-1]
    else:
        partial = np.argpartition(scores, -k)[-k:]
        best = partial[np.argsort(scores[partial])[::-1]]
    return [unique_doc_ids[int(index)] for index in best], [float(scores[int(index)]) for index in best]


def _base_output_row(
    *,
    event,
    answer_targets: dict[int, dict[str, Any]],
    model: str,
    top_doc_ids: list[str],
    top_scores: list[float],
    skipped_reason: str | None = None,
    search_seconds: float | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    answer = answer_targets.get(event.qid, {})
    row: dict[str, Any] = {
        "model": model,
        "event_index": event.event_index,
        "qid": event.qid,
        "query_key": event.query_text,
        "query_variant": answer.get("query_variant"),
        "rephrase_index": answer.get("rephrase_index"),
        "target_doc_id": event.clicked_doc_id,
        "source_pubid": answer.get("source_pubid", event.clicked_doc_id),
        "final_decision": answer.get("final_decision"),
        "long_answer": answer.get("long_answer"),
        "has_answer": bool(answer.get("has_answer")),
        "top10_doc_ids": top_doc_ids,
        "top10_scores": top_scores,
        "target_in_top10": event.clicked_doc_id in top_doc_ids,
        "skipped_reason": skipped_reason,
    }
    if search_seconds is not None:
        row["search_seconds"] = search_seconds
    if extra:
        row.update(extra)
    return row


def _write_rows(path: Path, rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    success = 0
    skipped = 0
    total_seconds = 0.0
    timed = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            count += 1
            success += int(bool(row.get("target_in_top10")))
            skipped += int(row.get("skipped_reason") is not None)
            if "search_seconds" in row:
                total_seconds += float(row["search_seconds"])
                timed += 1
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return {
        "path": str(path),
        "query_count": count,
        "target_in_top10_count": success,
        "target_in_top10_rate": success / count if count else 0.0,
        "skipped_count": skipped,
        "avg_search_seconds": total_seconds / timed if timed else None,
    }


def _merge_referral(candidate_list: dict[int, Any], referral: Any) -> None:
    current = candidate_list.get(referral.peer_id)
    if current is None or referral.similarity > current.similarity:
        candidate_list[referral.peer_id] = referral


def _best_known_peer_id(candidate_list: dict[int, Any]) -> int | None:
    if not candidate_list:
        return None
    return max(candidate_list.values(), key=lambda item: item.similarity).peer_id


def _run_best_first_search_continue_after_found(
    *,
    querier_peer_id: int,
    target_doc_id: str,
    max_cycles: int,
    query_peer,
    started_at: float,
):
    candidate_list: dict[int, Any] = {}
    queried_peer_ids: list[int] = []
    queried_set: set[int] = set()
    successful_peer_id: int | None = None
    cycles_until_first_found: int | None = None
    seconds_until_first_found: float | None = None

    current_peer_id = querier_peer_id
    termination_reason = "cap"
    while len(queried_peer_ids) < max_cycles:
        response = query_peer(current_peer_id)
        queried_peer_ids.append(current_peer_id)
        queried_set.add(current_peer_id)

        for referral in response.referrals:
            _merge_referral(candidate_list, referral)

        if (
            successful_peer_id is None
            and response.found
            and response.found_doc_id == target_doc_id
        ):
            successful_peer_id = current_peer_id
            cycles_until_first_found = len(queried_peer_ids)
            seconds_until_first_found = time.perf_counter() - started_at

        unqueried = [
            referral for peer_id, referral in candidate_list.items() if peer_id not in queried_set
        ]
        if not unqueried:
            termination_reason = "exhausted"
            break
        current_peer_id = max(unqueried, key=lambda item: item.similarity).peer_id

    from src.routing import SearchResult

    result = SearchResult(
        cycles_used=len(queried_peer_ids),
        result_symbol="+" if successful_peer_id is not None else "-",
        termination_reason=termination_reason,
        queried_peer_ids=queried_peer_ids,
        successful_peer_id=successful_peer_id,
        best_known_peer_id=successful_peer_id or _best_known_peer_id(candidate_list),
    )
    return result, {
        "cycles_until_first_found": cycles_until_first_found,
        "seconds_until_first_found": seconds_until_first_found,
        "first_found_peer_id": successful_peer_id,
        "continued_after_found": True,
        "full_search_termination_reason": termination_reason,
    }


def _low_coord_rows(
    *,
    events,
    manifest_path: Path,
    seeded_mapping: dict[int, list[str]],
    answer_targets: dict[int, dict[str, Any]],
    top_k: int,
    random_seed: int,
    max_neighbors: int,
    max_cycles: int,
    routing_mode: str,
    topology_mode: str,
    search_mode: str,
    fanout: int,
    referrals_per_response: int,
    similarity_threshold: float,
    continue_after_found: bool = False,
    drift_owner_injections: list[dict[str, int | str]] | None = None,
):
    network = LowCoordSemanticDHTNetwork.from_embedding_manifest(
        peer_ids={event.user_id for event in events},
        manifest_path=manifest_path,
        config=SimulationConfig(
            scenario="seeded_randomized_timeline",
            routing_mode=routing_mode,
            topology_mode=topology_mode,
            search_mode=search_mode,
            fanout=fanout,
            max_neighbors=max_neighbors,
            max_cycles=max_cycles,
            referrals_per_response=referrals_per_response,
            similarity_threshold=similarity_threshold,
            random_seed=random_seed,
        ),
    )
    network.seed_peer_docs(seeded_mapping)
    if drift_owner_injections:
        network.set_ownership_injection_schedule(drift_owner_injections)
    network.bootstrap_random_connections()
    for event in events:
        if drift_owner_injections:
            network._apply_pending_ownership_injections(event.event_index)
        if not event.searchable:
            yield _base_output_row(
                event=event,
                answer_targets=answer_targets,
                model="low_coord",
                top_doc_ids=[],
                top_scores=[],
                skipped_reason="non_searchable",
            )
            continue
        if event.query_text not in network.query_lookup:
            yield _base_output_row(
                event=event,
                answer_targets=answer_targets,
                model="low_coord",
                top_doc_ids=[],
                top_scores=[],
                skipped_reason="missing_query_embedding",
            )
            continue

        started_at = time.perf_counter()
        comparison_counts = VectorComparisonCounts()
        query_index = network.query_lookup.index_of(event.query_text)
        query_embedding = network.query_lookup.get_by_index(query_index)
        query_peer = lambda peer_id: network._query_peer(
            peer_id,
            query_embedding=query_embedding,
            query_index=query_index,
            target_doc_id=event.clicked_doc_id,
            original_peer_id=event.user_id,
            comparison_counts=comparison_counts,
        )
        first_found_extra: dict[str, Any] = {}
        if continue_after_found and search_mode == "best_first":
            search_result, first_found_extra = _run_best_first_search_continue_after_found(
                querier_peer_id=event.user_id,
                target_doc_id=event.clicked_doc_id,
                max_cycles=max_cycles,
                query_peer=query_peer,
                started_at=started_at,
            )
        elif search_mode == "fanout":
            search_result = run_fanout_search(
                querier_peer_id=event.user_id,
                target_doc_id=event.clicked_doc_id,
                max_cycles=max_cycles,
                fanout=fanout,
                query_peer=query_peer,
            )
        else:
            search_result = run_best_first_search(
                querier_peer_id=event.user_id,
                target_doc_id=event.clicked_doc_id,
                max_cycles=max_cycles,
                query_peer=query_peer,
            )
        candidate_doc_ids: set[str] = set()
        for peer_id in search_result.queried_peer_ids:
            if peer_id == event.user_id:
                continue
            candidate_doc_ids.update(network.peers[peer_id].owned_docs)
        top_doc_ids, top_scores = _rank_doc_ids(
            doc_ids=candidate_doc_ids,
            query_embedding=query_embedding,
            doc_lookup=network.doc_lookup,
            top_k=top_k,
        )
        elapsed = time.perf_counter() - started_at

        connect_peer_id = search_result.successful_peer_id or search_result.best_known_peer_id
        if (
            topology_mode == "forming_new_connections"
            and connect_peer_id is not None
            and connect_peer_id != event.user_id
        ):
            network.add_connection(event.user_id, connect_peer_id)
        if search_result.result_symbol == "+":
            network._store_clicked_doc(event.user_id, event.clicked_doc_id)

        yield _base_output_row(
            event=event,
            answer_targets=answer_targets,
            model="low_coord",
            top_doc_ids=top_doc_ids,
            top_scores=top_scores,
            search_seconds=elapsed,
            extra={
                "cycles_used": search_result.cycles_used,
                "termination_reason": search_result.termination_reason,
                "candidate_doc_count": len(candidate_doc_ids),
                "doc_vector_comparisons": comparison_counts.doc_vector_comparisons,
                "total_vector_comparisons": comparison_counts.total_vector_comparisons,
                **first_found_extra,
            },
        )


def _semantica_rows(
    *,
    events,
    manifest_path: Path,
    seeded_mapping: dict[int, list[str]],
    answer_targets: dict[int, dict[str, Any]],
    top_k: int,
    random_seed: int,
    max_cycles: int,
    record_limit_per_leafnode: int,
    delta: float,
    ncu: int,
    expansion_rounds: int,
):
    network = SemanticaNetwork.from_embedding_manifest(
        peer_ids={event.user_id for event in events},
        manifest_path=manifest_path,
        config=SemanticaSimulationConfig(
            scenario="seeded_randomized_timeline",
            random_seed=random_seed,
            max_cycles=max_cycles,
            record_limit_per_leafnode=record_limit_per_leafnode,
            delta=delta,
            ncu=ncu,
            expansion_rounds=expansion_rounds,
        ),
    )
    network.seed_peer_docs(seeded_mapping)
    network.build_trie()
    for event in events:
        if not event.searchable:
            yield _base_output_row(
                event=event,
                answer_targets=answer_targets,
                model="semantica",
                top_doc_ids=[],
                top_scores=[],
                skipped_reason="non_searchable",
            )
            continue
        if event.query_text not in network.query_lookup:
            yield _base_output_row(
                event=event,
                answer_targets=answer_targets,
                model="semantica",
                top_doc_ids=[],
                top_scores=[],
                skipped_reason="missing_query_embedding",
            )
            continue
        query_embedding = network.query_lookup.get(event.query_text)
        if event.user_id not in network.snapshot.peer_index:
            result = None
            contacted_peer_ids: list[int] = []
            termination = "querier_not_in_trie"
            cycles = 1
        else:
            result = run_closest_user_chain(
                network.snapshot,
                querier_id=event.user_id,
                target_doc_id=event.clicked_doc_id,
                query_embedding=query_embedding,
            )
            contacted_peer_ids = result.contacted_peer_ids
            termination = result.termination_reason
            cycles = result.cycles_used
        candidate_doc_ids: set[str] = set()
        for peer_id in contacted_peer_ids:
            candidate_doc_ids.update(network.peers[peer_id].owned_docs)
        top_doc_ids, top_scores = _rank_doc_ids(
            doc_ids=candidate_doc_ids,
            query_embedding=query_embedding,
            doc_lookup=network.doc_lookup,
            top_k=top_k,
        )
        yield _base_output_row(
            event=event,
            answer_targets=answer_targets,
            model="semantica",
            top_doc_ids=top_doc_ids,
            top_scores=top_scores,
            extra={
                "cycles_used": cycles,
                "termination_reason": termination,
                "candidate_doc_count": len(candidate_doc_ids),
                "chain_found": bool(result and result.result_symbol == "+"),
            },
        )


def _lsh_rows(
    *,
    events,
    manifest_path: Path,
    seeded_mapping: dict[int, list[str]],
    answer_targets: dict[int, dict[str, Any]],
    top_k: int,
    random_seed: int,
    lsh_k: int,
    lsh_l: int,
    lsh_w: float,
    bucket_replication: int,
):
    network = LSHDHTNetwork.from_embedding_manifest(
        peer_ids={event.user_id for event in events},
        manifest_path=manifest_path,
        config=LSHSimulationConfig(
            scenario="seeded_randomized_timeline",
            random_seed=random_seed,
            k=lsh_k,
            L=lsh_l,
            w=lsh_w,
            bucket_replication=bucket_replication,
        ),
    )
    network.seed_peer_docs(seeded_mapping)
    for event in events:
        if not event.searchable:
            yield _base_output_row(
                event=event,
                answer_targets=answer_targets,
                model="lsh",
                top_doc_ids=[],
                top_scores=[],
                skipped_reason="non_searchable",
            )
            continue
        if event.query_text not in network.query_lookup:
            yield _base_output_row(
                event=event,
                answer_targets=answer_targets,
                model="lsh",
                top_doc_ids=[],
                top_scores=[],
                skipped_reason="missing_query_embedding",
            )
            continue
        query_embedding = network.query_lookup.get(event.query_text)
        candidate_doc_ids: set[str] = set()
        contacted_owners: set[int] = set()
        for table_id, bucket_id in enumerate(network.lsh.hash_tables(query_embedding)):
            contacted_owners.update(network.bucket_owners(table_id, bucket_id))
            bucket = network.buckets.get((table_id, bucket_id))
            if not bucket:
                continue
            for doc_id, owners in bucket.items():
                if any(owner != event.user_id for owner in owners):
                    candidate_doc_ids.add(doc_id)
        top_doc_ids, top_scores = _rank_doc_ids(
            doc_ids=candidate_doc_ids,
            query_embedding=query_embedding,
            doc_lookup=network.doc_lookup,
            top_k=top_k,
        )
        yield _base_output_row(
            event=event,
            answer_targets=answer_targets,
            model="lsh",
            top_doc_ids=top_doc_ids,
            top_scores=top_scores,
            extra={
                "candidate_doc_count": len(candidate_doc_ids),
                "contacted_owner_count": len(contacted_owners - {event.user_id}),
            },
        )


def _hnsw_rows(
    *,
    events,
    manifest_path: Path,
    answer_targets: dict[int, dict[str, Any]],
    top_k: int,
    backend: str,
    ef_search: int,
    hnsw_m: int,
    construction_threads: int,
    search_threads: int,
):
    manifest = _read_json(manifest_path)
    query_lookup = EmbeddingLookup(
        keys_path=manifest["query_keys_path"],
        embeddings_path=manifest["query_embeddings_path"],
    )
    doc_keys = [str(value) for value in np.load(manifest["doc_keys_path"], allow_pickle=True).tolist()]
    doc_embeddings = np.asarray(np.load(manifest["doc_embeddings_path"], mmap_mode="r"), dtype=np.float32)

    backend_name = backend
    index = None
    faiss = None
    if backend == "auto":
        try:
            import hnswlib_trace  # type: ignore

            backend_name = "hnswlib_trace"
        except Exception:
            try:
                import hnswlib  # type: ignore

                backend_name = "hnswlib"
            except Exception:
                backend_name = "exact"
    if backend_name in {"hnswlib", "hnswlib_trace"}:
        if backend_name == "hnswlib_trace":
            import hnswlib_trace as hnsw_backend  # type: ignore
        else:
            import hnswlib as hnsw_backend  # type: ignore

        index = hnsw_backend.Index(space="ip", dim=int(doc_embeddings.shape[1]))
        index.init_index(
            max_elements=int(doc_embeddings.shape[0]),
            ef_construction=max(200, ef_search),
            M=hnsw_m,
        )
        index.add_items(
            doc_embeddings,
            ids=np.arange(doc_embeddings.shape[0]),
            num_threads=construction_threads,
        )
        index.set_ef(ef_search)
    elif backend_name == "faiss":
        import faiss  # type: ignore

        index = faiss.IndexHNSWFlat(int(doc_embeddings.shape[1]), hnsw_m, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efSearch = ef_search
        index.add(doc_embeddings)
    elif backend_name != "exact":
        raise ValueError(f"unsupported HNSW backend {backend!r}")

    for event in events:
        if event.query_text not in query_lookup:
            yield _base_output_row(
                event=event,
                answer_targets=answer_targets,
                model="hnsw",
                top_doc_ids=[],
                top_scores=[],
                skipped_reason="missing_query_embedding",
            )
            continue
        query_embedding = query_lookup.get(event.query_text).astype(np.float32, copy=False)
        started_at = time.perf_counter()
        if backend_name == "hnswlib":
            labels, distances = index.knn_query(
                query_embedding.reshape(1, -1),
                k=top_k,
                num_threads=search_threads,
            )
            labels = labels[0]
            scores = 1.0 - distances[0]
        elif backend_name == "faiss":
            scores, labels = index.search(query_embedding.reshape(1, -1), top_k)
            labels = labels[0]
            scores = scores[0]
        else:
            scores = doc_embeddings @ query_embedding
            k = min(top_k, scores.shape[0])
            labels = np.argpartition(scores, -k)[-k:]
            labels = labels[np.argsort(scores[labels])[::-1]]
            scores = scores[labels]
        elapsed = time.perf_counter() - started_at
        top_doc_ids = [doc_keys[int(index)] for index in labels.tolist()]
        top_scores = [float(value) for value in scores.tolist()]
        yield _base_output_row(
            event=event,
            answer_targets=answer_targets,
            model="hnsw",
            top_doc_ids=top_doc_ids,
            top_scores=top_scores,
            search_seconds=elapsed,
            extra={"hnsw_backend": backend_name, "ef_search": ef_search},
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PubMedQA top-k retrieval baselines")
    parser.add_argument("--dataset-dir", type=Path, default=None)
    parser.add_argument("--metadata", type=Path, default=None)
    parser.add_argument("--searchable-queries", type=Path, default=None)
    parser.add_argument("--embedding-manifest", type=Path, default=None)
    parser.add_argument("--seeded-peer-docs", type=Path, default=None)
    parser.add_argument("--answer-targets", type=Path, default=None)
    parser.add_argument("--drift-owner-injections", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--models", default="low_coord,semantica,lsh,hnsw")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--max-events", type=int, default=0)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--max-neighbors", type=int, default=60)
    parser.add_argument("--max-cycles", type=int, default=50)
    parser.add_argument("--routing-mode", choices=("semantic_query_memory", "replica_only"), default="semantic_query_memory")
    parser.add_argument("--topology-mode", choices=("forming_new_connections", "static_initial_connections"), default="forming_new_connections")
    parser.add_argument("--search-mode", choices=("best_first", "fanout"), default="best_first")
    parser.add_argument("--fanout", type=int, default=1)
    parser.add_argument("--referrals-per-response", type=int, default=10)
    parser.add_argument("--similarity-threshold", type=float, default=0.5)
    parser.add_argument("--low-coord-continue-after-found", action="store_true")
    parser.add_argument("--lsh-k", type=int, default=10)
    parser.add_argument("--lsh-L", type=int, default=20)
    parser.add_argument("--lsh-w", type=float, default=0.75)
    parser.add_argument("--bucket-replication", type=int, default=1)
    parser.add_argument("--semantica-record-limit-per-leafnode", type=int, default=50)
    parser.add_argument("--semantica-delta", type=float, default=0.003)
    parser.add_argument("--semantica-ncu", type=int, default=60)
    parser.add_argument("--semantica-expansion-rounds", type=int, default=10)
    parser.add_argument(
        "--hnsw-backend",
        choices=("auto", "hnswlib_trace", "hnswlib", "faiss", "exact"),
        default="auto",
    )
    parser.add_argument("--hnsw-ef-search", type=int, default=128)
    parser.add_argument("--hnsw-m", type=int, default=16)
    parser.add_argument("--construction-threads", type=int, default=8)
    parser.add_argument("--search-threads", type=int, default=8)
    args = parser.parse_args()

    if args.top_k <= 0:
        raise SystemExit("--top-k must be positive")
    dataset_dir = args.dataset_dir or resolve_dataset_dir("pubmedqa")
    metadata_path = args.metadata or dataset_dir / PUBMEDQA_RANDOMIZED_EVENTS_FILENAME
    searchable_path = args.searchable_queries or dataset_dir / PUBMEDQA_SEEDED_SEARCHABLE_FILENAME
    manifest_path = args.embedding_manifest or dataset_dir / "embeddings" / "manifest.json"
    seeded_path = args.seeded_peer_docs or dataset_dir / PUBMEDQA_PEER_FUTURE_DOCS_FILENAME
    answer_targets_path = args.answer_targets or dataset_dir / PUBMEDQA_ANSWER_TARGETS_FILENAME
    events = load_query_events(
        metadata_path,
        max_events=args.max_events or None,
        require_searchable_column=True,
        searchable_queries_path=searchable_path,
    )
    answer_targets = _load_answer_targets(answer_targets_path)
    seeded_mapping = _load_seeded_mapping(seeded_path)
    drift_owner_injections = (
        _load_ownership_injections(args.drift_owner_injections)
        if args.drift_owner_injections
        else None
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries: dict[str, Any] = {
        "dataset": "pubmedqa",
        "metadata_path": str(metadata_path),
        "searchable_queries_path": str(searchable_path),
        "embedding_manifest_path": str(manifest_path),
        "seeded_peer_docs_path": str(seeded_path),
        "answer_targets_path": str(answer_targets_path),
        "drift_owner_injections_path": (
            str(args.drift_owner_injections) if args.drift_owner_injections else None
        ),
        "drift_owner_injection_count": (
            len(drift_owner_injections) if drift_owner_injections else 0
        ),
        "top_k": args.top_k,
        "max_events": args.max_events,
        "models": {},
    }

    requested_models = [model.strip() for model in args.models.split(",") if model.strip()]
    for model in requested_models:
        output_path = args.output_dir / f"pubmedqa_top{args.top_k}_{model}.jsonl"
        if model == "low_coord":
            rows = _low_coord_rows(
                events=events,
                manifest_path=manifest_path,
                seeded_mapping=seeded_mapping,
                answer_targets=answer_targets,
                top_k=args.top_k,
                random_seed=args.random_seed,
                max_neighbors=args.max_neighbors,
                max_cycles=args.max_cycles,
                routing_mode=args.routing_mode,
                topology_mode=args.topology_mode,
                search_mode=args.search_mode,
                fanout=args.fanout,
                referrals_per_response=args.referrals_per_response,
                similarity_threshold=args.similarity_threshold,
                continue_after_found=args.low_coord_continue_after_found,
                drift_owner_injections=drift_owner_injections,
            )
        elif model == "semantica":
            rows = _semantica_rows(
                events=events,
                manifest_path=manifest_path,
                seeded_mapping=seeded_mapping,
                answer_targets=answer_targets,
                top_k=args.top_k,
                random_seed=args.random_seed,
                max_cycles=args.max_cycles,
                record_limit_per_leafnode=args.semantica_record_limit_per_leafnode,
                delta=args.semantica_delta,
                ncu=args.semantica_ncu,
                expansion_rounds=args.semantica_expansion_rounds,
            )
        elif model == "lsh":
            rows = _lsh_rows(
                events=events,
                manifest_path=manifest_path,
                seeded_mapping=seeded_mapping,
                answer_targets=answer_targets,
                top_k=args.top_k,
                random_seed=args.random_seed,
                lsh_k=args.lsh_k,
                lsh_l=args.lsh_L,
                lsh_w=args.lsh_w,
                bucket_replication=args.bucket_replication,
            )
        elif model == "hnsw":
            rows = _hnsw_rows(
                events=events,
                manifest_path=manifest_path,
                answer_targets=answer_targets,
                top_k=args.top_k,
                backend=args.hnsw_backend,
                ef_search=args.hnsw_ef_search,
                hnsw_m=args.hnsw_m,
                construction_threads=args.construction_threads,
                search_threads=args.search_threads,
            )
        else:
            raise SystemExit(f"unknown model {model!r}")
        summaries["models"][model] = _write_rows(output_path, rows)
        print(f"[pubmedqa-top10] wrote {model} rows to {output_path}", flush=True)

    summary_path = args.output_dir / "pubmedqa_top10_summary.json"
    summary_path.write_text(json.dumps(summaries, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[pubmedqa-top10] wrote summary to {summary_path}")


if __name__ == "__main__":
    main()


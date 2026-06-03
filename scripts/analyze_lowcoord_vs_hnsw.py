#!/usr/bin/env python3
"""Estimate Low-Coord comparison work and compare it with a dataset-level HNSW.

The Low-Coord simulator records hops/cycles, not the number of document vectors
scored while producing referrals. This script keeps that estimate separate from
the original result artifacts and adds one HNSW sweep per dataset. HNSW is a
plain dataset-level ANN baseline: it indexes every document that appears in the
dataset's seeded-searchable query set, then searches a consistent randomized
timeline query set for that dataset.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import sys
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.run_simulation import (  # noqa: E402
    _default_embedding_manifest_path,
    _default_metadata_path_for_scenario,
    _default_searchable_queries_path_for_scenario,
)
from src.dataset import QueryEvent, iter_query_events  # noqa: E402
from src.scenarios import SEEDED_SCENARIOS  # noqa: E402


DEFAULT_RESULTS_ROOT = Path("/var/scratch/pmneague/Low_Coord_Semantic_DHT/results")
DEFAULT_OUTPUT_DIRNAME = "hnsw_comparison_analysis"
HNSW_SCENARIO = "seeded_randomized_timeline"
DEFAULT_EF_SEARCH_VALUES = (8, 16, 32, 64, 128, 256, 512, 1024)
_UNIQUE_USER_COUNT_CACHE: dict[tuple[str, int, str], int] = {}


@dataclass(frozen=True, slots=True)
class ResultPaths:
    run_dir: Path
    manifest_path: Path
    metrics_path: Path
    query_log_path: Path


@dataclass(frozen=True, slots=True)
class LowCoordRun:
    run_dir: Path
    dataset: str
    amazon_category: str | None
    dataset_key: str
    scenario: str
    requested_scenario: str
    routing_mode: str
    topology_mode: str
    selection_mode: str
    max_neighbors: int
    max_routed_queries: int
    max_owned_docs: int
    max_events: int
    metadata_path: Path
    searchable_queries_path: Path | None
    embedding_manifest_path: Path
    seeded_peer_docs_path: Path | None
    executed_queries: int
    skipped_queries: int
    overall_recall: float
    overall_avg_cycles: float
    overall_skip_rate: float


@dataclass(frozen=True, slots=True)
class OwnershipEstimate:
    peer_count: int
    avg_docs_per_peer: float
    avg_docs_per_nonempty_peer: float
    avg_nonempty_peer_count: float
    method: str


@dataclass(frozen=True, slots=True)
class LowCoordEstimate:
    run: LowCoordRun
    ownership: OwnershipEstimate
    scanned_neighbors_per_cycle: float
    estimated_doc_comparisons_per_query: float
    estimated_routed_query_comparisons_upper_bound_per_query: float | None
    estimate_notes: str


@dataclass(frozen=True, slots=True)
class HnswDatasetTarget:
    dataset_key: str
    dataset: str
    amazon_category: str | None
    target_run_dir: Path
    target_recall: float
    embedding_manifest_path: Path
    metadata_path: Path
    searchable_queries_path: Path


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _open_json_gz_or_json(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def _dataset_key(dataset: str, amazon_category: str | None) -> str:
    if dataset == "amazon":
        return f"amazon_{amazon_category or 'unknown'}"
    return dataset


def _as_path(value: object | None) -> Path | None:
    if value is None:
        return None
    text = str(value).strip()
    return Path(text) if text else None


def _find_result_paths(results_root: Path, *, include_smoke: bool) -> list[ResultPaths]:
    result_paths: list[ResultPaths] = []
    for manifest_path in sorted(results_root.rglob("run_manifest.json")):
        run_dir = manifest_path.parent
        if not include_smoke and "smoke" in run_dir.name:
            continue
        metrics_path = run_dir / "metrics.json"
        query_log_path = run_dir / "query_log.jsonl"
        if metrics_path.exists() and query_log_path.exists():
            result_paths.append(
                ResultPaths(
                    run_dir=run_dir,
                    manifest_path=manifest_path,
                    metrics_path=metrics_path,
                    query_log_path=query_log_path,
                )
            )
    return result_paths


def _load_lowcoord_runs(results_root: Path, *, include_smoke: bool) -> list[LowCoordRun]:
    runs: list[LowCoordRun] = []
    for paths in _find_result_paths(results_root, include_smoke=include_smoke):
        manifest = _read_json(paths.manifest_path)
        metrics = _read_json(paths.metrics_path)
        if "routing_mode" not in manifest:
            continue
        if manifest.get("algorithm") in {"e2lsh_dht", "semantica"}:
            continue

        dataset = str(manifest.get("dataset", "aol4foltr"))
        amazon_category = manifest.get("amazon_category")
        amazon_category = str(amazon_category) if amazon_category else None
        config = manifest.get("config") or {}
        summary = manifest.get("summary") or {}
        metadata_path = _as_path(manifest.get("metadata"))
        if metadata_path is None:
            continue

        runs.append(
            LowCoordRun(
                run_dir=paths.run_dir,
                dataset=dataset,
                amazon_category=amazon_category,
                dataset_key=_dataset_key(dataset, amazon_category),
                scenario=str(manifest.get("scenario", config.get("scenario", ""))),
                requested_scenario=str(
                    manifest.get("requested_scenario", manifest.get("scenario", ""))
                ),
                routing_mode=str(manifest.get("routing_mode", config.get("routing_mode", ""))),
                topology_mode=str(manifest.get("topology_mode", config.get("topology_mode", ""))),
                selection_mode=str(
                    manifest.get("selection_mode", config.get("selection_mode", "similarity"))
                ),
                max_neighbors=int(config.get("max_neighbors", 60)),
                max_routed_queries=int(config.get("max_routed_queries", 0)),
                max_owned_docs=int(config.get("max_owned_docs", manifest.get("max_owned_docs") or 0)),
                max_events=int(manifest.get("max_events") or 0),
                metadata_path=metadata_path,
                searchable_queries_path=_as_path(manifest.get("searchable_queries")),
                embedding_manifest_path=Path(str(manifest.get("embedding_manifest"))),
                seeded_peer_docs_path=_as_path(manifest.get("seeded_peer_docs")),
                executed_queries=int(metrics.get("executed_queries", summary.get("executed_queries", 0))),
                skipped_queries=int(metrics.get("skipped_queries", summary.get("skipped_queries", 0))),
                overall_recall=float(
                    metrics.get("overall_recall", summary.get("overall_recall", 0.0))
                ),
                overall_avg_cycles=float(
                    metrics.get("overall_avg_cycles", summary.get("overall_avg_cycles", 0.0))
                ),
                overall_skip_rate=float(
                    metrics.get("overall_skip_rate", summary.get("overall_skip_rate", 0.0))
                ),
            )
        )
    return runs


def _load_executed_event_indices(query_log_path: Path) -> set[int]:
    indices: set[int] = set()
    with query_log_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            if not raw_line.strip():
                continue
            indices.add(int(json.loads(raw_line)["event_index"]))
    return indices


def _iter_run_events(run: LowCoordRun) -> Iterator[QueryEvent]:
    yield from iter_query_events(
        run.metadata_path,
        searchable_only=False,
        require_searchable_column=run.searchable_queries_path is not None,
        searchable_queries_path=run.searchable_queries_path,
    )


def _count_unique_users(run: LowCoordRun) -> int:
    cache_key = (
        str(run.metadata_path),
        int(run.max_events),
        str(run.searchable_queries_path or ""),
    )
    cached = _UNIQUE_USER_COUNT_CACHE.get(cache_key)
    if cached is not None:
        return cached
    users: set[int] = set()
    for count, event in enumerate(_iter_run_events(run), start=1):
        users.add(event.user_id)
        if run.max_events and count >= run.max_events:
            break
    result = len(users)
    _UNIQUE_USER_COUNT_CACHE[cache_key] = result
    return result


def _estimate_seeded_ownership(run: LowCoordRun) -> OwnershipEstimate:
    if run.seeded_peer_docs_path is None or not run.seeded_peer_docs_path.exists():
        return OwnershipEstimate(
            peer_count=0,
            avg_docs_per_peer=0.0,
            avg_docs_per_nonempty_peer=0.0,
            avg_nonempty_peer_count=0.0,
            method="seeded_peer_docs_missing",
        )
    with _open_json_gz_or_json(run.seeded_peer_docs_path) as handle:
        mapping = json.load(handle)
    nonempty_peer_count = len(mapping)
    total_docs = sum(len(doc_ids) for doc_ids in mapping.values())
    # Seeded mappings are built from the same eventual clicked-doc stream used
    # by the seeded simulations, so every simulated peer should have at least
    # one seeded doc. Avoid rescanning large metadata just to rediscover this.
    peer_count = nonempty_peer_count
    return OwnershipEstimate(
        peer_count=peer_count,
        avg_docs_per_peer=(total_docs / peer_count) if peer_count else 0.0,
        avg_docs_per_nonempty_peer=(total_docs / nonempty_peer_count)
        if nonempty_peer_count
        else 0.0,
        avg_nonempty_peer_count=float(nonempty_peer_count),
        method="seeded_peer_docs",
    )


def _estimate_runtime_ownership(run: LowCoordRun) -> OwnershipEstimate:
    query_log_path = run.run_dir / "query_log.jsonl"
    executed_indices = _load_executed_event_indices(query_log_path)
    if not executed_indices:
        return OwnershipEstimate(
            peer_count=0,
            avg_docs_per_peer=0.0,
            avg_docs_per_nonempty_peer=0.0,
            avg_nonempty_peer_count=0.0,
            method="runtime_no_executed_queries",
        )
    last_needed = max(executed_indices)
    docs_by_peer: dict[int, set[str] | OrderedDict[str, None]] = {}
    users: set[int] = set()
    total_owned_docs = 0
    nonempty_peer_count = 0
    sample_total_owned_docs = 0
    sample_nonempty_peers = 0
    sample_count = 0
    max_owned_docs = run.max_owned_docs if run.max_owned_docs > 0 else 0

    for count, event in enumerate(_iter_run_events(run), start=1):
        users.add(event.user_id)
        if event.event_index in executed_indices:
            sample_total_owned_docs += total_owned_docs
            sample_nonempty_peers += nonempty_peer_count
            sample_count += 1

        peer_docs = docs_by_peer.get(event.user_id)
        if peer_docs is None:
            peer_docs = OrderedDict() if max_owned_docs else set()
            docs_by_peer[event.user_id] = peer_docs

        if isinstance(peer_docs, OrderedDict):
            already_present = event.clicked_doc_id in peer_docs
            was_empty = len(peer_docs) == 0
            peer_docs[event.clicked_doc_id] = None
            peer_docs.move_to_end(event.clicked_doc_id)
            if not already_present:
                total_owned_docs += 1
                if was_empty:
                    nonempty_peer_count += 1
            if len(peer_docs) > max_owned_docs:
                peer_docs.popitem(last=False)
                total_owned_docs -= 1
        elif event.clicked_doc_id not in peer_docs:
            if not peer_docs:
                nonempty_peer_count += 1
            peer_docs.add(event.clicked_doc_id)
            total_owned_docs += 1

        if run.max_events and count >= run.max_events:
            break
        if event.event_index >= last_needed:
            # Keep scanning to max_events is unnecessary for query-time ownership samples.
            break

    peer_count = len(users)
    avg_total_owned = sample_total_owned_docs / sample_count if sample_count else 0.0
    avg_nonempty = sample_nonempty_peers / sample_count if sample_count else 0.0
    return OwnershipEstimate(
        peer_count=peer_count,
        avg_docs_per_peer=(avg_total_owned / peer_count) if peer_count else 0.0,
        avg_docs_per_nonempty_peer=(avg_total_owned / avg_nonempty) if avg_nonempty else 0.0,
        avg_nonempty_peer_count=avg_nonempty,
        method="runtime_replay_at_executed_queries",
    )


def estimate_lowcoord_run(run: LowCoordRun) -> LowCoordEstimate:
    print(
        f"[lowcoord] estimating {run.dataset_key} {run.run_dir.name} "
        f"scenario={run.scenario} selection={run.selection_mode}",
        flush=True,
    )
    if run.selection_mode == "random_walk":
        return LowCoordEstimate(
            run=run,
            ownership=OwnershipEstimate(
                peer_count=0,
                avg_docs_per_peer=0.0,
                avg_docs_per_nonempty_peer=0.0,
                avg_nonempty_peer_count=0.0,
                method="not_applicable_random_walk",
            ),
            scanned_neighbors_per_cycle=0.0,
            estimated_doc_comparisons_per_query=0.0,
            estimated_routed_query_comparisons_upper_bound_per_query=0.0,
            estimate_notes=(
                "random_walk samples active peers globally and checks target-doc "
                "ownership directly"
            ),
        )

    if run.scenario in SEEDED_SCENARIOS:
        ownership = _estimate_seeded_ownership(run)
    else:
        ownership = _estimate_runtime_ownership(run)

    scanned_neighbors = float(run.max_neighbors)
    doc_comparisons = run.overall_avg_cycles * scanned_neighbors * ownership.avg_docs_per_peer
    notes = (
        "estimated as avg_cycles * max_neighbors * time-weighted avg docs per peer; "
        "uses max_neighbors because bootstrap gives each queried peer up to that many "
        "outgoing connections"
    )

    routed_upper = None
    if run.routing_mode == "semantic_query_memory":
        routed_upper = run.overall_avg_cycles * run.max_routed_queries

    return LowCoordEstimate(
        run=run,
        ownership=ownership,
        scanned_neighbors_per_cycle=scanned_neighbors,
        estimated_doc_comparisons_per_query=doc_comparisons,
        estimated_routed_query_comparisons_upper_bound_per_query=routed_upper,
        estimate_notes=notes,
    )


def _write_lowcoord_estimates(path: Path, estimates: Iterable[LowCoordEstimate]) -> None:
    fieldnames = (
        "dataset_key",
        "dataset",
        "amazon_category",
        "run_dir",
        "scenario",
        "requested_scenario",
        "routing_mode",
        "topology_mode",
        "selection_mode",
        "executed_queries",
        "skipped_queries",
        "overall_recall",
        "overall_avg_cycles",
        "overall_skip_rate",
        "max_neighbors",
        "max_owned_docs",
        "ownership_method",
        "peer_count",
        "avg_docs_per_peer",
        "avg_docs_per_nonempty_peer",
        "avg_nonempty_peer_count",
        "scanned_neighbors_per_cycle",
        "estimated_doc_comparisons_per_query",
        "estimated_routed_query_comparisons_upper_bound_per_query",
        "estimate_notes",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for estimate in estimates:
            run = estimate.run
            writer.writerow(
                {
                    "dataset_key": run.dataset_key,
                    "dataset": run.dataset,
                    "amazon_category": run.amazon_category or "",
                    "run_dir": str(run.run_dir),
                    "scenario": run.scenario,
                    "requested_scenario": run.requested_scenario,
                    "routing_mode": run.routing_mode,
                    "topology_mode": run.topology_mode,
                    "selection_mode": run.selection_mode,
                    "executed_queries": run.executed_queries,
                    "skipped_queries": run.skipped_queries,
                    "overall_recall": run.overall_recall,
                    "overall_avg_cycles": run.overall_avg_cycles,
                    "overall_skip_rate": run.overall_skip_rate,
                    "max_neighbors": run.max_neighbors,
                    "max_owned_docs": run.max_owned_docs,
                    "ownership_method": estimate.ownership.method,
                    "peer_count": estimate.ownership.peer_count,
                    "avg_docs_per_peer": estimate.ownership.avg_docs_per_peer,
                    "avg_docs_per_nonempty_peer": estimate.ownership.avg_docs_per_nonempty_peer,
                    "avg_nonempty_peer_count": estimate.ownership.avg_nonempty_peer_count,
                    "scanned_neighbors_per_cycle": estimate.scanned_neighbors_per_cycle,
                    "estimated_doc_comparisons_per_query": (
                        estimate.estimated_doc_comparisons_per_query
                    ),
                    "estimated_routed_query_comparisons_upper_bound_per_query": (
                        ""
                        if estimate.estimated_routed_query_comparisons_upper_bound_per_query is None
                        else estimate.estimated_routed_query_comparisons_upper_bound_per_query
                    ),
                    "estimate_notes": estimate.estimate_notes,
                }
            )


def _best_runs_by_dataset(estimates: Iterable[LowCoordEstimate]) -> dict[str, LowCoordRun]:
    best: dict[str, LowCoordRun] = {}
    best_comparisons: dict[str, float] = {}
    for estimate in estimates:
        run = estimate.run
        existing = best.get(run.dataset_key)
        if existing is None:
            best[run.dataset_key] = run
            best_comparisons[run.dataset_key] = estimate.estimated_doc_comparisons_per_query
            continue
        current_key = (
            run.overall_recall,
            -run.overall_avg_cycles,
            -estimate.estimated_doc_comparisons_per_query,
            str(run.run_dir),
        )
        existing_key = (
            existing.overall_recall,
            -existing.overall_avg_cycles,
            -best_comparisons.get(run.dataset_key, 0.0),
            str(existing.run_dir),
        )
        if current_key > existing_key:
            best[run.dataset_key] = run
            best_comparisons[run.dataset_key] = estimate.estimated_doc_comparisons_per_query
    return best


def _dataset_hnsw_target(best_run: LowCoordRun) -> HnswDatasetTarget:
    metadata_path = _default_metadata_path_for_scenario(
        HNSW_SCENARIO,
        dataset=best_run.dataset,
        amazon_category=best_run.amazon_category,
    )
    searchable_path = _default_searchable_queries_path_for_scenario(
        HNSW_SCENARIO,
        dataset=best_run.dataset,
        amazon_category=best_run.amazon_category,
    )
    if searchable_path is None:
        raise ValueError(f"{best_run.dataset_key} has no seeded searchable sidecar")
    embedding_manifest_path = _default_embedding_manifest_path(
        dataset=best_run.dataset,
        amazon_category=best_run.amazon_category,
    )
    return HnswDatasetTarget(
        dataset_key=best_run.dataset_key,
        dataset=best_run.dataset,
        amazon_category=best_run.amazon_category,
        target_run_dir=best_run.run_dir,
        target_recall=best_run.overall_recall,
        embedding_manifest_path=embedding_manifest_path,
        metadata_path=metadata_path,
        searchable_queries_path=searchable_path,
    )


def _import_hnsw_backend(preferred_backend: str):
    if preferred_backend in {"auto", "hnswlib_trace"}:
        try:
            import hnswlib_trace  # type: ignore

            return "hnswlib_trace", hnswlib_trace
        except Exception:
            if preferred_backend == "hnswlib_trace":
                raise
    if preferred_backend in {"auto", "faiss"}:
        try:
            import faiss  # type: ignore

            return "faiss", faiss
        except Exception:
            if preferred_backend == "faiss":
                raise
    if preferred_backend in {"auto", "hnswlib"}:
        try:
            import hnswlib  # type: ignore

            return "hnswlib", hnswlib
        except Exception:
            if preferred_backend == "hnswlib":
                raise
    raise RuntimeError(
        "No HNSW backend is importable; install vendor/hnswlib_trace, faiss-cpu, or hnswlib."
    )


def _load_embedding_manifest(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_hnsw_query_and_doc_subset(
    target: HnswDatasetTarget,
    max_queries: int,
) -> tuple[np.ndarray, list[str], np.ndarray, np.ndarray, list[str]]:
    manifest = _load_embedding_manifest(target.embedding_manifest_path)
    query_keys = np.load(Path(manifest["query_keys_path"]), mmap_mode="r")
    query_embeddings = np.load(Path(manifest["query_embeddings_path"]), mmap_mode="r")
    doc_keys = np.load(Path(manifest["doc_keys_path"]), mmap_mode="r")
    doc_embeddings = np.load(Path(manifest["doc_embeddings_path"]), mmap_mode="r")

    query_index = {str(key): idx for idx, key in enumerate(query_keys.tolist())}
    doc_index = {str(key): idx for idx, key in enumerate(doc_keys.tolist())}

    retained_queries: list[np.ndarray] = []
    retained_targets: list[str] = []
    searchable_doc_ids: set[str] = set()
    skipped_missing_query = 0
    skipped_missing_doc = 0

    for event in iter_query_events(
        target.metadata_path,
        searchable_only=True,
        require_searchable_column=True,
        searchable_queries_path=target.searchable_queries_path,
    ):
        qidx = query_index.get(event.query_text)
        didx = doc_index.get(event.clicked_doc_id)
        if qidx is None:
            skipped_missing_query += 1
            continue
        if didx is None:
            skipped_missing_doc += 1
            continue
        searchable_doc_ids.add(event.clicked_doc_id)
        if not max_queries or len(retained_queries) < max_queries:
            retained_queries.append(np.asarray(query_embeddings[qidx], dtype=np.float32))
            retained_targets.append(event.clicked_doc_id)

    if not retained_queries:
        raise RuntimeError(f"No HNSW queries retained for {target.dataset_key}")

    ordered_doc_ids = sorted(searchable_doc_ids)
    subset_row_by_doc_id = {doc_id: idx for idx, doc_id in enumerate(ordered_doc_ids)}
    subset_indices = np.asarray([doc_index[doc_id] for doc_id in ordered_doc_ids], dtype=np.int64)
    subset_embeddings = np.asarray(doc_embeddings[subset_indices], dtype=np.float32)
    query_matrix = np.vstack(retained_queries).astype(np.float32, copy=False)
    target_indices = np.asarray(
        [subset_row_by_doc_id[doc_id] for doc_id in retained_targets],
        dtype=np.int64,
    )
    print(
        f"[hnsw:{target.dataset_key}] retained_queries={len(retained_queries):,} "
        f"indexed_docs={len(ordered_doc_ids):,} skipped_missing_query={skipped_missing_query:,} "
        f"skipped_missing_doc={skipped_missing_doc:,}",
        flush=True,
    )
    return subset_embeddings, ordered_doc_ids, query_matrix, target_indices, retained_targets


def _recall_rows_from_labels(
    labels: np.ndarray,
    target_indices: np.ndarray,
    *,
    top_k_values: tuple[int, ...],
    base_row: Mapping[str, Any],
    comparison_proxy: str,
    comparison_proxy_value_per_query: float | int | str,
    distance_computations_total: int | str = "",
    avg_distance_computations_per_query: float | str = "",
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    query_count = int(labels.shape[0])
    for top_k in top_k_values:
        if top_k > labels.shape[1]:
            raise ValueError(f"top_k={top_k} exceeds label width {labels.shape[1]}")
        label_view = np.ascontiguousarray(labels[:, :top_k])
        successes = int((labels[:, :top_k] == target_indices[:, None]).any(axis=1).sum())
        rows.append(
            {
                **dict(base_row),
                "top_k": top_k,
                "query_count": query_count,
                "recall": successes / query_count if query_count else 0.0,
                "successes": successes,
                "distance_computations_total": distance_computations_total,
                "avg_distance_computations_per_query": avg_distance_computations_per_query,
                "comparison_proxy": comparison_proxy,
                "comparison_proxy_value_per_query": comparison_proxy_value_per_query,
                "labels_sha256": hashlib.sha256(label_view.tobytes()).hexdigest(),
            }
        )
    return rows


def _search_faiss_flat(
    faiss: Any,
    doc_embeddings: np.ndarray,
    query_embeddings: np.ndarray,
    target_indices: np.ndarray,
    *,
    top_k_values: tuple[int, ...],
    batch_size: int,
) -> list[dict[str, Any]]:
    dim = int(doc_embeddings.shape[1])
    max_top_k = max(top_k_values)
    index = faiss.IndexFlatIP(dim)
    index.add(doc_embeddings)
    all_labels: list[np.ndarray] = []
    for start in range(0, query_embeddings.shape[0], batch_size):
        end = min(start + batch_size, query_embeddings.shape[0])
        _scores, labels = index.search(query_embeddings[start:end], max_top_k)
        all_labels.append(labels)
    labels_matrix = np.vstack(all_labels)
    rows = _recall_rows_from_labels(
        labels_matrix,
        target_indices,
        top_k_values=top_k_values,
        base_row={"backend": "faiss_flat", "ef_search": ""},
        comparison_proxy="exact_indexed_doc_count",
        comparison_proxy_value_per_query=int(doc_embeddings.shape[0]),
        distance_computations_total=int(doc_embeddings.shape[0] * query_embeddings.shape[0]),
        avg_distance_computations_per_query=int(doc_embeddings.shape[0]),
    )
    for row in rows:
        print(
            f"[hnsw:faiss_flat] top_k={row['top_k']} recall={row['recall']:.6f} "
            f"proxy={row['comparison_proxy_value_per_query']}",
            flush=True,
        )
    return rows


def _search_faiss_hnsw(
    faiss: Any,
    doc_embeddings: np.ndarray,
    query_embeddings: np.ndarray,
    target_indices: np.ndarray,
    *,
    ef_search_values: Iterable[int],
    hnsw_m: int,
    top_k_values: tuple[int, ...],
    batch_size: int,
) -> list[dict[str, Any]]:
    dim = int(doc_embeddings.shape[1])
    max_top_k = max(top_k_values)
    index = faiss.IndexHNSWFlat(dim, hnsw_m, faiss.METRIC_INNER_PRODUCT)
    index.add(doc_embeddings)
    rows: list[dict[str, Any]] = []
    for ef_search in ef_search_values:
        index.hnsw.efSearch = int(ef_search)
        if hasattr(faiss, "cvar") and hasattr(faiss.cvar, "hnsw_stats"):
            faiss.cvar.hnsw_stats.reset()
        all_labels: list[np.ndarray] = []
        for start in range(0, query_embeddings.shape[0], batch_size):
            end = min(start + batch_size, query_embeddings.shape[0])
            _scores, labels = index.search(query_embeddings[start:end], max_top_k)
            all_labels.append(labels)
        labels_matrix = np.vstack(all_labels)
        ndis = None
        if hasattr(faiss, "cvar") and hasattr(faiss.cvar, "hnsw_stats"):
            ndis = int(faiss.cvar.hnsw_stats.ndis)
        searched = int(query_embeddings.shape[0])
        new_rows = _recall_rows_from_labels(
            labels_matrix,
            target_indices,
            top_k_values=top_k_values,
            base_row={"backend": "faiss", "ef_search": int(ef_search)},
            comparison_proxy="faiss_hnsw_stats.ndis" if ndis is not None else "ef_search_proxy",
            comparison_proxy_value_per_query=(
                (ndis / searched) if ndis is not None and searched else int(ef_search)
            ),
            distance_computations_total=ndis if ndis is not None else "",
            avg_distance_computations_per_query=(
                (ndis / searched) if ndis is not None and searched else ""
            ),
        )
        rows.extend(new_rows)
        for row in new_rows:
            print(
                f"[hnsw:faiss] ef_search={ef_search} top_k={row['top_k']} "
                f"recall={row['recall']:.6f} proxy={row['comparison_proxy_value_per_query']}",
                flush=True,
            )
    return rows


def _search_hnswlib(
    hnswlib: Any,
    doc_embeddings: np.ndarray,
    query_embeddings: np.ndarray,
    target_indices: np.ndarray,
    *,
    ef_search_values: Iterable[int],
    hnsw_m: int,
    top_k_values: tuple[int, ...],
    batch_size: int,
    construction_threads: int,
    search_threads: int,
) -> list[dict[str, Any]]:
    dim = int(doc_embeddings.shape[1])
    max_top_k = max(top_k_values)
    index = hnswlib.Index(space="ip", dim=dim)
    max_ef = max(int(v) for v in ef_search_values)
    index.init_index(max_elements=doc_embeddings.shape[0], ef_construction=max(200, max_ef), M=hnsw_m)
    index.add_items(
        doc_embeddings,
        np.arange(doc_embeddings.shape[0], dtype=np.int64),
        num_threads=construction_threads,
    )
    rows: list[dict[str, Any]] = []
    for ef_search in ef_search_values:
        index.set_ef(int(ef_search))
        all_labels: list[np.ndarray] = []
        for start in range(0, query_embeddings.shape[0], batch_size):
            end = min(start + batch_size, query_embeddings.shape[0])
            labels, _distances = index.knn_query(
                query_embeddings[start:end],
                k=max_top_k,
                num_threads=search_threads,
            )
            all_labels.append(labels)
        labels_matrix = np.vstack(all_labels)
        new_rows = _recall_rows_from_labels(
            labels_matrix,
            target_indices,
            top_k_values=top_k_values,
            base_row={"backend": "hnswlib", "ef_search": int(ef_search)},
            comparison_proxy="ef_search_proxy",
            comparison_proxy_value_per_query=int(ef_search),
        )
        rows.extend(new_rows)
        for row in new_rows:
            print(
                f"[hnsw:hnswlib] ef_search={ef_search} top_k={row['top_k']} "
                f"recall={row['recall']:.6f} proxy={ef_search}",
                flush=True,
            )
    return rows


def _search_hnswlib_trace(
    hnswlib_trace: Any,
    doc_embeddings: np.ndarray,
    query_embeddings: np.ndarray,
    target_indices: np.ndarray,
    *,
    ef_search_values: Iterable[int],
    hnsw_m: int,
    return_k: int,
    batch_size: int,
    construction_threads: int,
    search_threads: int,
) -> list[dict[str, Any]]:
    dim = int(doc_embeddings.shape[1])
    index = hnswlib_trace.Index(space="ip", dim=dim)
    max_ef = max(int(v) for v in ef_search_values)
    index.init_index(
        max_elements=doc_embeddings.shape[0],
        ef_construction=max(200, max_ef),
        M=hnsw_m,
    )
    index.add_items(
        doc_embeddings,
        np.arange(doc_embeddings.shape[0], dtype=np.uint64),
        num_threads=construction_threads,
    )
    rows: list[dict[str, Any]] = []
    target_labels = target_indices.astype(np.uint64, copy=False)
    for ef_search in ef_search_values:
        index.set_ef(int(ef_search))
        all_labels: list[np.ndarray] = []
        successes = 0
        visited_total = 0
        target_visit_rank_total = 0
        query_count = 0
        for start in range(0, query_embeddings.shape[0], batch_size):
            end = min(start + batch_size, query_embeddings.shape[0])
            trace_result = index.knn_query_with_trace(
                query_embeddings[start:end],
                target_labels[start:end],
                k=return_k,
                num_threads=search_threads,
            )
            if len(trace_result) == 5:
                labels, _distances, visited_counts, target_visit_ranks, target_visited = trace_result
            else:
                labels, _distances, visited_counts, target_visited = trace_result
                target_visit_ranks = np.zeros_like(visited_counts, dtype=np.uint64)
            all_labels.append(labels)
            target_visited_array = np.asarray(target_visited, dtype=bool)
            successes += int(target_visited_array.sum())
            visited_total += int(np.asarray(visited_counts, dtype=np.uint64).sum())
            target_visit_rank_total += int(
                np.asarray(target_visit_ranks, dtype=np.uint64)[target_visited_array].sum()
            )
            query_count += end - start
        labels_matrix = np.vstack(all_labels)
        avg_visited = visited_total / query_count if query_count else 0.0
        avg_visits_until_target = target_visit_rank_total / successes if successes else 0.0
        target_visited_recall = successes / query_count if query_count else 0.0
        new_rows = _recall_rows_from_labels(
            labels_matrix,
            target_indices,
            top_k_values=tuple(value for value in (1, 10) if value <= return_k),
            base_row={
                "backend": "hnswlib_trace",
                "ef_search": int(ef_search),
                "return_k": return_k,
                "hnsw_success_criterion": "target_visited",
                "target_visited_recall": target_visited_recall,
                "visited_vectors_total": visited_total,
                "avg_visited_vectors_per_query": avg_visited,
                "target_visit_rank_total": target_visit_rank_total,
                "avg_visits_until_target": avg_visits_until_target,
            },
            comparison_proxy="visited_count",
            comparison_proxy_value_per_query=avg_visited,
        )
        rows.extend(new_rows)
        print(
            f"[hnsw:hnswlib_trace] ef_search={ef_search} "
            f"target_visited_recall={target_visited_recall:.6f} "
            f"avg_visited={avg_visited:.2f} "
            f"avg_until_target={avg_visits_until_target:.2f}",
            flush=True,
        )
    return rows


def _pick_hnsw_row(rows: list[dict[str, Any]], target_recall: float) -> dict[str, Any]:
    if not rows:
        raise ValueError("no HNSW sweep rows")
    traced_rows = [row for row in rows if row.get("backend") == "hnswlib_trace"]
    if traced_rows:
        rows = traced_rows
    hnsw_rows = [row for row in rows if row.get("backend") != "faiss_flat"]
    if hnsw_rows:
        rows = hnsw_rows
    return min(
        rows,
        key=lambda row: (
            abs(float(row.get("target_visited_recall", row["recall"])) - target_recall),
            float(row.get("comparison_proxy_value_per_query") or math.inf),
            int(row["ef_search"]),
        ),
    )


def _write_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def run_hnsw_for_target(
    target: HnswDatasetTarget,
    *,
    backend_name: str,
    backend: Any,
    ef_search_values: tuple[int, ...],
    hnsw_m: int,
    top_k_values: tuple[int, ...],
    return_k: int,
    batch_size: int,
    construction_threads: int,
    search_threads: int,
    max_queries: int,
    exact_flat: bool,
    output_dir: Path,
) -> dict[str, Any]:
    doc_embeddings, doc_ids, query_embeddings, target_indices, _target_doc_ids = (
        _load_hnsw_query_and_doc_subset(target, max_queries=max_queries)
    )
    rows: list[dict[str, Any]] = []
    if exact_flat:
        if backend_name not in {"faiss", "hnswlib_trace"}:
            raise RuntimeError("--exact-flat requires the FAISS or hnswlib_trace backend")
        if backend_name == "hnswlib_trace":
            _, faiss_backend = _import_hnsw_backend("faiss")
        else:
            faiss_backend = backend
        rows.extend(
            _search_faiss_flat(
                faiss_backend,
                doc_embeddings,
                query_embeddings,
                target_indices,
                top_k_values=top_k_values,
                batch_size=batch_size,
            )
        )
    if backend_name == "hnswlib_trace":
        rows.extend(
            _search_hnswlib_trace(
                backend,
                doc_embeddings,
                query_embeddings,
                target_indices,
                ef_search_values=ef_search_values,
                hnsw_m=hnsw_m,
                return_k=return_k,
                batch_size=batch_size,
                construction_threads=construction_threads,
                search_threads=search_threads,
            )
        )
    elif backend_name == "faiss":
        rows.extend(
            _search_faiss_hnsw(
                backend,
                doc_embeddings,
                query_embeddings,
                target_indices,
                ef_search_values=ef_search_values,
                hnsw_m=hnsw_m,
                top_k_values=top_k_values,
                batch_size=batch_size,
            )
        )
    else:
        rows.extend(
            _search_hnswlib(
                backend,
                doc_embeddings,
                query_embeddings,
                target_indices,
                ef_search_values=ef_search_values,
                hnsw_m=hnsw_m,
                top_k_values=top_k_values,
                batch_size=batch_size,
                construction_threads=construction_threads,
                search_threads=search_threads,
            )
        )
    sweep_path = output_dir / f"hnsw_sweep_{target.dataset_key}.csv"
    _write_csv(sweep_path, rows)
    selected = _pick_hnsw_row(rows, target.target_recall)
    return {
        "dataset_key": target.dataset_key,
        "dataset": target.dataset,
        "amazon_category": target.amazon_category,
        "target_lowcoord_run_dir": str(target.target_run_dir),
        "target_lowcoord_recall": target.target_recall,
        "metadata_path": str(target.metadata_path),
        "searchable_queries_path": str(target.searchable_queries_path),
        "embedding_manifest_path": str(target.embedding_manifest_path),
        "indexed_doc_count": len(doc_ids),
        "query_count": int(query_embeddings.shape[0]),
        "selected": selected,
        "sweep_path": str(sweep_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate Low-Coord document-comparison work and run one dataset-level "
            "HNSW comparison sweep per dataset."
        )
    )
    parser.add_argument("--results-root", default=str(DEFAULT_RESULTS_ROOT))
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--include-smoke",
        action="store_true",
        help="Include result dirs whose name contains 'smoke'. Default excludes them.",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help="Optional dataset keys to include, e.g. aol4foltr amazon_electronics mind.",
    )
    parser.add_argument(
        "--skip-hnsw",
        action="store_true",
        help="Only write Low-Coord comparison estimates.",
    )
    parser.add_argument(
        "--backend",
        choices=("auto", "hnswlib_trace", "faiss", "hnswlib"),
        default="auto",
        help="HNSW backend. auto prefers the vendored traced hnswlib extension.",
    )
    parser.add_argument(
        "--ef-search-values",
        default=",".join(str(v) for v in DEFAULT_EF_SEARCH_VALUES),
        help="Comma-separated HNSW efSearch sweep values.",
    )
    parser.add_argument("--hnsw-m", type=int, default=32)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument(
        "--return-k",
        type=int,
        default=10,
        help="Number of nearest labels to return from traced HNSW; success is still target_visited.",
    )
    parser.add_argument(
        "--top-k-values",
        default=None,
        help="Comma-separated recall@k values. Defaults to --top-k.",
    )
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument(
        "--search-threads",
        type=int,
        default=1,
        help="Threads used for HNSW query batches.",
    )
    parser.add_argument(
        "--construction-threads",
        type=int,
        default=1,
        help=(
            "Threads used for HNSW add_items/index construction. Multi-threaded "
            "construction can slightly change graph tie-breaking, so stock/patched "
            "checks should compare metric similarity rather than exact label parity."
        ),
    )
    parser.add_argument(
        "--exact-flat",
        action="store_true",
        help="Also run exact FAISS IndexFlatIP to measure the recall ceiling on the sampled queries.",
    )
    parser.add_argument(
        "--max-hnsw-queries",
        type=int,
        default=0,
        help="Limit retained HNSW queries for validation. 0 means all.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_root = Path(args.results_root)
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else results_root / DEFAULT_OUTPUT_DIRNAME
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_datasets = set(args.datasets or [])
    runs = _load_lowcoord_runs(results_root, include_smoke=args.include_smoke)
    if selected_datasets:
        runs = [run for run in runs if run.dataset_key in selected_datasets]
    if not runs:
        raise SystemExit(f"No finished Low-Coord runs found under {results_root}")
    print(f"[lowcoord] loaded {len(runs):,} finished Low-Coord runs", flush=True)

    estimates = [estimate_lowcoord_run(run) for run in runs]
    lowcoord_path = output_dir / "lowcoord_comparison_estimates.csv"
    _write_lowcoord_estimates(lowcoord_path, estimates)
    print(f"[lowcoord] wrote {lowcoord_path}", flush=True)

    if args.skip_hnsw:
        return

    backend_name, backend = _import_hnsw_backend(args.backend)
    ef_search_values = tuple(
        int(value.strip()) for value in args.ef_search_values.split(",") if value.strip()
    )
    if not ef_search_values:
        raise SystemExit("--ef-search-values must contain at least one integer")
    top_k_values = tuple(
        int(value.strip())
        for value in (args.top_k_values or str(args.top_k)).split(",")
        if value.strip()
    )
    if not top_k_values or min(top_k_values) < 1:
        raise SystemExit("--top-k-values/--top-k must contain positive integers")
    top_k_values = tuple(sorted(set(top_k_values)))
    if args.return_k < 1:
        raise SystemExit("--return-k must be positive")
    if args.search_threads < 1:
        raise SystemExit("--search-threads must be positive")
    if args.construction_threads < 1:
        raise SystemExit("--construction-threads must be positive")

    best_runs = _best_runs_by_dataset(estimates)
    summaries: list[dict[str, Any]] = []
    for dataset_key, best_run in sorted(best_runs.items()):
        print(
            f"[hnsw:{dataset_key}] target_recall={best_run.overall_recall:.6f} "
            f"from {best_run.run_dir}",
            flush=True,
        )
        target = _dataset_hnsw_target(best_run)
        summaries.append(
            run_hnsw_for_target(
                target,
                backend_name=backend_name,
                backend=backend,
                ef_search_values=ef_search_values,
                hnsw_m=args.hnsw_m,
                top_k_values=top_k_values,
                return_k=args.return_k,
                batch_size=args.batch_size,
                construction_threads=args.construction_threads,
                search_threads=args.search_threads,
                max_queries=args.max_hnsw_queries,
                exact_flat=args.exact_flat,
                output_dir=output_dir,
            )
        )

    summary_path = output_dir / "hnsw_best_dataset_comparison.json"
    summary_path.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    _write_csv(output_dir / "hnsw_best_dataset_comparison.csv", [s["selected"] | {
        "dataset_key": s["dataset_key"],
        "target_lowcoord_recall": s["target_lowcoord_recall"],
        "target_lowcoord_run_dir": s["target_lowcoord_run_dir"],
        "indexed_doc_count": s["indexed_doc_count"],
    } for s in summaries])
    print(f"[hnsw] wrote {summary_path}", flush=True)


if __name__ == "__main__":
    main()

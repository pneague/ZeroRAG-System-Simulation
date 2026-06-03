#!/usr/bin/env python3
"""Run HNSW top-k retrieval over the full PubMedQA original-question corpus."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from pubmedqa_embed_full_retrieval import (  # noqa: E402
    DEFAULT_DATA_ROOT,
    DEFAULT_OUTPUT_DIR,
    _checkpoint_paths,
    _load_manifest_rows,
    _read_json,
    build_manifest,
    run_embedding,
)
from src.embedder import DEFAULT_EMBEDDING_MODEL  # noqa: E402


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")


def _percentile(sorted_values: list[float], percentile: float) -> float | None:
    if not sorted_values:
        return None
    index = round((percentile / 100.0) * (len(sorted_values) - 1))
    return sorted_values[min(len(sorted_values) - 1, max(0, int(index)))]


def _latency_summary(seconds: list[float]) -> dict[str, float | None]:
    sorted_seconds = sorted(seconds)
    return {
        "mean_seconds": statistics.fmean(seconds) if seconds else None,
        "median_seconds": statistics.median(seconds) if seconds else None,
        "p95_seconds": _percentile(sorted_seconds, 95.0),
        "p99_seconds": _percentile(sorted_seconds, 99.0),
        "min_seconds": min(seconds) if seconds else None,
        "max_seconds": max(seconds) if seconds else None,
        "mean_ms": statistics.fmean(seconds) * 1000.0 if seconds else None,
        "median_ms": statistics.median(seconds) * 1000.0 if seconds else None,
        "p95_ms": _percentile(sorted_seconds, 95.0) * 1000.0 if seconds else None,
        "p99_ms": _percentile(sorted_seconds, 99.0) * 1000.0 if seconds else None,
        "min_ms": min(seconds) * 1000.0 if seconds else None,
        "max_ms": max(seconds) * 1000.0 if seconds else None,
    }


def run_hnsw(args: argparse.Namespace) -> dict[str, Any]:
    try:
        import hnswlib_trace as hnsw_backend  # type: ignore

        backend_name = "hnswlib_trace"
    except Exception:
        import hnswlib as hnsw_backend  # type: ignore

        backend_name = "hnswlib"

    paths = _checkpoint_paths(args.output_dir)
    state = _read_json(paths["state"]) if paths["state"].exists() else {}
    row_count = int(state.get("doc_rows_embedded", 0))
    query_count = int(state.get("query_rows_embedded", 0))
    if row_count == 0 or query_count != row_count:
        raise RuntimeError("Embeddings are incomplete. Run --stage embed or --stage all first.")

    run_manifest = _read_json(paths["manifest"])
    dim = int(run_manifest["embedding_dim"])
    rows = _load_manifest_rows(args.output_dir / "corpus_manifest.jsonl")
    if len(rows) != row_count:
        raise RuntimeError(f"Manifest row count {len(rows)} does not match embeddings {row_count}")

    doc_pubids = np.load(paths["doc_pubids"], allow_pickle=True).astype(str)
    query_pubids = np.load(paths["query_pubids"], allow_pickle=True).astype(str)
    doc_embeddings = np.memmap(paths["doc_embeddings"], mode="r", dtype=np.float32, shape=(row_count, dim))
    query_embeddings = np.memmap(paths["query_embeddings"], mode="r", dtype=np.float32, shape=(query_count, dim))

    started_build = time.perf_counter()
    index = hnsw_backend.Index(space="ip", dim=dim)
    index.init_index(
        max_elements=row_count,
        ef_construction=max(args.ef_construction, args.ef_search),
        M=args.hnsw_m,
    )
    index.add_items(
        doc_embeddings,
        np.arange(row_count, dtype=np.uint64),
        num_threads=args.construction_threads,
    )
    index.set_ef(args.ef_search)
    build_seconds = time.perf_counter() - started_build

    results_path = args.output_dir / f"hnsw_top{args.top_k}_original_questions.jsonl"
    success_count = 0
    latencies: list[float] = []
    search_started = time.perf_counter()
    with results_path.open("w", encoding="utf-8") as handle:
        for row_index in range(query_count):
            query = np.asarray(query_embeddings[row_index], dtype=np.float32)
            started = time.perf_counter()
            labels, distances = index.knn_query(
                query.reshape(1, -1),
                k=args.top_k,
                num_threads=args.search_threads,
            )
            elapsed = time.perf_counter() - started
            latencies.append(elapsed)

            top_indices = [int(value) for value in labels[0].tolist()]
            top_pubids = [str(doc_pubids[index]) for index in top_indices]
            target_pubid = str(query_pubids[row_index])
            target_in_top_k = target_pubid in top_pubids
            success_count += int(target_in_top_k)
            handle.write(
                json.dumps(
                    {
                        "row_index": row_index,
                        "config": rows[row_index].get("config"),
                        "source_pubid": target_pubid,
                        "top_doc_indices": top_indices,
                        "top_doc_pubids": top_pubids,
                        "top_scores": [float(1.0 - value) for value in distances[0].tolist()],
                        "target_in_top_k": target_in_top_k,
                        "search_seconds": elapsed,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )
    search_seconds = time.perf_counter() - search_started

    summary = {
        "dataset": "pubmedqa",
        "question_set": "original_questions_only",
        "document_set": "all_pubmedqa_rows",
        "query_count": query_count,
        "doc_count": row_count,
        "top_k": args.top_k,
        "success_count": success_count,
        "recall_at_k": success_count / query_count if query_count else 0.0,
        "hnsw_backend": backend_name,
        "hnsw_m": args.hnsw_m,
        "ef_search": args.ef_search,
        "ef_construction": max(args.ef_construction, args.ef_search),
        "construction_threads": args.construction_threads,
        "search_threads": args.search_threads,
        "build_seconds": build_seconds,
        "search_wall_seconds": search_seconds,
        "latency": _latency_summary(latencies),
        "results_path": str(results_path),
        "embedding_manifest_path": str(paths["manifest"]),
    }
    _write_json(args.output_dir / "hnsw_summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("manifest", "embed", "hnsw", "all"), default="all")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--model-batch-size", type=int, default=256)
    parser.add_argument("--embed-batch-size", type=int, default=256)
    parser.add_argument("--checkpoint-rows", type=int, default=512)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--hnsw-m", type=int, default=32)
    parser.add_argument("--ef-search", type=int, default=512)
    parser.add_argument("--ef-construction", type=int, default=512)
    parser.add_argument("--construction-threads", type=int, default=16)
    parser.add_argument("--search-threads", type=int, default=16)
    parser.add_argument("--force-manifest", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries: dict[str, Any] = {}
    if args.stage in {"manifest", "all"}:
        summaries["manifest"] = build_manifest(args)
    if args.stage in {"embed", "all"}:
        summaries["embedding"] = run_embedding(args)
    if args.stage in {"hnsw", "all"}:
        summaries["hnsw"] = run_hnsw(args)
    if summaries:
        _write_json(args.output_dir / "full_hnsw_pipeline_summary.json", summaries)
        print(json.dumps(summaries, indent=2, ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

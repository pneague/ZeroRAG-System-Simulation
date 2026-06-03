#!/usr/bin/env python3
"""Evaluate nano rephrased PubMedQA questions against full doc embeddings."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.embedder import DEFAULT_EMBEDDING_MODEL, SentenceTransformerEmbedder


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def _build_query_items(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for row in rows:
        for rephrase_index, text in enumerate(row["rephrasings"]):
            items.append(
                {
                    "source_pubid": str(row["pubid"]),
                    "source_question": row["question"],
                    "rephrase_index": rephrase_index,
                    "rephrased_question": str(text),
                }
            )
    return items


def embed_rephrases(args: argparse.Namespace, items: list[dict[str, Any]]) -> tuple[Path, Path]:
    embeddings_path = args.output_dir / "nano_rephrased_query_embeddings.npy"
    meta_path = args.output_dir / "nano_rephrased_query_meta.jsonl"
    _write_jsonl(meta_path, items)
    if embeddings_path.exists() and not args.force_embed:
        return embeddings_path, meta_path

    device = _resolve_device(args.device)
    embedder = SentenceTransformerEmbedder(
        model_name=args.embedding_model,
        batch_size=args.batch_size,
        device=device,
    )
    vectors = np.memmap(
        embeddings_path,
        mode="w+",
        dtype=np.float32,
        shape=(len(items), embedder.embedding_dim),
    )
    texts = [item["rephrased_question"] for item in items]
    for start in tqdm(range(0, len(texts), args.batch_size), desc="Embedding nano rephrases"):
        end = min(len(texts), start + args.batch_size)
        vectors[start:end] = embedder.encode(texts[start:end], show_progress_bar=False)
        vectors.flush()
    del vectors
    _write_json(
        args.output_dir / "nano_rephrased_embedding_manifest.json",
        {
            "embedding_model": args.embedding_model,
            "device": device,
            "embedding_dim": embedder.embedding_dim,
            "query_count": len(items),
            "embeddings_path": str(embeddings_path),
            "meta_path": str(meta_path),
        },
    )
    return embeddings_path, meta_path


def evaluate(args: argparse.Namespace, embeddings_path: Path, items: list[dict[str, Any]]) -> dict[str, Any]:
    import torch

    doc_pubids = np.load(args.doc_pubids, allow_pickle=True)
    doc_count = len(doc_pubids)
    query_count = len(items)
    query_embeddings = np.memmap(
        embeddings_path,
        mode="r",
        dtype=np.float32,
        shape=(query_count, args.embedding_dim),
    )
    doc_embeddings = np.memmap(
        args.doc_embeddings,
        mode="r",
        dtype=np.float32,
        shape=(doc_count, args.embedding_dim),
    )
    device = torch.device(_resolve_device(args.retrieval_device))
    pubid_to_index = {str(pubid): index for index, pubid in enumerate(doc_pubids)}
    missing_source = sorted({item["source_pubid"] for item in items if item["source_pubid"] not in pubid_to_index})
    if missing_source:
        raise RuntimeError(f"{len(missing_source)} source pubids missing from doc corpus; examples={missing_source[:10]}")

    results_path = args.output_dir / "nano_rephrased_full_doc_results.jsonl"
    mismatch_count = 0
    source_in_top10_count = 0
    top_examples: list[dict[str, Any]] = []

    with results_path.open("w", encoding="utf-8") as handle:
        for q_start in tqdm(range(0, query_count, args.query_batch_size), desc="Full-doc NN scan"):
            q_end = min(query_count, q_start + args.query_batch_size)
            q_np = np.asarray(query_embeddings[q_start:q_end], dtype=np.float32)
            q = torch.from_numpy(q_np).to(device)
            best_scores = torch.full((q_end - q_start,), -float("inf"), device=device)
            best_indices = torch.zeros((q_end - q_start,), dtype=torch.long, device=device)
            top10_scores = torch.full((q_end - q_start, 10), -float("inf"), device=device)
            top10_indices = torch.zeros((q_end - q_start, 10), dtype=torch.long, device=device)

            for d_start in range(0, doc_count, args.doc_chunk_size):
                d_end = min(doc_count, d_start + args.doc_chunk_size)
                d_np = np.asarray(doc_embeddings[d_start:d_end], dtype=np.float32)
                d = torch.from_numpy(d_np).to(device)
                with torch.no_grad():
                    scores = q @ d.T
                    local_best_scores, local_best_indices = torch.max(scores, dim=1)
                    improve = local_best_scores > best_scores
                    best_scores[improve] = local_best_scores[improve]
                    best_indices[improve] = d_start + local_best_indices[improve]

                    candidate_indices = torch.arange(d_start, d_end, device=device).unsqueeze(0).expand(scores.shape[0], -1)
                    merged_scores = torch.cat([top10_scores, scores], dim=1)
                    merged_indices = torch.cat([top10_indices, candidate_indices], dim=1)
                    top10_scores, top10_pos = torch.topk(merged_scores, k=10, dim=1)
                    top10_indices = torch.gather(merged_indices, 1, top10_pos)
                del d

            best_indices_np = best_indices.cpu().numpy()
            best_scores_np = best_scores.cpu().numpy()
            top10_indices_np = top10_indices.cpu().numpy()
            for local_index, item_index in enumerate(range(q_start, q_end)):
                item = items[item_index]
                source_pubid = item["source_pubid"]
                source_index = pubid_to_index[source_pubid]
                nearest_index = int(best_indices_np[local_index])
                nearest_pubid = str(doc_pubids[nearest_index])
                source_similarity = float(np.dot(q_np[local_index], np.asarray(doc_embeddings[source_index], dtype=np.float32)))
                nearest_similarity = float(best_scores_np[local_index])
                is_mismatch = nearest_pubid != source_pubid
                source_in_top10 = source_index in top10_indices_np[local_index]
                mismatch_count += int(is_mismatch)
                source_in_top10_count += int(source_in_top10)
                result = {
                    **item,
                    "nearest_pubid": nearest_pubid,
                    "nearest_similarity": nearest_similarity,
                    "source_similarity": source_similarity,
                    "margin": nearest_similarity - source_similarity,
                    "nearest_is_different_doc": is_mismatch,
                    "source_in_top10": source_in_top10,
                }
                handle.write(json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n")
                if is_mismatch:
                    top_examples.append(result)
            del q

    top_examples = sorted(top_examples, key=lambda row: row["margin"], reverse=True)[: args.example_count]
    report = {
        "query_count": query_count,
        "doc_count": doc_count,
        "queries_closest_to_different_doc": mismatch_count,
        "queries_closest_to_different_doc_fraction": mismatch_count / query_count,
        "queries_source_doc_in_top10": source_in_top10_count,
        "queries_source_doc_in_top10_fraction": source_in_top10_count / query_count,
        "results_path": str(results_path),
        "doc_embeddings": str(args.doc_embeddings),
        "doc_pubids": str(args.doc_pubids),
        "top_mismatch_examples": top_examples,
    }
    _write_json(args.output_dir / "nano_rephrased_full_doc_report.json", report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rephrased-queries", type=Path, required=True)
    parser.add_argument("--doc-embeddings", type=Path, required=True)
    parser.add_argument("--doc-pubids", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--embedding-dim", type=int, default=768)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--retrieval-device", default="auto")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--query-batch-size", type=int, default=256)
    parser.add_argument("--doc-chunk-size", type=int, default=16384)
    parser.add_argument("--example-count", type=int, default=20)
    parser.add_argument("--force-embed", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = _read_jsonl(args.rephrased_queries)
    items = _build_query_items(rows)
    embeddings_path, _ = embed_rephrases(args, items)
    report = evaluate(args, embeddings_path, items)
    print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()

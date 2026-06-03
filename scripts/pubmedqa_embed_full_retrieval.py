#!/usr/bin/env python3
"""Embed all PubMedQA documents and questions with checkpointed saves.

Uses sentence-transformers (MPS on Apple Silicon when available). After
embedding, finds the nearest document for each query by cosine similarity
(normalized dot product) and reports how many queries match a different
pubid than their source row.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from datasets import load_from_disk
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.embedder import DEFAULT_EMBEDDING_MODEL, SentenceTransformerEmbedder

PUBMEDQA_CONFIGS = ("pqa_labeled", "pqa_unlabeled", "pqa_artificial")
DEFAULT_DATA_ROOT = REPO_ROOT / "data" / "pubmedqa_rag" / "hf"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "pubmedqa_rag" / "full_retrieval"


@dataclass(frozen=True)
class CorpusRow:
    config: str
    row_index: int
    pubid: str
    question: str
    doc_text: str


def _clean_str(value: object) -> str:
    return str(value or "").strip()


def _compose_doc_text(context: dict[str, Any]) -> str:
    contexts = context.get("contexts") or []
    labels = context.get("labels") or []
    sections: list[str] = []
    for index, text in enumerate(contexts):
        text_value = _clean_str(text)
        if not text_value:
            continue
        label = _clean_str(labels[index]) if index < len(labels) else ""
        sections.append(f"{label}: {text_value}" if label else text_value)
    return "\n\n".join(sections)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_device(requested: str | None) -> str:
    if requested and requested != "auto":
        return requested
    try:
        import torch

        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


def load_corpus_rows(data_root: Path) -> list[CorpusRow]:
    rows: list[CorpusRow] = []
    for config in PUBMEDQA_CONFIGS:
        dataset = load_from_disk(str(data_root / config))
        for row_index, row in enumerate(dataset):
            context = row.get("context") or {}
            rows.append(
                CorpusRow(
                    config=config,
                    row_index=row_index,
                    pubid=_clean_str(row.get("pubid")),
                    question=_clean_str(row.get("question")),
                    doc_text=_compose_doc_text(context),
                )
            )
    return rows


def build_manifest(args: argparse.Namespace) -> dict[str, Any]:
    manifest_path = args.output_dir / "corpus_manifest.jsonl"
    if manifest_path.exists() and not args.force_manifest:
        count = sum(1 for line in manifest_path.open("r", encoding="utf-8") if line.strip())
        return {
            "corpus_manifest_path": str(manifest_path),
            "row_count": count,
            "reused_existing_manifest": True,
        }

    rows = load_corpus_rows(args.data_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    {
                        "config": row.config,
                        "row_index": row.row_index,
                        "pubid": row.pubid,
                        "question": row.question,
                        "doc_text": row.doc_text,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )
    return {
        "corpus_manifest_path": str(manifest_path),
        "row_count": len(rows),
        "reused_existing_manifest": False,
    }


def _load_manifest_rows(manifest_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _checkpoint_paths(output_dir: Path) -> dict[str, Path]:
    emb = output_dir / "embeddings"
    return {
        "state": output_dir / "embedding_state.json",
        "doc_pubids": emb / "doc_pubids.npy",
        "doc_embeddings": emb / "doc_embeddings.npy",
        "query_pubids": emb / "query_pubids.npy",
        "query_embeddings": emb / "query_embeddings.npy",
        "manifest": output_dir / "run_manifest.json",
    }


def _init_memmap(path: Path, *, count: int, dim: int, dtype: np.dtype) -> np.memmap:
    path.parent.mkdir(parents=True, exist_ok=True)
    return np.memmap(path, mode="r+", dtype=dtype, shape=(count, dim))


def _create_memmap(path: Path, *, count: int, dim: int, dtype: np.dtype) -> np.memmap:
    path.parent.mkdir(parents=True, exist_ok=True)
    return np.memmap(path, mode="w+", dtype=dtype, shape=(count, dim))


def embed_checkpointed(
    *,
    texts: list[str],
    pubids: list[str],
    embeddings_path: Path,
    pubids_path: Path,
    state_path: Path,
    state_key: str,
    embedder: SentenceTransformerEmbedder,
    embed_batch_size: int,
    checkpoint_rows: int,
) -> dict[str, Any]:
    if len(texts) != len(pubids):
        raise ValueError("texts and pubids length mismatch")

    dim = embedder.embedding_dim
    total = len(texts)
    embeddings_path.parent.mkdir(parents=True, exist_ok=True)
    pubids_path.parent.mkdir(parents=True, exist_ok=True)

    if pubids_path.exists():
        saved_pubids = np.load(pubids_path, allow_pickle=True)
        if len(saved_pubids) != total:
            raise RuntimeError(
                f"{pubids_path} has {len(saved_pubids)} ids but manifest has {total}"
            )
    else:
        np.save(pubids_path, np.asarray(pubids, dtype=str))

    if embeddings_path.exists():
        embeddings = _init_memmap(embeddings_path, count=total, dim=dim, dtype=np.float32)
    else:
        embeddings = _create_memmap(embeddings_path, count=total, dim=dim, dtype=np.float32)

    state = _read_json(state_path) if state_path.exists() else {}
    start = int(state.get(state_key, 0))
    if start >= total:
        return {
            "embeddings_path": str(embeddings_path),
            "pubids_path": str(pubids_path),
            "embedded_rows": total,
            "resumed_from": start,
            "already_complete": True,
        }

    for batch_start in tqdm(
        range(start, total, embed_batch_size),
        desc=f"Embedding {state_key}",
        initial=start // embed_batch_size,
        total=(total + embed_batch_size - 1) // embed_batch_size,
    ):
        batch_end = min(total, batch_start + embed_batch_size)
        batch_vectors = embedder.encode(
            texts[batch_start:batch_end],
            show_progress_bar=False,
        )
        embeddings[batch_start:batch_end] = batch_vectors
        embeddings.flush()

        if (batch_end % checkpoint_rows == 0) or batch_end == total:
            state[state_key] = batch_end
            state[f"{state_key}_updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            _write_json(state_path, state)

    del embeddings
    return {
        "embeddings_path": str(embeddings_path),
        "pubids_path": str(pubids_path),
        "embedded_rows": total,
        "resumed_from": start,
        "already_complete": start >= total,
    }


def run_embedding(args: argparse.Namespace) -> dict[str, Any]:
    paths = _checkpoint_paths(args.output_dir)
    manifest_path = args.output_dir / "corpus_manifest.jsonl"
    if not manifest_path.exists():
        raise RuntimeError(f"Missing corpus manifest: {manifest_path}")

    rows = _load_manifest_rows(manifest_path)
    pubids = [row["pubid"] for row in rows]
    doc_texts = [row["doc_text"] for row in rows]
    questions = [row["question"] for row in rows]

    device = resolve_device(args.device)
    embedder = SentenceTransformerEmbedder(
        model_name=args.embedding_model,
        batch_size=args.model_batch_size,
        device=device,
    )

    run_manifest = {
        "embedding_model": args.embedding_model,
        "device": device,
        "embedding_dim": embedder.embedding_dim,
        "row_count": len(rows),
        "embed_batch_size": args.embed_batch_size,
        "checkpoint_rows": args.checkpoint_rows,
    }
    _write_json(paths["manifest"], run_manifest)

    doc_summary = embed_checkpointed(
        texts=doc_texts,
        pubids=pubids,
        embeddings_path=paths["doc_embeddings"],
        pubids_path=paths["doc_pubids"],
        state_path=paths["state"],
        state_key="doc_rows_embedded",
        embedder=embedder,
        embed_batch_size=args.embed_batch_size,
        checkpoint_rows=args.checkpoint_rows,
    )
    query_summary = embed_checkpointed(
        texts=questions,
        pubids=pubids,
        embeddings_path=paths["query_embeddings"],
        pubids_path=paths["query_pubids"],
        state_path=paths["state"],
        state_key="query_rows_embedded",
        embedder=embedder,
        embed_batch_size=args.embed_batch_size,
        checkpoint_rows=args.checkpoint_rows,
    )
    return {"device": device, "documents": doc_summary, "queries": query_summary}


def _nearest_doc_indices(
    query_embeddings: np.ndarray,
    doc_embeddings: np.memmap,
    *,
    doc_chunk_size: int,
    query_batch_size: int,
    device: str,
) -> np.ndarray:
    n_queries = query_embeddings.shape[0]
    n_docs = doc_embeddings.shape[0]
    best_indices = np.zeros(n_queries, dtype=np.int64)
    best_scores = np.full(n_queries, -np.inf, dtype=np.float32)

    use_torch = device != "cpu"
    if use_torch:
        import torch

        torch_device = torch.device(device)
    else:
        torch = None
        torch_device = None

    for query_start in tqdm(
        range(0, n_queries, query_batch_size),
        desc="Retrieval nearest-doc scan",
    ):
        query_end = min(n_queries, query_start + query_batch_size)
        query_batch = np.asarray(query_embeddings[query_start:query_end], dtype=np.float32)
        batch_best_scores = np.full(query_end - query_start, -np.inf, dtype=np.float32)
        batch_best_indices = np.zeros(query_end - query_start, dtype=np.int64)

        for doc_start in range(0, n_docs, doc_chunk_size):
            doc_end = min(n_docs, doc_start + doc_chunk_size)
            doc_batch = np.asarray(doc_embeddings[doc_start:doc_end], dtype=np.float32)
            if use_torch:
                with torch.no_grad():
                    query_tensor = torch.from_numpy(query_batch).to(torch_device)
                    doc_tensor = torch.from_numpy(doc_batch).to(torch_device)
                    scores_tensor = query_tensor @ doc_tensor.T
                    local_best_tensor, local_argmax_tensor = torch.max(scores_tensor, dim=1)
                    local_best = local_best_tensor.cpu().numpy()
                    local_argmax = local_argmax_tensor.cpu().numpy()
            else:
                scores = query_batch @ doc_batch.T
                local_argmax = scores.argmax(axis=1)
                local_best = scores[np.arange(scores.shape[0]), local_argmax]
            improve = local_best > batch_best_scores
            batch_best_scores[improve] = local_best[improve]
            batch_best_indices[improve] = doc_start + local_argmax[improve]

        best_scores[query_start:query_end] = batch_best_scores
        best_indices[query_start:query_end] = batch_best_indices

    return best_indices


def run_evaluation(args: argparse.Namespace) -> dict[str, Any]:
    paths = _checkpoint_paths(args.output_dir)
    state = _read_json(paths["state"]) if paths["state"].exists() else {}
    row_count = int(state.get("doc_rows_embedded", 0))
    if row_count == 0 or state.get("query_rows_embedded", 0) != row_count:
        raise RuntimeError(
            "Embeddings are incomplete. Run --stage embed (or all) before evaluate."
        )

    doc_pubids = np.load(paths["doc_pubids"], allow_pickle=True)
    query_pubids = np.load(paths["query_pubids"], allow_pickle=True)
    if len(doc_pubids) != len(query_pubids):
        raise RuntimeError("doc_pubids and query_pubids length mismatch")

    run_manifest = _read_json(paths["manifest"])
    dim = int(run_manifest["embedding_dim"])

    doc_embeddings = np.memmap(
        paths["doc_embeddings"],
        mode="r",
        dtype=np.float32,
        shape=(row_count, dim),
    )
    query_embeddings = np.memmap(
        paths["query_embeddings"],
        mode="r",
        dtype=np.float32,
        shape=(row_count, dim),
    )
    retrieval_device = resolve_device(args.retrieval_device)

    nearest_indices = _nearest_doc_indices(
        query_embeddings,
        doc_embeddings,
        doc_chunk_size=args.doc_chunk_size,
        query_batch_size=args.retrieval_query_batch_size,
        device=retrieval_device,
    )

    drift_count = 0
    results_path = args.output_dir / "query_nearest_doc_results.jsonl"
    with results_path.open("w", encoding="utf-8") as handle:
        for index, nearest_index in enumerate(nearest_indices):
            source_pubid = str(query_pubids[index])
            nearest_pubid = str(doc_pubids[int(nearest_index)])
            is_drift = nearest_pubid != source_pubid
            drift_count += int(is_drift)
            source_score = float(
                np.dot(
                    np.asarray(query_embeddings[index], dtype=np.float32),
                    np.asarray(doc_embeddings[index], dtype=np.float32),
                )
            )
            nearest_score = float(
                np.dot(
                    np.asarray(query_embeddings[index], dtype=np.float32),
                    np.asarray(doc_embeddings[int(nearest_index)], dtype=np.float32),
                )
            )
            handle.write(
                json.dumps(
                    {
                        "row_index": index,
                        "source_pubid": source_pubid,
                        "nearest_pubid": nearest_pubid,
                        "nearest_is_different_doc": is_drift,
                        "source_similarity": source_score,
                        "nearest_similarity": nearest_score,
                        "margin": nearest_score - source_score,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )

    report = {
        "queries_evaluated": row_count,
        "queries_nearest_different_doc": drift_count,
        "queries_nearest_different_doc_fraction": drift_count / row_count if row_count else 0.0,
        "queries_nearest_correct_doc": row_count - drift_count,
        "results_path": str(results_path),
        "doc_chunk_size": args.doc_chunk_size,
        "retrieval_query_batch_size": args.retrieval_query_batch_size,
        "retrieval_device": retrieval_device,
    }
    _write_json(args.output_dir / "retrieval_report.json", report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("manifest", "embed", "evaluate", "all"),
        default="all",
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device: auto (prefer mps), mps, cuda, or cpu",
    )
    parser.add_argument(
        "--model-batch-size",
        type=int,
        default=256,
        help="SentenceTransformer internal encode batch size",
    )
    parser.add_argument(
        "--embed-batch-size",
        type=int,
        default=256,
        help="Rows embedded before flushing memmap checkpoint",
    )
    parser.add_argument(
        "--checkpoint-rows",
        type=int,
        default=512,
        help="Flush embedding checkpoint every N rows",
    )
    parser.add_argument("--doc-chunk-size", type=int, default=4096)
    parser.add_argument("--retrieval-query-batch-size", type=int, default=256)
    parser.add_argument(
        "--retrieval-device",
        default="auto",
        help="Device for nearest-doc scan: auto (prefer cuda/mps), cuda, mps, or cpu",
    )
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
    if args.stage in {"evaluate", "all"}:
        summaries["retrieval"] = run_evaluation(args)

    if summaries:
        _write_json(args.output_dir / "pipeline_summary.json", summaries)
        print(json.dumps(summaries, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()

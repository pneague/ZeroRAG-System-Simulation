#!/usr/bin/env python3
"""Build full-corpus PubMedQA all-MeSH artifacts with original and rephrased queries."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent
SCRIPT_DIR = REPO_ROOT / "scripts"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from precompute_pubmedqa_original_all_mesh import (  # noqa: E402
    DEFAULT_HF_ROOT,
    DEFAULT_SOURCE_FULL_DIR,
    FullPubMedQaRow,
    _build_topic_pools,
    _copy_embeddings_to_npy,
    _embedding_paths,
    _load_embedding_array,
    _load_full_rows,
    _open_text_writer,
    _query_key,
    _sample_trace,
    _write_json,
    _write_peer_future_docs,
)
from pubmedqa_embed_full_retrieval import _clean_str  # noqa: E402
from src.data_root import resolve_dataset_dir  # noqa: E402
from src.dataset import AMAZON_EVENTS_FIELDNAMES, SEARCHABLE_QUERY_FIELDNAMES, synthetic_timestamp_for_index  # noqa: E402
from src.embedder import DEFAULT_EMBEDDING_MODEL, SentenceTransformerEmbedder  # noqa: E402
from src.pubmedqa_dataset import (  # noqa: E402
    DEFAULT_PUBMEDQA_RANDOM_SEED,
    PUBMEDQA_ANSWER_TARGETS_FILENAME,
    PUBMEDQA_DOC_TOPICS_FILENAME,
    PUBMEDQA_EVENTS_FILENAME,
    PUBMEDQA_PEER_FUTURE_DOCS_FILENAME,
    PUBMEDQA_PEER_FUTURE_DOCS_SUMMARY_FILENAME,
    PUBMEDQA_PRECOMPUTE_SUMMARY_FILENAME,
    PUBMEDQA_QUERY_TOPICS_FILENAME,
    PUBMEDQA_RANDOMIZED_EVENTS_FILENAME,
    PUBMEDQA_SEEDED_SEARCHABLE_FILENAME,
    PUBMEDQA_TOPIC_MAP_FILENAME,
)


DEFAULT_REPHRASED_QUERIES = REPO_ROOT / "data" / "pubmedqa_rag" / "all_mesh_rephrasing_nano" / "rephrased_queries.jsonl"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _resolve_device(requested: str | None) -> str:
    if requested and requested != "auto":
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


def _rephrase_embedding_paths(embedding_dir: Path) -> dict[str, Path]:
    return {
        "state": embedding_dir / "rephrase_embedding_state.json",
        "manifest": embedding_dir / "rephrase_embedding_manifest.json",
        "source_indices": embedding_dir / "rephrase_source_indices.npy",
        "pubids": embedding_dir / "rephrase_pubids.npy",
        "embeddings": embedding_dir / "rephrased_query_embeddings.npy",
    }


def _load_rephrased_rows(path: Path) -> dict[tuple[str, int], dict[str, Any]]:
    rows: dict[tuple[str, int], dict[str, Any]] = {}
    for row in _iter_jsonl(path):
        key = (str(row["config"]), int(row["row_index"]))
        rephrasings = [_clean_str(value) for value in row.get("rephrasings") or []]
        if len(rephrasings) != 5:
            raise ValueError(f"{path} row {key} does not contain exactly 5 rephrasings")
        rows[key] = {**row, "rephrasings": rephrasings}
    if not rows:
        raise ValueError(f"{path} is empty")
    return rows


def _prepare_rephrase_inputs(
    *,
    rows: list[FullPubMedQaRow],
    rephrased_by_key: dict[tuple[str, int], dict[str, Any]],
) -> tuple[list[str], list[str], np.ndarray, dict[int, int]]:
    texts: list[str] = []
    pubids: list[str] = []
    source_indices: list[int] = []
    source_to_offset: dict[int, int] = {}
    missing: list[tuple[str, int]] = []

    for row in rows:
        if not row.meshes:
            continue
        key = (row.config, int(row.row_index))
        rephrased = rephrased_by_key.get(key)
        if rephrased is None:
            missing.append(key)
            continue
        if str(rephrased["pubid"]) != row.pubid:
            raise ValueError(f"Rephrased pubid mismatch at {key}: {rephrased['pubid']!r} != {row.pubid!r}")
        source_to_offset[int(row.source_index)] = len(texts)
        for rephrase in rephrased["rephrasings"]:
            texts.append(rephrase)
            pubids.append(row.pubid)
            source_indices.append(int(row.source_index))

    if missing:
        raise ValueError(f"Missing rephrased rows for {len(missing)} mesh-bearing rows, first={missing[:5]}")
    return texts, pubids, np.asarray(source_indices, dtype=np.int32), source_to_offset


def embed_rephrased_queries(args: argparse.Namespace) -> dict[str, Any]:
    paths = _embedding_paths(args.source_full_dir)
    rows, mesh_summary = _load_full_rows(hf_root=args.hf_root, source_full_dir=args.source_full_dir)
    rephrased_by_key = _load_rephrased_rows(args.rephrased_queries)
    texts, pubids, source_indices, _source_to_offset = _prepare_rephrase_inputs(
        rows=rows,
        rephrased_by_key=rephrased_by_key,
    )
    emb_manifest = _read_json(paths["manifest"])
    expected_dim = int(emb_manifest["embedding_dim"])

    output_dir = args.rephrase_embedding_dir or (args.rephrased_queries.parent / "embeddings")
    output_dir.mkdir(parents=True, exist_ok=True)
    out_paths = _rephrase_embedding_paths(output_dir)
    np.save(out_paths["source_indices"], source_indices)
    np.save(out_paths["pubids"], np.asarray(pubids, dtype=str))

    device = _resolve_device(args.device)
    embedder = SentenceTransformerEmbedder(
        model_name=args.embedding_model,
        batch_size=args.model_batch_size,
        device=device,
    )
    if embedder.embedding_dim != expected_dim:
        raise ValueError(
            f"Rephrase embedding dim {embedder.embedding_dim} != source full embedding dim {expected_dim}"
        )

    if out_paths["embeddings"].exists():
        embeddings = np.load(out_paths["embeddings"], mmap_mode="r+")
        if embeddings.shape != (len(texts), expected_dim):
            raise ValueError(f"{out_paths['embeddings']} shape {embeddings.shape} != {(len(texts), expected_dim)}")
    else:
        embeddings = np.lib.format.open_memmap(
            out_paths["embeddings"],
            mode="w+",
            dtype=np.float32,
            shape=(len(texts), expected_dim),
        )

    state = _read_json(out_paths["state"]) if out_paths["state"].exists() else {}
    start = int(state.get("rephrase_rows_embedded", 0))
    if start < len(texts):
        for batch_start in range(start, len(texts), args.embed_batch_size):
            batch_end = min(batch_start + args.embed_batch_size, len(texts))
            vectors = embedder.encode(texts[batch_start:batch_end], show_progress_bar=False)
            embeddings[batch_start:batch_end] = vectors
            embeddings.flush()
            if batch_end % args.checkpoint_rows == 0 or batch_end == len(texts):
                state["rephrase_rows_embedded"] = batch_end
                state["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                _write_json(out_paths["state"], state)
                print(
                    f"[pubmedqa-expanded-all-mesh] embedded rephrase rows {batch_end}/{len(texts)}",
                    flush=True,
                )
    del embeddings

    manifest = {
        "dataset": "pubmedqa",
        "pubmedqa_variant": "expanded_all_mesh_rephrase_source",
        "embedding_model": args.embedding_model,
        "embedding_dim": expected_dim,
        "device": device,
        "rephrased_queries_path": str(args.rephrased_queries),
        "mesh_bearing_source_rows": len(_source_to_offset),
        "rephrased_query_count": len(texts),
        "rephrase_pubids_path": str(out_paths["pubids"]),
        "rephrase_source_indices_path": str(out_paths["source_indices"]),
        "rephrased_query_embeddings_path": str(out_paths["embeddings"]),
        "mesh_summary": mesh_summary,
    }
    _write_json(out_paths["manifest"], manifest)
    return manifest


def _load_rephrase_embedding_lookup(
    *,
    embedding_dir: Path,
    rows: list[FullPubMedQaRow],
    dim: int,
) -> tuple[np.ndarray, np.ndarray]:
    paths = _rephrase_embedding_paths(embedding_dir)
    source_indices = np.load(paths["source_indices"])
    embeddings = np.load(paths["embeddings"], mmap_mode="r")
    if embeddings.shape[0] != source_indices.shape[0] or embeddings.shape[1] != dim:
        raise ValueError(
            f"Rephrase embedding/source shape mismatch: embeddings={embeddings.shape}, "
            f"source_indices={source_indices.shape}, dim={dim}"
        )
    source_to_offset = np.full(len(rows), -1, dtype=np.int64)
    for offset in range(0, int(source_indices.shape[0]), 5):
        source_index = int(source_indices[offset])
        if source_index < 0 or source_index >= len(rows):
            raise ValueError(f"Invalid rephrase source_index={source_index}")
        if np.any(source_indices[offset : offset + 5] != source_index):
            raise ValueError(f"Rephrase source indices are not grouped by five at offset={offset}")
        source_to_offset[source_index] = offset
    return embeddings, source_to_offset


def precompute_expanded_all_mesh(args: argparse.Namespace) -> dict[str, Any]:
    data_dir = args.output_dir or resolve_dataset_dir("pubmedqa")
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    paths = _embedding_paths(args.source_full_dir)
    state = _read_json(paths["state"])
    emb_manifest = _read_json(paths["manifest"])
    row_count = int(emb_manifest["row_count"])
    dim = int(emb_manifest["embedding_dim"])
    if int(state.get("doc_rows_embedded", 0)) != row_count:
        raise RuntimeError("Source full PubMedQA doc embeddings are incomplete")
    if int(state.get("query_rows_embedded", 0)) != row_count:
        raise RuntimeError("Source full PubMedQA query embeddings are incomplete")

    rows, mesh_summary = _load_full_rows(hf_root=args.hf_root, source_full_dir=args.source_full_dir)
    if len(rows) != row_count:
        raise ValueError(f"Loaded {len(rows)} rows but source embeddings have {row_count}")

    source_doc_pubids = np.load(paths["doc_pubids"], allow_pickle=True).astype(str)
    source_query_pubids = np.load(paths["query_pubids"], allow_pickle=True).astype(str)
    row_pubids = np.asarray([row.pubid for row in rows], dtype=str)
    if not np.array_equal(source_doc_pubids, row_pubids):
        raise ValueError("Source doc_pubids.npy is not aligned with full PubMedQA rows")
    if not np.array_equal(source_query_pubids, row_pubids):
        raise ValueError("Source query_pubids.npy is not aligned with full PubMedQA rows")

    rephrased_by_key = _load_rephrased_rows(args.rephrased_queries)
    _texts, _pubids, _source_indices, source_to_offset_dict = _prepare_rephrase_inputs(
        rows=rows,
        rephrased_by_key=rephrased_by_key,
    )
    rephrase_embedding_dir = args.rephrase_embedding_dir or (args.rephrased_queries.parent / "embeddings")
    rephrase_embeddings, source_to_offset = _load_rephrase_embedding_lookup(
        embedding_dir=rephrase_embedding_dir,
        rows=rows,
        dim=dim,
    )

    topic_terms, topic_id_by_mesh, doc_indices_by_topic = _build_topic_pools(rows)
    user_ids, topic_ids, doc_indices, sample_summary = _sample_trace(
        peer_count=args.peers,
        queries_per_peer=args.queries_per_peer,
        topic_count=len(topic_terms),
        doc_indices_by_topic=doc_indices_by_topic,
        alpha=args.alpha,
        random_seed=args.random_seed,
    )
    query_count = int(user_ids.shape[0])
    rng = np.random.default_rng(args.random_seed + 6)
    query_variants = rng.integers(0, 6, size=query_count, dtype=np.int8)
    qid_width = max(6, len(str(max(0, query_count - 1))))
    query_keys = np.asarray([_query_key(qid, qid_width) for qid in range(query_count)], dtype=str)
    target_pubids = np.asarray([rows[int(doc_index)].pubid for doc_index in doc_indices], dtype=str)

    missing_offsets = np.unique(doc_indices[(query_variants > 0) & (source_to_offset[doc_indices] < 0)])
    if missing_offsets.size:
        raise ValueError(f"Missing rephrase embeddings for sampled source rows: {missing_offsets[:10].tolist()}")

    docs_by_peer: dict[int, set[str]] = defaultdict(set)
    owners_by_doc: dict[str, set[int]] = defaultdict(set)
    for user_id, doc_id in zip(user_ids.tolist(), target_pubids.tolist(), strict=True):
        docs_by_peer[int(user_id)].add(str(doc_id))
        owners_by_doc[str(doc_id)].add(int(user_id))

    embeddings_dir = data_dir / "embeddings"
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    doc_keys_path = embeddings_dir / "doc_keys.npy"
    query_keys_path = embeddings_dir / "query_keys.npy"
    doc_embeddings_path = embeddings_dir / "doc_embeddings.npy"
    query_embeddings_path = embeddings_dir / "query_embeddings.npy"
    np.save(doc_keys_path, row_pubids)
    np.save(query_keys_path, query_keys)

    dtype = np.dtype(args.embedding_dtype)
    source_doc_embeddings = _load_embedding_array(paths["doc_embeddings"], rows=row_count, dim=dim)
    source_query_embeddings = _load_embedding_array(paths["query_embeddings"], rows=row_count, dim=dim)
    _copy_embeddings_to_npy(
        source=source_doc_embeddings,
        destination=doc_embeddings_path,
        dtype=dtype,
        batch_size=args.embedding_batch_size,
    )

    query_embeddings = np.lib.format.open_memmap(
        query_embeddings_path,
        mode="w+",
        dtype=dtype,
        shape=(query_count, dim),
    )
    for start in range(0, query_count, args.embedding_batch_size):
        end = min(start + args.embedding_batch_size, query_count)
        chunk_doc_indices = doc_indices[start:end]
        chunk_variants = query_variants[start:end]
        out = np.empty((end - start, dim), dtype=np.float32)
        original_mask = chunk_variants == 0
        if np.any(original_mask):
            out[original_mask] = np.asarray(source_query_embeddings[chunk_doc_indices[original_mask]], dtype=np.float32)
        if np.any(~original_mask):
            rephrase_rows = source_to_offset[chunk_doc_indices[~original_mask]] + (chunk_variants[~original_mask] - 1)
            out[~original_mask] = np.asarray(rephrase_embeddings[rephrase_rows], dtype=np.float32)
        query_embeddings[start:end] = out.astype(dtype, copy=False)
    query_embeddings.flush()
    del query_embeddings

    chronological_path = data_dir / PUBMEDQA_EVENTS_FILENAME
    randomized_path = data_dir / PUBMEDQA_RANDOMIZED_EVENTS_FILENAME
    searchable_path = data_dir / PUBMEDQA_SEEDED_SEARCHABLE_FILENAME
    query_topics_path = data_dir / PUBMEDQA_QUERY_TOPICS_FILENAME
    answer_targets_path = data_dir / PUBMEDQA_ANSWER_TARGETS_FILENAME
    permutation = np.random.default_rng(args.random_seed + 4).permutation(query_count)

    def event_row_for_qid(qid: int, event_index: int) -> dict[str, str | int]:
        return {
            "timestamp": synthetic_timestamp_for_index(event_index),
            "qid": qid,
            "user_id": int(user_ids[qid]),
            "query": str(query_keys[qid]),
            "target_doc_id": str(target_pubids[qid]),
            "candidate_doc_ids": "[]",
        }

    def searchable_for_qid(qid: int) -> bool:
        doc_id = str(target_pubids[qid])
        return bool(owners_by_doc[doc_id] - {int(user_ids[qid])})

    with _open_text_writer(chronological_path) as handle:
        writer = csv.DictWriter(handle, fieldnames=AMAZON_EVENTS_FIELDNAMES)
        writer.writeheader()
        for qid in range(query_count):
            writer.writerow(event_row_for_qid(qid, qid))

    topic_fields = (
        "qid",
        "user_id",
        "query",
        "generated_topic",
        "mesh_term",
        "target_doc_id",
        "target_doc_topic",
        "query_variant",
        "source_selected_index",
        "searchable",
    )
    searchable_count = 0
    with (
        _open_text_writer(randomized_path) as randomized_handle,
        _open_text_writer(searchable_path) as searchable_handle,
        _open_text_writer(query_topics_path) as topics_handle,
        answer_targets_path.open("w", encoding="utf-8") as answers_handle,
    ):
        randomized_writer = csv.DictWriter(randomized_handle, fieldnames=AMAZON_EVENTS_FIELDNAMES)
        searchable_writer = csv.DictWriter(searchable_handle, fieldnames=SEARCHABLE_QUERY_FIELDNAMES)
        topics_writer = csv.DictWriter(topics_handle, fieldnames=topic_fields)
        randomized_writer.writeheader()
        searchable_writer.writeheader()
        topics_writer.writeheader()
        for event_index, raw_qid in enumerate(permutation.tolist()):
            qid = int(raw_qid)
            source_row = rows[int(doc_indices[qid])]
            variant = int(query_variants[qid])
            rephrased_row = rephrased_by_key[(source_row.config, int(source_row.row_index))]
            searchable = searchable_for_qid(qid)
            searchable_count += int(searchable)
            variant_label = "original" if variant == 0 else f"rephrased_{variant}"

            randomized_writer.writerow(event_row_for_qid(qid, event_index))
            searchable_writer.writerow({"query_id": qid, "searchable": "true" if searchable else "false"})
            topics_writer.writerow(
                {
                    "qid": qid,
                    "user_id": int(user_ids[qid]),
                    "query": str(query_keys[qid]),
                    "generated_topic": int(topic_ids[qid]),
                    "mesh_term": topic_terms[int(topic_ids[qid])],
                    "target_doc_id": str(target_pubids[qid]),
                    "target_doc_topic": int(topic_ids[qid]),
                    "query_variant": variant_label,
                    "source_selected_index": int(source_row.source_index),
                    "searchable": "true" if searchable else "false",
                }
            )
            question = source_row.question if variant == 0 else rephrased_row["rephrasings"][variant - 1]
            answer_payload = {
                "event_index": event_index,
                "qid": qid,
                "query_key": str(query_keys[qid]),
                "query_variant": "original" if variant == 0 else "rephrased",
                "rephrase_index": None if variant == 0 else variant - 1,
                "source_selected_index": int(source_row.source_index),
                "source_pubid": source_row.pubid,
                "target_doc_id": str(target_pubids[qid]),
                "config": source_row.config,
                "row_index": int(source_row.row_index),
                "question": question,
                "original_question": source_row.question,
                "final_decision": source_row.final_decision,
                "long_answer": source_row.long_answer,
                "has_answer": bool(source_row.final_decision or source_row.long_answer),
            }
            answers_handle.write(json.dumps(answer_payload, ensure_ascii=False, sort_keys=True) + "\n")

    doc_topics_path = data_dir / PUBMEDQA_DOC_TOPICS_FILENAME
    with _open_text_writer(doc_topics_path) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("doc_id", "selected_index", "topic_id", "mesh_term", "is_primary"),
        )
        writer.writeheader()
        for row in rows:
            for offset, mesh in enumerate(row.meshes):
                writer.writerow(
                    {
                        "doc_id": row.pubid,
                        "selected_index": int(row.source_index),
                        "topic_id": topic_id_by_mesh[mesh],
                        "mesh_term": mesh,
                        "is_primary": "true" if offset == 0 else "false",
                    }
                )

    topic_map = {
        "topics": [
            {"topic_id": topic_id, "mesh_term": mesh, "doc_count": len(doc_indices_by_topic[topic_id])}
            for topic_id, mesh in enumerate(topic_terms)
        ]
    }
    _write_json(data_dir / PUBMEDQA_TOPIC_MAP_FILENAME, topic_map)
    peer_summary = _write_peer_future_docs(data_dir, docs_by_peer)
    topic_counts = Counter(int(topic_id) for topic_id in topic_ids.tolist())
    variant_counts = Counter(int(variant) for variant in query_variants.tolist())
    variant_count_labels = {
        ("original" if key == 0 else f"rephrased_{key}"): int(value)
        for key, value in sorted(variant_counts.items())
    }

    manifest = {
        "dataset": "pubmedqa",
        "pubmedqa_variant": "expanded_all_mesh",
        "metadata_path": str(chronological_path),
        "searchable_queries_path": str(searchable_path),
        "output_dir": str(embeddings_dir),
        "model": emb_manifest.get("embedding_model", DEFAULT_EMBEDDING_MODEL),
        "embedding_dim": dim,
        "embedding_dtype": str(dtype),
        "query_keys_path": str(query_keys_path),
        "query_embeddings_path": str(query_embeddings_path),
        "query_count": query_count,
        "doc_keys_path": str(doc_keys_path),
        "doc_embeddings_path": str(doc_embeddings_path),
        "doc_count": row_count,
        "user_count": args.peers,
        "queries_per_user": args.queries_per_peer,
        "dirichlet_alpha": args.alpha,
        "random_seed": args.random_seed,
        "source_full_dir": str(args.source_full_dir),
        "source_query_variant": "original_plus_5_rephrases",
        "rephrased_queries_path": str(args.rephrased_queries),
        "rephrase_embedding_manifest_path": str(_rephrase_embedding_paths(rephrase_embedding_dir)["manifest"]),
    }
    manifest_path = embeddings_dir / "manifest.json"
    _write_json(manifest_path, manifest)

    top_sampled_topics = [
        {"topic_id": topic_id, "mesh_term": topic_terms[topic_id], "query_count": int(count)}
        for topic_id, count in topic_counts.most_common(20)
    ]
    summary = {
        "dataset": "pubmedqa",
        "pubmedqa_variant": "expanded_all_mesh",
        "data_dir": str(data_dir),
        "source_full_dir": str(args.source_full_dir),
        "full_document_count": row_count,
        "topic_count": len(topic_terms),
        "mesh_summary": mesh_summary,
        "user_count": args.peers,
        "queries_per_user": args.queries_per_peer,
        "query_count": query_count,
        "dirichlet_alpha": args.alpha,
        "random_seed": args.random_seed,
        "searchable_query_count": searchable_count,
        "query_variant_counts": variant_count_labels,
        "sample_summary": {**sample_summary, "top_sampled_topics": top_sampled_topics},
        "manifest_path": str(manifest_path),
        "events_path": str(chronological_path),
        "randomized_events_path": str(randomized_path),
        "searchable_queries_path": str(searchable_path),
        "peer_future_docs_summary": peer_summary,
        "query_topics_path": str(query_topics_path),
        "doc_topics_path": str(doc_topics_path),
        "answer_targets_path": str(answer_targets_path),
        "rephrase_embedding_source_rows": len(source_to_offset_dict),
    }
    _write_json(data_dir / PUBMEDQA_PRECOMPUTE_SUMMARY_FILENAME, summary)
    print(f"[pubmedqa-expanded-all-mesh] wrote manifest to {manifest_path}", flush=True)
    print(f"[pubmedqa-expanded-all-mesh] wrote summary to {data_dir / PUBMEDQA_PRECOMPUTE_SUMMARY_FILENAME}", flush=True)
    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("rephrase-embed", "dataset", "all"), default="all")
    parser.add_argument("--output-dir", type=Path, default=None, help="Dataset dir; defaults via data-root resolver")
    parser.add_argument("--hf-root", type=Path, default=DEFAULT_HF_ROOT)
    parser.add_argument("--source-full-dir", type=Path, default=DEFAULT_SOURCE_FULL_DIR)
    parser.add_argument("--rephrased-queries", type=Path, default=DEFAULT_REPHRASED_QUERIES)
    parser.add_argument("--rephrase-embedding-dir", type=Path, default=None)
    parser.add_argument("--peers", type=int, default=50_000)
    parser.add_argument("--queries-per-peer", type=int, default=30)
    parser.add_argument("--alpha", type=float, default=0.00007)
    parser.add_argument("--random-seed", type=int, default=DEFAULT_PUBMEDQA_RANDOM_SEED)
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--model-batch-size", type=int, default=256)
    parser.add_argument("--embed-batch-size", type=int, default=256)
    parser.add_argument("--checkpoint-rows", type=int, default=4096)
    parser.add_argument("--embedding-dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument("--embedding-batch-size", type=int, default=4096)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    summaries: dict[str, Any] = {}
    if args.stage in {"rephrase-embed", "all"}:
        summaries["rephrase_embedding"] = embed_rephrased_queries(args)
    if args.stage in {"dataset", "all"}:
        summaries["dataset"] = precompute_expanded_all_mesh(args)
    if summaries:
        data_dir = Path(args.output_dir) if args.output_dir else Path(resolve_dataset_dir("pubmedqa"))
        _write_json(data_dir / "expanded_all_mesh_pipeline_summary.json", summaries)
        print(json.dumps(summaries, indent=2, ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

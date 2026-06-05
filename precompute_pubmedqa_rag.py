#!/usr/bin/env python3
"""Build simulator-ready PubMedQA RAG artifacts from the p80 rephrasing run."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, TextIO

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data_root import resolve_dataset_dir
from src.dataset import (
    AMAZON_EVENTS_FIELDNAMES,
    SEARCHABLE_QUERY_FIELDNAMES,
    synthetic_timestamp_for_index,
)
from src.path_defaults import default_results_root
from src.pubmedqa_dataset import (
    DEFAULT_PUBMEDQA_DIRICHLET_ALPHA,
    DEFAULT_PUBMEDQA_QUERIES_PER_USER,
    DEFAULT_PUBMEDQA_RANDOM_SEED,
    DEFAULT_PUBMEDQA_USER_COUNT,
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

DEFAULT_SELECTION_DIR = REPO_ROOT / "data" / "pubmedqa_rag" / "p80_500_batch_retrieval"
DEFAULT_HF_ROOT = REPO_ROOT / "data" / "pubmedqa_rag" / "hf"
DEFAULT_SOURCE_EMBEDDINGS_DIR = (
    default_results_root() / "pubmedqa_p80_500_rephrasing_retrieval" / "embeddings"
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _open_text_writer(path: Path) -> TextIO:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".gz":
        return gzip.open(path, "wt", newline="", encoding="utf-8")
    return path.open("w", newline="", encoding="utf-8")


def _query_key(qid: int, width: int) -> str:
    return f"pmq_q_{qid:0{width}d}"


def _load_pubids(path: Path) -> list[str]:
    return [str(value) for value in np.load(path, allow_pickle=True).tolist()]


def _load_embedding_array(path: Path, *, rows: int, dim: int) -> np.ndarray:
    try:
        loaded = np.load(path, mmap_mode="r")
        if loaded.shape != (rows, dim):
            raise ValueError(f"{path} shape {loaded.shape} != {(rows, dim)}")
        return loaded
    except ValueError:
        return np.memmap(path, mode="r", dtype=np.float32, shape=(rows, dim))


def _copy_embeddings_to_npy(
    *,
    source: np.ndarray,
    destination: Path,
    dtype: np.dtype,
    batch_size: int,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    target = np.lib.format.open_memmap(
        destination,
        mode="w+",
        dtype=dtype,
        shape=source.shape,
    )
    for start in range(0, source.shape[0], batch_size):
        end = min(start + batch_size, source.shape[0])
        target[start:end] = np.asarray(source[start:end], dtype=dtype)
    target.flush()
    del target


def _load_answer_lookup(
    *,
    hf_root: Path | None,
    selected_rows: list[dict[str, Any]],
) -> dict[tuple[str, int], dict[str, Any]]:
    answers: dict[tuple[str, int], dict[str, Any]] = {}
    for row in selected_rows:
        answers[(str(row["config"]), int(row["row_index"]))] = {
            "final_decision": row.get("final_decision"),
            "long_answer": row.get("long_answer"),
            "has_answer": bool(row.get("final_decision") or row.get("long_answer")),
        }
    if hf_root is None or not hf_root.exists():
        return answers
    try:
        from datasets import load_from_disk
    except ImportError:
        return answers

    needed_by_config: dict[str, set[int]] = defaultdict(set)
    for row in selected_rows:
        needed_by_config[str(row["config"])].add(int(row["row_index"]))
    for config, row_indices in needed_by_config.items():
        config_path = hf_root / config
        if not config_path.exists():
            continue
        dataset = load_from_disk(str(config_path))
        for row_index in row_indices:
            source_row = dataset[int(row_index)]
            answers[(config, int(row_index))] = {
                "final_decision": source_row.get("final_decision"),
                "long_answer": source_row.get("long_answer"),
                "has_answer": bool(source_row.get("final_decision") or source_row.get("long_answer")),
            }
    return answers


def _build_topic_vocab(selected_rows: list[dict[str, Any]]) -> tuple[list[str], dict[str, int], list[list[int]]]:
    terms = sorted(
        {
            str(mesh)
            for row in selected_rows
            for mesh in (row.get("matching_selected_meshes") or [])
            if str(mesh).strip()
        }
    )
    if not terms:
        raise ValueError("selected_manifest rows do not contain matching_selected_meshes")
    topic_id_by_mesh = {mesh: index for index, mesh in enumerate(terms)}
    doc_indices_by_topic: list[list[int]] = [[] for _ in terms]
    for doc_index, row in enumerate(selected_rows):
        for mesh in row.get("matching_selected_meshes") or []:
            topic_id = topic_id_by_mesh[str(mesh)]
            doc_indices_by_topic[topic_id].append(doc_index)
    empty_topics = [topic for topic, indices in enumerate(doc_indices_by_topic) if not indices]
    if empty_topics:
        raise AssertionError(f"empty PubMedQA topic pools: {empty_topics[:5]}")
    return terms, topic_id_by_mesh, doc_indices_by_topic


def _sample_trace(
    *,
    user_count: int,
    queries_per_user: int,
    topic_count: int,
    doc_indices_by_topic: list[list[int]],
    dirichlet_alpha: float,
    random_seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if user_count <= 0:
        raise ValueError("user_count must be positive")
    if queries_per_user <= 0:
        raise ValueError("queries_per_user must be positive")
    if dirichlet_alpha <= 0:
        raise ValueError("dirichlet_alpha must be positive")

    query_count = user_count * queries_per_user
    user_ids = np.repeat(np.arange(user_count, dtype=np.int32), queries_per_user)
    topic_ids = np.empty(query_count, dtype=np.int32)
    doc_indices = np.empty(query_count, dtype=np.int32)
    query_variants = np.empty(query_count, dtype=np.int8)

    rng = np.random.default_rng(random_seed + 2)
    topic_options = np.arange(topic_count, dtype=np.int32)
    alpha = np.full(topic_count, dirichlet_alpha, dtype=np.float64)
    write_offset = 0
    for _user_id in range(user_count):
        mixture = rng.dirichlet(alpha)
        drawn_topics = rng.choice(topic_options, size=queries_per_user, replace=True, p=mixture)
        for topic_id in drawn_topics.tolist():
            pool = doc_indices_by_topic[int(topic_id)]
            topic_ids[write_offset] = int(topic_id)
            doc_indices[write_offset] = int(pool[int(rng.integers(0, len(pool)))])
            query_variants[write_offset] = int(rng.integers(0, 6))
            write_offset += 1
    return user_ids, topic_ids, doc_indices, query_variants


def _write_peer_future_docs(data_dir: Path, docs_by_peer: dict[int, set[str]]) -> dict[str, Any]:
    payload = {str(peer_id): sorted(doc_ids) for peer_id, doc_ids in sorted(docs_by_peer.items())}
    path = data_dir / PUBMEDQA_PEER_FUTURE_DOCS_FILENAME
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True)

    counts = [len(doc_ids) for doc_ids in payload.values()]
    summary = {
        "peer_count": len(payload),
        "doc_assignment_count": int(sum(counts)),
        "min_docs_per_peer": min(counts) if counts else 0,
        "max_docs_per_peer": max(counts) if counts else 0,
        "avg_docs_per_peer": (sum(counts) / len(counts)) if counts else 0.0,
        "path": str(path),
    }
    _write_json(data_dir / PUBMEDQA_PEER_FUTURE_DOCS_SUMMARY_FILENAME, summary)
    return summary


def precompute_pubmedqa_rag_artifacts(
    *,
    data_dir: Path,
    selected_manifest_path: Path = DEFAULT_SELECTION_DIR / "selected_manifest.jsonl",
    rephrased_queries_path: Path = DEFAULT_SELECTION_DIR / "rephrased_queries.jsonl",
    source_embeddings_dir: Path = DEFAULT_SOURCE_EMBEDDINGS_DIR,
    hf_root: Path | None = DEFAULT_HF_ROOT,
    user_count: int = DEFAULT_PUBMEDQA_USER_COUNT,
    queries_per_user: int = DEFAULT_PUBMEDQA_QUERIES_PER_USER,
    dirichlet_alpha: float = DEFAULT_PUBMEDQA_DIRICHLET_ALPHA,
    random_seed: int = DEFAULT_PUBMEDQA_RANDOM_SEED,
    embedding_dtype: str = "float16",
    embedding_batch_size: int = 4096,
) -> dict[str, Any]:
    data_dir.mkdir(parents=True, exist_ok=True)
    selected_rows = list(_iter_jsonl(selected_manifest_path))
    rephrased_rows = list(_iter_jsonl(rephrased_queries_path))
    if len(selected_rows) != len(rephrased_rows):
        raise ValueError(
            f"selected/rephrased row mismatch: {len(selected_rows)} != {len(rephrased_rows)}"
        )
    if not selected_rows:
        raise ValueError("selected_manifest is empty")
    for index, (selected, rephrased) in enumerate(zip(selected_rows, rephrased_rows, strict=True)):
        if int(selected["selected_index"]) != index or int(rephrased["selected_index"]) != index:
            raise ValueError(f"selected_index alignment failed at row {index}")
        if len(rephrased.get("rephrasings") or []) != 5:
            raise ValueError(f"row {index} does not have exactly 5 rephrasings")

    emb_manifest = _read_json(source_embeddings_dir / "embedding_manifest.json")
    dim = int(emb_manifest["embedding_dim"])
    doc_count = len(selected_rows)
    doc_pubids = _load_pubids(source_embeddings_dir / "doc_pubids.npy")
    if doc_pubids != [str(row["pubid"]) for row in selected_rows]:
        raise ValueError("source doc_pubids.npy is not aligned with selected_manifest.jsonl")
    source_doc_embeddings = _load_embedding_array(
        source_embeddings_dir / "doc_embeddings.npy",
        rows=doc_count,
        dim=dim,
    )
    source_original_embeddings = _load_embedding_array(
        source_embeddings_dir / "original_query_embeddings.npy",
        rows=doc_count,
        dim=dim,
    )
    source_rephrased_embeddings = _load_embedding_array(
        source_embeddings_dir / "rephrased_query_embeddings.npy",
        rows=doc_count * 5,
        dim=dim,
    )

    topic_terms, topic_id_by_mesh, doc_indices_by_topic = _build_topic_vocab(selected_rows)
    user_ids, topic_ids, doc_indices, query_variants = _sample_trace(
        user_count=user_count,
        queries_per_user=queries_per_user,
        topic_count=len(topic_terms),
        doc_indices_by_topic=doc_indices_by_topic,
        dirichlet_alpha=dirichlet_alpha,
        random_seed=random_seed,
    )
    query_count = int(user_ids.shape[0])
    qid_width = max(6, len(str(max(0, query_count - 1))))
    query_keys = np.asarray([_query_key(qid, qid_width) for qid in range(query_count)], dtype=str)
    target_pubids = np.asarray([doc_pubids[int(doc_index)] for doc_index in doc_indices], dtype=str)

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
    np.save(doc_keys_path, np.asarray(doc_pubids, dtype=str))
    np.save(query_keys_path, query_keys)

    dtype = np.dtype(embedding_dtype)
    _copy_embeddings_to_npy(
        source=source_doc_embeddings,
        destination=doc_embeddings_path,
        dtype=dtype,
        batch_size=embedding_batch_size,
    )
    query_embeddings = np.lib.format.open_memmap(
        query_embeddings_path,
        mode="w+",
        dtype=dtype,
        shape=(query_count, dim),
    )
    for start in range(0, query_count, embedding_batch_size):
        end = min(start + embedding_batch_size, query_count)
        chunk_doc_indices = doc_indices[start:end]
        chunk_variants = query_variants[start:end]
        out = np.empty((end - start, dim), dtype=np.float32)
        original_mask = chunk_variants == 0
        if np.any(original_mask):
            out[original_mask] = np.asarray(
                source_original_embeddings[chunk_doc_indices[original_mask]],
                dtype=np.float32,
            )
        if np.any(~original_mask):
            rephrase_rows = chunk_doc_indices[~original_mask] * 5 + (chunk_variants[~original_mask] - 1)
            out[~original_mask] = np.asarray(source_rephrased_embeddings[rephrase_rows], dtype=np.float32)
        query_embeddings[start:end] = out.astype(dtype, copy=False)
    query_embeddings.flush()
    del query_embeddings

    chronological_path = data_dir / PUBMEDQA_EVENTS_FILENAME
    randomized_path = data_dir / PUBMEDQA_RANDOMIZED_EVENTS_FILENAME
    searchable_path = data_dir / PUBMEDQA_SEEDED_SEARCHABLE_FILENAME
    query_topics_path = data_dir / PUBMEDQA_QUERY_TOPICS_FILENAME
    answer_targets_path = data_dir / PUBMEDQA_ANSWER_TARGETS_FILENAME
    permutation = np.random.default_rng(random_seed + 4).permutation(query_count)
    answers = _load_answer_lookup(hf_root=hf_root, selected_rows=selected_rows)

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
            searchable = searchable_for_qid(qid)
            searchable_count += int(searchable)
            randomized_writer.writerow(event_row_for_qid(qid, event_index))
            searchable_writer.writerow(
                {"query_id": qid, "searchable": "true" if searchable else "false"}
            )
            doc_index = int(doc_indices[qid])
            variant = int(query_variants[qid])
            selected_row = selected_rows[doc_index]
            answer = answers.get((str(selected_row["config"]), int(selected_row["row_index"])), {})
            topics_writer.writerow(
                {
                    "qid": qid,
                    "user_id": int(user_ids[qid]),
                    "query": str(query_keys[qid]),
                    "generated_topic": int(topic_ids[qid]),
                    "mesh_term": topic_terms[int(topic_ids[qid])],
                    "target_doc_id": str(target_pubids[qid]),
                    "target_doc_topic": int(topic_ids[qid]),
                    "query_variant": "original" if variant == 0 else f"rephrased_{variant}",
                    "source_selected_index": doc_index,
                    "searchable": "true" if searchable else "false",
                }
            )
            answer_payload = {
                "event_index": event_index,
                "qid": qid,
                "query_key": str(query_keys[qid]),
                "query_variant": "original" if variant == 0 else "rephrased",
                "rephrase_index": None if variant == 0 else variant - 1,
                "source_selected_index": doc_index,
                "source_pubid": str(selected_row["pubid"]),
                "target_doc_id": str(target_pubids[qid]),
                "config": str(selected_row["config"]),
                "row_index": int(selected_row["row_index"]),
                "question": (
                    selected_row["question"]
                    if variant == 0
                    else rephrased_rows[doc_index]["rephrasings"][variant - 1]
                ),
                "original_question": selected_row["question"],
                "final_decision": answer.get("final_decision"),
                "long_answer": answer.get("long_answer"),
                "has_answer": bool(answer.get("has_answer")),
            }
            answers_handle.write(json.dumps(answer_payload, ensure_ascii=False, sort_keys=True) + "\n")

    doc_topics_path = data_dir / PUBMEDQA_DOC_TOPICS_FILENAME
    with _open_text_writer(doc_topics_path) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("doc_id", "selected_index", "topic_id", "mesh_term", "is_primary"),
        )
        writer.writeheader()
        for doc_index, row in enumerate(selected_rows):
            matches = [str(mesh) for mesh in row.get("matching_selected_meshes") or []]
            for offset, mesh in enumerate(matches):
                writer.writerow(
                    {
                        "doc_id": str(row["pubid"]),
                        "selected_index": doc_index,
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

    manifest = {
        "dataset": "pubmedqa",
        "metadata_path": str(chronological_path),
        "searchable_queries_path": str(searchable_path),
        "output_dir": str(embeddings_dir),
        "model": emb_manifest.get("embedding_model", "sentence-transformers/all-mpnet-base-v2"),
        "embedding_dim": dim,
        "embedding_dtype": str(dtype),
        "query_keys_path": str(query_keys_path),
        "query_embeddings_path": str(query_embeddings_path),
        "query_count": query_count,
        "doc_keys_path": str(doc_keys_path),
        "doc_embeddings_path": str(doc_embeddings_path),
        "doc_count": doc_count,
        "user_count": user_count,
        "queries_per_user": queries_per_user,
        "dirichlet_alpha": dirichlet_alpha,
        "random_seed": random_seed,
    }
    manifest_path = embeddings_dir / "manifest.json"
    _write_json(manifest_path, manifest)

    summary = {
        "dataset": "pubmedqa",
        "data_dir": str(data_dir),
        "selected_document_count": doc_count,
        "topic_count": len(topic_terms),
        "user_count": user_count,
        "queries_per_user": queries_per_user,
        "query_count": query_count,
        "dirichlet_alpha": dirichlet_alpha,
        "random_seed": random_seed,
        "searchable_query_count": searchable_count,
        "query_variant_counts": {str(key): int(value) for key, value in sorted(variant_counts.items())},
        "top_topic_query_counts": [
            {"topic_id": topic_id, "mesh_term": topic_terms[topic_id], "query_count": int(count)}
            for topic_id, count in topic_counts.most_common(20)
        ],
        "manifest_path": str(manifest_path),
        "events_path": str(chronological_path),
        "randomized_events_path": str(randomized_path),
        "searchable_queries_path": str(searchable_path),
        "peer_future_docs_summary": peer_summary,
        "query_topics_path": str(query_topics_path),
        "doc_topics_path": str(doc_topics_path),
        "answer_targets_path": str(answer_targets_path),
    }
    _write_json(data_dir / PUBMEDQA_PRECOMPUTE_SUMMARY_FILENAME, summary)
    print(f"[pubmedqa-precompute] wrote manifest to {manifest_path}")
    print(f"[pubmedqa-precompute] wrote summary to {data_dir / PUBMEDQA_PRECOMPUTE_SUMMARY_FILENAME}")
    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build PubMedQA RAG simulator artifacts")
    parser.add_argument("--output-dir", default=None, help="Dataset directory; defaults via data-root resolver")
    parser.add_argument("--selected-manifest", type=Path, default=DEFAULT_SELECTION_DIR / "selected_manifest.jsonl")
    parser.add_argument("--rephrased-queries", type=Path, default=DEFAULT_SELECTION_DIR / "rephrased_queries.jsonl")
    parser.add_argument("--source-embeddings-dir", type=Path, default=DEFAULT_SOURCE_EMBEDDINGS_DIR)
    parser.add_argument("--hf-root", type=Path, default=DEFAULT_HF_ROOT)
    parser.add_argument("--users", type=int, default=DEFAULT_PUBMEDQA_USER_COUNT)
    parser.add_argument("--queries-per-user", type=int, default=DEFAULT_PUBMEDQA_QUERIES_PER_USER)
    parser.add_argument("--dirichlet-alpha", type=float, default=DEFAULT_PUBMEDQA_DIRICHLET_ALPHA)
    parser.add_argument("--random-seed", type=int, default=DEFAULT_PUBMEDQA_RANDOM_SEED)
    parser.add_argument("--embedding-dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument("--embedding-batch-size", type=int, default=4096)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    data_dir = Path(args.output_dir) if args.output_dir else resolve_dataset_dir("pubmedqa")
    precompute_pubmedqa_rag_artifacts(
        data_dir=data_dir,
        selected_manifest_path=args.selected_manifest,
        rephrased_queries_path=args.rephrased_queries,
        source_embeddings_dir=args.source_embeddings_dir,
        hf_root=args.hf_root,
        user_count=args.users,
        queries_per_user=args.queries_per_user,
        dirichlet_alpha=args.dirichlet_alpha,
        random_seed=args.random_seed,
        embedding_dtype=args.embedding_dtype,
        embedding_batch_size=args.embedding_batch_size,
    )


if __name__ == "__main__":
    main()

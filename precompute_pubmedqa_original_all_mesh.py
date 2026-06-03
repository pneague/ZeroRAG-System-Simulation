#!/usr/bin/env python3
"""Build full-corpus PubMedQA original-query artifacts over all MeSH terms."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, TextIO

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent
SCRIPT_DIR = REPO_ROOT / "scripts"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from pubmedqa_embed_full_retrieval import PUBMEDQA_CONFIGS, _clean_str  # noqa: E402
from src.data_root import resolve_dataset_dir  # noqa: E402
from src.dataset import AMAZON_EVENTS_FIELDNAMES, SEARCHABLE_QUERY_FIELDNAMES, synthetic_timestamp_for_index  # noqa: E402
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

DEFAULT_HF_ROOT = REPO_ROOT / "data" / "pubmedqa_rag" / "hf"
DEFAULT_SOURCE_FULL_DIR = (
    Path("/var/scratch/pmneague/Low_Coord_Semantic_DHT/results")
    / "pubmedqa_full_hnsw_original_m32_ef512"
)


@dataclass(frozen=True, slots=True)
class FullPubMedQaRow:
    source_index: int
    config: str
    row_index: int
    pubid: str
    question: str
    final_decision: str | None
    long_answer: str | None
    meshes: tuple[str, ...]


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")


def _open_text_writer(path: Path) -> TextIO:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".gz":
        return gzip.open(path, "wt", newline="", encoding="utf-8")
    return path.open("w", newline="", encoding="utf-8")


def _query_key(qid: int, width: int) -> str:
    return f"pmq_q_{qid:0{width}d}"


def _embedding_paths(source_full_dir: Path) -> dict[str, Path]:
    embeddings_dir = source_full_dir / "embeddings"
    return {
        "state": source_full_dir / "embedding_state.json",
        "manifest": source_full_dir / "run_manifest.json",
        "corpus_manifest": source_full_dir / "corpus_manifest.jsonl",
        "doc_pubids": embeddings_dir / "doc_pubids.npy",
        "doc_embeddings": embeddings_dir / "doc_embeddings.npy",
        "query_pubids": embeddings_dir / "query_pubids.npy",
        "query_embeddings": embeddings_dir / "query_embeddings.npy",
    }


def _load_embedding_array(path: Path, *, rows: int, dim: int) -> np.ndarray:
    try:
        loaded = np.load(path, mmap_mode="r")
        if loaded.shape != (rows, dim):
            raise ValueError(f"{path} shape {loaded.shape} != {(rows, dim)}")
        return loaded
    except (OSError, ValueError):
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


def _load_full_rows(*, hf_root: Path, source_full_dir: Path) -> tuple[list[FullPubMedQaRow], dict[str, Any]]:
    from datasets import load_from_disk

    paths = _embedding_paths(source_full_dir)
    manifest_rows = list(_iter_jsonl(paths["corpus_manifest"]))
    manifest_by_key = {
        (str(row["config"]), int(row["row_index"])): row for row in manifest_rows
    }

    rows: list[FullPubMedQaRow] = []
    config_counts: dict[str, int] = {}
    rows_with_meshes = 0
    mesh_counts: Counter[str] = Counter()
    source_index = 0
    for config in PUBMEDQA_CONFIGS:
        dataset = load_from_disk(str(hf_root / config))
        config_counts[config] = len(dataset)
        for row_index, row in enumerate(dataset):
            manifest_row = manifest_by_key.get((config, row_index))
            if manifest_row is None:
                raise ValueError(f"Missing corpus manifest row for {config}:{row_index}")
            pubid = _clean_str(row.get("pubid"))
            if pubid != _clean_str(manifest_row.get("pubid")):
                raise ValueError(
                    f"HF/manifest pubid mismatch at {config}:{row_index}: "
                    f"{pubid!r} != {_clean_str(manifest_row.get('pubid'))!r}"
                )
            meshes = tuple(
                sorted(
                    {
                        _clean_str(mesh)
                        for mesh in ((row.get("context") or {}).get("meshes") or [])
                        if _clean_str(mesh)
                    }
                )
            )
            if meshes:
                rows_with_meshes += 1
                mesh_counts.update(meshes)
            rows.append(
                FullPubMedQaRow(
                    source_index=source_index,
                    config=config,
                    row_index=row_index,
                    pubid=pubid,
                    question=_clean_str(row.get("question")),
                    final_decision=row.get("final_decision"),
                    long_answer=row.get("long_answer"),
                    meshes=meshes,
                )
            )
            source_index += 1

    if len(rows) != len(manifest_rows):
        raise ValueError(f"HF row count {len(rows)} != corpus manifest row count {len(manifest_rows)}")

    summary = {
        "data_root": str(hf_root),
        "configs": config_counts,
        "row_count": len(rows),
        "rows_with_meshes": rows_with_meshes,
        "unique_mesh_terms": len(mesh_counts),
        "mesh_doc_frequency_min": min(mesh_counts.values()) if mesh_counts else 0,
        "mesh_doc_frequency_max": max(mesh_counts.values()) if mesh_counts else 0,
        "top_mesh_doc_frequencies": [
            {"mesh_term": mesh, "doc_frequency": int(count)}
            for mesh, count in mesh_counts.most_common(20)
        ],
    }
    return rows, summary


def _build_topic_pools(
    rows: list[FullPubMedQaRow],
) -> tuple[list[str], dict[str, int], list[list[int]]]:
    terms = sorted({mesh for row in rows for mesh in row.meshes})
    if not terms:
        raise ValueError("No MeSH terms found in full PubMedQA corpus")
    topic_id_by_mesh = {mesh: index for index, mesh in enumerate(terms)}
    doc_indices_by_topic: list[list[int]] = [[] for _ in terms]
    for row in rows:
        for mesh in row.meshes:
            doc_indices_by_topic[topic_id_by_mesh[mesh]].append(row.source_index)
    return terms, topic_id_by_mesh, doc_indices_by_topic


def _sample_trace(
    *,
    peer_count: int,
    queries_per_peer: int,
    topic_count: int,
    doc_indices_by_topic: list[list[int]],
    alpha: float,
    random_seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    if peer_count <= 0:
        raise ValueError("peer_count must be positive")
    if queries_per_peer <= 0:
        raise ValueError("queries_per_peer must be positive")
    if alpha <= 0:
        raise ValueError("alpha must be positive")

    query_count = peer_count * queries_per_peer
    user_ids = np.repeat(np.arange(peer_count, dtype=np.int32), queries_per_peer)
    topic_ids = np.empty(query_count, dtype=np.int32)
    doc_indices = np.empty(query_count, dtype=np.int32)
    topic_counts: Counter[int] = Counter()
    unique_topics_per_peer: list[int] = []

    rng = np.random.default_rng(random_seed)
    total_alpha = topic_count * alpha
    write_offset = 0
    for _user_id in range(peer_count):
        peer_topic_counts: Counter[int] = Counter()
        drawn_topics: list[int] = []
        for draw_index in range(queries_per_peer):
            existing_mass = draw_index
            if existing_mass and rng.random() < (existing_mass / (existing_mass + total_alpha)):
                local_topic_ids = list(peer_topic_counts.keys())
                local_counts = np.asarray(
                    [peer_topic_counts[topic_id] for topic_id in local_topic_ids],
                    dtype=np.float64,
                )
                topic_id = int(rng.choice(local_topic_ids, p=local_counts / local_counts.sum()))
            else:
                topic_id = int(rng.integers(0, topic_count))
            peer_topic_counts[topic_id] += 1
            drawn_topics.append(topic_id)
        unique_topics_per_peer.append(len(peer_topic_counts))
        for topic_id in drawn_topics:
            pool = doc_indices_by_topic[topic_id]
            topic_ids[write_offset] = topic_id
            doc_indices[write_offset] = int(pool[int(rng.integers(0, len(pool)))])
            topic_counts[topic_id] += 1
            write_offset += 1

    summary = {
        "avg_unique_topics_per_peer": float(np.mean(unique_topics_per_peer)) if unique_topics_per_peer else 0.0,
        "min_unique_topics_per_peer": int(min(unique_topics_per_peer)) if unique_topics_per_peer else 0,
        "max_unique_topics_per_peer": int(max(unique_topics_per_peer)) if unique_topics_per_peer else 0,
        "top_sampled_topic_ids": [
            {"topic_id": int(topic_id), "query_count": int(count)}
            for topic_id, count in topic_counts.most_common(20)
        ],
    }
    return user_ids, topic_ids, doc_indices, summary


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


def precompute_pubmedqa_original_all_mesh_artifacts(
    *,
    data_dir: Path,
    hf_root: Path = DEFAULT_HF_ROOT,
    source_full_dir: Path = DEFAULT_SOURCE_FULL_DIR,
    peer_count: int = 50_000,
    queries_per_peer: int = 20,
    alpha: float = 0.00007,
    random_seed: int = DEFAULT_PUBMEDQA_RANDOM_SEED,
    embedding_dtype: str = "float16",
    embedding_batch_size: int = 4096,
) -> dict[str, Any]:
    data_dir.mkdir(parents=True, exist_ok=True)
    paths = _embedding_paths(source_full_dir)
    state = json.loads(paths["state"].read_text(encoding="utf-8"))
    emb_manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    row_count = int(emb_manifest["row_count"])
    dim = int(emb_manifest["embedding_dim"])
    if int(state.get("doc_rows_embedded", 0)) != row_count:
        raise RuntimeError("Source full PubMedQA doc embeddings are incomplete")
    if int(state.get("query_rows_embedded", 0)) != row_count:
        raise RuntimeError("Source full PubMedQA query embeddings are incomplete")

    rows, mesh_summary = _load_full_rows(hf_root=hf_root, source_full_dir=source_full_dir)
    if len(rows) != row_count:
        raise ValueError(f"Loaded {len(rows)} PubMedQA rows but embeddings have {row_count}")

    source_doc_pubids = np.load(paths["doc_pubids"], allow_pickle=True).astype(str)
    source_query_pubids = np.load(paths["query_pubids"], allow_pickle=True).astype(str)
    row_pubids = np.asarray([row.pubid for row in rows], dtype=str)
    if not np.array_equal(source_doc_pubids, row_pubids):
        raise ValueError("Source doc_pubids.npy is not aligned with full PubMedQA rows")
    if not np.array_equal(source_query_pubids, row_pubids):
        raise ValueError("Source query_pubids.npy is not aligned with full PubMedQA rows")

    topic_terms, topic_id_by_mesh, doc_indices_by_topic = _build_topic_pools(rows)
    user_ids, topic_ids, doc_indices, sample_summary = _sample_trace(
        peer_count=peer_count,
        queries_per_peer=queries_per_peer,
        topic_count=len(topic_terms),
        doc_indices_by_topic=doc_indices_by_topic,
        alpha=alpha,
        random_seed=random_seed,
    )
    query_count = int(user_ids.shape[0])
    qid_width = max(6, len(str(max(0, query_count - 1))))
    query_keys = np.asarray([_query_key(qid, qid_width) for qid in range(query_count)], dtype=str)
    target_pubids = np.asarray([rows[int(doc_index)].pubid for doc_index in doc_indices], dtype=str)

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

    dtype = np.dtype(embedding_dtype)
    source_doc_embeddings = _load_embedding_array(paths["doc_embeddings"], rows=row_count, dim=dim)
    source_query_embeddings = _load_embedding_array(paths["query_embeddings"], rows=row_count, dim=dim)
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
        query_embeddings[start:end] = np.asarray(
            source_query_embeddings[doc_indices[start:end]],
            dtype=dtype,
        )
    query_embeddings.flush()
    del query_embeddings

    chronological_path = data_dir / PUBMEDQA_EVENTS_FILENAME
    randomized_path = data_dir / PUBMEDQA_RANDOMIZED_EVENTS_FILENAME
    searchable_path = data_dir / PUBMEDQA_SEEDED_SEARCHABLE_FILENAME
    query_topics_path = data_dir / PUBMEDQA_QUERY_TOPICS_FILENAME
    answer_targets_path = data_dir / PUBMEDQA_ANSWER_TARGETS_FILENAME
    permutation = np.random.default_rng(random_seed + 4).permutation(query_count)

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
            searchable = searchable_for_qid(qid)
            searchable_count += int(searchable)
            randomized_writer.writerow(event_row_for_qid(qid, event_index))
            searchable_writer.writerow(
                {"query_id": qid, "searchable": "true" if searchable else "false"}
            )
            topics_writer.writerow(
                {
                    "qid": qid,
                    "user_id": int(user_ids[qid]),
                    "query": str(query_keys[qid]),
                    "generated_topic": int(topic_ids[qid]),
                    "mesh_term": topic_terms[int(topic_ids[qid])],
                    "target_doc_id": str(target_pubids[qid]),
                    "target_doc_topic": int(topic_ids[qid]),
                    "query_variant": "original",
                    "source_selected_index": int(source_row.source_index),
                    "searchable": "true" if searchable else "false",
                }
            )
            answer_payload = {
                "event_index": event_index,
                "qid": qid,
                "query_key": str(query_keys[qid]),
                "query_variant": "original",
                "rephrase_index": None,
                "source_selected_index": int(source_row.source_index),
                "source_pubid": source_row.pubid,
                "target_doc_id": str(target_pubids[qid]),
                "config": source_row.config,
                "row_index": int(source_row.row_index),
                "question": source_row.question,
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

    manifest = {
        "dataset": "pubmedqa",
        "pubmedqa_variant": "original_all_mesh",
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
        "doc_count": row_count,
        "user_count": peer_count,
        "queries_per_user": queries_per_peer,
        "dirichlet_alpha": alpha,
        "random_seed": random_seed,
        "source_full_dir": str(source_full_dir),
        "source_query_variant": "original",
    }
    manifest_path = embeddings_dir / "manifest.json"
    _write_json(manifest_path, manifest)

    top_sampled_topics = [
        {"topic_id": topic_id, "mesh_term": topic_terms[topic_id], "query_count": int(count)}
        for topic_id, count in topic_counts.most_common(20)
    ]
    summary = {
        "dataset": "pubmedqa",
        "pubmedqa_variant": "original_all_mesh",
        "data_dir": str(data_dir),
        "source_full_dir": str(source_full_dir),
        "full_document_count": row_count,
        "topic_count": len(topic_terms),
        "mesh_summary": mesh_summary,
        "user_count": peer_count,
        "queries_per_user": queries_per_peer,
        "query_count": query_count,
        "dirichlet_alpha": alpha,
        "random_seed": random_seed,
        "searchable_query_count": searchable_count,
        "query_variant_counts": {"original": query_count},
        "sample_summary": {
            **sample_summary,
            "top_sampled_topics": top_sampled_topics,
        },
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
    print(f"[pubmedqa-original-all-mesh] wrote manifest to {manifest_path}", flush=True)
    print(
        f"[pubmedqa-original-all-mesh] wrote summary to {data_dir / PUBMEDQA_PRECOMPUTE_SUMMARY_FILENAME}",
        flush=True,
    )
    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=None, help="Dataset dir; defaults via data-root resolver")
    parser.add_argument("--hf-root", type=Path, default=DEFAULT_HF_ROOT)
    parser.add_argument("--source-full-dir", type=Path, default=DEFAULT_SOURCE_FULL_DIR)
    parser.add_argument("--peers", type=int, default=50_000)
    parser.add_argument("--queries-per-peer", type=int, default=20)
    parser.add_argument("--alpha", type=float, default=0.00007)
    parser.add_argument("--random-seed", type=int, default=DEFAULT_PUBMEDQA_RANDOM_SEED)
    parser.add_argument("--embedding-dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument("--embedding-batch-size", type=int, default=4096)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    data_dir = args.output_dir or resolve_dataset_dir("pubmedqa")
    precompute_pubmedqa_original_all_mesh_artifacts(
        data_dir=Path(data_dir),
        hf_root=args.hf_root,
        source_full_dir=args.source_full_dir,
        peer_count=args.peers,
        queries_per_peer=args.queries_per_peer,
        alpha=args.alpha,
        random_seed=args.random_seed,
        embedding_dtype=args.embedding_dtype,
        embedding_batch_size=args.embedding_batch_size,
    )


if __name__ == "__main__":
    main()

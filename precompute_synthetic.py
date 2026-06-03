#!/usr/bin/env python3
"""Precompute a fully synthetic embedding-space dataset.

The generator samples topic centroids on the unit sphere, samples documents and
queries around those centroids, assigns every query to the nearest sampled
document, then drops events whose target document was selected at most once.
It writes simulator-ready artifacts directly, including a float16 embedding
manifest, because there is no text embedding model involved.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterator, Sequence, TextIO

import numpy as np
from numpy.lib.format import open_memmap

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.data_root import resolve_dataset_dir
from src.dataset import (
    AMAZON_EVENTS_FIELDNAMES,
    SEARCHABLE_QUERY_FIELDNAMES,
    synthetic_timestamp_for_index,
)
from src.synthetic_dataset import (
    DEFAULT_DIRICHLET_ALPHA,
    DEFAULT_DOCS_PER_TOPIC,
    DEFAULT_EMBEDDING_DIM,
    DEFAULT_MIN_TARGET_COUNT,
    DEFAULT_QUERIES_PER_USER,
    DEFAULT_RANDOM_SEED,
    DEFAULT_SIGMA_SWEEP,
    DEFAULT_TOP_K,
    DEFAULT_TOPIC_COUNT,
    DEFAULT_USER_COUNT,
    SYNTHETIC_DOC_TOPICS_FILENAME,
    SYNTHETIC_DRIFT_OWNER_INJECTIONS_FILENAME,
    SYNTHETIC_EVENTS_FILENAME,
    SYNTHETIC_PEER_FUTURE_DOCS_FILENAME,
    SYNTHETIC_PEER_FUTURE_DOCS_SUMMARY_FILENAME,
    SYNTHETIC_PRECOMPUTE_SUMMARY_FILENAME,
    SYNTHETIC_QUERY_TOPICS_FILENAME,
    SYNTHETIC_RANDOMIZED_EVENTS_FILENAME,
    SYNTHETIC_SEEDED_SEARCHABLE_FILENAME,
    SYNTHETIC_SIGMA_SWEEP_FILENAME,
    SYNTHETIC_TOPICS_FILENAME,
    SyntheticConfig,
    batched_nearest_indices,
    doc_id_for_index,
    ensure_dataset_dir,
    fixed_width_for_count,
    nearest_doc_stats,
    parse_sigma_values,
    percentile_summary,
    query_key_for_id,
    sample_noisy_vectors,
    sample_unit_vectors,
)


def _open_text_writer(path: Path) -> TextIO:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".gz":
        return gzip.open(path, "wt", newline="", encoding="utf-8")
    return path.open("w", newline="", encoding="utf-8")


def _write_json_gz(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True)
        handle.write("\n")


def _unlink_if_exists(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def _generate_centroids(config: SyntheticConfig) -> np.ndarray:
    rng = np.random.default_rng(config.random_seed)
    return sample_unit_vectors(
        rng,
        count=config.topic_count,
        embedding_dim=config.embedding_dim,
    )


def _generate_doc_vectors(
    config: SyntheticConfig,
    *,
    centroids: np.ndarray,
    doc_counts_by_topic: Sequence[int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if doc_counts_by_topic is None:
        doc_topics = np.repeat(
            np.arange(config.topic_count, dtype=np.int32),
            config.docs_per_topic,
        )
    else:
        if len(doc_counts_by_topic) != config.topic_count:
            raise ValueError(
                "doc_counts_by_topic must contain exactly one count per topic "
                f"({config.topic_count})"
            )
        counts = np.asarray(doc_counts_by_topic, dtype=np.int64)
        if np.any(counts < 0):
            raise ValueError("doc_counts_by_topic cannot contain negative counts")
        doc_topics = np.repeat(np.arange(config.topic_count, dtype=np.int32), counts)
    rng = np.random.default_rng(config.random_seed + 1)
    docs = sample_noisy_vectors(
        rng,
        centroids=centroids,
        topic_indices=doc_topics,
        sigma=float(config.sigma or 0.0),
    )
    return docs, doc_topics


def _iter_query_batches(
    config: SyntheticConfig,
    *,
    centroids: np.ndarray,
    user_batch_size: int,
    drift_topic_id: int | None = None,
    drift_start_offset: int | None = None,
    drift_end_offset: int | None = None,
    drift_probability: float = 0.0,
) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    if user_batch_size <= 0:
        raise ValueError("user_batch_size must be positive")
    if config.sigma is None:
        raise ValueError("config.sigma must be chosen before query generation")

    rng = np.random.default_rng(config.random_seed + 2)
    if drift_topic_id is None:
        topic_options = np.arange(config.topic_count)
    else:
        topic_options = np.asarray(
            [topic_id for topic_id in range(config.topic_count) if topic_id != drift_topic_id],
            dtype=np.int32,
        )
    alpha = np.full(topic_options.shape[0], config.dirichlet_alpha, dtype=np.float64)
    for user_start in range(0, config.user_count, user_batch_size):
        user_end = min(user_start + user_batch_size, config.user_count)
        batch_user_count = user_end - user_start
        mixtures = rng.dirichlet(alpha, size=batch_user_count)
        topic_chunks: list[np.ndarray] = []
        for row in mixtures:
            topic_chunks.append(
                rng.choice(
                    topic_options,
                    size=config.queries_per_user,
                    replace=True,
                    p=row,
                ).astype(np.int32, copy=False)
            )
        query_topics = np.concatenate(topic_chunks).astype(np.int32, copy=False)
        qid_start = user_start * config.queries_per_user
        qids = np.arange(qid_start, qid_start + query_topics.shape[0], dtype=np.int64)
        if drift_topic_id is not None and drift_probability > 0.0:
            if drift_topic_id < 0 or drift_topic_id >= config.topic_count:
                raise ValueError(
                    f"drift_topic_id={drift_topic_id} is outside 0..{config.topic_count - 1}"
                )
            start_offset = 0 if drift_start_offset is None else drift_start_offset
            end_offset = config.queries_per_user if drift_end_offset is None else drift_end_offset
            query_offsets = qids % config.queries_per_user
            in_drift_window = (query_offsets >= start_offset) & (query_offsets < end_offset)
            if np.any(in_drift_window):
                drift_draws = rng.random(query_topics.shape[0])
                drift_mask = in_drift_window & (drift_draws < drift_probability)
                query_topics[drift_mask] = drift_topic_id
        user_ids = np.repeat(
            np.arange(user_start, user_end, dtype=np.int32),
            config.queries_per_user,
        )
        query_vectors = sample_noisy_vectors(
            rng,
            centroids=centroids,
            topic_indices=query_topics,
            sigma=config.sigma,
        )
        yield qids, user_ids, query_topics, query_vectors


def _choose_recommended_sigma(results: list[dict[str, object]]) -> float:
    if not results:
        raise ValueError("cannot choose sigma from empty sweep")
    viable = [
        result
        for result in results
        if 0.10 <= float(result["nearest_doc_cross_topic_rate"]) <= 0.30
        and float(result["doc_nearest_centroid_accuracy"]) >= 0.80
        and float(result["query_nearest_centroid_accuracy"]) >= 0.80
    ]
    if viable:
        return float(
            min(viable, key=lambda item: abs(float(item["nearest_doc_cross_topic_rate"]) - 0.20))[
                "sigma"
            ]
        )
    return float(
        min(results, key=lambda item: abs(float(item["nearest_doc_cross_topic_rate"]) - 0.20))[
            "sigma"
        ]
    )


def run_sigma_sweep(
    *,
    data_dir: Path,
    topic_count: int,
    docs_per_topic: int,
    embedding_dim: int,
    random_seed: int,
    sigma_values: tuple[float, ...],
    calibration_query_count: int,
    nearest_batch_size: int,
    top_k: int,
) -> dict[str, object]:
    if calibration_query_count <= 0:
        raise ValueError("calibration_query_count must be positive")

    centroid_rng = np.random.default_rng(random_seed)
    centroids = sample_unit_vectors(
        centroid_rng,
        count=topic_count,
        embedding_dim=embedding_dim,
    )
    doc_topics = np.repeat(np.arange(topic_count, dtype=np.int32), docs_per_topic)
    results: list[dict[str, object]] = []

    for sigma in sigma_values:
        doc_rng = np.random.default_rng(random_seed + 1)
        query_rng = np.random.default_rng(random_seed + 2)
        docs = sample_noisy_vectors(
            doc_rng,
            centroids=centroids,
            topic_indices=doc_topics,
            sigma=sigma,
        )
        query_topics = query_rng.integers(
            0,
            topic_count,
            size=calibration_query_count,
            dtype=np.int32,
        )
        queries = sample_noisy_vectors(
            query_rng,
            centroids=centroids,
            topic_indices=query_topics,
            sigma=sigma,
        )

        doc_nearest_centroids, _ = batched_nearest_indices(
            docs,
            centroids,
            batch_size=nearest_batch_size,
        )
        query_nearest_centroids, _ = batched_nearest_indices(
            queries,
            centroids,
            batch_size=nearest_batch_size,
        )
        stats = nearest_doc_stats(
            queries,
            query_topics=query_topics,
            docs=docs,
            doc_topics=doc_topics,
            batch_size=nearest_batch_size,
            top_k=top_k,
        )

        result = {
            "sigma": float(sigma),
            "doc_nearest_centroid_accuracy": float(np.mean(doc_nearest_centroids == doc_topics)),
            "query_nearest_centroid_accuracy": float(
                np.mean(query_nearest_centroids == query_topics)
            ),
            "nearest_doc_cross_topic_rate": float(stats.cross_topic_rate),
            "same_topic_top_k_rate": float(stats.same_topic_top_k_rate),
            "query_target_cosine": percentile_summary(stats.nearest_similarities),
        }
        results.append(result)
        print(
            "[synthetic-sigma] "
            f"sigma={sigma:g} cross_topic={result['nearest_doc_cross_topic_rate']:.3f} "
            f"doc_centroid_acc={result['doc_nearest_centroid_accuracy']:.3f} "
            f"query_centroid_acc={result['query_nearest_centroid_accuracy']:.3f}",
            flush=True,
        )

    recommended_sigma = _choose_recommended_sigma(results)
    payload = {
        "topic_count": topic_count,
        "docs_per_topic": docs_per_topic,
        "embedding_dim": embedding_dim,
        "random_seed": random_seed,
        "calibration_query_count": calibration_query_count,
        "top_k": top_k,
        "acceptance_target": {
            "nearest_doc_cross_topic_rate_min": 0.10,
            "nearest_doc_cross_topic_rate_max": 0.30,
            "centroid_accuracy_min": 0.80,
        },
        "recommended_sigma": recommended_sigma,
        "results": results,
    }
    output_path = data_dir / SYNTHETIC_SIGMA_SWEEP_FILENAME
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[synthetic-sigma] wrote sweep report to {output_path}")
    return payload


def _write_doc_artifacts(
    *,
    data_dir: Path,
    doc_vectors: np.ndarray,
    doc_topics: np.ndarray,
    target_counts: np.ndarray,
    retained_doc_mask: np.ndarray,
    doc_ids: list[str],
) -> dict[str, object]:
    topics_path = data_dir / SYNTHETIC_TOPICS_FILENAME
    # topics.npy is written by the caller because it owns the centroid matrix.

    doc_topics_path = data_dir / SYNTHETIC_DOC_TOPICS_FILENAME
    with _open_text_writer(doc_topics_path) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("doc_index", "doc_id", "topic_id", "target_count", "retained"),
        )
        writer.writeheader()
        for doc_index, doc_id in enumerate(doc_ids):
            writer.writerow(
                {
                    "doc_index": doc_index,
                    "doc_id": doc_id,
                    "topic_id": int(doc_topics[doc_index]),
                    "target_count": int(target_counts[doc_index]),
                    "retained": "true" if bool(retained_doc_mask[doc_index]) else "false",
                }
            )

    embeddings_dir = data_dir / "embeddings"
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    retained_indices = np.flatnonzero(retained_doc_mask)
    doc_key_width = fixed_width_for_count(len(doc_ids))
    retained_doc_ids = [doc_ids[int(index)] for index in retained_indices]
    doc_keys_path = embeddings_dir / "doc_ids.npy"
    np.save(
        doc_keys_path,
        np.asarray(retained_doc_ids, dtype=f"<U{len(doc_id_for_index(0, width=doc_key_width))}"),
    )

    doc_embeddings_path = embeddings_dir / "doc_embeddings.npy"
    doc_embeddings = open_memmap(
        doc_embeddings_path,
        mode="w+",
        dtype=np.float16,
        shape=(len(retained_indices), doc_vectors.shape[1]),
    )
    if len(retained_indices):
        doc_embeddings[:] = doc_vectors[retained_indices].astype(np.float16, copy=False)
    del doc_embeddings

    return {
        "topics_path": str(topics_path),
        "doc_topics_path": str(doc_topics_path),
        "doc_keys_path": str(doc_keys_path),
        "doc_embeddings_path": str(doc_embeddings_path),
        "retained_doc_ids": retained_doc_ids,
        "retained_doc_indices": retained_indices,
    }


def _build_retained_ownership(
    *,
    target_doc_indices: np.ndarray,
    retained_event_mask: np.ndarray,
    queries_per_user: int,
    doc_ids: list[str],
    user_count: int,
    random_seed: int,
    doc_topics: np.ndarray | None = None,
    drift_topic_id: int | None = None,
) -> tuple[dict[int, set[int]], dict[int, set[str]]]:
    if drift_topic_id is not None and doc_topics is None:
        raise ValueError("doc_topics is required when drift_topic_id is set")

    retained_doc_owners: dict[int, set[int]] = defaultdict(set)
    retained_peer_ids: set[int] = set()
    retained_qids = np.flatnonzero(retained_event_mask)
    for raw_qid in retained_qids:
        qid = int(raw_qid)
        user_id = qid // queries_per_user
        doc_index = int(target_doc_indices[qid])
        retained_peer_ids.add(user_id)
        retained_doc_owners[doc_index].add(user_id)

    owners_by_doc_index: dict[int, set[int]] = defaultdict(set)
    docs_by_peer: dict[int, set[str]] = defaultdict(set)
    for doc_index, event_owners in sorted(retained_doc_owners.items()):
        if (
            drift_topic_id is not None
            and doc_topics is not None
            and int(doc_topics[doc_index]) == drift_topic_id
        ):
            # Drift-topic docs are introduced at the global drift boundary by
            # a runtime injection artifact, not seeded in the initial inventory.
            continue
        else:
            owners = event_owners
        owners_by_doc_index[doc_index].update(owners)
        for user_id in owners:
            docs_by_peer[user_id].add(doc_ids[doc_index])
    return owners_by_doc_index, docs_by_peer


def _build_topic_doc_indices(doc_topics: np.ndarray, topic_count: int) -> list[np.ndarray]:
    return [
        np.flatnonzero(doc_topics == topic_id).astype(np.int32, copy=False)
        for topic_id in range(topic_count)
    ]


def _assign_capped_topic_targets(
    *,
    query_vectors: np.ndarray,
    user_ids: np.ndarray,
    query_topics: np.ndarray,
    doc_vectors: np.ndarray,
    topic_doc_indices: list[np.ndarray],
    used_docs_by_user_topic: dict[tuple[int, int], set[int]],
) -> np.ndarray:
    """Assign nearest same-topic docs without replacement per user/topic.

    If a user has already consumed every document in a topic, later query draws
    for that topic are marked ``-1`` and dropped before artifact writing.
    """

    assigned = np.full(query_topics.shape[0], -1, dtype=np.int32)
    for local_offset, raw_topic_id in enumerate(query_topics):
        topic_id = int(raw_topic_id)
        if topic_id < 0 or topic_id >= len(topic_doc_indices):
            raise ValueError(f"query topic {topic_id} is outside available topics")
        candidate_indices = topic_doc_indices[topic_id]
        if candidate_indices.size == 0:
            continue

        key = (int(user_ids[local_offset]), topic_id)
        used = used_docs_by_user_topic[key]
        if len(used) >= candidate_indices.size:
            continue

        if used:
            available = np.asarray(
                [int(index) for index in candidate_indices.tolist() if int(index) not in used],
                dtype=np.int32,
            )
        else:
            available = candidate_indices
        if available.size == 0:
            continue

        similarities = doc_vectors[available] @ query_vectors[local_offset]
        selected = int(available[int(np.argmax(similarities))])
        assigned[local_offset] = selected
        used.add(selected)
    return assigned


def _write_peer_future_docs(
    *,
    docs_by_peer: dict[int, set[str]],
    output_path: Path,
    summary_path: Path,
) -> dict[str, object]:
    payload = {
        str(peer_id): sorted(doc_ids)
        for peer_id, doc_ids in sorted(docs_by_peer.items())
        if doc_ids
    }
    _write_json_gz(output_path, payload)
    total_pairs = sum(len(doc_ids) for doc_ids in payload.values())
    summary = {
        "output_path": str(output_path),
        "peer_count": len(payload),
        "total_unique_peer_doc_pairs": total_pairs,
        "avg_docs_per_peer": total_pairs / len(payload) if payload else 0.0,
        "max_docs_per_peer": max((len(doc_ids) for doc_ids in payload.values()), default=0),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _write_drift_owner_injections(
    *,
    output_path: Path,
    drift_doc_indices: np.ndarray,
    doc_ids: list[str],
    owner_pool: Sequence[int],
    event_index: int,
    random_seed: int,
) -> dict[str, object]:
    if drift_doc_indices.size == 0:
        injections: list[dict[str, int | str]] = []
    else:
        if not owner_pool:
            raise ValueError("cannot create drift owner injections without any retained peers")
        rng = np.random.default_rng(random_seed + 3)
        replace = len(owner_pool) < int(drift_doc_indices.size)
        selected_peers = rng.choice(
            np.asarray(list(owner_pool), dtype=np.int64),
            size=int(drift_doc_indices.size),
            replace=replace,
        )
        injections = [
            {
                "event_index": event_index,
                "peer_id": int(peer_id),
                "doc_id": doc_ids[int(doc_index)],
            }
            for doc_index, peer_id in zip(drift_doc_indices.tolist(), selected_peers.tolist())
        ]

    _write_json_gz(output_path, injections)
    return {
        "output_path": str(output_path),
        "injection_count": len(injections),
        "event_index": event_index,
        "unique_doc_count": len({str(item["doc_id"]) for item in injections}),
        "unique_peer_count": len({int(item["peer_id"]) for item in injections}),
    }


def _write_retained_query_artifacts(
    *,
    config: SyntheticConfig,
    data_dir: Path,
    centroids: np.ndarray,
    doc_vectors: np.ndarray,
    doc_topics: np.ndarray,
    target_doc_indices: np.ndarray,
    query_vectors_by_qid: np.ndarray,
    retained_event_mask: np.ndarray,
    owners_by_doc_index: dict[int, set[int]],
    doc_ids: list[str],
    retained_doc_ids: list[str],
    user_batch_size: int,
    drift_topic_id: int | None = None,
    drift_start_fraction: float = 0.0,
    drift_end_fraction: float = 1.0,
    drift_probability: float = 0.0,
) -> dict[str, object]:
    retained_event_count = int(np.sum(retained_event_mask))
    embeddings_dir = data_dir / "embeddings"
    embeddings_dir.mkdir(parents=True, exist_ok=True)

    query_width = fixed_width_for_count(config.query_count, minimum=10)
    query_key_len = len(query_key_for_id(0, width=query_width))
    query_keys_path = embeddings_dir / "query_texts.npy"
    query_embeddings_path = embeddings_dir / "query_embeddings.npy"
    query_keys = open_memmap(
        query_keys_path,
        mode="w+",
        dtype=f"<U{query_key_len}",
        shape=(retained_event_count,),
    )
    query_embeddings = open_memmap(
        query_embeddings_path,
        mode="w+",
        dtype=np.float16,
        shape=(retained_event_count, config.embedding_dim),
    )

    events_path = data_dir / SYNTHETIC_EVENTS_FILENAME
    randomized_path = data_dir / SYNTHETIC_RANDOMIZED_EVENTS_FILENAME
    searchable_path = data_dir / SYNTHETIC_SEEDED_SEARCHABLE_FILENAME
    query_topics_path = data_dir / SYNTHETIC_QUERY_TOPICS_FILENAME

    searchable_count = 0
    write_index = 0
    retained_doc_id_set = set(retained_doc_ids)
    records: list[tuple[int, int, int, int]] = []
    for raw_qid in np.flatnonzero(retained_event_mask):
        qid = int(raw_qid)
        user_id = qid // config.queries_per_user
        doc_index = int(target_doc_indices[qid])
        doc_id = doc_ids[doc_index]
        if doc_id not in retained_doc_id_set:
            raise AssertionError(f"retained event references missing doc {doc_id}")
        records.append((qid, user_id, doc_index, int(doc_topics[doc_index])))

    if drift_topic_id is not None:
        drift_doc_indices = np.flatnonzero(
            np.asarray([doc_id in retained_doc_id_set for doc_id in doc_ids])
            & (doc_topics == drift_topic_id)
        ).astype(np.int32, copy=False)
        if drift_doc_indices.size == 0:
            raise ValueError("concept drift requires at least one retained drift-topic doc")
    else:
        drift_doc_indices = np.asarray([], dtype=np.int32)

    drift_start_event_index = (
        int(retained_event_count * drift_start_fraction) if drift_topic_id is not None else None
    )
    drift_end_event_index = (
        int(retained_event_count * drift_end_fraction) if drift_topic_id is not None else None
    )
    global_order = list(range(len(records)))
    if drift_topic_id is not None:
        order_rng = np.random.default_rng(config.random_seed + 5)
        global_order = order_rng.permutation(len(records)).tolist()
    drift_rng = np.random.default_rng(config.random_seed + 6)
    drift_doc_rng = np.random.default_rng(config.random_seed + 7)
    drift_draws = (
        drift_rng.random(len(records)) if drift_topic_id is not None else np.zeros(len(records))
    )
    drift_target_positions = (
        drift_doc_rng.integers(0, int(drift_doc_indices.size), size=len(records))
        if drift_doc_indices.size
        else np.zeros(len(records), dtype=np.int64)
    )

    def build_record(
        *,
        source_index: int,
        event_index: int,
        allow_drift_override: bool,
    ) -> tuple[dict[str, str | int], dict[str, str | int], str, np.ndarray, bool]:
        qid, user_id, base_doc_index, base_topic = records[source_index]
        doc_index = base_doc_index
        generated_topic = base_topic
        query_vector = np.asarray(query_vectors_by_qid[qid], dtype=np.float32)
        is_drift_topic = False
        if (
            allow_drift_override
            and drift_topic_id is not None
            and drift_start_event_index is not None
            and drift_end_event_index is not None
            and drift_start_event_index <= event_index < drift_end_event_index
            and float(drift_draws[source_index]) < drift_probability
        ):
            generated_topic = drift_topic_id
            doc_index = int(drift_doc_indices[int(drift_target_positions[source_index])])
            query_vector = sample_noisy_vectors(
                drift_rng,
                centroids=centroids,
                topic_indices=np.asarray([drift_topic_id], dtype=np.int32),
                sigma=config.sigma,
            )[0]
            is_drift_topic = True
        elif drift_topic_id is not None and generated_topic == drift_topic_id:
            is_drift_topic = True

        doc_id = doc_ids[doc_index]
        query_key = query_key_for_id(qid, width=query_width)
        other_owners = owners_by_doc_index[doc_index] - {user_id}
        searchable = bool(other_owners) or (
            is_drift_topic
            and drift_topic_id is not None
            and drift_start_event_index is not None
            and event_index >= drift_start_event_index
        )
        row = {
            "timestamp": synthetic_timestamp_for_index(event_index),
            "qid": qid,
            "user_id": user_id,
            "query": query_key,
            "target_doc_id": doc_id,
            "candidate_doc_ids": "[]",
        }
        topic_row = {
            "qid": qid,
            "user_id": user_id,
            "query": query_key,
            "generated_topic": generated_topic,
            "target_doc_id": doc_id,
            "target_doc_topic": int(doc_topics[doc_index]),
            "query_target_similarity": f"{float(query_vector @ doc_vectors[doc_index]):.8f}",
            "searchable": "true" if searchable else "false",
            "is_drift_topic": "true" if is_drift_topic else "false",
        }
        return row, topic_row, query_key, query_vector, searchable

    with (
        _open_text_writer(events_path) as events_handle,
        _open_text_writer(randomized_path) as randomized_handle,
        _open_text_writer(searchable_path) as searchable_handle,
        _open_text_writer(query_topics_path) as query_topics_handle,
    ):
        events_writer = csv.DictWriter(events_handle, fieldnames=AMAZON_EVENTS_FIELDNAMES)
        randomized_writer = csv.DictWriter(randomized_handle, fieldnames=AMAZON_EVENTS_FIELDNAMES)
        searchable_writer = csv.DictWriter(
            searchable_handle,
            fieldnames=SEARCHABLE_QUERY_FIELDNAMES,
        )
        query_topics_writer = csv.DictWriter(
            query_topics_handle,
            fieldnames=(
                "qid",
                "user_id",
                "query",
                "generated_topic",
                "target_doc_id",
                "target_doc_topic",
                "query_target_similarity",
                "searchable",
                "is_drift_topic",
            ),
        )
        events_writer.writeheader()
        randomized_writer.writeheader()
        searchable_writer.writeheader()
        query_topics_writer.writeheader()

        for source_index in range(len(records)):
            row, _topic_row, _query_key, _query_vector, _searchable = build_record(
                source_index=source_index,
                event_index=source_index,
                allow_drift_override=False,
            )
            events_writer.writerow(row)

        for event_index, source_index in enumerate(global_order):
            row, topic_row, query_key, query_vector, searchable = build_record(
                source_index=source_index,
                event_index=event_index,
                allow_drift_override=drift_topic_id is not None,
            )
            randomized_writer.writerow(row)
            searchable_writer.writerow(
                {
                    "query_id": row["qid"],
                    "searchable": "true" if searchable else "false",
                }
            )
            query_topics_writer.writerow(topic_row)
            query_keys[write_index] = query_key
            query_embeddings[write_index] = query_vector.astype(np.float16, copy=False)
            searchable_count += int(searchable)
            write_index += 1

    if write_index != retained_event_count:
        raise AssertionError(
            f"wrote {write_index} retained queries, expected {retained_event_count}"
        )
    del query_keys
    del query_embeddings

    return {
        "events_path": str(events_path),
        "randomized_events_path": str(randomized_path),
        "searchable_queries_path": str(searchable_path),
        "query_topics_path": str(query_topics_path),
        "query_keys_path": str(query_keys_path),
        "query_embeddings_path": str(query_embeddings_path),
        "retained_query_count": retained_event_count,
        "searchable_query_count": searchable_count,
        "drift_start_event_index": drift_start_event_index,
        "drift_end_event_index": drift_end_event_index,
    }


def precompute_synthetic_artifacts(
    *,
    data_dir: Path | None = None,
    topic_count: int = DEFAULT_TOPIC_COUNT,
    docs_per_topic: int = DEFAULT_DOCS_PER_TOPIC,
    user_count: int = DEFAULT_USER_COUNT,
    queries_per_user: int = DEFAULT_QUERIES_PER_USER,
    dirichlet_alpha: float = DEFAULT_DIRICHLET_ALPHA,
    embedding_dim: int = DEFAULT_EMBEDDING_DIM,
    sigma: float | None = None,
    random_seed: int = DEFAULT_RANDOM_SEED,
    min_target_count: int = DEFAULT_MIN_TARGET_COUNT,
    run_sweep: bool = True,
    sigma_sweep_values: tuple[float, ...] = DEFAULT_SIGMA_SWEEP,
    calibration_query_count: int = 10_000,
    nearest_batch_size: int = 1024,
    user_batch_size: int = 128,
    top_k: int = DEFAULT_TOP_K,
    target_assignment: str = "nearest",
    doc_counts_by_topic: Sequence[int] | None = None,
    drift_topic_id: int | None = None,
    drift_start_fraction: float = 0.0,
    drift_end_fraction: float = 1.0,
    drift_probability: float = 0.0,
) -> dict[str, object]:
    if data_dir is None:
        data_dir = resolve_dataset_dir("synthetic")
    data_dir = ensure_dataset_dir(data_dir)
    print(f"[synthetic-precompute] data_dir={data_dir}")

    sweep_payload: dict[str, object] | None = None
    if run_sweep:
        if doc_counts_by_topic is not None:
            raise ValueError("sigma sweep currently requires uniform docs_per_topic")
        sweep_payload = run_sigma_sweep(
            data_dir=data_dir,
            topic_count=topic_count,
            docs_per_topic=docs_per_topic,
            embedding_dim=embedding_dim,
            random_seed=random_seed,
            sigma_values=sigma_sweep_values,
            calibration_query_count=calibration_query_count,
            nearest_batch_size=nearest_batch_size,
            top_k=top_k,
        )
        if sigma is None:
            sigma = float(sweep_payload["recommended_sigma"])
            print(f"[synthetic-precompute] selected sigma={sigma:g} from sweep")
    elif sigma is None:
        raise ValueError("sigma must be provided when run_sweep=False")
    if target_assignment not in {"nearest", "capped_topic"}:
        raise ValueError("target_assignment must be either 'nearest' or 'capped_topic'")
    if not 0.0 <= drift_start_fraction <= 1.0:
        raise ValueError("drift_start_fraction must be in [0, 1]")
    if not 0.0 <= drift_end_fraction <= 1.0:
        raise ValueError("drift_end_fraction must be in [0, 1]")
    if drift_end_fraction < drift_start_fraction:
        raise ValueError("drift_end_fraction must be >= drift_start_fraction")
    if not 0.0 <= drift_probability <= 1.0:
        raise ValueError("drift_probability must be in [0, 1]")

    config = SyntheticConfig(
        topic_count=topic_count,
        docs_per_topic=docs_per_topic,
        user_count=user_count,
        queries_per_user=queries_per_user,
        dirichlet_alpha=dirichlet_alpha,
        embedding_dim=embedding_dim,
        sigma=float(sigma),
        random_seed=random_seed,
        min_target_count=min_target_count,
    )
    centroids = _generate_centroids(config)
    np.save(data_dir / SYNTHETIC_TOPICS_FILENAME, centroids.astype(np.float32, copy=False))
    doc_vectors, doc_topics = _generate_doc_vectors(
        config,
        centroids=centroids,
        doc_counts_by_topic=doc_counts_by_topic,
    )
    doc_count = int(doc_vectors.shape[0])
    if doc_count == 0:
        raise ValueError("synthetic dataset must contain at least one document")
    topic_doc_indices = _build_topic_doc_indices(doc_topics, config.topic_count)
    drift_start_offset = int(config.queries_per_user * drift_start_fraction)
    drift_end_offset = int(config.queries_per_user * drift_end_fraction)
    doc_width = fixed_width_for_count(doc_count)
    doc_ids = [doc_id_for_index(index, width=doc_width) for index in range(doc_count)]

    tmp_dir = data_dir / ".synthetic_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    target_path = tmp_dir / "target_doc_indices.npy"
    query_topic_path = tmp_dir / "query_topic_indices.npy"
    query_vector_path = tmp_dir / "query_vectors.npy"
    _unlink_if_exists(target_path)
    _unlink_if_exists(query_topic_path)
    _unlink_if_exists(query_vector_path)
    target_doc_indices = open_memmap(
        target_path,
        mode="w+",
        dtype=np.int32,
        shape=(config.query_count,),
    )
    target_doc_indices[:] = -1
    query_topic_indices = open_memmap(
        query_topic_path,
        mode="w+",
        dtype=np.int32,
        shape=(config.query_count,),
    )
    query_vectors_by_qid = open_memmap(
        query_vector_path,
        mode="w+",
        dtype=np.float16,
        shape=(config.query_count, config.embedding_dim),
    )

    print(
        "[synthetic-precompute] assigning targets for "
        f"{config.query_count:,} queries over {doc_count:,} docs",
        flush=True,
    )
    used_docs_by_user_topic: dict[tuple[int, int], set[int]] = defaultdict(set)
    capped_dropped_query_count = 0
    for qids, user_ids, query_topics, query_vectors in _iter_query_batches(
        config,
        centroids=centroids,
        user_batch_size=user_batch_size,
        drift_topic_id=drift_topic_id,
        drift_start_offset=drift_start_offset,
        drift_end_offset=drift_end_offset,
        drift_probability=0.0 if drift_topic_id is not None else drift_probability,
    ):
        if target_assignment == "capped_topic":
            nearest_indices = _assign_capped_topic_targets(
                query_vectors=query_vectors,
                user_ids=user_ids,
                query_topics=query_topics,
                doc_vectors=doc_vectors,
                topic_doc_indices=topic_doc_indices,
                used_docs_by_user_topic=used_docs_by_user_topic,
            )
            capped_dropped_query_count += int(np.sum(nearest_indices < 0))
        else:
            nearest_indices, _nearest_sims = batched_nearest_indices(
                query_vectors,
                doc_vectors,
                batch_size=nearest_batch_size,
            )
        start = int(qids[0])
        end = int(qids[-1]) + 1
        target_doc_indices[start:end] = nearest_indices
        query_topic_indices[start:end] = query_topics
        query_vectors_by_qid[start:end] = query_vectors.astype(np.float16, copy=False)
        if end % max(1, config.queries_per_user * 1000) == 0 or end == config.query_count:
            print(
                f"[synthetic-precompute] assigned {end:,}/{config.query_count:,} queries",
                flush=True,
            )

    target_index_arr = np.asarray(target_doc_indices, dtype=np.int32)
    valid_target_mask = target_index_arr >= 0
    target_counts = np.bincount(
        target_index_arr[valid_target_mask],
        minlength=doc_count,
    )
    retained_doc_mask = target_counts > config.min_target_count
    if drift_topic_id is not None:
        retained_doc_mask[doc_topics == drift_topic_id] = True
    retained_event_mask = np.zeros(config.query_count, dtype=bool)
    retained_event_mask[valid_target_mask] = retained_doc_mask[target_index_arr[valid_target_mask]]
    retained_doc_count = int(np.sum(retained_doc_mask))
    retained_query_count = int(np.sum(retained_event_mask))
    print(
        "[synthetic-precompute] retained "
        f"{retained_doc_count:,}/{doc_count:,} docs and "
        f"{retained_query_count:,}/{config.query_count:,} query events",
        flush=True,
    )

    doc_artifacts = _write_doc_artifacts(
        data_dir=data_dir,
        doc_vectors=doc_vectors,
        doc_topics=doc_topics,
        target_counts=target_counts,
        retained_doc_mask=retained_doc_mask,
        doc_ids=doc_ids,
    )
    owners_by_doc_index, docs_by_peer = _build_retained_ownership(
        target_doc_indices=target_doc_indices,
        retained_event_mask=retained_event_mask,
        queries_per_user=config.queries_per_user,
        doc_ids=doc_ids,
        user_count=config.user_count,
        random_seed=config.random_seed,
        doc_topics=doc_topics,
        drift_topic_id=drift_topic_id,
    )
    peer_summary = _write_peer_future_docs(
        docs_by_peer=docs_by_peer,
        output_path=data_dir / SYNTHETIC_PEER_FUTURE_DOCS_FILENAME,
        summary_path=data_dir / SYNTHETIC_PEER_FUTURE_DOCS_SUMMARY_FILENAME,
    )
    query_artifacts = _write_retained_query_artifacts(
        config=config,
        data_dir=data_dir,
        centroids=centroids,
        doc_vectors=doc_vectors,
        doc_topics=doc_topics,
        target_doc_indices=target_doc_indices,
        query_vectors_by_qid=query_vectors_by_qid,
        retained_event_mask=retained_event_mask,
        owners_by_doc_index=owners_by_doc_index,
        doc_ids=doc_ids,
        retained_doc_ids=doc_artifacts["retained_doc_ids"],
        user_batch_size=user_batch_size,
        drift_topic_id=drift_topic_id,
        drift_start_fraction=drift_start_fraction,
        drift_end_fraction=drift_end_fraction,
        drift_probability=drift_probability,
    )
    retained_peer_ids = sorted(
        {int(qid) // config.queries_per_user for qid in np.flatnonzero(retained_event_mask)}
    )
    retained_doc_id_set = set(doc_artifacts["retained_doc_ids"])
    drift_doc_indices = np.flatnonzero(
        np.asarray([doc_id in retained_doc_id_set for doc_id in doc_ids])
        & (doc_topics == drift_topic_id)
    ).astype(np.int32, copy=False) if drift_topic_id is not None else np.asarray([], dtype=np.int32)
    drift_owner_injections_summary = None
    if drift_topic_id is not None:
        drift_owner_injections_summary = _write_drift_owner_injections(
            output_path=data_dir / SYNTHETIC_DRIFT_OWNER_INJECTIONS_FILENAME,
            drift_doc_indices=drift_doc_indices,
            doc_ids=doc_ids,
            owner_pool=retained_peer_ids,
            event_index=int(query_artifacts["drift_start_event_index"]),
            random_seed=config.random_seed,
        )

    manifest = {
        "dataset": "synthetic",
        "metadata_path": str(data_dir / SYNTHETIC_EVENTS_FILENAME),
        "searchable_queries_path": str(data_dir / SYNTHETIC_SEEDED_SEARCHABLE_FILENAME),
        "output_dir": str(data_dir / "embeddings"),
        "model": "synthetic-centroid-gaussian",
        "embedding_dim": config.embedding_dim,
        "embedding_dtype": "float16",
        "topic_count": config.topic_count,
        "docs_per_topic": config.docs_per_topic,
        "doc_counts_by_topic": (
            [int(value) for value in doc_counts_by_topic] if doc_counts_by_topic is not None else None
        ),
        "user_count": config.user_count,
        "queries_per_user": config.queries_per_user,
        "dirichlet_alpha": config.dirichlet_alpha,
        "sigma": config.sigma,
        "random_seed": config.random_seed,
        "target_assignment": target_assignment,
        "drift_topic_id": drift_topic_id,
        "drift_start_fraction": drift_start_fraction,
        "drift_end_fraction": drift_end_fraction,
        "drift_probability": drift_probability,
        "drift_start_event_index": query_artifacts["drift_start_event_index"],
        "drift_end_event_index": query_artifacts["drift_end_event_index"],
        "drift_owner_injections": (
            drift_owner_injections_summary["output_path"]
            if drift_owner_injections_summary is not None
            else None
        ),
        "query_keys_path": query_artifacts["query_keys_path"],
        "query_embeddings_path": query_artifacts["query_embeddings_path"],
        "query_count": query_artifacts["retained_query_count"],
        "doc_keys_path": doc_artifacts["doc_keys_path"],
        "doc_embeddings_path": doc_artifacts["doc_embeddings_path"],
        "doc_count": len(doc_artifacts["retained_doc_ids"]),
    }
    manifest_path = data_dir / "embeddings" / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    summary = {
        "dataset": "synthetic",
        "data_dir": str(data_dir),
        "topic_count": config.topic_count,
        "docs_per_topic": config.docs_per_topic,
        "doc_counts_by_topic": (
            [int(value) for value in doc_counts_by_topic] if doc_counts_by_topic is not None else None
        ),
        "doc_count_before_filter": doc_count,
        "retained_doc_count": retained_doc_count,
        "dropped_doc_count": doc_count - retained_doc_count,
        "user_count": config.user_count,
        "queries_per_user": config.queries_per_user,
        "query_count_before_filter": config.query_count,
        "retained_query_count": retained_query_count,
        "dropped_query_count": config.query_count - retained_query_count,
        "searchable_query_count": query_artifacts["searchable_query_count"],
        "dirichlet_alpha": config.dirichlet_alpha,
        "embedding_dim": config.embedding_dim,
        "sigma": config.sigma,
        "random_seed": config.random_seed,
        "min_target_count": config.min_target_count,
        "target_assignment": target_assignment,
        "capped_topic_dropped_query_count": capped_dropped_query_count,
        "drift_topic_id": drift_topic_id,
        "drift_start_fraction": drift_start_fraction,
        "drift_end_fraction": drift_end_fraction,
        "drift_probability": drift_probability,
        "drift_start_event_index": query_artifacts["drift_start_event_index"],
        "drift_end_event_index": query_artifacts["drift_end_event_index"],
        "drift_owner_injections_summary": drift_owner_injections_summary,
        "target_count_summary": percentile_summary(target_counts),
        "sigma_sweep_path": str(data_dir / SYNTHETIC_SIGMA_SWEEP_FILENAME) if sweep_payload else None,
        "recommended_sigma": sweep_payload.get("recommended_sigma") if sweep_payload else None,
        "manifest_path": str(manifest_path),
        "peer_future_docs_summary": peer_summary,
        "events_path": query_artifacts["events_path"],
        "randomized_events_path": query_artifacts["randomized_events_path"],
        "searchable_queries_path": query_artifacts["searchable_queries_path"],
        "doc_topics_path": doc_artifacts["doc_topics_path"],
        "query_topics_path": query_artifacts["query_topics_path"],
        "topics_path": str(data_dir / SYNTHETIC_TOPICS_FILENAME),
    }
    summary_path = data_dir / SYNTHETIC_PRECOMPUTE_SUMMARY_FILENAME
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[synthetic-precompute] wrote manifest to {manifest_path}")
    print(f"[synthetic-precompute] wrote summary to {summary_path}")

    del target_doc_indices
    del query_topic_indices
    del query_vectors_by_qid
    _unlink_if_exists(target_path)
    _unlink_if_exists(query_topic_path)
    _unlink_if_exists(query_vector_path)
    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a synthetic embedding-space dataset")
    parser.add_argument("--output-dir", default=None, help="Dataset directory; defaults via data-root resolver")
    parser.add_argument("--topics", type=int, default=DEFAULT_TOPIC_COUNT)
    parser.add_argument("--docs-per-topic", type=int, default=DEFAULT_DOCS_PER_TOPIC)
    parser.add_argument("--users", type=int, default=DEFAULT_USER_COUNT)
    parser.add_argument("--queries-per-user", type=int, default=DEFAULT_QUERIES_PER_USER)
    parser.add_argument("--dirichlet-alpha", type=float, default=DEFAULT_DIRICHLET_ALPHA)
    parser.add_argument("--embedding-dim", type=int, default=DEFAULT_EMBEDDING_DIM)
    parser.add_argument("--sigma", type=float, default=None, help="Gaussian stddev; defaults to sigma sweep recommendation")
    parser.add_argument("--random-seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument(
        "--min-target-count",
        type=int,
        default=DEFAULT_MIN_TARGET_COUNT,
        help="Drop docs targeted at most this many times, and drop their events",
    )
    parser.add_argument(
        "--sigma-sweep",
        default=",".join(str(value) for value in DEFAULT_SIGMA_SWEEP),
        help="Comma-separated candidate sigmas for overlap calibration",
    )
    parser.add_argument("--skip-sigma-sweep", action="store_true")
    parser.add_argument("--calibration-query-count", type=int, default=10_000)
    parser.add_argument("--nearest-batch-size", type=int, default=1024)
    parser.add_argument("--user-batch-size", type=int, default=128)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument(
        "--target-assignment",
        choices=("nearest", "capped_topic"),
        default="nearest",
        help=(
            "nearest keeps the original global nearest-doc target rule; capped_topic "
            "chooses same-topic docs without replacement per user/topic and drops "
            "overflow queries"
        ),
    )
    parser.add_argument(
        "--doc-counts-by-topic",
        default=None,
        help="Optional comma-separated per-topic doc counts; disables sigma sweep",
    )
    parser.add_argument("--drift-topic-id", type=int, default=None)
    parser.add_argument("--drift-start-fraction", type=float, default=0.0)
    parser.add_argument("--drift-end-fraction", type=float, default=1.0)
    parser.add_argument("--drift-probability", type=float, default=0.0)
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    if args.skip_sigma_sweep and args.sigma is None:
        parser.error("--sigma is required when --skip-sigma-sweep is set")
    doc_counts_by_topic = None
    if args.doc_counts_by_topic:
        doc_counts_by_topic = [
            int(value.strip())
            for value in args.doc_counts_by_topic.split(",")
            if value.strip()
        ]
        if not args.skip_sigma_sweep:
            parser.error("--doc-counts-by-topic requires --skip-sigma-sweep")
    data_dir = Path(args.output_dir) if args.output_dir else resolve_dataset_dir("synthetic")
    precompute_synthetic_artifacts(
        data_dir=data_dir,
        topic_count=args.topics,
        docs_per_topic=args.docs_per_topic,
        user_count=args.users,
        queries_per_user=args.queries_per_user,
        dirichlet_alpha=args.dirichlet_alpha,
        embedding_dim=args.embedding_dim,
        sigma=args.sigma,
        random_seed=args.random_seed,
        min_target_count=args.min_target_count,
        run_sweep=not args.skip_sigma_sweep,
        sigma_sweep_values=parse_sigma_values(args.sigma_sweep),
        calibration_query_count=args.calibration_query_count,
        nearest_batch_size=args.nearest_batch_size,
        user_batch_size=args.user_batch_size,
        top_k=args.top_k,
        target_assignment=args.target_assignment,
        doc_counts_by_topic=doc_counts_by_topic,
        drift_topic_id=args.drift_topic_id,
        drift_start_fraction=args.drift_start_fraction,
        drift_end_fraction=args.drift_end_fraction,
        drift_probability=args.drift_probability,
    )


if __name__ == "__main__":
    main()

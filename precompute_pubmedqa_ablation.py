#!/usr/bin/env python3
"""Precompute PubMedQA expanded all-MeSH ablation traces.

This script builds simulator-facing PubMedQA variants without recopying the
full document embedding matrix. Variant manifests point at the existing expanded
all-MeSH document keys/embeddings and write only per-trace query embeddings.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent
SCRIPT_DIR = REPO_ROOT / "scripts"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from precompute_pubmedqa_expanded_all_mesh import (  # noqa: E402
    DEFAULT_REPHRASED_QUERIES,
    _load_rephrase_embedding_lookup,
    _load_rephrased_rows,
    _rephrase_embedding_paths,
)
from precompute_pubmedqa_original_all_mesh import (  # noqa: E402
    DEFAULT_HF_ROOT,
    DEFAULT_SOURCE_FULL_DIR,
    _build_topic_pools,
    _embedding_paths,
    _load_embedding_array,
    _load_full_rows,
    _open_text_writer,
    _query_key,
    _write_json,
    _write_peer_future_docs,
)
from src.dataset import AMAZON_EVENTS_FIELDNAMES, SEARCHABLE_QUERY_FIELDNAMES, synthetic_timestamp_for_index  # noqa: E402
from src.embedder import DEFAULT_EMBEDDING_MODEL  # noqa: E402
from src.pubmedqa_dataset import (  # noqa: E402
    DEFAULT_PUBMEDQA_RANDOM_SEED,
    PUBMEDQA_ANSWER_TARGETS_FILENAME,
    PUBMEDQA_DOC_TOPICS_FILENAME,
    PUBMEDQA_EVENTS_FILENAME,
    PUBMEDQA_PEER_FUTURE_DOCS_FILENAME,
    PUBMEDQA_PRECOMPUTE_SUMMARY_FILENAME,
    PUBMEDQA_QUERY_TOPICS_FILENAME,
    PUBMEDQA_RANDOMIZED_EVENTS_FILENAME,
    PUBMEDQA_SEEDED_SEARCHABLE_FILENAME,
    PUBMEDQA_TOPIC_MAP_FILENAME,
)


DEFAULT_BASE_DATA_ROOT = Path(
    os.environ.get(
        "LOW_COORD_DHT_PUBMEDQA_BASE_DATA_ROOT",
        "/var/scratch/pmneague/Low_Coord_Semantic_DHT/data/pubmedqa_expanded_all_mesh_alpha0p00007_q30",
    )
)
DEFAULT_OUTPUT_ROOT = Path(
    os.environ.get(
        "LOW_COORD_DHT_PUBMEDQA_ABLATION_DATA_ROOT",
        str(DEFAULT_BASE_DATA_ROOT / "ablations"),
    )
)
DEFAULT_BASE_DATASET_DIR = DEFAULT_BASE_DATA_ROOT / "pubmedqa"
USER_SCALABILITY_USERS = (1_000, 5_000, 10_000, 50_000, 100_000)
DOC_SCALABILITY_DOCS_PER_PEER = (10, 20, 30, 40, 50)
BASE_PEERS = int(os.environ.get("LOW_COORD_DHT_PUBMEDQA_PEERS", "50000"))
BASE_DOCS_PER_PEER = 30
DEFAULT_ALPHA = 0.00007
CHURN_SCHEDULE_FILENAME = "churn_schedule.json.gz"
DRIFT_OWNER_INJECTIONS_FILENAME = "drift_owner_injections.json.gz"
CONCEPT_DRIFT_LABEL = os.environ.get(
    "LOW_COORD_DHT_PUBMEDQA_CONCEPT_DRIFT_LABEL",
    "auto_mesh_window40_60",
)
DRIFT_START_FRACTION = float(os.environ.get("LOW_COORD_DHT_DRIFT_START_FRACTION", "0.40"))
DRIFT_END_FRACTION = float(os.environ.get("LOW_COORD_DHT_DRIFT_END_FRACTION", "0.60"))
DRIFT_WINDOW_QUERY_FRACTION = 0.30
DRIFT_TARGET_DOC_COUNT = 10
CONCEPT_DRIFT_MESH = os.environ.get(
    "LOW_COORD_DHT_PUBMEDQA_CONCEPT_DRIFT_MESH",
    "Peptostreptococcus",
)
CONCEPT_DRIFT_STREAM_MODE = os.environ.get(
    "LOW_COORD_DHT_PUBMEDQA_CONCEPT_DRIFT_STREAM_MODE",
    "reshuffled_window",
)
DRIFT_INJECTION_REPLICAS_PER_DOC = int(
    os.environ.get("LOW_COORD_DHT_DRIFT_INJECTION_REPLICAS_PER_DOC", "1")
)


@dataclass(frozen=True, slots=True)
class PubmedQaAblationSetting:
    experiment: str
    label: str
    peers: int = BASE_PEERS
    docs_per_peer: int = BASE_DOCS_PER_PEER
    replacement_fraction: float = 0.0
    drift_start_fraction: float = DRIFT_START_FRACTION
    drift_end_fraction: float = DRIFT_END_FRACTION


@dataclass(frozen=True, slots=True)
class QuerySpec:
    peer_id: int
    doc_index: int
    topic_id: int


def iter_settings(experiment: str) -> Iterable[PubmedQaAblationSetting]:
    if experiment == "base":
        yield PubmedQaAblationSetting(experiment=experiment, label="base")
    elif experiment == "user_scalability":
        for peers in USER_SCALABILITY_USERS:
            yield PubmedQaAblationSetting(
                experiment=experiment,
                label=f"users{peers}",
                peers=peers,
            )
    elif experiment == "doc_scalability":
        for docs_per_peer in DOC_SCALABILITY_DOCS_PER_PEER:
            yield PubmedQaAblationSetting(
                experiment=experiment,
                label=f"docs_per_peer{docs_per_peer}",
                docs_per_peer=docs_per_peer,
            )
    elif experiment == "churn":
        yield PubmedQaAblationSetting(
            experiment=experiment,
            label="drop50_gradual_replace50",
            replacement_fraction=0.5,
        )
    elif experiment == "concept_drift":
        yield PubmedQaAblationSetting(
            experiment=experiment,
            label=CONCEPT_DRIFT_LABEL,
        )
    else:
        raise ValueError(f"unknown PubMedQA ablation experiment {experiment!r}")


def setting_by_index(experiment: str, index: int) -> PubmedQaAblationSetting:
    settings = list(iter_settings(experiment))
    if index < 0 or index >= len(settings):
        raise SystemExit(
            f"index {index} out of range for {experiment}; expected 0..{len(settings) - 1}"
        )
    return settings[index]


def data_dir_for_setting(output_root: Path, setting: PubmedQaAblationSetting) -> Path:
    return output_root / setting.experiment / setting.label / "pubmedqa"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _draw_peer_topics(
    *,
    topic_count: int,
    draws: int,
    alpha: float,
    rng: np.random.Generator,
) -> list[int]:
    peer_topic_counts: Counter[int] = Counter()
    drawn_topics: list[int] = []
    total_alpha = topic_count * alpha
    for draw_index in range(draws):
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
    return drawn_topics


def _choose_unique_doc(
    *,
    preferred_pool: list[int],
    global_pool: np.ndarray,
    used_doc_indices: set[int],
    rng: np.random.Generator,
) -> int:
    if len(used_doc_indices) < len(preferred_pool):
        for _attempt in range(max(8, len(preferred_pool) * 2)):
            candidate = int(preferred_pool[int(rng.integers(0, len(preferred_pool)))])
            if candidate not in used_doc_indices:
                return candidate
        remaining = [doc_index for doc_index in preferred_pool if doc_index not in used_doc_indices]
        if remaining:
            return int(remaining[int(rng.integers(0, len(remaining)))])

    if len(used_doc_indices) >= int(global_pool.shape[0]):
        raise ValueError("Cannot draw another unique document for this peer")
    for _attempt in range(max(32, int(global_pool.shape[0] // 8))):
        candidate = int(global_pool[int(rng.integers(0, global_pool.shape[0]))])
        if candidate not in used_doc_indices:
            return candidate
    remaining_global = [int(doc_index) for doc_index in global_pool.tolist() if int(doc_index) not in used_doc_indices]
    return remaining_global[int(rng.integers(0, len(remaining_global)))]


def _sample_exact_inventories(
    *,
    peer_count: int,
    docs_per_peer: int,
    topic_count: int,
    doc_indices_by_topic: list[list[int]],
    alpha: float,
    random_seed: int,
    peer_id_offset: int = 0,
) -> tuple[dict[int, list[int]], dict[int, list[int]], dict[str, Any]]:
    if peer_count <= 0:
        raise ValueError("peer_count must be positive")
    if docs_per_peer <= 0:
        raise ValueError("docs_per_peer must be positive")
    global_pool = np.asarray(
        sorted({doc_index for pool in doc_indices_by_topic for doc_index in pool}),
        dtype=np.int32,
    )
    if docs_per_peer > int(global_pool.shape[0]):
        raise ValueError("docs_per_peer exceeds the PubMedQA document pool")

    rng = np.random.default_rng(random_seed)
    docs_by_peer: dict[int, list[int]] = {}
    topics_by_peer: dict[int, list[int]] = {}
    unique_topics_per_peer: list[int] = []
    topic_counts: Counter[int] = Counter()
    fallback_doc_draws = 0

    for local_peer_id in range(peer_count):
        peer_id = peer_id_offset + local_peer_id
        topic_ids = _draw_peer_topics(
            topic_count=topic_count,
            draws=docs_per_peer,
            alpha=alpha,
            rng=rng,
        )
        used_doc_indices: set[int] = set()
        doc_indices: list[int] = []
        for topic_id in topic_ids:
            before = len(used_doc_indices)
            doc_index = _choose_unique_doc(
                preferred_pool=doc_indices_by_topic[topic_id],
                global_pool=global_pool,
                used_doc_indices=used_doc_indices,
                rng=rng,
            )
            fallback_doc_draws += int(doc_index not in doc_indices_by_topic[topic_id])
            used_doc_indices.add(doc_index)
            if len(used_doc_indices) == before:
                raise RuntimeError("unique document draw did not advance")
            doc_indices.append(doc_index)
            topic_counts[topic_id] += 1
        docs_by_peer[peer_id] = doc_indices
        topics_by_peer[peer_id] = topic_ids
        unique_topics_per_peer.append(len(set(topic_ids)))

    summary = {
        "avg_unique_topics_per_peer": float(np.mean(unique_topics_per_peer)) if unique_topics_per_peer else 0.0,
        "min_unique_topics_per_peer": int(min(unique_topics_per_peer)) if unique_topics_per_peer else 0,
        "max_unique_topics_per_peer": int(max(unique_topics_per_peer)) if unique_topics_per_peer else 0,
        "fallback_doc_draws": int(fallback_doc_draws),
        "top_sampled_topic_ids": [
            {"topic_id": int(topic_id), "query_count": int(count)}
            for topic_id, count in topic_counts.most_common(20)
        ],
    }
    return docs_by_peer, topics_by_peer, summary


def _query_specs_for_inventories(
    docs_by_peer: dict[int, list[int]],
    topics_by_peer: dict[int, list[int]],
) -> list[QuerySpec]:
    specs: list[QuerySpec] = []
    for peer_id in sorted(docs_by_peer):
        for doc_index, topic_id in zip(docs_by_peer[peer_id], topics_by_peer[peer_id], strict=True):
            specs.append(QuerySpec(peer_id=peer_id, doc_index=doc_index, topic_id=topic_id))
    return specs


def _build_churn_specs_and_schedule(
    *,
    initial_specs: list[QuerySpec],
    initial_docs_by_peer: dict[int, list[int]],
    initial_topics_by_peer: dict[int, list[int]],
    replacement_docs_by_peer: dict[int, list[int]],
    replacement_topics_by_peer: dict[int, list[int]],
    rows,
    churn_fraction: float,
    random_seed: int,
) -> tuple[list[QuerySpec], dict[str, Any]]:
    initial_peer_ids = sorted(initial_docs_by_peer)
    drop_count = int(round(len(initial_peer_ids) * churn_fraction))
    if drop_count <= 0:
        raise ValueError("churn_fraction produced no dropped peers")

    rng = np.random.default_rng(random_seed)
    dropped_peer_ids = [int(peer_id) for peer_id in rng.choice(initial_peer_ids, size=drop_count, replace=False).tolist()]
    dropped_set = set(dropped_peer_ids)
    survivor_ids = [peer_id for peer_id in initial_peer_ids if peer_id not in dropped_set]

    first_half_count = len(initial_specs) // 2
    first_indices = rng.choice(len(initial_specs), size=first_half_count, replace=False)
    first_specs = [initial_specs[int(index)] for index in first_indices.tolist()]

    survivor_specs = _query_specs_for_inventories(
        {peer_id: initial_docs_by_peer[peer_id] for peer_id in survivor_ids},
        {peer_id: initial_topics_by_peer[peer_id] for peer_id in survivor_ids},
    )
    replacement_specs = _query_specs_for_inventories(replacement_docs_by_peer, replacement_topics_by_peer)
    second_pool = survivor_specs + replacement_specs
    if len(second_pool) < first_half_count:
        raise ValueError("Not enough survivor/replacement events to fill post-churn half")
    second_indices = rng.choice(len(second_pool), size=first_half_count, replace=False)
    second_specs = [second_pool[int(index)] for index in second_indices.tolist()]

    replacement_peer_ids = sorted(replacement_docs_by_peer)
    activation_span = max(1, first_half_count)
    first_replacement_event: dict[int, int] = {}
    for offset, spec in enumerate(second_specs):
        if spec.peer_id in replacement_docs_by_peer and spec.peer_id not in first_replacement_event:
            first_replacement_event[spec.peer_id] = first_half_count + offset
    replacement_activations = []
    for offset, replacement_peer_id in enumerate(replacement_peer_ids):
        linear_event_index = first_half_count + int(((offset + 1) * activation_span) / len(replacement_peer_ids)) - 1
        event_index = first_replacement_event.get(replacement_peer_id, linear_event_index)
        replacement_activations.append(
            {
                "event_index": int(min(first_half_count * 2 - 1, max(first_half_count, event_index))),
                "dropped_peer_id": int(dropped_peer_ids[offset]),
                "replacement_peer_id": int(replacement_peer_id),
                "doc_ids": [rows[doc_index].pubid for doc_index in replacement_docs_by_peer[replacement_peer_id]],
            }
        )

    schedule = {
        "mode": "sudden_drop_gradual_replace",
        "drop_event_index": int(first_half_count),
        "drop_fraction": churn_fraction,
        "dropped_peer_ids": dropped_peer_ids,
        "replacement_activations": replacement_activations,
        "initial_peer_count": len(initial_peer_ids),
        "replacement_peer_count": len(replacement_peer_ids),
        "event_count": int(first_half_count * 2),
        "notes": {
            "searchability": "Runtime searchability must use active non-querier owners only.",
            "drop": "All dropped peers deactivate at drop_event_index before that event is processed.",
            "replacement": "Replacement peers activate linearly over the post-drop half of the trace.",
        },
    }
    return first_specs + second_specs, schedule


def _doc_indices_containing_mesh(
    rows: list[Any],
    *,
    mesh: str,
    used_doc_indices: set[int],
) -> set[int]:
    return {
        int(doc_index)
        for doc_index in used_doc_indices
        if mesh in getattr(rows[int(doc_index)], "meshes", ())
    }


def _select_concept_drift_topic(
    *,
    rows: list[Any],
    topic_terms: list[str],
    query_specs: list[QuerySpec],
    target_unique_doc_count: int,
    preferred_mesh: str | None = None,
) -> dict[str, Any]:
    used_doc_indices = {int(spec.doc_index) for spec in query_specs}
    topic_id_by_mesh = {mesh: topic_id for topic_id, mesh in enumerate(topic_terms)}
    sampled_counts: Counter[int] = Counter()
    event_counts: Counter[int] = Counter()
    unique_doc_counts: Counter[int] = Counter()
    for spec in query_specs:
        sampled_counts[int(spec.topic_id)] += 1
        for mesh in getattr(rows[int(spec.doc_index)], "meshes", ()):
            topic_id = topic_id_by_mesh[mesh]
            event_counts[topic_id] += 1
    for doc_index in used_doc_indices:
        for mesh in getattr(rows[int(doc_index)], "meshes", ()):
            unique_doc_counts[topic_id_by_mesh[mesh]] += 1

    candidates: list[dict[str, Any]] = []
    for topic_id, mesh in enumerate(topic_terms):
        event_count = int(event_counts.get(topic_id, 0))
        unique_doc_count = int(unique_doc_counts.get(topic_id, 0))
        if event_count <= 0 or unique_doc_count <= 0:
            continue
        candidates.append(
            {
                "topic_id": int(topic_id),
                "mesh_term": mesh,
                "sampled_topic_count": int(sampled_counts.get(topic_id, 0)),
                "heldout_event_count": event_count,
                "heldout_unique_doc_count": unique_doc_count,
            }
        )
    if not candidates:
        raise ValueError("No sampled MeSH topic is available for concept drift")
    if preferred_mesh:
        for candidate in candidates:
            if str(candidate["mesh_term"]) == preferred_mesh:
                return candidate
        raise ValueError(f"Requested concept-drift MeSH {preferred_mesh!r} is not present in sampled docs")

    fitting = [
        candidate
        for candidate in candidates
        if int(candidate["heldout_unique_doc_count"]) >= target_unique_doc_count
    ]
    pool = fitting or candidates
    return max(
        pool,
        key=lambda candidate: (
            -abs(int(candidate["heldout_unique_doc_count"]) - target_unique_doc_count),
            int(candidate["heldout_event_count"]),
            int(candidate["sampled_topic_count"]),
            -int(candidate["topic_id"]),
        ),
    )


def _build_concept_drift_specs_and_injections(
    *,
    initial_specs: list[QuerySpec],
    docs_by_peer: dict[int, list[int]],
    topics_by_peer: dict[int, list[int]],
    rows: list[Any],
    topic_terms: list[str],
    drift_start_fraction: float,
    drift_end_fraction: float,
    random_seed: int,
    event_order_seed: int | None = None,
    stream_mode: str = "reshuffled_window",
) -> tuple[list[QuerySpec], dict[int, list[int]], dict[int, list[int]], list[dict[str, int | str]], dict[str, Any]]:
    if not 0.0 <= drift_start_fraction <= 1.0:
        raise ValueError("drift_start_fraction must be in [0, 1]")
    if not 0.0 <= drift_end_fraction <= 1.0:
        raise ValueError("drift_end_fraction must be in [0, 1]")
    if drift_end_fraction <= drift_start_fraction:
        raise ValueError("drift_end_fraction must be greater than drift_start_fraction")

    event_count = len(initial_specs)
    drift_start_event_index = int(event_count * drift_start_fraction)
    drift_end_event_index = int(event_count * drift_end_fraction)
    drift_window_slots = drift_end_event_index - drift_start_event_index
    if drift_window_slots <= 0:
        raise ValueError("concept drift window contains no events")
    target_drift_query_count = int(round(drift_window_slots * DRIFT_WINDOW_QUERY_FRACTION))
    if target_drift_query_count <= 0:
        raise ValueError("concept drift query fraction produced no target-topic queries")

    target = _select_concept_drift_topic(
        rows=rows,
        topic_terms=topic_terms,
        query_specs=initial_specs,
        target_unique_doc_count=DRIFT_TARGET_DOC_COUNT,
        preferred_mesh=CONCEPT_DRIFT_MESH,
    )
    target_topic_id = int(target["topic_id"])
    target_mesh = str(target["mesh_term"])
    used_doc_indices = {int(spec.doc_index) for spec in initial_specs}
    heldout_doc_indices = _doc_indices_containing_mesh(
        rows,
        mesh=target_mesh,
        used_doc_indices=used_doc_indices,
    )
    if not heldout_doc_indices:
        raise ValueError(f"Selected drift MeSH {target_mesh!r} has no sampled docs")

    filtered_docs_by_peer: dict[int, list[int]] = {}
    filtered_topics_by_peer: dict[int, list[int]] = {}
    removed_owners_by_doc: dict[int, set[int]] = defaultdict(set)
    for peer_id in sorted(docs_by_peer):
        kept_docs: list[int] = []
        kept_topics: list[int] = []
        for doc_index, topic_id in zip(docs_by_peer[peer_id], topics_by_peer[peer_id], strict=True):
            doc_index = int(doc_index)
            if doc_index in heldout_doc_indices:
                removed_owners_by_doc[doc_index].add(int(peer_id))
                continue
            kept_docs.append(doc_index)
            kept_topics.append(int(topic_id))
        filtered_docs_by_peer[int(peer_id)] = kept_docs
        filtered_topics_by_peer[int(peer_id)] = kept_topics

    non_drift_specs = [
        spec for spec in initial_specs if int(spec.doc_index) not in heldout_doc_indices
    ]
    drift_specs = [
        QuerySpec(
            peer_id=int(spec.peer_id),
            doc_index=int(spec.doc_index),
            topic_id=target_topic_id,
        )
        for spec in initial_specs
        if int(spec.doc_index) in heldout_doc_indices
    ]
    if not drift_specs:
        raise ValueError(f"Selected drift MeSH {target_mesh!r} has no sampled query specs")

    rng = np.random.default_rng(random_seed)
    drift_positions: list[int] = []
    repeated_drift_query_fill_count = 0
    duplicated_non_drift_fill_count = 0
    existing_drift_positions: list[int] = []
    if stream_mode == "base_event_stream":
        order_seed = random_seed if event_order_seed is None else event_order_seed
        permutation = np.random.default_rng(order_seed + 4).permutation(len(initial_specs))
        query_specs = [initial_specs[int(index)] for index in permutation.tolist()]
        existing_drift_positions = [
            event_index
            for event_index in range(drift_start_event_index, drift_end_event_index)
            if int(query_specs[event_index].doc_index) in heldout_doc_indices
        ]
        if len(existing_drift_positions) > target_drift_query_count:
            raise ValueError(
                "Base event stream already has more post-start drift-topic queries "
                f"({len(existing_drift_positions)}) than the requested target ({target_drift_query_count})"
            )
        remaining_drift_count = target_drift_query_count - len(existing_drift_positions)
        candidate_positions = np.asarray(
            [
                event_index
                for event_index in range(drift_start_event_index, drift_end_event_index)
                if int(query_specs[event_index].doc_index) not in heldout_doc_indices
            ],
            dtype=np.int64,
        )
        if remaining_drift_count > int(candidate_positions.shape[0]):
            raise ValueError("Not enough post-start non-drift events to build requested drift fraction")
        sampled_positions = (
            rng.choice(candidate_positions, size=remaining_drift_count, replace=False).tolist()
            if remaining_drift_count
            else []
        )
        drift_positions = sorted(int(position) for position in existing_drift_positions + sampled_positions)
        heldout_doc_array = np.asarray(sorted(heldout_doc_indices), dtype=np.int32)
        assigned_doc_indices = rng.choice(
            heldout_doc_array,
            size=len(drift_positions),
            replace=True,
        ).tolist()
        repeated_drift_query_fill_count = max(0, len(drift_positions) - len(set(int(index) for index in assigned_doc_indices)))
        for event_index, doc_index in zip(drift_positions, assigned_doc_indices, strict=True):
            original_spec = query_specs[int(event_index)]
            query_specs[int(event_index)] = QuerySpec(
                peer_id=int(original_spec.peer_id),
                doc_index=int(doc_index),
                topic_id=target_topic_id,
            )
    elif stream_mode == "reshuffled_window":
        non_drift_order = rng.permutation(len(non_drift_specs)).tolist()
        shuffled_non_drift = [non_drift_specs[int(index)] for index in non_drift_order]
        if len(drift_specs) >= target_drift_query_count:
            drift_indices = rng.permutation(len(drift_specs)).tolist()[:target_drift_query_count]
            repeated_drift_query_fill_count = 0
        else:
            drift_indices = rng.choice(
                len(drift_specs),
                size=target_drift_query_count,
                replace=True,
            ).tolist()
            repeated_drift_query_fill_count = target_drift_query_count - len(drift_specs)
        shuffled_drift = [drift_specs[int(index)] for index in drift_indices]
        required_non_drift_count = event_count - len(shuffled_drift)
        duplicated_non_drift_fill_count = max(0, required_non_drift_count - len(shuffled_non_drift))
        if duplicated_non_drift_fill_count:
            fill_indices = rng.choice(
                len(shuffled_non_drift),
                size=duplicated_non_drift_fill_count,
                replace=True,
            )
            shuffled_non_drift.extend(shuffled_non_drift[int(index)] for index in fill_indices.tolist())
        shuffled_non_drift = shuffled_non_drift[:required_non_drift_count]

        pre_specs = shuffled_non_drift[:drift_start_event_index]
        cursor = drift_start_event_index
        window_non_drift_count = drift_window_slots - len(shuffled_drift)
        window_specs = shuffled_drift + shuffled_non_drift[cursor : cursor + window_non_drift_count]
        rng.shuffle(window_specs)
        cursor += window_non_drift_count
        post_specs = shuffled_non_drift[cursor:]
        query_specs = pre_specs + window_specs + post_specs
        drift_positions = [
            event_index
            for event_index, spec in enumerate(query_specs)
            if (
                drift_start_event_index <= event_index < drift_end_event_index
                and int(spec.doc_index) in heldout_doc_indices
            )
        ]
    else:
        raise ValueError(
            "LOW_COORD_DHT_PUBMEDQA_CONCEPT_DRIFT_STREAM_MODE must be "
            "'reshuffled_window' or 'base_event_stream'"
        )
    if len(query_specs) != event_count:
        raise AssertionError(f"concept drift trace length changed from {event_count} to {len(query_specs)}")

    owner_pool = sorted(filtered_docs_by_peer)
    injections: list[dict[str, int | str]] = []
    for doc_index in sorted(heldout_doc_indices):
        removed_owners = removed_owners_by_doc.get(doc_index, set())
        candidates = [
            peer_id
            for peer_id in owner_pool
            if peer_id not in removed_owners
        ]
        if not candidates:
            candidates = owner_pool
        if len(candidates) >= DRIFT_INJECTION_REPLICAS_PER_DOC:
            selected_peers = rng.choice(
                candidates,
                size=DRIFT_INJECTION_REPLICAS_PER_DOC,
                replace=False,
            ).tolist()
        else:
            selected_peers = rng.choice(
                candidates,
                size=DRIFT_INJECTION_REPLICAS_PER_DOC,
                replace=True,
            ).tolist()
        for peer_id in selected_peers:
            injections.append(
                {
                    "event_index": drift_start_event_index,
                    "peer_id": int(peer_id),
                    "doc_id": str(rows[doc_index].pubid),
                }
            )

    drift_query_count = sum(
        drift_start_event_index <= event_index < drift_end_event_index
        and int(spec.doc_index) in heldout_doc_indices
        for event_index, spec in enumerate(query_specs)
    )
    drift_positions_array = np.asarray(drift_positions, dtype=np.int64)
    summary = {
        **target,
        "stream_mode": stream_mode,
        "base_event_order_preserved": stream_mode == "base_event_stream",
        "drift_start_fraction": drift_start_fraction,
        "drift_end_fraction": drift_end_fraction,
        "drift_start_event_index": drift_start_event_index,
        "drift_end_event_index": drift_end_event_index,
        "drift_window_slots": drift_window_slots,
        "target_drift_query_fraction": DRIFT_WINDOW_QUERY_FRACTION,
        "target_drift_query_count": target_drift_query_count,
        "global_drift_position_count": len(drift_positions),
        "drift_position_seed": random_seed,
        "drift_positions_sha256": hashlib.sha256(drift_positions_array.tobytes()).hexdigest(),
        "drift_positions_sample": [int(position) for position in drift_positions[:10]],
        "existing_post_start_drift_position_count": len(existing_drift_positions),
        "target_heldout_doc_count": DRIFT_TARGET_DOC_COUNT,
        "heldout_doc_count": len(heldout_doc_indices),
        "heldout_event_count": len(drift_specs),
        "drift_query_count": int(drift_query_count),
        "repeated_drift_query_fill_count": int(repeated_drift_query_fill_count),
        "duplicated_non_drift_fill_count": int(duplicated_non_drift_fill_count),
        "injection_count": len(injections),
        "injection_replicas_per_doc": DRIFT_INJECTION_REPLICAS_PER_DOC,
        "unique_injection_peer_count": len({int(item["peer_id"]) for item in injections}),
        "removed_peer_doc_pairs": sum(len(owners) for owners in removed_owners_by_doc.values()),
        "notes": {
            "heldout_docs": "Every sampled doc whose MeSH list contains the target MeSH is removed from seeded ownership, including multi-MeSH docs.",
            "spreading": f"Each held-out doc is injected into {DRIFT_INJECTION_REPLICAS_PER_DOC} random peer inventories at the drift start; subsequent spreading happens through successful retrieval in the simulator.",
        },
    }
    return query_specs, filtered_docs_by_peer, filtered_topics_by_peer, injections, summary


def _write_gzip_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, sort_keys=True)


def _open_text_reader(path: Path):
    return gzip.open(path, "rt", encoding="utf-8", newline="") if path.suffix == ".gz" else path.open("r", encoding="utf-8", newline="")


def _load_answer_targets_by_qid(path: Path) -> dict[int, dict[str, Any]]:
    answers: dict[int, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                answers[int(row["qid"])] = row
    return answers


def _load_peer_future_docs(path: Path) -> dict[int, set[str]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    return {int(peer_id): {str(doc_id) for doc_id in doc_ids} for peer_id, doc_ids in payload.items()}


def _precompute_base_stream_concept_drift(
    *,
    setting: PubmedQaAblationSetting,
    data_dir: Path,
    base_dataset_dir: Path,
    rows: list[Any],
    topic_terms: list[str],
    topic_id_by_mesh: dict[str, int],
    doc_indices_by_topic: list[list[int]],
    docs_by_peer_indices: dict[int, list[int]],
    topics_by_peer: dict[int, list[int]],
    sample_summary: dict[str, Any],
    mesh_summary: dict[str, Any],
    base_manifest: dict[str, Any],
    emb_manifest: dict[str, Any],
    source_full_dir: Path,
    rephrased_queries: Path,
    rephrase_dir: Path,
    alpha: float,
    random_seed: int,
    event_order_seed: int | None,
    churn_random_seed: int,
    embedding_dtype: str,
    embedding_batch_size: int,
) -> dict[str, Any]:
    data_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = data_dir / "embeddings" / "manifest.json"
    event_count = setting.peers * setting.docs_per_peer
    drift_start_event_index = int(event_count * setting.drift_start_fraction)
    drift_end_event_index = int(event_count * setting.drift_end_fraction)
    drift_window_slots = drift_end_event_index - drift_start_event_index
    target_drift_query_count = int(round(drift_window_slots * DRIFT_WINDOW_QUERY_FRACTION))

    target = _select_concept_drift_topic(
        rows=rows,
        topic_terms=topic_terms,
        query_specs=_query_specs_for_inventories(docs_by_peer_indices, topics_by_peer),
        target_unique_doc_count=DRIFT_TARGET_DOC_COUNT,
        preferred_mesh=CONCEPT_DRIFT_MESH,
    )
    target_topic_id = int(target["topic_id"])
    target_mesh = str(target["mesh_term"])
    used_doc_indices = {doc_index for doc_indices in docs_by_peer_indices.values() for doc_index in doc_indices}
    heldout_doc_indices = _doc_indices_containing_mesh(
        rows,
        mesh=target_mesh,
        used_doc_indices=used_doc_indices,
    )
    if not heldout_doc_indices:
        raise ValueError(f"Selected drift MeSH {target_mesh!r} has no sampled docs")
    pubid_to_doc_index = {str(row.pubid): int(index) for index, row in enumerate(rows)}
    heldout_pubids = {str(rows[doc_index].pubid) for doc_index in heldout_doc_indices}

    base_docs_by_peer_pubids = _load_peer_future_docs(base_dataset_dir / PUBMEDQA_PEER_FUTURE_DOCS_FILENAME)
    removed_owners_by_doc: dict[int, set[int]] = defaultdict(set)
    docs_by_peer_pubids: dict[int, set[str]] = {}
    owners_by_doc: dict[str, set[int]] = defaultdict(set)
    for peer_id, base_pubids in sorted(base_docs_by_peer_pubids.items()):
        kept_pubids: set[str] = set()
        for pubid in base_pubids:
            doc_index = pubid_to_doc_index[str(pubid)]
            if str(pubid) in heldout_pubids:
                removed_owners_by_doc[doc_index].add(int(peer_id))
                continue
            kept_pubids.add(str(pubid))
        docs_by_peer_pubids[int(peer_id)] = kept_pubids
        for pubid in kept_pubids:
            owners_by_doc[pubid].add(int(peer_id))

    metadata_template_path = base_dataset_dir / PUBMEDQA_RANDOMIZED_EVENTS_FILENAME
    base_event_rows: list[dict[str, str]] = []
    with _open_text_reader(metadata_template_path) as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            base_event_rows.append(dict(row))
    if len(base_event_rows) != event_count:
        raise ValueError(f"Base stream has {len(base_event_rows)} events; expected {event_count}")

    rng = np.random.default_rng(churn_random_seed)
    existing_drift_positions = [
        event_index
        for event_index in range(drift_start_event_index, drift_end_event_index)
        if str(base_event_rows[event_index]["target_doc_id"]) in heldout_pubids
    ]
    if len(existing_drift_positions) > target_drift_query_count:
        raise ValueError("Base stream already exceeds requested drift-topic query count")
    remaining_drift_count = target_drift_query_count - len(existing_drift_positions)
    candidate_positions = np.asarray(
        [
            event_index
            for event_index in range(drift_start_event_index, drift_end_event_index)
            if str(base_event_rows[event_index]["target_doc_id"]) not in heldout_pubids
        ],
        dtype=np.int64,
    )
    sampled_positions = (
        rng.choice(candidate_positions, size=remaining_drift_count, replace=False).tolist()
        if remaining_drift_count
        else []
    )
    drift_positions = sorted(int(position) for position in existing_drift_positions + sampled_positions)
    heldout_doc_array = np.asarray(sorted(heldout_doc_indices), dtype=np.int32)
    assigned_doc_indices = [
        int(index)
        for index in rng.choice(heldout_doc_array, size=len(drift_positions), replace=True).tolist()
    ]
    assigned_doc_by_position = dict(zip(drift_positions, assigned_doc_indices, strict=True))

    owner_pool = sorted(docs_by_peer_pubids)
    injections: list[dict[str, int | str]] = []
    injected_owners_by_doc: dict[str, set[int]] = defaultdict(set)
    for doc_index in sorted(heldout_doc_indices):
        removed_owners = removed_owners_by_doc.get(doc_index, set())
        candidates = [peer_id for peer_id in owner_pool if peer_id not in removed_owners]
        if not candidates:
            candidates = owner_pool
        replace = len(candidates) < DRIFT_INJECTION_REPLICAS_PER_DOC
        selected_peers = rng.choice(
            candidates,
            size=DRIFT_INJECTION_REPLICAS_PER_DOC,
            replace=replace,
        ).tolist()
        doc_id = str(rows[doc_index].pubid)
        for peer_id in selected_peers:
            injections.append(
                {
                    "event_index": drift_start_event_index,
                    "peer_id": int(peer_id),
                    "doc_id": doc_id,
                }
            )
            injected_owners_by_doc[doc_id].add(int(peer_id))

    answer_targets_path = base_dataset_dir / PUBMEDQA_ANSWER_TARGETS_FILENAME
    base_answers = _load_answer_targets_by_qid(answer_targets_path)
    answers_by_doc: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for answer in base_answers.values():
        answers_by_doc[str(answer.get("target_doc_id"))].append(answer)
    for doc_id in heldout_pubids:
        if not answers_by_doc.get(doc_id):
            raise ValueError(f"No base answer target found for held-out doc {doc_id}")

    base_query_keys = np.load(str(base_manifest["query_keys_path"]), allow_pickle=True).astype(str)
    base_query_key_to_index = {str(key): int(index) for index, key in enumerate(base_query_keys.tolist())}
    base_query_embeddings = np.load(str(base_manifest["query_embeddings_path"]), mmap_mode="r")
    dim = int(base_manifest.get("embedding_dim") or emb_manifest["embedding_dim"])
    dtype = np.dtype(embedding_dtype)
    embeddings_dir = data_dir / "embeddings"
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    query_keys_path = embeddings_dir / "query_keys.npy"
    query_embeddings_path = embeddings_dir / "query_embeddings.npy"
    output_query_keys = np.empty(event_count, dtype="<U32")
    query_embeddings = np.lib.format.open_memmap(
        query_embeddings_path,
        mode="w+",
        dtype=dtype,
        shape=(event_count, dim),
    )

    chronological_path = data_dir / PUBMEDQA_EVENTS_FILENAME
    randomized_path = data_dir / PUBMEDQA_RANDOMIZED_EVENTS_FILENAME
    searchable_path = data_dir / PUBMEDQA_SEEDED_SEARCHABLE_FILENAME
    query_topics_path = data_dir / PUBMEDQA_QUERY_TOPICS_FILENAME
    output_answers_path = data_dir / PUBMEDQA_ANSWER_TARGETS_FILENAME

    searchable_count = 0
    topic_counts: Counter[int] = Counter()
    variant_counts: Counter[str] = Counter()
    drift_position_set = set(drift_positions)
    drift_doc_assignment_counts: Counter[str] = Counter()

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
        "is_drift_topic",
    )
    with (
        _open_text_writer(chronological_path) as chronological_handle,
        _open_text_writer(randomized_path) as randomized_handle,
        _open_text_writer(searchable_path) as searchable_handle,
        _open_text_writer(query_topics_path) as topics_handle,
        output_answers_path.open("w", encoding="utf-8") as answers_handle,
    ):
        chronological_writer = csv.DictWriter(chronological_handle, fieldnames=AMAZON_EVENTS_FIELDNAMES)
        randomized_writer = csv.DictWriter(randomized_handle, fieldnames=AMAZON_EVENTS_FIELDNAMES)
        searchable_writer = csv.DictWriter(searchable_handle, fieldnames=SEARCHABLE_QUERY_FIELDNAMES)
        topics_writer = csv.DictWriter(topics_handle, fieldnames=topic_fields)
        chronological_writer.writeheader()
        randomized_writer.writeheader()
        searchable_writer.writeheader()
        topics_writer.writeheader()

        for start in range(0, event_count, embedding_batch_size):
            end = min(start + embedding_batch_size, event_count)
            chunk_embeddings = np.empty((end - start, dim), dtype=np.float32)
            for offset, event_index in enumerate(range(start, end)):
                base_row = dict(base_event_rows[event_index])
                user_id = int(base_row["user_id"])
                is_drift_position = event_index in drift_position_set
                if is_drift_position:
                    doc_index = int(assigned_doc_by_position[event_index])
                    doc_id = str(rows[doc_index].pubid)
                    answer_pool = answers_by_doc[doc_id]
                    answer_template = answer_pool[event_index % len(answer_pool)]
                    qid = event_count + event_index
                    query_key = f"pmq_drift_{event_index:07d}"
                    embedding_key = str(answer_template["query_key"])
                    answer_row = dict(answer_template)
                    answer_row.update(
                        {
                            "event_index": event_index,
                            "qid": qid,
                            "query_key": query_key,
                            "source_pubid": doc_id,
                            "target_doc_id": doc_id,
                        }
                    )
                    is_searchable = bool(injected_owners_by_doc[doc_id] - {user_id})
                    topic_id = target_topic_id
                    mesh_term = target_mesh
                    source_selected_index = int(rows[doc_index].source_index)
                    variant_label = str(answer_row.get("query_variant") or "drift_reused")
                    drift_doc_assignment_counts[doc_id] += 1
                else:
                    qid = int(base_row["qid"])
                    query_key = str(base_row["query"])
                    doc_id = str(base_row["target_doc_id"])
                    answer_row = dict(base_answers[qid])
                    embedding_key = query_key
                    doc_index = pubid_to_doc_index[doc_id]
                    mesh_term = getattr(rows[doc_index], "meshes", ("",))[0]
                    topic_id = topic_id_by_mesh[mesh_term]
                    source_selected_index = int(rows[doc_index].source_index)
                    variant_label = str(answer_row.get("query_variant") or "unknown")
                    is_searchable = bool(owners_by_doc[doc_id] - {user_id})
                output_query_keys[event_index] = query_key
                chunk_embeddings[offset] = np.asarray(
                    base_query_embeddings[base_query_key_to_index[embedding_key]],
                    dtype=np.float32,
                )
                output_row = {
                    "timestamp": synthetic_timestamp_for_index(event_index),
                    "qid": qid,
                    "user_id": user_id,
                    "query": query_key,
                    "target_doc_id": doc_id,
                    "candidate_doc_ids": "[]",
                }
                chronological_writer.writerow(output_row)
                randomized_writer.writerow(output_row)
                searchable_writer.writerow({"query_id": qid, "searchable": "true" if is_searchable else "false"})
                searchable_count += int(is_searchable)
                topic_counts[int(topic_id)] += 1
                variant_counts[variant_label] += 1
                topics_writer.writerow(
                    {
                        "qid": qid,
                        "user_id": user_id,
                        "query": query_key,
                        "generated_topic": int(topic_id),
                        "mesh_term": mesh_term,
                        "target_doc_id": doc_id,
                        "target_doc_topic": int(topic_id),
                        "query_variant": variant_label,
                        "source_selected_index": source_selected_index,
                        "searchable": "true" if is_searchable else "false",
                        "is_drift_topic": "true" if is_drift_position else "false",
                    }
                )
                answer_row["event_index"] = event_index
                answers_handle.write(json.dumps(answer_row, ensure_ascii=False, sort_keys=True) + "\n")
            query_embeddings[start:end] = chunk_embeddings.astype(dtype, copy=False)
    query_embeddings.flush()
    del query_embeddings
    np.save(query_keys_path, output_query_keys)

    with _open_text_writer(data_dir / PUBMEDQA_DOC_TOPICS_FILENAME) as handle:
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
    peer_summary = _write_peer_future_docs(data_dir, docs_by_peer_pubids)
    drift_owner_injections_path = data_dir / DRIFT_OWNER_INJECTIONS_FILENAME
    _write_gzip_json(drift_owner_injections_path, injections)

    drift_positions_array = np.asarray(drift_positions, dtype=np.int64)
    concept_drift_summary = {
        **target,
        "stream_mode": "base_event_stream",
        "base_event_order_preserved": True,
        "base_metadata_template_path": str(metadata_template_path),
        "drift_start_fraction": setting.drift_start_fraction,
        "drift_end_fraction": setting.drift_end_fraction,
        "drift_start_event_index": drift_start_event_index,
        "drift_end_event_index": drift_end_event_index,
        "drift_window_slots": drift_window_slots,
        "target_drift_query_fraction": DRIFT_WINDOW_QUERY_FRACTION,
        "target_drift_query_count": target_drift_query_count,
        "global_drift_position_count": len(drift_positions),
        "drift_position_seed": churn_random_seed,
        "drift_positions_sha256": hashlib.sha256(drift_positions_array.tobytes()).hexdigest(),
        "drift_positions_sample": [int(position) for position in drift_positions[:10]],
        "existing_post_start_drift_position_count": len(existing_drift_positions),
        "target_heldout_doc_count": DRIFT_TARGET_DOC_COUNT,
        "heldout_doc_count": len(heldout_doc_indices),
        "heldout_event_count": int(sum(str(row["target_doc_id"]) in heldout_pubids for row in base_event_rows)),
        "drift_query_count": len(drift_positions),
        "repeated_drift_query_fill_count": max(0, len(drift_positions) - len(set(assigned_doc_indices))),
        "duplicated_non_drift_fill_count": 0,
        "injection_count": len(injections),
        "injection_replicas_per_doc": DRIFT_INJECTION_REPLICAS_PER_DOC,
        "unique_injection_peer_count": len({int(item["peer_id"]) for item in injections}),
        "removed_peer_doc_pairs": sum(len(owners) for owners in removed_owners_by_doc.values()),
        "drift_doc_assignment_counts": dict(sorted(drift_doc_assignment_counts.items())),
        "notes": {
            "heldout_docs": "Every sampled doc whose MeSH list contains the target MeSH is removed from seeded ownership.",
            "event_stream": "Non-drift positions are copied from the base PubMedQA metadata stream; selected post-start positions are retargeted globally.",
        },
    }
    top_sampled_topics = [
        {"topic_id": topic_id, "mesh_term": topic_terms[topic_id], "query_count": int(count)}
        for topic_id, count in topic_counts.most_common(20)
    ]
    variant_count_labels = {str(key): int(value) for key, value in sorted(variant_counts.items())}
    manifest = {
        "dataset": "pubmedqa",
        "pubmedqa_variant": f"expanded_all_mesh_ablation_{setting.experiment}",
        "metadata_path": str(chronological_path),
        "searchable_queries_path": str(searchable_path),
        "output_dir": str(embeddings_dir),
        "model": emb_manifest.get("embedding_model", DEFAULT_EMBEDDING_MODEL),
        "embedding_dim": dim,
        "embedding_dtype": str(dtype),
        "query_keys_path": str(query_keys_path),
        "query_embeddings_path": str(query_embeddings_path),
        "query_count": event_count,
        "doc_keys_path": str(base_manifest["doc_keys_path"]),
        "doc_embeddings_path": str(base_manifest["doc_embeddings_path"]),
        "doc_count": int(base_manifest["doc_count"]),
        "user_count": setting.peers,
        "docs_per_peer": setting.docs_per_peer,
        "dirichlet_alpha": alpha,
        "random_seed": random_seed,
        "event_order_seed": random_seed if event_order_seed is None else event_order_seed,
        "source_full_dir": str(source_full_dir),
        "base_dataset_dir": str(base_dataset_dir),
        "source_query_variant": "base_stream_plus_drift_retargets",
        "rephrased_queries_path": str(rephrased_queries),
        "rephrase_embedding_manifest_path": str(_rephrase_embedding_paths(rephrase_dir)["manifest"]),
        "churn_schedule_path": None,
        "drift_owner_injections": str(drift_owner_injections_path),
        "concept_drift": concept_drift_summary,
    }
    _write_json(manifest_path, manifest)
    summary = {
        "dataset": "pubmedqa",
        "pubmedqa_variant": f"expanded_all_mesh_ablation_{setting.experiment}",
        "setting": asdict(setting),
        "data_dir": str(data_dir),
        "source_full_dir": str(source_full_dir),
        "base_dataset_dir": str(base_dataset_dir),
        "full_document_count": int(base_manifest["doc_count"]),
        "topic_count": len(topic_terms),
        "mesh_summary": mesh_summary,
        "user_count": setting.peers,
        "docs_per_peer": setting.docs_per_peer,
        "query_count": event_count,
        "dirichlet_alpha": alpha,
        "random_seed": random_seed,
        "event_order_seed": random_seed if event_order_seed is None else event_order_seed,
        "churn_random_seed": churn_random_seed,
        "searchable_query_count": searchable_count,
        "query_variant_counts": variant_count_labels,
        "sample_summary": {**sample_summary, "concept_drift": concept_drift_summary, "top_sampled_topics": top_sampled_topics},
        "manifest_path": str(manifest_path),
        "events_path": str(chronological_path),
        "randomized_events_path": str(randomized_path),
        "searchable_queries_path": str(searchable_path),
        "peer_future_docs_summary": peer_summary,
        "query_topics_path": str(query_topics_path),
        "doc_topics_path": str(data_dir / PUBMEDQA_DOC_TOPICS_FILENAME),
        "answer_targets_path": str(output_answers_path),
        "churn_schedule_path": None,
        "drift_owner_injections_path": str(drift_owner_injections_path),
        "concept_drift": concept_drift_summary,
    }
    _write_json(data_dir / PUBMEDQA_PRECOMPUTE_SUMMARY_FILENAME, summary)
    _write_json(
        data_dir / "pubmedqa_ablation_setting.json",
        {"setting": asdict(setting), "summary_path": str(data_dir / PUBMEDQA_PRECOMPUTE_SUMMARY_FILENAME)},
    )
    print(f"[pubmedqa-ablation] wrote manifest to {manifest_path}", flush=True)
    print(f"[pubmedqa-ablation] wrote summary to {data_dir / PUBMEDQA_PRECOMPUTE_SUMMARY_FILENAME}", flush=True)
    return summary


def precompute_setting(
    setting: PubmedQaAblationSetting,
    *,
    output_root: Path,
    base_dataset_dir: Path,
    hf_root: Path,
    source_full_dir: Path,
    rephrased_queries: Path,
    rephrase_embedding_dir: Path | None,
    alpha: float,
    random_seed: int,
    event_order_seed: int | None,
    churn_random_seed: int,
    embedding_dtype: str,
    embedding_batch_size: int,
    force: bool,
) -> dict[str, Any]:
    data_dir = data_dir_for_setting(output_root, setting)
    manifest_path = data_dir / "embeddings" / "manifest.json"
    if manifest_path.exists() and not force:
        summary_path = data_dir / PUBMEDQA_PRECOMPUTE_SUMMARY_FILENAME
        if summary_path.exists():
            return _read_json(summary_path)
        return {"data_dir": str(data_dir), "manifest_path": str(manifest_path), "skipped": True}

    data_dir.mkdir(parents=True, exist_ok=True)
    paths = _embedding_paths(source_full_dir)
    state = _read_json(paths["state"])
    emb_manifest = _read_json(paths["manifest"])
    row_count = int(emb_manifest["row_count"])
    dim = int(emb_manifest["embedding_dim"])
    if int(state.get("doc_rows_embedded", 0)) != row_count:
        raise RuntimeError("Source full PubMedQA doc embeddings are incomplete")
    if int(state.get("query_rows_embedded", 0)) != row_count:
        raise RuntimeError("Source full PubMedQA query embeddings are incomplete")

    rows, mesh_summary = _load_full_rows(hf_root=hf_root, source_full_dir=source_full_dir)
    source_doc_pubids = np.load(paths["doc_pubids"], allow_pickle=True).astype(str)
    source_query_pubids = np.load(paths["query_pubids"], allow_pickle=True).astype(str)
    row_pubids = np.asarray([row.pubid for row in rows], dtype=str)
    if len(rows) != row_count:
        raise ValueError(f"Loaded {len(rows)} rows but source embeddings have {row_count}")
    if not np.array_equal(source_doc_pubids, row_pubids):
        raise ValueError("Source doc_pubids.npy is not aligned with full PubMedQA rows")
    if not np.array_equal(source_query_pubids, row_pubids):
        raise ValueError("Source query_pubids.npy is not aligned with full PubMedQA rows")

    base_manifest = _read_json(base_dataset_dir / "embeddings" / "manifest.json")
    doc_keys_path = str(base_manifest.get("doc_keys_path") or paths["doc_pubids"])
    doc_embeddings_path = str(base_manifest.get("doc_embeddings_path") or paths["doc_embeddings"])

    rephrased_by_key = _load_rephrased_rows(rephrased_queries)
    rephrase_dir = rephrase_embedding_dir or (rephrased_queries.parent / "embeddings")
    rephrase_embeddings, source_to_offset = _load_rephrase_embedding_lookup(
        embedding_dir=rephrase_dir,
        rows=rows,
        dim=dim,
    )
    source_query_embeddings = _load_embedding_array(paths["query_embeddings"], rows=row_count, dim=dim)

    topic_terms, topic_id_by_mesh, doc_indices_by_topic = _build_topic_pools(rows)
    docs_by_peer_indices, topics_by_peer, sample_summary = _sample_exact_inventories(
        peer_count=setting.peers,
        docs_per_peer=setting.docs_per_peer,
        topic_count=len(topic_terms),
        doc_indices_by_topic=doc_indices_by_topic,
        alpha=alpha,
        random_seed=random_seed,
    )
    if setting.experiment == "concept_drift" and CONCEPT_DRIFT_STREAM_MODE == "base_event_stream":
        return _precompute_base_stream_concept_drift(
            setting=setting,
            data_dir=data_dir,
            base_dataset_dir=base_dataset_dir,
            rows=rows,
            topic_terms=topic_terms,
            topic_id_by_mesh=topic_id_by_mesh,
            doc_indices_by_topic=doc_indices_by_topic,
            docs_by_peer_indices=docs_by_peer_indices,
            topics_by_peer=topics_by_peer,
            sample_summary=sample_summary,
            mesh_summary=mesh_summary,
            base_manifest=base_manifest,
            emb_manifest=emb_manifest,
            source_full_dir=source_full_dir,
            rephrased_queries=rephrased_queries,
            rephrase_dir=rephrase_dir,
            alpha=alpha,
            random_seed=random_seed,
            event_order_seed=event_order_seed,
            churn_random_seed=churn_random_seed,
            embedding_dtype=embedding_dtype,
            embedding_batch_size=embedding_batch_size,
        )
    query_specs = _query_specs_for_inventories(docs_by_peer_indices, topics_by_peer)
    all_docs_by_peer_indices = dict(docs_by_peer_indices)
    all_topics_by_peer = dict(topics_by_peer)
    churn_schedule: dict[str, Any] | None = None
    drift_owner_injections: list[dict[str, int | str]] | None = None
    concept_drift_summary: dict[str, Any] | None = None
    if setting.experiment == "concept_drift":
        (
            query_specs,
            docs_by_peer_indices,
            topics_by_peer,
            drift_owner_injections,
            concept_drift_summary,
        ) = _build_concept_drift_specs_and_injections(
            initial_specs=query_specs,
            docs_by_peer=docs_by_peer_indices,
            topics_by_peer=topics_by_peer,
            rows=rows,
            topic_terms=topic_terms,
            drift_start_fraction=setting.drift_start_fraction,
            drift_end_fraction=setting.drift_end_fraction,
            random_seed=churn_random_seed,
            event_order_seed=random_seed if event_order_seed is None else event_order_seed,
            stream_mode=CONCEPT_DRIFT_STREAM_MODE,
        )
        sample_summary = {
            **sample_summary,
            "concept_drift": concept_drift_summary,
        }
        all_docs_by_peer_indices = dict(docs_by_peer_indices)
        all_topics_by_peer = dict(topics_by_peer)
    elif setting.experiment == "churn":
        replacement_count = int(round(setting.peers * setting.replacement_fraction))
        replacement_docs_by_peer, replacement_topics_by_peer, replacement_summary = _sample_exact_inventories(
            peer_count=replacement_count,
            docs_per_peer=setting.docs_per_peer,
            topic_count=len(topic_terms),
            doc_indices_by_topic=doc_indices_by_topic,
            alpha=alpha,
            random_seed=random_seed + 10_000,
            peer_id_offset=setting.peers,
        )
        all_docs_by_peer_indices.update(replacement_docs_by_peer)
        all_topics_by_peer.update(replacement_topics_by_peer)
        query_specs, churn_schedule = _build_churn_specs_and_schedule(
            initial_specs=query_specs,
            initial_docs_by_peer=docs_by_peer_indices,
            initial_topics_by_peer=topics_by_peer,
            replacement_docs_by_peer=replacement_docs_by_peer,
            replacement_topics_by_peer=replacement_topics_by_peer,
            rows=rows,
            churn_fraction=setting.replacement_fraction,
            random_seed=churn_random_seed,
        )
        sample_summary = {
            **sample_summary,
            "replacement_sample_summary": replacement_summary,
            "post_churn_event_count": len(query_specs) // 2,
        }
    else:
        order_seed = random_seed if event_order_seed is None else event_order_seed
        permutation = np.random.default_rng(order_seed + 4).permutation(len(query_specs))
        query_specs = [query_specs[int(index)] for index in permutation.tolist()]

    query_count = len(query_specs)
    qid_width = max(6, len(str(max(0, query_count - 1))))
    query_keys = np.asarray([_query_key(qid, qid_width) for qid in range(query_count)], dtype=str)
    target_doc_indices = np.asarray([spec.doc_index for spec in query_specs], dtype=np.int32)
    target_pubids = np.asarray([rows[int(doc_index)].pubid for doc_index in target_doc_indices], dtype=str)
    query_variants = np.random.default_rng(random_seed + 6).integers(0, 6, size=query_count, dtype=np.int8)
    missing_offsets = np.unique(target_doc_indices[(query_variants > 0) & (source_to_offset[target_doc_indices] < 0)])
    if missing_offsets.size:
        raise ValueError(f"Missing rephrase embeddings for sampled source rows: {missing_offsets[:10].tolist()}")

    owners_by_doc: dict[str, set[int]] = defaultdict(set)
    docs_by_peer_pubids: dict[int, set[str]] = {}
    for peer_id, doc_indices in all_docs_by_peer_indices.items():
        pubids = {rows[int(doc_index)].pubid for doc_index in doc_indices}
        docs_by_peer_pubids[peer_id] = pubids
        for pubid in pubids:
            owners_by_doc[pubid].add(peer_id)

    embeddings_dir = data_dir / "embeddings"
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    query_keys_path = embeddings_dir / "query_keys.npy"
    query_embeddings_path = embeddings_dir / "query_embeddings.npy"
    np.save(query_keys_path, query_keys)

    dtype = np.dtype(embedding_dtype)
    query_embeddings = np.lib.format.open_memmap(
        query_embeddings_path,
        mode="w+",
        dtype=dtype,
        shape=(query_count, dim),
    )
    for start in range(0, query_count, embedding_batch_size):
        end = min(start + embedding_batch_size, query_count)
        chunk_doc_indices = target_doc_indices[start:end]
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

    def event_row_for_qid(qid: int, event_index: int) -> dict[str, str | int]:
        spec = query_specs[qid]
        return {
            "timestamp": synthetic_timestamp_for_index(event_index),
            "qid": qid,
            "user_id": int(spec.peer_id),
            "query": str(query_keys[qid]),
            "target_doc_id": str(target_pubids[qid]),
            "candidate_doc_ids": "[]",
        }

    def is_concept_drift_qid(qid: int) -> bool:
        if concept_drift_summary is None:
            return False
        spec = query_specs[qid]
        row = rows[int(spec.doc_index)]
        target_mesh = str(concept_drift_summary["mesh_term"])
        return (
            target_mesh in getattr(row, "meshes", ())
            and int(concept_drift_summary["drift_start_event_index"])
            <= qid
            < int(concept_drift_summary["drift_end_event_index"])
        )

    def searchable_for_qid(qid: int) -> bool:
        spec = query_specs[qid]
        doc_id = str(target_pubids[qid])
        if is_concept_drift_qid(qid):
            return True
        return bool(owners_by_doc[doc_id] - {int(spec.peer_id)})

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
        "is_drift_topic",
    )
    searchable_count = 0
    topic_counts: Counter[int] = Counter()
    variant_counts: Counter[int] = Counter()
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
        for event_index, spec in enumerate(query_specs):
            qid = event_index
            source_row = rows[int(spec.doc_index)]
            variant = int(query_variants[qid])
            rephrased_row = rephrased_by_key[(source_row.config, int(source_row.row_index))]
            searchable = searchable_for_qid(qid)
            searchable_count += int(searchable)
            topic_counts[int(spec.topic_id)] += 1
            variant_counts[variant] += 1
            variant_label = "original" if variant == 0 else f"rephrased_{variant}"
            is_drift_topic = is_concept_drift_qid(qid)
            randomized_writer.writerow(event_row_for_qid(qid, event_index))
            searchable_writer.writerow({"query_id": qid, "searchable": "true" if searchable else "false"})
            topics_writer.writerow(
                {
                    "qid": qid,
                    "user_id": int(spec.peer_id),
                    "query": str(query_keys[qid]),
                    "generated_topic": int(spec.topic_id),
                    "mesh_term": topic_terms[int(spec.topic_id)],
                    "target_doc_id": str(target_pubids[qid]),
                    "target_doc_topic": int(spec.topic_id),
                    "query_variant": variant_label,
                    "source_selected_index": int(source_row.source_index),
                    "searchable": "true" if searchable else "false",
                    "is_drift_topic": "true" if is_drift_topic else "false",
                }
            )
            question = source_row.question if variant == 0 else rephrased_row["rephrasings"][variant - 1]
            answers_handle.write(
                json.dumps(
                    {
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
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )

    with _open_text_writer(data_dir / PUBMEDQA_DOC_TOPICS_FILENAME) as handle:
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
    peer_summary = _write_peer_future_docs(data_dir, docs_by_peer_pubids)
    churn_schedule_path = None
    if churn_schedule is not None:
        churn_schedule_path = data_dir / CHURN_SCHEDULE_FILENAME
        _write_gzip_json(churn_schedule_path, churn_schedule)
    drift_owner_injections_path = None
    if drift_owner_injections is not None:
        drift_owner_injections_path = data_dir / DRIFT_OWNER_INJECTIONS_FILENAME
        _write_gzip_json(drift_owner_injections_path, drift_owner_injections)

    variant_count_labels = {
        ("original" if key == 0 else f"rephrased_{key}"): int(value)
        for key, value in sorted(variant_counts.items())
    }
    top_sampled_topics = [
        {"topic_id": topic_id, "mesh_term": topic_terms[topic_id], "query_count": int(count)}
        for topic_id, count in topic_counts.most_common(20)
    ]

    manifest = {
        "dataset": "pubmedqa",
        "pubmedqa_variant": f"expanded_all_mesh_ablation_{setting.experiment}",
        "metadata_path": str(chronological_path),
        "searchable_queries_path": str(searchable_path),
        "output_dir": str(embeddings_dir),
        "model": emb_manifest.get("embedding_model", DEFAULT_EMBEDDING_MODEL),
        "embedding_dim": dim,
        "embedding_dtype": str(dtype),
        "query_keys_path": str(query_keys_path),
        "query_embeddings_path": str(query_embeddings_path),
        "query_count": query_count,
        "doc_keys_path": doc_keys_path,
        "doc_embeddings_path": doc_embeddings_path,
        "doc_count": row_count,
        "user_count": setting.peers,
        "docs_per_peer": setting.docs_per_peer,
        "dirichlet_alpha": alpha,
        "random_seed": random_seed,
        "event_order_seed": random_seed if event_order_seed is None else event_order_seed,
        "source_full_dir": str(source_full_dir),
        "base_dataset_dir": str(base_dataset_dir),
        "source_query_variant": "original_plus_5_rephrases",
        "rephrased_queries_path": str(rephrased_queries),
        "rephrase_embedding_manifest_path": str(_rephrase_embedding_paths(rephrase_dir)["manifest"]),
        "churn_schedule_path": str(churn_schedule_path) if churn_schedule_path else None,
        "drift_owner_injections": (
            str(drift_owner_injections_path) if drift_owner_injections_path else None
        ),
        "concept_drift": concept_drift_summary,
    }
    _write_json(manifest_path, manifest)

    summary = {
        "dataset": "pubmedqa",
        "pubmedqa_variant": f"expanded_all_mesh_ablation_{setting.experiment}",
        "setting": asdict(setting),
        "data_dir": str(data_dir),
        "source_full_dir": str(source_full_dir),
        "base_dataset_dir": str(base_dataset_dir),
        "full_document_count": row_count,
        "topic_count": len(topic_terms),
        "mesh_summary": mesh_summary,
        "user_count": setting.peers,
        "docs_per_peer": setting.docs_per_peer,
        "query_count": query_count,
        "dirichlet_alpha": alpha,
        "random_seed": random_seed,
        "event_order_seed": random_seed if event_order_seed is None else event_order_seed,
        "churn_random_seed": churn_random_seed,
        "searchable_query_count": searchable_count,
        "query_variant_counts": variant_count_labels,
        "sample_summary": {**sample_summary, "top_sampled_topics": top_sampled_topics},
        "manifest_path": str(manifest_path),
        "events_path": str(chronological_path),
        "randomized_events_path": str(randomized_path),
        "searchable_queries_path": str(searchable_path),
        "peer_future_docs_summary": peer_summary,
        "query_topics_path": str(query_topics_path),
        "doc_topics_path": str(data_dir / PUBMEDQA_DOC_TOPICS_FILENAME),
        "answer_targets_path": str(answer_targets_path),
        "churn_schedule_path": str(churn_schedule_path) if churn_schedule_path else None,
        "drift_owner_injections_path": (
            str(drift_owner_injections_path) if drift_owner_injections_path else None
        ),
        "concept_drift": concept_drift_summary,
    }
    _write_json(data_dir / PUBMEDQA_PRECOMPUTE_SUMMARY_FILENAME, summary)
    _write_json(
        data_dir / "pubmedqa_ablation_setting.json",
        {"setting": asdict(setting), "summary_path": str(data_dir / PUBMEDQA_PRECOMPUTE_SUMMARY_FILENAME)},
    )
    print(f"[pubmedqa-ablation] wrote manifest to {manifest_path}", flush=True)
    print(f"[pubmedqa-ablation] wrote summary to {data_dir / PUBMEDQA_PRECOMPUTE_SUMMARY_FILENAME}", flush=True)
    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment",
        choices=("base", "user_scalability", "doc_scalability", "churn", "concept_drift"),
        required=True,
    )
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--print-matrix", action="store_true")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--base-dataset-dir", type=Path, default=DEFAULT_BASE_DATASET_DIR)
    parser.add_argument("--hf-root", type=Path, default=DEFAULT_HF_ROOT)
    parser.add_argument("--source-full-dir", type=Path, default=DEFAULT_SOURCE_FULL_DIR)
    parser.add_argument("--rephrased-queries", type=Path, default=DEFAULT_REPHRASED_QUERIES)
    parser.add_argument("--rephrase-embedding-dir", type=Path, default=None)
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument("--random-seed", type=int, default=DEFAULT_PUBMEDQA_RANDOM_SEED)
    parser.add_argument("--event-order-seed", type=int, default=None)
    parser.add_argument("--churn-random-seed", type=int, default=42)
    parser.add_argument("--embedding-dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument("--embedding-batch-size", type=int, default=4096)
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if args.print_matrix:
        print(json.dumps([asdict(setting) for setting in iter_settings(args.experiment)], indent=2))
        return

    setting = setting_by_index(args.experiment, args.index)
    summary = precompute_setting(
        setting,
        output_root=args.output_root,
        base_dataset_dir=args.base_dataset_dir,
        hf_root=args.hf_root,
        source_full_dir=args.source_full_dir,
        rephrased_queries=args.rephrased_queries,
        rephrase_embedding_dir=args.rephrase_embedding_dir,
        alpha=args.alpha,
        random_seed=args.random_seed,
        event_order_seed=args.event_order_seed,
        churn_random_seed=args.churn_random_seed,
        embedding_dtype=args.embedding_dtype,
        embedding_batch_size=args.embedding_batch_size,
        force=args.force,
    )
    print(json.dumps({"setting": asdict(setting), "summary": summary}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

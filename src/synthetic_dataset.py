"""Helpers for fully synthetic embedding-space datasets.

The synthetic dataset bypasses text embedding and writes the same artifact
shape as the real dataset preprocessors: events, sidecars, peer-doc maps, and
an embedding manifest. Vectors live on the unit sphere because the simulator's
semantic comparisons assume L2-normalized sentence-transformer embeddings.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from src.scenarios import SYNTHETIC_SUPPORTED_SCENARIOS as SYNTHETIC_SUPPORTED_SCENARIOS  # noqa: F401


SYNTHETIC_EVENTS_FILENAME = "events.csv"
SYNTHETIC_RANDOMIZED_EVENTS_FILENAME = "metadata_seeded_randomized.csv.gz"
SYNTHETIC_SEEDED_SEARCHABLE_FILENAME = "seeded_searchable_queries.csv.gz"
SYNTHETIC_PEER_FUTURE_DOCS_FILENAME = "peer_future_doc_ids.json.gz"
SYNTHETIC_PEER_FUTURE_DOCS_SUMMARY_FILENAME = "peer_future_doc_ids_summary.json"
SYNTHETIC_DRIFT_OWNER_INJECTIONS_FILENAME = "drift_owner_injections.json.gz"
SYNTHETIC_DOC_TOPICS_FILENAME = "doc_topics.csv.gz"
SYNTHETIC_QUERY_TOPICS_FILENAME = "query_topics.csv.gz"
SYNTHETIC_TOPICS_FILENAME = "topics.npy"
SYNTHETIC_SIGMA_SWEEP_FILENAME = "sigma_sweep.json"
SYNTHETIC_PRECOMPUTE_SUMMARY_FILENAME = "synthetic_precompute_summary.json"


DEFAULT_TOPIC_COUNT = 200
DEFAULT_DOCS_PER_TOPIC = 50
DEFAULT_USER_COUNT = 20_000
DEFAULT_QUERIES_PER_USER = 25
DEFAULT_DIRICHLET_ALPHA = 0.01
DEFAULT_EMBEDDING_DIM = 768
DEFAULT_RANDOM_SEED = 42
DEFAULT_SIGMA_SWEEP = (0.01, 0.02, 0.03, 0.05, 0.075, 0.10)
DEFAULT_MIN_TARGET_COUNT = 1
DEFAULT_TOP_K = 10


@dataclass(frozen=True, slots=True)
class SyntheticConfig:
    topic_count: int = DEFAULT_TOPIC_COUNT
    docs_per_topic: int = DEFAULT_DOCS_PER_TOPIC
    user_count: int = DEFAULT_USER_COUNT
    queries_per_user: int = DEFAULT_QUERIES_PER_USER
    dirichlet_alpha: float = DEFAULT_DIRICHLET_ALPHA
    embedding_dim: int = DEFAULT_EMBEDDING_DIM
    sigma: float | None = None
    random_seed: int = DEFAULT_RANDOM_SEED
    min_target_count: int = DEFAULT_MIN_TARGET_COUNT

    @property
    def doc_count(self) -> int:
        return self.topic_count * self.docs_per_topic

    @property
    def query_count(self) -> int:
        return self.user_count * self.queries_per_user


@dataclass(frozen=True, slots=True)
class NearestDocStats:
    nearest_indices: np.ndarray
    nearest_similarities: np.ndarray
    cross_topic_rate: float
    same_topic_top_k_rate: float


def doc_id_for_index(index: int, *, width: int) -> str:
    return f"doc_{index:0{width}d}"


def query_key_for_id(qid: int, *, width: int) -> str:
    return f"q_{qid:0{width}d}"


def fixed_width_for_count(count: int, *, minimum: int = 6) -> int:
    return max(minimum, len(str(max(0, count - 1))))


def l2_normalize(vectors: np.ndarray, *, eps: float = 1e-12) -> np.ndarray:
    """Return row-wise L2-normalized float32 vectors."""

    arr = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return (arr / norms).astype(np.float32, copy=False)


def sample_unit_vectors(
    rng: np.random.Generator,
    *,
    count: int,
    embedding_dim: int,
) -> np.ndarray:
    vectors = rng.standard_normal(size=(count, embedding_dim)).astype(np.float32)
    return l2_normalize(vectors)


def sample_noisy_vectors(
    rng: np.random.Generator,
    *,
    centroids: np.ndarray,
    topic_indices: np.ndarray,
    sigma: float,
) -> np.ndarray:
    base = np.asarray(centroids, dtype=np.float32)[np.asarray(topic_indices, dtype=np.int64)]
    if sigma <= 0:
        return np.array(base, dtype=np.float32, copy=True)
    noise = rng.normal(loc=0.0, scale=sigma, size=base.shape).astype(np.float32)
    return l2_normalize(base + noise)


def batched_nearest_indices(
    queries: np.ndarray,
    candidates: np.ndarray,
    *,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return nearest candidate index and cosine similarity for each query."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    query_arr = np.asarray(queries, dtype=np.float32)
    candidate_arr = np.asarray(candidates, dtype=np.float32)
    candidate_t = candidate_arr.T
    nearest_indices = np.empty(query_arr.shape[0], dtype=np.int32)
    nearest_similarities = np.empty(query_arr.shape[0], dtype=np.float32)
    for start in range(0, query_arr.shape[0], batch_size):
        end = min(start + batch_size, query_arr.shape[0])
        sims = query_arr[start:end] @ candidate_t
        local_indices = np.argmax(sims, axis=1)
        nearest_indices[start:end] = local_indices.astype(np.int32, copy=False)
        nearest_similarities[start:end] = sims[
            np.arange(end - start), local_indices
        ].astype(np.float32, copy=False)
    return nearest_indices, nearest_similarities


def nearest_doc_stats(
    queries: np.ndarray,
    *,
    query_topics: np.ndarray,
    docs: np.ndarray,
    doc_topics: np.ndarray,
    batch_size: int,
    top_k: int = DEFAULT_TOP_K,
) -> NearestDocStats:
    if top_k <= 0:
        raise ValueError("top_k must be positive")

    query_arr = np.asarray(queries, dtype=np.float32)
    doc_arr = np.asarray(docs, dtype=np.float32)
    query_topic_arr = np.asarray(query_topics, dtype=np.int32)
    doc_topic_arr = np.asarray(doc_topics, dtype=np.int32)
    doc_t = doc_arr.T
    k = min(top_k, doc_arr.shape[0])

    nearest_indices = np.empty(query_arr.shape[0], dtype=np.int32)
    nearest_similarities = np.empty(query_arr.shape[0], dtype=np.float32)
    cross_topic_count = 0
    same_topic_top_k_count = 0

    for start in range(0, query_arr.shape[0], batch_size):
        end = min(start + batch_size, query_arr.shape[0])
        sims = query_arr[start:end] @ doc_t
        local_indices = np.argmax(sims, axis=1)
        batch_topics = query_topic_arr[start:end]
        nearest_indices[start:end] = local_indices.astype(np.int32, copy=False)
        nearest_similarities[start:end] = sims[
            np.arange(end - start), local_indices
        ].astype(np.float32, copy=False)
        cross_topic_count += int(np.sum(doc_topic_arr[local_indices] != batch_topics))

        if k == doc_arr.shape[0]:
            top_indices = np.argsort(sims, axis=1)[:, -k:]
        else:
            top_indices = np.argpartition(sims, -k, axis=1)[:, -k:]
        same_topic_top_k_count += int(
            np.sum(np.any(doc_topic_arr[top_indices] == batch_topics[:, None], axis=1))
        )

    total = max(1, query_arr.shape[0])
    return NearestDocStats(
        nearest_indices=nearest_indices,
        nearest_similarities=nearest_similarities,
        cross_topic_rate=cross_topic_count / total,
        same_topic_top_k_rate=same_topic_top_k_count / total,
    )


def percentile_summary(values: Sequence[float] | np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float32)
    if arr.size == 0:
        return {
            "min": 0.0,
            "p05": 0.0,
            "p25": 0.0,
            "median": 0.0,
            "p75": 0.0,
            "p95": 0.0,
            "max": 0.0,
            "mean": 0.0,
        }
    return {
        "min": float(np.min(arr)),
        "p05": float(np.percentile(arr, 5)),
        "p25": float(np.percentile(arr, 25)),
        "median": float(np.percentile(arr, 50)),
        "p75": float(np.percentile(arr, 75)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
    }


def parse_sigma_values(raw_values: str | Iterable[float]) -> tuple[float, ...]:
    if isinstance(raw_values, str):
        values = [part.strip() for part in raw_values.split(",") if part.strip()]
        if not values:
            raise ValueError("sigma list must contain at least one value")
        return tuple(float(value) for value in values)
    parsed = tuple(float(value) for value in raw_values)
    if not parsed:
        raise ValueError("sigma list must contain at least one value")
    return parsed


def ensure_dataset_dir(path: str | Path) -> Path:
    data_dir = Path(path)
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir

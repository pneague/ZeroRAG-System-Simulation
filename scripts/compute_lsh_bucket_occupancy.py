#!/usr/bin/env python3
"""Compute document-only LSH bucket occupancy statistics.

This does not load query events and does not run retrieval. It hashes every
document embedding into the configured E2LSH tables and summarizes how many
documents land in each non-empty bucket.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data_root import describe_resolution, resolve_dataset_dir
from src.lsh import E2LSHFamily
from src.network import EmbeddingLookup


def _default_embedding_manifest_path(*, dataset: str, amazon_category: str | None) -> Path:
    if dataset == "amazon":
        return resolve_dataset_dir("amazon", amazon_category) / "embeddings" / "manifest.json"
    if dataset == "mind":
        return resolve_dataset_dir("mind") / "embeddings" / "manifest.json"
    if dataset == "synthetic":
        return resolve_dataset_dir("synthetic") / "embeddings" / "manifest.json"
    if dataset == "pubmedqa":
        return resolve_dataset_dir("pubmedqa") / "embeddings" / "manifest.json"
    return resolve_dataset_dir("aol4foltr") / "embeddings" / "manifest.json"


def _resolve_manifest_paths(manifest_path: Path) -> tuple[Path, Path, dict[str, Any]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_dir = manifest_path.parent

    def resolve(value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else manifest_dir / path

    return resolve(manifest["doc_keys_path"]), resolve(manifest["doc_embeddings_path"]), manifest


def _percentiles(values: np.ndarray, points: tuple[float, ...]) -> dict[str, float]:
    if values.size == 0:
        return {f"p{point:g}": 0.0 for point in points}
    quantiles = np.percentile(values, points)
    return {f"p{point:g}": float(value) for point, value in zip(points, quantiles)}


def _open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def _load_doc_filter(path: Path) -> set[str]:
    """Load a doc-id filter from a JSON/JSON.GZ mapping or list.

    The seeded AOL peer-doc artifact is a JSON object mapping peer ids to lists
    of doc ids. We also accept a top-level list for ad hoc filters.
    """

    with _open_text(path) as handle:
        payload = json.load(handle)
    doc_ids: set[str] = set()
    if isinstance(payload, dict):
        for value in payload.values():
            if isinstance(value, list):
                doc_ids.update(str(doc_id) for doc_id in value)
            elif isinstance(value, dict):
                for nested_value in value.values():
                    if isinstance(nested_value, list):
                        doc_ids.update(str(doc_id) for doc_id in nested_value)
    elif isinstance(payload, list):
        doc_ids.update(str(doc_id) for doc_id in payload)
    else:
        raise ValueError(f"Unsupported doc filter payload type in {path}: {type(payload).__name__}")
    return doc_ids


def compute_occupancy(
    *,
    manifest_path: Path,
    output_dir: Path,
    dataset: str,
    amazon_category: str | None,
    w: float,
    k: int,
    L: int,
    random_seed: int,
    chunk_size: int,
    doc_filter_path: Path | None,
) -> dict[str, Any]:
    doc_keys_path, doc_embeddings_path, manifest = _resolve_manifest_paths(manifest_path)
    doc_lookup = EmbeddingLookup(keys_path=doc_keys_path, embeddings_path=doc_embeddings_path)
    embeddings = doc_lookup.embeddings
    embedding_total_docs = int(embeddings.shape[0])
    embedding_dim = int(embeddings.shape[1])
    doc_filter_count = None
    missing_filter_doc_count = 0

    if doc_filter_path is not None:
        doc_filter = _load_doc_filter(doc_filter_path)
        doc_filter_count = len(doc_filter)
        selected_indices = sorted(
            doc_lookup.index[str(doc_id)] for doc_id in doc_filter if str(doc_id) in doc_lookup.index
        )
        missing_filter_doc_count = doc_filter_count - len(selected_indices)
        if not selected_indices:
            raise ValueError(f"Doc filter {doc_filter_path} did not match any embedding keys")
        selected_indices_array = np.asarray(selected_indices, dtype=np.int64)
    else:
        selected_indices_array = np.arange(embedding_total_docs, dtype=np.int64)

    total_docs = int(selected_indices_array.shape[0])

    family = E2LSHFamily.build(d=embedding_dim, k=k, L=L, w=w, seed=random_seed)
    table_counts: list[Counter[tuple[int, ...]]] = [Counter() for _ in range(L)]

    for start in range(0, total_docs, chunk_size):
        stop = min(start + chunk_size, total_docs)
        chunk_indices = selected_indices_array[start:stop]
        bucket_ids = family.hash_tables_batch(np.asarray(embeddings[chunk_indices], dtype=np.float32))
        for table_id in range(L):
            unique_buckets, counts = np.unique(
                bucket_ids[:, table_id, :],
                axis=0,
                return_counts=True,
            )
            table_counter = table_counts[table_id]
            for bucket_id, count in zip(unique_buckets, counts):
                table_counter[tuple(int(value) for value in bucket_id.tolist())] += int(count)
        print(
            f"[bucket-occupancy] hashed docs {stop:,}/{total_docs:,} "
            f"for w={w:g} k={k} L={L}",
            flush=True,
        )

    per_table_rows: list[dict[str, Any]] = []
    all_bucket_counts: list[int] = []
    for table_id, table_counter in enumerate(table_counts):
        counts = np.asarray(list(table_counter.values()), dtype=np.int64)
        all_bucket_counts.extend(int(value) for value in counts.tolist())
        percentile_values = _percentiles(counts, (50, 90, 95, 99))
        per_table_rows.append(
            {
                "table_id": table_id,
                "nonempty_bucket_count": int(counts.size),
                "mean_bucket_docs": float(counts.mean()) if counts.size else 0.0,
                "median_bucket_docs": percentile_values["p50"],
                "p90_bucket_docs": percentile_values["p90"],
                "p95_bucket_docs": percentile_values["p95"],
                "p99_bucket_docs": percentile_values["p99"],
                "max_bucket_docs": int(counts.max()) if counts.size else 0,
            }
        )

    all_counts = np.asarray(all_bucket_counts, dtype=np.int64)
    percentile_values = _percentiles(all_counts, (50, 75, 90, 95, 99, 99.9))
    bucket_size_histogram = Counter(int(value) for value in all_bucket_counts)
    output_dir.mkdir(parents=True, exist_ok=True)
    label = f"{dataset}_w{w:g}_k{k}_L{L}"
    safe_label = re.sub(r"[^A-Za-z0-9_.-]+", "_", label)
    summary_json_path = output_dir / f"{safe_label}_bucket_occupancy_summary.json"
    summary_csv_path = output_dir / f"{safe_label}_bucket_occupancy_summary.csv"
    histogram_csv_path = output_dir / f"{safe_label}_bucket_size_histogram.csv"
    table_csv_path = output_dir / f"{safe_label}_per_table_bucket_occupancy.csv"

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": dataset,
        "amazon_category": amazon_category,
        "embedding_manifest": str(manifest_path),
        "doc_keys_path": str(doc_keys_path),
        "doc_embeddings_path": str(doc_embeddings_path),
        "embedding_manifest_metadata": manifest,
        "lsh_config": {"w": w, "k": k, "L": L, "random_seed": random_seed},
        "doc_filter_path": str(doc_filter_path) if doc_filter_path is not None else None,
        "doc_filter_count": doc_filter_count,
        "missing_filter_doc_count": missing_filter_doc_count,
        "embedding_total_docs": embedding_total_docs,
        "total_docs": total_docs,
        "embedding_dim": embedding_dim,
        "table_count": L,
        "nonempty_bucket_count_total_across_tables": int(all_counts.size),
        "mean_nonempty_buckets_per_table": float(all_counts.size / L) if L else 0.0,
        "mean_bucket_docs": float(all_counts.mean()) if all_counts.size else 0.0,
        "median_bucket_docs": percentile_values["p50"],
        "p75_bucket_docs": percentile_values["p75"],
        "p90_bucket_docs": percentile_values["p90"],
        "p95_bucket_docs": percentile_values["p95"],
        "p99_bucket_docs": percentile_values["p99"],
        "p99_9_bucket_docs": percentile_values["p99.9"],
        "max_bucket_docs": int(all_counts.max()) if all_counts.size else 0,
        "mean_bucket_pct_of_docs": (
            100.0 * float(all_counts.mean()) / total_docs if all_counts.size and total_docs else 0.0
        ),
        "median_bucket_pct_of_docs": 100.0 * percentile_values["p50"] / total_docs if total_docs else 0.0,
        "p95_bucket_pct_of_docs": 100.0 * percentile_values["p95"] / total_docs if total_docs else 0.0,
        "p99_bucket_pct_of_docs": 100.0 * percentile_values["p99"] / total_docs if total_docs else 0.0,
        "max_bucket_pct_of_docs": (
            100.0 * float(all_counts.max()) / total_docs if all_counts.size and total_docs else 0.0
        ),
        "summary_json_path": str(summary_json_path),
        "summary_csv_path": str(summary_csv_path),
        "histogram_csv_path": str(histogram_csv_path),
        "per_table_csv_path": str(table_csv_path),
    }

    summary_json_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    with summary_csv_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "dataset",
            "amazon_category",
            "w",
            "k",
            "L",
            "random_seed",
            "total_docs",
            "nonempty_bucket_count_total_across_tables",
            "mean_nonempty_buckets_per_table",
            "mean_bucket_docs",
            "median_bucket_docs",
            "p75_bucket_docs",
            "p90_bucket_docs",
            "p95_bucket_docs",
            "p99_bucket_docs",
            "p99_9_bucket_docs",
            "max_bucket_docs",
            "mean_bucket_pct_of_docs",
            "median_bucket_pct_of_docs",
            "p95_bucket_pct_of_docs",
            "p99_bucket_pct_of_docs",
            "max_bucket_pct_of_docs",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        row = {key: summary.get(key) for key in fieldnames}
        row.update(summary["lsh_config"])
        writer.writerow(row)

    with histogram_csv_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["bucket_docs", "bucket_pct_of_docs", "bucket_count", "bucket_fraction"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        denominator = int(all_counts.size) or 1
        for bucket_docs, bucket_count in sorted(bucket_size_histogram.items()):
            writer.writerow(
                {
                    "bucket_docs": bucket_docs,
                    "bucket_pct_of_docs": 100.0 * bucket_docs / total_docs if total_docs else 0.0,
                    "bucket_count": bucket_count,
                    "bucket_fraction": bucket_count / denominator,
                }
            )

    with table_csv_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "table_id",
            "nonempty_bucket_count",
            "mean_bucket_docs",
            "median_bucket_docs",
            "p90_bucket_docs",
            "p95_bucket_docs",
            "p99_bucket_docs",
            "max_bucket_docs",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(per_table_rows)

    print(json.dumps(summary, indent=2, default=str))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute full-document LSH bucket occupancy.")
    parser.add_argument("--dataset", default="aol4foltr")
    parser.add_argument("--amazon-category", default=os.environ.get("LOW_COORD_DHT_AMAZON_CATEGORY"))
    parser.add_argument("--embedding-manifest", default=None)
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "results" / "lsh_bucket_occupancy"),
    )
    parser.add_argument("--lsh-w", type=float, required=True)
    parser.add_argument("--lsh-k", type=int, required=True)
    parser.add_argument("--lsh-L", type=int, required=True)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--chunk-size", type=int, default=8192)
    parser.add_argument(
        "--doc-filter-json-gz",
        default=None,
        help=(
            "Optional JSON/JSON.GZ doc-id filter. For seeded AOL, pass "
            "peer_future_doc_ids.json.gz to restrict occupancy to seeded owned docs."
        ),
    )
    args = parser.parse_args()

    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be > 0")

    amazon_category = args.amazon_category if args.dataset == "amazon" else None
    manifest_path = Path(args.embedding_manifest) if args.embedding_manifest else _default_embedding_manifest_path(
        dataset=args.dataset,
        amazon_category=amazon_category,
    )
    print(describe_resolution(args.dataset, amazon_category))
    compute_occupancy(
        manifest_path=manifest_path,
        output_dir=Path(args.output_dir),
        dataset=args.dataset,
        amazon_category=amazon_category,
        w=args.lsh_w,
        k=args.lsh_k,
        L=args.lsh_L,
        random_seed=args.random_seed,
        chunk_size=args.chunk_size,
        doc_filter_path=Path(args.doc_filter_json_gz) if args.doc_filter_json_gz else None,
    )


if __name__ == "__main__":
    main()

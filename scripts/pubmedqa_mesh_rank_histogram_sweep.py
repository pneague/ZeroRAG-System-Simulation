#!/usr/bin/env python3
"""Sample PubMedQA peers over all MeSH terms and plot topic-rank histograms."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from plot_topic_rank_histogram import write_topic_rank_histogram  # noqa: E402
from pubmedqa_p80_batch_retrieval import PUBMEDQA_CONFIGS, _clean_str  # noqa: E402


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")


def _alpha_label(alpha: float) -> str:
    text = f"{alpha:.10f}".rstrip("0").rstrip(".")
    return "alpha_" + text.replace(".", "p")


def _load_mesh_doc_pools(data_root: Path) -> tuple[list[str], list[list[str]], dict[str, Any]]:
    from datasets import load_from_disk

    doc_pools: dict[str, list[str]] = {}
    config_counts: dict[str, int] = {}
    row_count = 0
    rows_with_meshes = 0
    for config in PUBMEDQA_CONFIGS:
        dataset = load_from_disk(str(data_root / config))
        config_counts[config] = len(dataset)
        for row in dataset:
            row_count += 1
            pubid = _clean_str(row.get("pubid"))
            meshes = sorted(
                {
                    _clean_str(mesh)
                    for mesh in ((row.get("context") or {}).get("meshes") or [])
                    if _clean_str(mesh)
                }
            )
            if not pubid or not meshes:
                continue
            rows_with_meshes += 1
            for mesh in meshes:
                doc_pools.setdefault(mesh, []).append(pubid)

    topics = sorted(doc_pools)
    pools = [doc_pools[topic] for topic in topics]
    summary = {
        "data_root": str(data_root),
        "configs": config_counts,
        "row_count": row_count,
        "rows_with_meshes": rows_with_meshes,
        "unique_mesh_terms": len(topics),
        "mesh_doc_frequency_min": min((len(pool) for pool in pools), default=0),
        "mesh_doc_frequency_max": max((len(pool) for pool in pools), default=0),
        "top_mesh_doc_frequencies": [
            {"mesh_term": mesh, "doc_frequency": int(count)}
            for mesh, count in Counter({topic: len(pool) for topic, pool in zip(topics, pools, strict=True)}).most_common(20)
        ],
    }
    return topics, pools, summary


def _write_sampled_query_topics(
    *,
    path: Path,
    topics: list[str],
    doc_pools: list[list[str]],
    peer_count: int,
    queries_per_peer: int,
    alpha: float,
    random_seed: int,
) -> dict[str, Any]:
    if peer_count <= 0:
        raise ValueError("peer_count must be positive")
    if queries_per_peer <= 0:
        raise ValueError("queries_per_peer must be positive")
    if alpha <= 0:
        raise ValueError("alpha must be positive")
    if not topics:
        raise ValueError("no MeSH topics available")

    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(random_seed)
    topic_count = len(topics)
    total_alpha = topic_count * alpha
    topic_counts: Counter[int] = Counter()
    unique_topics_per_peer: list[int] = []
    qid = 0
    with gzip.open(path, "wt", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
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
            ),
        )
        writer.writeheader()
        for user_id in range(peer_count):
            # Collapsed Dirichlet-multinomial sampling avoids materializing a full
            # topic probability vector for each peer while preserving the same law.
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
                    local_probs = local_counts / local_counts.sum()
                    topic_id = int(rng.choice(local_topic_ids, p=local_probs))
                else:
                    topic_id = int(rng.integers(0, topic_count))
                peer_topic_counts[topic_id] += 1
                drawn_topics.append(topic_id)
            unique_topics_per_peer.append(len(set(drawn_topics)))
            for topic_id in drawn_topics:
                pool = doc_pools[int(topic_id)]
                target_doc_id = pool[int(rng.integers(0, len(pool)))]
                topic_counts[int(topic_id)] += 1
                writer.writerow(
                    {
                        "qid": qid,
                        "user_id": user_id,
                        "query": f"mesh_sample_q_{qid:06d}",
                        "generated_topic": int(topic_id),
                        "mesh_term": topics[int(topic_id)],
                        "target_doc_id": target_doc_id,
                        "target_doc_topic": int(topic_id),
                        "query_variant": "original",
                        "source_selected_index": "",
                        "searchable": "true",
                    }
                )
                qid += 1

    return {
        "alpha": alpha,
        "random_seed": random_seed,
        "peer_count": peer_count,
        "queries_per_peer": queries_per_peer,
        "query_count": qid,
        "query_topics_path": str(path),
        "avg_unique_topics_per_peer": float(np.mean(unique_topics_per_peer)) if unique_topics_per_peer else 0.0,
        "top_sampled_topics": [
            {"topic_id": topic_id, "mesh_term": topics[topic_id], "query_count": int(count)}
            for topic_id, count in topic_counts.most_common(20)
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / "data" / "pubmedqa_rag" / "hf")
    parser.add_argument("--temp-dir", type=Path, required=True)
    parser.add_argument("--figure-root", type=Path, required=True)
    parser.add_argument("--peer-count", type=int, default=50_000)
    parser.add_argument("--queries-per-peer", type=int, default=10)
    parser.add_argument("--alphas", default="0.001,0.0005,0.00001")
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--keep-temp", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    alphas = [float(value.strip()) for value in args.alphas.split(",") if value.strip()]
    if not alphas:
        raise SystemExit("--alphas must contain at least one value")

    args.temp_dir.mkdir(parents=True, exist_ok=True)
    args.figure_root.mkdir(parents=True, exist_ok=True)
    topics, doc_pools, mesh_summary = _load_mesh_doc_pools(args.data_root)
    run_summaries = []
    try:
        for offset, alpha in enumerate(alphas):
            label = _alpha_label(alpha)
            query_topics_path = args.temp_dir / f"pubmedqa_all_mesh_{label}_query_topics.csv.gz"
            sample_summary = _write_sampled_query_topics(
                path=query_topics_path,
                topics=topics,
                doc_pools=doc_pools,
                peer_count=args.peer_count,
                queries_per_peer=args.queries_per_peer,
                alpha=alpha,
                random_seed=args.random_seed + offset,
            )
            output_prefix = args.figure_root / f"pubmedqa_all_mesh_topic_rank_histogram_{label}"
            histogram_summary = write_topic_rank_histogram(
                query_topics_path=query_topics_path,
                output_prefix=output_prefix,
                title=f"PubMedQA all MeSH: topic-rank histogram (alpha={alpha:g})",
            )
            run_summaries.append(
                {
                    **sample_summary,
                    "histogram_csv_path": histogram_summary["csv_path"],
                    "histogram_png_path": histogram_summary["png_path"],
                    "histogram_pdf_path": histogram_summary["pdf_path"],
                    "histogram_summary_path": str(
                        output_prefix.with_name(output_prefix.name + "_summary.json")
                    ),
                    "histogram_avg_nonempty_topics_per_peer": histogram_summary[
                        "avg_nonempty_topics_per_peer"
                    ],
                }
            )
    finally:
        if not args.keep_temp and args.temp_dir.exists():
            shutil.rmtree(args.temp_dir)

    summary = {
        "mesh_summary": mesh_summary,
        "peer_count": args.peer_count,
        "queries_per_peer": args.queries_per_peer,
        "alphas": alphas,
        "temp_dir_deleted": not args.keep_temp,
        "runs": run_summaries,
    }
    summary_path = args.figure_root / "pubmedqa_all_mesh_topic_rank_histogram_sweep_summary.json"
    _write_json(summary_path, summary)
    print(json.dumps({"summary_path": str(summary_path), "runs": run_summaries}, indent=2), flush=True)


if __name__ == "__main__":
    main()

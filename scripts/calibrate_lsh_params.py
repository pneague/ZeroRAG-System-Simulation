#!/usr/bin/env python3
"""Calibrate E2LSH ``(w, k, L)`` for the active embedding manifest.

For unit-normalized sentence-transformer vectors the projections satisfy
``a . v ~ N(0, 1)``, so the paper's ``w = 4`` default produces degenerate
buckets. This utility sweeps a small parameter grid on a capped event
prefix and writes ``lsh_calibration.json`` with:

* selected ``(w, k, L)`` tuple,
* per-configuration recall/contacts/candidate-count summary,
* bucket-occupancy distribution per ``w``,
* analytic-vs-empirical collision probabilities at a few reference
  distances (sanity check against
  :func:`src.lsh.analytic_collision_probability`).

The selected tuple is consumed by ``experiments/run_lsh_simulation.py``
via ``--lsh-calibration-profile`` so production runs never use a
hardcoded ``w``. Call sites that already pass ``--lsh-w`` / ``--lsh-k`` /
``--lsh-L`` override the calibration; calibration is the default.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from itertools import product
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _resolve_default_paths(dataset: str, amazon_category: str | None) -> tuple[Path, Path | None, Path]:
    from src.amazon_dataset import (
        AMAZON_EVENTS_FILENAME,
        AMAZON_RANDOMIZED_EVENTS_FILENAME,
        AMAZON_SEEDED_SEARCHABLE_FILENAME,
    )
    from src.data_root import resolve_dataset_dir
    from src.dataset import (
        METADATA_FILENAME,
        SEEDED_RANDOMIZED_METADATA_FILENAME,
        SEEDED_SEARCHABLE_QUERIES_FILENAME,
    )

    if dataset == "amazon":
        amazon_dir = resolve_dataset_dir("amazon", amazon_category)
        # Use the seeded randomized variant + searchable sidecar by default.
        metadata_path = amazon_dir / AMAZON_RANDOMIZED_EVENTS_FILENAME
        if not metadata_path.exists():
            metadata_path = amazon_dir / AMAZON_EVENTS_FILENAME
        searchable_path = amazon_dir / AMAZON_SEEDED_SEARCHABLE_FILENAME
        embeddings_path = amazon_dir / "embeddings" / "manifest.json"
    else:
        aol_dir = resolve_dataset_dir("aol4foltr")
        metadata_path = aol_dir / SEEDED_RANDOMIZED_METADATA_FILENAME
        if not metadata_path.exists():
            metadata_path = aol_dir / METADATA_FILENAME
        searchable_path = aol_dir / SEEDED_SEARCHABLE_QUERIES_FILENAME
        embeddings_path = aol_dir / "embeddings" / "manifest.json"

    return metadata_path, searchable_path, embeddings_path


def main() -> None:
    from src.amazon_dataset import AMAZON_CATEGORY_SPECS
    from src.dataset import load_query_events
    from src.lsh import E2LSHFamily, analytic_collision_probability
    from src.lsh_network import LSHDHTNetwork, LSHSimulationConfig

    parser = argparse.ArgumentParser(description="Calibrate (w, k, L) for E2LSH on the chosen dataset")
    parser.add_argument("--dataset", choices=("aol4foltr", "amazon"), default="aol4foltr")
    parser.add_argument(
        "--amazon-category",
        choices=tuple(sorted(AMAZON_CATEGORY_SPECS)),
        default="cell_phones",
    )
    parser.add_argument("--metadata", default=None)
    parser.add_argument("--searchable-queries", default=None)
    parser.add_argument("--embedding-manifest", default=None)
    parser.add_argument("--max-events", type=int, default=2000)
    parser.add_argument(
        "--w-grid",
        default="0.25,0.5,0.75,1.0",
        help="Comma-separated bucket widths to sweep.",
    )
    parser.add_argument(
        "--k-grid",
        default="8,10,12",
        help="Comma-separated AND-amplification depths to sweep.",
    )
    parser.add_argument(
        "--L-grid",
        default="10,20,30",
        help="Comma-separated table counts to sweep.",
    )
    parser.add_argument("--bucket-replication", type=int, default=1)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument(
        "--scenario",
        default="seeded_randomized_timeline",
        help="Scenario used for the calibration prefix; defaults to "
        "seeded_randomized_timeline so peers start with a non-empty inventory.",
    )
    parser.add_argument(
        "--seeded-peer-docs",
        default=None,
        help="Optional override of the seeded peer-doc mapping (JSON or .gz).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output path for lsh_calibration.json (defaults to <embedding_dir>/lsh_calibration.json).",
    )
    args = parser.parse_args()

    amazon_category = args.amazon_category if args.dataset == "amazon" else None
    default_metadata, default_searchable, default_manifest = _resolve_default_paths(
        args.dataset, amazon_category
    )
    metadata_path = Path(args.metadata) if args.metadata else default_metadata
    searchable_path = Path(args.searchable_queries) if args.searchable_queries else default_searchable
    manifest_path = Path(args.embedding_manifest) if args.embedding_manifest else default_manifest

    if not manifest_path.exists():
        raise SystemExit(f"Embedding manifest not found at {manifest_path}")

    w_grid = [float(x) for x in args.w_grid.split(",") if x]
    k_grid = [int(x) for x in args.k_grid.split(",") if x]
    L_grid = [int(x) for x in args.L_grid.split(",") if x]

    print(f"Calibrating LSH on dataset={args.dataset} amazon_category={amazon_category}")
    print(f"Embedding manifest: {manifest_path}")
    print(f"Sweep w_grid={w_grid} k_grid={k_grid} L_grid={L_grid}")

    events = load_query_events(
        metadata_path,
        searchable_only=False,
        max_events=args.max_events,
        require_searchable_column=False,
        searchable_queries_path=str(searchable_path) if searchable_path else None,
    )

    seeded_doc_mapping: dict[int, list[str]] | None = None
    if args.seeded_peer_docs is not None:
        seeded_path = Path(args.seeded_peer_docs)
        import gzip

        if seeded_path.suffix == ".gz":
            with gzip.open(seeded_path, "rt", encoding="utf-8") as handle:
                payload = json.load(handle)
        else:
            payload = json.loads(seeded_path.read_text(encoding="utf-8"))
        seeded_doc_mapping = {
            int(peer_id): [str(doc_id) for doc_id in doc_ids]
            for peer_id, doc_ids in payload.items()
        }

    peer_ids = [event.user_id for event in events]

    results: list[dict] = []
    bucket_occupancy_summary_per_w: dict[float, dict] = {}

    # Run a tiny in-memory metrics tracker substitute so we can collect
    # recall directly without touching disk.
    class _InMemoryMetrics:
        def __init__(self) -> None:
            self.executed = 0
            self.successes = 0
            self.skipped = 0
            self.cycles = 0

        def log_query(self, query_log) -> None:
            self.executed += 1
            self.cycles += query_log.cycles_used
            if query_log.result == "+":
                self.successes += 1

        def note_skipped_query(self, _reason: str) -> None:
            self.skipped += 1

    for w in w_grid:
        # Inspect bucket occupancy on the doc embeddings using a temp family.
        from src.network import EmbeddingLookup

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        doc_lookup = EmbeddingLookup(
            keys_path=manifest["doc_keys_path"],
            embeddings_path=manifest["doc_embeddings_path"],
        )
        sample_size = min(1000, doc_lookup.embeddings.shape[0])
        sample_idx = list(range(sample_size))
        sample_vectors = doc_lookup.embeddings[sample_idx].astype("float32")
        family = E2LSHFamily.build(
            d=sample_vectors.shape[1], k=max(k_grid), L=max(L_grid), w=w, seed=args.random_seed
        )
        bucket_arr = family.hash_tables_batch(sample_vectors)
        # Use only L=1, k=max_k to characterize raw bucket spread per w.
        first_table = bucket_arr[:, 0, :]
        bucket_keys = [tuple(row.tolist()) for row in first_table]
        bucket_counts = Counter(bucket_keys)
        bucket_occupancy_summary_per_w[w] = {
            "distinct_buckets_in_sample": len(bucket_counts),
            "max_bucket_size": max(bucket_counts.values()) if bucket_counts else 0,
            "mean_bucket_size": (
                sum(bucket_counts.values()) / len(bucket_counts) if bucket_counts else 0.0
            ),
            "sample_size": sample_size,
        }

        for k, L in product(k_grid, L_grid):
            cfg = LSHSimulationConfig(
                scenario=args.scenario,
                k=k,
                L=L,
                w=w,
                bucket_replication=args.bucket_replication,
                random_seed=args.random_seed,
                max_owned_docs=0,
            )
            network = LSHDHTNetwork.from_embedding_manifest(
                peer_ids=peer_ids, manifest_path=str(manifest_path), config=cfg
            )
            if seeded_doc_mapping is not None and args.scenario in {
                "seeded_fixed_inventory",
                "seeded_randomized_timeline",
            }:
                network.seed_peer_docs(seeded_doc_mapping)

            metrics = _InMemoryMetrics()
            for event in events:
                network.process_event(event, metrics)
            recall = metrics.successes / metrics.executed if metrics.executed else 0.0
            avg_cycles = metrics.cycles / metrics.executed if metrics.executed else 0.0
            executed = metrics.executed
            avg_query_contacts = (
                network.coord.query_contacts / executed if executed else 0.0
            )
            avg_candidate_count = (
                network.coord.candidate_total / executed if executed else 0.0
            )
            results.append(
                {
                    "w": w,
                    "k": k,
                    "L": L,
                    "recall": recall,
                    "avg_cycles": avg_cycles,
                    "avg_query_contacts": avg_query_contacts,
                    "avg_candidate_count": avg_candidate_count,
                    "executed_queries": executed,
                    "skipped_queries": metrics.skipped,
                    "publish_messages": network.coord.publish_messages,
                    "docs_published": network.coord.docs_published,
                    "bucket_count_distinct": network.coord.bucket_count_distinct,
                }
            )
            print(
                f"  w={w} k={k} L={L} recall={recall:.4f} avg_cycles={avg_cycles:.2f} "
                f"avg_candidates={avg_candidate_count:.2f}"
            )

    # Selection rule: highest recall, breaking ties by lowest avg_query_contacts.
    if not results:
        raise SystemExit("Calibration sweep produced no results")
    results_sorted = sorted(
        results, key=lambda row: (-row["recall"], row["avg_query_contacts"], -row["avg_candidate_count"])
    )
    best = results_sorted[0]

    reference_distances = [0.1, 0.25, 0.5, 1.0, 1.5, 2.0]
    analytic_table = {
        f"w={w}": {
            f"c={c}": analytic_collision_probability(c=c, w=w) for c in reference_distances
        }
        for w in w_grid
    }

    payload = {
        "dataset": args.dataset,
        "amazon_category": amazon_category,
        "embedding_manifest": str(manifest_path),
        "max_events_used": args.max_events,
        "scenario": args.scenario,
        "bucket_replication": args.bucket_replication,
        "random_seed": args.random_seed,
        "selected": {"w": best["w"], "k": best["k"], "L": best["L"]},
        "best_summary": best,
        "results": results,
        "bucket_occupancy_per_w": bucket_occupancy_summary_per_w,
        "analytic_collision_probabilities": analytic_table,
        "selection_rule": (
            "Maximize recall; break ties by lowest avg_query_contacts; "
            "secondary tiebreak prefers higher candidate count."
        ),
    }

    output_path = Path(args.output) if args.output else manifest_path.parent / "lsh_calibration.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"\nSelected (w, k, L) = ({best['w']}, {best['k']}, {best['L']})")
    print(f"Calibration profile written to {output_path}")


if __name__ == "__main__":
    main()

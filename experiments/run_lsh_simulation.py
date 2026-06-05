#!/usr/bin/env python3
"""Run the Low_Coord_Semantic_DHT E2LSH-DHT baseline simulation.

Mirrors :mod:`experiments.run_simulation` for the LSH baseline. Reuses the
exact same dataset loaders, embedding manifests, scenario rules, and
:class:`MetricsTracker`, but routes queries through
:class:`src.lsh_network.LSHDHTNetwork` instead of the semantic peer
graph.

Output layout (compatible with the semantic baseline's tooling):

* ``query_log.jsonl`` -- per-query ``(event_index, query_id,
  querier_peer_id, clicked_doc_id, cycles_used, result, termination_reason)``;
  identical schema to the semantic runs.
* ``metrics.json`` -- rolling-window ``recall``, ``avg_cycles``,
  ``skip_rate``; identical schema.
* ``run_manifest.json`` -- adds ``algorithm``, ``lsh_config``, and
  ``metric_semantics`` blocks so analysis can disambiguate algorithms.
* ``lsh_metrics.json`` -- LSH-only coordination counters (publish
  messages, query contacts, candidate and distinct-doc counts) that don't fit the existing
  schema.
"""

from __future__ import annotations

import argparse
import fcntl
import gzip
import json
import os
import random
import socket
import subprocess
import sys
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.path_defaults import default_experiments_registry  # noqa: E402

DEFAULT_EXPERIMENTS_REGISTRY = str(default_experiments_registry())
REGISTRY_SCHEMA_VERSION = 1
ALGORITHM_NAME = "e2lsh_dht"


from src.scenarios import (  # noqa: E402
    AMAZON_SUPPORTED_SCENARIOS,
    FIFO_SCENARIOS,
    PUBMEDQA_SUPPORTED_SCENARIOS,
    RANDOMIZED_TIMELINE_SCENARIOS,
    RUNTIME_SEARCHABLE_SCENARIOS,
    SEEDED_SCENARIOS,
    SYNTHETIC_SUPPORTED_SCENARIOS,
)


def _default_metadata_path_for_scenario(
    scenario: str,
    *,
    dataset: str,
    amazon_category: str | None,
) -> Path:
    from src.amazon_dataset import (
        AMAZON_EVENTS_FILENAME,
        AMAZON_RANDOMIZED_EVENTS_FILENAME,
    )
    from src.data_root import resolve_dataset_dir
    from src.dataset import METADATA_FILENAME, SEEDED_RANDOMIZED_METADATA_FILENAME

    if dataset == "amazon":
        amazon_dir = resolve_dataset_dir("amazon", amazon_category)
        if scenario in RANDOMIZED_TIMELINE_SCENARIOS:
            return amazon_dir / AMAZON_RANDOMIZED_EVENTS_FILENAME
        return amazon_dir / AMAZON_EVENTS_FILENAME

    if dataset == "mind":
        from src.mind_dataset import MIND_EVENTS_FILENAME, MIND_RANDOMIZED_EVENTS_FILENAME

        mind_dir = resolve_dataset_dir("mind")
        if scenario in RANDOMIZED_TIMELINE_SCENARIOS:
            return mind_dir / MIND_RANDOMIZED_EVENTS_FILENAME
        return mind_dir / MIND_EVENTS_FILENAME

    if dataset == "synthetic":
        from src.synthetic_dataset import (
            SYNTHETIC_EVENTS_FILENAME,
            SYNTHETIC_RANDOMIZED_EVENTS_FILENAME,
        )

        synthetic_dir = resolve_dataset_dir("synthetic")
        if scenario in RANDOMIZED_TIMELINE_SCENARIOS:
            return synthetic_dir / SYNTHETIC_RANDOMIZED_EVENTS_FILENAME
        return synthetic_dir / SYNTHETIC_EVENTS_FILENAME

    if dataset == "pubmedqa":
        from src.pubmedqa_dataset import (
            PUBMEDQA_EVENTS_FILENAME,
            PUBMEDQA_RANDOMIZED_EVENTS_FILENAME,
        )

        pubmedqa_dir = resolve_dataset_dir("pubmedqa")
        if scenario in RANDOMIZED_TIMELINE_SCENARIOS:
            return pubmedqa_dir / PUBMEDQA_RANDOMIZED_EVENTS_FILENAME
        return pubmedqa_dir / PUBMEDQA_EVENTS_FILENAME

    aol_dir = resolve_dataset_dir("aol")
    if scenario in RANDOMIZED_TIMELINE_SCENARIOS:
        return aol_dir / SEEDED_RANDOMIZED_METADATA_FILENAME
    return aol_dir / METADATA_FILENAME


def _default_searchable_queries_path_for_scenario(
    scenario: str,
    *,
    dataset: str,
    amazon_category: str | None,
) -> Path | None:
    from src.amazon_dataset import AMAZON_SEEDED_SEARCHABLE_FILENAME
    from src.data_root import resolve_dataset_dir
    from src.dataset import (
        CHRONOLOGICAL_SEARCHABLE_QUERIES_FILENAME,
        SEEDED_SEARCHABLE_QUERIES_FILENAME,
    )

    if scenario in RUNTIME_SEARCHABLE_SCENARIOS:
        return None
    if dataset == "amazon":
        amazon_dir = resolve_dataset_dir("amazon", amazon_category)
        return amazon_dir / AMAZON_SEEDED_SEARCHABLE_FILENAME
    if dataset == "mind":
        from src.mind_dataset import MIND_SEEDED_SEARCHABLE_FILENAME

        mind_dir = resolve_dataset_dir("mind")
        if scenario == "empty_start_chronological":
            return mind_dir / CHRONOLOGICAL_SEARCHABLE_QUERIES_FILENAME
        return mind_dir / MIND_SEEDED_SEARCHABLE_FILENAME
    if dataset == "synthetic":
        from src.synthetic_dataset import SYNTHETIC_SEEDED_SEARCHABLE_FILENAME

        return resolve_dataset_dir("synthetic") / SYNTHETIC_SEEDED_SEARCHABLE_FILENAME
    if dataset == "pubmedqa":
        from src.pubmedqa_dataset import PUBMEDQA_SEEDED_SEARCHABLE_FILENAME

        return resolve_dataset_dir("pubmedqa") / PUBMEDQA_SEEDED_SEARCHABLE_FILENAME
    aol_dir = resolve_dataset_dir("aol")
    if scenario == "empty_start_chronological":
        return aol_dir / CHRONOLOGICAL_SEARCHABLE_QUERIES_FILENAME
    return aol_dir / SEEDED_SEARCHABLE_QUERIES_FILENAME


def _default_embedding_manifest_path(*, dataset: str, amazon_category: str | None) -> Path:
    from src.data_root import resolve_dataset_dir

    if dataset == "amazon":
        return resolve_dataset_dir("amazon", amazon_category) / "embeddings" / "manifest.json"
    if dataset == "mind":
        return resolve_dataset_dir("mind") / "embeddings" / "manifest.json"
    if dataset == "synthetic":
        return resolve_dataset_dir("synthetic") / "embeddings" / "manifest.json"
    if dataset == "pubmedqa":
        return resolve_dataset_dir("pubmedqa") / "embeddings" / "manifest.json"
    return resolve_dataset_dir("aol") / "embeddings" / "manifest.json"


def _default_seeded_peer_docs_path(*, dataset: str, amazon_category: str | None) -> Path:
    from src.amazon_dataset import AMAZON_PEER_FUTURE_DOCS_FILENAME
    from src.data_root import resolve_dataset_dir

    if dataset == "amazon":
        return resolve_dataset_dir("amazon", amazon_category) / AMAZON_PEER_FUTURE_DOCS_FILENAME
    if dataset == "mind":
        from src.mind_dataset import MIND_PEER_FUTURE_DOCS_FILENAME

        return resolve_dataset_dir("mind") / MIND_PEER_FUTURE_DOCS_FILENAME
    if dataset == "synthetic":
        from src.synthetic_dataset import SYNTHETIC_PEER_FUTURE_DOCS_FILENAME

        return resolve_dataset_dir("synthetic") / SYNTHETIC_PEER_FUTURE_DOCS_FILENAME
    if dataset == "pubmedqa":
        from src.pubmedqa_dataset import PUBMEDQA_PEER_FUTURE_DOCS_FILENAME

        return resolve_dataset_dir("pubmedqa") / PUBMEDQA_PEER_FUTURE_DOCS_FILENAME
    return resolve_dataset_dir("aol") / "peer_future_doc_ids.json.gz"


def _default_relevant_docs_by_qid_path(*, dataset: str, amazon_category: str | None) -> Path | None:
    if dataset != "amazon":
        return None
    from src.data_root import resolve_dataset_dir
    from src.dataset import RELEVANT_DOC_IDS_BY_QID_FILENAME

    return resolve_dataset_dir("amazon", amazon_category) / RELEVANT_DOC_IDS_BY_QID_FILENAME


def _git_commit_or_none() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent.parent,
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    commit = result.stdout.strip()
    return commit or None


def _append_to_experiments_registry(registry_path: Path, record: dict) -> None:
    """Identical body to the semantic runner so both algorithms share the
    central registry envelope."""

    try:
        registry_path.parent.mkdir(parents=True, exist_ok=True)
        with open(registry_path, "a+", encoding="utf-8") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            try:
                handle.seek(0)
                raw = handle.read().strip()
                payload: dict
                if raw:
                    try:
                        payload = json.loads(raw)
                        if not isinstance(payload, dict):
                            raise ValueError("registry root is not an object")
                    except (json.JSONDecodeError, ValueError) as exc:
                        backup_path = registry_path.with_suffix(
                            registry_path.suffix + f".bak.{int(time.time())}"
                        )
                        try:
                            backup_path.write_text(raw, encoding="utf-8")
                        except OSError:
                            pass
                        print(
                            f"[registry] corrupted registry quarantined to {backup_path}: {exc}",
                            file=sys.stderr,
                        )
                        payload = {"schema_version": REGISTRY_SCHEMA_VERSION, "runs": []}
                else:
                    payload = {"schema_version": REGISTRY_SCHEMA_VERSION, "runs": []}

                payload.setdefault("schema_version", REGISTRY_SCHEMA_VERSION)
                runs = payload.setdefault("runs", [])
                if not isinstance(runs, list):
                    runs = []
                    payload["runs"] = runs
                runs.append(record)

                handle.seek(0)
                handle.truncate()
                json.dump(payload, handle, indent=2, default=str)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)
    except Exception as exc:
        print(
            f"[registry] failed to append run record to {registry_path}: {exc}",
            file=sys.stderr,
        )
        traceback.print_exc(file=sys.stderr)


def _load_seeded_doc_mapping(path: str | Path) -> dict[int, list[str]]:
    mapping_path = Path(path)
    if mapping_path.suffix == ".gz":
        with gzip.open(mapping_path, "rt", encoding="utf-8") as handle:
            payload = json.load(handle)
    else:
        payload = json.loads(mapping_path.read_text(encoding="utf-8"))
    return {int(peer_id): [str(doc_id) for doc_id in doc_ids] for peer_id, doc_ids in payload.items()}


def _sample_distinct_candidate_event_indices(
    *,
    events,
    query_lookup,
    scenario: str,
    sample_size: int,
    random_seed: int,
) -> set[int]:
    """Pick a deterministic bounded sample for exact candidate-union counting.

    The seeded LSH sweeps use sidecar searchability, so executable queries can
    be identified before the event loop. Runtime-searchable scenarios may skip
    additional events based on evolving ownership; sampling from the superset is
    still bounded and deterministic.
    """

    if sample_size <= 0:
        return set()

    if scenario in RUNTIME_SEARCHABLE_SCENARIOS:
        eligible = [
            int(event.event_index)
            for event in events
            if event.query_text in query_lookup
        ]
    else:
        eligible = [
            int(event.event_index)
            for event in events
            if event.searchable and event.query_text in query_lookup
        ]
    if len(eligible) <= sample_size:
        return set(eligible)
    rng = random.Random(random_seed)
    return set(rng.sample(eligible, sample_size))


def _load_calibration_profile(profile_path: str | None) -> dict | None:
    if not profile_path:
        return None
    raw = Path(profile_path).read_text(encoding="utf-8")
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError(
            f"Calibration profile at {profile_path} must be a JSON object, got {type(payload).__name__}"
        )
    return payload


def _resolve_lsh_params(
    *,
    cli_k: int | None,
    cli_L: int | None,
    cli_w: float | None,
    calibration_profile: dict | None,
) -> tuple[int, int, float, dict]:
    """CLI flags win; otherwise pull from calibration profile; otherwise fall
    back to last-resort defaults that work for normalized embeddings."""

    chosen: dict[str, object] = {}

    if calibration_profile is not None:
        chosen.update(
            {
                "from_calibration_profile": True,
                "calibration_w": calibration_profile.get("w"),
                "calibration_k": calibration_profile.get("k"),
                "calibration_L": calibration_profile.get("L"),
                "calibration_dataset": calibration_profile.get("dataset"),
                "calibration_amazon_category": calibration_profile.get("amazon_category"),
            }
        )
        prof_w = calibration_profile.get("w")
        prof_k = calibration_profile.get("k")
        prof_L = calibration_profile.get("L")
    else:
        chosen["from_calibration_profile"] = False
        prof_w = prof_k = prof_L = None

    k = cli_k if cli_k is not None else (int(prof_k) if prof_k is not None else 10)
    L = cli_L if cli_L is not None else (int(prof_L) if prof_L is not None else 20)
    if cli_w is not None:
        w = float(cli_w)
    elif prof_w is not None:
        w = float(prof_w)
    else:
        # Last-resort default tuned for unit-normalized sentence-transformer
        # embeddings (a.v ~ N(0, 1) so w in [0.25, 1.0] avoids degeneracy).
        w = 0.75

    chosen.update({"k": k, "L": L, "w": w})
    return k, L, w, chosen


def main() -> None:
    from src.amazon_dataset import AMAZON_CATEGORY_SPECS
    from src.scenarios import ALL_SCENARIOS, DEFAULT_SCENARIO

    dataset_choices = ("aol", "amazon", "mind", "synthetic", "pubmedqa")
    amazon_category_choices = tuple(sorted(AMAZON_CATEGORY_SPECS))

    parser = argparse.ArgumentParser(description="Run the Low_Coord_Semantic_DHT E2LSH baseline")
    parser.add_argument(
        "--dataset",
        choices=dataset_choices,
        default=os.environ.get("LOW_COORD_DHT_DATASET", "aol"),
    )
    parser.add_argument(
        "--amazon-category",
        choices=amazon_category_choices,
        default=os.environ.get("LOW_COORD_DHT_AMAZON_CATEGORY", "cell_phones"),
    )
    parser.add_argument("--metadata", default=None)
    parser.add_argument("--searchable-queries", default=None)
    parser.add_argument("--embedding-manifest", default=None)
    parser.add_argument("--output-dir", default=str(Path("results") / "zerorag_lsh_dht"))
    parser.add_argument("--max-events", type=int, default=0, help="0 means all events")
    parser.add_argument("--metrics-window", type=int, default=1000)
    parser.add_argument(
        "--query-log-mode",
        choices=("full", "window", "none"),
        default=os.environ.get("LOW_COORD_DHT_QUERY_LOG_MODE", "full"),
        help=(
            "Query logging mode for MetricsTracker. Use 'none' for large sweeps "
            "when only aggregate metrics are needed."
        ),
    )
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument(
        "--scenario",
        choices=ALL_SCENARIOS,
        default=DEFAULT_SCENARIO,
    )
    parser.add_argument(
        "--max-owned-docs",
        type=int,
        default=None,
        help="Maximum docs a peer can hold locally; defaults to 20 for fifo_* scenarios and unlimited otherwise",
    )
    parser.add_argument("--seeded-peer-docs", default=None)
    parser.add_argument(
        "--retrieval-mode",
        choices=("single_target", "multi_relevant"),
        default=os.environ.get("LOW_COORD_DHT_RETRIEVAL_MODE"),
    )
    parser.add_argument(
        "--relevant-docs-by-qid",
        default=os.environ.get("LOW_COORD_DHT_RELEVANT_DOCS_BY_QID"),
    )

    # LSH-only knobs
    parser.add_argument(
        "--lsh-k",
        type=int,
        default=None,
        help="Number of base hashes concatenated per table (AND amplification). "
        "Defaults to calibration profile or 10.",
    )
    parser.add_argument(
        "--lsh-L",
        type=int,
        default=None,
        help="Number of independent hash tables (OR amplification). "
        "Defaults to calibration profile or 20.",
    )
    parser.add_argument(
        "--lsh-w",
        type=float,
        default=None,
        help="Bucket width w (paper symbol). Accepts the alias --lsh-r. "
        "Defaults to calibration profile or 0.75 (tuned for L2-normalized embeddings).",
    )
    parser.add_argument(
        "--lsh-r",
        type=float,
        default=None,
        help="Alias of --lsh-w to match the CharlesLiu7/p-stable-lsh-python parameter spelling.",
    )
    parser.add_argument(
        "--bucket-replication",
        type=int,
        default=int(os.environ.get("LOW_COORD_DHT_BUCKET_REPLICATION", "1")),
    )
    parser.add_argument(
        "--candidate-rerank-limit",
        type=int,
        default=int(os.environ.get("LOW_COORD_DHT_RERANK_LIMIT", "0")),
        help="Optional diagnostic: cap on the number of candidates re-ranked exactly. "
        "0 disables and matches the candidate-membership success criterion.",
    )
    parser.add_argument(
        "--distinct-candidate-sample-size",
        type=int,
        default=int(os.environ.get("LOW_COORD_DHT_DISTINCT_CANDIDATE_SAMPLE_SIZE", "5000")),
        help=(
            "Number of executed queries sampled for exact distinct candidate-doc "
            "union size. Set 0 to disable this expensive diagnostic."
        ),
    )
    parser.add_argument(
        "--lsh-calibration-profile",
        default=os.environ.get("LOW_COORD_DHT_LSH_CALIBRATION_PROFILE"),
        help="Path to lsh_calibration.json from scripts/calibrate_lsh_params.py",
    )

    parser.add_argument(
        "--experiments-registry",
        default=os.environ.get("LOW_COORD_DHT_EXPERIMENTS_REGISTRY", DEFAULT_EXPERIMENTS_REGISTRY),
    )
    args = parser.parse_args()

    if args.lsh_w is not None and args.lsh_r is not None and float(args.lsh_w) != float(args.lsh_r):
        raise SystemExit(
            f"Conflicting --lsh-w={args.lsh_w} and --lsh-r={args.lsh_r}; provide only one."
        )
    cli_w = args.lsh_w if args.lsh_w is not None else args.lsh_r

    if args.dataset == "amazon" and args.scenario not in AMAZON_SUPPORTED_SCENARIOS:
        raise SystemExit(
            f"Amazon dataset only supports scenarios {sorted(AMAZON_SUPPORTED_SCENARIOS)}; "
            f"got --scenario={args.scenario}. Amazon qrels carry no real timestamps "
            "so chronological scenarios do not apply."
        )
    if args.dataset == "synthetic" and args.scenario not in SYNTHETIC_SUPPORTED_SCENARIOS:
        raise SystemExit(
            f"Synthetic dataset only supports scenarios {sorted(SYNTHETIC_SUPPORTED_SCENARIOS)}; "
            f"got --scenario={args.scenario}. Synthetic events have no real chronology "
            "so chronological scenarios do not apply."
        )
    if args.dataset == "pubmedqa" and args.scenario not in PUBMEDQA_SUPPORTED_SCENARIOS:
        raise SystemExit(
            f"PubMedQA dataset only supports scenarios {sorted(PUBMEDQA_SUPPORTED_SCENARIOS)}; "
            f"got --scenario={args.scenario}. PubMedQA events use synthetic timestamps "
            "so chronological scenarios do not apply."
        )

    requested_scenario = args.scenario
    if args.dataset == "amazon" and args.scenario == "seeded_fixed_inventory":
        args.scenario = "seeded_randomized_timeline"
        print(
            "[amazon] aliasing scenario seeded_fixed_inventory -> "
            "seeded_randomized_timeline (no real timeline on this dataset)"
        )
    if args.dataset == "synthetic" and args.scenario == "seeded_fixed_inventory":
        args.scenario = "seeded_randomized_timeline"
        print(
            "[synthetic] aliasing scenario seeded_fixed_inventory -> "
            "seeded_randomized_timeline (no real timeline on this dataset)"
        )
    if args.dataset == "pubmedqa" and args.scenario == "seeded_fixed_inventory":
        args.scenario = "seeded_randomized_timeline"
        print(
            "[pubmedqa] aliasing scenario seeded_fixed_inventory -> "
            "seeded_randomized_timeline (synthetic trace has no real timeline)"
        )

    amazon_category = args.amazon_category if args.dataset == "amazon" else None
    retrieval_mode = args.retrieval_mode
    if retrieval_mode is None:
        retrieval_mode = (
            "multi_relevant"
            if args.dataset == "amazon" and args.scenario in SEEDED_SCENARIOS
            else "single_target"
        )
    if retrieval_mode == "multi_relevant" and not (
        args.dataset == "amazon" and args.scenario in SEEDED_SCENARIOS
    ):
        raise SystemExit(
            "retrieval_mode=multi_relevant is only supported for Amazon seeded runs"
        )

    run_id = str(uuid.uuid4())
    started_at = datetime.now(timezone.utc)
    started_at_iso = started_at.isoformat()

    from tqdm import tqdm

    from src.data_root import describe_resolution
    from src.dataset import load_query_events, load_relevant_doc_ids_by_qid
    from src.lsh_network import LSHDHTNetwork, LSHSimulationConfig
    from src.metrics import MetricsTracker

    metadata_path = args.metadata or str(
        _default_metadata_path_for_scenario(
            args.scenario, dataset=args.dataset, amazon_category=amazon_category
        )
    )
    default_searchable_queries_path = _default_searchable_queries_path_for_scenario(
        args.scenario, dataset=args.dataset, amazon_category=amazon_category
    )
    searchable_queries_path = args.searchable_queries or (
        str(default_searchable_queries_path) if default_searchable_queries_path is not None else None
    )
    if retrieval_mode == "multi_relevant":
        searchable_queries_path = None
    embedding_manifest_path = args.embedding_manifest or str(
        _default_embedding_manifest_path(dataset=args.dataset, amazon_category=amazon_category)
    )
    seeded_peer_docs_path = args.seeded_peer_docs or str(
        _default_seeded_peer_docs_path(dataset=args.dataset, amazon_category=amazon_category)
    )
    relevant_docs_by_qid_path = args.relevant_docs_by_qid or (
        str(_default_relevant_docs_by_qid_path(dataset=args.dataset, amazon_category=amazon_category))
        if retrieval_mode == "multi_relevant"
        else None
    )
    print(describe_resolution(args.dataset, amazon_category))

    searchable_mode = (
        "runtime"
        if args.scenario in RUNTIME_SEARCHABLE_SCENARIOS or retrieval_mode == "multi_relevant"
        else "sidecar"
    )
    max_owned_docs = args.max_owned_docs if args.max_owned_docs is not None else (
        20 if args.scenario in FIFO_SCENARIOS else 0
    )

    calibration_profile = _load_calibration_profile(args.lsh_calibration_profile)
    k, L, w, lsh_param_provenance = _resolve_lsh_params(
        cli_k=args.lsh_k, cli_L=args.lsh_L, cli_w=cli_w, calibration_profile=calibration_profile
    )

    events = load_query_events(
        metadata_path,
        searchable_only=False,
        max_events=args.max_events or None,
        require_searchable_column=searchable_mode == "sidecar",
        searchable_queries_path=searchable_queries_path,
    )
    if searchable_mode == "runtime":
        print(f"Loaded {len(events):,} events from {metadata_path} with runtime searchability checks")
    else:
        print(
            f"Loaded {len(events):,} events from {metadata_path} "
            f"with searchable annotations from {searchable_queries_path}"
        )

    relevant_doc_ids_by_qid = (
        load_relevant_doc_ids_by_qid(relevant_docs_by_qid_path)
        if retrieval_mode == "multi_relevant" and relevant_docs_by_qid_path
        else {}
    )

    config = LSHSimulationConfig(
        scenario=args.scenario,
        k=k,
        L=L,
        w=w,
        bucket_replication=args.bucket_replication,
        candidate_rerank_limit=args.candidate_rerank_limit,
        metrics_window=args.metrics_window,
        random_seed=args.random_seed,
        max_owned_docs=max_owned_docs,
        distinct_candidate_sample_size=args.distinct_candidate_sample_size,
        retrieval_mode=retrieval_mode,
        relevant_doc_ids_by_qid=relevant_doc_ids_by_qid,
    )
    peer_ids = [event.user_id for event in events]
    network = LSHDHTNetwork.from_embedding_manifest(
        peer_ids=peer_ids,
        manifest_path=embedding_manifest_path,
        config=config,
    )

    seeded_stats: dict[str, int] | None = None
    if args.scenario in SEEDED_SCENARIOS:
        seeded_mapping = _load_seeded_doc_mapping(seeded_peer_docs_path)
        seeded_stats = network.seed_peer_docs(seeded_mapping)

    distinct_candidate_sample_event_indices = _sample_distinct_candidate_event_indices(
        events=events,
        query_lookup=network.query_lookup,
        scenario=args.scenario,
        sample_size=config.distinct_candidate_sample_size,
        random_seed=args.random_seed,
    )
    network.set_distinct_candidate_sample_event_indices(distinct_candidate_sample_event_indices)
    print(
        "Exact distinct candidate-doc counting sampled "
        f"{len(distinct_candidate_sample_event_indices):,} events "
        f"(requested {config.distinct_candidate_sample_size:,})."
    )

    metrics = MetricsTracker(
        output_dir=args.output_dir,
        window_size=config.metrics_window,
        query_log_mode=args.query_log_mode,
    )
    for event in tqdm(events, desc="Simulating events"):
        network.process_event(event, metrics)
    payload = metrics.finalize()

    finished_at = datetime.now(timezone.utc)
    finished_at_iso = finished_at.isoformat()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    registry_path_str = args.experiments_registry.strip() if args.experiments_registry else ""
    registry_enabled = bool(registry_path_str)
    registry_path = Path(registry_path_str) if registry_enabled else None

    metric_semantics = {
        "algorithm": ALGORITHM_NAME,
        "cycles_used": (
            "1 (querier's local hash + candidate aggregation) + count of distinct "
            "non-querier bucket-owner peers contacted. Aligned with the semantic "
            "baseline's convention of counting the querier as the first cycle."
        ),
        "result_symbol": (
            "'+' iff the clicked target doc appears in the non-local LSH candidate "
            "set returned by the L bucket lookups; '-' otherwise. Matches the "
            "semantic baseline's recall semantics."
        ),
        "distinct_candidate_docs": (
            "Exact count of unique document ids in the union of all matched LSH "
            "buckets, measured only on a deterministic bounded sample of executed "
            "queries because materializing the full union is expensive for dense "
            "LSH buckets. avg_candidate_count keeps the older summed-bucket-size "
            "convention for every executed query."
        ),
        "distinct_docs_published": (
            "Number of unique document ids ever published into the LSH index "
            "during the run, independent of how many owners or repeated publish "
            "operations each doc produced."
        ),
        "termination_reason": (
            "'found' if target was in candidate set; 'only_querier_owns' if all "
            "candidates were owned only by the querier; 'not_in_buckets' if no "
            "non-local candidate matched."
        ),
        "skip_reasons": (
            "Identical to the semantic baseline: local_held / "
            "missing_query_embedding / non_searchable, evaluated in the same "
            "order. FIFO refresh on local_held preserved."
        ),
        "parallelism_caveat": (
            "Cycles count peer contacts (units of coordination), not sequential "
            "round-trips. Semantic walks are sequential; LSH bucket lookups are "
            "parallelizable. Latency comparisons require separate analysis."
        ),
    }

    lsh_config_block = {
        "k": config.k,
        "L": config.L,
        "w": config.w,
        "bucket_replication": config.bucket_replication,
        "candidate_rerank_limit": config.candidate_rerank_limit,
        "distinct_candidate_sample_size": config.distinct_candidate_sample_size,
        "distinct_candidate_sample_event_count": len(distinct_candidate_sample_event_indices),
        "param_provenance": lsh_param_provenance,
        "calibration_profile_path": args.lsh_calibration_profile,
        "embedding_dim": network.embedding_dim,
        "base_hash_count": network.lsh.base_hash_count(),
    }

    run_manifest = {
        "run_id": run_id,
        "algorithm": ALGORITHM_NAME,
        "started_at": started_at_iso,
        "finished_at": finished_at_iso,
        "experiments_registry_path": str(registry_path) if registry_path is not None else None,
        "dataset": args.dataset,
        "amazon_category": amazon_category,
        "metadata": metadata_path,
        "searchable_queries": searchable_queries_path,
        "searchable_mode": searchable_mode,
        "retrieval_mode": retrieval_mode,
        "embedding_manifest": embedding_manifest_path,
        "output_dir": args.output_dir,
        "max_events": args.max_events,
        "scenario": args.scenario,
        "requested_scenario": requested_scenario,
        "max_owned_docs": max_owned_docs,
        "seeded_peer_docs": seeded_peer_docs_path if args.scenario in SEEDED_SCENARIOS else None,
        "relevant_docs_by_qid": relevant_docs_by_qid_path,
        "lsh_config": lsh_config_block,
        "metric_semantics": metric_semantics,
        "config": {
            "scenario": config.scenario,
            "k": config.k,
            "L": config.L,
            "w": config.w,
            "bucket_replication": config.bucket_replication,
            "candidate_rerank_limit": config.candidate_rerank_limit,
            "metrics_window": config.metrics_window,
            "random_seed": config.random_seed,
            "max_owned_docs": config.max_owned_docs,
            "distinct_candidate_sample_size": config.distinct_candidate_sample_size,
            "retrieval_mode": config.retrieval_mode,
        },
        "summary": {
            "executed_queries": payload["executed_queries"],
            "skipped_queries": payload["skipped_queries"],
            "skipped_by_reason": payload["skipped_by_reason"],
            "overall_recall": payload["overall_recall"],
            "overall_retrieval_recall": payload["overall_retrieval_recall"],
            "total_retrieved_relevant_doc_count": payload[
                "total_retrieved_relevant_doc_count"
            ],
            "overall_avg_cycles": payload["overall_avg_cycles"],
            "overall_skip_rate": payload["overall_skip_rate"],
        },
    }
    if seeded_stats is not None:
        run_manifest["seeded_stats"] = seeded_stats
    (output_dir / "run_manifest.json").write_text(
        json.dumps(run_manifest, indent=2, default=str), encoding="utf-8"
    )

    coord = network.coord
    avg_query_contacts = (
        coord.query_contacts / coord.queries_executed if coord.queries_executed else 0.0
    )
    avg_candidate_count = (
        coord.candidate_total / coord.queries_executed if coord.queries_executed else 0.0
    )
    avg_distinct_candidate_docs = (
        coord.distinct_candidate_doc_total / coord.distinct_candidate_doc_samples
        if coord.distinct_candidate_doc_samples
        else 0.0
    )
    avg_publishes_per_doc = (
        coord.publish_messages / coord.docs_published if coord.docs_published else 0.0
    )
    lsh_metrics_payload = {
        "algorithm": ALGORITHM_NAME,
        "publish_messages": coord.publish_messages,
        "docs_published": coord.docs_published,
        "distinct_docs_published": network.distinct_docs_published,
        "distinct_docs_currently_owned": len(network.doc_owners),
        "avg_publishes_per_doc": avg_publishes_per_doc,
        "queries_executed": coord.queries_executed,
        "queries_skipped": coord.queries_skipped,
        "query_contacts_total": coord.query_contacts,
        "avg_query_contacts": avg_query_contacts,
        "candidate_total": coord.candidate_total,
        "avg_candidate_count": avg_candidate_count,
        "distinct_candidate_doc_total": coord.distinct_candidate_doc_total,
        "distinct_candidate_doc_samples": coord.distinct_candidate_doc_samples,
        "distinct_candidate_sample_size_requested": config.distinct_candidate_sample_size,
        "distinct_candidate_sample_event_count": len(distinct_candidate_sample_event_indices),
        "avg_distinct_candidate_docs": avg_distinct_candidate_docs,
        "bucket_count_distinct": coord.bucket_count_distinct,
        "lsh_config": lsh_config_block,
    }
    (output_dir / "lsh_metrics.json").write_text(
        json.dumps(lsh_metrics_payload, indent=2, default=str), encoding="utf-8"
    )
    print(json.dumps(run_manifest["summary"], indent=2))
    print("LSH coordination summary:")
    print(json.dumps(lsh_metrics_payload, indent=2, default=str))

    if registry_path is not None:
        registry_record = {
            "run_id": run_id,
            "algorithm": ALGORITHM_NAME,
            "started_at": started_at_iso,
            "finished_at": finished_at_iso,
            "host": socket.gethostname(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "git_commit": _git_commit_or_none(),
            "dataset": args.dataset,
            "amazon_category": amazon_category,
            "scenario": args.scenario,
            "output_dir": str(output_dir.resolve()),
            "run_manifest_path": str((output_dir / "run_manifest.json").resolve()),
            "manifest": run_manifest,
            "lsh_metrics": lsh_metrics_payload,
        }
        _append_to_experiments_registry(registry_path, registry_record)


if __name__ == "__main__":
    main()

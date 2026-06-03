#!/usr/bin/env python3
"""Run the Low_Coord_Semantic_DHT time-ordered simulation."""

from __future__ import annotations

import argparse
import fcntl
import gzip
import json
import os
import shutil
import socket
import subprocess
import sys
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


DEFAULT_EXPERIMENTS_REGISTRY = "/var/scratch/pmneague/Low_Coord_Semantic_DHT/experiments_registry.json"
REGISTRY_SCHEMA_VERSION = 1


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _cleanup_generated_path_after_startup(path: str | None, *, output_dir: str | Path) -> dict[str, object] | None:
    if not path:
        return None
    cleanup_path = Path(path).expanduser().resolve()
    output_path = Path(output_dir).expanduser().resolve()
    payload: dict[str, object] = {
        "requested_path": str(path),
        "resolved_path": str(cleanup_path),
        "deleted": False,
        "reason": None,
    }
    if not cleanup_path.exists():
        payload["reason"] = "missing"
        print(f"[startup-cleanup] skipped missing path {cleanup_path}", flush=True)
        return payload
    if cleanup_path == Path(cleanup_path.anchor):
        raise SystemExit(f"Refusing to cleanup filesystem root: {cleanup_path}")
    if output_path == cleanup_path or output_path.is_relative_to(cleanup_path):
        raise SystemExit(
            f"Refusing to cleanup {cleanup_path} because output dir {output_path} is inside it"
        )

    if cleanup_path.is_dir():
        shutil.rmtree(cleanup_path)
    else:
        cleanup_path.unlink()
    payload["deleted"] = True
    payload["reason"] = "deleted_after_startup"
    print(f"[startup-cleanup] deleted generated path {cleanup_path}", flush=True)
    return payload


from src.scenarios import (  # noqa: E402  (sys.path mutation above is intentional)
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

    aol_dir = resolve_dataset_dir("aol4foltr")
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
    aol_dir = resolve_dataset_dir("aol4foltr")
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
    return resolve_dataset_dir("aol4foltr") / "embeddings" / "manifest.json"


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
    return resolve_dataset_dir("aol4foltr") / "peer_future_doc_ids.json.gz"


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
    """Append ``record`` to the central experiments registry under an exclusive flock.

    Failures here never abort the simulation; a corrupted registry is quarantined
    and replaced with a fresh envelope so future runs can keep appending.
    """

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


def _load_ownership_injections(path: str | Path) -> list[dict[str, object]]:
    injections_path = Path(path)
    if injections_path.suffix == ".gz":
        with gzip.open(injections_path, "rt", encoding="utf-8") as handle:
            payload = json.load(handle)
    else:
        payload = json.loads(injections_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"{injections_path} must contain a list of ownership injections")
    return [
        {
            "event_index": int(item["event_index"]),
            "peer_id": int(item["peer_id"]),
            "doc_id": str(item["doc_id"]),
        }
        for item in payload
    ]


def _load_churn_schedule(path: str | Path) -> dict[str, object]:
    schedule_path = Path(path)
    if schedule_path.suffix == ".gz":
        with gzip.open(schedule_path, "rt", encoding="utf-8") as handle:
            payload = json.load(handle)
    else:
        payload = json.loads(schedule_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{schedule_path} must contain a JSON object")
    return payload


class ChurnController:
    def __init__(
        self,
        *,
        network,
        event_count: int,
        fraction: float,
        mode: str,
        random_seed: int,
        schedule: dict[str, object] | None = None,
        schedule_path: str | Path | None = None,
    ) -> None:
        import numpy as np

        if mode not in {"none", "drop", "replace"}:
            raise ValueError(f"unsupported churn mode {mode!r}")
        if not 0.0 <= fraction <= 1.0:
            raise ValueError("churn fraction must be in [0, 1]")
        self.network = network
        self.mode = mode
        self.records: list[dict[str, int | str | None]] = []
        self._cursor = 0
        self.schedule_path = str(schedule_path) if schedule_path is not None else None
        self._schedule: list[dict[str, object]] = []
        if schedule is not None:
            self._load_explicit_schedule(schedule)
            return
        if mode == "none" or fraction == 0.0 or event_count == 0:
            return

        rng = np.random.default_rng(random_seed)
        active_peer_ids = sorted(network.active_peer_ids)
        drop_count = int(round(len(active_peer_ids) * fraction))
        if drop_count <= 0:
            return
        dropped = rng.choice(active_peer_ids, size=drop_count, replace=False).astype(int)
        drop_steps = rng.integers(0, event_count, size=drop_count).astype(int)
        replacement_start = max(network.peers) + 1 if network.peers else 0
        for offset, (event_index, peer_id) in enumerate(zip(drop_steps.tolist(), dropped.tolist())):
            replacement_id = replacement_start + offset if mode == "replace" else None
            self._schedule.append(
                {
                    "type": "legacy_peer_churn",
                    "event_index": int(event_index),
                    "peer_id": int(peer_id),
                    "replacement_peer_id": replacement_id,
                }
            )
        self._schedule.sort(key=lambda item: int(item["event_index"]))

    def _load_explicit_schedule(self, schedule: dict[str, object]) -> None:
        drop_event_index = int(schedule["drop_event_index"])
        dropped_peer_ids = [int(peer_id) for peer_id in schedule.get("dropped_peer_ids", [])]
        if dropped_peer_ids:
            self._schedule.append(
                {
                    "type": "drop_batch",
                    "event_index": drop_event_index,
                    "peer_ids": dropped_peer_ids,
                }
            )

        activations = schedule.get("replacement_activations", [])
        if not isinstance(activations, list):
            raise ValueError("replacement_activations must be a list")
        for activation in activations:
            if not isinstance(activation, dict):
                raise ValueError("replacement activation entries must be objects")
            replacement_peer_id = int(activation["replacement_peer_id"])
            if replacement_peer_id in self.network.active_peer_ids:
                self.network.deactivate_peer(replacement_peer_id)
            self._schedule.append(
                {
                    "type": "replacement_activation",
                    "event_index": int(activation["event_index"]),
                    "replacement_peer_id": replacement_peer_id,
                    "dropped_peer_id": (
                        int(activation["dropped_peer_id"])
                        if activation.get("dropped_peer_id") is not None
                        else None
                    ),
                    "doc_ids": [str(doc_id) for doc_id in activation.get("doc_ids", [])],
                }
            )
        self._schedule.sort(
            key=lambda item: (
                int(item["event_index"]),
                0 if item["type"] == "drop_batch" else 1,
                int(item.get("replacement_peer_id", -1)),
            )
        )

    def apply_until(self, event_index: int) -> None:
        while (
            self._cursor < len(self._schedule)
            and int(self._schedule[self._cursor]["event_index"]) <= event_index
        ):
            action = self._schedule[self._cursor]
            action_type = str(action["type"])
            scheduled_event_index = int(action["event_index"])
            before = self.network.active_single_owner_doc_count()
            if action_type == "drop_batch":
                dropped_peer_ids = [int(peer_id) for peer_id in action["peer_ids"]]
                for peer_id in dropped_peer_ids:
                    self.network.deactivate_peer(peer_id)
                after = self.network.active_single_owner_doc_count()
                self.records.append(
                    {
                        "event_index": scheduled_event_index,
                        "event_type": "drop_batch",
                        "dropped_peer_count": len(dropped_peer_ids),
                        "dropped_peer_id": None,
                        "replacement_peer_id": None,
                        "active_peer_count": len(self.network.active_peer_ids),
                        "single_owner_doc_count": after,
                        "newly_single_owner_doc_delta": after - before,
                    }
                )
            elif action_type == "replacement_activation":
                replacement_peer_id = int(action["replacement_peer_id"])
                self.network.activate_peer_with_docs(
                    replacement_peer_id,
                    [str(doc_id) for doc_id in action.get("doc_ids", [])],
                )
                after = self.network.active_single_owner_doc_count()
                self.records.append(
                    {
                        "event_index": scheduled_event_index,
                        "event_type": "replacement_activation",
                        "dropped_peer_id": action.get("dropped_peer_id"),
                        "replacement_peer_id": replacement_peer_id,
                        "active_peer_count": len(self.network.active_peer_ids),
                        "single_owner_doc_count": after,
                        "newly_single_owner_doc_delta": after - before,
                    }
                )
            elif action_type == "legacy_peer_churn":
                peer_id = int(action["peer_id"])
                replacement_id = action.get("replacement_peer_id")
                if replacement_id is None:
                    self.network.deactivate_peer(peer_id)
                else:
                    self.network.replace_peer(peer_id, int(replacement_id))
                after = self.network.active_single_owner_doc_count()
                self.records.append(
                    {
                        "event_index": scheduled_event_index,
                        "event_type": "legacy_peer_churn",
                        "dropped_peer_id": peer_id,
                        "replacement_peer_id": (
                            int(replacement_id) if replacement_id is not None else None
                        ),
                        "active_peer_count": len(self.network.active_peer_ids),
                        "single_owner_doc_count": after,
                        "newly_single_owner_doc_delta": after - before,
                    }
                )
            else:
                raise ValueError(f"unknown churn action type {action_type!r}")
            self._cursor += 1

    def finalize(self, output_dir: str | Path) -> dict[str, object] | None:
        if self.mode == "none" and not self._schedule:
            return None
        payload: dict[str, object] = {
            "mode": self.mode,
            "schedule_path": self.schedule_path,
            "scheduled_churn_events": len(self._schedule),
            "applied_churn_events": len(self.records),
            "final_active_peer_count": len(self.network.active_peer_ids),
            "final_single_owner_doc_count": self.network.active_single_owner_doc_count(),
            "events": self.records,
        }
        path = Path(output_dir) / "churn_metrics.json"
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        payload["output_path"] = str(path)
        return payload


def main() -> None:
    from src.amazon_dataset import AMAZON_CATEGORY_SPECS, AMAZON_SUPPORTED_SCENARIOS

    default_scenario = "empty_start_chronological"
    scenario_choices = (
        default_scenario,
        "fifo_chronological",
        "empty_start_randomized_timeline",
        "fifo_randomized_timeline",
        "seeded_fixed_inventory",
        "seeded_randomized_timeline",
    )
    dataset_choices = ("aol4foltr", "amazon", "mind", "synthetic", "pubmedqa")
    amazon_category_choices = tuple(sorted(AMAZON_CATEGORY_SPECS))

    parser = argparse.ArgumentParser(description="Run the Low_Coord_Semantic_DHT simulation")
    parser.add_argument(
        "--dataset",
        choices=dataset_choices,
        default=os.environ.get("LOW_COORD_DHT_DATASET", "aol4foltr"),
        help=(
            "Which dataset to run on. aol4foltr and mind have real chronology; amazon "
            "uses real user-item purchases/reviews + synthetic generated "
            "queries + UCSD metadata joined by asin and only supports "
            "non-chronological scenarios; synthetic is fully generated in "
            "embedding space and PubMedQA is a synthetic user trace over real "
            "questions/docs, so both only support non-chronological scenarios."
        ),
    )
    parser.add_argument(
        "--amazon-category",
        choices=amazon_category_choices,
        default=os.environ.get("LOW_COORD_DHT_AMAZON_CATEGORY", "cell_phones"),
        help="Amazon benchmark category (only used when --dataset=amazon)",
    )
    parser.add_argument(
        "--metadata",
        default=None,
        help="Metadata CSV(.gz) path; defaults to the resolved dataset dir",
    )
    parser.add_argument(
        "--searchable-queries",
        default=None,
        help="Searchable sidecar CSV(.gz) path keyed by query_id",
    )
    parser.add_argument(
        "--embedding-manifest",
        default=None,
        help="Embedding manifest generated by compute_embeddings.py",
    )
    parser.add_argument("--output-dir", default=str(Path("results") / "low_coord_semantic_dht"))
    parser.add_argument("--max-events", type=int, default=0, help="0 means all events")
    parser.add_argument("--max-neighbors", type=int, default=60)
    parser.add_argument("--max-cycles", type=int, default=50)
    parser.add_argument("--referrals-per-response", type=int, default=10)
    parser.add_argument("--similarity-threshold", type=float, default=0.5)
    parser.add_argument("--metrics-window", type=int, default=1000)
    parser.add_argument(
        "--query-log-mode",
        choices=("full", "window", "none"),
        default=os.environ.get("LOW_COORD_DHT_QUERY_LOG_MODE", "full"),
        help=(
            "Per-query logging mode: full writes query_log.jsonl, window writes "
            "only metrics_windows.jsonl rolling summaries, none writes no "
            "streaming query log."
        ),
    )
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--max-routed-queries", type=int, default=1000)
    parser.add_argument(
        "--routing-mode",
        choices=("semantic_query_memory", "replica_only"),
        default="semantic_query_memory",
        help="How peers generate referrals during distributed search",
    )
    parser.add_argument(
        "--retrieval-mode",
        choices=("single_target", "multi_relevant"),
        default=os.environ.get("LOW_COORD_DHT_RETRIEVAL_MODE"),
        help=(
            "Retrieval/evaluation objective. Defaults to multi_relevant for "
            "Amazon seeded runs and single_target otherwise."
        ),
    )
    parser.add_argument(
        "--relevant-docs-by-qid",
        default=os.environ.get("LOW_COORD_DHT_RELEVANT_DOCS_BY_QID"),
        help="Amazon multi_relevant JSON(.gz) mapping qid -> relevant doc ids",
    )
    parser.add_argument(
        "--topology-mode",
        choices=("forming_new_connections", "static_initial_connections"),
        default="forming_new_connections",
        help="Whether peer connections keep evolving after the initial random bootstrap",
    )
    parser.add_argument(
        "--selection-mode",
        choices=("similarity", "random_walk"),
        default="similarity",
        help=(
            "How the querier searches for the target peer. "
            "'similarity' (default) ranks by closest-doc-vs-query similarity; "
            "'random_walk' samples up to max_cycles active peers uniformly at random, "
            "excluding the querier, and checks target-doc ownership directly."
        ),
    )
    parser.add_argument(
        "--search-mode",
        choices=("best_first", "fanout"),
        default=os.environ.get("LOW_COORD_DHT_SEARCH_MODE", "best_first"),
        help=(
            "Distributed search policy. 'best_first' preserves the original "
            "single-peer walk; 'fanout' queries the top fanout candidates per "
            "parallel round."
        ),
    )
    parser.add_argument(
        "--fanout",
        type=int,
        default=int(os.environ.get("LOW_COORD_DHT_FANOUT", "1")),
        help="Number of top candidates to query per fanout round",
    )
    parser.add_argument(
        "--profile-search",
        action="store_true",
        default=_env_flag("LOW_COORD_DHT_PROFILE_SEARCH"),
        help="Record aggregate wall-clock time spent inside routed search",
    )
    parser.add_argument(
        "--execution-mode",
        choices=("sequential", "epoch_parallel"),
        default=os.environ.get("LOW_COORD_DHT_EXECUTION_MODE", "sequential"),
        help=(
            "How events are executed. 'epoch_parallel' runs contiguous epochs "
            "against stale epoch-start network snapshots and merges deltas in "
            "event order."
        ),
    )
    parser.add_argument(
        "--epoch-parallelism",
        type=int,
        default=int(os.environ.get("LOW_COORD_DHT_EPOCH_PARALLELISM", "1")),
        help="Worker process count for --execution-mode epoch_parallel",
    )
    parser.add_argument(
        "--epoch-chunk-size",
        type=int,
        default=int(os.environ.get("LOW_COORD_DHT_EPOCH_CHUNK_SIZE", "500")),
        help="Events per worker chunk for --execution-mode epoch_parallel",
    )
    parser.add_argument(
        "--epoch-count",
        type=int,
        default=int(os.environ.get("LOW_COORD_DHT_EPOCH_COUNT", "0")),
        help=(
            "Optional number of epoch_parallel epochs to run. If --max-events "
            "is unset, max events becomes parallelism * chunk size * epoch count."
        ),
    )
    parser.add_argument(
        "--scenario",
        choices=scenario_choices,
        default=default_scenario,
        help="Experiment scenario to run",
    )
    parser.add_argument(
        "--max-owned-docs",
        type=int,
        default=None,
        help="Maximum docs a peer can hold locally; defaults to 20 for fifo_* scenarios and unlimited otherwise",
    )
    parser.add_argument(
        "--seeded-peer-docs",
        default=None,
        help=(
            "Path to JSON(.gz) mapping for seeded scenarios; defaults to the "
            "resolved dataset dir's peer_future_doc_ids.json.gz"
        ),
    )
    parser.add_argument(
        "--drift-owner-injections",
        default=os.environ.get("LOW_COORD_DHT_DRIFT_OWNER_INJECTIONS"),
        help=(
            "Optional JSON(.gz) list of event-indexed owner injections; "
            "defaults to drift_owner_injections.json.gz beside --metadata when present"
        ),
    )
    parser.add_argument(
        "--experiments-registry",
        default=os.environ.get("LOW_COORD_DHT_EXPERIMENTS_REGISTRY", DEFAULT_EXPERIMENTS_REGISTRY),
        help=(
            "Path to a central JSON registry that records every run. "
            "Set to an empty string to disable registry writes."
        ),
    )
    parser.add_argument(
        "--churn-mode",
        choices=("none", "drop", "replace"),
        default=os.environ.get("LOW_COORD_DHT_CHURN_MODE", "none"),
        help="Synthetic ablation churn mode; non-none recomputes searchability from active owners",
    )
    parser.add_argument(
        "--churn-fraction",
        type=float,
        default=float(os.environ.get("LOW_COORD_DHT_CHURN_FRACTION", "0.5")),
        help="Fraction of peers to drop uniformly over event time when churn is enabled",
    )
    parser.add_argument(
        "--churn-random-seed",
        type=int,
        default=int(os.environ.get("LOW_COORD_DHT_CHURN_RANDOM_SEED", "42")),
        help="Random seed for peer drop order and drop times",
    )
    parser.add_argument(
        "--churn-schedule",
        default=os.environ.get("LOW_COORD_DHT_CHURN_SCHEDULE"),
        help=(
            "Optional JSON(.gz) explicit churn schedule. When present, "
            "searchability is recomputed from active owners and replacements "
            "are activated according to the schedule."
        ),
    )
    parser.add_argument(
        "--cleanup-after-startup",
        default=os.environ.get("LOW_COORD_DHT_CLEANUP_AFTER_STARTUP"),
        help=(
            "Optional generated-data path to delete after all simulator inputs "
            "are loaded and before bootstrapping / event execution starts."
        ),
    )
    parser.add_argument(
        "--load-query-embeddings-into-memory",
        action="store_true",
        default=_env_flag("LOW_COORD_DHT_LOAD_QUERY_EMBEDDINGS_INTO_MEMORY"),
        help=(
            "Load query embedding keys/vectors into RAM instead of memory-mapping "
            "them. Use this when --cleanup-after-startup deletes generated query "
            "embedding files during the run."
        ),
    )
    parser.add_argument(
        "--topic-metadata",
        default=os.environ.get("LOW_COORD_DHT_TOPIC_METADATA"),
        help="Optional synthetic query_topics.csv(.gz) path for focus-topic metrics",
    )
    parser.add_argument(
        "--topic-metrics-topic-id",
        type=int,
        default=(
            int(os.environ["LOW_COORD_DHT_TOPIC_METRICS_TOPIC_ID"])
            if "LOW_COORD_DHT_TOPIC_METRICS_TOPIC_ID" in os.environ
            else None
        ),
        help="Optional topic id to treat as the focus topic; defaults to is_drift_topic",
    )
    args = parser.parse_args()

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
    if args.execution_mode == "epoch_parallel":
        if args.churn_mode != "none" or args.churn_schedule:
            raise SystemExit("churn ablations require --execution-mode=sequential")
        if args.scenario in RUNTIME_SEARCHABLE_SCENARIOS:
            raise SystemExit(
                "epoch_parallel currently supports only sidecar-searchable seeded scenarios; "
                f"got --scenario={args.scenario}"
            )
        if args.scenario not in SEEDED_SCENARIOS:
            raise SystemExit(
                "epoch_parallel currently supports only seeded scenarios; "
                f"got --scenario={args.scenario}"
            )
        if args.selection_mode != "similarity":
            raise SystemExit("epoch_parallel currently requires --selection-mode=similarity")

    # Amazon has no real timeline, so seeded_fixed_inventory and
    # seeded_randomized_timeline mean the same thing on this dataset (both run
    # over the same seeded inventory in the same randomized order). Collapse
    # them onto a single canonical scenario so the two names produce identical
    # runs and downstream analysis cannot accidentally double-count them.
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
    from src.metrics import MetricsTracker
    from src.network import LowCoordSemanticDHTNetwork, SimulationConfig

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
    drift_owner_injections_path = args.drift_owner_injections
    if drift_owner_injections_path is None:
        try:
            from src.synthetic_dataset import SYNTHETIC_DRIFT_OWNER_INJECTIONS_FILENAME

            candidate = Path(metadata_path).parent / SYNTHETIC_DRIFT_OWNER_INJECTIONS_FILENAME
            if candidate.exists():
                drift_owner_injections_path = str(candidate)
        except ImportError:
            drift_owner_injections_path = None
    churn_schedule_path = args.churn_schedule
    if churn_schedule_path is None:
        candidate = Path(metadata_path).parent / "churn_schedule.json.gz"
        if candidate.exists():
            churn_schedule_path = str(candidate)
    churn_schedule = _load_churn_schedule(churn_schedule_path) if churn_schedule_path else None
    effective_churn_mode = args.churn_mode
    if churn_schedule is not None and effective_churn_mode == "none":
        effective_churn_mode = "replace"
    print(describe_resolution(args.dataset, amazon_category))

    searchable_mode = (
        "runtime"
        if (
            args.scenario in RUNTIME_SEARCHABLE_SCENARIOS
            or effective_churn_mode != "none"
            or retrieval_mode == "multi_relevant"
        )
        else "sidecar"
    )
    max_owned_docs = args.max_owned_docs if args.max_owned_docs is not None else (
        20 if args.scenario in FIFO_SCENARIOS else 0
    )

    effective_max_events = args.max_events
    if (
        effective_max_events == 0
        and args.execution_mode == "epoch_parallel"
        and args.epoch_count > 0
    ):
        effective_max_events = args.epoch_parallelism * args.epoch_chunk_size * args.epoch_count

    events = load_query_events(
        metadata_path,
        searchable_only=False,
        max_events=effective_max_events or None,
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

    config = SimulationConfig(
        scenario=args.scenario,
        max_neighbors=args.max_neighbors,
        max_cycles=args.max_cycles,
        referrals_per_response=args.referrals_per_response,
        similarity_threshold=args.similarity_threshold,
        metrics_window=args.metrics_window,
        random_seed=args.random_seed,
        max_routed_queries=args.max_routed_queries,
        routing_mode=args.routing_mode,
        topology_mode=args.topology_mode,
        selection_mode=args.selection_mode,
        search_mode=args.search_mode,
        fanout=args.fanout,
        execution_mode=args.execution_mode,
        epoch_parallelism=args.epoch_parallelism,
        epoch_chunk_size=args.epoch_chunk_size,
        epoch_count=args.epoch_count,
        max_owned_docs=max_owned_docs,
        profile_search=args.profile_search,
        churn_mode=effective_churn_mode,
        retrieval_mode=retrieval_mode,
        relevant_doc_ids_by_qid=relevant_doc_ids_by_qid,
    )
    peer_ids = [event.user_id for event in events]
    network = LowCoordSemanticDHTNetwork.from_embedding_manifest(
        peer_ids=peer_ids,
        manifest_path=embedding_manifest_path,
        config=config,
        load_query_embeddings_into_memory=args.load_query_embeddings_into_memory,
    )
    seeded_stats: dict[str, int] | None = None
    if args.scenario in SEEDED_SCENARIOS:
        seeded_mapping = _load_seeded_doc_mapping(seeded_peer_docs_path)
        seeded_stats = network.seed_peer_docs(seeded_mapping)
    ownership_injection_count = 0
    if drift_owner_injections_path:
        ownership_injections = _load_ownership_injections(drift_owner_injections_path)
        network.set_ownership_injection_schedule(ownership_injections)
        ownership_injection_count = len(ownership_injections)
        print(
            f"[network.injections] loaded {ownership_injection_count:,} ownership injections "
            f"from {drift_owner_injections_path}"
        )
    churn_controller = ChurnController(
        network=network,
        event_count=len(events),
        fraction=args.churn_fraction,
        mode=effective_churn_mode,
        random_seed=args.churn_random_seed,
        schedule=churn_schedule,
        schedule_path=churn_schedule_path,
    )
    topic_metrics = None
    if args.topic_metadata:
        from src.ablation_metrics import TopicMetricsTracker, load_topic_flags

        topic_metrics = TopicMetricsTracker(
            output_dir=args.output_dir,
            topic_flags=load_topic_flags(
                args.topic_metadata,
                topic_id=args.topic_metrics_topic_id,
            ),
        )
    cleanup_after_startup_summary = _cleanup_generated_path_after_startup(
        args.cleanup_after_startup,
        output_dir=args.output_dir,
    )
    if args.selection_mode == "random_walk":
        print("[network.bootstrap] skipped for selection_mode=random_walk")
    else:
        network.bootstrap_random_connections()

    metrics = MetricsTracker(
        output_dir=args.output_dir,
        window_size=config.metrics_window,
        query_log_mode=args.query_log_mode,
    )
    epoch_parallel_summary = None
    if args.execution_mode == "epoch_parallel":
        from src.parallel_network import LowCoordEpochParallelDriver

        driver = LowCoordEpochParallelDriver(
            network=network,
            parallelism=args.epoch_parallelism,
            chunk_size=args.epoch_chunk_size,
            epoch_count=args.epoch_count,
        )
        driver.run(events, metrics)
        epoch_parallel_summary = {
            "epochs_processed": driver.summary.epochs_processed,
            "events_processed": driver.summary.events_processed,
            "worker_chunks": driver.summary.worker_chunks,
            "total_worker_search_seconds": driver.summary.total_worker_search_seconds,
            "max_worker_search_seconds": driver.summary.max_worker_search_seconds,
            "epoch_size": driver.epoch_size,
        }
    else:
        for event in tqdm(events, desc="Simulating events"):
            churn_controller.apply_until(event.event_index)
            previous_executed = metrics.executed_queries
            previous_skipped = metrics.skipped_queries
            network.process_event(event, metrics)
            if topic_metrics is not None:
                topic_metrics.note_after_event(
                    event,
                    metrics=metrics,
                    previous_executed=previous_executed,
                    previous_skipped=previous_skipped,
                )
    payload = metrics.finalize()
    churn_summary = churn_controller.finalize(args.output_dir)
    topic_metrics_summary = topic_metrics.finalize() if topic_metrics is not None else None
    search_profile = dict(network.search_profile)
    if search_profile["executed_queries"]:
        search_profile["avg_search_seconds"] = (
            search_profile["total_search_seconds"] / search_profile["executed_queries"]
        )
    else:
        search_profile["avg_search_seconds"] = 0.0

    finished_at = datetime.now(timezone.utc)
    finished_at_iso = finished_at.isoformat()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    registry_path_str = args.experiments_registry.strip() if args.experiments_registry else ""
    registry_enabled = bool(registry_path_str)
    registry_path = Path(registry_path_str) if registry_enabled else None
    run_manifest = {
        "run_id": run_id,
        "started_at": started_at_iso,
        "finished_at": finished_at_iso,
        "experiments_registry_path": str(registry_path) if registry_path is not None else None,
        "dataset": args.dataset,
        "amazon_category": amazon_category,
        "metadata": metadata_path,
        "searchable_queries": searchable_queries_path,
        "searchable_mode": searchable_mode,
        "embedding_manifest": embedding_manifest_path,
        "load_query_embeddings_into_memory": args.load_query_embeddings_into_memory,
        "output_dir": args.output_dir,
        "max_events": args.max_events,
        "effective_max_events": effective_max_events,
        "query_log_mode": args.query_log_mode,
        "query_log_path": payload.get("query_log_path"),
        "window_log_path": payload.get("window_log_path"),
        "cleanup_after_startup": cleanup_after_startup_summary,
        "scenario": args.scenario,
        "requested_scenario": requested_scenario,
        "routing_mode": args.routing_mode,
        "topology_mode": args.topology_mode,
        "selection_mode": args.selection_mode,
        "retrieval_mode": retrieval_mode,
        "search_mode": args.search_mode,
        "fanout": args.fanout,
        "execution_mode": args.execution_mode,
        "epoch_parallelism": args.epoch_parallelism,
        "epoch_chunk_size": args.epoch_chunk_size,
        "epoch_count": args.epoch_count,
        "max_owned_docs": max_owned_docs,
        "seeded_peer_docs": seeded_peer_docs_path if args.scenario in SEEDED_SCENARIOS else None,
        "drift_owner_injections": drift_owner_injections_path,
        "ownership_injection_count": ownership_injection_count,
        "relevant_docs_by_qid": relevant_docs_by_qid_path,
        "churn": {
            "mode": effective_churn_mode,
            "requested_mode": args.churn_mode,
            "fraction": args.churn_fraction,
            "random_seed": args.churn_random_seed,
            "schedule": churn_schedule_path,
        },
        "topic_metadata": args.topic_metadata,
        "topic_metrics_topic_id": args.topic_metrics_topic_id,
        "config": {
            "scenario": config.scenario,
            "routing_mode": config.routing_mode,
            "topology_mode": config.topology_mode,
            "selection_mode": config.selection_mode,
            "retrieval_mode": config.retrieval_mode,
            "search_mode": config.search_mode,
            "fanout": config.fanout,
            "execution_mode": config.execution_mode,
            "epoch_parallelism": config.epoch_parallelism,
            "epoch_chunk_size": config.epoch_chunk_size,
            "epoch_count": config.epoch_count,
            "max_neighbors": config.max_neighbors,
            "max_cycles": config.max_cycles,
            "referrals_per_response": config.referrals_per_response,
            "similarity_threshold": config.similarity_threshold,
            "metrics_window": config.metrics_window,
            "random_seed": config.random_seed,
            "max_routed_queries": config.max_routed_queries,
            "max_owned_docs": config.max_owned_docs,
            "profile_search": config.profile_search,
            "churn_mode": config.churn_mode,
        },
        "notes": {
            "fanout_cycles": (
                "When search_mode='fanout', cycles_used is the number of parallel "
                "fanout rounds, not the number of queried peers."
            ),
            "epoch_parallel": (
                "When execution_mode='epoch_parallel', workers query a read-only "
                "epoch-start network snapshot and the master applies routed-query, "
                "topology, and clicked-doc deltas in event_index order at the epoch "
                "barrier. The graph is stale within each epoch."
            ),
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
        "search_profile": search_profile,
        "epoch_parallel_summary": epoch_parallel_summary,
        "churn_summary": churn_summary,
        "topic_metrics_summary": topic_metrics_summary,
    }
    if seeded_stats is not None:
        run_manifest["seeded_stats"] = seeded_stats
    (output_dir / "run_manifest.json").write_text(
        json.dumps(run_manifest, indent=2, default=str), encoding="utf-8"
    )
    if config.profile_search:
        (output_dir / "search_profile.json").write_text(
            json.dumps(search_profile, indent=2, default=str), encoding="utf-8"
        )
        print(json.dumps({"search_profile": search_profile}, indent=2))
    print(json.dumps(run_manifest["summary"], indent=2))

    if registry_path is not None:
        registry_record = {
            "run_id": run_id,
            "started_at": started_at_iso,
            "finished_at": finished_at_iso,
            "host": socket.gethostname(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "git_commit": _git_commit_or_none(),
            "dataset": args.dataset,
            "amazon_category": amazon_category,
            "scenario": args.scenario,
            "routing_mode": args.routing_mode,
            "topology_mode": args.topology_mode,
            "selection_mode": args.selection_mode,
            "output_dir": str(output_dir.resolve()),
            "run_manifest_path": str((output_dir / "run_manifest.json").resolve()),
            "manifest": run_manifest,
        }
        _append_to_experiments_registry(registry_path, registry_record)


if __name__ == "__main__":
    main()


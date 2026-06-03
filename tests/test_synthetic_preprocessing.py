from __future__ import annotations

import csv
import gzip
import json
import os
import sys
import tempfile
import unittest
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data_root import resolve_dataset_dir
from src.dataset import load_query_events
from src.synthetic_dataset import (
    SYNTHETIC_DRIFT_OWNER_INJECTIONS_FILENAME,
    SYNTHETIC_EVENTS_FILENAME,
    SYNTHETIC_PEER_FUTURE_DOCS_FILENAME,
    SYNTHETIC_RANDOMIZED_EVENTS_FILENAME,
    SYNTHETIC_SEEDED_SEARCHABLE_FILENAME,
)


class SyntheticResolverTests(unittest.TestCase):
    def setUp(self) -> None:
        self._saved_env = os.environ.get("LOW_COORD_DHT_DATA_ROOT")

    def tearDown(self) -> None:
        if self._saved_env is None:
            os.environ.pop("LOW_COORD_DHT_DATA_ROOT", None)
        else:
            os.environ["LOW_COORD_DHT_DATA_ROOT"] = self._saved_env

    def test_env_override_resolves_synthetic_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            os.environ["LOW_COORD_DHT_DATA_ROOT"] = tmp
            resolved, source = resolve_dataset_dir("synthetic", return_source=True)
            self.assertEqual(source, "env_override")
            self.assertEqual(resolved, Path(tmp) / "synthetic")


class SyntheticPrecomputeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)
        self._saved_env = os.environ.get("LOW_COORD_DHT_DATA_ROOT")
        os.environ["LOW_COORD_DHT_DATA_ROOT"] = str(self.temp_path)

    def tearDown(self) -> None:
        if self._saved_env is None:
            os.environ.pop("LOW_COORD_DHT_DATA_ROOT", None)
        else:
            os.environ["LOW_COORD_DHT_DATA_ROOT"] = self._saved_env
        self.temp_dir.cleanup()

    def test_precompute_pipeline_end_to_end_small_config(self) -> None:
        from precompute_synthetic import precompute_synthetic_artifacts

        data_dir = resolve_dataset_dir("synthetic")
        summary = precompute_synthetic_artifacts(
            data_dir=data_dir,
            topic_count=3,
            docs_per_topic=3,
            user_count=6,
            queries_per_user=20,
            dirichlet_alpha=0.3,
            embedding_dim=16,
            sigma=0.02,
            random_seed=11,
            min_target_count=1,
            run_sweep=True,
            sigma_sweep_values=(0.01, 0.05),
            calibration_query_count=12,
            nearest_batch_size=4,
            user_batch_size=2,
            top_k=3,
        )

        self.assertEqual(summary["dataset"], "synthetic")
        self.assertLess(summary["retained_query_count"], summary["query_count_before_filter"] + 1)
        self.assertGreater(summary["retained_query_count"], 0)

        events_path = data_dir / SYNTHETIC_RANDOMIZED_EVENTS_FILENAME
        searchable_path = data_dir / SYNTHETIC_SEEDED_SEARCHABLE_FILENAME
        events = load_query_events(
            events_path,
            searchable_only=False,
            require_searchable_column=True,
            searchable_queries_path=searchable_path,
        )
        self.assertEqual(len(events), summary["retained_query_count"])
        self.assertEqual(len({event.qid for event in events}), len(events))
        self.assertTrue((data_dir / SYNTHETIC_EVENTS_FILENAME).exists())

        manifest = json.loads((data_dir / "embeddings" / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["dataset"], "synthetic")
        self.assertEqual(manifest["embedding_dtype"], "float16")
        self.assertEqual(manifest["query_count"], len(events))

        query_embeddings = np.load(manifest["query_embeddings_path"], mmap_mode="r")
        doc_embeddings = np.load(manifest["doc_embeddings_path"], mmap_mode="r")
        self.assertEqual(query_embeddings.dtype, np.float16)
        self.assertEqual(doc_embeddings.dtype, np.float16)
        self.assertEqual(query_embeddings.shape[0], len(events))
        self.assertEqual(doc_embeddings.shape[0], manifest["doc_count"])
        self.assertTrue(
            np.allclose(
                np.linalg.norm(np.asarray(query_embeddings, dtype=np.float32), axis=1),
                1.0,
                atol=2e-3,
            )
        )
        self.assertTrue(
            np.allclose(
                np.linalg.norm(np.asarray(doc_embeddings, dtype=np.float32), axis=1),
                1.0,
                atol=2e-3,
            )
        )

        target_counts: dict[str, int] = defaultdict(int)
        owners_by_doc: dict[str, set[int]] = defaultdict(set)
        for event in events:
            target_counts[event.clicked_doc_id] += 1
            owners_by_doc[event.clicked_doc_id].add(event.user_id)
        self.assertTrue(all(count > 1 for count in target_counts.values()))

        doc_keys = {str(value) for value in np.load(manifest["doc_keys_path"]).tolist()}
        self.assertEqual(doc_keys, set(target_counts))

        with gzip.open(data_dir / SYNTHETIC_PEER_FUTURE_DOCS_FILENAME, "rt", encoding="utf-8") as handle:
            peer_docs = json.load(handle)
        expected_peer_docs: dict[str, set[str]] = defaultdict(set)
        for event in events:
            expected_peer_docs[str(event.user_id)].add(event.clicked_doc_id)
        self.assertEqual(
            {peer: sorted(docs) for peer, docs in expected_peer_docs.items()},
            peer_docs,
        )

        with gzip.open(searchable_path, "rt", newline="", encoding="utf-8") as handle:
            sidecar = {int(row["query_id"]): row["searchable"] == "true" for row in csv.DictReader(handle)}
        for event in events:
            expected_searchable = bool(owners_by_doc[event.clicked_doc_id] - {event.user_id})
            self.assertEqual(sidecar[event.qid], expected_searchable)
            self.assertEqual(event.searchable, expected_searchable)

    def test_capped_topic_assignment_never_replaces_overflow_draws(self) -> None:
        from precompute_synthetic import precompute_synthetic_artifacts

        data_dir = resolve_dataset_dir("synthetic")
        summary = precompute_synthetic_artifacts(
            data_dir=data_dir,
            topic_count=2,
            docs_per_topic=2,
            user_count=4,
            queries_per_user=8,
            dirichlet_alpha=0.05,
            embedding_dim=16,
            sigma=0.02,
            random_seed=17,
            min_target_count=0,
            run_sweep=False,
            nearest_batch_size=4,
            user_batch_size=2,
            target_assignment="capped_topic",
        )

        self.assertEqual(summary["target_assignment"], "capped_topic")
        self.assertGreater(summary["capped_topic_dropped_query_count"], 0)

        per_user_topic_docs: dict[tuple[int, int], set[str]] = defaultdict(set)
        with gzip.open(data_dir / "query_topics.csv.gz", "rt", newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                key = (int(row["user_id"]), int(row["generated_topic"]))
                per_user_topic_docs[key].add(row["target_doc_id"])

        self.assertTrue(per_user_topic_docs)
        self.assertTrue(all(len(doc_ids) <= 2 for doc_ids in per_user_topic_docs.values()))

    def test_concept_drift_metadata_marks_focus_topic(self) -> None:
        from precompute_synthetic import precompute_synthetic_artifacts
        from src.ablation_metrics import load_topic_flags

        data_dir = resolve_dataset_dir("synthetic")
        summary = precompute_synthetic_artifacts(
            data_dir=data_dir,
            topic_count=4,
            docs_per_topic=3,
            doc_counts_by_topic=[3, 3, 3, 1],
            user_count=4,
            queries_per_user=4,
            dirichlet_alpha=0.3,
            embedding_dim=16,
            sigma=0.02,
            random_seed=23,
            min_target_count=0,
            run_sweep=False,
            nearest_batch_size=4,
            user_batch_size=2,
            target_assignment="capped_topic",
            drift_topic_id=3,
            drift_start_fraction=0.0,
            drift_end_fraction=1.0,
            drift_probability=1.0,
        )

        self.assertEqual(summary["drift_topic_id"], 3)
        flags = load_topic_flags(data_dir / "query_topics.csv.gz")
        self.assertTrue(flags)
        self.assertTrue(all(flags.values()))

    def test_concept_drift_window_applies_global_event_timeline(self) -> None:
        from precompute_synthetic import precompute_synthetic_artifacts

        data_dir = resolve_dataset_dir("synthetic")
        summary = precompute_synthetic_artifacts(
            data_dir=data_dir,
            topic_count=3,
            docs_per_topic=4,
            user_count=20,
            queries_per_user=8,
            dirichlet_alpha=0.3,
            embedding_dim=16,
            sigma=0.02,
            random_seed=29,
            min_target_count=0,
            run_sweep=False,
            nearest_batch_size=4,
            user_batch_size=2,
            target_assignment="capped_topic",
            drift_topic_id=2,
            drift_start_fraction=0.25,
            drift_end_fraction=0.75,
            drift_probability=0.5,
        )

        rows = []
        with gzip.open(data_dir / "query_topics.csv.gz", "rt", newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))

        start = summary["drift_start_event_index"]
        end = summary["drift_end_event_index"]
        before = rows[:start]
        during = rows[start:end]
        after = rows[end:]
        self.assertFalse(any(int(row["generated_topic"]) == 2 for row in before))
        self.assertFalse(any(int(row["generated_topic"]) == 2 for row in after))

        drift_count = sum(int(row["generated_topic"]) == 2 for row in during)
        self.assertGreater(drift_count, 0)
        self.assertAlmostEqual(drift_count / len(during), 0.5, delta=0.2)

    def test_concept_drift_docs_have_runtime_owner_injections(self) -> None:
        from precompute_synthetic import precompute_synthetic_artifacts

        data_dir = resolve_dataset_dir("synthetic")
        summary = precompute_synthetic_artifacts(
            data_dir=data_dir,
            topic_count=3,
            docs_per_topic=4,
            user_count=8,
            queries_per_user=8,
            dirichlet_alpha=0.3,
            embedding_dim=16,
            sigma=0.02,
            random_seed=37,
            min_target_count=0,
            run_sweep=False,
            nearest_batch_size=4,
            user_batch_size=2,
            target_assignment="capped_topic",
            drift_topic_id=2,
            drift_start_fraction=0.25,
            drift_end_fraction=0.75,
            drift_probability=0.5,
        )

        drift_doc_ids: set[str] = set()
        with gzip.open(data_dir / "doc_topics.csv.gz", "rt", newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if int(row["topic_id"]) == 2 and row["retained"] == "true":
                    drift_doc_ids.add(row["doc_id"])

        with gzip.open(data_dir / SYNTHETIC_PEER_FUTURE_DOCS_FILENAME, "rt", encoding="utf-8") as handle:
            peer_docs = json.load(handle)
        seeded_doc_ids = {doc_id for doc_ids in peer_docs.values() for doc_id in doc_ids}

        self.assertTrue(drift_doc_ids)
        self.assertTrue(drift_doc_ids.isdisjoint(seeded_doc_ids))

        with gzip.open(
            data_dir / SYNTHETIC_DRIFT_OWNER_INJECTIONS_FILENAME,
            "rt",
            encoding="utf-8",
        ) as handle:
            injections = json.load(handle)
        self.assertEqual(len(injections), len(drift_doc_ids))
        self.assertEqual({item["doc_id"] for item in injections}, drift_doc_ids)
        self.assertTrue(
            all(item["event_index"] == summary["drift_start_event_index"] for item in injections)
        )

    def test_drift_owner_injection_schedule_seeds_at_runtime(self) -> None:
        from experiments.run_simulation import _load_ownership_injections
        from precompute_synthetic import precompute_synthetic_artifacts
        from src.metrics import MetricsTracker
        from src.network import LowCoordSemanticDHTNetwork, SimulationConfig

        data_dir = resolve_dataset_dir("synthetic")
        summary = precompute_synthetic_artifacts(
            data_dir=data_dir,
            topic_count=3,
            docs_per_topic=4,
            user_count=8,
            queries_per_user=8,
            dirichlet_alpha=0.3,
            embedding_dim=16,
            sigma=0.02,
            random_seed=39,
            min_target_count=0,
            run_sweep=False,
            nearest_batch_size=4,
            user_batch_size=2,
            target_assignment="capped_topic",
            drift_topic_id=2,
            drift_start_fraction=0.25,
            drift_end_fraction=0.75,
            drift_probability=1.0,
        )
        events = load_query_events(
            data_dir / SYNTHETIC_RANDOMIZED_EVENTS_FILENAME,
            searchable_only=False,
            require_searchable_column=True,
            searchable_queries_path=data_dir / SYNTHETIC_SEEDED_SEARCHABLE_FILENAME,
        )
        injections = _load_ownership_injections(data_dir / SYNTHETIC_DRIFT_OWNER_INJECTIONS_FILENAME)
        first_injection = injections[0]
        first_drift_event = next(event for event in events if event.event_index >= first_injection["event_index"])

        config = SimulationConfig(
            scenario="seeded_randomized_timeline",
            max_neighbors=2,
            max_cycles=1,
            random_seed=39,
        )
        network = LowCoordSemanticDHTNetwork.from_embedding_manifest(
            peer_ids=[event.user_id for event in events],
            manifest_path=summary["manifest_path"],
            config=config,
        )
        self.assertFalse(network.peers[first_injection["peer_id"]].has_doc(first_injection["doc_id"]))
        network.set_ownership_injection_schedule(injections)

        metrics = MetricsTracker(output_dir=data_dir / "metrics_injection", window_size=10)
        network.process_event(first_drift_event, metrics)
        metrics.finalize()

        self.assertTrue(network.peers[first_injection["peer_id"]].has_doc(first_injection["doc_id"]))

    def test_successful_single_target_search_acquires_doc(self) -> None:
        from precompute_synthetic import precompute_synthetic_artifacts
        from src.metrics import MetricsTracker
        from src.network import LowCoordSemanticDHTNetwork, SimulationConfig

        data_dir = resolve_dataset_dir("synthetic")
        summary = precompute_synthetic_artifacts(
            data_dir=data_dir,
            topic_count=3,
            docs_per_topic=3,
            user_count=4,
            queries_per_user=8,
            dirichlet_alpha=0.3,
            embedding_dim=16,
            sigma=0.02,
            random_seed=41,
            min_target_count=0,
            run_sweep=False,
            nearest_batch_size=4,
            user_batch_size=2,
        )
        events = load_query_events(
            data_dir / SYNTHETIC_RANDOMIZED_EVENTS_FILENAME,
            searchable_only=False,
            require_searchable_column=True,
            searchable_queries_path=data_dir / SYNTHETIC_SEEDED_SEARCHABLE_FILENAME,
        )
        event = next(event for event in events if event.searchable)
        owner_id = next(peer_id for peer_id in {candidate.user_id for candidate in events} if peer_id != event.user_id)
        config = SimulationConfig(
            scenario="seeded_randomized_timeline",
            max_neighbors=2,
            max_cycles=3,
            random_seed=41,
        )
        network = LowCoordSemanticDHTNetwork.from_embedding_manifest(
            peer_ids=[candidate.user_id for candidate in events],
            manifest_path=summary["manifest_path"],
            config=config,
        )
        network.seed_peer_docs({owner_id: [event.clicked_doc_id]})
        network.add_connection(event.user_id, owner_id)

        metrics = MetricsTracker(output_dir=data_dir / "metrics_success", window_size=10)
        network.process_event(event, metrics)
        metrics.finalize()

        self.assertTrue(network.peers[event.user_id].has_doc(event.clicked_doc_id))

    def test_failed_single_target_search_does_not_acquire_doc(self) -> None:
        from precompute_synthetic import precompute_synthetic_artifacts
        from src.metrics import MetricsTracker
        from src.network import LowCoordSemanticDHTNetwork, SimulationConfig

        data_dir = resolve_dataset_dir("synthetic")
        summary = precompute_synthetic_artifacts(
            data_dir=data_dir,
            topic_count=3,
            docs_per_topic=3,
            user_count=4,
            queries_per_user=8,
            dirichlet_alpha=0.3,
            embedding_dim=16,
            sigma=0.02,
            random_seed=43,
            min_target_count=0,
            run_sweep=False,
            nearest_batch_size=4,
            user_batch_size=2,
        )
        events = load_query_events(
            data_dir / SYNTHETIC_RANDOMIZED_EVENTS_FILENAME,
            searchable_only=False,
            require_searchable_column=True,
            searchable_queries_path=data_dir / SYNTHETIC_SEEDED_SEARCHABLE_FILENAME,
        )
        event = next(event for event in events if event.searchable)
        config = SimulationConfig(
            scenario="seeded_randomized_timeline",
            max_neighbors=2,
            max_cycles=3,
            random_seed=43,
        )
        network = LowCoordSemanticDHTNetwork.from_embedding_manifest(
            peer_ids=[candidate.user_id for candidate in events],
            manifest_path=summary["manifest_path"],
            config=config,
        )

        metrics = MetricsTracker(output_dir=data_dir / "metrics_failure", window_size=10)
        network.process_event(event, metrics)
        metrics.finalize()

        self.assertFalse(network.peers[event.user_id].has_doc(event.clicked_doc_id))

    def test_churn_deactivation_removes_active_doc_ownership(self) -> None:
        from experiments.run_simulation import _load_seeded_doc_mapping
        from precompute_synthetic import precompute_synthetic_artifacts
        from src.network import LowCoordSemanticDHTNetwork, SimulationConfig

        data_dir = resolve_dataset_dir("synthetic")
        summary = precompute_synthetic_artifacts(
            data_dir=data_dir,
            topic_count=3,
            docs_per_topic=3,
            user_count=6,
            queries_per_user=8,
            dirichlet_alpha=0.3,
            embedding_dim=16,
            sigma=0.02,
            random_seed=31,
            min_target_count=0,
            run_sweep=False,
            nearest_batch_size=4,
            user_batch_size=2,
        )
        events = load_query_events(
            data_dir / SYNTHETIC_RANDOMIZED_EVENTS_FILENAME,
            searchable_only=False,
            require_searchable_column=True,
            searchable_queries_path=data_dir / SYNTHETIC_SEEDED_SEARCHABLE_FILENAME,
        )
        config = SimulationConfig(
            scenario="seeded_randomized_timeline",
            max_neighbors=2,
            random_seed=31,
            churn_mode="drop",
        )
        network = LowCoordSemanticDHTNetwork.from_embedding_manifest(
            peer_ids=[event.user_id for event in events],
            manifest_path=summary["manifest_path"],
            config=config,
        )
        network.seed_peer_docs(_load_seeded_doc_mapping(data_dir / SYNTHETIC_PEER_FUTURE_DOCS_FILENAME))
        network.bootstrap_random_connections()

        peer_id = next(peer_id for peer_id, peer in network.peers.items() if peer.owned_docs)
        owned_doc_ids = set(network.peers[peer_id].owned_docs)
        network.deactivate_peer(peer_id)

        self.assertNotIn(peer_id, network.active_peer_ids)
        for doc_id in owned_doc_ids:
            self.assertNotIn(peer_id, network.doc_owners.get(doc_id, set()))

    def test_run_simulation_defaults_resolve_synthetic_paths(self) -> None:
        from experiments.run_simulation import (
            _default_embedding_manifest_path,
            _default_metadata_path_for_scenario,
            _default_searchable_queries_path_for_scenario,
            _default_seeded_peer_docs_path,
        )

        metadata = _default_metadata_path_for_scenario(
            "seeded_randomized_timeline",
            dataset="synthetic",
            amazon_category=None,
        )
        self.assertEqual(metadata.name, SYNTHETIC_RANDOMIZED_EVENTS_FILENAME)
        self.assertEqual(
            _default_searchable_queries_path_for_scenario(
                "seeded_randomized_timeline",
                dataset="synthetic",
                amazon_category=None,
            ).name,
            SYNTHETIC_SEEDED_SEARCHABLE_FILENAME,
        )
        self.assertEqual(
            _default_embedding_manifest_path(dataset="synthetic", amazon_category=None).name,
            "manifest.json",
        )
        self.assertEqual(
            _default_seeded_peer_docs_path(dataset="synthetic", amazon_category=None).name,
            SYNTHETIC_PEER_FUTURE_DOCS_FILENAME,
        )


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import json
import tempfile
import unittest
from datetime import date, datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.run_simulation import ChurnController  # noqa: E402
from precompute_pubmedqa_ablation import (  # noqa: E402
    _build_churn_specs_and_schedule,
    _build_concept_drift_specs_and_injections,
    _query_specs_for_inventories,
    _sample_exact_inventories,
)
from src.dataset import QueryEvent  # noqa: E402
from src.metrics import MetricsTracker  # noqa: E402
from src.network import ZeroRAGNetwork, SimulationConfig  # noqa: E402


class PubmedQaAblationPrecomputeTests(unittest.TestCase):
    def test_exact_inventory_sampler_returns_requested_docs_per_peer(self) -> None:
        docs_by_peer, topics_by_peer, summary = _sample_exact_inventories(
            peer_count=5,
            docs_per_peer=4,
            topic_count=3,
            doc_indices_by_topic=[
                [0, 1, 2, 3],
                [4, 5, 6, 7],
                [8, 9, 10, 11],
            ],
            alpha=0.1,
            random_seed=7,
        )

        self.assertEqual(sorted(docs_by_peer), [0, 1, 2, 3, 4])
        self.assertTrue(all(len(doc_ids) == 4 for doc_ids in docs_by_peer.values()))
        self.assertTrue(all(len(set(doc_ids)) == 4 for doc_ids in docs_by_peer.values()))
        self.assertTrue(all(len(topic_ids) == 4 for topic_ids in topics_by_peer.values()))
        self.assertIn("avg_unique_topics_per_peer", summary)

    def test_churn_schedule_activates_replacements_before_their_first_query(self) -> None:
        rows = [SimpleNamespace(pubid=f"doc_{idx}") for idx in range(20)]
        initial_docs = {
            0: [0, 1, 2, 3],
            1: [4, 5, 6, 7],
            2: [8, 9, 10, 11],
            3: [12, 13, 14, 15],
        }
        initial_topics = {peer_id: [0, 0, 1, 1] for peer_id in initial_docs}
        replacement_docs = {4: [0, 4, 8, 12], 5: [1, 5, 9, 13]}
        replacement_topics = {4: [0, 1, 2, 3], 5: [0, 1, 2, 3]}
        query_specs, schedule = _build_churn_specs_and_schedule(
            initial_specs=_query_specs_for_inventories(initial_docs, initial_topics),
            initial_docs_by_peer=initial_docs,
            initial_topics_by_peer=initial_topics,
            replacement_docs_by_peer=replacement_docs,
            replacement_topics_by_peer=replacement_topics,
            rows=rows,
            churn_fraction=0.5,
            random_seed=11,
        )

        drop_index = int(schedule["drop_event_index"])
        self.assertEqual(drop_index, len(query_specs) // 2)
        activations = {
            int(row["replacement_peer_id"]): int(row["event_index"])
            for row in schedule["replacement_activations"]
        }
        for event_index, spec in enumerate(query_specs):
            if spec.peer_id in replacement_docs:
                self.assertGreaterEqual(event_index, activations[spec.peer_id])

    def test_concept_drift_removes_all_docs_containing_target_mesh(self) -> None:
        rows = [
            SimpleNamespace(pubid=f"doc_{idx}", meshes=meshes)
            for idx, meshes in enumerate(
                [
                    ("A", "Target"),
                    ("Target",),
                    ("B",),
                    ("C", "Target"),
                    ("C",),
                    ("B",),
                    ("A",),
                    ("C",),
                    ("B",),
                    ("C",),
                    ("A",),
                ]
            )
        ]
        topic_terms = ["A", "Target", "B", "C"]
        docs_by_peer = {
            0: [0, 1, 2, 4],
            1: [3, 5, 6, 7],
            2: [0, 8, 9, 10],
        }
        topics_by_peer = {
            0: [1, 1, 2, 3],
            1: [1, 2, 0, 3],
            2: [0, 2, 3, 1],
        }
        initial_specs = _query_specs_for_inventories(docs_by_peer, topics_by_peer)

        query_specs, filtered_docs, _filtered_topics, injections, summary = (
            _build_concept_drift_specs_and_injections(
                initial_specs=initial_specs,
                docs_by_peer=docs_by_peer,
                topics_by_peer=topics_by_peer,
                rows=rows,
                topic_terms=topic_terms,
                drift_start_fraction=0.25,
                drift_end_fraction=0.75,
                random_seed=13,
            )
        )

        selected_mesh = str(summary["mesh_term"])
        heldout_doc_ids = {
            f"doc_{idx}"
            for idx, row in enumerate(rows)
            if selected_mesh in row.meshes
            and any(idx in doc_indices for doc_indices in docs_by_peer.values())
        }
        for doc_indices in filtered_docs.values():
            self.assertTrue(
                all(selected_mesh not in rows[doc_index].meshes for doc_index in doc_indices)
            )
        self.assertEqual({item["doc_id"] for item in injections}, heldout_doc_ids)
        self.assertEqual(summary["injection_replicas_per_doc"], 1)
        for doc_id in heldout_doc_ids:
            self.assertEqual(sum(item["doc_id"] == doc_id for item in injections), 1)
        self.assertEqual(summary["injection_count"], len(heldout_doc_ids))

        start = int(summary["drift_start_event_index"])
        end = int(summary["drift_end_event_index"])
        drift_events = 0
        for event_index, spec in enumerate(query_specs):
            has_target_mesh = selected_mesh in rows[int(spec.doc_index)].meshes
            if has_target_mesh:
                drift_events += 1
                self.assertGreaterEqual(event_index, start)
                self.assertLess(event_index, end)
            elif event_index < start:
                self.assertFalse(has_target_mesh)
        self.assertEqual(drift_events, summary["target_drift_query_count"])
        self.assertEqual(summary["drift_query_count"], summary["target_drift_query_count"])

    def test_concept_drift_resampled_queries_replace_instead_of_append(self) -> None:
        rows = [
            SimpleNamespace(pubid=f"doc_{idx}", meshes=("Target",) if idx < 2 else ("Other",))
            for idx in range(20)
        ]
        topic_terms = ["Target", "Other"]
        docs_by_peer = {
            0: [0, 2, 3, 4, 5],
            1: [1, 6, 7, 8, 9],
            2: [10, 11, 12, 13, 14],
            3: [15, 16, 17, 18, 19],
        }
        topics_by_peer = {
            peer_id: [0 if doc_index < 2 else 1 for doc_index in doc_indices]
            for peer_id, doc_indices in docs_by_peer.items()
        }
        initial_specs = _query_specs_for_inventories(docs_by_peer, topics_by_peer)

        with (
            patch("precompute_pubmedqa_ablation.DRIFT_TARGET_DOC_COUNT", 2),
            patch("precompute_pubmedqa_ablation.DRIFT_WINDOW_QUERY_FRACTION", 0.5),
        ):
            query_specs, _filtered_docs, _filtered_topics, _injections, summary = (
                _build_concept_drift_specs_and_injections(
                    initial_specs=initial_specs,
                    docs_by_peer=docs_by_peer,
                    topics_by_peer=topics_by_peer,
                    rows=rows,
                    topic_terms=topic_terms,
                    drift_start_fraction=0.25,
                    drift_end_fraction=0.75,
                    random_seed=17,
                )
            )

        self.assertEqual(len(query_specs), len(initial_specs))
        self.assertEqual(summary["mesh_term"], "Target")
        self.assertEqual(summary["heldout_event_count"], 2)
        self.assertEqual(summary["target_drift_query_count"], 5)
        self.assertEqual(summary["repeated_drift_query_fill_count"], 3)

        start = int(summary["drift_start_event_index"])
        end = int(summary["drift_end_event_index"])
        drift_events = [
            event_index
            for event_index, spec in enumerate(query_specs)
            if rows[int(spec.doc_index)].meshes == ("Target",)
        ]
        self.assertEqual(len(drift_events), 5)
        self.assertTrue(all(start <= event_index < end for event_index in drift_events))


class PubmedQaChurnRuntimeTests(unittest.TestCase):
    def _make_network(self, root: Path) -> ZeroRAGNetwork:
        doc_keys = np.asarray(["doc_a", "doc_b", "doc_c"], dtype=str)
        query_keys = np.asarray(["q0"], dtype=str)
        doc_embeddings = np.eye(3, dtype=np.float32)
        query_embeddings = np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32)
        np.save(root / "doc_keys.npy", doc_keys)
        np.save(root / "query_keys.npy", query_keys)
        np.save(root / "doc_embeddings.npy", doc_embeddings)
        np.save(root / "query_embeddings.npy", query_embeddings)
        manifest = {
            "doc_keys_path": str(root / "doc_keys.npy"),
            "query_keys_path": str(root / "query_keys.npy"),
            "doc_embeddings_path": str(root / "doc_embeddings.npy"),
            "query_embeddings_path": str(root / "query_embeddings.npy"),
        }
        manifest_path = root / "manifest.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        network = ZeroRAGNetwork.from_embedding_manifest(
            peer_ids=[0, 1, 2],
            manifest_path=manifest_path,
            config=SimulationConfig(
                scenario="seeded_randomized_timeline",
                max_neighbors=2,
                max_cycles=2,
                random_seed=5,
                churn_mode="replace",
            ),
        )
        network.seed_peer_docs({0: ["doc_a"], 1: ["doc_b"], 2: ["doc_c"]})
        return network

    def test_explicit_churn_schedule_controls_active_doc_ownership(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            network = self._make_network(root)
            schedule = {
                "drop_event_index": 2,
                "dropped_peer_ids": [0],
                "replacement_activations": [
                    {
                        "event_index": 4,
                        "dropped_peer_id": 0,
                        "replacement_peer_id": 2,
                        "doc_ids": ["doc_c"],
                    }
                ],
            }
            controller = ChurnController(
                network=network,
                event_count=6,
                fraction=0.5,
                mode="replace",
                random_seed=5,
                schedule=schedule,
                schedule_path=root / "churn_schedule.json.gz",
            )

            self.assertNotIn(2, network.active_peer_ids)
            self.assertNotIn(2, network.doc_owners.get("doc_c", set()))
            self.assertTrue(network._is_runtime_searchable(1, "doc_a"))

            controller.apply_until(2)
            self.assertNotIn(0, network.active_peer_ids)
            self.assertFalse(network._is_runtime_searchable(1, "doc_a"))

            controller.apply_until(4)
            self.assertIn(2, network.active_peer_ids)
            self.assertIn(2, network.doc_owners.get("doc_c", set()))
            self.assertTrue(network._is_runtime_searchable(1, "doc_c"))

            payload = controller.finalize(root)
            self.assertIsNotNone(payload)
            self.assertTrue((root / "churn_metrics.json").exists())
            self.assertEqual(payload["applied_churn_events"], 2)


class PubmedQaConceptDriftRuntimeTests(unittest.TestCase):
    def test_missed_drift_doc_does_not_spread_to_querier(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            doc_keys = np.asarray(["doc_target"], dtype=str)
            query_keys = np.asarray(["q0"], dtype=str)
            np.save(root / "doc_keys.npy", doc_keys)
            np.save(root / "query_keys.npy", query_keys)
            np.save(root / "doc_embeddings.npy", np.asarray([[1.0, 0.0]], dtype=np.float32))
            np.save(root / "query_embeddings.npy", np.asarray([[1.0, 0.0]], dtype=np.float32))
            manifest_path = root / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "doc_keys_path": str(root / "doc_keys.npy"),
                        "query_keys_path": str(root / "query_keys.npy"),
                        "doc_embeddings_path": str(root / "doc_embeddings.npy"),
                        "query_embeddings_path": str(root / "query_embeddings.npy"),
                    }
                ),
                encoding="utf-8",
            )
            network = ZeroRAGNetwork.from_embedding_manifest(
                peer_ids=[0, 1],
                manifest_path=manifest_path,
                config=SimulationConfig(
                    scenario="seeded_randomized_timeline",
                    max_neighbors=0,
                    max_cycles=1,
                    random_seed=3,
                ),
            )
            network.seed_peer_docs({1: ["doc_target"]})
            metrics = MetricsTracker(output_dir=root / "metrics", window_size=10)
            event = QueryEvent(
                event_index=0,
                qid=0,
                user_id=0,
                timestamp=datetime(2000, 1, 1),
                day=date(2000, 1, 1),
                query_text="q0",
                clicked_doc_id="doc_target",
                candidate_doc_ids=(),
                searchable=True,
            )

            network.process_event(event, metrics)

            self.assertEqual(metrics.executed_queries, 1)
            self.assertEqual(metrics.last_query_log.result, "-")
            self.assertFalse(network.peers[0].has_doc("doc_target"))
            metrics.finalize()


class PubmedQaAblationRunnerTests(unittest.TestCase):
    def test_pubmedqa_runners_do_not_force_continue_after_found(self) -> None:
        runner_paths = [
            REPO_ROOT / "scripts" / "run_pubmedqa_ablation_churn.slurm",
            REPO_ROOT / "scripts" / "run_pubmedqa_ablation_concept_drift.slurm",
            REPO_ROOT / "scripts" / "run_pubmedqa_ablation_degree_array.slurm",
            REPO_ROOT / "scripts" / "run_pubmedqa_ablation_doc_scalability_array.slurm",
            REPO_ROOT / "scripts" / "run_pubmedqa_ablation_fanout_array.slurm",
            REPO_ROOT / "scripts" / "run_pubmedqa_ablation_query_memory_limit_array.slurm",
            REPO_ROOT / "scripts" / "run_pubmedqa_ablation_user_scalability_array.slurm",
        ]
        for path in runner_paths:
            with self.subTest(path=path.name):
                contents = path.read_text(encoding="utf-8")
                self.assertNotIn("low-coord-continue-after-found", contents)
                self.assertNotIn("pubmedqa_top10_retrieval.py", contents)
                self.assertIn("experiments/run_simulation.py", contents)
                self.assertIn('--metrics-window "${LOW_COORD_DHT_METRICS_WINDOW:-100}"', contents)
                self.assertIn(
                    '--query-log-mode "${LOW_COORD_DHT_QUERY_LOG_MODE:-window}"',
                    contents,
                )

    def test_generated_ablation_runners_cleanup_after_startup(self) -> None:
        generated_runner_paths = [
            REPO_ROOT / "scripts" / "run_pubmedqa_ablation_churn.slurm",
            REPO_ROOT / "scripts" / "run_pubmedqa_ablation_concept_drift.slurm",
            REPO_ROOT / "scripts" / "run_pubmedqa_ablation_doc_scalability_array.slurm",
            REPO_ROOT / "scripts" / "run_pubmedqa_ablation_user_scalability_array.slurm",
        ]
        for path in generated_runner_paths:
            with self.subTest(path=path.name):
                contents = path.read_text(encoding="utf-8")
                self.assertIn('--cleanup-after-startup "${DATA_PARENT}"', contents)
                self.assertIn("--load-query-embeddings-into-memory", contents)

    def test_query_memory_runner_sweeps_requested_limits(self) -> None:
        contents = (
            REPO_ROOT / "scripts" / "run_pubmedqa_ablation_query_memory_limit_array.slurm"
        ).read_text(encoding="utf-8")
        self.assertIn("QUERY_MEMORY_LIMITS=(0 10 100 1000 10000)", contents)
        self.assertIn('--max-routed-queries "${LIMIT}"', contents)
        self.assertIn("query_memory_limit_${LIMIT}", contents)


if __name__ == "__main__":
    unittest.main()

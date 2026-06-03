from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from datetime import date, datetime
from pathlib import Path
from unittest.mock import patch

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.dataset import QueryEvent
from src.metrics import MetricsTracker, QueryLog
from src.network import LowCoordSemanticDHTNetwork, SimulationConfig
from src.parallel_network import LowCoordEpochParallelDriver
from src.peer import Peer, Referral, RoutedQuery, VectorComparisonCounts
from src.routing import SearchResult


class ReplicaOptimizationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)
        self.manifest_path = self._write_manifest()

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _write_manifest(self) -> Path:
        query_keys_path = self.temp_path / "query_keys.npy"
        query_embeddings_path = self.temp_path / "query_embeddings.npy"
        doc_keys_path = self.temp_path / "doc_keys.npy"
        doc_embeddings_path = self.temp_path / "doc_embeddings.npy"

        np.save(query_keys_path, np.asarray(["q_doc_a", "q_doc_b", "q_diag"], dtype=str))
        np.save(
            query_embeddings_path,
            np.asarray(
                [
                    [1.0, 0.0],
                    [0.0, 1.0],
                    [1.0, 1.0],
                ],
                dtype=np.float32,
            ),
        )
        np.save(doc_keys_path, np.asarray(["doc_a", "doc_b"], dtype=str))
        np.save(
            doc_embeddings_path,
            np.asarray(
                [
                    [1.0, 0.0],
                    [0.0, 1.0],
                ],
                dtype=np.float32,
            ),
        )

        manifest_path = self.temp_path / "manifest.json"
        manifest_path.write_text(
            json.dumps(
                {
                    "query_keys_path": str(query_keys_path),
                    "query_embeddings_path": str(query_embeddings_path),
                    "doc_keys_path": str(doc_keys_path),
                    "doc_embeddings_path": str(doc_embeddings_path),
                }
            ),
            encoding="utf-8",
        )
        return manifest_path

    def _make_network(
        self,
        *,
        scenario: str = "seeded_fixed_inventory",
        routing_mode: str = "semantic_query_memory",
        topology_mode: str = "forming_new_connections",
        selection_mode: str = "similarity",
        max_neighbors: int = 60,
        search_mode: str = "best_first",
        fanout: int = 1,
    ) -> LowCoordSemanticDHTNetwork:
        return LowCoordSemanticDHTNetwork.from_embedding_manifest(
            peer_ids=[1, 2, 3],
            manifest_path=self.manifest_path,
            config=SimulationConfig(
                scenario=scenario,
                routing_mode=routing_mode,
                topology_mode=topology_mode,
                selection_mode=selection_mode,
                search_mode=search_mode,
                fanout=fanout,
                max_neighbors=max_neighbors,
                max_cycles=5,
                referrals_per_response=5,
                similarity_threshold=0.1,
                metrics_window=10,
                random_seed=42,
                max_routed_queries=10,
            ),
        )

    def _make_event(
        self,
        *,
        event_index: int,
        qid: int,
        user_id: int,
        query_text: str,
        clicked_doc_id: str,
        searchable: bool = True,
    ) -> QueryEvent:
        timestamp = datetime(2024, 1, 1, 0, 0, 0)
        return QueryEvent(
            event_index=event_index,
            qid=qid,
            user_id=user_id,
            timestamp=timestamp,
            day=date(2024, 1, 1),
            query_text=query_text,
            clicked_doc_id=clicked_doc_id,
            candidate_doc_ids=(),
            searchable=searchable,
        )

    def _write_metadata_fixture(self) -> tuple[Path, Path]:
        metadata_path = self.temp_path / "metadata.csv"
        metadata_path.write_text(
            "\n".join(
                (
                    "qid,user_id,timestamp,query,target_doc_id,candidate_doc_ids",
                    '1,1,2024-01-01 00:00:00,q_doc_a,doc_a,"[]"',
                )
            )
            + "\n",
            encoding="utf-8",
        )
        searchable_path = self.temp_path / "searchable.csv"
        searchable_path.write_text(
            "\n".join(
                (
                    "query_id,searchable",
                    "1,true",
                )
            )
            + "\n",
            encoding="utf-8",
        )
        return metadata_path, searchable_path

    def test_owned_doc_catalog_preserves_tie_order_and_updates_existing_rows(self) -> None:
        peer = Peer(peer_id=7)
        query = np.asarray([1.0, 0.0], dtype=np.float32)

        peer.add_owned_doc("doc_a", query)
        peer.add_owned_doc("doc_b", query)
        first_referral = peer.best_owned_doc_referral(query, source="replica")

        self.assertIsNotNone(first_referral)
        self.assertEqual(first_referral.closest_doc_id, "doc_a")

        peer.add_owned_doc("doc_a", np.asarray([0.0, 1.0], dtype=np.float32))
        second_referral = peer.best_owned_doc_referral(query, source="replica")

        self.assertIsNotNone(second_referral)
        self.assertEqual(second_referral.closest_doc_id, "doc_b")
        self.assertEqual(peer.owned_doc_catalog.doc_ids, ["doc_a", "doc_b"])
        self.assertEqual(peer.owned_doc_catalog.size, 2)

    def test_fifo_owned_docs_refresh_and_evict_oldest(self) -> None:
        peer = Peer(peer_id=7, max_owned_docs=2)
        doc_a = np.asarray([1.0, 0.0], dtype=np.float32)
        doc_b = np.asarray([0.0, 1.0], dtype=np.float32)
        doc_c = np.asarray([1.0, 1.0], dtype=np.float32)

        self.assertIsNone(peer.add_owned_doc("doc_a", doc_a))
        self.assertIsNone(peer.add_owned_doc("doc_b", doc_b))
        self.assertEqual(list(peer.fifo_order.keys()), ["doc_a", "doc_b"])

        self.assertIsNone(peer.add_owned_doc("doc_a", doc_a))
        self.assertEqual(list(peer.fifo_order.keys()), ["doc_b", "doc_a"])

        evicted = peer.add_owned_doc("doc_c", doc_c)
        self.assertEqual(evicted, "doc_b")
        self.assertEqual(set(peer.owned_docs), {"doc_a", "doc_c"})
        self.assertEqual(set(peer.owned_doc_catalog.doc_ids), {"doc_a", "doc_c"})
        self.assertEqual(peer.owned_doc_catalog.size, 2)

    def test_connected_neighbor_uses_current_owned_docs_without_replica_copies(self) -> None:
        network = self._make_network()
        network.add_connection(1, 2)

        network.peers[2].add_owned_doc("doc_a", network.doc_lookup.get("doc_a"))
        first_referrals = network._best_replica_referrals(1, network.query_lookup.get("q_doc_a"))
        self.assertEqual([(ref.peer_id, ref.closest_doc_id) for ref in first_referrals], [(2, "doc_a")])

        network.peers[2].add_owned_doc("doc_b", network.doc_lookup.get("doc_b"))
        second_referrals = network._best_replica_referrals(1, network.query_lookup.get("q_doc_b"))
        self.assertEqual([(ref.peer_id, ref.closest_doc_id) for ref in second_referrals], [(2, "doc_b")])

    def test_connection_eviction_removes_old_neighbor_from_replica_scoring(self) -> None:
        network = self._make_network(max_neighbors=1)
        network.peers[2].add_owned_doc("doc_a", network.doc_lookup.get("doc_a"))
        network.peers[3].add_owned_doc("doc_b", network.doc_lookup.get("doc_b"))

        network.add_connection(1, 2)
        network.add_connection(1, 3)

        referrals = network._best_replica_referrals(1, network.query_lookup.get("q_diag"))
        self.assertEqual([ref.peer_id for ref in referrals], [3])

    def test_seeded_inventory_is_visible_after_bootstrap_connection(self) -> None:
        network = self._make_network(scenario="seeded_fixed_inventory")
        network.seed_peer_docs({2: ["doc_a"]})
        network.add_connection(1, 2)

        referrals = network._best_replica_referrals(1, network.query_lookup.get("q_doc_a"))

        self.assertEqual(len(referrals), 1)
        self.assertEqual(referrals[0].peer_id, 2)
        self.assertEqual(referrals[0].closest_doc_id, "doc_a")

    def test_routing_mode_gates_routed_query_memory(self) -> None:
        query_embedding = np.asarray([1.0, 0.0], dtype=np.float32)
        embeddings_by_index = {0: query_embedding}
        resolve = embeddings_by_index.__getitem__

        semantic_peer = Peer(peer_id=2, max_routed_queries=10)
        semantic_peer.respond_to_query(
            query_embedding=query_embedding,
            query_index=0,
            target_doc_id="doc_a",
            original_peer_id=9,
            referral_limit=5,
            similarity_threshold=0.1,
            resolve_query_embedding=resolve,
            replica_referrals=[],
            allow_local_hit=False,
            use_routed_query_memory=True,
        )
        semantic_response = semantic_peer.respond_to_query(
            query_embedding=query_embedding,
            query_index=0,
            target_doc_id="doc_b",
            original_peer_id=8,
            referral_limit=5,
            similarity_threshold=0.1,
            resolve_query_embedding=resolve,
            replica_referrals=[],
            allow_local_hit=False,
            use_routed_query_memory=True,
        )

        self.assertEqual([ref.peer_id for ref in semantic_response.referrals], [9])

        zero_memory_peer = Peer(peer_id=2, max_routed_queries=0)
        zero_memory_peer.respond_to_query(
            query_embedding=query_embedding,
            query_index=0,
            target_doc_id="doc_a",
            original_peer_id=9,
            referral_limit=5,
            similarity_threshold=0.1,
            resolve_query_embedding=resolve,
            replica_referrals=[],
            allow_local_hit=False,
            use_routed_query_memory=True,
        )
        zero_memory_response = zero_memory_peer.respond_to_query(
            query_embedding=query_embedding,
            query_index=0,
            target_doc_id="doc_b",
            original_peer_id=8,
            referral_limit=5,
            similarity_threshold=0.1,
            resolve_query_embedding=resolve,
            replica_referrals=[],
            allow_local_hit=False,
            use_routed_query_memory=True,
        )

        self.assertEqual(zero_memory_peer.routed_queries, [])
        self.assertEqual(zero_memory_response.referrals, [])

        replica_only_peer = Peer(peer_id=2, max_routed_queries=10)
        replica_only_peer.respond_to_query(
            query_embedding=query_embedding,
            query_index=0,
            target_doc_id="doc_a",
            original_peer_id=9,
            referral_limit=5,
            similarity_threshold=0.1,
            resolve_query_embedding=resolve,
            replica_referrals=[],
            allow_local_hit=False,
            use_routed_query_memory=False,
        )
        replica_only_response = replica_only_peer.respond_to_query(
            query_embedding=query_embedding,
            query_index=0,
            target_doc_id="doc_b",
            original_peer_id=8,
            referral_limit=5,
            similarity_threshold=0.1,
            resolve_query_embedding=resolve,
            replica_referrals=[],
            allow_local_hit=False,
            use_routed_query_memory=False,
        )

        self.assertEqual(replica_only_response.referrals, [])

    def test_peer_referral_limit_samples_exact_similarity_ties(self) -> None:
        peer = Peer(peer_id=2, max_routed_queries=10)
        referrals = [
            Referral(
                peer_id=peer_id,
                closest_doc_id=f"doc_{peer_id}",
                similarity=1.0,
                source="routed_query",
            )
            for peer_id in range(20)
        ]

        selected = peer._merge_referrals(
            referrals,
            limit=10,
            rng=np.random.default_rng(42),
        )

        selected_peer_ids = [referral.peer_id for referral in selected]
        self.assertEqual(len(selected_peer_ids), 10)
        self.assertEqual(len(set(selected_peer_ids)), 10)
        self.assertTrue(set(selected_peer_ids).issubset(set(range(20))))
        self.assertNotEqual(selected_peer_ids, list(range(10)))

    def test_query_frontier_randomizes_exact_similarity_ties(self) -> None:
        network = self._make_network()
        referrals = [
            Referral(peer_id=1, closest_doc_id="doc_a", similarity=0.9, source="routed_query"),
            Referral(peer_id=2, closest_doc_id="doc_b", similarity=0.9, source="routed_query"),
            Referral(peer_id=3, closest_doc_id="doc_c", similarity=0.8, source="replica"),
        ]

        network.rng = np.random.default_rng(1)
        first = network._select_random_best_referral(referrals)
        network.rng = np.random.default_rng(2)
        second = network._select_random_best_referral(referrals)

        self.assertIn(first.peer_id, {1, 2})
        self.assertIn(second.peer_id, {1, 2})
        self.assertNotEqual(first.peer_id, 3)
        self.assertNotEqual(second.peer_id, 3)

        winner = network._select_random_best_referral(
            [
                Referral(peer_id=1, closest_doc_id="doc_a", similarity=0.9, source="routed_query"),
                Referral(peer_id=2, closest_doc_id="doc_b", similarity=1.0, source="routed_query"),
            ]
        )
        self.assertEqual(winner.peer_id, 2)

    def test_routed_query_memory_stores_query_index_not_embedding(self) -> None:
        """Routed-query entries must store an int reference, not a copied vector.

        This is the single biggest semantic-mode memory win and is what allowed
        the Amazon ``all`` semantic runs to OOM at ~17-18% before the
        refactor; if a regression ever puts an ndarray back here the test
        will catch it before another multi-day run is wasted.
        """

        query_embedding = np.asarray([1.0, 0.0], dtype=np.float32)
        resolve = {7: query_embedding}.__getitem__

        peer = Peer(peer_id=2, max_routed_queries=10)
        peer.respond_to_query(
            query_embedding=query_embedding,
            query_index=7,
            target_doc_id="doc_a",
            original_peer_id=9,
            referral_limit=5,
            similarity_threshold=0.1,
            resolve_query_embedding=resolve,
            replica_referrals=[],
            allow_local_hit=False,
            use_routed_query_memory=True,
        )

        self.assertEqual(len(peer.routed_queries), 1)
        stored = peer.routed_queries[0]
        self.assertIsInstance(stored, RoutedQuery)
        self.assertEqual(stored.query_index, 7)
        self.assertNotIn("query_embedding", RoutedQuery.__slots__)

    def test_compute_query_response_returns_routed_delta_without_mutating_memory(self) -> None:
        query_embedding = np.asarray([1.0, 0.0], dtype=np.float32)
        resolve = {7: query_embedding}.__getitem__
        peer = Peer(peer_id=2, max_routed_queries=10)

        computation = peer.compute_query_response(
            query_embedding=query_embedding,
            query_index=7,
            target_doc_id="doc_a",
            original_peer_id=9,
            referral_limit=5,
            similarity_threshold=0.1,
            resolve_query_embedding=resolve,
            replica_referrals=[],
            allow_local_hit=False,
            use_routed_query_memory=True,
        )

        self.assertEqual(peer.routed_queries, [])
        self.assertIsNotNone(computation.routed_query)
        self.assertEqual(computation.routed_query.query_index, 7)
        peer.record_routed_query(computation.routed_query)
        self.assertEqual(len(peer.routed_queries), 1)

    def test_network_commits_routed_query_memory_only_after_success(self) -> None:
        network = self._make_network(topology_mode="static_initial_connections")
        network.seed_peer_docs({2: ["doc_a"]})
        network.add_connection(1, 2)
        metrics = MetricsTracker(output_dir=self.temp_path / "success_memory", window_size=10)

        network.process_event(
            self._make_event(
                event_index=0,
                qid=1,
                user_id=1,
                query_text="q_doc_a",
                clicked_doc_id="doc_b",
            ),
            metrics,
        )

        self.assertEqual(network.peers[1].routed_queries, [])
        self.assertEqual(network.peers[2].routed_queries, [])

        network.process_event(
            self._make_event(
                event_index=1,
                qid=1,
                user_id=1,
                query_text="q_doc_a",
                clicked_doc_id="doc_a",
            ),
            metrics,
        )

        self.assertEqual(len(network.peers[1].routed_queries), 1)
        self.assertEqual(len(network.peers[2].routed_queries), 1)

    def test_routed_query_referrals_use_resolved_embedding_for_similarity(self) -> None:
        """The resolver must be the source of truth for similarity scoring.

        We make the resolver lie about the stored query (return a vector
        orthogonal to the live one) so the only way the similarity check can
        pass is if the peer is actually fetching the embedding via the
        resolver instead of relying on a stale cached copy.
        """

        live_query = np.asarray([1.0, 0.0], dtype=np.float32)
        # Resolver returns the *live* vector for the stored entry, so a
        # subsequent query that is identical to the stored one will hit the
        # similarity threshold and produce a routed_query referral.
        resolve = {0: live_query}.__getitem__

        peer = Peer(peer_id=2, max_routed_queries=10)
        peer.respond_to_query(
            query_embedding=live_query,
            query_index=0,
            target_doc_id="doc_a",
            original_peer_id=9,
            referral_limit=5,
            similarity_threshold=0.5,
            resolve_query_embedding=resolve,
            replica_referrals=[],
            allow_local_hit=False,
            use_routed_query_memory=True,
        )

        # Now resolver lies - the stored query is "really" orthogonal. The
        # peer must consult the resolver, not any cached embedding, so the
        # routed_query referral should drop out under the same threshold.
        lying_resolver = {0: np.asarray([0.0, 1.0], dtype=np.float32)}.__getitem__
        response_with_lying_resolver = peer.respond_to_query(
            query_embedding=live_query,
            query_index=0,
            target_doc_id="doc_b",
            original_peer_id=8,
            referral_limit=5,
            similarity_threshold=0.5,
            resolve_query_embedding=lying_resolver,
            replica_referrals=[],
            allow_local_hit=False,
            use_routed_query_memory=True,
        )

        self.assertEqual(response_with_lying_resolver.referrals, [])

    def test_peer_response_counts_local_and_routed_query_vector_comparisons(self) -> None:
        query_embedding = np.asarray([1.0, 0.0], dtype=np.float32)
        resolve = {
            0: query_embedding,
            1: np.asarray([0.0, 1.0], dtype=np.float32),
        }.__getitem__
        peer = Peer(peer_id=2, max_routed_queries=10)
        peer.add_owned_doc("doc_a", query_embedding)
        peer.add_owned_doc("doc_b", np.asarray([0.0, 1.0], dtype=np.float32))
        peer.routed_queries = [
            RoutedQuery(
                query_index=0,
                original_peer_id=9,
                forwarded_to_peer_id=None,
                clicked_doc_id="doc_a",
            ),
            RoutedQuery(
                query_index=1,
                original_peer_id=10,
                forwarded_to_peer_id=None,
                clicked_doc_id="doc_b",
            ),
        ]
        counts = VectorComparisonCounts()

        response = peer.respond_to_query(
            query_embedding=query_embedding,
            query_index=0,
            target_doc_id="missing_doc",
            original_peer_id=8,
            referral_limit=5,
            similarity_threshold=0.5,
            resolve_query_embedding=resolve,
            replica_referrals=[],
            allow_local_hit=True,
            use_routed_query_memory=True,
            comparison_counts=counts,
        )

        self.assertFalse(response.found)
        self.assertEqual(counts.local_doc_vector_comparisons, 2)
        self.assertEqual(counts.routed_query_memory_comparisons, 2)
        self.assertEqual(counts.neighbor_doc_vector_comparisons, 0)
        self.assertEqual(counts.doc_vector_comparisons, 2)
        self.assertEqual(counts.total_vector_comparisons, 4)

    def test_replica_scoring_counts_neighbor_doc_vector_comparisons(self) -> None:
        network = self._make_network()
        network.add_connection(1, 2)
        network.add_connection(1, 3)
        network.peers[2].add_owned_doc("doc_a", network.doc_lookup.get("doc_a"))
        network.peers[3].add_owned_doc("doc_a", network.doc_lookup.get("doc_a"))
        network.peers[3].add_owned_doc("doc_b", network.doc_lookup.get("doc_b"))
        counts = VectorComparisonCounts()

        referrals = network._best_replica_referrals(
            1,
            network.query_lookup.get("q_doc_a"),
            comparison_counts=counts,
        )

        self.assertEqual(len(referrals), 2)
        self.assertEqual(counts.neighbor_doc_vector_comparisons, 3)
        self.assertEqual(counts.local_doc_vector_comparisons, 0)
        self.assertEqual(counts.routed_query_memory_comparisons, 0)
        self.assertEqual(counts.total_vector_comparisons, 3)

    def test_metrics_tracker_persists_vector_comparison_averages(self) -> None:
        metrics = MetricsTracker(output_dir=self.temp_path / "comparison_metrics", window_size=2)

        metrics.log_query(
            QueryLog(
                event_index=0,
                query_id=1,
                querier_peer_id=1,
                clicked_doc_id="doc_a",
                cycles_used=2,
                result="+",
                termination_reason="found",
                doc_vector_comparisons=5,
                local_doc_vector_comparisons=2,
                neighbor_doc_vector_comparisons=3,
                routed_query_memory_comparisons=7,
                total_vector_comparisons=12,
            )
        )
        metrics.log_query(
            QueryLog(
                event_index=1,
                query_id=2,
                querier_peer_id=1,
                clicked_doc_id="doc_b",
                cycles_used=4,
                result="-",
                termination_reason="cap",
                doc_vector_comparisons=9,
                local_doc_vector_comparisons=4,
                neighbor_doc_vector_comparisons=5,
                routed_query_memory_comparisons=3,
                total_vector_comparisons=12,
            )
        )

        payload = metrics.finalize()
        self.assertEqual(payload["overall_avg_doc_vector_comparisons"], 7)
        self.assertEqual(payload["overall_avg_local_doc_vector_comparisons"], 3)
        self.assertEqual(payload["overall_avg_neighbor_doc_vector_comparisons"], 4)
        self.assertEqual(payload["overall_avg_routed_query_memory_comparisons"], 5)
        self.assertEqual(payload["overall_avg_total_vector_comparisons"], 12)
        self.assertEqual(payload["windows"][0]["avg_doc_vector_comparisons"], 7)
        query_log_lines = (self.temp_path / "comparison_metrics" / "query_log.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
        first_log = json.loads(query_log_lines[0])
        self.assertEqual(first_log["local_doc_vector_comparisons"], 2)
        self.assertEqual(first_log["total_vector_comparisons"], 12)

    def test_metrics_tracker_window_mode_omits_per_query_log(self) -> None:
        output_dir = self.temp_path / "window_metrics"
        metrics = MetricsTracker(
            output_dir=output_dir,
            window_size=2,
            query_log_mode="window",
        )

        metrics.log_query(
            QueryLog(
                event_index=10,
                query_id=1,
                querier_peer_id=1,
                clicked_doc_id="doc_a",
                cycles_used=2,
                result="+",
                termination_reason="found",
            )
        )
        metrics.log_query(
            QueryLog(
                event_index=11,
                query_id=2,
                querier_peer_id=1,
                clicked_doc_id="doc_b",
                cycles_used=4,
                result="-",
                termination_reason="cap",
            )
        )

        payload = metrics.finalize()
        self.assertEqual(payload["query_log_mode"], "window")
        self.assertIsNone(payload["query_log_path"])
        self.assertFalse((output_dir / "query_log.jsonl").exists())

        window_log_path = output_dir / "metrics_windows.jsonl"
        self.assertEqual(payload["window_log_path"], str(window_log_path))
        window_lines = window_log_path.read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(window_lines), 1)
        window = json.loads(window_lines[0])
        self.assertEqual(window["executed_queries"], 2)
        self.assertEqual(window["event_start"], 10)
        self.assertEqual(window["event_end"], 11)
        self.assertEqual(window["recall"], 0.5)

    def test_forming_new_connections_rotates_neighbor_after_query(self) -> None:
        network = self._make_network(topology_mode="forming_new_connections", max_neighbors=1)
        network.add_connection(1, 2)
        metrics = MetricsTracker(output_dir=self.temp_path / "forming", window_size=10)
        event = self._make_event(
            event_index=0,
            qid=1,
            user_id=1,
            query_text="q_doc_a",
            clicked_doc_id="doc_a",
        )

        with patch(
            "src.network.run_best_first_search",
            return_value=SearchResult(
                cycles_used=1,
                result_symbol="-",
                termination_reason="exhausted",
                queried_peer_ids=[1],
                successful_peer_id=None,
                best_known_peer_id=3,
            ),
        ):
            network.process_event(event, metrics)

        self.assertEqual(list(network.peers[1].connections), [3])
        metrics.finalize()

    def test_static_initial_connections_preserves_bootstrap_neighbors_after_query(self) -> None:
        network = self._make_network(topology_mode="static_initial_connections", max_neighbors=1)
        network.add_connection(1, 2)
        metrics = MetricsTracker(output_dir=self.temp_path / "static", window_size=10)
        event = self._make_event(
            event_index=0,
            qid=1,
            user_id=1,
            query_text="q_doc_a",
            clicked_doc_id="doc_a",
        )

        with patch(
            "src.network.run_best_first_search",
            return_value=SearchResult(
                cycles_used=1,
                result_symbol="-",
                termination_reason="exhausted",
                queried_peer_ids=[1],
                successful_peer_id=None,
                best_known_peer_id=3,
            ),
        ):
            network.process_event(event, metrics)

        self.assertEqual(list(network.peers[1].connections), [2])
        metrics.finalize()

    def test_fanout_search_mode_uses_fanout_routing(self) -> None:
        network = self._make_network(
            topology_mode="forming_new_connections",
            max_neighbors=1,
            search_mode="fanout",
            fanout=10,
        )
        network.add_connection(1, 2)
        metrics = MetricsTracker(output_dir=self.temp_path / "fanout", window_size=10)
        event = self._make_event(
            event_index=0,
            qid=1,
            user_id=1,
            query_text="q_doc_a",
            clicked_doc_id="doc_a",
        )

        with patch(
            "src.network.run_fanout_search",
            return_value=SearchResult(
                cycles_used=1,
                result_symbol="-",
                termination_reason="exhausted",
                queried_peer_ids=[1],
                successful_peer_id=None,
                best_known_peer_id=3,
            ),
        ) as fanout_search:
            network.process_event(event, metrics)

        fanout_search.assert_called_once()
        self.assertEqual(fanout_search.call_args.kwargs["fanout"], 10)
        self.assertEqual(list(network.peers[1].connections), [3])
        metrics.finalize()

    def test_random_walk_probes_active_peers_without_embeddings_or_topology_update(self) -> None:
        network = self._make_network(
            routing_mode="semantic_query_memory",
            selection_mode="random_walk",
            max_neighbors=1,
        )
        network.add_connection(1, 3)
        network.peers[2].add_owned_doc("doc_a", network.doc_lookup.get("doc_a"))
        network.peers[1].record_routed_query(
            RoutedQuery(
                query_index=network.query_lookup.index_of("q_doc_a"),
                original_peer_id=2,
                forwarded_to_peer_id=None,
                clicked_doc_id="doc_a",
            )
        )
        metrics = MetricsTracker(output_dir=self.temp_path / "random_probe", window_size=10)
        event = self._make_event(
            event_index=0,
            qid=1,
            user_id=1,
            query_text="missing_query_embedding",
            clicked_doc_id="doc_a",
        )

        with patch("src.network.run_best_first_search") as best_first_search:
            network.process_event(event, metrics)

        payload = metrics.finalize()
        query_log = json.loads(
            (self.temp_path / "random_probe" / "query_log.jsonl").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(payload["executed_queries"], 1)
        self.assertEqual(payload["skipped_queries"], 0)
        self.assertEqual(query_log["result"], "+")
        self.assertEqual(query_log["total_vector_comparisons"], 0)
        self.assertEqual(query_log["routed_query_memory_comparisons"], 0)
        self.assertEqual(list(network.peers[1].connections), [3])
        best_first_search.assert_not_called()

    def test_random_walk_excludes_original_querier_from_probe_candidates(self) -> None:
        network = self._make_network(selection_mode="random_walk", routing_mode="replica_only")
        network.peers[1].add_owned_doc("doc_a", network.doc_lookup.get("doc_a"))
        metrics = MetricsTracker(
            output_dir=self.temp_path / "random_probe_excludes_self",
            window_size=10,
        )
        event = self._make_event(
            event_index=0,
            qid=1,
            user_id=1,
            query_text="missing_query_embedding",
            clicked_doc_id="doc_a",
        )

        network.process_event(event, metrics)

        payload = metrics.finalize()
        query_log = json.loads(
            (
                self.temp_path / "random_probe_excludes_self" / "query_log.jsonl"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(payload["executed_queries"], 1)
        self.assertEqual(query_log["result"], "-")
        self.assertEqual(query_log["termination_reason"], "exhausted")
        self.assertEqual(query_log["cycles_used"], 2)

    def test_epoch_parallel_driver_applies_worker_deltas_in_event_order(self) -> None:
        network = self._make_network(
            scenario="seeded_randomized_timeline",
            search_mode="fanout",
            fanout=2,
        )
        network.seed_peer_docs({2: ["doc_a"]})
        network.add_connection(1, 2)
        metrics = MetricsTracker(output_dir=self.temp_path / "epoch_parallel", window_size=10)
        events = [
            self._make_event(
                event_index=0,
                qid=1,
                user_id=1,
                query_text="q_doc_a",
                clicked_doc_id="doc_a",
            ),
            self._make_event(
                event_index=1,
                qid=2,
                user_id=1,
                query_text="q_doc_a",
                clicked_doc_id="doc_a",
            ),
        ]
        driver = LowCoordEpochParallelDriver(
            network=network,
            parallelism=2,
            chunk_size=1,
            epoch_count=1,
        )

        driver.run(events, metrics)
        payload = metrics.finalize()

        self.assertEqual(driver.summary.epochs_processed, 1)
        self.assertEqual(driver.summary.events_processed, 2)
        self.assertEqual(payload["executed_queries"], 2)
        self.assertEqual(payload["skipped_queries"], 0)
        self.assertEqual(payload["overall_recall"], 1.0)
        self.assertEqual(len(network.peers[1].routed_queries), 2)
        self.assertEqual(len(network.peers[2].routed_queries), 2)
        query_log_lines = (self.temp_path / "epoch_parallel" / "query_log.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
        self.assertEqual([json.loads(line)["event_index"] for line in query_log_lines], [0, 1])

    def test_run_manifest_includes_topology_mode(self) -> None:
        metadata_path, searchable_path = self._write_metadata_fixture()
        seeded_peer_docs_path = self.temp_path / "seeded_peer_docs.json"
        seeded_peer_docs_path.write_text("{}", encoding="utf-8")
        output_dir = self.temp_path / "cli_run"

        subprocess.run(
            [
                sys.executable,
                str(REPO_ROOT / "experiments" / "run_simulation.py"),
                "--metadata",
                str(metadata_path),
                "--searchable-queries",
                str(searchable_path),
                "--embedding-manifest",
                str(self.manifest_path),
                "--output-dir",
                str(output_dir),
                "--max-events",
                "1",
                "--scenario",
                "seeded_fixed_inventory",
                "--seeded-peer-docs",
                str(seeded_peer_docs_path),
                "--topology-mode",
                "static_initial_connections",
                "--search-mode",
                "fanout",
                "--fanout",
                "10",
                "--max-cycles",
                "10",
                "--profile-search",
                "--execution-mode",
                "epoch_parallel",
                "--epoch-parallelism",
                "1",
                "--epoch-chunk-size",
                "1",
                "--epoch-count",
                "1",
            ],
            check=True,
            cwd=REPO_ROOT,
        )

        manifest = json.loads((output_dir / "run_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["topology_mode"], "static_initial_connections")
        self.assertEqual(manifest["config"]["topology_mode"], "static_initial_connections")
        self.assertEqual(manifest["search_mode"], "fanout")
        self.assertEqual(manifest["fanout"], 10)
        self.assertEqual(manifest["config"]["max_cycles"], 10)
        self.assertEqual(manifest["config"]["search_mode"], "fanout")
        self.assertEqual(manifest["config"]["fanout"], 10)
        self.assertEqual(manifest["execution_mode"], "epoch_parallel")
        self.assertEqual(manifest["epoch_parallelism"], 1)
        self.assertEqual(manifest["epoch_chunk_size"], 1)
        self.assertEqual(manifest["epoch_count"], 1)
        self.assertEqual(manifest["config"]["execution_mode"], "epoch_parallel")
        self.assertIsNotNone(manifest["epoch_parallel_summary"])
        self.assertTrue(manifest["search_profile"]["enabled"])
        self.assertTrue((output_dir / "search_profile.json").exists())

    def test_runtime_searchable_fifo_scenario_uses_default_fifo_cap_in_manifest(self) -> None:
        metadata_path, _ = self._write_metadata_fixture()
        output_dir = self.temp_path / "fifo_cli_run"

        subprocess.run(
            [
                sys.executable,
                str(REPO_ROOT / "experiments" / "run_simulation.py"),
                "--metadata",
                str(metadata_path),
                "--embedding-manifest",
                str(self.manifest_path),
                "--output-dir",
                str(output_dir),
                "--max-events",
                "1",
                "--scenario",
                "fifo_chronological",
            ],
            check=True,
            cwd=REPO_ROOT,
        )

        manifest = json.loads((output_dir / "run_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["searchable_mode"], "runtime")
        self.assertIsNone(manifest["searchable_queries"])
        self.assertEqual(manifest["max_owned_docs"], 20)
        self.assertEqual(manifest["config"]["max_owned_docs"], 20)

    def test_fifo_runtime_scenario_skips_local_held_and_refreshes_doc_order(self) -> None:
        network = self._make_network(scenario="fifo_chronological")
        network.peers[2].add_owned_doc("doc_a", network.doc_lookup.get("doc_a"))
        network.doc_owners["doc_a"].add(2)
        network.add_connection(1, 2)
        metrics = MetricsTracker(output_dir=self.temp_path / "fifo_runtime", window_size=10)

        first_event = self._make_event(
            event_index=0,
            qid=1,
            user_id=1,
            query_text="q_doc_a",
            clicked_doc_id="doc_a",
            searchable=False,
        )
        second_event = self._make_event(
            event_index=1,
            qid=2,
            user_id=1,
            query_text="q_doc_a",
            clicked_doc_id="doc_a",
            searchable=False,
        )

        network.process_event(first_event, metrics)
        network.process_event(second_event, metrics)
        payload = metrics.finalize()

        self.assertEqual(payload["executed_queries"], 1)
        self.assertEqual(payload["skipped_queries"], 1)
        self.assertEqual(payload["skipped_by_reason"], {"local_held": 1})
        self.assertTrue(network.peers[1].has_doc("doc_a"))
        self.assertEqual(list(network.peers[1].fifo_order.keys()), ["doc_a"])

    def test_scenarios_and_routing_modes_preserve_expected_query_behavior(self) -> None:
        for scenario in (
            "empty_start_chronological",
            "seeded_fixed_inventory",
            "seeded_randomized_timeline",
        ):
            for routing_mode in ("semantic_query_memory", "replica_only"):
                with self.subTest(scenario=scenario, routing_mode=routing_mode):
                    network = self._make_network(scenario=scenario, routing_mode=routing_mode)
                    network.peers[2].add_owned_doc("doc_a", network.doc_lookup.get("doc_a"))
                    network.add_connection(1, 2)

                    metrics = MetricsTracker(
                        output_dir=self.temp_path / f"{scenario}_{routing_mode}",
                        window_size=10,
                    )

                    first_event = self._make_event(
                        event_index=0,
                        qid=1,
                        user_id=1,
                        query_text="q_doc_a",
                        clicked_doc_id="doc_a",
                    )
                    second_event = self._make_event(
                        event_index=1,
                        qid=2,
                        user_id=1,
                        query_text="q_doc_a",
                        clicked_doc_id="doc_a",
                    )

                    network.process_event(first_event, metrics)
                    network.process_event(second_event, metrics)
                    payload = metrics.finalize()

                    if scenario == "empty_start_chronological":
                        self.assertEqual(payload["executed_queries"], 1)
                        self.assertEqual(payload["skipped_queries"], 1)
                        self.assertEqual(payload["skipped_by_reason"], {"local_held": 1})
                    else:
                        self.assertEqual(payload["executed_queries"], 2)
                        self.assertEqual(payload["skipped_queries"], 0)
                        self.assertEqual(payload["skipped_by_reason"], {})


if __name__ == "__main__":
    unittest.main()

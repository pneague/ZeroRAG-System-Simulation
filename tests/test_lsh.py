"""Tests for the E2LSH-DHT baseline.

Covers the four claims the plan makes:

* :class:`src.lsh.E2LSHFamily` is deterministic for a given seed and
  produces ``L`` tuples of length ``k``.
* Bucket-owner assignment in :class:`src.lsh_network.LSHDHTNetwork` is
  deterministic, uses sparse real peer ids correctly, and respects
  ``bucket_replication``.
* End-to-end smoke on a small synthetic dataset produces non-zero recall
  and ``cycles_used`` bounded by ``1 + L * bucket_replication``.
* Calibration sanity: for normalized vectors, picking ``w`` from the
  recommended range avoids degenerate bucket collapse and produces
  non-trivial candidate counts.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from collections import Counter
from datetime import date, datetime
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.dataset import QueryEvent
from src.lsh import E2LSHFamily, analytic_collision_probability
from src.lsh_network import LSHDHTNetwork, LSHSimulationConfig
from src.metrics import MetricsTracker


class E2LSHFamilyTests(unittest.TestCase):
    def test_deterministic_seed(self) -> None:
        a = E2LSHFamily.build(d=16, k=4, L=8, w=0.5, seed=123)
        b = E2LSHFamily.build(d=16, k=4, L=8, w=0.5, seed=123)
        self.assertTrue(np.array_equal(a.A, b.A))
        self.assertTrue(np.array_equal(a.B, b.B))

    def test_hash_table_shapes(self) -> None:
        family = E2LSHFamily.build(d=8, k=5, L=7, w=0.75, seed=42)
        vec = np.linspace(-1.0, 1.0, 8).astype(np.float32)
        bucket_ids = family.hash_tables(vec)
        self.assertEqual(len(bucket_ids), 7)
        for bucket_id in bucket_ids:
            self.assertEqual(len(bucket_id), 5)
            for value in bucket_id:
                self.assertIsInstance(value, int)

    def test_paper_alias_r_equivalent_to_w(self) -> None:
        a = E2LSHFamily.build(d=4, k=3, L=2, w=0.4, seed=1)
        b = E2LSHFamily.build(d=4, k=3, L=2, r=0.4, seed=1)
        vec = np.array([0.1, -0.2, 0.3, -0.4], dtype=np.float32)
        self.assertEqual(a.hash_tables(vec), b.hash_tables(vec))

    def test_analytic_collision_probability_monotone(self) -> None:
        # Closer points must collide more often than far ones.
        small = analytic_collision_probability(c=0.1, w=0.5)
        large = analytic_collision_probability(c=2.0, w=0.5)
        self.assertGreater(small, large)
        self.assertGreaterEqual(small, 0.0)
        self.assertLessEqual(small, 1.0)


class _SyntheticEmbedding(unittest.TestCase):
    """Shared helper that materializes a tiny embedding manifest on disk."""

    @classmethod
    def _write_manifest(
        cls,
        tmp: Path,
        *,
        d: int,
        n_queries: int,
        n_docs: int,
        seed: int,
    ) -> tuple[Path, list[str], list[str], np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed)
        query_keys = [f"q{i}" for i in range(n_queries)]
        doc_keys = [f"d{i}" for i in range(n_docs)]

        # Construct queries and docs so that q_i is the closest doc to d_i.
        # Put the first n_docs queries near the corresponding docs and the
        # rest as random non-correlated queries.
        doc_emb = rng.standard_normal(size=(n_docs, d)).astype(np.float32)
        doc_emb /= np.linalg.norm(doc_emb, axis=1, keepdims=True)

        query_emb = np.zeros((n_queries, d), dtype=np.float32)
        for i in range(n_queries):
            base = doc_emb[i % n_docs]
            jitter = rng.normal(scale=0.05, size=d).astype(np.float32)
            vec = base + jitter
            vec /= np.linalg.norm(vec)
            query_emb[i] = vec

        np.save(tmp / "qk.npy", np.asarray(query_keys, dtype=str))
        np.save(tmp / "qe.npy", query_emb)
        np.save(tmp / "dk.npy", np.asarray(doc_keys, dtype=str))
        np.save(tmp / "de.npy", doc_emb)
        manifest_path = tmp / "manifest.json"
        manifest_path.write_text(
            json.dumps(
                {
                    "query_keys_path": str(tmp / "qk.npy"),
                    "query_embeddings_path": str(tmp / "qe.npy"),
                    "doc_keys_path": str(tmp / "dk.npy"),
                    "doc_embeddings_path": str(tmp / "de.npy"),
                }
            ),
            encoding="utf-8",
        )
        return manifest_path, query_keys, doc_keys, query_emb, doc_emb


class BucketOwnershipTests(_SyntheticEmbedding):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmp.name)
        self.manifest_path, *_ = self._write_manifest(
            self.tmp_path, d=8, n_queries=8, n_docs=8, seed=11
        )

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _make_network(
        self, *, peer_ids, bucket_replication: int = 1
    ) -> LSHDHTNetwork:
        cfg = LSHSimulationConfig(
            scenario="empty_start_chronological",
            k=3,
            L=4,
            w=0.5,
            bucket_replication=bucket_replication,
            random_seed=7,
        )
        return LSHDHTNetwork.from_embedding_manifest(
            peer_ids=peer_ids, manifest_path=str(self.manifest_path), config=cfg
        )

    def test_owner_uses_real_sparse_peer_ids(self) -> None:
        # Sparse peer ids that do NOT span a contiguous range.
        sparse_peer_ids = [101, 257, 8888, 999_999]
        network = self._make_network(peer_ids=sparse_peer_ids)
        for table_id in range(4):
            for bucket_id in [(0, 0, 0), (1, -1, 2), (5, 5, 5)]:
                owners = network.bucket_owners(table_id, bucket_id)
                self.assertEqual(len(owners), 1)
                self.assertIn(owners[0], sparse_peer_ids)

    def test_owner_assignment_deterministic(self) -> None:
        peer_ids = list(range(10))
        net_a = self._make_network(peer_ids=peer_ids)
        net_b = self._make_network(peer_ids=peer_ids)
        bucket_id = (3, -2, 1)
        for table_id in range(4):
            self.assertEqual(
                net_a.bucket_owners(table_id, bucket_id),
                net_b.bucket_owners(table_id, bucket_id),
            )

    def test_replication_returns_distinct_owners(self) -> None:
        peer_ids = list(range(20))
        network = self._make_network(peer_ids=peer_ids, bucket_replication=3)
        owners = network.bucket_owners(0, (1, 2, 3))
        self.assertEqual(len(owners), 3)
        self.assertEqual(len(set(owners)), 3)


class EndToEndSmokeTests(_SyntheticEmbedding):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmp.name)
        self.manifest_path, self.query_keys, self.doc_keys, _, _ = self._write_manifest(
            self.tmp_path, d=32, n_queries=50, n_docs=100, seed=17
        )

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _make_event(self, idx: int, *, user_id: int, doc_id: str, query_text: str) -> QueryEvent:
        moment = datetime(2025, 1, 1, 0, 0, idx)
        return QueryEvent(
            event_index=idx,
            qid=idx,
            user_id=user_id,
            timestamp=moment,
            day=date(2025, 1, 1),
            query_text=query_text,
            clicked_doc_id=doc_id,
            candidate_doc_ids=(),
            searchable=True,
        )

    def test_smoke_recall_and_cycle_bound(self) -> None:
        peer_ids = list(range(10))
        n_events = 50
        events = []
        for idx in range(n_events):
            doc_idx = idx % len(self.doc_keys)
            query_text = self.query_keys[idx % len(self.query_keys)]
            user_id = peer_ids[idx % len(peer_ids)]
            events.append(
                self._make_event(
                    idx, user_id=user_id, doc_id=self.doc_keys[doc_idx], query_text=query_text
                )
            )

        # Seed every doc on multiple peers (skipping the peer that will
        # later click it, so the search has to find a non-local owner).
        cfg = LSHSimulationConfig(
            scenario="empty_start_chronological",
            k=4,
            L=12,
            w=0.5,
            bucket_replication=1,
            random_seed=3,
            metrics_window=10,
        )
        network = LSHDHTNetwork.from_embedding_manifest(
            peer_ids=[event.user_id for event in events],
            manifest_path=str(self.manifest_path),
            config=cfg,
        )
        seeded_mapping: dict[int, list[str]] = {peer_id: [] for peer_id in peer_ids}
        for doc_idx, doc_id in enumerate(self.doc_keys):
            click_peer = peer_ids[doc_idx % len(peer_ids)]
            seed_owners = [pid for pid in peer_ids if pid != click_peer][:3]
            for owner in seed_owners:
                seeded_mapping[owner].append(doc_id)
        network.seed_peer_docs(seeded_mapping)

        out_dir = Path(self.tmp_path) / "smoke_out"
        metrics = MetricsTracker(output_dir=out_dir, window_size=cfg.metrics_window)
        for event in events:
            network.process_event(event, metrics)
        payload = metrics.finalize()
        self.assertGreater(payload["overall_recall"], 0.0)
        # cycles_used = 1 (querier) + #distinct non-querier owners contacted,
        # bounded above by 1 + L * bucket_replication.
        max_cycles = 1 + cfg.L * cfg.bucket_replication
        for window in payload["windows"]:
            self.assertLessEqual(window["avg_cycles"], max_cycles)
        self.assertEqual(network.coord.queries_executed, payload["executed_queries"])
        self.assertEqual(network.coord.queries_skipped, payload["skipped_queries"])
        self.assertGreater(network.coord.distinct_candidate_doc_total, 0)
        self.assertLessEqual(
            network.coord.distinct_candidate_doc_total,
            network.coord.candidate_total,
        )
        self.assertGreater(network.distinct_docs_published, 0)
        self.assertLessEqual(network.distinct_docs_published, len(self.doc_keys))

    def test_metrics_json_schema_matches_semantic(self) -> None:
        # The plotting tools key on these fields; if any drift, the LSH runs
        # would silently break cross-algorithm overlays.
        cfg = LSHSimulationConfig(
            scenario="empty_start_chronological",
            k=3,
            L=5,
            w=0.5,
            random_seed=5,
            metrics_window=5,
        )
        events = [
            self._make_event(
                idx,
                user_id=idx % 5,
                doc_id=self.doc_keys[idx % len(self.doc_keys)],
                query_text=self.query_keys[idx % len(self.query_keys)],
            )
            for idx in range(20)
        ]
        network = LSHDHTNetwork.from_embedding_manifest(
            peer_ids=[event.user_id for event in events],
            manifest_path=str(self.manifest_path),
            config=cfg,
        )
        out_dir = Path(self.tmp_path) / "schema_out"
        metrics = MetricsTracker(output_dir=out_dir, window_size=cfg.metrics_window)
        for event in events:
            network.process_event(event, metrics)
        payload = metrics.finalize()
        for key in (
            "executed_queries",
            "skipped_queries",
            "overall_recall",
            "overall_avg_cycles",
            "overall_skip_rate",
            "windows",
        ):
            self.assertIn(key, payload)
        for window in payload["windows"]:
            for key in ("window_index", "recall", "avg_cycles", "skip_rate"):
                self.assertIn(key, window)


class CalibrationSanityTests(_SyntheticEmbedding):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmp.name)
        self.manifest_path, *_ = self._write_manifest(
            self.tmp_path, d=16, n_queries=10, n_docs=200, seed=99
        )

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _bucket_distribution(self, *, w: float, k: int) -> Counter:
        from src.network import EmbeddingLookup

        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        doc_lookup = EmbeddingLookup(
            keys_path=manifest["doc_keys_path"],
            embeddings_path=manifest["doc_embeddings_path"],
        )
        family = E2LSHFamily.build(
            d=int(doc_lookup.embeddings.shape[1]),
            k=k,
            L=1,
            w=w,
            seed=42,
        )
        bucket_arr = family.hash_tables_batch(doc_lookup.embeddings.astype(np.float32))
        return Counter(tuple(row.tolist()) for row in bucket_arr[:, 0, :])

    def test_w_in_recommended_range_avoids_collapse(self) -> None:
        # Normalized vectors with paper default w=4 collapse into too few
        # buckets. The recommended range (0.25..1.0) should produce a
        # diverse bucket distribution.
        diverse = self._bucket_distribution(w=0.5, k=8)
        degenerate = self._bucket_distribution(w=4.0, k=8)
        self.assertGreater(len(diverse), len(degenerate))
        # And the diverse setting must not place every doc in one bucket.
        self.assertGreater(len(diverse), 1)


if __name__ == "__main__":
    unittest.main()

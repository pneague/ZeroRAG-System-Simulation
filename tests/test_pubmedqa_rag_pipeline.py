from __future__ import annotations

import csv
import gzip
import json
import os
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

from precompute_pubmedqa_original_all_mesh import (  # noqa: E402
    FullPubMedQaRow,
    precompute_pubmedqa_original_all_mesh_artifacts,
)
from precompute_pubmedqa_rag import precompute_pubmedqa_rag_artifacts
from scripts.plot_topic_rank_histogram import build_topic_rank_rows
from scripts.pubmedqa_top10_retrieval import (
    _hnsw_rows,
    _load_answer_targets,
    _zerorag_rows,
    _lsh_rows,
    _semantica_rows,
)
from src.data_root import resolve_dataset_dir
from src.dataset import QueryEvent, load_query_events
from src.pubmedqa_dataset import (
    PUBMEDQA_ANSWER_TARGETS_FILENAME,
    PUBMEDQA_QUERY_TOPICS_FILENAME,
    PUBMEDQA_RANDOMIZED_EVENTS_FILENAME,
    PUBMEDQA_SEEDED_SEARCHABLE_FILENAME,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _normalized_eye(rows: int, dim: int) -> np.ndarray:
    vectors = np.zeros((rows, dim), dtype=np.float32)
    for row in range(rows):
        vectors[row, row % dim] = 1.0
    return vectors


class PubMedQaRagPipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self._saved_env = os.environ.get("LOW_COORD_DHT_DATA_ROOT")
        os.environ["LOW_COORD_DHT_DATA_ROOT"] = str(self.root)

    def tearDown(self) -> None:
        if self._saved_env is None:
            os.environ.pop("LOW_COORD_DHT_DATA_ROOT", None)
        else:
            os.environ["LOW_COORD_DHT_DATA_ROOT"] = self._saved_env
        self.temp_dir.cleanup()

    def _fixture_paths(self) -> tuple[Path, Path, Path]:
        selected_path = self.root / "selected_manifest.jsonl"
        rephrased_path = self.root / "rephrased_queries.jsonl"
        embeddings_dir = self.root / "source_embeddings"
        embeddings_dir.mkdir()
        selected_rows = [
            {
                "selected_index": 0,
                "config": "pqa_labeled",
                "row_index": 0,
                "pubid": "doc0",
                "question": "Question 0?",
                "doc_text": "Doc 0",
                "meshes": ["A"],
                "matching_selected_meshes": ["A"],
                "final_decision": "yes",
                "long_answer": "Answer 0",
            },
            {
                "selected_index": 1,
                "config": "pqa_labeled",
                "row_index": 1,
                "pubid": "doc1",
                "question": "Question 1?",
                "doc_text": "Doc 1",
                "meshes": ["A"],
                "matching_selected_meshes": ["A"],
                "final_decision": "no",
                "long_answer": "Answer 1",
            },
            {
                "selected_index": 2,
                "config": "pqa_labeled",
                "row_index": 2,
                "pubid": "doc2",
                "question": "Question 2?",
                "doc_text": "Doc 2",
                "meshes": ["B"],
                "matching_selected_meshes": ["B"],
                "final_decision": "maybe",
                "long_answer": "Answer 2",
            },
            {
                "selected_index": 3,
                "config": "pqa_labeled",
                "row_index": 3,
                "pubid": "doc3",
                "question": "Question 3?",
                "doc_text": "Doc 3",
                "meshes": ["B"],
                "matching_selected_meshes": ["B"],
                "final_decision": "yes",
                "long_answer": "Answer 3",
            },
        ]
        _write_jsonl(selected_path, selected_rows)
        _write_jsonl(
            rephrased_path,
            [
                {
                    "selected_index": row["selected_index"],
                    "pubid": row["pubid"],
                    "config": row["config"],
                    "row_index": row["row_index"],
                    "question": row["question"],
                    "rephrasings": [f"{row['question']} rephrase {idx}" for idx in range(5)],
                }
                for row in selected_rows
            ],
        )
        dim = 4
        doc_embeddings = _normalized_eye(4, dim)
        np.save(embeddings_dir / "doc_pubids.npy", np.asarray([row["pubid"] for row in selected_rows]))
        np.save(embeddings_dir / "doc_embeddings.npy", doc_embeddings)
        np.save(embeddings_dir / "original_query_embeddings.npy", doc_embeddings)
        np.save(embeddings_dir / "rephrased_query_embeddings.npy", np.repeat(doc_embeddings, 5, axis=0))
        (embeddings_dir / "embedding_manifest.json").write_text(
            json.dumps({"embedding_dim": dim, "embedding_model": "fixture"}),
            encoding="utf-8",
        )
        return selected_path, rephrased_path, embeddings_dir

    def test_precompute_outputs_preserve_rephrased_answer_targets(self) -> None:
        selected_path, rephrased_path, embeddings_dir = self._fixture_paths()
        data_dir = resolve_dataset_dir("pubmedqa")
        summary = precompute_pubmedqa_rag_artifacts(
            data_dir=data_dir,
            selected_manifest_path=selected_path,
            rephrased_queries_path=rephrased_path,
            source_embeddings_dir=embeddings_dir,
            hf_root=None,
            user_count=4,
            queries_per_user=6,
            dirichlet_alpha=0.01,
            random_seed=7,
            embedding_dtype="float32",
            embedding_batch_size=4,
        )
        self.assertEqual(summary["dataset"], "pubmedqa")
        events = load_query_events(
            data_dir / PUBMEDQA_RANDOMIZED_EVENTS_FILENAME,
            searchable_queries_path=data_dir / PUBMEDQA_SEEDED_SEARCHABLE_FILENAME,
            require_searchable_column=True,
        )
        self.assertEqual(len(events), 24)

        answer_rows = list(_load_answer_targets(data_dir / PUBMEDQA_ANSWER_TARGETS_FILENAME).values())
        self.assertEqual(len(answer_rows), 24)
        for row in answer_rows:
            self.assertEqual(row["target_doc_id"], row["source_pubid"])
            if row["query_variant"] == "rephrased":
                self.assertIsNotNone(row["rephrase_index"])
                self.assertNotEqual(row["question"], row["original_question"])

        manifest = json.loads((data_dir / "embeddings" / "manifest.json").read_text(encoding="utf-8"))
        query_keys = np.load(manifest["query_keys_path"], allow_pickle=True)
        query_embeddings = np.load(manifest["query_embeddings_path"], mmap_mode="r")
        self.assertEqual(query_keys.shape[0], 24)
        self.assertEqual(query_embeddings.shape, (24, 4))

    def test_topic_rank_histogram_reads_pubmedqa_topics(self) -> None:
        selected_path, rephrased_path, embeddings_dir = self._fixture_paths()
        data_dir = resolve_dataset_dir("pubmedqa")
        precompute_pubmedqa_rag_artifacts(
            data_dir=data_dir,
            selected_manifest_path=selected_path,
            rephrased_queries_path=rephrased_path,
            source_embeddings_dir=embeddings_dir,
            hf_root=None,
            user_count=3,
            queries_per_user=4,
            random_seed=11,
            embedding_dtype="float32",
        )
        rows, summary = build_topic_rank_rows(data_dir / PUBMEDQA_QUERY_TOPICS_FILENAME)
        self.assertGreaterEqual(len(rows), 1)
        self.assertEqual(summary["query_count"], 12)
        self.assertEqual(summary["peer_count"], 3)
        self.assertEqual(rows[0]["topic_rank"], 1)

    def test_original_all_mesh_precompute_uses_only_original_queries(self) -> None:
        data_dir = resolve_dataset_dir("pubmedqa")
        source_dir = self.root / "full_source"
        emb_dir = source_dir / "embeddings"
        emb_dir.mkdir(parents=True)
        full_rows = [
            FullPubMedQaRow(0, "pqa_labeled", 0, "doc0", "Question 0?", "yes", "A0", ("A",)),
            FullPubMedQaRow(1, "pqa_labeled", 1, "doc1", "Question 1?", "no", "A1", ("A",)),
            FullPubMedQaRow(2, "pqa_labeled", 2, "doc2", "Question 2?", "yes", "A2", ("B",)),
            FullPubMedQaRow(3, "pqa_labeled", 3, "doc3", "Question 3?", "no", "A3", ("B",)),
        ]
        with (source_dir / "corpus_manifest.jsonl").open("w", encoding="utf-8") as handle:
            for row in full_rows:
                handle.write(
                    json.dumps(
                        {
                            "config": row.config,
                            "row_index": row.row_index,
                            "pubid": row.pubid,
                            "question": row.question,
                            "doc_text": row.question,
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
        (source_dir / "embedding_state.json").write_text(
            json.dumps({"doc_rows_embedded": 4, "query_rows_embedded": 4}),
            encoding="utf-8",
        )
        (source_dir / "run_manifest.json").write_text(
            json.dumps({"row_count": 4, "embedding_dim": 4, "embedding_model": "fixture"}),
            encoding="utf-8",
        )
        vectors = _normalized_eye(4, 4)
        np.save(emb_dir / "doc_pubids.npy", np.asarray([row.pubid for row in full_rows], dtype=str))
        np.save(emb_dir / "query_pubids.npy", np.asarray([row.pubid for row in full_rows], dtype=str))
        np.save(emb_dir / "doc_embeddings.npy", vectors)
        np.save(emb_dir / "query_embeddings.npy", vectors)

        with patch(
            "precompute_pubmedqa_original_all_mesh._load_full_rows",
            return_value=(
                full_rows,
                {
                    "row_count": 4,
                    "rows_with_meshes": 4,
                    "unique_mesh_terms": 2,
                },
            ),
        ):
            summary = precompute_pubmedqa_original_all_mesh_artifacts(
                data_dir=data_dir,
                hf_root=self.root / "hf",
                source_full_dir=source_dir,
                peer_count=4,
                queries_per_peer=5,
                alpha=0.00007,
                random_seed=23,
                embedding_dtype="float32",
                embedding_batch_size=4,
            )

        self.assertEqual(summary["query_count"], 20)
        self.assertEqual(summary["query_variant_counts"], {"original": 20})
        answer_rows = list(_load_answer_targets(data_dir / PUBMEDQA_ANSWER_TARGETS_FILENAME).values())
        self.assertEqual(len(answer_rows), 20)
        self.assertTrue(all(row["query_variant"] == "original" for row in answer_rows))
        manifest = json.loads((data_dir / "embeddings" / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["doc_count"], 4)
        self.assertEqual(manifest["query_count"], 20)

    def test_hnsw_exact_top10_rows_include_timing(self) -> None:
        selected_path, rephrased_path, embeddings_dir = self._fixture_paths()
        data_dir = resolve_dataset_dir("pubmedqa")
        precompute_pubmedqa_rag_artifacts(
            data_dir=data_dir,
            selected_manifest_path=selected_path,
            rephrased_queries_path=rephrased_path,
            source_embeddings_dir=embeddings_dir,
            hf_root=None,
            user_count=2,
            queries_per_user=3,
            random_seed=13,
            embedding_dtype="float32",
        )
        events = load_query_events(
            data_dir / PUBMEDQA_RANDOMIZED_EVENTS_FILENAME,
            searchable_queries_path=data_dir / PUBMEDQA_SEEDED_SEARCHABLE_FILENAME,
            require_searchable_column=True,
        )
        answer_targets = _load_answer_targets(data_dir / PUBMEDQA_ANSWER_TARGETS_FILENAME)
        rows = list(
            _hnsw_rows(
                events=events,
                manifest_path=data_dir / "embeddings" / "manifest.json",
                answer_targets=answer_targets,
                top_k=2,
                backend="exact",
                ef_search=8,
                hnsw_m=8,
                construction_threads=1,
                search_threads=1,
            )
        )
        self.assertEqual(len(rows), 6)
        self.assertTrue(all("search_seconds" in row for row in rows))
        self.assertTrue(all(len(row["top10_doc_ids"]) == 2 for row in rows))

    def test_distributed_top10_rows_have_expected_schema(self) -> None:
        selected_path, rephrased_path, embeddings_dir = self._fixture_paths()
        data_dir = resolve_dataset_dir("pubmedqa")
        precompute_pubmedqa_rag_artifacts(
            data_dir=data_dir,
            selected_manifest_path=selected_path,
            rephrased_queries_path=rephrased_path,
            source_embeddings_dir=embeddings_dir,
            hf_root=None,
            user_count=4,
            queries_per_user=4,
            random_seed=17,
            embedding_dtype="float32",
        )
        events = load_query_events(
            data_dir / PUBMEDQA_RANDOMIZED_EVENTS_FILENAME,
            searchable_queries_path=data_dir / PUBMEDQA_SEEDED_SEARCHABLE_FILENAME,
            require_searchable_column=True,
            max_events=5,
        )
        answer_targets = _load_answer_targets(data_dir / PUBMEDQA_ANSWER_TARGETS_FILENAME)
        with gzip.open(data_dir / "peer_future_doc_ids.json.gz", "rt", encoding="utf-8") as handle:
            seeded_mapping = {int(peer): docs for peer, docs in json.load(handle).items()}
        manifest_path = data_dir / "embeddings" / "manifest.json"

        zerorag = list(
            _zerorag_rows(
                events=events,
                manifest_path=manifest_path,
                seeded_mapping=seeded_mapping,
                answer_targets=answer_targets,
                top_k=2,
                random_seed=17,
                max_neighbors=2,
                max_cycles=3,
                routing_mode="semantic_query_memory",
                topology_mode="forming_new_connections",
                search_mode="best_first",
                fanout=1,
                referrals_per_response=2,
                similarity_threshold=0.0,
                continue_after_found=True,
            )
        )
        lsh = list(
            _lsh_rows(
                events=events,
                manifest_path=manifest_path,
                seeded_mapping=seeded_mapping,
                answer_targets=answer_targets,
                top_k=2,
                random_seed=17,
                lsh_k=2,
                lsh_l=3,
                lsh_w=1.0,
                bucket_replication=1,
            )
        )
        semantica = list(
            _semantica_rows(
                events=events,
                manifest_path=manifest_path,
                seeded_mapping=seeded_mapping,
                answer_targets=answer_targets,
                top_k=2,
                random_seed=17,
                max_cycles=3,
                record_limit_per_leafnode=50,
                delta=0.003,
                ncu=3,
                expansion_rounds=1,
            )
        )
        self.assertEqual(len(zerorag), len(events))
        self.assertEqual(len(lsh), len(events))
        self.assertEqual(len(semantica), len(events))
        self.assertTrue(all(row["model"] == "zerorag" for row in zerorag))
        self.assertTrue(any("search_seconds" in row for row in zerorag))
        self.assertTrue(
            all(
                "continued_after_found" in row
                for row in zerorag
                if row.get("skipped_reason") is None
            )
        )
        self.assertTrue(all("top10_doc_ids" in row for row in lsh + semantica))

    def test_zerorag_top10_applies_drift_injection_before_event(self) -> None:
        manifest_dir = self.root / "injection_fixture"
        manifest_dir.mkdir()
        np.save(manifest_dir / "doc_keys.npy", np.asarray(["target", "other"], dtype=str))
        np.save(manifest_dir / "query_keys.npy", np.asarray(["q"], dtype=str))
        np.save(
            manifest_dir / "doc_embeddings.npy",
            np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        )
        np.save(
            manifest_dir / "query_embeddings.npy",
            np.asarray([[1.0, 0.0]], dtype=np.float32),
        )
        manifest_path = manifest_dir / "manifest.json"
        manifest_path.write_text(
            json.dumps(
                {
                    "doc_keys_path": str(manifest_dir / "doc_keys.npy"),
                    "query_keys_path": str(manifest_dir / "query_keys.npy"),
                    "doc_embeddings_path": str(manifest_dir / "doc_embeddings.npy"),
                    "query_embeddings_path": str(manifest_dir / "query_embeddings.npy"),
                }
            ),
            encoding="utf-8",
        )
        now = datetime(2026, 1, 1)
        events = [
            QueryEvent(idx, idx, user_id, now, date(2026, 1, 1), "q", "target", (), True)
            for idx, user_id in enumerate([0, 0, 1])
        ]

        rows = list(
            _zerorag_rows(
                events=events,
                manifest_path=manifest_path,
                seeded_mapping={0: [], 1: ["other"]},
                answer_targets={},
                top_k=1,
                random_seed=3,
                max_neighbors=1,
                max_cycles=2,
                routing_mode="semantic_query_memory",
                topology_mode="static_initial_connections",
                search_mode="best_first",
                fanout=1,
                referrals_per_response=1,
                similarity_threshold=-1.0,
                continue_after_found=True,
                drift_owner_injections=[
                    {"event_index": 1, "peer_id": 1, "doc_id": "target"}
                ],
            )
        )

        self.assertFalse(rows[0]["target_in_top10"])
        self.assertTrue(rows[1]["target_in_top10"])


if __name__ == "__main__":
    unittest.main()

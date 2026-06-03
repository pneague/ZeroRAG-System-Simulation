from __future__ import annotations

import csv
import gzip
import json
import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from precompute_searchable import build_chronological_searchable_queries
from precompute_seeded_randomized import build_seeded_randomized_artifacts
from src.dataset import load_query_events


class DatasetPreprocessingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _write_metadata(self, path: Path, rows: list[dict[str, str]]) -> None:
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "timestamp",
                    "qid",
                    "user_id",
                    "query",
                    "target_doc_id",
                    "candidate_doc_ids",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)

    def _read_csv_gz_rows(self, path: Path) -> list[dict[str, str]]:
        with gzip.open(path, "rt", newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))

    def test_chronological_searchable_sidecar_drives_loader(self) -> None:
        metadata_path = self.temp_path / "metadata.csv"
        searchable_path = self.temp_path / "chronological_searchable_queries.csv.gz"
        self._write_metadata(
            metadata_path,
            [
                {
                    "timestamp": "2024-01-01 00:00:00",
                    "qid": "1",
                    "user_id": "1",
                    "query": "q_shared_a",
                    "target_doc_id": "doc_shared",
                    "candidate_doc_ids": "[]",
                },
                {
                    "timestamp": "2024-01-01 00:00:01",
                    "qid": "2",
                    "user_id": "2",
                    "query": "q_shared_b",
                    "target_doc_id": "doc_shared",
                    "candidate_doc_ids": "[]",
                },
                {
                    "timestamp": "2024-01-02 00:00:00",
                    "qid": "3",
                    "user_id": "1",
                    "query": "q_lonely",
                    "target_doc_id": "doc_lonely",
                    "candidate_doc_ids": "[]",
                },
            ],
        )

        build_chronological_searchable_queries(metadata_path, searchable_path)

        sidecar_rows = self._read_csv_gz_rows(searchable_path)
        self.assertEqual(list(sidecar_rows[0].keys()), ["query_id", "searchable"])
        self.assertEqual(
            {int(row["query_id"]): row["searchable"] for row in sidecar_rows},
            {1: "true", 2: "true", 3: "false"},
        )

        events = load_query_events(
            metadata_path,
            searchable_only=True,
            require_searchable_column=True,
            searchable_queries_path=searchable_path,
        )
        self.assertEqual([event.qid for event in events], [1, 2])

    def test_seeded_randomized_artifacts_are_deterministic_and_compact(self) -> None:
        metadata_path = self.temp_path / "metadata.csv"
        seeded_peer_docs_path = self.temp_path / "peer_future_doc_ids.json.gz"
        randomized_a_path = self.temp_path / "metadata_seeded_randomized_a.csv.gz"
        randomized_b_path = self.temp_path / "metadata_seeded_randomized_b.csv.gz"
        searchable_a_path = self.temp_path / "seeded_searchable_queries_a.csv.gz"
        searchable_b_path = self.temp_path / "seeded_searchable_queries_b.csv.gz"

        self._write_metadata(
            metadata_path,
            [
                {
                    "timestamp": "2024-01-01 00:00:00",
                    "qid": "1",
                    "user_id": "1",
                    "query": "q_doc_shared_1",
                    "target_doc_id": "doc_shared",
                    "candidate_doc_ids": "[]",
                },
                {
                    "timestamp": "2024-01-01 00:00:01",
                    "qid": "2",
                    "user_id": "1",
                    "query": "q_doc_only_user_1",
                    "target_doc_id": "doc_only_user_1",
                    "candidate_doc_ids": "[]",
                },
                {
                    "timestamp": "2024-01-01 00:00:02",
                    "qid": "3",
                    "user_id": "2",
                    "query": "q_doc_shared_2",
                    "target_doc_id": "doc_shared",
                    "candidate_doc_ids": "[]",
                },
                {
                    "timestamp": "2024-01-01 00:00:03",
                    "qid": "4",
                    "user_id": "3",
                    "query": "q_doc_only_user_3",
                    "target_doc_id": "doc_only_user_3",
                    "candidate_doc_ids": "[]",
                },
            ],
        )
        with gzip.open(seeded_peer_docs_path, "wt", encoding="utf-8") as handle:
            json.dump(
                {
                    "1": ["doc_shared", "doc_only_user_1"],
                    "2": ["doc_shared"],
                    "3": ["doc_only_user_3"],
                },
                handle,
            )

        build_seeded_randomized_artifacts(
            metadata_path,
            seeded_peer_docs_path,
            randomized_a_path,
            searchable_a_path,
            random_seed=7,
        )
        build_seeded_randomized_artifacts(
            metadata_path,
            seeded_peer_docs_path,
            randomized_b_path,
            searchable_b_path,
            random_seed=7,
        )

        randomized_rows_a = self._read_csv_gz_rows(randomized_a_path)
        randomized_rows_b = self._read_csv_gz_rows(randomized_b_path)
        self.assertEqual(randomized_rows_a, randomized_rows_b)
        self.assertEqual(
            sorted(int(row["qid"]) for row in randomized_rows_a),
            [1, 2, 3, 4],
        )
        timestamps = [datetime.strptime(row["timestamp"], "%Y-%m-%d %H:%M:%S") for row in randomized_rows_a]
        self.assertEqual(timestamps, sorted(timestamps))

        searchable_rows = self._read_csv_gz_rows(searchable_a_path)
        self.assertEqual(list(searchable_rows[0].keys()), ["query_id", "searchable"])
        self.assertEqual(
            {int(row["query_id"]): row["searchable"] for row in searchable_rows},
            {1: "true", 2: "false", 3: "true", 4: "false"},
        )

        events = load_query_events(
            randomized_a_path,
            searchable_only=True,
            require_searchable_column=True,
            searchable_queries_path=searchable_a_path,
        )
        self.assertEqual(sorted(event.qid for event in events), [1, 3])


if __name__ == "__main__":
    unittest.main()

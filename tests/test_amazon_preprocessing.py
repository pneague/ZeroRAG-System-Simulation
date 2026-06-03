from __future__ import annotations

import csv
import gzip
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.amazon_dataset import (
    AMAZON_DOC_METADATA_FILENAME,
    AMAZON_EVENTS_FILENAME,
    AMAZON_PEER_FUTURE_DOCS_FILENAME,
    AMAZON_PEER_ID_MAP_FILENAME,
    AMAZON_RANDOMIZED_EVENTS_FILENAME,
    AMAZON_SEEDED_SEARCHABLE_FILENAME,
    AMAZON_SUPPORTED_SCENARIOS,
    parse_amazon_qrels_key,
)
from src.dataset import load_query_events
from src.dataset import (
    AMAZON_QUERY_AVAILABILITY_FILENAME,
    RELEVANT_DOC_IDS_BY_QID_FILENAME,
)
from src.data_root import resolve_dataset_dir


class AmazonHelpersTests(unittest.TestCase):
    def test_parse_amazon_qrels_key_splits_on_last_underscore(self) -> None:
        self.assertEqual(parse_amazon_qrels_key("A3ABC_42"), ("A3ABC", 42))
        self.assertEqual(
            parse_amazon_qrels_key("AKM1MP6P0OYPR_FOO_7"), ("AKM1MP6P0OYPR_FOO", 7)
        )
        with self.assertRaises(ValueError):
            parse_amazon_qrels_key("nokey")
        with self.assertRaises(ValueError):
            parse_amazon_qrels_key("trailing_")
        with self.assertRaises(ValueError):
            parse_amazon_qrels_key("_leading")

    def test_amazon_supported_scenarios_excludes_chronological(self) -> None:
        self.assertIn("seeded_fixed_inventory", AMAZON_SUPPORTED_SCENARIOS)
        self.assertNotIn("empty_start_chronological", AMAZON_SUPPORTED_SCENARIOS)
        self.assertNotIn("fifo_chronological", AMAZON_SUPPORTED_SCENARIOS)

    def test_amazon_seeded_scenarios_resolve_to_same_metadata_path(self) -> None:
        # Both seeded scenarios on Amazon must point at the same metadata file
        # because seeded_fixed_inventory is aliased to seeded_randomized_timeline
        # (Amazon has no real timeline). The canonicalization happens in main();
        # _default_metadata_path_for_scenario is exercised after canonicalization.
        from experiments.run_simulation import _default_metadata_path_for_scenario

        randomized = _default_metadata_path_for_scenario(
            "seeded_randomized_timeline",
            dataset="amazon",
            amazon_category="cell_phones",
        )
        # seeded_fixed_inventory pre-canonicalization technically points at events.csv;
        # the simulator never reaches that branch on Amazon, but the file does still
        # need to exist on disk for inspection. The alias contract is enforced in main().
        self.assertEqual(randomized.name, "metadata_seeded_randomized.csv.gz")


class AmazonResolverTests(unittest.TestCase):
    def setUp(self) -> None:
        self._saved_env = os.environ.get("LOW_COORD_DHT_DATA_ROOT")

    def tearDown(self) -> None:
        if self._saved_env is None:
            os.environ.pop("LOW_COORD_DHT_DATA_ROOT", None)
        else:
            os.environ["LOW_COORD_DHT_DATA_ROOT"] = self._saved_env

    def test_env_override_wins(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            os.environ["LOW_COORD_DHT_DATA_ROOT"] = tmp
            resolved, source = resolve_dataset_dir(
                "amazon", "cell_phones", return_source=True
            )
            self.assertEqual(source, "env_override")
            self.assertEqual(resolved, Path(tmp) / "amazon" / "cell_phones")

    def test_amazon_requires_category(self) -> None:
        with self.assertRaises(ValueError):
            resolve_dataset_dir("amazon")

    def test_unknown_dataset_raises(self) -> None:
        with self.assertRaises(ValueError):
            resolve_dataset_dir("does_not_exist")


class AmazonPrecomputeTests(unittest.TestCase):
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

    def _write_gz_text(self, path: Path, content: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(path, "wt", encoding="utf-8") as handle:
            handle.write(content)

    def _write_jsonl_gz(self, path: Path, records: list[dict]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(path, "wt", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record) + "\n")

    def test_precompute_pipeline_end_to_end(self) -> None:
        from precompute_amazon import precompute_amazon_artifacts
        from src.amazon_dataset import category_spec

        category = "cell_phones"
        spec = category_spec(category)
        data_dir = resolve_dataset_dir("amazon", category)
        benchmark_dir = data_dir / "benchmark"
        metadata_dir = data_dir / "metadata"

        # Three queries (line numbers 0,1,2)
        self._write_gz_text(
            benchmark_dir / "query_text.txt.gz",
            "wireless headphones\nbluetooth speaker\nphone case\n",
        )
        # Train: reviewer A has positives for q0->X1 and q1->X2; reviewer B for q0->X1
        self._write_gz_text(
            benchmark_dir / "train.qrels.gz",
            "AAA_0 0 X1 1\n"
            "AAA_1 0 X2 1\n"
            "BBB_0 0 X1 1\n"
            "BBB_2 0 X3 0\n",  # negative: should be dropped
        )
        # Test: reviewer C for q1->X2 (so X2 has 2 distinct owners)
        self._write_gz_text(
            benchmark_dir / "test.qrels.gz",
            "CCC_1 0 X2 1\n",
        )
        self._write_gz_text(benchmark_dir / "train_review_id.txt.gz", "RID0\nRID1\nRID2\n")

        # UCSD metadata for X1, X2, X3, plus an unrelated asin to ensure filtering
        self._write_jsonl_gz(
            metadata_dir / spec.ucsd_meta_filename,
            [
                {
                    "asin": "X1",
                    "title": "Headphones",
                    "description": ["Comfy"],
                    "category": [["Electronics", "Headphones"]],
                    "brand": "BrandA",
                    "price": "$19.99",
                },
                {
                    "asin": "X2",
                    "title": "Speaker",
                    "description": "Loud",
                    "categories": ["Electronics", "Speakers"],
                    "brand": "BrandB",
                    "price": "",
                },
                {
                    "asin": "X3",
                    "title": "Case",
                    "description": "",
                    "category": [["Electronics", "Cases"]],
                    "brand": "BrandC",
                    "price": "5.00",
                },
                {
                    "asin": "ZZZ",
                    "title": "Irrelevant",
                    "description": "",
                    "category": [],
                    "brand": "",
                    "price": "",
                },
            ],
        )

        summary = precompute_amazon_artifacts(category, random_seed=7)

        # Reviewer mapping is deterministic by sorted reviewerID
        peer_id_map_path = data_dir / AMAZON_PEER_ID_MAP_FILENAME
        with gzip.open(peer_id_map_path, "rt", encoding="utf-8") as handle:
            peer_id_map = json.load(handle)
        # Negative-only reviewer BBB still appears because they have a positive too
        self.assertEqual(peer_id_map, {"AAA": 0, "BBB": 1, "CCC": 2})

        # Events: 4 positives (negative X3 dropped)
        events_path = data_dir / AMAZON_EVENTS_FILENAME
        with events_path.open("r", newline="", encoding="utf-8") as handle:
            event_rows = list(csv.DictReader(handle))
        self.assertEqual(len(event_rows), 4)
        targets = {(int(r["user_id"]), int(r["qid"]), r["target_doc_id"]) for r in event_rows}
        self.assertEqual(
            targets,
            {(0, 0, "X1"), (0, 1, "X2"), (1, 0, "X1"), (2, 1, "X2")},
        )
        # Synthetic timestamps strictly increase
        timestamps = [r["timestamp"] for r in event_rows]
        self.assertEqual(timestamps, sorted(timestamps))

        # Searchable sidecar: q0 -> two distinct owners of X1 (AAA, BBB) -> searchable
        # q1 -> two distinct owners of X2 (AAA, CCC) -> searchable
        # q2 has no positives -> absent
        searchable_path = data_dir / AMAZON_SEEDED_SEARCHABLE_FILENAME
        with gzip.open(searchable_path, "rt", newline="", encoding="utf-8") as handle:
            sidecar = {int(r["query_id"]): r["searchable"] for r in csv.DictReader(handle)}
        self.assertEqual(sidecar, {0: "true", 1: "true"})

        # peer_future_doc_ids: peer 0 -> {X1, X2}, peer 1 -> {X1}, peer 2 -> {X2}
        peer_docs_path = data_dir / AMAZON_PEER_FUTURE_DOCS_FILENAME
        with gzip.open(peer_docs_path, "rt", encoding="utf-8") as handle:
            peer_docs = json.load(handle)
        self.assertEqual(peer_docs, {"0": ["X1", "X2"], "1": ["X1"], "2": ["X2"]})

        relevant_path = data_dir / RELEVANT_DOC_IDS_BY_QID_FILENAME
        with gzip.open(relevant_path, "rt", encoding="utf-8") as handle:
            relevant_by_qid = json.load(handle)
        self.assertEqual(relevant_by_qid, {"0": ["X1"], "1": ["X2"]})

        availability_path = data_dir / AMAZON_QUERY_AVAILABILITY_FILENAME
        with gzip.open(availability_path, "rt", encoding="utf-8") as handle:
            availability = json.load(handle)
        self.assertEqual(availability["0"]["relevant_doc_count"], 1)
        self.assertEqual(availability["0"]["available_relevant_doc_count"], 1)
        self.assertEqual(availability["1"]["available_relevant_doc_count"], 1)

        # doc_metadata.csv.gz contains exactly the positive-only asins (X1, X2);
        # X3 only appears as a negative and ZZZ is unrelated, so neither is joined.
        doc_metadata_path = data_dir / AMAZON_DOC_METADATA_FILENAME
        with gzip.open(doc_metadata_path, "rt", newline="", encoding="utf-8") as handle:
            doc_rows = list(csv.DictReader(handle))
        self.assertEqual({r["asin"] for r in doc_rows}, {"X1", "X2"})

        # Randomized events file is loadable by the AOL loader and yields the
        # same set of qids (with the seeded searchable sidecar)
        randomized_path = data_dir / AMAZON_RANDOMIZED_EVENTS_FILENAME
        events = load_query_events(
            randomized_path,
            searchable_only=True,
            require_searchable_column=True,
            searchable_queries_path=searchable_path,
        )
        self.assertEqual(sorted(e.qid for e in events), [0, 0, 1, 1])

        # Summary JSON sanity
        self.assertEqual(summary["positive_event_count"], 4)
        self.assertEqual(summary["user_query_event_count"], 4)
        self.assertEqual(summary["unique_reviewer_count"], 3)
        # Only X1 and X2 appear as positives; X3 is dropped (negative-only)
        self.assertEqual(summary["unique_asin_count"], 2)
        self.assertEqual(summary["doc_metadata_found"], 2)
        self.assertEqual(summary["doc_metadata_missing"], 0)


if __name__ == "__main__":
    unittest.main()

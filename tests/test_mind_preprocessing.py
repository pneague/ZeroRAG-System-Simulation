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

from src.data_root import resolve_dataset_dir
from src.dataset import (
    CHRONOLOGICAL_SEARCHABLE_QUERIES_FILENAME,
    load_query_events,
)
from src.mind_dataset import (
    MIND_DOC_METADATA_FILENAME,
    MIND_EVENTS_FILENAME,
    MIND_PEER_FUTURE_DOCS_FILENAME,
    MIND_PEER_ID_MAP_FILENAME,
    MIND_RANDOMIZED_EVENTS_FILENAME,
    MIND_SEEDED_SEARCHABLE_FILENAME,
    compose_article_text,
    iter_positive_clicks,
)


class MindHelpersTests(unittest.TestCase):
    def test_compose_article_text_uses_title_then_abstract(self) -> None:
        self.assertEqual(
            compose_article_text("Title", "Abstract"),
            "Title\n\nAbstract",
        )
        self.assertEqual(compose_article_text("Title", ""), "Title")

    def test_news_parser_accepts_missing_entity_columns(self) -> None:
        from src.mind_dataset import iter_news

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "news.tsv"
            path.write_text(
                "N1\tnews\tworld\tTitle\tAbstract\thttps://example.com\n",
                encoding="utf-8",
            )
            articles = list(iter_news(path))
        self.assertEqual(len(articles), 1)
        self.assertEqual(articles[0].text, "Title\n\nAbstract")

    def test_positive_click_parser_keeps_only_label_one(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "behaviors.tsv"
            path.write_text(
                "1\tU1\t11/13/2019 8:36:57 AM\tN0\tN1-0 N2-1 N3-0\n",
                encoding="utf-8",
            )
            clicks = list(iter_positive_clicks(path, split="train"))
        self.assertEqual(len(clicks), 1)
        self.assertEqual(clicks[0].clicked_news_id, "N2")
        self.assertEqual(clicks[0].impression_position, 1)


class MindResolverTests(unittest.TestCase):
    def setUp(self) -> None:
        self._saved_env = os.environ.get("LOW_COORD_DHT_DATA_ROOT")

    def tearDown(self) -> None:
        if self._saved_env is None:
            os.environ.pop("LOW_COORD_DHT_DATA_ROOT", None)
        else:
            os.environ["LOW_COORD_DHT_DATA_ROOT"] = self._saved_env

    def test_env_override_resolves_mind_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            os.environ["LOW_COORD_DHT_DATA_ROOT"] = tmp
            resolved, source = resolve_dataset_dir("mind", return_source=True)
            self.assertEqual(source, "env_override")
            self.assertEqual(resolved, Path(tmp) / "mind")

    def test_embedding_smoke_defaults_use_separate_output_dir(self) -> None:
        import argparse
        from compute_embeddings import _resolve_mind_defaults

        with tempfile.TemporaryDirectory() as tmp:
            os.environ["LOW_COORD_DHT_DATA_ROOT"] = tmp
            args = argparse.Namespace(
                metadata=None,
                searchable_queries=None,
                output_dir=None,
                mind_doc_metadata=None,
                limit_docs=100,
            )
            _metadata, _searchable, output_dir, _doc_metadata = _resolve_mind_defaults(args)
            self.assertEqual(output_dir, Path(tmp) / "mind" / "embeddings_smoke_100")


class MindPrecomputeTests(unittest.TestCase):
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

    def _write_raw_split(self, split: str, *, news_rows: list[str], behavior_rows: list[str]) -> None:
        split_dir = resolve_dataset_dir("mind") / "raw" / split
        split_dir.mkdir(parents=True, exist_ok=True)
        (split_dir / "news.tsv").write_text("".join(news_rows), encoding="utf-8")
        (split_dir / "behaviors.tsv").write_text("".join(behavior_rows), encoding="utf-8")

    def test_precompute_pipeline_combines_train_and_dev(self) -> None:
        from precompute_mind import precompute_mind_artifacts

        news_rows = [
            "N1\tnews\tworld\tShared title\tShared abstract\thttps://n1\t[]\t[]\n",
            "N2\tnews\ttech\tSolo title\tSolo abstract\thttps://n2\t[]\t[]\n",
            "N3\tnews\tarts\tUnused title\tUnused abstract\thttps://n3\t[]\t[]\n",
        ]
        self._write_raw_split(
            "train",
            news_rows=news_rows,
            behavior_rows=[
                "10\tUA\t11/13/2019 8:36:57 AM\t\tN1-1 N2-0\n",
                "11\tUA\t11/14/2019 9:00:00 AM\tN1\tN2-1\n",
            ],
        )
        self._write_raw_split(
            "dev",
            news_rows=news_rows,
            behavior_rows=[
                "20\tUB\t11/13/2019 8:40:00 AM\t\tN1-1 N3-0\n",
            ],
        )

        data_dir = resolve_dataset_dir("mind")
        summary = precompute_mind_artifacts(random_seed=7)

        with gzip.open(data_dir / MIND_PEER_ID_MAP_FILENAME, "rt", encoding="utf-8") as handle:
            peer_id_map = json.load(handle)
        self.assertEqual(peer_id_map, {"UA": 0, "UB": 1})

        with (data_dir / MIND_EVENTS_FILENAME).open("r", newline="", encoding="utf-8") as handle:
            event_rows = list(csv.DictReader(handle))
        self.assertEqual(len(event_rows), 3)
        self.assertEqual([row["target_doc_id"] for row in event_rows], ["N1", "N1", "N2"])
        self.assertEqual(event_rows[0]["query"], "Shared title\n\nShared abstract")
        self.assertEqual(event_rows[2]["query"], "Solo title\n\nSolo abstract")

        with gzip.open(data_dir / MIND_SEEDED_SEARCHABLE_FILENAME, "rt", newline="", encoding="utf-8") as handle:
            sidecar = {int(row["query_id"]): row["searchable"] for row in csv.DictReader(handle)}
        self.assertEqual(sidecar, {0: "true", 1: "true", 2: "false"})

        with gzip.open(data_dir / MIND_PEER_FUTURE_DOCS_FILENAME, "rt", encoding="utf-8") as handle:
            peer_docs = json.load(handle)
        self.assertEqual(peer_docs, {"0": ["N1", "N2"], "1": ["N1"]})

        with gzip.open(data_dir / MIND_DOC_METADATA_FILENAME, "rt", newline="", encoding="utf-8") as handle:
            doc_rows = list(csv.DictReader(handle))
        self.assertEqual({row["news_id"] for row in doc_rows}, {"N1", "N2", "N3"})
        self.assertEqual(
            {row["text"] for row in doc_rows},
            {
                "Shared title\n\nShared abstract",
                "Solo title\n\nSolo abstract",
                "Unused title\n\nUnused abstract",
            },
        )

        randomized_path = data_dir / MIND_RANDOMIZED_EVENTS_FILENAME
        events = load_query_events(
            randomized_path,
            searchable_only=True,
            require_searchable_column=True,
            searchable_queries_path=data_dir / MIND_SEEDED_SEARCHABLE_FILENAME,
        )
        self.assertEqual(sorted(event.qid for event in events), [0, 1])

        chronological_path = data_dir / CHRONOLOGICAL_SEARCHABLE_QUERIES_FILENAME
        self.assertTrue(chronological_path.exists())
        self.assertEqual(summary["positive_event_count"], 3)
        self.assertEqual(summary["article_count"], 3)
        self.assertEqual(summary["unique_clicked_doc_count"], 2)


if __name__ == "__main__":
    unittest.main()

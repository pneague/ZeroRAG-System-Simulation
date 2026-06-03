"""Constants for the PubMedQA RAG simulator dataset."""

from __future__ import annotations

PUBMEDQA_EVENTS_FILENAME = "events.csv"
PUBMEDQA_RANDOMIZED_EVENTS_FILENAME = "metadata_seeded_randomized.csv.gz"
PUBMEDQA_SEEDED_SEARCHABLE_FILENAME = "seeded_searchable_queries.csv.gz"
PUBMEDQA_PEER_FUTURE_DOCS_FILENAME = "peer_future_doc_ids.json.gz"
PUBMEDQA_PEER_FUTURE_DOCS_SUMMARY_FILENAME = "peer_future_doc_ids_summary.json"
PUBMEDQA_DOC_TOPICS_FILENAME = "doc_topics.csv.gz"
PUBMEDQA_QUERY_TOPICS_FILENAME = "query_topics.csv.gz"
PUBMEDQA_TOPIC_MAP_FILENAME = "mesh_topic_map.json"
PUBMEDQA_ANSWER_TARGETS_FILENAME = "answer_targets.jsonl"
PUBMEDQA_PRECOMPUTE_SUMMARY_FILENAME = "pubmedqa_precompute_summary.json"

DEFAULT_PUBMEDQA_USER_COUNT = 20_000
DEFAULT_PUBMEDQA_QUERIES_PER_USER = 25
DEFAULT_PUBMEDQA_DIRICHLET_ALPHA = 0.01
DEFAULT_PUBMEDQA_RANDOM_SEED = 42


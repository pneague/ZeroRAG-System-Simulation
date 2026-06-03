#!/usr/bin/env python3
"""Compute query and document embeddings for the Low_Coord_Semantic_DHT pipeline.

Supports three datasets:

* ``aol4foltr`` (default) - documents are pulled from the configured
  ``ir_datasets`` corpus (typically ``aol-ia``).
* ``amazon`` - documents are loaded from the per-category
  ``doc_metadata.csv.gz`` produced by :mod:`precompute_amazon`. The Amazon
  benchmark builds events from real user-item purchases/reviews + synthetic
  generated queries + UCSD product metadata joined by ``asin``; we never
  describe Amazon events as 'user searched for product'.
* ``mind`` - documents are loaded from ``doc_metadata.csv.gz`` produced by
  :mod:`precompute_mind`. Queries are exactly the clicked article title+abstract
  text, so MIND query embeddings and document embeddings are computed from the
  same composed article strings.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.amazon_dataset import (
    AMAZON_CATEGORY_SPECS,
    AMAZON_DOC_METADATA_FILENAME,
    AMAZON_EVENTS_FILENAME,
    AMAZON_SEEDED_SEARCHABLE_FILENAME,
    compose_doc_text,
)
from src.data_root import resolve_dataset_dir
from src.mind_dataset import (
    MIND_DOC_METADATA_FILENAME,
    MIND_EVENTS_FILENAME,
    MIND_SEEDED_SEARCHABLE_FILENAME,
)


def _require_ir_datasets():
    try:
        import ir_datasets
    except ImportError as exc:
        raise RuntimeError(
            "The ir_datasets package is required for AOL-IA document access. "
            "Install it with 'pip install ir-datasets>=0.5.8'."
        ) from exc
    return ir_datasets


def _iter_batches(values: list[str], batch_size: int):
    for start in range(0, len(values), batch_size):
        end = min(len(values), start + batch_size)
        yield start, end, values[start:end]


def _normalize_doc_text(doc) -> str:
    parts: list[str] = []
    for field_name in ("title", "body", "text", "url"):
        value = getattr(doc, field_name, None)
        if value:
            parts.append(str(value).strip())
    return "\n\n".join(part for part in parts if part)


def _load_aol_docstore(dataset_name: str):
    ir_datasets = _require_ir_datasets()
    dataset = ir_datasets.load(dataset_name)
    return dataset, dataset.docs_store()


def _collect_all_aol_doc_ids(dataset_name: str) -> list[str]:
    ir_datasets = _require_ir_datasets()
    dataset = ir_datasets.load(dataset_name)
    return [str(doc.doc_id) for doc in dataset.docs_iter()]


def _load_amazon_doc_texts(metadata_csv_path: Path) -> dict[str, str]:
    with gzip.open(metadata_csv_path, "rt", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        texts: dict[str, str] = {}
        for row in reader:
            asin = (row.get("asin") or "").strip()
            if not asin:
                continue
            texts[asin] = compose_doc_text(row)
    return texts


class _AmazonDocstoreAdapter:
    """Tiny adapter that mimics the ``ir_datasets`` ``docstore.get`` interface."""

    class _Doc:
        __slots__ = ("doc_id", "text")

        def __init__(self, doc_id: str, text: str) -> None:
            self.doc_id = doc_id
            self.text = text

    def __init__(self, texts: dict[str, str]) -> None:
        self._texts = texts

    def get(self, doc_id: str):
        text = self._texts.get(doc_id)
        if text is None:
            return None
        return self._Doc(doc_id, text)


def _load_mind_doc_texts(metadata_csv_path: Path) -> dict[str, str]:
    with gzip.open(metadata_csv_path, "rt", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        texts: dict[str, str] = {}
        for row in reader:
            news_id = (row.get("news_id") or "").strip()
            if not news_id:
                continue
            texts[news_id] = (row.get("text") or "").strip()
    return texts


class _MindDocstoreAdapter:
    """Tiny adapter that mimics the ``ir_datasets`` ``docstore.get`` interface."""

    class _Doc:
        __slots__ = ("doc_id", "text")

        def __init__(self, doc_id: str, text: str) -> None:
            self.doc_id = doc_id
            self.text = text

    def __init__(self, texts: dict[str, str]) -> None:
        self._texts = texts

    def get(self, doc_id: str):
        text = self._texts.get(doc_id)
        if text is None:
            return None
        return self._Doc(doc_id, text)


def _embed_queries(
    query_texts: list[str],
    *,
    embedder,
    output_dir: Path,
) -> dict[str, str | int]:
    import numpy as np
    from numpy.lib.format import open_memmap
    from tqdm import tqdm

    keys_path = output_dir / "query_texts.npy"
    embeddings_path = output_dir / "query_embeddings.npy"
    np.save(keys_path, np.asarray(query_texts, dtype=str))
    embeddings = open_memmap(
        embeddings_path,
        mode="w+",
        dtype=np.float32,
        shape=(len(query_texts), embedder.embedding_dim),
    )
    for start, end, batch in tqdm(
        _iter_batches(query_texts, embedder.batch_size),
        total=math.ceil(len(query_texts) / embedder.batch_size) if query_texts else 0,
        desc="Embedding queries",
    ):
        embeddings[start:end] = embedder.encode(batch, show_progress_bar=False)
    del embeddings
    return {
        "query_keys_path": str(keys_path),
        "query_embeddings_path": str(embeddings_path),
        "query_count": len(query_texts),
    }


def _embed_documents(
    doc_ids: list[str],
    *,
    embedder,
    docstore,
    output_dir: Path,
) -> dict[str, str | int | list[str]]:
    import numpy as np
    from numpy.lib.format import open_memmap
    from tqdm import tqdm

    keys_path = output_dir / "doc_ids.npy"
    embeddings_path = output_dir / "doc_embeddings.npy"
    missing_path = output_dir / "missing_doc_ids.json"

    np.save(keys_path, np.asarray(doc_ids, dtype=str))
    embeddings = open_memmap(
        embeddings_path,
        mode="w+",
        dtype=np.float32,
        shape=(len(doc_ids), embedder.embedding_dim),
    )

    missing_doc_ids: list[str] = []
    for start, end, batch_doc_ids in tqdm(
        _iter_batches(doc_ids, embedder.batch_size),
        total=math.ceil(len(doc_ids) / embedder.batch_size) if doc_ids else 0,
        desc="Embedding documents",
    ):
        texts: list[str] = []
        missing_positions: list[int] = []
        for doc_id in batch_doc_ids:
            doc = docstore.get(doc_id)
            if doc is None:
                missing_doc_ids.append(doc_id)
                missing_positions.append(len(texts))
                texts.append("")
                continue
            text = _normalize_doc_text(doc) if not hasattr(doc, "text") or not isinstance(doc.text, str) else doc.text
            if not text:
                missing_doc_ids.append(doc_id)
                missing_positions.append(len(texts))
            texts.append(text)

        batch_vectors = embedder.encode(texts, show_progress_bar=False)
        for position in missing_positions:
            batch_vectors[position] = 0.0
        embeddings[start:end] = batch_vectors

    del embeddings
    missing_path.write_text(json.dumps(missing_doc_ids, indent=2), encoding="utf-8")
    return {
        "doc_keys_path": str(keys_path),
        "doc_embeddings_path": str(embeddings_path),
        "doc_count": len(doc_ids),
        "missing_doc_ids_path": str(missing_path),
        "missing_doc_count": len(missing_doc_ids),
    }


def _resolve_aol_defaults(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    aol_dir = resolve_dataset_dir("aol4foltr")
    metadata_path = Path(args.metadata) if args.metadata else aol_dir / "metadata.csv"
    searchable_queries = (
        Path(args.searchable_queries)
        if args.searchable_queries
        else aol_dir / "chronological_searchable_queries.csv.gz"
    )
    output_dir = Path(args.output_dir) if args.output_dir else aol_dir / "embeddings"
    return metadata_path, searchable_queries, output_dir


def _resolve_amazon_defaults(args: argparse.Namespace) -> tuple[Path, Path, Path, Path]:
    if args.amazon_category not in AMAZON_CATEGORY_SPECS:
        raise SystemExit(
            f"--amazon-category must be one of {sorted(AMAZON_CATEGORY_SPECS)}"
        )
    amazon_dir = resolve_dataset_dir("amazon", args.amazon_category)
    metadata_path = (
        Path(args.metadata) if args.metadata else amazon_dir / AMAZON_EVENTS_FILENAME
    )
    searchable_queries = (
        Path(args.searchable_queries)
        if args.searchable_queries
        else amazon_dir / AMAZON_SEEDED_SEARCHABLE_FILENAME
    )
    output_dir = Path(args.output_dir) if args.output_dir else amazon_dir / "embeddings"
    doc_metadata_path = (
        Path(args.amazon_doc_metadata)
        if args.amazon_doc_metadata
        else amazon_dir / AMAZON_DOC_METADATA_FILENAME
    )
    return metadata_path, searchable_queries, output_dir, doc_metadata_path


def _resolve_mind_defaults(args: argparse.Namespace) -> tuple[Path, Path, Path, Path]:
    mind_dir = resolve_dataset_dir("mind")
    metadata_path = Path(args.metadata) if args.metadata else mind_dir / MIND_EVENTS_FILENAME
    searchable_queries = (
        Path(args.searchable_queries)
        if args.searchable_queries
        else mind_dir / MIND_SEEDED_SEARCHABLE_FILENAME
    )
    if args.output_dir:
        output_dir = Path(args.output_dir)
    elif args.limit_docs:
        output_dir = mind_dir / f"embeddings_smoke_{args.limit_docs}"
    else:
        output_dir = mind_dir / "embeddings"
    doc_metadata_path = (
        Path(args.mind_doc_metadata)
        if args.mind_doc_metadata
        else mind_dir / MIND_DOC_METADATA_FILENAME
    )
    return metadata_path, searchable_queries, output_dir, doc_metadata_path


def main() -> None:
    default_model = "sentence-transformers/all-mpnet-base-v2"

    parser = argparse.ArgumentParser(
        description="Compute query and document embeddings for AOL4FOLTR, Amazon, or MIND"
    )
    parser.add_argument(
        "--dataset",
        choices=("aol4foltr", "amazon", "mind"),
        default="aol4foltr",
        help="Which dataset to embed (default: aol4foltr)",
    )
    parser.add_argument(
        "--amazon-category",
        choices=sorted(AMAZON_CATEGORY_SPECS),
        default="cell_phones",
        help="Amazon benchmark category (used when --dataset=amazon)",
    )
    parser.add_argument(
        "--metadata",
        default=None,
        help=(
            "Override the metadata CSV(.gz) path; default is resolved per dataset "
            "via src.data_root.resolve_dataset_dir"
        ),
    )
    parser.add_argument(
        "--searchable-queries",
        default=None,
        help="Searchable sidecar CSV(.gz) path; default resolved per dataset",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Embedding output directory; default is <data>/embeddings",
    )
    parser.add_argument(
        "--amazon-doc-metadata",
        default=None,
        help=(
            "Override the Amazon doc_metadata.csv.gz path; default resolved "
            "per dataset"
        ),
    )
    parser.add_argument(
        "--mind-doc-metadata",
        default=None,
        help=(
            "Override the MIND doc_metadata.csv.gz path; default resolved "
            "per dataset"
        ),
    )
    parser.add_argument("--model", default=default_model)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default=None, help="Torch device (e.g. cpu, cuda)")
    parser.add_argument(
        "--doc-dataset",
        default="aol-ia",
        help="ir_datasets identifier for AOL document contents (ignored for amazon and mind)",
    )
    parser.add_argument(
        "--searchable-only",
        action="store_true",
        help="Only embed query/doc ids referenced by searchable events",
    )
    parser.add_argument(
        "--all-docs",
        action="store_true",
        help=(
            "AOL only: embed the full ir_datasets corpus instead of the metadata "
            "doc ids. For Amazon and MIND, all docs in doc_metadata.csv.gz are "
            "embedded regardless of this flag."
        ),
    )
    parser.add_argument(
        "--limit-docs",
        type=int,
        default=0,
        help=(
            "MIND only: cap document embedding to the first N article ids and "
            "embed the corresponding article texts as queries. Intended for "
            "remote smoke tests."
        ),
    )
    parser.add_argument(
        "--queries-only",
        action="store_true",
        help="Only embed queries and skip document embedding",
    )
    parser.add_argument(
        "--documents-only",
        action="store_true",
        help="Only embed documents and skip query embedding",
    )
    args = parser.parse_args()
    if args.queries_only and args.documents_only:
        parser.error("--queries-only and --documents-only cannot be used together")

    from src.dataset import collect_unique_queries_and_doc_ids
    from src.embedder import SentenceTransformerEmbedder

    if args.limit_docs < 0:
        parser.error("--limit-docs must be non-negative")
    if args.limit_docs and args.dataset != "mind":
        parser.error("--limit-docs is only supported with --dataset mind")

    if args.dataset == "amazon":
        metadata_path, searchable_queries, output_dir, doc_metadata_path = (
            _resolve_amazon_defaults(args)
        )
    elif args.dataset == "mind":
        metadata_path, searchable_queries, output_dir, doc_metadata_path = (
            _resolve_mind_defaults(args)
        )
    else:
        metadata_path, searchable_queries, output_dir = _resolve_aol_defaults(args)
        doc_metadata_path = None

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Dataset: {args.dataset}")
    print(f"Metadata path: {metadata_path}")
    print(f"Searchable queries: {searchable_queries}")
    print(f"Output dir: {output_dir}")

    print(f"Collecting unique queries and doc ids from {metadata_path}")
    query_texts, doc_ids = collect_unique_queries_and_doc_ids(
        metadata_path,
        searchable_only=args.searchable_only,
        include_candidates=True,
        require_searchable_column=args.searchable_only,
        searchable_queries_path=searchable_queries,
    )

    if args.dataset == "amazon":
        amazon_doc_texts = _load_amazon_doc_texts(doc_metadata_path)
        print(
            f"Loaded {len(amazon_doc_texts):,} product texts from {doc_metadata_path}"
        )
        # Embed every asin we have a metadata row for so the simulator's lookup
        # is complete; --all-docs is honored implicitly.
        doc_ids = sorted(amazon_doc_texts)
        docstore = _AmazonDocstoreAdapter(amazon_doc_texts)
    elif args.dataset == "mind":
        mind_doc_texts = _load_mind_doc_texts(doc_metadata_path)
        print(f"Loaded {len(mind_doc_texts):,} article texts from {doc_metadata_path}")
        doc_ids = sorted(mind_doc_texts)
        if args.limit_docs:
            doc_ids = doc_ids[: args.limit_docs]
            # MIND queries are exactly the clicked article text. For smoke runs,
            # limit query embeddings to the same article texts as the doc cap.
            query_texts = sorted({mind_doc_texts[doc_id] for doc_id in doc_ids})
            print(
                f"MIND smoke cap active: embedding {len(doc_ids):,} docs and "
                f"{len(query_texts):,} corresponding query texts"
            )
        docstore = _MindDocstoreAdapter(mind_doc_texts)
    else:
        docstore = None
        if args.all_docs:
            print(f"Collecting all document ids from {args.doc_dataset}")
            doc_ids = _collect_all_aol_doc_ids(args.doc_dataset)

    print(f"Unique query texts: {len(query_texts):,}")
    print(f"Unique doc ids: {len(doc_ids):,}")

    embedder = SentenceTransformerEmbedder(
        model_name=args.model,
        batch_size=args.batch_size,
        device=args.device,
    )

    manifest: dict[str, object] = {
        "dataset": args.dataset,
        "amazon_category": args.amazon_category if args.dataset == "amazon" else None,
        "metadata_path": str(metadata_path),
        "searchable_queries_path": str(searchable_queries) if searchable_queries else None,
        "output_dir": str(output_dir),
        "model": args.model,
        "embedding_dim": embedder.embedding_dim,
        "batch_size": args.batch_size,
        "device": args.device,
        "doc_dataset": args.doc_dataset if args.dataset == "aol4foltr" else None,
        "amazon_doc_metadata_path": str(doc_metadata_path) if args.dataset == "amazon" else None,
        "mind_doc_metadata_path": str(doc_metadata_path) if args.dataset == "mind" else None,
        "limit_docs": args.limit_docs,
        "searchable_only": args.searchable_only,
        "all_docs": args.all_docs,
        "queries_only": args.queries_only,
        "documents_only": args.documents_only,
    }
    if not args.documents_only:
        manifest.update(_embed_queries(query_texts, embedder=embedder, output_dir=output_dir))
    if not args.queries_only:
        if args.dataset == "amazon":
            manifest.update(
                _embed_documents(doc_ids, embedder=embedder, docstore=docstore, output_dir=output_dir)
            )
        elif args.dataset == "mind":
            manifest.update(
                _embed_documents(doc_ids, embedder=embedder, docstore=docstore, output_dir=output_dir)
            )
        else:
            _, ir_docstore = _load_aol_docstore(args.doc_dataset)
            manifest.update(
                _embed_documents(doc_ids, embedder=embedder, docstore=ir_docstore, output_dir=output_dir)
            )

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Embedding manifest written to {manifest_path}")


if __name__ == "__main__":
    main()

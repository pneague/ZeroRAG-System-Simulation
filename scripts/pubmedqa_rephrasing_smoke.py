#!/usr/bin/env python3
"""Run the PubMedQA p40-p60 MeSH rephrasing smoke test.

The script is intentionally resumable:

1. Build the eligible and 500-row smoke manifests.
2. Ask an OpenAI chat model for five rephrasings per smoke row.
3. Validate outputs.
4. Locally embed the 500 docs, originals, and rephrasings.
5. Report nearest-document drift for the rephrased queries.

The OpenAI API key is read from OPENAI_API_KEY, or from
``<output-dir>/openai_api_key`` when the env var is unset.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import requests
from datasets import load_from_disk
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.embedder import DEFAULT_EMBEDDING_MODEL, SentenceTransformerEmbedder


PUBMEDQA_CONFIGS = ("pqa_labeled", "pqa_unlabeled", "pqa_artificial")
DEFAULT_DATA_ROOT = REPO_ROOT / "data" / "pubmedqa_rag" / "hf"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "pubmedqa_rag" / "rephrasing_smoke"
DEFAULT_MODEL = "gpt-5.4-mini"
SYSTEM_PROMPT = (
    "You rewrite biomedical questions while preserving their meaning. "
    "Return only valid JSON."
)
USER_PROMPT_PREFIX = (
    "Given the PubMedQA question below, produce exactly 5 distinct rephrasings "
    "that preserve the original biomedical meaning. Do not answer the question. "
    'Return JSON with key "rephrasings" whose value is an array of 5 strings.'
    "\n\nQuestion: "
)


@dataclass(frozen=True)
class PubMedQaRow:
    config: str
    row_index: int
    pubid: str
    question: str
    doc_text: str
    meshes: tuple[str, ...]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")


def _percentile(values: list[int], p: float) -> float:
    if not values:
        raise ValueError("Cannot compute percentile of an empty sequence")
    ordered = sorted(values)
    k = (len(ordered) - 1) * (p / 100.0)
    lower = int(math.floor(k))
    upper = min(lower + 1, len(ordered) - 1)
    if lower == upper:
        return float(ordered[lower])
    return ordered[lower] + (k - lower) * (ordered[upper] - ordered[lower])


def _clean_str(value: object) -> str:
    return str(value or "").strip()


def _compose_doc_text(context: dict[str, Any]) -> str:
    contexts = context.get("contexts") or []
    labels = context.get("labels") or []
    sections: list[str] = []
    for index, text in enumerate(contexts):
        text_value = _clean_str(text)
        if not text_value:
            continue
        label = _clean_str(labels[index]) if index < len(labels) else ""
        sections.append(f"{label}: {text_value}" if label else text_value)
    return "\n\n".join(sections)


def _load_pubmedqa_rows(data_root: Path) -> list[PubMedQaRow]:
    rows: list[PubMedQaRow] = []
    for config in PUBMEDQA_CONFIGS:
        dataset = load_from_disk(str(data_root / config))
        for row_index, row in enumerate(dataset):
            context = row.get("context") or {}
            meshes = tuple(
                sorted(
                    {
                        _clean_str(mesh)
                        for mesh in (context.get("meshes") or [])
                        if _clean_str(mesh)
                    }
                )
            )
            rows.append(
                PubMedQaRow(
                    config=config,
                    row_index=row_index,
                    pubid=_clean_str(row.get("pubid")),
                    question=_clean_str(row.get("question")),
                    doc_text=_compose_doc_text(context),
                    meshes=meshes,
                )
            )
    return rows


def _manifest_row(
    row: PubMedQaRow,
    *,
    order_index: int,
    matching_meshes: Iterable[str],
) -> dict[str, Any]:
    return {
        "order_index": order_index,
        "config": row.config,
        "row_index": row.row_index,
        "pubid": row.pubid,
        "question": row.question,
        "doc_text": row.doc_text,
        "matching_p40_p60_meshes": sorted(matching_meshes),
        "meshes": list(row.meshes),
    }


def build_manifests(args: argparse.Namespace) -> dict[str, Any]:
    rows = _load_pubmedqa_rows(args.data_root)
    mesh_doc_counts: Counter[str] = Counter()
    for row in rows:
        mesh_doc_counts.update(row.meshes)

    p40 = _percentile(list(mesh_doc_counts.values()), 40)
    p60 = _percentile(list(mesh_doc_counts.values()), 60)
    selected_meshes = {
        mesh for mesh, count in mesh_doc_counts.items() if p40 <= count <= p60
    }

    eligible: list[dict[str, Any]] = []
    for order_index, row in enumerate(rows):
        matching = set(row.meshes) & selected_meshes
        if matching:
            eligible.append(
                _manifest_row(row, order_index=order_index, matching_meshes=matching)
            )

    smoke = eligible[: args.smoke_size]
    eligible_path = args.output_dir / "eligible_manifest.jsonl"
    smoke_path = args.output_dir / f"smoke_manifest_{args.smoke_size}.jsonl"
    summary_path = args.output_dir / "manifest_summary.json"

    _write_jsonl(eligible_path, eligible)
    _write_jsonl(smoke_path, smoke)
    summary = {
        "data_root": str(args.data_root),
        "eligible_manifest_path": str(eligible_path),
        "smoke_manifest_path": str(smoke_path),
        "total_documents": len(rows),
        "unique_mesh_terms": len(mesh_doc_counts),
        "p40_doc_frequency": p40,
        "p60_doc_frequency": p60,
        "selected_mesh_terms": len(selected_meshes),
        "eligible_documents": len(eligible),
        "smoke_documents": len(smoke),
    }
    _write_json(summary_path, summary)
    return summary


def _strip_json_fences(text: str) -> str:
    stripped = text.strip()
    match = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", stripped, flags=re.DOTALL)
    return match.group(1).strip() if match else stripped


def _validate_rephrasings(content: str) -> list[str]:
    payload = json.loads(_strip_json_fences(content))
    rephrasings = payload.get("rephrasings")
    if not isinstance(rephrasings, list):
        raise ValueError("Missing JSON array field: rephrasings")
    cleaned = [_clean_str(item) for item in rephrasings]
    cleaned = [item for item in cleaned if item]
    if len(cleaned) != 5:
        raise ValueError(f"Expected exactly 5 non-empty rephrasings, got {len(cleaned)}")
    return cleaned


def _openai_headers(api_key: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }


def _verify_model(api_key: str, model: str, *, timeout_seconds: int) -> None:
    url = f"https://api.openai.com/v1/models/{model}"
    response = requests.get(url, headers=_openai_headers(api_key), timeout=timeout_seconds)
    if response.status_code == 404:
        raise RuntimeError(
            f"OpenAI model {model!r} was not found for this API key. "
            "Stop here and choose the exact available model identifier before substituting."
        )
    if response.status_code >= 400:
        raise RuntimeError(
            f"Model verification failed for {model!r}: "
            f"HTTP {response.status_code}: {response.text[:500]}"
        )


def _chat_completion(
    *,
    api_key: str,
    model: str,
    question: str,
    timeout_seconds: int,
    max_completion_tokens: int,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT_PREFIX + question},
        ],
        "response_format": {"type": "json_object"},
        "max_completion_tokens": max_completion_tokens,
    }
    response = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers=_openai_headers(api_key),
        json=payload,
        timeout=timeout_seconds,
    )
    if response.status_code >= 400:
        raise RuntimeError(f"HTTP {response.status_code}: {response.text[:1000]}")
    return response.json()


def _extract_content(response: dict[str, Any]) -> str:
    try:
        return str(response["choices"][0]["message"]["content"])
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError(f"Unexpected OpenAI response shape: {response}") from exc


def _token_estimate(text: str, *, encoding_name: str = "cl100k_base") -> int:
    try:
        import tiktoken

        encoding = tiktoken.get_encoding(encoding_name)
        return len(encoding.encode(text))
    except Exception:
        return max(1, round(len(text) / 4))


def _estimate_request_tokens(question: str) -> int:
    return (
        _token_estimate(SYSTEM_PROMPT)
        + _token_estimate(USER_PROMPT_PREFIX + question)
        + 10
    )


def _estimate_output_tokens(question: str) -> int:
    return 5 * _token_estimate(question) + 15


def _resolve_api_key(output_dir: Path) -> str:
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if api_key:
        return api_key
    key_path = output_dir / "openai_api_key"
    if key_path.is_file():
        return key_path.read_text(encoding="utf-8").strip()
    raise RuntimeError(
        "No OpenAI API key found. Set OPENAI_API_KEY or create "
        f"{key_path} with the key on a single line."
    )


def run_rephrasing(args: argparse.Namespace) -> dict[str, Any]:
    api_key = _resolve_api_key(args.output_dir)

    _verify_model(api_key, args.model, timeout_seconds=args.timeout_seconds)

    manifest_path = args.output_dir / f"smoke_manifest_{args.smoke_size}.jsonl"
    rows = _read_jsonl(manifest_path)
    if not rows:
        raise RuntimeError(f"Smoke manifest is empty or missing: {manifest_path}")

    raw_path = args.output_dir / "raw_openai_responses.jsonl"
    output_path = args.output_dir / "rephrased_queries.jsonl"
    failures_path = args.output_dir / "failed_rephrasing_rows.jsonl"
    summary_path = args.output_dir / "rephrasing_summary.json"

    completed = {row["pubid"] for row in _read_jsonl(output_path)}
    failures: list[dict[str, Any]] = []
    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0
    estimated_input_tokens = 0
    estimated_output_tokens = 0

    for row in tqdm(rows, desc="Rephrasing PubMedQA smoke rows"):
        pubid = row["pubid"]
        question = row["question"]
        estimated_input_tokens += _estimate_request_tokens(question)
        estimated_output_tokens += _estimate_output_tokens(question)
        if pubid in completed:
            continue

        last_error = ""
        for attempt in range(1, args.max_retries + 2):
            try:
                response = _chat_completion(
                    api_key=api_key,
                    model=args.model,
                    question=question,
                    timeout_seconds=args.timeout_seconds,
                    max_completion_tokens=args.max_completion_tokens,
                )
                content = _extract_content(response)
                rephrasings = _validate_rephrasings(content)
                usage = response.get("usage") or {}
                prompt_tokens += int(usage.get("prompt_tokens") or 0)
                completion_tokens += int(usage.get("completion_tokens") or 0)
                total_tokens += int(usage.get("total_tokens") or 0)
                _append_jsonl(
                    raw_path,
                    {
                        "pubid": pubid,
                        "response": response,
                    },
                )
                _append_jsonl(
                    output_path,
                    {
                        "config": row["config"],
                        "row_index": row["row_index"],
                        "order_index": row["order_index"],
                        "pubid": pubid,
                        "question": question,
                        "rephrasings": rephrasings,
                        "matching_p40_p60_meshes": row["matching_p40_p60_meshes"],
                    },
                )
                completed.add(pubid)
                time.sleep(args.request_delay_seconds)
                break
            except Exception as exc:  # noqa: BLE001 - capture row-level retry details.
                last_error = str(exc)
                if attempt > args.max_retries:
                    failure = {
                        "pubid": pubid,
                        "question": question,
                        "error": last_error,
                    }
                    failures.append(failure)
                    _append_jsonl(failures_path, failure)
                    break
                time.sleep(min(60.0, args.retry_delay_seconds * attempt))

    valid_rows = _read_jsonl(output_path)
    failed_rows = _read_jsonl(failures_path)
    summary = {
        "model": args.model,
        "smoke_manifest_path": str(manifest_path),
        "raw_responses_path": str(raw_path),
        "rephrased_queries_path": str(output_path),
        "failed_rows_path": str(failures_path),
        "requested_rows": len(rows),
        "completed_rows": len(valid_rows),
        "failed_rows": len(failed_rows),
        "api_prompt_tokens": prompt_tokens,
        "api_completion_tokens": completion_tokens,
        "api_total_tokens": total_tokens,
        "estimated_input_tokens": estimated_input_tokens,
        "estimated_output_tokens": estimated_output_tokens,
        "estimated_total_tokens": estimated_input_tokens + estimated_output_tokens,
    }
    _write_json(summary_path, summary)
    return summary


def _load_smoke_manifest_by_pubid(args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    manifest_path = args.output_dir / f"smoke_manifest_{args.smoke_size}.jsonl"
    rows = _read_jsonl(manifest_path)
    return {row["pubid"]: row for row in rows}


def _encode_texts(
    texts: list[str],
    *,
    embedder: SentenceTransformerEmbedder,
    desc: str,
) -> np.ndarray:
    batches: list[np.ndarray] = []
    for start in tqdm(range(0, len(texts), embedder.batch_size), desc=desc):
        batch = texts[start : start + embedder.batch_size]
        batches.append(embedder.encode(batch, show_progress_bar=False))
    return np.vstack(batches).astype(np.float32) if batches else np.zeros((0, 0), dtype=np.float32)


def _normalize_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for row in rows:
        rephrasings = row.get("rephrasings")
        if not isinstance(rephrasings, list) or len(rephrasings) != 5:
            raise ValueError(f"Invalid rephrasing row for pubid={row.get('pubid')}")
        normalized.append(row)
    return normalized


def run_embedding_report(args: argparse.Namespace) -> dict[str, Any]:
    manifest_by_pubid = _load_smoke_manifest_by_pubid(args)
    rephrased_path = args.output_dir / "rephrased_queries.jsonl"
    rephrased_rows = _normalize_rows(_read_jsonl(rephrased_path))
    if len(rephrased_rows) != args.smoke_size:
        raise RuntimeError(
            f"Expected {args.smoke_size} validated rows before embedding, "
            f"found {len(rephrased_rows)} in {rephrased_path}"
        )

    doc_pubids = [row["pubid"] for row in rephrased_rows]
    doc_texts = [manifest_by_pubid[pubid]["doc_text"] for pubid in doc_pubids]
    original_questions = [row["question"] for row in rephrased_rows]
    rephrased_items: list[dict[str, Any]] = []
    rephrased_texts: list[str] = []
    for row in rephrased_rows:
        for index, text in enumerate(row["rephrasings"]):
            rephrased_items.append(
                {
                    "source_pubid": row["pubid"],
                    "source_question": row["question"],
                    "rephrase_index": index,
                    "rephrased_question": text,
                }
            )
            rephrased_texts.append(text)

    embedder = SentenceTransformerEmbedder(
        model_name=args.embedding_model,
        batch_size=args.embedding_batch_size,
        device=args.device,
    )

    doc_embeddings = _encode_texts(doc_texts, embedder=embedder, desc="Embedding docs")
    original_embeddings = _encode_texts(
        original_questions,
        embedder=embedder,
        desc="Embedding original questions",
    )
    rephrased_embeddings = _encode_texts(
        rephrased_texts,
        embedder=embedder,
        desc="Embedding rephrased questions",
    )

    embedding_dir = args.output_dir / "embeddings"
    embedding_dir.mkdir(parents=True, exist_ok=True)
    np.save(embedding_dir / "doc_pubids.npy", np.asarray(doc_pubids, dtype=str))
    np.save(embedding_dir / "doc_embeddings.npy", doc_embeddings)
    np.save(embedding_dir / "original_question_embeddings.npy", original_embeddings)
    np.save(embedding_dir / "rephrased_question_embeddings.npy", rephrased_embeddings)

    source_index = {pubid: index for index, pubid in enumerate(doc_pubids)}
    similarities = rephrased_embeddings @ doc_embeddings.T
    nearest_indices = similarities.argmax(axis=1)

    results_path = args.output_dir / "rephrased_closeness_results.jsonl"
    drifted = 0
    evaluated = 0
    examples: list[dict[str, Any]] = []
    with results_path.open("w", encoding="utf-8") as handle:
        for item_index, item in enumerate(rephrased_items):
            source_pubid = item["source_pubid"]
            source_doc_index = source_index[source_pubid]
            nearest_index = int(nearest_indices[item_index])
            nearest_pubid = doc_pubids[nearest_index]
            source_similarity = float(similarities[item_index, source_doc_index])
            nearest_similarity = float(similarities[item_index, nearest_index])
            is_drift = nearest_pubid != source_pubid
            drifted += int(is_drift)
            evaluated += 1
            result = {
                **item,
                "nearest_pubid": nearest_pubid,
                "nearest_similarity": nearest_similarity,
                "source_similarity": source_similarity,
                "margin": nearest_similarity - source_similarity,
                "nearest_is_different_doc": is_drift,
            }
            handle.write(json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n")
            if is_drift:
                examples.append(result)

    original_similarities = original_embeddings @ doc_embeddings.T
    original_nearest = original_similarities.argmax(axis=1)
    original_drift = int(
        sum(doc_pubids[int(nearest)] != doc_pubids[index] for index, nearest in enumerate(original_nearest))
    )

    examples = sorted(examples, key=lambda row: row["margin"], reverse=True)[: args.example_count]
    report = {
        "embedding_model": args.embedding_model,
        "embedding_dir": str(embedding_dir),
        "rephrased_results_path": str(results_path),
        "documents_embedded": len(doc_texts),
        "original_questions_embedded": len(original_questions),
        "rephrased_questions_embedded": len(rephrased_texts),
        "rephrased_queries_evaluated": evaluated,
        "rephrased_queries_closer_to_different_doc": drifted,
        "rephrased_queries_closer_to_different_doc_fraction": drifted / evaluated if evaluated else 0.0,
        "original_questions_evaluated": len(original_questions),
        "original_questions_closer_to_different_doc": original_drift,
        "original_questions_closer_to_different_doc_fraction": (
            original_drift / len(original_questions) if original_questions else 0.0
        ),
        "top_drift_examples": examples,
    }
    _write_json(args.output_dir / "closeness_report.json", report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("manifests", "rephrase", "embed-report", "all"),
        default="all",
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--smoke-size", type=int, default=500)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--timeout-seconds", type=int, default=90)
    parser.add_argument("--max-completion-tokens", type=int, default=512)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--request-delay-seconds", type=float, default=0.1)
    parser.add_argument("--retry-delay-seconds", type=float, default=2.0)
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--embedding-batch-size", type=int, default=32)
    parser.add_argument("--device", default=None)
    parser.add_argument("--example-count", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summaries: dict[str, Any] = {}
    if args.stage in {"manifests", "all"}:
        summaries["manifest"] = build_manifests(args)
    if args.stage in {"rephrase", "all"}:
        summaries["rephrasing"] = run_rephrasing(args)
    if args.stage in {"embed-report", "all"}:
        summaries["closeness"] = run_embedding_report(args)

    if summaries:
        _write_json(args.output_dir / "smoke_run_summary.json", summaries)
        print(json.dumps(summaries, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()

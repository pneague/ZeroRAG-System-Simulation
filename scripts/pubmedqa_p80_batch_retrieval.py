#!/usr/bin/env python3
"""PubMedQA p80 MeSH Batch API rephrasing and retrieval evaluation.

Pipeline:

1. Select the 500 MeSH terms starting at the 80th-percentile frequency rank.
2. Build an OpenAI Batch API input file asking for five rephrasings per question.
3. Submit/poll/download/validate the batch.
4. Embed selected docs, original questions, and rephrased questions with checkpoints.
5. Compute nearest-document mismatch rates for original and rephrased queries.

The OpenAI key is read from OPENAI_API_KEY or an explicit --api-key-file.
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
from typing import Any, Iterable, Sequence

import numpy as np
import requests
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.embedder import DEFAULT_EMBEDDING_MODEL, SentenceTransformerEmbedder

PUBMEDQA_CONFIGS = ("pqa_labeled", "pqa_unlabeled", "pqa_artificial")
DEFAULT_DATA_ROOT = REPO_ROOT / "data" / "pubmedqa_rag" / "hf"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "pubmedqa_rag" / "p80_500_batch_retrieval"
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
TERMINAL_BATCH_STATES = {"completed", "failed", "expired", "cancelled", "cancelling"}


@dataclass(frozen=True)
class PubMedQaRow:
    config: str
    row_index: int
    pubid: str
    question: str
    doc_text: str
    meshes: tuple[str, ...]


def _clean_str(value: object) -> str:
    return str(value or "").strip()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


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


def load_pubmedqa_rows(data_root: Path) -> list[PubMedQaRow]:
    from datasets import load_from_disk

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


def build_selection(args: argparse.Namespace) -> dict[str, Any]:
    selected_manifest_path = args.output_dir / "selected_manifest.jsonl"
    selected_meshes_path = args.output_dir / "selected_meshes.json"
    summary_path = args.output_dir / "selection_summary.json"

    rows = load_pubmedqa_rows(args.data_root)
    mesh_doc_counts: Counter[str] = Counter()
    for row in rows:
        mesh_doc_counts.update(row.meshes)

    terms_by_freq = sorted(mesh_doc_counts.items(), key=lambda item: (item[1], item[0]))
    start_rank = int(math.floor((args.mesh_quantile / 100.0) * (len(terms_by_freq) - 1)))
    selected_pairs = terms_by_freq[start_rank : start_rank + args.mesh_count]
    selected_meshes = {mesh for mesh, _ in selected_pairs}

    selected_rows: list[dict[str, Any]] = []
    for global_row_index, row in enumerate(rows):
        matching_meshes = sorted(set(row.meshes) & selected_meshes)
        if not matching_meshes:
            continue
        selected_rows.append(
            {
                "selected_index": len(selected_rows),
                "global_row_index": global_row_index,
                "config": row.config,
                "row_index": row.row_index,
                "pubid": row.pubid,
                "question": row.question,
                "doc_text": row.doc_text,
                "meshes": list(row.meshes),
                "matching_selected_meshes": matching_meshes,
            }
        )

    _write_jsonl(selected_manifest_path, selected_rows)
    _write_json(
        selected_meshes_path,
        {
            "mesh_quantile": args.mesh_quantile,
            "mesh_count": args.mesh_count,
            "start_rank": start_rank,
            "selected_meshes": [
                {"mesh": mesh, "doc_frequency": count} for mesh, count in selected_pairs
            ],
        },
    )
    summary = {
        "data_root": str(args.data_root),
        "mesh_quantile": args.mesh_quantile,
        "mesh_count": args.mesh_count,
        "start_rank": start_rank,
        "unique_mesh_terms": len(mesh_doc_counts),
        "selected_mesh_doc_frequency_min": min(count for _, count in selected_pairs),
        "selected_mesh_doc_frequency_max": max(count for _, count in selected_pairs),
        "selected_documents": len(selected_rows),
        "unique_questions": len({row["question"] for row in selected_rows}),
        "selected_manifest_path": str(selected_manifest_path),
        "selected_meshes_path": str(selected_meshes_path),
    }
    _write_json(summary_path, summary)
    return summary


def _batch_custom_id(row: dict[str, Any]) -> str:
    return f"pubmedqa:{row['pubid']}:{row['config']}:{row['row_index']}"


def build_batch_input(args: argparse.Namespace) -> dict[str, Any]:
    selected_manifest_path = args.output_dir / "selected_manifest.jsonl"
    batch_input_path = args.output_dir / "openai_batch_input.jsonl"
    rows = list(_iter_jsonl(selected_manifest_path))
    with batch_input_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            body = {
                "model": args.model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": USER_PROMPT_PREFIX + row["question"]},
                ],
                "response_format": {"type": "json_object"},
                "max_completion_tokens": args.max_completion_tokens,
            }
            handle.write(
                json.dumps(
                    {
                        "custom_id": _batch_custom_id(row),
                        "method": "POST",
                        "url": "/v1/chat/completions",
                        "body": body,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )
    summary = {
        "batch_input_path": str(batch_input_path),
        "request_count": len(rows),
        "model": args.model,
    }
    _write_json(args.output_dir / "batch_input_summary.json", summary)
    return summary


def _resolve_api_key(args: argparse.Namespace) -> str:
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if api_key:
        return api_key
    if args.api_key_file and args.api_key_file.is_file():
        return args.api_key_file.read_text(encoding="utf-8").strip()
    raise RuntimeError("No OpenAI API key found. Set OPENAI_API_KEY or pass --api-key-file.")


def _openai_headers(api_key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {api_key}"}


def _openai_json_headers(api_key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}


def verify_model(api_key: str, model: str, *, timeout_seconds: int) -> None:
    response = requests.get(
        f"https://api.openai.com/v1/models/{model}",
        headers=_openai_headers(api_key),
        timeout=timeout_seconds,
    )
    if response.status_code == 404:
        raise RuntimeError(f"OpenAI model {model!r} was not found for this API key.")
    if response.status_code >= 400:
        raise RuntimeError(
            f"Model verification failed: HTTP {response.status_code}: {response.text[:500]}"
        )


def upload_batch_file(api_key: str, batch_input_path: Path, *, timeout_seconds: int) -> dict[str, Any]:
    with batch_input_path.open("rb") as handle:
        response = requests.post(
            "https://api.openai.com/v1/files",
            headers=_openai_headers(api_key),
            data={"purpose": "batch"},
            files={"file": (batch_input_path.name, handle, "application/jsonl")},
            timeout=timeout_seconds,
        )
    if response.status_code >= 400:
        raise RuntimeError(f"Batch file upload failed: HTTP {response.status_code}: {response.text[:1000]}")
    return response.json()


def create_batch(api_key: str, input_file_id: str, args: argparse.Namespace) -> dict[str, Any]:
    payload = {
        "input_file_id": input_file_id,
        "endpoint": "/v1/chat/completions",
        "completion_window": args.completion_window,
        "metadata": {
            "dataset": "pubmedqa",
            "mesh_selection": f"p{args.mesh_quantile}_start_{args.mesh_count}",
            "model": args.model,
        },
    }
    response = requests.post(
        "https://api.openai.com/v1/batches",
        headers=_openai_json_headers(api_key),
        json=payload,
        timeout=args.timeout_seconds,
    )
    if response.status_code >= 400:
        raise RuntimeError(f"Batch creation failed: HTTP {response.status_code}: {response.text[:1000]}")
    return response.json()


def submit_batch(args: argparse.Namespace) -> dict[str, Any]:
    state_path = args.output_dir / "batch_state.json"
    if state_path.exists() and not args.force_batch_submit:
        return _read_json(state_path)
    api_key = _resolve_api_key(args)
    verify_model(api_key, args.model, timeout_seconds=args.timeout_seconds)
    batch_input_path = args.output_dir / "openai_batch_input.jsonl"
    uploaded_file = upload_batch_file(api_key, batch_input_path, timeout_seconds=args.timeout_seconds)
    batch = create_batch(api_key, uploaded_file["id"], args)
    state = {
        "model": args.model,
        "input_file": uploaded_file,
        "batch": batch,
        "batch_id": batch["id"],
        "submitted_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    _write_json(state_path, state)
    _write_json(args.output_dir / "batch_status.json", batch)
    return state


def poll_batch(args: argparse.Namespace) -> dict[str, Any]:
    api_key = _resolve_api_key(args)
    state = _read_json(args.output_dir / "batch_state.json")
    batch_id = state["batch_id"]
    deadline = time.monotonic() + args.max_poll_seconds
    while True:
        response = requests.get(
            f"https://api.openai.com/v1/batches/{batch_id}",
            headers=_openai_headers(api_key),
            timeout=args.timeout_seconds,
        )
        if response.status_code >= 400:
            raise RuntimeError(f"Batch status failed: HTTP {response.status_code}: {response.text[:1000]}")
        status = response.json()
        _write_json(args.output_dir / "batch_status.json", status)
        print(
            f"batch_id={batch_id} status={status.get('status')} "
            f"request_counts={status.get('request_counts')}",
            flush=True,
        )
        if status.get("status") in TERMINAL_BATCH_STATES:
            return status
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Batch {batch_id} did not finish within {args.max_poll_seconds}s")
        time.sleep(args.poll_interval_seconds)


def download_batch_output(args: argparse.Namespace) -> dict[str, Any]:
    api_key = _resolve_api_key(args)
    status = _read_json(args.output_dir / "batch_status.json")
    if status.get("status") != "completed":
        raise RuntimeError(f"Cannot download incomplete batch with status={status.get('status')!r}")
    output_file_id = status.get("output_file_id")
    if not output_file_id:
        raise RuntimeError("Completed batch has no output_file_id")
    output_path = args.output_dir / "openai_batch_output.jsonl"
    response = requests.get(
        f"https://api.openai.com/v1/files/{output_file_id}/content",
        headers=_openai_headers(api_key),
        timeout=args.download_timeout_seconds,
    )
    if response.status_code >= 400:
        raise RuntimeError(f"Batch output download failed: HTTP {response.status_code}: {response.text[:1000]}")
    output_path.write_bytes(response.content)

    error_file_id = status.get("error_file_id")
    error_path = None
    if error_file_id:
        error_response = requests.get(
            f"https://api.openai.com/v1/files/{error_file_id}/content",
            headers=_openai_headers(api_key),
            timeout=args.download_timeout_seconds,
        )
        if error_response.status_code < 400:
            error_path = args.output_dir / "openai_batch_errors.jsonl"
            error_path.write_bytes(error_response.content)
    summary = {
        "output_file_id": output_file_id,
        "output_path": str(output_path),
        "error_file_id": error_file_id,
        "error_path": str(error_path) if error_path else None,
    }
    _write_json(args.output_dir / "batch_download_summary.json", summary)
    return summary


def _strip_json_fences(text: str) -> str:
    stripped = text.strip()
    match = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", stripped, flags=re.DOTALL)
    return match.group(1).strip() if match else stripped


def _validate_rephrasings(content: str) -> list[str]:
    payload = json.loads(_strip_json_fences(content))
    values = payload.get("rephrasings")
    if not isinstance(values, list):
        raise ValueError("Missing rephrasings array")
    cleaned = [_clean_str(value) for value in values if _clean_str(value)]
    if len(cleaned) != 5:
        raise ValueError(f"Expected exactly 5 rephrasings, got {len(cleaned)}")
    return cleaned


def validate_batch_output(args: argparse.Namespace) -> dict[str, Any]:
    manifest_rows = list(_iter_jsonl(args.output_dir / "selected_manifest.jsonl"))
    rows_by_custom_id = {_batch_custom_id(row): row for row in manifest_rows}
    output_path = args.output_dir / "openai_batch_output.jsonl"
    rephrased_path = args.output_dir / "rephrased_queries.jsonl"
    invalid_path = args.output_dir / "invalid_batch_rows.jsonl"
    retry_input_path = args.output_dir / "openai_batch_retry_input.jsonl"

    normalized: list[dict[str, Any]] = []
    invalid: list[dict[str, Any]] = []
    seen: set[str] = set()
    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0

    for batch_row in _iter_jsonl(output_path):
        custom_id = batch_row.get("custom_id")
        seen.add(custom_id)
        manifest_row = rows_by_custom_id.get(custom_id)
        if manifest_row is None:
            invalid.append({"custom_id": custom_id, "error": "unknown custom_id", "row": batch_row})
            continue
        response = batch_row.get("response") or {}
        if response.get("status_code") != 200:
            invalid.append({"custom_id": custom_id, "error": "non_200_response", "row": batch_row})
            continue
        body = response.get("body") or {}
        try:
            content = body["choices"][0]["message"]["content"]
            rephrasings = _validate_rephrasings(content)
        except Exception as exc:  # noqa: BLE001 - preserve validation failure details.
            invalid.append({"custom_id": custom_id, "error": str(exc), "row": batch_row})
            continue
        usage = body.get("usage") or {}
        prompt_tokens += int(usage.get("prompt_tokens") or 0)
        completion_tokens += int(usage.get("completion_tokens") or 0)
        total_tokens += int(usage.get("total_tokens") or 0)
        normalized.append(
            {
                "custom_id": custom_id,
                "selected_index": manifest_row["selected_index"],
                "config": manifest_row["config"],
                "row_index": manifest_row["row_index"],
                "pubid": manifest_row["pubid"],
                "question": manifest_row["question"],
                "rephrasings": rephrasings,
                "matching_selected_meshes": manifest_row["matching_selected_meshes"],
            }
        )

    missing_ids = sorted(set(rows_by_custom_id) - seen)
    invalid.extend({"custom_id": custom_id, "error": "missing_output"} for custom_id in missing_ids)
    _write_jsonl(rephrased_path, sorted(normalized, key=lambda row: row["selected_index"]))
    _write_jsonl(invalid_path, invalid)

    retry_rows = [rows_by_custom_id[row["custom_id"]] for row in invalid if row["custom_id"] in rows_by_custom_id]
    if retry_rows:
        with retry_input_path.open("w", encoding="utf-8") as handle:
            for row in retry_rows:
                body = {
                    "model": args.model,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": USER_PROMPT_PREFIX + row["question"]},
                    ],
                    "response_format": {"type": "json_object"},
                    "max_completion_tokens": args.max_completion_tokens,
                }
                handle.write(
                    json.dumps(
                        {
                            "custom_id": _batch_custom_id(row),
                            "method": "POST",
                            "url": "/v1/chat/completions",
                            "body": body,
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                    + "\n"
                )

    summary = {
        "requested_rows": len(manifest_rows),
        "valid_rows": len(normalized),
        "invalid_or_missing_rows": len(invalid),
        "rephrased_queries": len(normalized) * 5,
        "rephrased_queries_path": str(rephrased_path),
        "invalid_rows_path": str(invalid_path),
        "retry_input_path": str(retry_input_path) if retry_rows else None,
        "api_prompt_tokens": prompt_tokens,
        "api_completion_tokens": completion_tokens,
        "api_total_tokens": total_tokens,
    }
    _write_json(args.output_dir / "batch_validation_summary.json", summary)
    return summary


def resolve_device(requested: str | None) -> str:
    if requested and requested != "auto":
        return requested
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def _checkpoint_paths(output_dir: Path) -> dict[str, Path]:
    emb_dir = output_dir / "embeddings"
    return {
        "state": emb_dir / "embedding_state.json",
        "doc_pubids": emb_dir / "doc_pubids.npy",
        "doc_embeddings": emb_dir / "doc_embeddings.npy",
        "original_pubids": emb_dir / "original_query_pubids.npy",
        "original_embeddings": emb_dir / "original_query_embeddings.npy",
        "rephrased_meta": emb_dir / "rephrased_query_meta.jsonl",
        "rephrased_embeddings": emb_dir / "rephrased_query_embeddings.npy",
        "embedding_manifest": emb_dir / "embedding_manifest.json",
    }


def _load_state(path: Path) -> dict[str, Any]:
    return _read_json(path) if path.exists() else {}


def _save_pubids(path: Path, pubids: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = np.load(path, allow_pickle=True)
        if len(existing) != len(pubids):
            raise RuntimeError(f"{path} length mismatch: {len(existing)} != {len(pubids)}")
        return
    np.save(path, np.asarray(pubids, dtype=str))


def _open_memmap(path: Path, *, rows: int, dim: int) -> np.memmap:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "r+" if path.exists() else "w+"
    return np.memmap(path, mode=mode, dtype=np.float32, shape=(rows, dim))


def embed_texts_checkpointed(
    *,
    texts: list[str],
    output_path: Path,
    state_path: Path,
    state_key: str,
    embedder: SentenceTransformerEmbedder,
    batch_size: int,
    checkpoint_rows: int,
) -> dict[str, Any]:
    dim = embedder.embedding_dim
    state = _load_state(state_path)
    start = int(state.get(state_key, 0))
    vectors = _open_memmap(output_path, rows=len(texts), dim=dim)
    if start >= len(texts):
        return {"path": str(output_path), "rows": len(texts), "resumed_from": start, "complete": True}

    for batch_start in tqdm(
        range(start, len(texts), batch_size),
        desc=f"Embedding {state_key}",
    ):
        batch_end = min(len(texts), batch_start + batch_size)
        vectors[batch_start:batch_end] = embedder.encode(
            texts[batch_start:batch_end],
            show_progress_bar=False,
        )
        vectors.flush()
        if batch_end == len(texts) or batch_end % checkpoint_rows == 0:
            state[state_key] = batch_end
            state[f"{state_key}_updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            _write_json(state_path, state)
    del vectors
    return {"path": str(output_path), "rows": len(texts), "resumed_from": start, "complete": True}


def run_embeddings(args: argparse.Namespace) -> dict[str, Any]:
    manifest_rows = list(_iter_jsonl(args.output_dir / "selected_manifest.jsonl"))
    rephrased_rows = list(_iter_jsonl(args.output_dir / "rephrased_queries.jsonl"))
    if len(rephrased_rows) != len(manifest_rows):
        raise RuntimeError(
            f"Expected {len(manifest_rows)} validated rephrasing rows, found {len(rephrased_rows)}"
        )
    paths = _checkpoint_paths(args.output_dir)
    doc_pubids = [row["pubid"] for row in manifest_rows]
    original_pubids = [row["pubid"] for row in rephrased_rows]
    rephrased_texts: list[str] = []
    rephrased_meta: list[dict[str, Any]] = []
    for row in rephrased_rows:
        for rephrase_index, text in enumerate(row["rephrasings"]):
            rephrased_texts.append(text)
            rephrased_meta.append(
                {
                    "source_pubid": row["pubid"],
                    "source_selected_index": row["selected_index"],
                    "rephrase_index": rephrase_index,
                    "rephrased_question": text,
                }
            )

    _save_pubids(paths["doc_pubids"], doc_pubids)
    _save_pubids(paths["original_pubids"], original_pubids)
    _write_jsonl(paths["rephrased_meta"], rephrased_meta)

    device = resolve_device(args.device)
    embedder = SentenceTransformerEmbedder(
        model_name=args.embedding_model,
        batch_size=args.model_batch_size,
        device=device,
    )
    _write_json(
        paths["embedding_manifest"],
        {
            "embedding_model": args.embedding_model,
            "device": device,
            "embedding_dim": embedder.embedding_dim,
            "doc_rows": len(manifest_rows),
            "original_query_rows": len(rephrased_rows),
            "rephrased_query_rows": len(rephrased_texts),
            "batch_size": args.embed_batch_size,
        },
    )
    summaries = {
        "docs": embed_texts_checkpointed(
            texts=[row["doc_text"] for row in manifest_rows],
            output_path=paths["doc_embeddings"],
            state_path=paths["state"],
            state_key="doc_rows_embedded",
            embedder=embedder,
            batch_size=args.embed_batch_size,
            checkpoint_rows=args.checkpoint_rows,
        ),
        "original_queries": embed_texts_checkpointed(
            texts=[row["question"] for row in rephrased_rows],
            output_path=paths["original_embeddings"],
            state_path=paths["state"],
            state_key="original_query_rows_embedded",
            embedder=embedder,
            batch_size=args.embed_batch_size,
            checkpoint_rows=args.checkpoint_rows,
        ),
        "rephrased_queries": embed_texts_checkpointed(
            texts=rephrased_texts,
            output_path=paths["rephrased_embeddings"],
            state_path=paths["state"],
            state_key="rephrased_query_rows_embedded",
            embedder=embedder,
            batch_size=args.embed_batch_size,
            checkpoint_rows=args.checkpoint_rows,
        ),
    }
    _write_json(args.output_dir / "embedding_summary.json", summaries)
    return summaries


def _nearest_indices(
    *,
    query_embeddings: np.memmap,
    doc_embeddings: np.memmap,
    query_batch_size: int,
    doc_chunk_size: int,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    use_torch = device != "cpu"
    if use_torch:
        import torch

        torch_device = torch.device(device)
    else:
        torch = None
        torch_device = None
    best_indices = np.zeros(query_embeddings.shape[0], dtype=np.int64)
    best_scores = np.full(query_embeddings.shape[0], -np.inf, dtype=np.float32)

    for query_start in tqdm(range(0, query_embeddings.shape[0], query_batch_size), desc="NN scan"):
        query_end = min(query_embeddings.shape[0], query_start + query_batch_size)
        q = np.asarray(query_embeddings[query_start:query_end], dtype=np.float32)
        batch_best_scores = np.full(query_end - query_start, -np.inf, dtype=np.float32)
        batch_best_indices = np.zeros(query_end - query_start, dtype=np.int64)
        if use_torch:
            q_tensor = torch.from_numpy(q).to(torch_device)
        for doc_start in range(0, doc_embeddings.shape[0], doc_chunk_size):
            doc_end = min(doc_embeddings.shape[0], doc_start + doc_chunk_size)
            d = np.asarray(doc_embeddings[doc_start:doc_end], dtype=np.float32)
            if use_torch:
                with torch.no_grad():
                    d_tensor = torch.from_numpy(d).to(torch_device)
                    scores_tensor = q_tensor @ d_tensor.T
                    local_best_tensor, local_argmax_tensor = torch.max(scores_tensor, dim=1)
                    local_best = local_best_tensor.cpu().numpy()
                    local_argmax = local_argmax_tensor.cpu().numpy()
                del d_tensor, scores_tensor
            else:
                scores = q @ d.T
                local_argmax = scores.argmax(axis=1)
                local_best = scores[np.arange(scores.shape[0]), local_argmax]
            improve = local_best > batch_best_scores
            batch_best_scores[improve] = local_best[improve]
            batch_best_indices[improve] = doc_start + local_argmax[improve]
        best_indices[query_start:query_end] = batch_best_indices
        best_scores[query_start:query_end] = batch_best_scores
        if use_torch:
            del q_tensor
    return best_indices, best_scores


def evaluate_group(
    *,
    name: str,
    query_embeddings_path: Path,
    query_pubids: list[str],
    doc_embeddings: np.memmap,
    doc_pubids: np.ndarray,
    dim: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    query_embeddings = np.memmap(
        query_embeddings_path,
        mode="r",
        dtype=np.float32,
        shape=(len(query_pubids), dim),
    )
    device = resolve_device(args.retrieval_device)
    nearest_indices, nearest_scores = _nearest_indices(
        query_embeddings=query_embeddings,
        doc_embeddings=doc_embeddings,
        query_batch_size=args.retrieval_query_batch_size,
        doc_chunk_size=args.doc_chunk_size,
        device=device,
    )
    results_path = args.output_dir / f"{name}_nearest_doc_results.jsonl"
    mismatch_count = 0
    top_examples: list[dict[str, Any]] = []
    with results_path.open("w", encoding="utf-8") as handle:
        for index, nearest_index in enumerate(nearest_indices):
            source_pubid = str(query_pubids[index])
            nearest_pubid = str(doc_pubids[int(nearest_index)])
            source_doc_index = index if name == "original_queries" else None
            if name == "rephrased_queries":
                # Rephrased queries are ordered in groups of five per selected document.
                source_doc_index = index // 5
            source_score = float(
                np.dot(
                    np.asarray(query_embeddings[index], dtype=np.float32),
                    np.asarray(doc_embeddings[int(source_doc_index)], dtype=np.float32),
                )
            )
            nearest_score = float(nearest_scores[index])
            is_mismatch = nearest_pubid != source_pubid
            mismatch_count += int(is_mismatch)
            result = {
                "query_index": index,
                "source_pubid": source_pubid,
                "nearest_pubid": nearest_pubid,
                "nearest_is_different_doc": is_mismatch,
                "source_similarity": source_score,
                "nearest_similarity": nearest_score,
                "margin": nearest_score - source_score,
            }
            handle.write(json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n")
            if is_mismatch:
                top_examples.append(result)
    top_examples = sorted(top_examples, key=lambda row: row["margin"], reverse=True)[: args.example_count]
    return {
        "queries_evaluated": len(query_pubids),
        "queries_nearest_different_doc": mismatch_count,
        "queries_nearest_different_doc_fraction": mismatch_count / len(query_pubids) if query_pubids else 0.0,
        "results_path": str(results_path),
        "retrieval_device": device,
        "top_mismatch_examples": top_examples,
    }


def run_retrieval(args: argparse.Namespace) -> dict[str, Any]:
    paths = _checkpoint_paths(args.output_dir)
    emb_manifest = _read_json(paths["embedding_manifest"])
    dim = int(emb_manifest["embedding_dim"])
    doc_pubids = np.load(paths["doc_pubids"], allow_pickle=True)
    original_pubids = [str(value) for value in np.load(paths["original_pubids"], allow_pickle=True)]
    rephrased_pubids = [
        row["source_pubid"] for row in _iter_jsonl(paths["rephrased_meta"])
    ]
    doc_embeddings = np.memmap(
        paths["doc_embeddings"],
        mode="r",
        dtype=np.float32,
        shape=(len(doc_pubids), dim),
    )
    report = {
        "original_queries": evaluate_group(
            name="original_queries",
            query_embeddings_path=paths["original_embeddings"],
            query_pubids=original_pubids,
            doc_embeddings=doc_embeddings,
            doc_pubids=doc_pubids,
            dim=dim,
            args=args,
        ),
        "rephrased_queries": evaluate_group(
            name="rephrased_queries",
            query_embeddings_path=paths["rephrased_embeddings"],
            query_pubids=rephrased_pubids,
            doc_embeddings=doc_embeddings,
            doc_pubids=doc_pubids,
            dim=dim,
            args=args,
        ),
    }
    validation_path = args.output_dir / "batch_validation_summary.json"
    if validation_path.exists():
        report["openai_batch"] = _read_json(validation_path)
    _write_json(args.output_dir / "retrieval_report.json", report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=(
            "select",
            "batch-input",
            "batch-submit",
            "batch-poll",
            "batch-download",
            "batch-validate",
            "embed",
            "retrieve",
            "all",
        ),
        default="all",
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--mesh-quantile", type=float, default=80.0)
    parser.add_argument("--mesh-count", type=int, default=500)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-completion-tokens", type=int, default=512)
    parser.add_argument("--completion-window", default="24h")
    parser.add_argument("--api-key-file", type=Path, default=None)
    parser.add_argument("--timeout-seconds", type=int, default=90)
    parser.add_argument("--download-timeout-seconds", type=int, default=600)
    parser.add_argument("--poll-interval-seconds", type=int, default=120)
    parser.add_argument("--max-poll-seconds", type=int, default=90_000)
    parser.add_argument("--force-batch-submit", action="store_true")
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--retrieval-device", default="auto")
    parser.add_argument("--model-batch-size", type=int, default=256)
    parser.add_argument("--embed-batch-size", type=int, default=256)
    parser.add_argument("--checkpoint-rows", type=int, default=1024)
    parser.add_argument("--doc-chunk-size", type=int, default=16384)
    parser.add_argument("--retrieval-query-batch-size", type=int, default=256)
    parser.add_argument("--example-count", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries: dict[str, Any] = {}
    if args.stage in {"select", "all"}:
        summaries["selection"] = build_selection(args)
    if args.stage in {"batch-input", "all"}:
        summaries["batch_input"] = build_batch_input(args)
    if args.stage in {"batch-submit", "all"}:
        summaries["batch_submit"] = submit_batch(args)
    if args.stage in {"batch-poll", "all"}:
        summaries["batch_poll"] = poll_batch(args)
    if args.stage in {"batch-download", "all"}:
        summaries["batch_download"] = download_batch_output(args)
    if args.stage in {"batch-validate", "all"}:
        summaries["batch_validation"] = validate_batch_output(args)
    if args.stage in {"embed", "all"}:
        summaries["embeddings"] = run_embeddings(args)
    if args.stage in {"retrieve", "all"}:
        summaries["retrieval"] = run_retrieval(args)
    _write_json(args.output_dir / "pipeline_summary.json", summaries)
    print(json.dumps(summaries, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()

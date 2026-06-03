#!/usr/bin/env python3
"""Download and validate the all-MeSH PubMedQA Batch API rephrasing outputs."""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Iterable
from urllib.error import HTTPError
from urllib.request import Request, urlopen


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "pubmedqa_rag" / "all_mesh_rephrasing_nano"
DEFAULT_API_KEY_FILE = REPO_ROOT / "data" / "pubmedqa_rag" / "rephrasing_smoke" / "openai_api_key"
TERMINAL_BATCH_STATES = {"completed", "failed", "expired", "cancelled"}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _clean_str(value: object) -> str:
    return str(value or "").strip()


def _resolve_api_key(args: argparse.Namespace) -> str:
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if api_key:
        return api_key
    if args.api_key_file and args.api_key_file.is_file():
        return args.api_key_file.read_text(encoding="utf-8").strip()
    raise RuntimeError("No OpenAI API key found. Set OPENAI_API_KEY or pass --api-key-file.")


def _openai_request(
    *,
    api_key: str,
    url: str,
    timeout_seconds: int,
) -> bytes:
    request = Request(url, headers={"Authorization": f"Bearer {api_key}"})
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            return response.read()
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"OpenAI request failed: HTTP {exc.code}: {body[:1000]}") from exc


def _openai_json(*, api_key: str, url: str, timeout_seconds: int) -> dict[str, Any]:
    return json.loads(_openai_request(api_key=api_key, url=url, timeout_seconds=timeout_seconds).decode("utf-8"))


def poll_batches(args: argparse.Namespace) -> dict[str, Any]:
    state_path = args.output_dir / "batch_state.json"
    state = _read_json(state_path)
    api_key = _resolve_api_key(args)
    results: list[dict[str, Any]] = []
    totals = {"total": 0, "completed": 0, "failed": 0}

    for batch_record in state.get("batches", []):
        batch_id = str(batch_record["batch_id"])
        status = _openai_json(
            api_key=api_key,
            url=f"https://api.openai.com/v1/batches/{batch_id}",
            timeout_seconds=args.timeout_seconds,
        )
        counts = status.get("request_counts") or {}
        for key in totals:
            totals[key] += int(counts.get(key) or 0)
        batch_record["batch"] = status
        results.append(
            {
                "chunk_index": int(batch_record["chunk_index"]),
                "batch_id": batch_id,
                "status": status.get("status"),
                "request_counts": counts,
                "output_file_id": status.get("output_file_id"),
                "error_file_id": status.get("error_file_id"),
                "errors": status.get("errors"),
            }
        )

    if all(result["status"] == "completed" for result in results):
        state["status"] = "completed"
    elif any(result["status"] in TERMINAL_BATCH_STATES - {"completed"} for result in results):
        state["status"] = "failed"
    else:
        state["status"] = "in_progress"
    state["last_polled_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    _write_json(state_path, state)

    summary = {
        "overall_status": state["status"],
        "aggregate_request_counts": totals,
        "batches": results,
    }
    _write_json(args.output_dir / "batch_poll_summary.json", summary)
    return summary


def download_outputs(args: argparse.Namespace) -> dict[str, Any]:
    poll_summary = poll_batches(args)
    incomplete = [row for row in poll_summary["batches"] if row["status"] != "completed"]
    if incomplete:
        raise RuntimeError(f"Cannot download incomplete batches: {incomplete}")

    state = _read_json(args.output_dir / "batch_state.json")
    api_key = _resolve_api_key(args)
    output_dir = args.output_dir / "batch_outputs"
    error_dir = args.output_dir / "batch_errors"
    downloaded: list[dict[str, Any]] = []

    for batch_record in state["batches"]:
        chunk_index = int(batch_record["chunk_index"])
        batch = batch_record["batch"]
        output_file_id = batch.get("output_file_id")
        if not output_file_id:
            raise RuntimeError(f"Completed batch {batch_record['batch_id']} has no output_file_id")

        output_path = output_dir / f"openai_batch_output_{chunk_index:03d}.jsonl"
        if not output_path.exists() or args.force_download:
            content = _openai_request(
                api_key=api_key,
                url=f"https://api.openai.com/v1/files/{output_file_id}/content",
                timeout_seconds=args.download_timeout_seconds,
            )
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(content)

        error_file_id = batch.get("error_file_id")
        error_path = None
        if error_file_id:
            error_path = error_dir / f"openai_batch_errors_{chunk_index:03d}.jsonl"
            if not error_path.exists() or args.force_download:
                content = _openai_request(
                    api_key=api_key,
                    url=f"https://api.openai.com/v1/files/{error_file_id}/content",
                    timeout_seconds=args.download_timeout_seconds,
                )
                error_path.parent.mkdir(parents=True, exist_ok=True)
                error_path.write_bytes(content)

        downloaded.append(
            {
                "chunk_index": chunk_index,
                "batch_id": batch_record["batch_id"],
                "request_count": int(batch_record["request_count"]),
                "output_file_id": output_file_id,
                "output_path": str(output_path),
                "error_file_id": error_file_id,
                "error_path": str(error_path) if error_path else None,
            }
        )

    summary = {"downloaded": downloaded}
    _write_json(args.output_dir / "batch_download_summary.json", summary)
    return summary


def _batch_custom_id(row: dict[str, Any]) -> str:
    return f"pubmedqa_allmesh:{row['pubid']}:{row['config']}:{row['row_index']}"


def _strip_json_fences(text: str) -> str:
    stripped = text.strip()
    match = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", stripped, flags=re.DOTALL)
    return match.group(1).strip() if match else stripped


def _validate_rephrasings(content: str) -> list[str]:
    payload = json.loads(_strip_json_fences(content))
    values = payload.get("rephrasings")
    if values is None:
        list_values = [value for value in payload.values() if isinstance(value, list)]
        if len(list_values) == 1:
            values = list_values[0]
    if not isinstance(values, list):
        raise ValueError("Missing rephrasings array")
    cleaned = [_clean_str(value) for value in values if _clean_str(value)]
    if len(cleaned) != 5:
        raise ValueError(f"Expected exactly 5 rephrasings, got {len(cleaned)}")
    return cleaned


def validate_outputs(args: argparse.Namespace) -> dict[str, Any]:
    manifest_rows = list(_iter_jsonl(args.output_dir / "selected_manifest.jsonl"))
    rows_by_custom_id = {_batch_custom_id(row): row for row in manifest_rows}
    output_paths = sorted((args.output_dir / "batch_outputs").glob("openai_batch_output_*.jsonl"))
    if not output_paths:
        raise RuntimeError(f"No downloaded batch outputs found under {args.output_dir / 'batch_outputs'}")

    normalized: list[dict[str, Any]] = []
    invalid: list[dict[str, Any]] = []
    seen: set[str] = set()
    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0

    for output_path in output_paths:
        for batch_row in _iter_jsonl(output_path):
            custom_id = batch_row.get("custom_id")
            seen.add(str(custom_id))
            manifest_row = rows_by_custom_id.get(str(custom_id))
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
            except Exception as exc:  # noqa: BLE001 - keep exact validation failure.
                invalid.append({"custom_id": custom_id, "error": str(exc), "row": batch_row})
                continue

            usage = body.get("usage") or {}
            prompt_tokens += int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
            completion_tokens += int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
            total_tokens += int(usage.get("total_tokens") or 0)
            normalized.append(
                {
                    "custom_id": custom_id,
                    "selected_index": int(manifest_row["selected_index"]),
                    "config": str(manifest_row["config"]),
                    "row_index": int(manifest_row["row_index"]),
                    "pubid": str(manifest_row["pubid"]),
                    "question": str(manifest_row["question"]),
                    "meshes": list(manifest_row.get("meshes") or []),
                    "rephrasings": rephrasings,
                }
            )

    missing_ids = sorted(set(rows_by_custom_id) - seen)
    invalid.extend({"custom_id": custom_id, "error": "missing_output"} for custom_id in missing_ids)
    normalized.sort(key=lambda row: row["selected_index"])

    rephrased_path = args.output_dir / "rephrased_queries.jsonl"
    invalid_path = args.output_dir / "invalid_batch_rows.jsonl"
    _write_jsonl(rephrased_path, normalized)
    _write_jsonl(invalid_path, invalid)

    summary = {
        "requested_rows": len(manifest_rows),
        "output_files": [str(path) for path in output_paths],
        "valid_rows": len(normalized),
        "invalid_or_missing_rows": len(invalid),
        "rephrased_queries": len(normalized) * 5,
        "rephrased_queries_path": str(rephrased_path),
        "invalid_rows_path": str(invalid_path),
        "api_prompt_tokens": prompt_tokens,
        "api_completion_tokens": completion_tokens,
        "api_total_tokens": total_tokens,
    }
    _write_json(args.output_dir / "batch_validation_summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("poll", "download", "validate", "all"), default="all")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--api-key-file", type=Path, default=DEFAULT_API_KEY_FILE)
    parser.add_argument("--timeout-seconds", type=int, default=90)
    parser.add_argument("--download-timeout-seconds", type=int, default=600)
    parser.add_argument("--force-download", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summaries: dict[str, Any] = {}
    if args.stage in {"poll", "download", "all"}:
        summaries["poll"] = poll_batches(args)
    if args.stage in {"download", "all"}:
        summaries["download"] = download_outputs(args)
    if args.stage in {"validate", "all"}:
        summaries["validate"] = validate_outputs(args)
    if summaries:
        _write_json(args.output_dir / "all_mesh_batch_rephrasing_summary.json", summaries)
        print(json.dumps(summaries, indent=2, ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

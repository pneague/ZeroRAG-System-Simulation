#!/usr/bin/env python3
"""Measure the longest Phi-3.5 evaluation prompt after chat templating."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.path_defaults import default_scratch_root  # noqa: E402


DEFAULT_SCRATCH = default_scratch_root()
DEFAULT_DATASET_DIR = DEFAULT_SCRATCH / "data/pubmedqa_phi35_top5_last200k_q30"
SHORT_COT_DIRECTIVE = (
    "Provide a brief justification (max 2 sentences) and then write exactly one line in the form "
    "'Final answer: <yes|no|maybe>'.\n\nAnswer:\n"
)


def _canonical_method(method: str) -> str:
    return "zerorag" if method in {"lowcoord", "low_coord"} else method


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _wrap_prompt(prompt: str, tokenizer: Any) -> str:
    prompt = prompt.rstrip()
    answer_marker = "Answer:"
    if prompt.endswith(answer_marker):
        prompt = prompt[: -len(answer_marker)].rstrip() + "\n\n" + SHORT_COT_DIRECTIVE
    else:
        prompt = prompt + "\n\n" + SHORT_COT_DIRECTIVE
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    return prompt


def _measure_batch(
    *,
    tokenizer: Any,
    prompts: list[str],
    row_summaries: list[dict[str, Any]],
    max_tokens: int,
    max_row: dict[str, Any] | None,
) -> tuple[int, dict[str, Any] | None]:
    encoded = tokenizer(prompts, add_special_tokens=True, truncation=False)
    for idx, (input_ids, row_summary) in enumerate(zip(encoded["input_ids"], row_summaries, strict=True)):
        token_count = len(input_ids)
        if token_count > max_tokens:
            max_tokens = token_count
            max_row = {**row_summary, "token_count": token_count, "prompt_chars": len(prompts[idx])}
    return max_tokens, max_row


def _measure_file(path: Path, method: str, tokenizer: Any, batch_size: int, progress_every: int) -> dict[str, Any]:
    started = time.perf_counter()
    max_tokens = 0
    max_row: dict[str, Any] | None = None
    rows = 0
    prompts: list[str] = []
    row_summaries: list[dict[str, Any]] = []
    for row in _iter_jsonl(path):
        rows += 1
        prompt = _wrap_prompt(str(row["prompt"]), tokenizer)
        prompts.append(prompt)
        row_summaries.append(
            {
                "method": method,
                "event_index": row.get("event_index"),
                "line_number": row.get("line_number"),
                "qid": row.get("qid"),
                "checkpoint": row.get("checkpoint"),
            }
        )
        if len(prompts) >= batch_size:
            max_tokens, max_row = _measure_batch(
                tokenizer=tokenizer,
                prompts=prompts,
                row_summaries=row_summaries,
                max_tokens=max_tokens,
                max_row=max_row,
            )
            prompts = []
            row_summaries = []
        if progress_every > 0 and rows % progress_every == 0:
            elapsed = time.perf_counter() - started
            print(
                json.dumps(
                    {
                        "method": method,
                        "rows": rows,
                        "max_tokens": max_tokens,
                        "elapsed_seconds": elapsed,
                        "rows_per_second": rows / elapsed if elapsed else 0.0,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    if prompts:
        max_tokens, max_row = _measure_batch(
            tokenizer=tokenizer,
            prompts=prompts,
            row_summaries=row_summaries,
            max_tokens=max_tokens,
            max_row=max_row,
        )
    elapsed = time.perf_counter() - started
    return {
        "method": method,
        "path": str(path),
        "rows": rows,
        "max_tokens": max_tokens,
        "max_row": max_row,
        "elapsed_seconds": elapsed,
        "rows_per_second": rows / elapsed if elapsed else 0.0,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--model-name", default="microsoft/Phi-3.5-mini-instruct")
    parser.add_argument("--methods", nargs="+", default=["zerorag", "hnsw"])
    parser.add_argument("--output-file", type=Path, default=None)
    parser.add_argument("--report-file", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--progress-every", type=int, default=10_000)
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def main() -> None:
    from transformers import AutoTokenizer

    args = parse_args()
    output_file = args.output_file or args.dataset_dir / "max_input_tokens.txt"
    report_file = args.report_file or args.dataset_dir / "max_input_tokens_report.json"
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=args.trust_remote_code,
        use_fast=True,
    )

    requested_methods: list[str] = []
    for method in args.methods:
        canonical = _canonical_method(method)
        if canonical not in requested_methods:
            requested_methods.append(canonical)

    method_reports = []
    for method in requested_methods:
        path = args.dataset_dir / f"{method}_examples.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"Missing examples file: {path}")
        method_reports.append(
            _measure_file(
                path=path,
                method=method,
                tokenizer=tokenizer,
                batch_size=args.batch_size,
                progress_every=args.progress_every,
            )
        )

    global_max = max(report["max_tokens"] for report in method_reports)
    max_input_tokens = global_max + 1
    report = {
        "dataset_dir": str(args.dataset_dir),
        "model_name": args.model_name,
        "methods": method_reports,
        "global_max_tokens": global_max,
        "max_input_tokens": max_input_tokens,
        "short_cot_directive": SHORT_COT_DIRECTIVE,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(f"{max_input_tokens}\n", encoding="utf-8")
    report_file.parent.mkdir(parents=True, exist_ok=True)
    report_file.write_text(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Run Phi-3.5 multi-choice evaluation over PubMedQA top-k contexts."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.path_defaults import default_scratch_root  # noqa: E402


DEFAULT_SCRATCH = default_scratch_root()
DEFAULT_DATASET_DIR = DEFAULT_SCRATCH / "data/pubmedqa_phi35_top5_checkpoints_q30_4x5k"
DEFAULT_OUTPUT_DIR = DEFAULT_SCRATCH / "results/pubmedqa_phi35_top5_checkpoints_q30_4x5k"
LABELS = ("yes", "no", "maybe")
SHORT_COT_DIRECTIVE = (
    "Provide a brief justification (max 2 sentences) and then write exactly one line in the form "
    "'Final answer: <yes|no|maybe>'.\n\nAnswer:\n"
)
STOP_STRINGS = ("Final answer: yes", "Final answer: no", "Final answer: maybe")


def _canonical_method(method: str) -> str:
    return "zerorag" if method in {"lowcoord", "low_coord"} else method


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _append_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")


def _prediction_path(output_dir: Path, method: str) -> Path:
    return output_dir / f"{method}_phi35_predictions.jsonl"


def _load_completed(path: Path) -> tuple[set[int], Counter[str], Counter[str], int, int]:
    completed: set[int] = set()
    pred_counts: Counter[str] = Counter()
    correct_by_label: Counter[str] = Counter()
    parsed = 0
    total = 0
    if not path.exists():
        return completed, pred_counts, correct_by_label, parsed, total
    for row in _iter_jsonl(path):
        total += 1
        completed.add(int(row["event_index"]))
        parsed_answer = row.get("predicted_answer")
        if parsed_answer in LABELS:
            parsed += 1
            pred_counts[str(parsed_answer)] += 1
        if row.get("is_correct") and row.get("gold_answer") in LABELS:
            correct_by_label[str(row["gold_answer"])] += 1
    return completed, pred_counts, correct_by_label, parsed, total


def _parse_answer(text: str) -> tuple[str | None, str]:
    lowered = text.lower()
    final = re.findall(r"final\s+answer\s*[:\-]?\s*(yes|no|maybe)\b", lowered)
    if final:
        return final[-1], "final_answer"
    explicit = re.findall(r"answer\s*[:\-]?\s*(yes|no|maybe)\b", lowered)
    if explicit:
        return explicit[-1], "explicit"
    choices = re.findall(r"\b(yes|no|maybe)\b", lowered)
    if choices:
        return choices[-1], "last_label"
    return None, "invalid"


def _batched(items: list[dict[str, Any]], batch_size: int) -> Iterable[list[dict[str, Any]]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def _load_model(args: argparse.Namespace):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    dtype = torch.float16 if args.torch_dtype == "float16" else torch.bfloat16 if args.torch_dtype == "bfloat16" else "auto"
    device = "cuda" if torch.cuda.is_available() and args.device == "auto" else args.device
    if device == "auto":
        device = "cpu"

    model_kwargs: dict[str, Any] = {
        "trust_remote_code": args.trust_remote_code,
        "torch_dtype": dtype,
        "attn_implementation": args.attn_implementation,
        "low_cpu_mem_usage": True,
    }
    if device == "cuda":
        try:
            model = AutoModelForCausalLM.from_pretrained(
                args.model_name,
                device_map="auto",
                **model_kwargs,
            )
        except Exception:
            model = AutoModelForCausalLM.from_pretrained(args.model_name, **model_kwargs).to("cuda")
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model_name, **model_kwargs).to(device)
    model.eval()
    return tokenizer, model


def _wrap_prompt(prompt: str, tokenizer) -> str:
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


def _generate_batch(
    *,
    tokenizer,
    model,
    examples: list[dict[str, Any]],
    max_new_tokens: int,
    max_input_tokens: int,
) -> list[str]:
    import torch

    prompts = []
    for example in examples:
        prompts.append(_wrap_prompt(str(example["prompt"]), tokenizer))
    encoded = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=max_input_tokens > 0,
        max_length=max_input_tokens if max_input_tokens > 0 else None,
    )
    device = next(model.parameters()).device
    encoded = {key: value.to(device) for key, value in encoded.items()}
    input_width = int(encoded["input_ids"].shape[1])
    with torch.inference_mode():
        generated = model.generate(
            **encoded,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            stop_strings=list(STOP_STRINGS),
            tokenizer=tokenizer,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    suffix = generated[:, input_width:].detach().cpu()
    decoded = tokenizer.batch_decode(suffix, skip_special_tokens=True)
    del generated, suffix, encoded
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return decoded


def _generate_batch_with_oom_retry(
    *,
    tokenizer,
    model,
    examples: list[dict[str, Any]],
    max_new_tokens: int,
    max_input_tokens: int,
    method: str,
    progress_log: Path,
) -> tuple[list[str], int]:
    import gc
    import torch

    try:
        return (
            _generate_batch(
                tokenizer=tokenizer,
                model=model,
                examples=examples,
                max_new_tokens=max_new_tokens,
                max_input_tokens=max_input_tokens,
            ),
            0,
        )
    except torch.cuda.OutOfMemoryError:
        gc.collect()
        torch.cuda.empty_cache()
        if len(examples) <= 1:
            raise
        midpoint = len(examples) // 2
        _append_jsonl(
            progress_log,
            [
                {
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "event": "oom_retry",
                    "method": method,
                    "failed_batch_size": len(examples),
                    "retry_batch_sizes": [midpoint, len(examples) - midpoint],
                    "last_event_index": int(examples[-1]["event_index"]),
                }
            ],
        )
        left, left_retries = _generate_batch_with_oom_retry(
            tokenizer=tokenizer,
            model=model,
            examples=examples[:midpoint],
            max_new_tokens=max_new_tokens,
            max_input_tokens=max_input_tokens,
            method=method,
            progress_log=progress_log,
        )
        right, right_retries = _generate_batch_with_oom_retry(
            tokenizer=tokenizer,
            model=model,
            examples=examples[midpoint:],
            max_new_tokens=max_new_tokens,
            max_input_tokens=max_input_tokens,
            method=method,
            progress_log=progress_log,
        )
        return left + right, left_retries + right_retries + 1


def _example_stream(path: Path, completed: set[int], limit: int) -> Iterable[dict[str, Any]]:
    yielded = 0
    for row in _iter_jsonl(path):
        event_index = int(row["event_index"])
        if event_index in completed:
            continue
        yield row
        yielded += 1
        if limit > 0 and yielded >= limit:
            break


def _run_method(
    *,
    method: str,
    examples_path: Path,
    output_dir: Path,
    progress_log: Path,
    tokenizer,
    model,
    args: argparse.Namespace,
) -> dict[str, Any]:
    prediction_path = _prediction_path(output_dir, method)
    completed, _pred_counts, _correct_by_label, parsed_existing, total_existing = _load_completed(prediction_path)
    started = time.perf_counter()
    completed_this_run = 0
    parsed_this_run = 0
    invalid_this_run = 0
    evaluable_this_run = 0
    correct_this_run = 0

    pending_iter = _example_stream(examples_path, completed, args.limit_per_method)
    while True:
        batch = []
        try:
            for _ in range(args.batch_size):
                batch.append(next(pending_iter))
        except StopIteration:
            pass
        if not batch:
            break

        generated_texts, oom_retries = _generate_batch_with_oom_retry(
            tokenizer=tokenizer,
            model=model,
            examples=batch,
            max_new_tokens=args.max_new_tokens,
            max_input_tokens=args.max_input_tokens,
            method=method,
            progress_log=progress_log,
        )
        prediction_rows: list[dict[str, Any]] = []
        for example, generated_text in zip(batch, generated_texts, strict=True):
            predicted_answer, parse_status = _parse_answer(generated_text)
            gold_answer = str(example.get("gold_answer") or "").strip().lower()
            is_evaluable = gold_answer in LABELS
            is_correct = is_evaluable and predicted_answer == gold_answer
            prediction_rows.append(
                {
                    "method": method,
                    "checkpoint": example.get("checkpoint"),
                    "event_index": int(example["event_index"]),
                    "line_number": int(example["line_number"]),
                    "qid": int(example["qid"]),
                    "question": example.get("question"),
                    "gold_answer": gold_answer,
                    "is_evaluable": is_evaluable,
                    "predicted_answer": predicted_answer,
                    "parse_status": parse_status,
                    "is_correct": bool(is_correct),
                    "target_doc_id": example.get("target_doc_id"),
                    "target_in_top10": bool(example.get("target_in_top10")),
                    "target_in_context": bool(example.get("target_in_context")),
                    "top10_doc_ids": example.get("top10_doc_ids"),
                    "top10_scores": example.get("top10_scores"),
                    "query_variant": example.get("query_variant"),
                    "rephrase_index": example.get("rephrase_index"),
                    "skipped_reason": example.get("skipped_reason"),
                    "raw_generation": generated_text[:1000],
                }
            )
            completed_this_run += 1
            parsed_this_run += int(predicted_answer in LABELS)
            invalid_this_run += int(predicted_answer not in LABELS)
            evaluable_this_run += int(is_evaluable)
            correct_this_run += int(is_correct)
            completed.add(int(example["event_index"]))

        _append_jsonl(prediction_path, prediction_rows)
        elapsed = time.perf_counter() - started
        total_completed = total_existing + completed_this_run
        total_parsed = parsed_existing + parsed_this_run
        progress = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "method": method,
            "last_event_index": int(batch[-1]["event_index"]),
            "batch_size": len(batch),
            "completed_this_run": completed_this_run,
            "completed_total_in_output": total_completed,
            "elapsed_seconds_this_run": elapsed,
            "examples_per_second_this_run": completed_this_run / elapsed if elapsed else 0.0,
            "parsed_this_run": parsed_this_run,
            "parsed_total_in_output": total_parsed,
            "invalid_this_run": invalid_this_run,
            "evaluable_this_run": evaluable_this_run,
            "running_accuracy_this_run": correct_this_run / evaluable_this_run if evaluable_this_run else 0.0,
            "prediction_path": str(prediction_path),
            "last_checkpoint": batch[-1].get("checkpoint"),
            "oom_retries": oom_retries,
            "max_input_tokens": args.max_input_tokens,
        }
        _append_jsonl(progress_log, [progress])

    return {
        "method": method,
        "prediction_path": str(prediction_path),
        "completed_this_run": completed_this_run,
        "completed_total_in_output": len(completed),
    }


def _precision_recall_f1(tp: int, fp: int, fn: int) -> dict[str, float]:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def _summarize_prediction_rows(rows: list[dict[str, Any]], prediction_path: str | None = None) -> dict[str, Any]:
    total = len(rows)
    evaluable_rows = [row for row in rows if row.get("gold_answer") in LABELS]
    parsed_rows = [row for row in rows if row.get("predicted_answer") in LABELS]
    evaluable_parsed_rows = [row for row in evaluable_rows if row.get("predicted_answer") in LABELS]
    correct = sum(1 for row in evaluable_rows if row.get("is_correct"))
    gold_counts = Counter(str(row.get("gold_answer")) for row in evaluable_rows)
    pred_counts = Counter(str(row.get("predicted_answer")) for row in parsed_rows)
    confusion: dict[str, dict[str, int]] = {gold: {pred: 0 for pred in LABELS} for gold in LABELS}
    invalid_by_gold: Counter[str] = Counter()
    for row in evaluable_rows:
        gold = str(row.get("gold_answer"))
        pred = row.get("predicted_answer")
        if gold not in LABELS:
            continue
        if pred in LABELS:
            confusion[gold][str(pred)] += 1
        else:
            invalid_by_gold[gold] += 1

    per_class: dict[str, dict[str, float]] = {}
    for label in LABELS:
        tp = confusion[label][label]
        fp = sum(confusion[other][label] for other in LABELS if other != label)
        fn = sum(confusion[label][other] for other in LABELS if other != label) + invalid_by_gold[label]
        per_class[label] = _precision_recall_f1(tp, fp, fn)
    macro_f1 = sum(per_class[label]["f1"] for label in LABELS) / len(LABELS)
    payload: dict[str, Any] = {
        "total": total,
        "evaluable": len(evaluable_rows),
        "unevaluable": total - len(evaluable_rows),
        "parsed": len(parsed_rows),
        "invalid": total - len(parsed_rows),
        "parse_rate": len(parsed_rows) / total if total else 0.0,
        "evaluable_parsed": len(evaluable_parsed_rows),
        "evaluable_parse_rate": len(evaluable_parsed_rows) / len(evaluable_rows) if evaluable_rows else 0.0,
        "accuracy": correct / len(evaluable_rows) if evaluable_rows else 0.0,
        "correct": correct,
        "gold_counts": dict(gold_counts),
        "prediction_counts": dict(pred_counts),
        "confusion_matrix": confusion,
        "invalid_by_gold": dict(invalid_by_gold),
        "per_class": per_class,
        "macro_f1": macro_f1,
    }
    if prediction_path is not None:
        payload["prediction_path"] = prediction_path
    return payload


def compute_metrics(output_dir: Path, methods: list[str]) -> dict[str, Any]:
    by_method: dict[str, dict[str, Any]] = {}
    by_checkpoint: dict[str, dict[str, dict[str, Any]]] = {}
    predictions_by_method: dict[str, dict[int, dict[str, Any]]] = {}
    for method in methods:
        path = _prediction_path(output_dir, method)
        rows = list(_iter_jsonl(path)) if path.exists() else []
        predictions_by_method[method] = {int(row["event_index"]): row for row in rows}
        by_method[method] = _summarize_prediction_rows(rows, str(path))
        checkpoint_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            checkpoint_rows[str(row.get("checkpoint") or "unknown")].append(row)
        by_checkpoint[method] = {
            checkpoint: _summarize_prediction_rows(grouped_rows)
            for checkpoint, grouped_rows in sorted(checkpoint_rows.items())
        }

    paired: dict[str, Any] = {}
    if len(methods) >= 2:
        left, right = methods[0], methods[1]
        shared = sorted(set(predictions_by_method[left]) & set(predictions_by_method[right]))
        both_correct = left_only = right_only = both_wrong = 0
        shared_evaluable = 0
        for event_index in shared:
            left_row = predictions_by_method[left][event_index]
            right_row = predictions_by_method[right][event_index]
            if left_row.get("gold_answer") not in LABELS or right_row.get("gold_answer") not in LABELS:
                continue
            shared_evaluable += 1
            left_correct = bool(left_row.get("is_correct"))
            right_correct = bool(right_row.get("is_correct"))
            both_correct += int(left_correct and right_correct)
            left_only += int(left_correct and not right_correct)
            right_only += int(right_correct and not left_correct)
            both_wrong += int(not left_correct and not right_correct)
        paired = {
            "methods": [left, right],
            "shared_events": len(shared),
            "shared_evaluable_events": shared_evaluable,
            "both_correct": both_correct,
            f"{left}_only_correct": left_only,
            f"{right}_only_correct": right_only,
            "both_wrong": both_wrong,
        }

    metrics = {
        "methods": by_method,
        "by_checkpoint": by_checkpoint,
        "paired": paired,
        "labels": list(LABELS),
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    _write_json(output_dir / "phi35_eval_metrics.json", metrics)
    with (output_dir / "phi35_eval_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("method", "total", "evaluable", "accuracy", "macro_f1", "parse_rate", "invalid"),
        )
        writer.writeheader()
        for method, payload in by_method.items():
            writer.writerow(
                {
                    "method": method,
                    "total": payload["total"],
                    "evaluable": payload["evaluable"],
                    "accuracy": payload["accuracy"],
                    "macro_f1": payload["macro_f1"],
                    "parse_rate": payload["parse_rate"],
                    "invalid": payload["invalid"],
                }
            )
    with (output_dir / "phi35_eval_checkpoints.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("method", "checkpoint", "total", "evaluable", "accuracy", "macro_f1", "parse_rate", "invalid"),
        )
        writer.writeheader()
        for method, checkpoint_payloads in by_checkpoint.items():
            for checkpoint, payload in checkpoint_payloads.items():
                writer.writerow(
                    {
                        "method": method,
                        "checkpoint": checkpoint,
                        "total": payload["total"],
                        "evaluable": payload["evaluable"],
                        "accuracy": payload["accuracy"],
                        "macro_f1": payload["macro_f1"],
                        "parse_rate": payload["parse_rate"],
                        "invalid": payload["invalid"],
                    }
                )
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-name", default="microsoft/Phi-3.5-mini-instruct")
    parser.add_argument("--methods", nargs="+", default=["zerorag", "hnsw"])
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--limit-per-method", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--max-input-tokens", type=int, default=0)
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    parser.add_argument("--torch-dtype", default="float16", choices=("float16", "bfloat16", "auto"))
    parser.add_argument("--attn-implementation", default="eager")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--progress-log", type=Path, default=DEFAULT_OUTPUT_DIR / "phi35_progress.jsonl")
    parser.add_argument("--metrics-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    requested_methods: list[str] = []
    for method in args.methods:
        canonical = _canonical_method(method)
        if canonical not in requested_methods:
            requested_methods.append(canonical)
    args.methods = requested_methods
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.metrics_only:
        tokenizer, model = _load_model(args)
        run_summary = {}
        for method in args.methods:
            examples_path = args.dataset_dir / f"{method}_examples.jsonl"
            if not examples_path.exists():
                raise FileNotFoundError(f"Missing examples file: {examples_path}")
            run_summary[method] = _run_method(
                method=method,
                examples_path=examples_path,
                output_dir=args.output_dir,
                progress_log=args.progress_log,
                tokenizer=tokenizer,
                model=model,
                args=args,
            )
        _write_json(args.output_dir / "phi35_run_summary.json", run_summary)
    metrics = compute_metrics(args.output_dir, args.methods)
    print(json.dumps(metrics, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()

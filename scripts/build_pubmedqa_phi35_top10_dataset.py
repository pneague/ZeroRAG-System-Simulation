#!/usr/bin/env python3
"""Build PubMedQA top-k-context examples for Phi-3.5 evaluation."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


PROMPT_TEMPLATE = """Here are the relevant documents:
{retrieved_context}

You are a helpful medical expert, and your task is to answer a multi-choice medical question using the relevant documents.

Please first think step-by-step and then choose the answer from the provided options.
Your responses will be used for research purposes only, so please have a definite answer.

Here is the question:
{question}

Here are the potential choices:
yes
no
maybe

Answer:
"""


DEFAULT_ROOT = Path("/home/pmneague/DecentANN_remote/Low_Coord_Semantic_DHT")
DEFAULT_SCRATCH = Path("/var/scratch/pmneague/Low_Coord_Semantic_DHT")
DEFAULT_Q30_HOME_RESULTS = DEFAULT_ROOT / "results/pubmedqa_expanded_all_mesh_alpha0p00007_q30"
DEFAULT_Q30_SCRATCH_DATA = DEFAULT_SCRATCH / "data/pubmedqa_expanded_all_mesh_alpha0p00007_q30/pubmedqa"
DEFAULT_SOURCE_FULL_DIR = DEFAULT_SCRATCH / "results/pubmedqa_full_hnsw_original_m32_ef512"
DEFAULT_OUTPUT_DIR = DEFAULT_SCRATCH / "data/pubmedqa_phi35_top5_checkpoints_q30_4x5k"
DEFAULT_SLICES = "q300k:295001-300000,q600k:595001-600000,q900k:895001-900000,q1200k:1195001-1200000"
DEFAULT_TOP_K = 5
VALID_LABELS = {"yes", "no", "maybe"}


@dataclass(frozen=True)
class SliceSpec:
    label: str
    start_line: int
    end_line: int

    @property
    def expected_rows(self) -> int:
        return self.end_line - self.start_line + 1

    def contains(self, line_number: int) -> bool:
        return self.start_line <= line_number <= self.end_line

    def to_json(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "expected_rows": self.expected_rows,
        }


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")


def _parse_slices(value: str | None, start_line: int | None, end_line: int | None) -> list[SliceSpec]:
    if value:
        specs: list[SliceSpec] = []
        for chunk in value.split(","):
            chunk = chunk.strip()
            if not chunk:
                continue
            if ":" not in chunk or "-" not in chunk:
                raise ValueError(f"Invalid slice spec {chunk!r}; expected label:start-end")
            label, bounds = chunk.split(":", 1)
            start_text, end_text = bounds.split("-", 1)
            spec = SliceSpec(label.strip(), int(start_text), int(end_text))
            if not spec.label:
                raise ValueError(f"Invalid slice spec {chunk!r}; label is empty")
            if spec.start_line <= 0 or spec.end_line < spec.start_line:
                raise ValueError(f"Invalid slice bounds for {spec.label}: {spec.start_line}-{spec.end_line}")
            specs.append(spec)
        if not specs:
            raise ValueError("--slices did not contain any usable slice specs")
        return specs
    if start_line is not None or end_line is not None:
        if start_line is None or end_line is None:
            raise ValueError("--start-line and --end-line must be provided together")
        if start_line <= 0 or end_line < start_line:
            raise ValueError("--start-line must be positive and <= --end-line")
        return [SliceSpec("selected", start_line, end_line)]
    return _parse_slices(DEFAULT_SLICES, None, None)


def _empty_counts() -> dict[str, int]:
    return {
        "rows": 0,
        "skipped_rows": 0,
        "empty_top10_rows": 0,
        "target_in_context_rows": 0,
    }


def _empty_example_counts() -> dict[str, int]:
    return {
        "examples": 0,
        "skipped_unlabeled": 0,
        "missing_doc_refs": 0,
        "missing_gold_answer": 0,
        "empty_context_examples": 0,
    }


def _slice_for_line(line_number: int, slices: list[SliceSpec]) -> SliceSpec | None:
    for spec in slices:
        if spec.contains(line_number):
            return spec
    return None


def _load_selected_retrieval_rows(
    path: Path,
    *,
    method: str,
    slices: list[SliceSpec],
    top_k: int,
) -> tuple[list[dict[str, Any]], set[int], set[str], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    qids: set[int] = set()
    doc_ids: set[str] = set()
    overall = _empty_counts()
    by_checkpoint = {spec.label: _empty_counts() for spec in slices}
    max_end_line = max(spec.end_line for spec in slices)
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line_number > max_end_line:
                break
            spec = _slice_for_line(line_number, slices)
            if spec is None:
                continue
            if not line.strip():
                continue
            row = json.loads(line)
            top_doc_ids = [str(doc_id) for doc_id in (row.get("top10_doc_ids") or [])[:top_k]]
            top_scores = [float(score) for score in (row.get("top10_scores") or [])[: len(top_doc_ids)]]
            qid = int(row["qid"])
            target_doc_id = str(row.get("target_doc_id", ""))
            target_in_context = bool(target_doc_id and target_doc_id in top_doc_ids)
            selected = {
                "line_number": line_number,
                "method": method,
                "checkpoint": spec.label,
                "event_index": int(row["event_index"]),
                "qid": qid,
                "target_doc_id": target_doc_id,
                "top10_doc_ids": top_doc_ids,
                "top10_scores": top_scores,
                "target_in_top10": target_in_context,
                "target_in_context": target_in_context,
                "skipped_reason": row.get("skipped_reason"),
                "query_variant": row.get("query_variant"),
                "rephrase_index": row.get("rephrase_index"),
            }
            rows.append(selected)
            qids.add(qid)
            doc_ids.update(top_doc_ids)
            for stats in (overall, by_checkpoint[spec.label]):
                stats["rows"] += 1
                stats["skipped_rows"] += int(row.get("skipped_reason") is not None)
                stats["empty_top10_rows"] += int(not top_doc_ids)
                stats["target_in_context_rows"] += int(target_in_context)
    for spec in slices:
        observed = by_checkpoint[spec.label]["rows"]
        if observed != spec.expected_rows:
            raise ValueError(
                f"{path} yielded {observed} rows for {spec.label}, expected {spec.expected_rows} "
                f"({spec.start_line}-{spec.end_line})"
            )
    stats = {**overall, "by_checkpoint": by_checkpoint}
    return rows, qids, doc_ids, stats


def _load_sampled_retrieval_rows(
    path: Path,
    *,
    method: str,
    sample_size: int,
    sample_label: str,
    random_seed: int,
    top_k: int,
    valid_answers: dict[int, dict[str, Any]] | None,
) -> tuple[list[dict[str, Any]], set[int], set[str], dict[str, Any]]:
    if sample_size <= 0:
        raise ValueError("--sample-size must be positive")
    rng = random.Random(random_seed)
    reservoir: list[dict[str, Any]] = []
    candidates = 0
    raw_rows = 0
    skipped_unlabeled_candidates = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            raw_rows += 1
            row = json.loads(line)
            qid = int(row["qid"])
            if valid_answers is not None and qid not in valid_answers:
                skipped_unlabeled_candidates += 1
                continue
            candidates += 1
            top_doc_ids = [str(doc_id) for doc_id in (row.get("top10_doc_ids") or [])[:top_k]]
            top_scores = [float(score) for score in (row.get("top10_scores") or [])[: len(top_doc_ids)]]
            target_doc_id = str(row.get("target_doc_id", ""))
            target_in_context = bool(target_doc_id and target_doc_id in top_doc_ids)
            selected = {
                "line_number": line_number,
                "method": method,
                "checkpoint": sample_label,
                "event_index": int(row["event_index"]),
                "qid": qid,
                "target_doc_id": target_doc_id,
                "top10_doc_ids": top_doc_ids,
                "top10_scores": top_scores,
                "target_in_top10": target_in_context,
                "target_in_context": target_in_context,
                "skipped_reason": row.get("skipped_reason"),
                "query_variant": row.get("query_variant"),
                "rephrase_index": row.get("rephrase_index"),
            }
            if len(reservoir) < sample_size:
                reservoir.append(selected)
            else:
                replace_at = rng.randrange(candidates)
                if replace_at < sample_size:
                    reservoir[replace_at] = selected
    if len(reservoir) != sample_size:
        raise ValueError(f"{path} yielded only {len(reservoir)} sample rows from {candidates} candidates")
    reservoir.sort(key=lambda item: int(item["line_number"]))
    qids = {int(row["qid"]) for row in reservoir}
    doc_ids = {doc_id for row in reservoir for doc_id in row["top10_doc_ids"]}
    overall = _empty_counts()
    for row in reservoir:
        overall["rows"] += 1
        overall["skipped_rows"] += int(row.get("skipped_reason") is not None)
        overall["empty_top10_rows"] += int(not row["top10_doc_ids"])
        overall["target_in_context_rows"] += int(bool(row.get("target_in_context")))
    stats = {
        **overall,
        "raw_rows_scanned": raw_rows,
        "sample_candidates": candidates,
        "skipped_unlabeled_candidates": skipped_unlabeled_candidates,
        "sample_size": sample_size,
        "sample_seed": random_seed,
        "by_checkpoint": {sample_label: overall.copy()},
    }
    return reservoir, qids, doc_ids, stats


def _load_answer_targets(path: Path, qids: set[int]) -> dict[int, dict[str, Any]]:
    answers: dict[int, dict[str, Any]] = {}
    remaining = set(qids)
    for row in _iter_jsonl(path):
        qid = int(row["qid"])
        if qid not in remaining:
            continue
        answers[qid] = row
        remaining.remove(qid)
        if not remaining:
            break
    if remaining:
        raise ValueError(f"Missing answer targets for {len(remaining)} qids, first={sorted(remaining)[:10]}")
    return answers


def _load_valid_answer_targets(path: Path) -> dict[int, dict[str, Any]]:
    answers: dict[int, dict[str, Any]] = {}
    for row in _iter_jsonl(path):
        gold_answer = str(row.get("final_decision") or "").strip().lower()
        if gold_answer in VALID_LABELS:
            answers[int(row["qid"])] = row
    if not answers:
        raise ValueError(f"No valid labeled answer targets found in {path}")
    return answers


def _load_doc_texts(path: Path, doc_ids: set[str]) -> dict[str, str]:
    docs: dict[str, str] = {}
    remaining = set(doc_ids)
    for row in _iter_jsonl(path):
        pubid = str(row.get("pubid", ""))
        if pubid not in remaining:
            continue
        docs[pubid] = str(row.get("doc_text") or "")
        remaining.remove(pubid)
        if not remaining:
            break
    return docs


def _format_context(
    *,
    top_doc_ids: list[str],
    top_scores: list[float],
    doc_texts: dict[str, str],
    max_doc_chars: int,
) -> tuple[str, list[str]]:
    sections: list[str] = []
    missing: list[str] = []
    for rank, doc_id in enumerate(top_doc_ids, start=1):
        score = top_scores[rank - 1] if rank - 1 < len(top_scores) else None
        text = doc_texts.get(doc_id)
        if text is None:
            missing.append(doc_id)
            text = "[missing document text]"
        elif max_doc_chars > 0 and len(text) > max_doc_chars:
            text = text[:max_doc_chars].rstrip() + "\n[truncated]"
        score_text = "nan" if score is None else f"{score:.6f}"
        sections.append(f"[Document {rank} | doc_id={doc_id} | score={score_text}]\n{text}")
    return "\n\n".join(sections), missing


def _write_examples(
    *,
    rows: list[dict[str, Any]],
    answers: dict[int, dict[str, Any]],
    doc_texts: dict[str, str],
    output_path: Path,
    max_doc_chars: int,
    labeled_only: bool,
) -> dict[str, Any]:
    stats = _empty_example_counts()
    by_checkpoint: dict[str, dict[str, int]] = {}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            checkpoint = str(row.get("checkpoint") or "selected")
            checkpoint_stats = by_checkpoint.setdefault(checkpoint, _empty_example_counts())
            answer = answers[int(row["qid"])]
            question = str(answer.get("question") or answer.get("original_question") or row.get("query_key") or "")
            gold_answer = str(answer.get("final_decision") or "").strip().lower()
            if labeled_only and gold_answer not in VALID_LABELS:
                stats["skipped_unlabeled"] += 1
                checkpoint_stats["skipped_unlabeled"] += 1
                continue
            retrieved_context, missing_doc_ids = _format_context(
                top_doc_ids=row["top10_doc_ids"],
                top_scores=row["top10_scores"],
                doc_texts=doc_texts,
                max_doc_chars=max_doc_chars,
            )
            prompt = PROMPT_TEMPLATE.format(
                retrieved_context=retrieved_context,
                question=question,
            )
            payload = {
                **row,
                "question": question,
                "original_question": answer.get("original_question"),
                "gold_answer": gold_answer,
                "long_answer": answer.get("long_answer"),
                "has_answer": bool(answer.get("has_answer")),
                "source_pubid": answer.get("source_pubid"),
                "config": answer.get("config"),
                "row_index": answer.get("row_index"),
                "retrieved_context": retrieved_context,
                "missing_doc_ids": missing_doc_ids,
                "prompt": prompt,
            }
            handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
            for current_stats in (stats, checkpoint_stats):
                current_stats["examples"] += 1
                current_stats["missing_doc_refs"] += len(missing_doc_ids)
                current_stats["missing_gold_answer"] += int(gold_answer not in VALID_LABELS)
                current_stats["empty_context_examples"] += int(not retrieved_context)
    stats["by_checkpoint"] = by_checkpoint
    return stats


def build_dataset(args: argparse.Namespace) -> dict[str, Any]:
    slices = _parse_slices(args.slices, args.start_line, args.end_line)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    all_method_inputs = {
        "zerorag": args.zerorag_jsonl,
        "hnsw": args.hnsw_jsonl,
    }
    unknown_methods = sorted(set(args.methods) - set(all_method_inputs))
    if unknown_methods:
        raise ValueError(f"Unknown methods requested: {unknown_methods}")
    method_inputs = {method: all_method_inputs[method] for method in args.methods}
    selected_rows: dict[str, list[dict[str, Any]]] = {}
    retrieval_stats: dict[str, dict[str, int]] = {}
    all_qids: set[int] = set()
    all_doc_ids: set[str] = set()
    valid_answers = _load_valid_answer_targets(args.answer_targets) if args.sample_size > 0 and args.labeled_only else None
    for method, path in method_inputs.items():
        if args.sample_size > 0:
            rows, qids, doc_ids, stats = _load_sampled_retrieval_rows(
                path,
                method=method,
                sample_size=args.sample_size,
                sample_label=args.sample_label,
                random_seed=args.random_seed,
                top_k=args.top_k,
                valid_answers=valid_answers,
            )
        else:
            rows, qids, doc_ids, stats = _load_selected_retrieval_rows(
                path,
                method=method,
                slices=slices,
                top_k=args.top_k,
            )
        selected_rows[method] = rows
        retrieval_stats[method] = stats
        all_qids.update(qids)
        all_doc_ids.update(doc_ids)

    answers = (
        {qid: valid_answers[qid] for qid in all_qids}
        if valid_answers is not None
        else _load_answer_targets(args.answer_targets, all_qids)
    )
    doc_texts = _load_doc_texts(args.corpus_manifest, all_doc_ids)

    example_paths: dict[str, str] = {}
    example_stats: dict[str, dict[str, Any]] = {}
    for method, rows in selected_rows.items():
        output_path = output_dir / f"{method}_examples.jsonl"
        example_paths[method] = str(output_path)
        example_stats[method] = _write_examples(
            rows=rows,
            answers=answers,
            doc_texts=doc_texts,
            output_path=output_path,
            max_doc_chars=args.max_doc_chars,
            labeled_only=args.labeled_only,
        )

    missing_doc_ids = sorted(all_doc_ids - set(doc_texts))
    manifest = {
        "dataset": "pubmedqa",
        "variant": (
            f"expanded_all_mesh_q30_phi35_top{args.top_k}_sample{args.sample_size}"
            if args.sample_size > 0
            else f"expanded_all_mesh_q30_phi35_top{args.top_k}_checkpoints_4x5k"
        ),
        "slices": [spec.to_json() for spec in slices] if args.sample_size <= 0 else [],
        "sample": {
            "enabled": args.sample_size > 0,
            "size": args.sample_size,
            "label": args.sample_label,
            "random_seed": args.random_seed,
            "labeled_only": bool(args.labeled_only),
        },
        "top_k": args.top_k,
        "max_doc_chars": args.max_doc_chars,
        "labeled_only": bool(args.labeled_only),
        "prompt_template": PROMPT_TEMPLATE,
        "inputs": {method: str(path) for method, path in method_inputs.items()},
        "answer_targets_path": str(args.answer_targets),
        "corpus_manifest_path": str(args.corpus_manifest),
        "output_dir": str(output_dir),
        "example_paths": example_paths,
        "retrieval_stats": retrieval_stats,
        "example_stats": example_stats,
        "unique_qids": len(all_qids),
        "unique_doc_ids": len(all_doc_ids),
        "loaded_doc_texts": len(doc_texts),
        "missing_doc_ids": missing_doc_ids[:100],
        "missing_doc_id_count": len(missing_doc_ids),
    }
    _write_json(output_dir / "manifest.json", manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zerorag-jsonl", type=Path, default=DEFAULT_Q30_HOME_RESULTS / "zerorag/pubmedqa_top10_low_coord.jsonl")
    parser.add_argument("--hnsw-jsonl", type=Path, default=DEFAULT_Q30_HOME_RESULTS / "hnsw_m16_ef64/pubmedqa_top10_hnsw.jsonl")
    parser.add_argument("--methods", nargs="+", default=["zerorag", "hnsw"], choices=("zerorag", "hnsw"))
    parser.add_argument("--answer-targets", type=Path, default=DEFAULT_Q30_SCRATCH_DATA / "answer_targets.jsonl")
    parser.add_argument("--corpus-manifest", type=Path, default=DEFAULT_SOURCE_FULL_DIR / "corpus_manifest.jsonl")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--slices",
        default=None,
        help=(
            "Comma-separated label:start-end slice specs. Defaults to the four q30 checkpoints: "
            f"{DEFAULT_SLICES}. If omitted with --start-line/--end-line, builds one legacy slice."
        ),
    )
    parser.add_argument("--start-line", type=int, default=None)
    parser.add_argument("--end-line", type=int, default=None)
    parser.add_argument("--sample-size", type=int, default=0, help="Randomly sample this many retrieval rows instead of using slices.")
    parser.add_argument("--sample-label", default="sample5k", help="Checkpoint label assigned to sampled rows.")
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument(
        "--max-doc-chars",
        type=int,
        default=0,
        help="Per-document character cap for retrieved_context; 0 keeps full document text.",
    )
    parser.add_argument(
        "--include-unlabeled",
        dest="labeled_only",
        action="store_false",
        help="Include PubMedQA rows without yes/no/maybe final_decision. Default excludes them.",
    )
    parser.set_defaults(labeled_only=True)
    return parser.parse_args()


def main() -> None:
    manifest = build_dataset(parse_args())
    print(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()

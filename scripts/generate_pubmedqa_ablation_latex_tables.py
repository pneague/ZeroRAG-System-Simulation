#!/usr/bin/env python3
"""Generate compact PubMedQA ablation tables with replicate uncertainty.

The generator reads completed ``metrics.json`` files for seed 42 plus repeat
roots for seeds 43 and 44. For every ablation setup it computes, per seed, the
executed-query-weighted final-10% recall and final-10% average request count
(``avg_cycles``), then reports mean +/- sample standard deviation.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, stdev

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.path_defaults import default_results_root  # noqa: E402


DEFAULT_RESULTS_ROOT = default_results_root()
DEFAULT_ROOTS = {
    42: DEFAULT_RESULTS_ROOT
    / "pubmedqa_expanded_all_mesh_alpha0p00007_q30/compact_retry/ablations",
    43: DEFAULT_RESULTS_ROOT
    / "pubmedqa_expanded_all_mesh_alpha0p00007_q30/compact_retry_seed_repeats/seed43/ablations",
    44: DEFAULT_RESULTS_ROOT
    / "pubmedqa_expanded_all_mesh_alpha0p00007_q30/compact_retry_seed_repeats/seed44/ablations",
}

FAMILY_ORDER = {
    "query_memory_limit": 10,
    "user_scalability": 20,
    "doc_scalability": 30,
    "degree": 40,
    "fanout": 50,
    "churn": 60,
}

FAMILY_TITLE = {
    "query_memory_limit": "Query-memory limit ablation",
    "user_scalability": "User scalability ablation",
    "doc_scalability": "Documents-per-peer scalability ablation",
    "degree": "Graph degree ablation",
    "fanout": "Fanout ablation",
    "churn": "Churn ablation",
}

FAMILY_PARAM_COLUMN = {
    "query_memory_limit": "Max routed queries",
    "user_scalability": "Users",
    "doc_scalability": "Docs / peer",
    "degree": "Graph degree",
    "fanout": "Fanout (rounds)",
    "churn": "Churn setting",
}

EXPECTED_RUNS = {
    "query_memory_limit": [
        "query_memory_limit_0",
        "query_memory_limit_10",
        "query_memory_limit_100",
        "query_memory_limit_1000",
        "query_memory_limit_10000",
    ],
    "user_scalability": [
        "users1000",
        "users5000",
        "users10000",
        "users50000",
        "users100000",
    ],
    "doc_scalability": [
        "docs_per_peer10",
        "docs_per_peer20",
        "docs_per_peer30",
        "docs_per_peer40",
        "docs_per_peer50",
    ],
    "degree": [
        "degree3",
        "degree5",
        "degree10",
        "degree25",
        "degree50",
        "degree75",
        "degree100",
    ],
    "fanout": [
        "fanout3_rounds10",
        "fanout6_rounds5",
        "fanout10_rounds3",
    ],
    "churn": [
        "drop50_gradual_replace50",
    ],
}


@dataclass(frozen=True)
class PerSeedMetrics:
    seed: int
    metrics_path: str
    total_events: int
    executed_queries: int
    skipped_queries: int
    recall_last_10pct: float
    avg_requests_last_10pct: float
    unsearchable_rate: float


@dataclass(frozen=True)
class SummaryMetric:
    mean: float
    std: float


@dataclass(frozen=True)
class TableRow:
    family: str
    run_name: str
    ablation_value: str
    recall_last_10pct: SummaryMetric
    avg_requests_last_10pct: SummaryMetric
    unsearchable_rate: SummaryMetric | None
    per_seed: list[PerSeedMetrics]


def _latex_escape(value: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in value)


def _natural_int(pattern: str, text: str) -> int | None:
    match = re.search(pattern, text)
    return int(match.group(1)) if match else None


def _format_count(value: int) -> str:
    return f"{value:,}".replace(",", r"{,}")


def _format_pct_summary(metric: SummaryMetric) -> str:
    if math.isnan(metric.std):
        return f"{metric.mean * 100:.1f}\\%"
    return f"{metric.mean * 100:.1f} $\\pm$ {metric.std * 100:.1f}\\%"


def _format_float_summary(metric: SummaryMetric) -> str:
    if math.isnan(metric.std):
        return f"{metric.mean:.2f}"
    return f"{metric.mean:.2f} $\\pm$ {metric.std:.2f}"


def _ablation_value(family: str, name: str) -> tuple[str, tuple]:
    if family == "query_memory_limit":
        value = _natural_int(r"query_memory_limit_(\d+)", name)
        if value is not None:
            return _format_count(value), (value, name)
    if family == "user_scalability":
        value = _natural_int(r"users(\d+)", name)
        if value is not None:
            return _format_count(value), (value, name)
    if family == "doc_scalability":
        value = _natural_int(r"docs_per_peer(\d+)", name)
        if value is not None:
            return str(value), (value, name)
    if family == "degree":
        value = _natural_int(r"degree(\d+)", name)
        if value is not None:
            return str(value), (value, name)
    if family == "fanout":
        fanout = _natural_int(r"fanout(\d+)", name)
        rounds = _natural_int(r"rounds(\d+)", name)
        if fanout is not None and rounds is not None:
            return f"{fanout} ({rounds})", (fanout, name)
    if family == "churn":
        return name.replace("_", " "), (0, name)
    return name.replace("_", " "), (0, name)


def _total_events_from_windows(metrics: dict) -> int:
    windows = metrics.get("windows") or []
    if not windows:
        return int(metrics.get("executed_queries") or 0) + int(metrics.get("skipped_queries") or 0)
    return max(int(window["event_end"]) for window in windows) + 1


def _weighted_last_10(metrics: dict, key: str) -> tuple[float, float]:
    windows = metrics.get("windows") or []
    if not windows:
        raise ValueError("metrics.json has no windows array; cannot compute final-10% metric")
    total_events = _total_events_from_windows(metrics)
    interval_start = 0.9 * total_events
    interval_end = float(total_events)
    numerator = 0.0
    denominator = 0.0
    for window in windows:
        start = float(window["event_start"])
        end = float(window["event_end"])
        if end <= start:
            end = start + 1.0
        overlap = max(0.0, min(end, interval_end) - max(start, interval_start))
        if overlap <= 0:
            continue
        fraction = overlap / (end - start)
        weight = float(window.get("executed_queries", 0)) * fraction
        numerator += float(window[key]) * weight
        denominator += weight
    if denominator <= 0:
        raise ValueError("final-10% interval has no executed queries")
    return numerator / denominator, denominator


def _unsearchable_rate(metrics: dict) -> float:
    executed = int(metrics.get("executed_queries") or 0)
    skipped = int(metrics.get("skipped_queries") or 0)
    total = executed + skipped
    return skipped / total if total else 0.0


def _summary(values: list[float]) -> SummaryMetric:
    if len(values) < 2:
        return SummaryMetric(mean=values[0], std=math.nan)
    return SummaryMetric(mean=mean(values), std=stdev(values))


def _parse_seed_root(raw: str) -> tuple[int, Path]:
    seed_text, sep, path_text = raw.partition("=")
    if not sep:
        raise argparse.ArgumentTypeError("seed root must have the form SEED=/path/to/ablations")
    try:
        seed = int(seed_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid seed {seed_text!r}") from exc
    return seed, Path(path_text)


def _metrics_path(root: Path, family: str, run_name: str) -> Path:
    return root / family / run_name / "metrics.json"


def _load_per_seed(seed: int, root: Path, family: str, run_name: str) -> PerSeedMetrics:
    path = _metrics_path(root, family, run_name)
    if not path.exists():
        raise FileNotFoundError(str(path))
    metrics = json.loads(path.read_text(encoding="utf-8"))
    recall, _ = _weighted_last_10(metrics, "recall")
    avg_requests, _ = _weighted_last_10(metrics, "avg_cycles")
    return PerSeedMetrics(
        seed=seed,
        metrics_path=str(path),
        total_events=_total_events_from_windows(metrics),
        executed_queries=int(metrics.get("executed_queries") or 0),
        skipped_queries=int(metrics.get("skipped_queries") or 0),
        recall_last_10pct=recall,
        avg_requests_last_10pct=avg_requests,
        unsearchable_rate=_unsearchable_rate(metrics),
    )


def _load_rows(roots: dict[int, Path], *, allow_missing: bool) -> tuple[list[TableRow], list[str]]:
    rows: list[TableRow] = []
    missing: list[str] = []
    for family in sorted(EXPECTED_RUNS, key=lambda item: FAMILY_ORDER[item]):
        for run_name in EXPECTED_RUNS[family]:
            per_seed: list[PerSeedMetrics] = []
            for seed, root in sorted(roots.items()):
                try:
                    per_seed.append(_load_per_seed(seed, root, family, run_name))
                except FileNotFoundError as exc:
                    missing.append(f"seed {seed}: {exc}")
            if missing and not allow_missing:
                continue
            if not per_seed:
                continue
            ablation_value, _ = _ablation_value(family, run_name)
            rows.append(
                TableRow(
                    family=family,
                    run_name=run_name,
                    ablation_value=ablation_value,
                    recall_last_10pct=_summary([item.recall_last_10pct for item in per_seed]),
                    avg_requests_last_10pct=_summary(
                        [item.avg_requests_last_10pct for item in per_seed]
                    ),
                    unsearchable_rate=(
                        _summary([item.unsearchable_rate for item in per_seed])
                        if family in {"user_scalability", "doc_scalability"}
                        else None
                    ),
                    per_seed=per_seed,
                )
            )
    return sorted(
        rows,
        key=lambda row: (FAMILY_ORDER[row.family], _ablation_value(row.family, row.run_name)[1]),
    ), missing


def _family_label(family: str) -> str:
    return f"tab:pubmedqa-{family.replace('_', '-')}"


def _write_latex(rows: list[TableRow], output_path: Path) -> None:
    by_family: dict[str, list[TableRow]] = {}
    for row in rows:
        by_family.setdefault(row.family, []).append(row)

    lines = [
        "% Auto-generated by scripts/generate_pubmedqa_ablation_latex_tables.py",
        "% Recall and avg. requests are executed-query-weighted means over events 90--100%.",
        "% Values are averaged across available seeds; +/- is shown only when multiple seeds are present.",
        "% Requires \\usepackage{booktabs}.",
        "",
    ]
    for family in sorted(by_family, key=lambda item: FAMILY_ORDER[item]):
        include_unsearchable = family in {"user_scalability", "doc_scalability"}
        lines.append("\\begin{table}[t]")
        lines.append("\\centering")
        lines.append("\\small")
        lines.append(
            "\\caption{"
            + _latex_escape(FAMILY_TITLE[family])
            + ". Recall and average requests are both computed over the final 10\\% of each run's event timeline.}"
        )
        lines.append("\\label{" + _family_label(family) + "}")
        if include_unsearchable:
            lines.append("\\begin{tabular}{lrrr}")
            lines.append("\\toprule")
            lines.append(
                f"{_latex_escape(FAMILY_PARAM_COLUMN[family])} & "
                "Recall (last 10\\%) & Avg. requests (last 10\\%) & Unsearchable rate \\\\"
            )
        else:
            lines.append("\\begin{tabular}{lrr}")
            lines.append("\\toprule")
            lines.append(
                f"{_latex_escape(FAMILY_PARAM_COLUMN[family])} & "
                "Recall (last 10\\%) & Avg. requests (last 10\\%) \\\\"
            )
        lines.append("\\midrule")
        for row in by_family[family]:
            cells = [
                _latex_escape(row.ablation_value),
                _format_pct_summary(row.recall_last_10pct),
                _format_float_summary(row.avg_requests_last_10pct),
            ]
            if include_unsearchable:
                assert row.unsearchable_rate is not None
                cells.append(_format_pct_summary(row.unsearchable_rate))
            lines.append(" & ".join(cells) + " \\\\")
        lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def _json_safe(value):
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    return value


def _write_audit(rows: list[TableRow], output_path: Path, missing: list[str]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "missing": missing,
        "rows": [asdict(row) for row in rows],
    }
    output_path.write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=True), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-root", action="append", type=_parse_seed_root)
    parser.add_argument("--allow-missing", action="store_true")
    parser.add_argument("--output-tex", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    roots = dict(args.seed_root) if args.seed_root else dict(DEFAULT_ROOTS)
    rows, missing = _load_rows(roots, allow_missing=args.allow_missing)
    if missing and not args.allow_missing:
        for item in missing:
            print(item)
        raise SystemExit(f"Missing {len(missing)} required replicate metrics files")
    _write_latex(rows, args.output_tex)
    _write_audit(rows, args.output_json, missing)
    print(
        json.dumps(
            {
                "rows": len(rows),
                "missing": len(missing),
                "seeds": sorted(roots),
                "output_tex": str(args.output_tex),
                "output_json": str(args.output_json),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

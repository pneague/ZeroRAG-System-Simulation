#!/usr/bin/env python3
"""Regenerate the three-method comparison graphs (ZeroRAG / Semantica / LSH).

Defaults match the analysis_requested bundle on DAS scratch; override paths for local runs.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.path_defaults import default_results_root, default_scratch_root  # noqa: E402

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ALPHA_LSH_DEFAULT = 0.12
ALPHA_PRIMARY_DEFAULT = 0.55
LW_LSH_DEFAULT = 0.85
LW_PRIMARY_DEFAULT = 2.3


@dataclass(frozen=True)
class RunSpec:
    dataset: str
    method: str
    label: str
    run_dir: Path
    metric: str


@dataclass(frozen=True)
class HnswSpec:
    dataset: str
    label: str
    result_dir: Path
    metric: str


def load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def lsh_label(run_dir: Path) -> str:
    man = load_json(run_dir / "run_manifest.json")
    cfg = man.get("lsh_config") or man.get("config") or {}
    if cfg:
        return f"LSH w={cfg.get('w')} k={cfg.get('k')} L={cfg.get('L')}"
    m = re.search(r"w([0-9.]+)_k(\d+)_L(\d+)", run_dir.name)
    return f"LSH w={m.group(1)} k={m.group(2)} L={m.group(3)}" if m else "LSH"


def fmt_num(v):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "n/a"
    if abs(v) >= 1000:
        return f"{v:,.0f}"
    if abs(v) >= 10:
        return f"{v:.1f}"
    return f"{v:.2f}"


def avg_checked(run_dir: Path, method: str):
    metrics = load_json(run_dir / "metrics.json")
    if method == "HNSW":
        selected = load_hnsw_selected(run_dir)
        value = selected.get("comparison_proxy_value_per_query") or selected.get(
            "avg_visited_vectors_per_query"
        )
        if isinstance(value, (int, float)):
            return float(value), "visited vectors"
        if isinstance(value, str) and value:
            return float(value), "visited vectors"
        return None, "visited vectors"
    if method == "ZeroRAG":
        value = metrics.get("overall_avg_total_vector_comparisons")
        if isinstance(value, (int, float)) and value > 0:
            return float(value), "doc/query vectors"
        return None, "doc/query vectors"
    if method == "LSH":
        value = load_json(run_dir / "lsh_metrics.json").get("avg_candidate_count")
        if isinstance(value, (int, float)):
            return float(value), "bucket candidates"
        return None, "bucket candidates"
    sem = load_json(run_dir / "semantica_metrics.json")
    q, hops = sem.get("queries_executed"), sem.get("chain_hops_total")
    if isinstance(q, (int, float)) and q and isinstance(hops, (int, float)):
        return float(hops) / float(q), "chain peer vectors"
    cyc = metrics.get("overall_avg_cycles")
    if isinstance(cyc, (int, float)):
        return max(0.0, float(cyc) - 1.0), "chain peer vectors"
    return None, "chain peer vectors"


def rolling_from_log(path: Path, metric: str, window: int, stride: int):
    vals = deque()
    success_sum = ret_sum = avail_sum = 0
    vec_sum, vec_count = 0.0, 0
    executed = 0
    points = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            ev = json.loads(line)
            executed += 1
            tvc = ev.get("total_vector_comparisons")
            if isinstance(tvc, (int, float)):
                vec_sum += float(tvc)
                vec_count += 1
            if metric == "recall":
                val = 1 if ev.get("result") == "+" else 0
                vals.append(val)
                success_sum += val
                if len(vals) > window:
                    success_sum -= vals.popleft()
                if len(vals) == window and executed % stride == 0:
                    points.append((executed // stride, success_sum / window))
            else:
                pair = (
                    int(ev.get("retrieved_relevant_doc_count") or 0),
                    int(ev.get("available_relevant_doc_count") or 0),
                )
                vals.append(pair)
                ret_sum += pair[0]
                avail_sum += pair[1]
                if len(vals) > window:
                    old_ret, old_avail = vals.popleft()
                    ret_sum -= old_ret
                    avail_sum -= old_avail
                if len(vals) == window and executed % stride == 0:
                    pct = (ret_sum / avail_sum) * 100.0 if avail_sum else 0.0
                    points.append((executed // stride, pct))
    return points, (vec_sum / vec_count if vec_count else None), executed


def series_for(specs: list[RunSpec], window: int, stride: int):
    series, rows = [], []
    for spec in specs:
        q = spec.run_dir / "query_log.jsonl"
        print(f"[plots] {spec.label}", flush=True)
        points, avg_from_log, executed_from_log = rolling_from_log(q, spec.metric, window, stride)
        avg, unit = avg_checked(spec.run_dir, spec.method)
        if avg is None and avg_from_log is not None:
            avg = avg_from_log
        metrics = load_json(spec.run_dir / "metrics.json")
        man = load_json(spec.run_dir / "run_manifest.json")
        summary = man.get("summary") if isinstance(man.get("summary"), dict) else {}
        rr = metrics.get("overall_retrieval_recall") or summary.get("overall_retrieval_recall")
        row = {
            "dataset": spec.dataset,
            "method": spec.method,
            "label": spec.label,
            "legend_label": f"{spec.label} (avg checked: {fmt_num(avg)} {unit})",
            "run_dir": str(spec.run_dir),
            "status": "ok",
            "metric": spec.metric,
            "point_count": len(points),
            "executed_queries": metrics.get("executed_queries")
            or summary.get("executed_queries")
            or executed_from_log,
            "overall_recall": metrics.get("overall_recall") or summary.get("overall_recall"),
            "overall_retrieval_pct": rr * 100.0 if isinstance(rr, (int, float)) else None,
            "overall_avg_cycles": metrics.get("overall_avg_cycles") or summary.get("overall_avg_cycles"),
            "overall_skip_rate": metrics.get("overall_skip_rate") or summary.get("overall_skip_rate"),
            "avg_checked": avg,
            "avg_checked_unit": unit,
        }
        rows.append(row)
        line_style = "-"
        if "no query memory" in spec.label:
            line_style = ":"
        elif "no new connections" in spec.label:
            line_style = ":"
        series.append(
            {
                "method": spec.method,
                "label": spec.label,
                "legend": row["legend_label"],
                "points": points,
                "line_style": line_style,
            }
        )
    return series, rows


def load_hnsw_selected(result_dir: Path) -> dict:
    payload = load_json(result_dir / "hnsw_best_dataset_comparison.json")
    if isinstance(payload, list) and payload:
        selected = payload[0].get("selected", {})
        return selected if isinstance(selected, dict) else {}
    return {}


def append_hnsw_baseline(
    specs: list[HnswSpec],
    series: list[dict],
    rows: list[dict],
    *,
    y_multiplier: float = 1.0,
) -> None:
    plotted_points = [point for item in series for point in item["points"]]
    if not plotted_points:
        return
    x_min = min(x for x, _ in plotted_points)
    x_max = max(x for x, _ in plotted_points)
    for spec in specs:
        selected = load_hnsw_selected(spec.result_dir)
        if not selected:
            print(f"[plots] skipping missing HNSW {spec.result_dir}", flush=True)
            continue
        recall = selected.get("target_visited_recall", selected.get("recall"))
        if recall in (None, ""):
            print(f"[plots] skipping HNSW without recall {spec.result_dir}", flush=True)
            continue
        ef_search = int(selected.get("ef_search", 0))
        avg, unit = avg_checked(spec.result_dir, "HNSW")
        y_value = float(recall) * y_multiplier
        label = f"{spec.label} ef={ef_search}"
        row = {
            "dataset": spec.dataset,
            "method": "HNSW",
            "label": label,
            "legend_label": f"{label} (avg checked: {fmt_num(avg)} {unit})",
            "run_dir": str(spec.result_dir),
            "status": "aggregate_baseline",
            "metric": spec.metric,
            "point_count": 2,
            "executed_queries": selected.get("query_count"),
            "overall_recall": float(recall),
            "overall_retrieval_pct": y_value if y_multiplier == 100.0 else None,
            "overall_avg_cycles": None,
            "overall_skip_rate": None,
            "avg_checked": avg,
            "avg_checked_unit": unit,
            "hnsw_backend": selected.get("backend"),
            "hnsw_success_criterion": selected.get("hnsw_success_criterion"),
            "hnsw_top_k": selected.get("top_k"),
        }
        rows.append(row)
        series.append(
            {
                "method": "HNSW",
                "label": label,
                "legend": row["legend_label"],
                "points": [(x_min, y_value), (x_max, y_value)],
                "line_style": "--",
            }
        )


def plot(
    series,
    path: Path,
    title: str,
    ylabel: str,
    ylim: tuple[float, float],
    notes: str,
    *,
    window: int,
    stride: int,
    alpha_lsh: float,
    alpha_primary: float,
    lw_lsh: float,
    lw_primary: float,
):
    fig, ax = plt.subplots(figsize=(13, 7))
    for item in series:
        is_lsh = item["method"] == "LSH"
        pts = item["points"]
        alpha = alpha_lsh if is_lsh else alpha_primary
        lw = lw_lsh if is_lsh else lw_primary
        z = 1 if is_lsh else (2 if item["method"] == "HNSW" else 3)
        if item.get("label") == "ZeroRAG no query memory":
            z = 4
        elif item.get("label") == "ZeroRAG no new connections":
            z = 4
        elif item.get("label") == "ZeroRAG":
            z = 5
        ax.plot(
            [x for x, _ in pts],
            [y for _, y in pts],
            label=item["legend"],
            linewidth=lw,
            alpha=alpha,
            linestyle=item.get("line_style", "-"),
            zorder=z,
        )
    if stride == window:
        ax.set_xlabel(f"{window}-query rolling-window index")
    else:
        ax.set_xlabel(f"Sample index ({window}-query rolling mean, sampled every {stride} queries)")
    ax.set_ylabel(ylabel)
    ax.set_ylim(*ylim)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def manifest_series_entries(rows: list[dict], plot_series: list[dict]) -> list[dict]:
    """Align with historical analysis_manifest.json schema under \"series\"."""
    entries = []
    for row, ps in zip(rows, plot_series):
        entries.append(
            {
                "label": row["label"],
                "method": row["method"],
                "avg_checked": row["avg_checked"],
                "avg_checked_unit": row["avg_checked_unit"],
                "overall_recall": row["overall_recall"],
                "overall_retrieval_pct": row["overall_retrieval_pct"],
                "overall_avg_cycles": row["overall_avg_cycles"],
                "legend_label": row["legend_label"],
                "points": ps["points"],
            }
        )
    return entries


def write_csv(path: Path, rows: list[dict]):
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def lsh_specs(dataset: str, pattern: str, results_root: Path) -> list[RunSpec]:
    def key(p: Path):
        m = re.search(r"w([0-9.]+)_k(\d+)_L(\d+)", p.name)
        return (float(m.group(1)), int(m.group(2)), int(m.group(3))) if m else (999.0, 999, 999)

    return [
        RunSpec(dataset, "LSH", lsh_label(p), p, "recall")
        for p in sorted(results_root.glob(pattern), key=key)
    ]


def existing_spec(
    dataset: str,
    method: str,
    label: str,
    run_dir: Path,
    metric: str,
) -> list[RunSpec]:
    if (run_dir / "query_log.jsonl").exists():
        return [RunSpec(dataset, method, label, run_dir, metric)]
    print(f"[plots] skipping missing run {run_dir}", flush=True)
    return []


def default_specs(results_root: Path) -> tuple[list[RunSpec], list[RunSpec], list[RunSpec]]:
    aol_specs = [
        RunSpec(
            "AOL",
            "ZeroRAG",
            "ZeroRAG",
            results_root
            / "aol4foltr_seeded_randomized_timeline_semantic_query_memory_forming_new_connections_exact_vector_comparisons_20260507T134722_run",
            "recall",
        ),
        RunSpec(
            "AOL",
            "Semantica",
            "Semantica",
            results_root / "semantica_aol_seeded_randomized_timeline_run",
            "recall",
        ),
    ]
    aol_specs += existing_spec(
        "AOL",
        "ZeroRAG",
        "ZeroRAG no query memory",
        results_root
        / "aol4foltr_seeded_randomized_timeline_replica_only_forming_new_connections_exact_vector_comparisons_20260507T134722_run",
        "recall",
    )
    aol_specs += existing_spec(
        "AOL",
        "ZeroRAG",
        "ZeroRAG no new connections",
        results_root
        / "aol4foltr_seeded_randomized_timeline_semantic_query_memory_static_initial_connections_exact_vector_comparisons_20260518_run",
        "recall",
    )
    aol_specs += lsh_specs("AOL", "lsh_seeded_randomized_full_w*_k*_L*_run", results_root)

    amazon_specs = [
        RunSpec(
            "Amazon electronics",
            "ZeroRAG",
            "ZeroRAG",
            results_root
            / "amazon_electronics_multi_relevant_lowcoord_query_memory_forming_best_contributor_run",
            "retrieval_pct",
        ),
        RunSpec(
            "Amazon electronics",
            "Semantica",
            "Semantica",
            results_root / "amazon_electronics_multi_relevant_semantica_run",
            "retrieval_pct",
        ),
    ]
    amazon_specs += existing_spec(
        "Amazon electronics",
        "ZeroRAG",
        "ZeroRAG no query memory",
        results_root
        / "amazon_electronics_multi_relevant_lowcoord_replica_only_forming_best_contributor_run",
        "retrieval_pct",
    )
    amazon_specs += existing_spec(
        "Amazon electronics",
        "ZeroRAG",
        "ZeroRAG no new connections",
        results_root / "amazon_electronics_multi_relevant_lowcoord_query_memory_static_run",
        "retrieval_pct",
    )
    amazon_specs += existing_spec(
        "Amazon electronics",
        "ZeroRAG",
        "ZeroRAG no query memory, no new connections",
        results_root / "amazon_electronics_multi_relevant_lowcoord_replica_only_static_run",
        "retrieval_pct",
    )
    amazon_specs += [
        RunSpec("Amazon electronics", "LSH", lsh_label(p), p, "retrieval_pct")
        for p in sorted(results_root.glob("amazon_electronics_multi_relevant_lsh*_run"))
    ]

    synth_specs = [
        RunSpec(
            "Synthetic",
            "ZeroRAG",
            "ZeroRAG",
            results_root / "synthetic_seeded_randomized_semantic_forming_best_first_run",
            "recall",
        ),
        RunSpec(
            "Synthetic",
            "Semantica",
            "Semantica",
            results_root / "synthetic_semantica_seeded_randomized_timeline_run",
            "recall",
        ),
    ]
    synth_specs += existing_spec(
        "Synthetic",
        "ZeroRAG",
        "ZeroRAG no query memory",
        results_root / "synthetic_seeded_randomized_replica_forming_best_first_run",
        "recall",
    )
    synth_specs += existing_spec(
        "Synthetic",
        "ZeroRAG",
        "ZeroRAG no new connections",
        results_root / "synthetic_seeded_randomized_semantic_static_best_first_run",
        "recall",
    )
    synth_specs += lsh_specs("Synthetic", "synthetic_lsh_seeded_randomized_w*_k*_L*_run", results_root)

    return aol_specs, amazon_specs, synth_specs


def default_hnsw_specs(results_root: Path) -> tuple[list[HnswSpec], list[HnswSpec], list[HnswSpec]]:
    return (
        [
            HnswSpec(
                "AOL",
                "HNSW",
                results_root / "hnsw_full_aol4foltr_trace_threads16",
                "recall",
            )
        ],
        [
            HnswSpec(
                "Amazon electronics",
                "HNSW",
                results_root / "hnsw_full_amazon_electronics_trace_threads16",
                "retrieval_pct",
            )
        ],
        [
            HnswSpec(
                "Synthetic",
                "HNSW",
                results_root / "hnsw_full_synthetic_trace_threads16",
                "recall",
            )
        ],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root",
        type=Path,
        default=default_results_root(),
        help="Directory containing run_* folders with query_log.jsonl",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=default_scratch_root() / "analysis_requested_2026-05-18",
        help="Output directory for PNGs, CSV, partial manifest patch",
    )
    parser.add_argument("--window", type=int, default=1000, help="Rolling window size (queries)")
    parser.add_argument(
        "--stride",
        type=int,
        default=None,
        help="Sample every N executed queries (default: same as --window)",
    )
    parser.add_argument("--alpha-primary", type=float, default=ALPHA_PRIMARY_DEFAULT)
    parser.add_argument("--alpha-lsh", type=float, default=ALPHA_LSH_DEFAULT)
    parser.add_argument("--lw-primary", type=float, default=LW_PRIMARY_DEFAULT)
    parser.add_argument("--lw-lsh", type=float, default=LW_LSH_DEFAULT)
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=None,
        help="Optional analysis_manifest.json to patch (defaults to --out-dir/analysis_manifest.json)",
    )
    args = parser.parse_args()
    window = args.window
    stride = args.stride if args.stride is not None else window

    results_root = args.results_root
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    suffix = f"rolling{window}"
    graph1 = out_dir / f"graph1_aol_recall_{suffix}.png"
    graph2 = out_dir / f"graph2_amazon_retrieval_pct_{suffix}.png"
    graph3 = out_dir / f"graph3_synthetic_recall_{suffix}.png"

    note_style = (
        f"Opacity: ZeroRAG/Semantica alpha={args.alpha_primary}, LSH alpha={args.alpha_lsh}. "
        "ZeroRAG no query memory uses replica_only routing; ZeroRAG no new connections uses static_initial_connections topology. "
        "LSH checked count is summed bucket-candidate entries; Semantica checked count is chain peer-vector hops. "
        "HNSW dashed lines are aggregate target-visited baselines from traced HNSW sweeps."
    )

    aol_specs, amazon_specs, synth_specs = default_specs(results_root)
    aol_hnsw_specs, amazon_hnsw_specs, synth_hnsw_specs = default_hnsw_specs(results_root)

    aol_series, aol_rows = series_for(aol_specs, window, stride)
    append_hnsw_baseline(aol_hnsw_specs, aol_series, aol_rows)
    plot(
        aol_series,
        graph1,
        "AOL: Recall, ZeroRAG Variants vs Semantica vs LSH/HNSW Baselines",
        "Recall (rolling mean)",
        (0, 1.02),
        note_style,
        window=window,
        stride=stride,
        alpha_lsh=args.alpha_lsh,
        alpha_primary=args.alpha_primary,
        lw_lsh=args.lw_lsh,
        lw_primary=args.lw_primary,
    )

    amazon_series, amazon_rows = series_for(amazon_specs, window, stride)
    append_hnsw_baseline(amazon_hnsw_specs, amazon_series, amazon_rows, y_multiplier=100.0)
    amazon_note = (
        "Partial ZeroRAG snapshot. "
        + note_style.replace("Semantica checked count", "Legend avg checked semantics unchanged").replace(
            "LSH checked count is summed bucket-candidate entries; ",
            "Metric is sum(retrieved relevant docs)/sum(available relevant docs) per rolling window. ",
        )
        + " Amazon HNSW uses target-visited recall scaled to percent, not the multi-relevant denominator."
    )
    plot(
        amazon_series,
        graph2,
        "Amazon Electronics: ZeroRAG Variants vs Semantica vs LSH/HNSW Baselines",
        "Relevant docs retrieved (%)",
        (0, 100),
        amazon_note,
        window=window,
        stride=stride,
        alpha_lsh=args.alpha_lsh,
        alpha_primary=args.alpha_primary,
        lw_lsh=args.lw_lsh,
        lw_primary=args.lw_primary,
    )

    synth_series, synth_rows = series_for(synth_specs, window, stride)
    append_hnsw_baseline(synth_hnsw_specs, synth_series, synth_rows)
    plot(
        synth_series,
        graph3,
        "Synthetic: Recall, ZeroRAG Variants vs Semantica vs LSH/HNSW Baselines",
        "Recall (rolling mean)",
        (0, 1.02),
        note_style,
        window=window,
        stride=stride,
        alpha_lsh=args.alpha_lsh,
        alpha_primary=args.alpha_primary,
        lw_lsh=args.lw_lsh,
        lw_primary=args.lw_primary,
    )

    write_csv(out_dir / "method_summary.csv", aol_rows + amazon_rows + synth_rows)

    manifest_path = args.manifest_path or (out_dir / "analysis_manifest.json")
    manifest = load_json(manifest_path)
    manifest["window_size"] = window
    manifest["stride"] = stride
    manifest["graphs"] = {
        "graph1": str(graph1),
        "graph2": str(graph2),
        "graph3": str(graph3),
    }
    manifest["series"] = {
        "aol": manifest_series_entries(aol_rows, aol_series),
        "amazon": manifest_series_entries(amazon_rows, amazon_series),
        "synthetic": manifest_series_entries(synth_rows, synth_series),
    }
    manifest.setdefault("notes", {})["graph_style_update"] = (
        f"Rolling window={window}, stride={stride}; ZeroRAG/Semantica alpha={args.alpha_primary}, "
        f"LSH alpha={args.alpha_lsh} (linewidth primary={args.lw_primary}, LSH={args.lw_lsh}); "
        "ZeroRAG replica_only/static_initial_connections variants included where matching logs are present; "
        "HNSW aggregate dashed baselines included where traced sweep outputs are present."
    )
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("[plots] done", out_dir, flush=True)


if __name__ == "__main__":
    main()

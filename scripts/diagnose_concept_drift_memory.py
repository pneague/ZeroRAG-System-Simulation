#!/usr/bin/env python3
"""Deep diagnostic over concept-drift query logs for the memory-variant ablations.

Single pass per query_log.jsonl. Emits, per run:
  - overall recall, termination-reason breakdown, avg cycles
  - memory-consultation stats (routed_query_memory_comparisons)
  - temporal decile recall curve (learning over time)
  - drift-window topic vs non-topic split with per-doc recall, term reasons, cycle histogram
  - "success used memory" proxy: fraction of successes whose response chain touched routed-query memory
"""
from __future__ import annotations

import argparse
import gzip
import json
from collections import Counter, defaultdict
from pathlib import Path


def load_topic_docs(doc_topics_path: Path, mesh_term: str) -> set[str]:
    docs: set[str] = set()
    with gzip.open(doc_topics_path, "rt") as f:
        f.readline()
        for line in f:
            parts = line.rstrip("\n").split(",")
            if len(parts) >= 4 and parts[3] == mesh_term:
                docs.add(parts[0])
    return docs


def cyc_bucket(c: int) -> str:
    if c <= 1:
        return "1"
    if c <= 2:
        return "2"
    if c <= 5:
        return "3-5"
    if c <= 10:
        return "6-10"
    if c <= 25:
        return "11-25"
    if c <= 49:
        return "26-49"
    return "50(cap)"


def analyze(log_path: Path, topic_docs: set[str], w0: int, w1: int, total_events: int) -> dict:
    decile_size = max(1, total_events // 10)

    overall = dict(n=0, succ=0, term=Counter(), cyc=0,
                   mem_consulted=0, mem_consulted_succ=0,
                   mem_comp_sum=0, succ_cyc=0)
    deciles = [dict(n=0, succ=0) for _ in range(10)]

    topic = dict(n=0, succ=0, term=Counter(), cyc=0, cyc_succ=0, cyc_succ_n=0,
                 cyc_hist=Counter(), exh_cyc=[], per_doc=Counter(), per_doc_succ=Counter(),
                 mem_consulted=0, mem_consulted_succ=0)
    other_w = dict(n=0, succ=0, term=Counter(), cyc=0)

    with log_path.open() as f:
        for line in f:
            e = json.loads(line)
            ei = e["event_index"]
            succ = e["result"] == "+"
            cyc = e.get("cycles_used", 0)
            memc = e.get("routed_query_memory_comparisons", 0) or 0

            overall["n"] += 1
            overall["cyc"] += cyc
            overall["term"][e.get("termination_reason")] += 1
            overall["mem_comp_sum"] += memc
            if memc > 0:
                overall["mem_consulted"] += 1
            if succ:
                overall["succ"] += 1
                overall["succ_cyc"] += cyc
                if memc > 0:
                    overall["mem_consulted_succ"] += 1
            d = min(9, ei // decile_size)
            deciles[d]["n"] += 1
            if succ:
                deciles[d]["succ"] += 1

            if w0 <= ei < w1:
                is_topic = e["clicked_doc_id"] in topic_docs
                grp = topic if is_topic else other_w
                grp["n"] += 1
                grp["cyc"] += cyc
                grp["term"][e.get("termination_reason")] += 1
                if succ:
                    grp["succ"] += 1
                if is_topic:
                    d_id = e["clicked_doc_id"]
                    topic["per_doc"][d_id] += 1
                    if succ:
                        topic["per_doc_succ"][d_id] += 1
                        topic["cyc_succ"] += cyc
                        topic["cyc_succ_n"] += 1
                        topic["cyc_hist"][cyc_bucket(cyc)] += 1
                        if memc > 0:
                            topic["mem_consulted_succ"] += 1
                    elif e.get("termination_reason") == "exhausted":
                        topic["exh_cyc"].append(cyc)
                    if memc > 0:
                        topic["mem_consulted"] += 1

    return {
        "overall": overall,
        "deciles": deciles,
        "topic": topic,
        "other_w": other_w,
    }


def fmt_overall(o: dict) -> str:
    n = o["n"] or 1
    s = o["succ"] or 1
    return (
        f"  n={o['n']:,} recall={o['succ']/n:.4f} avg_cycles={o['cyc']/n:.2f} "
        f"avg_cycles_succ={o['succ_cyc']/s:.2f}\n"
        f"  termination={dict(o['term'])}\n"
        f"  mem_consulted_frac={o['mem_consulted']/n:.4f} "
        f"mem_consulted_among_succ={o['mem_consulted_succ']/s:.4f} "
        f"avg_mem_comparisons={o['mem_comp_sum']/n:.1f}"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--doc-topics", type=Path, required=True)
    ap.add_argument("--mesh-term", default="Peptostreptococcus")
    ap.add_argument("--w0", type=int, default=600000)
    ap.add_argument("--w1", type=int, default=900000)
    ap.add_argument("--total-events", type=int, default=1500000)
    ap.add_argument("--runs", nargs="+", required=True,
                    help="label=path/to/query_log.jsonl pairs")
    args = ap.parse_args()

    topic_docs = load_topic_docs(args.doc_topics, args.mesh_term)
    print(f"# topic docs for {args.mesh_term}: {len(topic_docs)} -> {sorted(topic_docs)}\n")

    for spec in args.runs:
        label, _, path = spec.partition("=")
        res = analyze(Path(path), topic_docs, args.w0, args.w1, args.total_events)
        o = res["overall"]
        t = res["topic"]
        ow = res["other_w"]
        print("=" * 78)
        print(f"RUN: {label}")
        print("OVERALL")
        print(fmt_overall(o))
        print("  decile recall:", " ".join(
            f"{i}:{(dd['succ']/dd['n'] if dd['n'] else 0):.2f}" for i, dd in enumerate(res["deciles"])
        ))
        tn = t["n"] or 1
        print(f"DRIFT-WINDOW TOPIC  n={t['n']:,} recall={t['succ']/tn:.4f} "
              f"avg_cycles={t['cyc']/tn:.2f} term={dict(t['term'])}")
        print(f"  mem_consulted_among_topic_succ={t['mem_consulted_succ']/(t['succ'] or 1):.4f}")
        print(f"  topic success cycle hist={dict(t['cyc_hist'])}")
        if t["exh_cyc"]:
            ex = sorted(t["exh_cyc"])
            mid = ex[len(ex) // 2]
            print(f"  topic exhausted: count={len(ex)} median_cyc={mid} "
                  f"min={ex[0]} max={ex[-1]}")
        print("  per-doc recall:")
        for d_id in sorted(t["per_doc"]):
            tot = t["per_doc"][d_id]
            sc = t["per_doc_succ"][d_id]
            print(f"    {d_id}: {sc/tot:.4f} (n={tot})")
        own = ow["n"] or 1
        print(f"DRIFT-WINDOW NON-TOPIC  n={ow['n']:,} recall={ow['succ']/own:.4f} "
              f"avg_cycles={ow['cyc']/own:.2f} term={dict(ow['term'])}")
        print()


if __name__ == "__main__":
    main()

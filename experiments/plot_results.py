#!/usr/bin/env python3
"""Plot convergence curves from Low_Coord_Semantic_DHT metrics."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main() -> None:
    default_output_root = Path(
        os.environ.get("LOW_COORD_DHT_OUTPUT_DIR", Path("results") / "zerorag_semantic_dht")
    )
    parser = argparse.ArgumentParser(description="Plot Low_Coord_Semantic_DHT metrics")
    parser.add_argument(
        "--metrics-json",
        default=str(default_output_root / "metrics.json"),
        help="Path to metrics.json",
    )
    parser.add_argument(
        "--output-dir",
        default=str(default_output_root / "plots"),
        help="Directory for generated plots",
    )
    args = parser.parse_args()

    import matplotlib.pyplot as plt

    payload = json.loads(Path(args.metrics_json).read_text(encoding="utf-8"))
    windows = payload.get("windows", [])
    if not windows:
        raise SystemExit(f"No windows found in {args.metrics_json}")

    x = [window["window_index"] for window in windows]
    recalls = [window["recall"] for window in windows]
    retrieval_recalls = [
        window.get("retrieval_recall", 0.0)
        for window in windows
    ]
    avg_cycles = [window["avg_cycles"] for window in windows]
    skip_rates = [window["skip_rate"] for window in windows]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 5))
    plt.plot(x, recalls, label="Recall")
    plt.xlabel("Window index")
    plt.ylabel("Recall")
    plt.title("Recall convergence")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "recall_convergence.png", dpi=150)
    plt.close()

    if any(value > 0 for value in retrieval_recalls):
        plt.figure(figsize=(10, 5))
        plt.plot(x, retrieval_recalls, label="Retrieval recall", color="tab:purple")
        plt.xlabel("Window index")
        plt.ylabel("Retrieval recall")
        plt.title("Multi-relevant retrieval recall convergence")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / "retrieval_recall_convergence.png", dpi=150)
        plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(x, avg_cycles, label="Average cycles", color="tab:orange")
    plt.xlabel("Window index")
    plt.ylabel("Average cycles")
    plt.title("Cycle usage convergence")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "cycles_convergence.png", dpi=150)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(x, skip_rates, label="Skip rate", color="tab:green")
    plt.xlabel("Window index")
    plt.ylabel("Skip rate")
    plt.title("Skip-rate trend")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "skip_rate.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    main()

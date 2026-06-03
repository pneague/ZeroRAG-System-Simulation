#!/usr/bin/env python3
"""Plot Phi-3.5 checkpoint-level QA metrics."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


DEFAULT_SCRATCH = Path("/var/scratch/pmneague/Low_Coord_Semantic_DHT")
DEFAULT_DATASET_DIR = DEFAULT_SCRATCH / "data/pubmedqa_phi35_top5_checkpoints_q30_4x5k"
DEFAULT_OUTPUT_DIR = DEFAULT_SCRATCH / "results/pubmedqa_phi35_top5_checkpoints_q30_4x5k"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _checkpoint_order(dataset_dir: Path, metrics: dict[str, Any]) -> list[str]:
    manifest_path = dataset_dir / "manifest.json"
    if manifest_path.exists():
        manifest = _read_json(manifest_path)
        labels = [str(item["label"]) for item in manifest.get("slices", []) if item.get("label")]
        if labels:
            return labels
    seen: set[str] = set()
    for method_payload in metrics.get("by_checkpoint", {}).values():
        seen.update(str(label) for label in method_payload)
    return sorted(seen)


def _write_tidy_csv(output_path: Path, metrics: dict[str, Any], order: list[str]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("method", "checkpoint", "total", "evaluable", "accuracy", "macro_f1", "parse_rate", "invalid"),
        )
        writer.writeheader()
        for method, checkpoint_payloads in metrics.get("by_checkpoint", {}).items():
            for checkpoint in order:
                payload = checkpoint_payloads.get(checkpoint)
                if not payload:
                    continue
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


def _plot_metric(
    *,
    metrics: dict[str, Any],
    order: list[str],
    metric: str,
    ylabel: str,
    output_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    x_values = list(range(len(order)))
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    for method, checkpoint_payloads in metrics.get("by_checkpoint", {}).items():
        y_values: list[float] = []
        labels: list[str] = []
        for checkpoint in order:
            payload = checkpoint_payloads.get(checkpoint)
            if not payload:
                y_values.append(float("nan"))
                labels.append("")
                continue
            y_values.append(float(payload.get(metric, 0.0)))
            labels.append(f"{float(payload.get(metric, 0.0)):.3f}\nn={int(payload.get('evaluable', 0))}")
        ax.plot(x_values, y_values, marker="o", linewidth=2, label=method)
        for x_value, y_value, label in zip(x_values, y_values, labels, strict=True):
            if label and y_value == y_value:
                ax.annotate(label, (x_value, y_value), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8)

    ax.set_xticks(x_values)
    ax.set_xticklabels(order)
    ax.set_xlabel("Query checkpoint")
    ax.set_ylabel(ylabel)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics_path = args.output_dir / "phi35_eval_metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing metrics file: {metrics_path}")
    metrics = _read_json(metrics_path)
    order = _checkpoint_order(args.dataset_dir, metrics)
    figures_dir = args.output_dir / "figures"
    _write_tidy_csv(figures_dir / "phi35_checkpoint_metrics.csv", metrics, order)
    _plot_metric(
        metrics=metrics,
        order=order,
        metric="macro_f1",
        ylabel="Macro-F1",
        output_path=figures_dir / "phi35_checkpoint_macro_f1.png",
    )
    _plot_metric(
        metrics=metrics,
        order=order,
        metric="accuracy",
        ylabel="Accuracy",
        output_path=figures_dir / "phi35_checkpoint_accuracy.png",
    )
    print(json.dumps({"figures_dir": str(figures_dir), "checkpoints": order}, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Generate remote scenario comparison plots, copy them locally, and open them."""

from __future__ import annotations

import argparse
import subprocess
import sys
import textwrap
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.path_defaults import (  # noqa: E402
    default_remote_host,
    default_remote_root,
    default_remote_scratch,
)


DEFAULT_REMOTE_HOST = default_remote_host()
DEFAULT_REMOTE_ROOT = default_remote_root()
DEFAULT_REMOTE_SCRATCH = default_remote_scratch()


def _run(command: list[str]) -> None:
    subprocess.run(command, check=True)


def _remote_plot_script(
    remote_results_root: str,
    remote_plot_dir: str,
    window_size: int,
    x_min: int | None,
    x_max: int | None,
    comparison_axis: str,
    routing_mode: str,
    topology_mode: str,
    scenario: str | None,
) -> str:
    return textwrap.dedent(
        f"""
        import json
        from pathlib import Path

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        results_root = Path({remote_results_root!r})
        plot_dir = Path({remote_plot_dir!r})
        plot_dir.mkdir(parents=True, exist_ok=True)
        x_min = {x_min!r}
        x_max = {x_max!r}
        comparison_axis = {comparison_axis!r}
        selected_routing_mode = {routing_mode!r}
        selected_topology_mode = {topology_mode!r}
        selected_scenario = {scenario!r}

        def run_dir_name(scenario_key: str, routing_mode: str, topology_mode: str) -> str:
            name = scenario_key
            if routing_mode == "replica_only":
                name += "_replica_only"
            name += f"_{{topology_mode}}_run"
            return name

        scenario_titles = [
            ("empty_start_chronological", "Chronological"),
            ("seeded_fixed_inventory", "Seeded Chronological"),
            ("seeded_randomized_timeline", "Seeded Randomized"),
        ]
        if selected_scenario is not None:
            scenario_titles = [
                (scenario_key, title)
                for scenario_key, title in scenario_titles
                if scenario_key == selected_scenario
            ]
            if not scenario_titles:
                raise SystemExit(f"Unsupported scenario={{selected_scenario}}")

        if comparison_axis == "routing_mode":
            left_label = "semantic_query_memory"
            right_label = "replica_only"
            filename_suffix = "routing_mode_comparison"
            scenario_specs = [
                (
                    scenario_key,
                    title,
                    results_root
                    / run_dir_name(scenario_key, "semantic_query_memory", selected_topology_mode)
                    / "metrics.json",
                    results_root
                    / run_dir_name(scenario_key, "replica_only", selected_topology_mode)
                    / "metrics.json",
                )
                for scenario_key, title in scenario_titles
            ]
        elif comparison_axis == "topology_mode":
            left_label = "forming_new_connections"
            right_label = "static_initial_connections"
            filename_suffix = "topology_mode_comparison"
            scenario_specs = [
                (
                    scenario_key,
                    title,
                    results_root
                    / run_dir_name(scenario_key, selected_routing_mode, "forming_new_connections")
                    / "metrics.json",
                    results_root
                    / run_dir_name(scenario_key, selected_routing_mode, "static_initial_connections")
                    / "metrics.json",
                )
                for scenario_key, title in scenario_titles
            ]
        else:
            raise SystemExit(f"Unsupported comparison_axis={{comparison_axis}}")

        metric_specs = [
            ("recall", "Recall"),
            ("avg_cycles", "Average cycles"),
            ("skip_rate", "Skip rate"),
        ]

        def load_status(run_dir: Path) -> str:
            final_path = run_dir / "metrics.json"
            partial_path = run_dir / "metrics_partial.json"
            query_log_path = run_dir / "query_log.jsonl"
            window_log_path = run_dir / "metrics_windows.jsonl"
            if final_path.exists():
                return "final"
            if partial_path.exists():
                return "partial"
            if window_log_path.exists():
                return "streaming_windows"
            if query_log_path.exists():
                return "streaming"
            raise SystemExit(
                f"Missing metrics.json, metrics_partial.json, metrics_windows.jsonl, "
                f"and query_log.jsonl in {{run_dir}}"
            )

        def build_windows(run_dir: Path, window_size: int) -> list[dict[str, float | int]]:
            if window_size <= 0:
                raise SystemExit("window_size must be positive")

            query_log_path = run_dir / "query_log.jsonl"
            if not query_log_path.exists():
                window_log_path = run_dir / "metrics_windows.jsonl"
                metrics_path = run_dir / "metrics.json"
                windows = []
                if window_log_path.exists():
                    with window_log_path.open("r", encoding="utf-8") as handle:
                        for raw_line in handle:
                            line = raw_line.strip()
                            if line:
                                windows.append(json.loads(line))
                elif metrics_path.exists():
                    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
                    raw_windows = payload.get("windows", [])
                    if isinstance(raw_windows, list):
                        windows = [window for window in raw_windows if isinstance(window, dict)]
                if not windows:
                    raise SystemExit(f"Missing rolling metrics windows in {{run_dir}}")
                return windows

            windows: list[dict[str, float | int]] = []
            window_executed = 0
            window_skipped = 0
            window_cycles = 0
            window_successes = 0
            previous_event_index = -1

            with query_log_path.open("r", encoding="utf-8") as handle:
                for raw_line in handle:
                    line = raw_line.strip()
                    if not line:
                        continue
                    event = json.loads(line)
                    event_index = int(event["event_index"])
                    skipped_since_last = event_index - previous_event_index - 1
                    if skipped_since_last < 0:
                        raise SystemExit(
                            f"Non-monotone event_index in {{query_log_path}}: "
                            f"{{event_index}} after {{previous_event_index}}"
                        )

                    previous_event_index = event_index
                    window_skipped += skipped_since_last
                    window_executed += 1
                    window_cycles += int(event["cycles_used"])
                    window_successes += int(event["result"] == "+")

                    if window_executed >= window_size:
                        skip_total = window_executed + window_skipped
                        windows.append(
                            {{
                                "window_index": len(windows),
                                "recall": window_successes / window_executed,
                                "avg_cycles": window_cycles / window_executed,
                                "skip_rate": window_skipped / skip_total if skip_total else 0.0,
                            }}
                        )
                        window_executed = 0
                        window_skipped = 0
                        window_cycles = 0
                        window_successes = 0

            if window_executed or window_skipped:
                skip_total = window_executed + window_skipped
                windows.append(
                    {{
                        "window_index": len(windows),
                        "recall": window_successes / window_executed if window_executed else 0.0,
                        "avg_cycles": window_cycles / window_executed if window_executed else 0.0,
                        "skip_rate": window_skipped / skip_total if skip_total else 0.0,
                    }}
                )

            if not windows:
                raise SystemExit(f"No executed queries found in {{query_log_path}}")

            return windows

        for scenario_key, title, semantic_path, replica_path in scenario_specs:
            semantic_status = load_status(semantic_path.parent)
            replica_status = load_status(replica_path.parent)

            semantic_windows = build_windows(semantic_path.parent, {window_size})
            replica_windows = build_windows(replica_path.parent, {window_size})

            x_semantic = [window["window_index"] for window in semantic_windows]
            x_replica = [window["window_index"] for window in replica_windows]

            fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
            suffix = ""
            if semantic_status != "final" or replica_status != "final":
                suffix = " (partial metrics present)"
            fig.suptitle(
                f"{{title}}: {{left_label}} vs {{right_label}} "
                f"(window={window_size}){{suffix}}"
            )

            for axis, (metric_key, metric_label) in zip(axes, metric_specs):
                axis.plot(
                    x_semantic,
                    [window[metric_key] for window in semantic_windows],
                    label=left_label,
                    linewidth=2,
                    alpha=0.5,
                )
                axis.plot(
                    x_replica,
                    [window[metric_key] for window in replica_windows],
                    label=right_label,
                    linewidth=2,
                    alpha=0.5,
                )
                axis.set_ylabel(metric_label)
                axis.grid(True, alpha=0.3)
                axis.legend()
                if x_min is not None or x_max is not None:
                    axis.set_xlim(left=x_min, right=x_max)

            axes[-1].set_xlabel("Window index")
            fig.tight_layout()
            fig.savefig(plot_dir / f"{{scenario_key}}_{{filename_suffix}}.png", dpi=150)
            plt.close(fig)
        """
    )


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Generate three remote comparison plots and open them locally."
    )
    parser.add_argument("--host", default=DEFAULT_REMOTE_HOST, help="SSH host for DAS login node")
    parser.add_argument("--remote-root", default=DEFAULT_REMOTE_ROOT, help="Remote repo root")
    parser.add_argument(
        "--remote-scratch",
        default=DEFAULT_REMOTE_SCRATCH,
        help="Remote scratch root containing results and logs",
    )
    parser.add_argument(
        "--local-output-dir",
        default=str(repo_root / "results" / "remote_comparison_plots"),
        help="Local directory where plots are copied",
    )
    parser.add_argument(
        "--no-open",
        action="store_true",
        help="Generate and copy plots without opening them locally",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=1000,
        help="Executed queries to average into one plotted data point",
    )
    parser.add_argument(
        "--x-min",
        type=int,
        help="Optional lower bound for the plotted window index range",
    )
    parser.add_argument(
        "--x-max",
        type=int,
        help="Optional upper bound for the plotted window index range",
    )
    parser.add_argument(
        "--comparison-axis",
        choices=("routing_mode", "topology_mode"),
        default="routing_mode",
        help="Which experiment axis to compare across the three scenarios",
    )
    parser.add_argument(
        "--routing-mode",
        choices=("semantic_query_memory", "replica_only"),
        default="semantic_query_memory",
        help="Routing mode to hold fixed when --comparison-axis=topology_mode",
    )
    parser.add_argument(
        "--topology-mode",
        choices=("forming_new_connections", "static_initial_connections"),
        default="forming_new_connections",
        help="Topology mode to hold fixed when --comparison-axis=routing_mode",
    )
    parser.add_argument(
        "--scenario",
        choices=("empty_start_chronological", "seeded_fixed_inventory", "seeded_randomized_timeline"),
        help="Optional scenario to plot instead of generating all three scenario plots",
    )
    args = parser.parse_args()
    if args.x_min is not None and args.x_max is not None and args.x_min > args.x_max:
        raise SystemExit("--x-min cannot be greater than --x-max")

    local_output_dir = Path(args.local_output_dir)
    local_output_dir.mkdir(parents=True, exist_ok=True)

    remote_python = f"{args.remote_root}/../.venvs/low-coord-dht/bin/python"
    remote_results_root = f"{args.remote_scratch}/results"
    remote_plot_dir = f"{args.remote_scratch}/comparison_plots"
    remote_script = _remote_plot_script(
        remote_results_root,
        remote_plot_dir,
        args.window_size,
        args.x_min,
        args.x_max,
        args.comparison_axis,
        args.routing_mode,
        args.topology_mode,
        args.scenario,
    )

    _run(
        [
            "ssh",
            args.host,
            (
                f"cd {args.remote_root} && "
                f"{remote_python} - <<'PY'\n{remote_script}\nPY"
            ),
        ]
    )

    scenario_keys = (
        [args.scenario]
        if args.scenario is not None
        else [
            "empty_start_chronological",
            "seeded_fixed_inventory",
            "seeded_randomized_timeline",
        ]
    )
    plot_names = [
        f"{scenario_key}_{args.comparison_axis}_comparison.png" for scenario_key in scenario_keys
    ]
    remote_paths = [f"{args.host}:{remote_plot_dir}/{plot_name}" for plot_name in plot_names]
    _run(["scp", *remote_paths, str(local_output_dir)])

    if not args.no_open:
        for plot_name in plot_names:
            _run(["open", str(local_output_dir / plot_name)])

    print(f"Copied plots to {local_output_dir}")


if __name__ == "__main__":
    main()

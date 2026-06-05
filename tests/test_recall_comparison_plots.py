from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.plot_recall_comparison_suite import (  # noqa: E402
    PlotResult,
    RecallWindow,
    discover_runs,
    drift_marker_positions,
    generate_primary_plots,
    load_recall_windows,
    write_plot_manifest,
)


def _write_query_log(path: Path, results: list[str]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for event_index, result in enumerate(results):
            payload = {
                "event_index": event_index,
                "query_id": event_index,
                "querier_peer_id": 1,
                "clicked_doc_id": f"doc-{event_index}",
                "cycles_used": 1,
                "result": result,
                "termination_reason": "found" if result == "+" else "exhausted",
            }
            handle.write(json.dumps(payload) + "\n")


def _write_manifest(run_dir: Path, payload: dict) -> None:
    run_dir.mkdir(parents=True)
    (run_dir / "run_manifest.json").write_text(json.dumps(payload), encoding="utf-8")
    _write_query_log(run_dir / "query_log.jsonl", ["+", "-"])


class RecallWindowTests(unittest.TestCase):
    def test_load_recall_windows_replays_query_log_at_requested_size(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            query_log = Path(tmp) / "query_log.jsonl"
            _write_query_log(query_log, ["+", "-", "+", "+", "-"])

            windows = load_recall_windows(query_log, window_size=2)

        self.assertEqual([window.executed_queries for window in windows], [2, 2, 1])
        self.assertEqual([window.successes for window in windows], [1, 2, 0])
        self.assertEqual([window.recall for window in windows], [0.5, 1.0, 0.0])


class RunDiscoveryTests(unittest.TestCase):
    def test_discover_runs_classifies_algorithms_and_fallbacks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_manifest(
                root / "seeded_randomized_timeline_forming_new_connections_run",
                {
                    "dataset": "aol4foltr",
                    "scenario": "seeded_randomized_timeline",
                    "routing_mode": "semantic_query_memory",
                    "topology_mode": "forming_new_connections",
                    "summary": {"overall_recall": 0.4, "executed_queries": 2},
                },
            )
            _write_manifest(
                root / "lsh_seeded_randomized_run",
                {
                    "algorithm": "e2lsh_dht",
                    "dataset": "aol4foltr",
                    "scenario": "seeded_randomized_timeline",
                    "summary": {"overall_recall": 0.5, "executed_queries": 2},
                },
            )
            _write_manifest(
                root / "semantica_aol_seeded_randomized_timeline_run",
                {
                    "algorithm": "semantica_chain",
                    "dataset": "aol4foltr",
                    "scenario": "seeded_randomized_timeline",
                    "summary": {"overall_recall": 0.6, "executed_queries": 2},
                },
            )
            _write_manifest(
                root / "seeded_randomized_timeline_random_walk_run",
                {
                    "dataset": "aol4foltr",
                    "scenario": "seeded_randomized_timeline",
                    "selection_mode": "random_walk",
                    "summary": {"overall_recall": 0.3, "executed_queries": 2},
                },
            )

            records = discover_runs(root)

        self.assertEqual(
            sorted(record.method for record in records),
            ["lsh", "random_walk", "semantica", "zerorag"],
        )
        zerorag = next(record for record in records if record.method == "zerorag")
        self.assertEqual(zerorag.selection_mode, "similarity")
        self.assertEqual(zerorag.search_mode, "best_first")

    def test_primary_plot_reports_missing_runs_without_creating_empty_plot(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output_dir = root / "plots"

            plots = generate_primary_plots(
                [],
                output_dir=output_dir,
                datasets=["synthetic"],
                scenarios=["fifo_randomized_timeline"],
                window_size=100,
            )

        self.assertEqual(len(plots), 1)
        self.assertIsNone(plots[0].path)
        self.assertIn("Semantica omitted", " ".join(plots[0].notes))
        self.assertTrue(any("missing synthetic/fifo_randomized_timeline" in note for note in plots[0].notes))


class ConceptDriftPlotTests(unittest.TestCase):
    def test_drift_marker_positions_use_window_index_domain(self) -> None:
        windows = [
            RecallWindow(
                window_index=index,
                event_start=index * 100,
                event_end=index * 100 + 99,
                executed_queries=100,
                successes=50,
                recall=0.5,
            )
            for index in range(11)
        ]

        markers = drift_marker_positions(
            windows,
            start_fraction=0.4,
            end_fraction=0.6,
        )

        self.assertEqual(markers, [(4.0, "concept appears"), (6.0, "concept disappears")])


class PlotManifestTests(unittest.TestCase):
    def test_write_plot_manifest_records_window_and_missing_notes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_path = write_plot_manifest(
                output_dir=root / "plots",
                results_root=root / "results",
                window_size=100,
                records=[],
                plots=[
                    PlotResult(
                        kind="primary",
                        name="primary_synthetic_fifo_randomized_timeline",
                        path=None,
                        records=[],
                        notes=["Semantica omitted: current runner is fixed-inventory only."],
                    )
                ],
            )
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))

        self.assertEqual(payload["window_size"], 100)
        self.assertEqual(payload["generated_plot_count"], 0)
        self.assertEqual(
            payload["plots"][0]["notes"],
            ["Semantica omitted: current runner is fixed-inventory only."],
        )


if __name__ == "__main__":
    unittest.main()

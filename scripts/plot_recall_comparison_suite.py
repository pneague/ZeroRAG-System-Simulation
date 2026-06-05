#!/usr/bin/env python3
"""Plot rolling-recall comparison suites from existing simulation outputs."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable, TextIO


DEFAULT_DATASETS = ("aol4foltr", "synthetic")
DEFAULT_SCENARIOS = ("seeded_randomized_timeline", "fifo_randomized_timeline")
DEFAULT_WINDOW_SIZE = 100
SEMANTICA_FIFO_NOTE = "Semantica omitted: current runner is fixed-inventory only."


@dataclass(frozen=True, slots=True)
class RecallWindow:
    window_index: int
    event_start: int
    event_end: int
    executed_queries: int
    successes: int
    recall: float


@dataclass(frozen=True, slots=True)
class RunRecord:
    run_dir: Path
    manifest_path: Path
    query_log_path: Path
    manifest: dict
    dataset: str
    scenario: str
    method: str
    routing_mode: str
    topology_mode: str
    selection_mode: str
    search_mode: str
    fanout: int
    max_neighbors: int
    max_owned_docs: int
    max_routed_queries: int
    churn_mode: str
    overall_recall: float
    executed_queries: int
    label: str


@dataclass(frozen=True, slots=True)
class PlotResult:
    kind: str
    name: str
    path: str | None
    records: list[str]
    notes: list[str]


def _json_load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _nested_get(payload: dict, keys: Iterable[str], default: object = None) -> object:
    current: object = payload
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def _as_int(value: object, default: int = 0) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_float(value: object, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _infer_dataset(manifest: dict, run_dir: Path) -> str:
    dataset = manifest.get("dataset")
    if isinstance(dataset, str) and dataset:
        return dataset
    haystack = " ".join(
        str(value)
        for value in (
            run_dir.name,
            manifest.get("metadata"),
            manifest.get("embedding_manifest"),
            manifest.get("output_dir"),
        )
        if value is not None
    ).lower()
    if "synthetic" in haystack:
        return "synthetic"
    if "amazon" in haystack:
        return "amazon"
    if "mind" in haystack:
        return "mind"
    return "aol4foltr"


def _infer_scenario(manifest: dict, run_dir: Path) -> str:
    scenario = manifest.get("scenario") or manifest.get("requested_scenario")
    if isinstance(scenario, str) and scenario:
        return scenario
    name = run_dir.name
    for candidate in (
        "seeded_randomized_timeline",
        "fifo_randomized_timeline",
        "empty_start_randomized_timeline",
        "fifo_chronological",
        "empty_start_chronological",
        "seeded_fixed_inventory",
    ):
        if candidate in name:
            return candidate
    return "unknown"


def _method_from_manifest(manifest: dict, run_dir: Path) -> str:
    algorithm = manifest.get("algorithm")
    if algorithm == "e2lsh_dht" or "lsh_" in run_dir.name:
        return "lsh"
    if algorithm == "semantica_chain" or "semantica" in run_dir.name:
        return "semantica"
    selection_mode = (
        manifest.get("selection_mode")
        or _nested_get(manifest, ("config", "selection_mode"))
        or ("random_walk" if "random_walk" in run_dir.name else "similarity")
    )
    if selection_mode == "random_walk":
        return "random_walk"
    if algorithm == "ZeroRAG" or algorithm == "zerorag" or "zerorag" in run_dir.name:
        return "zerorag"
    return "zerorag"


def _default_label(record: RunRecord) -> str:
    if record.method == "zerorag":
        if record.routing_mode == "replica_only":
            return "Replica-only"
        if record.topology_mode == "static_initial_connections":
            return "Static topology"
        if record.search_mode == "fanout":
            return f"ZeroRAG fanout={record.fanout}"
        return "ZeroRAG"
    if record.method == "lsh":
        lsh_config = record.manifest.get("lsh_config") or record.manifest.get("config") or {}
        bits = []
        for key in ("w", "k", "L"):
            if key in lsh_config:
                bits.append(f"{key}={lsh_config[key]}")
        return f"LSH ({', '.join(bits)})" if bits else "LSH"
    if record.method == "semantica":
        return "Semantica"
    if record.method == "random_walk":
        return "Random Walk"
    return record.method


def _record_from_manifest(manifest_path: Path) -> RunRecord:
    manifest = _json_load(manifest_path)
    run_dir = manifest_path.parent
    dataset = _infer_dataset(manifest, run_dir)
    scenario = _infer_scenario(manifest, run_dir)
    method = _method_from_manifest(manifest, run_dir)
    config = manifest.get("config") if isinstance(manifest.get("config"), dict) else {}
    churn = manifest.get("churn") if isinstance(manifest.get("churn"), dict) else {}

    routing_mode = str(manifest.get("routing_mode") or config.get("routing_mode") or "semantic_query_memory")
    topology_mode = str(
        manifest.get("topology_mode") or config.get("topology_mode") or "forming_new_connections"
    )
    selection_mode = str(
        manifest.get("selection_mode")
        or config.get("selection_mode")
        or ("random_walk" if method == "random_walk" else "similarity")
    )
    search_mode = str(manifest.get("search_mode") or config.get("search_mode") or "best_first")
    fanout = _as_int(manifest.get("fanout") or config.get("fanout"), 1)
    max_neighbors = _as_int(config.get("max_neighbors"), 0)
    max_owned_docs = _as_int(manifest.get("max_owned_docs") or config.get("max_owned_docs"), 0)
    max_routed_queries = _as_int(config.get("max_routed_queries"), 0)
    churn_mode = str(churn.get("mode") or config.get("churn_mode") or "none")
    summary = manifest.get("summary") if isinstance(manifest.get("summary"), dict) else {}

    record = RunRecord(
        run_dir=run_dir,
        manifest_path=manifest_path,
        query_log_path=run_dir / "query_log.jsonl",
        manifest=manifest,
        dataset=dataset,
        scenario=scenario,
        method=method,
        routing_mode=routing_mode,
        topology_mode=topology_mode,
        selection_mode=selection_mode,
        search_mode=search_mode,
        fanout=fanout,
        max_neighbors=max_neighbors,
        max_owned_docs=max_owned_docs,
        max_routed_queries=max_routed_queries,
        churn_mode=churn_mode,
        overall_recall=_as_float(summary.get("overall_recall"), 0.0),
        executed_queries=_as_int(summary.get("executed_queries"), 0),
        label="",
    )
    return RunRecord(**{**asdict(record), "label": _default_label(record)})


def discover_runs(results_root: Path, *, include_smoke: bool = False) -> list[RunRecord]:
    records: list[RunRecord] = []
    for manifest_path in sorted(results_root.rglob("run_manifest.json")):
        if not include_smoke and "smoke" in str(manifest_path.parent):
            continue
        try:
            record = _record_from_manifest(manifest_path)
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            continue
        records.append(record)
    return records


def load_recall_windows(query_log_path: Path, *, window_size: int) -> list[RecallWindow]:
    if window_size <= 0:
        raise ValueError("window_size must be positive")
    windows: list[RecallWindow] = []
    window_successes = 0
    window_executed = 0
    window_start_event: int | None = None
    window_end_event = -1

    with query_log_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            event = json.loads(line)
            event_index = int(event["event_index"])
            if window_start_event is None:
                window_start_event = event_index
            window_end_event = event_index
            window_executed += 1
            window_successes += int(event.get("result") == "+")

            if window_executed >= window_size:
                windows.append(
                    RecallWindow(
                        window_index=len(windows),
                        event_start=window_start_event,
                        event_end=window_end_event,
                        executed_queries=window_executed,
                        successes=window_successes,
                        recall=window_successes / window_executed,
                    )
                )
                window_successes = 0
                window_executed = 0
                window_start_event = None

    if window_executed:
        windows.append(
            RecallWindow(
                window_index=len(windows),
                event_start=window_start_event if window_start_event is not None else -1,
                event_end=window_end_event,
                executed_queries=window_executed,
                successes=window_successes,
                recall=window_successes / window_executed,
            )
        )
    if not windows:
        raise ValueError(f"No executed queries found in {query_log_path}")
    return windows


def _window_payload_to_recall_window(payload: dict, fallback_index: int) -> RecallWindow:
    executed = _as_int(payload.get("executed_queries"), 0)
    recall = _as_float(payload.get("recall"), 0.0)
    successes = _as_int(payload.get("successes"), int(round(recall * executed)))
    return RecallWindow(
        window_index=_as_int(payload.get("window_index"), fallback_index),
        event_start=_as_int(payload.get("event_start"), -1),
        event_end=_as_int(payload.get("event_end"), -1),
        executed_queries=executed,
        successes=successes,
        recall=recall,
    )


def load_recall_windows_from_compact_metrics(run_dir: Path) -> list[RecallWindow]:
    window_log_path = run_dir / "metrics_windows.jsonl"
    windows: list[RecallWindow] = []
    if window_log_path.exists():
        with window_log_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                windows.append(_window_payload_to_recall_window(json.loads(line), len(windows)))
    else:
        metrics_path = run_dir / "metrics.json"
        if metrics_path.exists():
            payload = _json_load(metrics_path)
            raw_windows = payload.get("windows", [])
            if isinstance(raw_windows, list):
                for raw_window in raw_windows:
                    if isinstance(raw_window, dict):
                        windows.append(
                            _window_payload_to_recall_window(raw_window, len(windows))
                        )
    if not windows:
        raise ValueError(f"No compact metrics windows found in {run_dir}")
    return windows


def _open_text(path: Path) -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(path, "rt", newline="", encoding="utf-8")
    return path.open("r", newline="", encoding="utf-8")


def load_topic_flags(path: Path, *, topic_id: int | None = None) -> dict[int, bool]:
    flags: dict[int, bool] = {}
    with _open_text(path) as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        if "qid" not in fieldnames:
            return flags
        for row in reader:
            qid = int(row["qid"])
            if topic_id is not None and "generated_topic" in fieldnames:
                flags[qid] = int(row["generated_topic"]) == topic_id
            else:
                flags[qid] = str(row.get("is_drift_topic", "")).strip().lower() in {
                    "1",
                    "true",
                    "yes",
                    "y",
                }
    return flags


def load_filtered_recall_windows(
    query_log_path: Path,
    *,
    window_size: int,
    include_query: Callable[[int], bool],
) -> list[RecallWindow]:
    windows: list[RecallWindow] = []
    window_successes = 0
    window_executed = 0
    window_start_event: int | None = None
    window_end_event = -1

    with query_log_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            event = json.loads(line)
            if not include_query(int(event["query_id"])):
                continue
            event_index = int(event["event_index"])
            if window_start_event is None:
                window_start_event = event_index
            window_end_event = event_index
            window_executed += 1
            window_successes += int(event.get("result") == "+")
            if window_executed >= window_size:
                windows.append(
                    RecallWindow(
                        window_index=len(windows),
                        event_start=window_start_event,
                        event_end=window_end_event,
                        executed_queries=window_executed,
                        successes=window_successes,
                        recall=window_successes / window_executed,
                    )
                )
                window_successes = 0
                window_executed = 0
                window_start_event = None

    if window_executed:
        windows.append(
            RecallWindow(
                window_index=len(windows),
                event_start=window_start_event if window_start_event is not None else -1,
                event_end=window_end_event,
                executed_queries=window_executed,
                successes=window_successes,
                recall=window_successes / window_executed,
            )
        )
    return windows


def _has_query_log(record: RunRecord) -> bool:
    return (
        record.query_log_path.exists()
        or (record.run_dir / "metrics_windows.jsonl").exists()
        or (record.run_dir / "metrics.json").exists()
    )


def _select_best(records: Iterable[RunRecord]) -> RunRecord | None:
    candidates = [record for record in records if _has_query_log(record)]
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda record: (
            record.overall_recall,
            record.executed_queries,
            str(record.run_dir),
        ),
    )


def _zerorag_records(records: Iterable[RunRecord]) -> list[RunRecord]:
    return [
        record
        for record in records
        if record.method == "zerorag"
        and record.routing_mode == "semantic_query_memory"
        and record.selection_mode == "similarity"
        and record.search_mode == "best_first"
    ]


def _plot_curves(
    series: list[tuple[str, RunRecord, list[RecallWindow]]],
    *,
    output_path: Path,
    title: str,
    window_size: int,
    notes: list[str] | None = None,
    vertical_markers: list[tuple[float, str]] | None = None,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 6))
    for label, _record, windows in series:
        ax.plot(
            [window.window_index for window in windows],
            [window.recall for window in windows],
            label=label,
            linewidth=1.8,
            alpha=0.85,
        )
    if vertical_markers:
        for x_value, label in vertical_markers:
            ax.axvline(x=x_value, color="tab:red", linestyle=":", linewidth=1.5)
            ax.text(
                x_value,
                0.02,
                label,
                rotation=90,
                va="bottom",
                ha="right",
                transform=ax.get_xaxis_transform(),
            )
    ax.set_title(title)
    ax.set_xlabel("Window index")
    ax.set_ylabel(f"Recall rate, rolling window = {window_size} executed queries")
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend()
    if notes:
        fig.text(0.01, 0.01, " | ".join(notes), fontsize=8)
    fig.tight_layout(rect=(0, 0.04 if notes else 0, 1, 1))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _windows_for_record(record: RunRecord, window_size: int) -> list[RecallWindow]:
    if not record.query_log_path.exists():
        return load_recall_windows_from_compact_metrics(record.run_dir)
    return load_recall_windows(record.query_log_path, window_size=window_size)


def _safe_windows(record: RunRecord, window_size: int, missing: list[str]) -> list[RecallWindow] | None:
    try:
        return _windows_for_record(record, window_size)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        missing.append(f"{record.run_dir}: {exc}")
        return None


def _find_primary_record(
    records: list[RunRecord],
    *,
    dataset: str,
    scenario: str,
    method: str,
) -> RunRecord | None:
    pool = [
        record
        for record in records
        if record.dataset == dataset and record.scenario == scenario and record.method == method
    ]
    if method == "zerorag":
        pool = [
            record
            for record in pool
            if record.routing_mode == "semantic_query_memory"
            and record.topology_mode == "forming_new_connections"
            and record.selection_mode == "similarity"
            and record.search_mode == "best_first"
        ]
    elif method == "random_walk":
        pool = [record for record in pool if record.selection_mode == "random_walk"]
    return _select_best(pool)


def generate_primary_plots(
    records: list[RunRecord],
    *,
    output_dir: Path,
    datasets: Iterable[str],
    scenarios: Iterable[str],
    window_size: int,
) -> list[PlotResult]:
    results: list[PlotResult] = []
    for dataset in datasets:
        for scenario in scenarios:
            requested_methods = ["zerorag", "lsh", "random_walk"]
            notes: list[str] = []
            if scenario == "seeded_randomized_timeline":
                requested_methods.append("semantica")
            else:
                notes.append(SEMANTICA_FIFO_NOTE)

            series: list[tuple[str, RunRecord, list[RecallWindow]]] = []
            missing: list[str] = []
            for method in requested_methods:
                record = _find_primary_record(
                    records,
                    dataset=dataset,
                    scenario=scenario,
                    method=method,
                )
                if record is None:
                    missing.append(f"missing {dataset}/{scenario}/{method}")
                    continue
                windows = _safe_windows(record, window_size, missing)
                if windows is not None:
                    series.append((record.label, record, windows))

            name = f"primary_{dataset}_{scenario}"
            if series:
                output_path = output_dir / f"{name}.png"
                _plot_curves(
                    series,
                    output_path=output_path,
                    title=f"{dataset}: {scenario} recall comparison",
                    window_size=window_size,
                    notes=notes,
                )
                path: str | None = str(output_path)
            else:
                path = None
            results.append(
                PlotResult(
                    kind="primary",
                    name=name,
                    path=path,
                    records=[str(record.run_dir) for _label, record, _windows in series],
                    notes=notes + missing,
                )
            )
    return results


def _comparison_plot(
    records: list[RunRecord],
    *,
    output_dir: Path,
    name: str,
    title: str,
    window_size: int,
    candidates: list[tuple[str, Callable[[RunRecord], bool]]],
) -> PlotResult:
    series: list[tuple[str, RunRecord, list[RecallWindow]]] = []
    missing: list[str] = []
    for label, predicate in candidates:
        record = _select_best(record for record in records if predicate(record))
        if record is None:
            missing.append(f"missing {label}")
            continue
        windows = _safe_windows(record, window_size, missing)
        if windows is not None:
            series.append((label, record, windows))

    path: str | None = None
    if series:
        output_path = output_dir / f"{name}.png"
        _plot_curves(
            series,
            output_path=output_path,
            title=title,
            window_size=window_size,
        )
        path = str(output_path)
    return PlotResult(
        kind="ablation",
        name=name,
        path=path,
        records=[str(record.run_dir) for _label, record, _windows in series],
        notes=missing,
    )


def _parse_int_after(pattern: str, text: str) -> int | None:
    match = re.search(pattern, text)
    if not match:
        return None
    return int(match.group(1))


def _ablation_value(record: RunRecord, ablation: str) -> int | str | None:
    name = record.run_dir.name
    if ablation == "peer_count":
        return _parse_int_after(r"users(\d+)", name) or _as_int(
            _nested_get(record.manifest, ("seeded_stats", "seeded_peer_count")),
            0,
        )
    if ablation == "corpus_size":
        return _parse_int_after(r"docs(\d+)", name)
    if ablation == "graph_degree":
        return _parse_int_after(r"degree(\d+)", name) or record.max_neighbors
    if ablation == "churn":
        if record.churn_mode != "none":
            return record.churn_mode
        if "replacement_churn" in name:
            return "replace"
        if "churn" in name:
            return "drop"
    if ablation == "local_document_capacity":
        return record.max_owned_docs or _parse_int_after(r"owned(\d+)", name)
    if ablation == "routed_query_memory_capacity":
        return record.max_routed_queries or _parse_int_after(r"query_memory_limit_(\d+)", name)
    if ablation == "fanout":
        if record.search_mode == "fanout" or "fanout" in name:
            return f"fanout={record.fanout}, cycles={_nested_get(record.manifest, ('config', 'max_cycles'), '')}"
    return None


def _ablation_records(records: list[RunRecord], ablation: str) -> list[tuple[str, RunRecord]]:
    selected: list[tuple[str, RunRecord]] = []
    if ablation == "fanout":
        pool = [
            record
            for record in records
            if record.method == "zerorag"
            and record.routing_mode == "semantic_query_memory"
            and record.selection_mode == "similarity"
        ]
    else:
        pool = _zerorag_records(records)
    for record in pool:
        value = _ablation_value(record, ablation)
        if not value:
            continue
        if ablation in {"peer_count", "corpus_size"} and "synthetic_ablation" not in record.run_dir.name:
            continue
        if ablation == "graph_degree" and "degree" not in record.run_dir.name:
            continue
        if ablation == "local_document_capacity" and "fifo_limit" not in record.run_dir.name:
            continue
        if ablation == "routed_query_memory_capacity" and "query_memory_limit" not in record.run_dir.name:
            continue
        if ablation == "fanout" and "fanout" not in record.run_dir.name:
            continue
        selected.append((str(value), record))
    return sorted(selected, key=lambda item: (_sort_key(item[0]), str(item[1].run_dir)))


def _sort_key(value: str) -> tuple[int, object]:
    try:
        return (0, int(re.sub(r"\D", "", value)))
    except ValueError:
        return (1, value)


def _plot_ablation_group(
    records: list[RunRecord],
    *,
    output_dir: Path,
    ablation: str,
    title: str,
    window_size: int,
) -> PlotResult:
    series: list[tuple[str, RunRecord, list[RecallWindow]]] = []
    missing: list[str] = []
    for label, record in _ablation_records(records, ablation):
        windows = _safe_windows(record, window_size, missing)
        if windows is not None:
            series.append((label, record, windows))
    path: str | None = None
    if series:
        output_path = output_dir / f"ablation_{ablation}.png"
        _plot_curves(
            series,
            output_path=output_path,
            title=title,
            window_size=window_size,
        )
        path = str(output_path)
    else:
        missing.append(f"missing ablation runs for {ablation}")
    return PlotResult(
        kind="ablation",
        name=f"ablation_{ablation}",
        path=path,
        records=[str(record.run_dir) for _label, record, _windows in series],
        notes=missing,
    )


def drift_marker_positions(
    windows: list[RecallWindow],
    *,
    start_fraction: float,
    end_fraction: float,
) -> list[tuple[float, str]]:
    if not windows:
        return []
    max_window = max(window.window_index for window in windows)
    return [
        (max_window * start_fraction, "concept appears"),
        (max_window * end_fraction, "concept disappears"),
    ]


def _load_drift_setting(record: RunRecord) -> tuple[float, float]:
    metadata = record.manifest.get("metadata")
    if isinstance(metadata, str):
        data_dir = Path(metadata).parent
        setting_path = data_dir / "synthetic_ablation_setting.json"
        if setting_path.exists():
            try:
                setting_payload = _json_load(setting_path)
                setting = setting_payload.get("setting", {})
                return (
                    _as_float(setting.get("drift_start_fraction"), 0.40),
                    _as_float(setting.get("drift_end_fraction"), 0.60),
                )
            except (OSError, json.JSONDecodeError):
                pass
    return (0.40, 0.60)


def _plot_concept_drift(
    records: list[RunRecord],
    *,
    output_dir: Path,
    window_size: int,
) -> PlotResult:
    candidates = [
        record
        for record in _zerorag_records(records)
        if "concept_drift" in record.run_dir.name or record.manifest.get("topic_metadata")
    ]
    record = _select_best(candidates)
    if record is None:
        return PlotResult(
            kind="ablation",
            name="ablation_concept_drift",
            path=None,
            records=[],
            notes=["missing concept drift run"],
        )

    notes: list[str] = []
    windows = _safe_windows(record, window_size, notes)
    if windows is None:
        return PlotResult(
            kind="ablation",
            name="ablation_concept_drift",
            path=None,
            records=[str(record.run_dir)],
            notes=notes,
        )

    series: list[tuple[str, RunRecord, list[RecallWindow]]] = [("All queries", record, windows)]
    topic_metadata = record.manifest.get("topic_metadata")
    if isinstance(topic_metadata, str):
        topic_path = Path(topic_metadata)
        if topic_path.exists():
            flags = load_topic_flags(
                topic_path,
                topic_id=(
                    _as_int(record.manifest.get("topic_metrics_topic_id"))
                    if record.manifest.get("topic_metrics_topic_id") is not None
                    else None
                ),
            )
            if flags and record.query_log_path.exists():
                focus_windows = load_filtered_recall_windows(
                    record.query_log_path,
                    window_size=window_size,
                    include_query=lambda qid: flags.get(qid, False),
                )
                if focus_windows:
                    series.append(("Drift topic queries", record, focus_windows))
    start_fraction, end_fraction = _load_drift_setting(record)
    markers = drift_marker_positions(
        windows,
        start_fraction=start_fraction,
        end_fraction=end_fraction,
    )
    output_path = output_dir / "ablation_concept_drift.png"
    _plot_curves(
        series,
        output_path=output_path,
        title="Synthetic concept drift recall",
        window_size=window_size,
        vertical_markers=markers,
    )
    return PlotResult(
        kind="ablation",
        name="ablation_concept_drift",
        path=str(output_path),
        records=[str(record.run_dir)],
        notes=[f"drift markers at fractions {start_fraction:.2f}, {end_fraction:.2f}"],
    )


def generate_ablation_plots(
    records: list[RunRecord],
    *,
    output_dir: Path,
    window_size: int,
) -> list[PlotResult]:
    results: list[PlotResult] = []
    for dataset in DEFAULT_DATASETS:
        for scenario in DEFAULT_SCENARIOS:
            base_predicate = (
                lambda record, dataset=dataset, scenario=scenario: record.dataset == dataset
                and record.scenario == scenario
                and record.method == "zerorag"
                and record.routing_mode == "semantic_query_memory"
                and record.topology_mode == "forming_new_connections"
                and record.selection_mode == "similarity"
                and record.search_mode == "best_first"
            )
            replica_predicate = (
                lambda record, dataset=dataset, scenario=scenario: record.dataset == dataset
                and record.scenario == scenario
                and record.method == "zerorag"
                and record.routing_mode == "replica_only"
                and record.topology_mode == "forming_new_connections"
                and record.selection_mode == "similarity"
                and record.search_mode == "best_first"
            )
            static_predicate = (
                lambda record, dataset=dataset, scenario=scenario: record.dataset == dataset
                and record.scenario == scenario
                and record.method == "zerorag"
                and record.routing_mode == "semantic_query_memory"
                and record.topology_mode == "static_initial_connections"
                and record.selection_mode == "similarity"
                and record.search_mode == "best_first"
            )
            results.append(
                _comparison_plot(
                    records,
                    output_dir=output_dir,
                    name=f"ablation_{dataset}_{scenario}_replica_only",
                    title=f"{dataset}: {scenario} ZeroRAG vs replica-only",
                    window_size=window_size,
                    candidates=[
                        ("ZeroRAG", base_predicate),
                        ("Replica-only", replica_predicate),
                    ],
                )
            )
            results.append(
                _comparison_plot(
                    records,
                    output_dir=output_dir,
                    name=f"ablation_{dataset}_{scenario}_static_topology",
                    title=f"{dataset}: {scenario} ZeroRAG vs static topology",
                    window_size=window_size,
                    candidates=[
                        ("Forming topology", base_predicate),
                        ("Static topology", static_predicate),
                    ],
                )
            )

    for ablation, title in (
        ("peer_count", "Synthetic peer-count recall sensitivity"),
        ("corpus_size", "Synthetic corpus-size recall sensitivity"),
        ("graph_degree", "Synthetic graph-degree recall sensitivity"),
        ("churn", "Synthetic churn recall sensitivity"),
        ("local_document_capacity", "Synthetic local document capacity recall sensitivity"),
        ("routed_query_memory_capacity", "Synthetic routed-query memory capacity recall sensitivity"),
        ("fanout", "Synthetic fanout recall sensitivity"),
    ):
        results.append(
            _plot_ablation_group(
                records,
                output_dir=output_dir,
                ablation=ablation,
                title=title,
                window_size=window_size,
            )
        )
    results.append(_plot_concept_drift(records, output_dir=output_dir, window_size=window_size))
    return results


def write_plot_manifest(
    *,
    output_dir: Path,
    results_root: Path,
    window_size: int,
    records: list[RunRecord],
    plots: list[PlotResult],
) -> Path:
    payload = {
        "results_root": str(results_root),
        "window_size": window_size,
        "discovered_run_count": len(records),
        "generated_plot_count": sum(1 for plot in plots if plot.path),
        "plots": [asdict(plot) for plot in plots],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "plot_manifest.json"
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return manifest_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate rolling-recall comparison graphs from existing DHT result dirs."
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("results"),
        help="Directory containing run subdirectories with run_manifest.json and query_log.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results") / "recall_comparison_suite",
        help="Directory for generated plots and plot_manifest.json",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=DEFAULT_WINDOW_SIZE,
        help="Executed queries per rolling recall point",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(DEFAULT_DATASETS),
        help="Datasets to include in primary comparison plots",
    )
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=list(DEFAULT_SCENARIOS),
        help="Scenarios to include in primary comparison plots",
    )
    parser.add_argument(
        "--include-smoke",
        action="store_true",
        help="Include result directories whose path contains 'smoke'",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    records = discover_runs(args.results_root, include_smoke=args.include_smoke)
    plots: list[PlotResult] = []
    plots.extend(
        generate_primary_plots(
            records,
            output_dir=args.output_dir,
            datasets=args.datasets,
            scenarios=args.scenarios,
            window_size=args.window_size,
        )
    )
    plots.extend(
        generate_ablation_plots(
            records,
            output_dir=args.output_dir,
            window_size=args.window_size,
        )
    )
    manifest_path = write_plot_manifest(
        output_dir=args.output_dir,
        results_root=args.results_root,
        window_size=args.window_size,
        records=records,
        plots=plots,
    )
    generated = sum(1 for plot in plots if plot.path)
    missing = sum(1 for plot in plots if not plot.path)
    print(
        json.dumps(
            {
                "plot_manifest": str(manifest_path),
                "discovered_run_count": len(records),
                "generated_plot_count": generated,
                "missing_plot_count": missing,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

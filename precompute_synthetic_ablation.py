#!/usr/bin/env python3
"""Precompute synthetic datasets for the planned ablation matrices."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

sys.path.insert(0, str(Path(__file__).resolve().parent))

from precompute_synthetic import precompute_synthetic_artifacts
from src.path_defaults import default_data_root
from src.synthetic_dataset import (
    DEFAULT_DIRICHLET_ALPHA,
    DEFAULT_DOCS_PER_TOPIC,
    DEFAULT_EMBEDDING_DIM,
    DEFAULT_MIN_TARGET_COUNT,
    DEFAULT_QUERIES_PER_USER,
    DEFAULT_RANDOM_SEED,
    DEFAULT_TOPIC_COUNT,
    DEFAULT_USER_COUNT,
)


DEFAULT_ABLATION_SIGMA = 0.05
DEFAULT_DATA_ROOT = default_data_root()
BASELINE_USERS = DEFAULT_USER_COUNT
ABLATION_QUERIES_PER_USER = DEFAULT_QUERIES_PER_USER
USER_SCALABILITY_USERS = (100, 500, 1_000, 5_000, 10_000, 50_000, 100_000)
DOC_SCALABILITY_DOCS_PER_TOPIC = (5, 10, 20, 50, 100, 200, 500)
DEGREE_SENSITIVITY_MAX_NEIGHBORS = (5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100)


@dataclass(frozen=True, slots=True)
class SyntheticAblationSetting:
    experiment: str
    label: str
    users: int = BASELINE_USERS
    topics: int = DEFAULT_TOPIC_COUNT
    docs_per_topic: int = DEFAULT_DOCS_PER_TOPIC
    queries_per_user: int = ABLATION_QUERIES_PER_USER
    target_assignment: str = "nearest"
    max_neighbors: int = 60
    churn_mode: str = "none"
    drift_topic_id: int | None = None
    drift_docs: int = 0
    drift_start_fraction: float = 0.0
    drift_end_fraction: float = 1.0
    drift_probability: float = 0.0

    @property
    def doc_counts_by_topic(self) -> list[int] | None:
        if self.drift_topic_id is None or self.drift_docs <= 0:
            return None
        return [self.docs_per_topic for _ in range(self.topics - 1)] + [self.drift_docs]


def iter_settings(experiment: str) -> Iterable[SyntheticAblationSetting]:
    if experiment == "base":
        yield SyntheticAblationSetting(
            experiment=experiment,
            label="default",
            users=BASELINE_USERS,
        )
    elif experiment == "user_scalability":
        for users in USER_SCALABILITY_USERS:
            yield SyntheticAblationSetting(
                experiment=experiment,
                label=f"users{users}",
                users=users,
            )
    elif experiment == "doc_scalability":
        for docs_per_topic in DOC_SCALABILITY_DOCS_PER_TOPIC:
            yield SyntheticAblationSetting(
                experiment=experiment,
                label=f"docs{docs_per_topic * DEFAULT_TOPIC_COUNT}",
                users=BASELINE_USERS,
                docs_per_topic=docs_per_topic,
                target_assignment="capped_topic",
            )
    elif experiment == "degree_sensitivity":
        for max_neighbors in DEGREE_SENSITIVITY_MAX_NEIGHBORS:
            yield SyntheticAblationSetting(
                experiment=experiment,
                label=f"degree{max_neighbors}",
                users=BASELINE_USERS,
                max_neighbors=max_neighbors,
            )
    elif experiment == "churn":
        yield SyntheticAblationSetting(
            experiment=experiment,
            label="drop50",
            users=BASELINE_USERS,
            churn_mode="drop",
        )
    elif experiment == "replacement_churn":
        yield SyntheticAblationSetting(
            experiment=experiment,
            label="drop50_replace50",
            users=BASELINE_USERS,
            churn_mode="replace",
        )
    elif experiment == "concept_drift":
        yield SyntheticAblationSetting(
            experiment=experiment,
            label="news_topic",
            users=BASELINE_USERS,
            topics=DEFAULT_TOPIC_COUNT + 1,
            target_assignment="capped_topic",
            drift_topic_id=DEFAULT_TOPIC_COUNT,
            drift_docs=10,
            drift_start_fraction=0.40,
            drift_end_fraction=0.60,
            drift_probability=0.20,
        )
    else:
        raise ValueError(f"unknown experiment {experiment!r}")


def setting_by_index(experiment: str, index: int) -> SyntheticAblationSetting:
    settings = list(iter_settings(experiment))
    if index < 0 or index >= len(settings):
        raise SystemExit(
            f"index {index} out of range for {experiment}; expected 0..{len(settings) - 1}"
        )
    return settings[index]


def data_dir_for_setting(data_root: Path, setting: SyntheticAblationSetting) -> Path:
    return data_root / "synthetic_ablation" / setting.experiment / setting.label


def precompute_setting(
    setting: SyntheticAblationSetting,
    *,
    data_root: Path,
    sigma: float,
    random_seed: int,
    min_target_count: int,
    dirichlet_alpha: float,
    embedding_dim: int,
    nearest_batch_size: int,
    user_batch_size: int,
    force: bool,
) -> dict[str, object]:
    data_dir = data_dir_for_setting(data_root, setting)
    manifest_path = data_dir / "embeddings" / "manifest.json"
    if manifest_path.exists() and not force:
        summary_path = data_dir / "synthetic_precompute_summary.json"
        if summary_path.exists():
            return json.loads(summary_path.read_text(encoding="utf-8"))
        return {"data_dir": str(data_dir), "manifest_path": str(manifest_path), "skipped": True}

    summary = precompute_synthetic_artifacts(
        data_dir=data_dir,
        topic_count=setting.topics,
        docs_per_topic=setting.docs_per_topic,
        user_count=setting.users,
        queries_per_user=setting.queries_per_user,
        dirichlet_alpha=dirichlet_alpha,
        embedding_dim=embedding_dim,
        sigma=sigma,
        random_seed=random_seed,
        min_target_count=min_target_count,
        run_sweep=False,
        nearest_batch_size=nearest_batch_size,
        user_batch_size=user_batch_size,
        target_assignment=setting.target_assignment,
        doc_counts_by_topic=setting.doc_counts_by_topic,
        drift_topic_id=setting.drift_topic_id,
        drift_start_fraction=setting.drift_start_fraction,
        drift_end_fraction=setting.drift_end_fraction,
        drift_probability=setting.drift_probability,
    )
    descriptor = {
        "setting": asdict(setting),
        "summary_path": str(data_dir / "synthetic_precompute_summary.json"),
        "manifest_path": str(manifest_path),
    }
    (data_dir / "synthetic_ablation_setting.json").write_text(
        json.dumps(descriptor, indent=2),
        encoding="utf-8",
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Precompute one synthetic ablation setting")
    parser.add_argument(
        "--experiment",
        choices=(
            "base",
            "user_scalability",
            "doc_scalability",
            "degree_sensitivity",
            "churn",
            "replacement_churn",
            "concept_drift",
        ),
        required=True,
    )
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--print-matrix", action="store_true")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--sigma", type=float, default=DEFAULT_ABLATION_SIGMA)
    parser.add_argument("--random-seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument("--min-target-count", type=int, default=DEFAULT_MIN_TARGET_COUNT)
    parser.add_argument("--dirichlet-alpha", type=float, default=DEFAULT_DIRICHLET_ALPHA)
    parser.add_argument("--embedding-dim", type=int, default=DEFAULT_EMBEDDING_DIM)
    parser.add_argument("--nearest-batch-size", type=int, default=1024)
    parser.add_argument("--user-batch-size", type=int, default=128)
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.print_matrix:
        print(json.dumps([asdict(setting) for setting in iter_settings(args.experiment)], indent=2))
        return

    setting = setting_by_index(args.experiment, args.index)
    summary = precompute_setting(
        setting,
        data_root=args.data_root,
        sigma=args.sigma,
        random_seed=args.random_seed,
        min_target_count=args.min_target_count,
        dirichlet_alpha=args.dirichlet_alpha,
        embedding_dim=args.embedding_dim,
        nearest_batch_size=args.nearest_batch_size,
        user_batch_size=args.user_batch_size,
        force=args.force,
    )
    print(json.dumps({"setting": asdict(setting), "summary": summary}, indent=2))


if __name__ == "__main__":
    main()

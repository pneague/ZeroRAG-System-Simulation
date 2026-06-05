#!/usr/bin/env python3
"""Build seed-specific randomized event metadata for repeat experiments."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, TextIO

import numpy as np


def _open_text(path: Path, mode: str) -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(path, mode, encoding="utf-8", newline="")
    return path.open(mode, encoding="utf-8", newline="")


def _synthetic_timestamp(index: int) -> str:
    base = datetime(2006, 1, 1, tzinfo=timezone.utc)
    return (base + timedelta(seconds=index)).strftime("%Y-%m-%d %H:%M:%S")


def _read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with _open_text(path, "rt") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path} has no CSV header")
        return list(reader.fieldnames), [dict(row) for row in reader]


def _shuffle_rows(
    rows: list[dict[str, str]],
    *,
    random_seed: int,
    shuffle_within_user: bool,
    user_field: str,
) -> list[dict[str, str]]:
    rng = np.random.default_rng(random_seed)
    if not shuffle_within_user:
        return [rows[int(index)] for index in rng.permutation(len(rows))]

    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    missing_user_rows: list[dict[str, str]] = []
    for row in rows:
        user_id = row.get(user_field, "")
        if user_id == "":
            missing_user_rows.append(row)
        else:
            grouped[user_id].append(row)

    shuffled: list[dict[str, str]] = []
    for user_id in sorted(grouped):
        user_rows = grouped[user_id]
        if len(user_rows) > 1:
            user_rows = [user_rows[int(index)] for index in rng.permutation(len(user_rows))]
        shuffled.extend(user_rows)
    shuffled.extend(missing_user_rows)
    return [shuffled[int(index)] for index in rng.permutation(len(shuffled))]


def _write_rows(path: Path, fieldnames: Iterable[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with _open_text(path, "wt") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Source events/metadata CSV(.gz)")
    parser.add_argument("--output", type=Path, required=True, help="Output shuffled CSV(.gz)")
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument("--random-seed", type=int, required=True)
    parser.add_argument(
        "--shuffle-within-user",
        action="store_true",
        help="First shuffle each user's rows, then globally shuffle all rows.",
    )
    parser.add_argument("--user-field", default="user_id")
    parser.add_argument(
        "--preserve-timestamps",
        action="store_true",
        help="Keep source timestamp values instead of assigning synthetic order timestamps.",
    )
    args = parser.parse_args()

    fieldnames, rows = _read_rows(args.input)
    if not rows:
        raise ValueError(f"{args.input} contains no rows")
    shuffled = _shuffle_rows(
        rows,
        random_seed=args.random_seed,
        shuffle_within_user=args.shuffle_within_user,
        user_field=args.user_field,
    )
    if "timestamp" in fieldnames and not args.preserve_timestamps:
        for index, row in enumerate(shuffled):
            row["timestamp"] = _synthetic_timestamp(index)
    _write_rows(args.output, fieldnames, shuffled)

    summary = {
        "input": str(args.input),
        "output": str(args.output),
        "random_seed": args.random_seed,
        "row_count": len(shuffled),
        "shuffle_within_user": args.shuffle_within_user,
        "user_field": args.user_field,
        "preserve_timestamps": args.preserve_timestamps,
        "fieldnames": fieldnames,
    }
    summary_path = args.summary_json or args.output.with_suffix(args.output.suffix + ".summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"[seed-repeat-metadata] wrote {len(shuffled):,} rows to {args.output} "
        f"(seed={args.random_seed})",
        flush=True,
    )


if __name__ == "__main__":
    main()

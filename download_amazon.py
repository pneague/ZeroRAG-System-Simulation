#!/usr/bin/env python3
"""Download the Amazon Product Search benchmark + UCSD per-category metadata.

The benchmark itself comes from the QingyaoAi/Amazon-Product-Search-Datasets
GitHub repo, where each category lives under a directory named
``reviews_<Category>_5.json.gz/`` (the ``.gz`` suffix is part of the
directory name in that repo). The ancillary product metadata is the matching
``meta_<Category>.json.gz`` file from the official UCSD Amazon Review Data
2018 release.

The script writes downloads under ``<data_root>/amazon/<category>/``, where
``<data_root>`` is resolved by :func:`src.data_root.resolve_dataset_dir`
so the same command works locally and on remote scratch.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.amazon_dataset import (
    AMAZON_BENCHMARK_FILENAMES,
    AMAZON_CATEGORY_SPECS,
    category_spec,
)
from src.data_root import resolve_dataset_dir


GITHUB_RAW_BASE = (
    "https://raw.githubusercontent.com/QingyaoAi/Amazon-Product-Search-Datasets/master"
)
# The QingyaoAi benchmark is built on the 2014 Amazon 5-core review dump, whose
# matching per-category metadata lives on Stanford SNAP. The earlier 2018
# release at jmcauley.ucsd.edu/data/amazon_v2/metaFiles2 covered only ~59% of
# the qrels asins because the catalog was re-crawled four years later. Do NOT
# point this back at the 2018 host without re-validating coverage.
UCSD_META_BASE = "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles"
CHUNK_SIZE = 1024 * 1024
INSECURE_UCSD_ENV_VAR = "LOW_COORD_DHT_INSECURE_UCSD"


def _env_truthy(value: str | None) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "on"}


def _download(url: str, output_path: Path, *, force: bool, verify: bool = True) -> None:
    import requests
    from tqdm import tqdm

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not force and output_path.exists() and output_path.stat().st_size > 0:
        print(f"Using cached {output_path}")
        return

    if not verify:
        try:
            import urllib3
            from urllib3.exceptions import InsecureRequestWarning

            urllib3.disable_warnings(InsecureRequestWarning)
        except Exception:
            pass

    with requests.get(url, stream=True, timeout=120, verify=verify) as response:
        response.raise_for_status()
        total = int(response.headers.get("Content-Length", "0"))
        with output_path.open("wb") as handle, tqdm(
            total=total,
            unit="B",
            unit_scale=True,
            desc=output_path.name,
        ) as progress:
            for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                if not chunk:
                    continue
                handle.write(chunk)
                progress.update(len(chunk))


def download_category(
    category: str,
    *,
    output_dir: Path | None = None,
    skip_benchmark: bool = False,
    skip_metadata: bool = False,
    force: bool = False,
    insecure_ucsd: bool = False,
) -> Path:
    if category == "all":
        raise SystemExit(
            "--category=all is not valid for download_amazon.py; download each "
            "real category (cell_phones, electronics, kindle, cds) and then run "
            "precompute_amazon_all.py to build the combined dataset."
        )
    spec = category_spec(category)
    if output_dir is None:
        output_dir = resolve_dataset_dir("amazon", category)
    output_dir = Path(output_dir)
    benchmark_dir = output_dir / "benchmark"
    metadata_dir = output_dir / "metadata"

    print(f"[amazon] category={category} ({spec.display_name})")
    print(f"[amazon] writing to {output_dir}")

    if not skip_benchmark:
        for filename in AMAZON_BENCHMARK_FILENAMES:
            url = f"{GITHUB_RAW_BASE}/{spec.benchmark_dir}/{filename}"
            _download(url, benchmark_dir / filename, force=force, verify=True)

    if not skip_metadata:
        url = f"{UCSD_META_BASE}/{spec.ucsd_meta_filename}"
        # The 2014 SNAP metadata host is plain HTTP, so TLS verification has no
        # effect; insecure_ucsd is accepted for backward compatibility only.
        _download(
            url,
            metadata_dir / spec.ucsd_meta_filename,
            force=force,
            verify=not insecure_ucsd,
        )

    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Download Amazon Product Search benchmark + UCSD per-category metadata. "
            "Pick one category and keep the category aligned everywhere downstream."
        )
    )
    parser.add_argument(
        "--category",
        choices=sorted(AMAZON_CATEGORY_SPECS),
        default="cell_phones",
        help="Amazon benchmark category (default: cell_phones)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Override the resolved output directory. By default this is "
            "<data_root>/amazon/<category>/ where <data_root> follows the "
            "LOW_COORD_DHT_DATA_ROOT / local / scratch resolution rule."
        ),
    )
    parser.add_argument(
        "--skip-benchmark",
        action="store_true",
        help="Do not download the QingyaoAi benchmark files",
    )
    parser.add_argument(
        "--skip-metadata",
        action="store_true",
        help="Do not download the UCSD metadata file",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if the destination file already exists",
    )
    parser.add_argument(
        "--insecure-ucsd",
        action="store_true",
        default=_env_truthy(os.environ.get(INSECURE_UCSD_ENV_VAR)),
        help=(
            "Kept for backward compatibility. The 2014 SNAP host "
            "(snap.stanford.edu) serves over plain HTTP, so TLS verification "
            "is not used for the metadata download regardless of this flag. "
            f"Also accepted via {INSECURE_UCSD_ENV_VAR}=1."
        ),
    )
    args = parser.parse_args()

    download_category(
        args.category,
        output_dir=Path(args.output_dir) if args.output_dir else None,
        skip_benchmark=args.skip_benchmark,
        skip_metadata=args.skip_metadata,
        force=args.force,
        insecure_ucsd=args.insecure_ucsd,
    )


if __name__ == "__main__":
    main()

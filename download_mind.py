#!/usr/bin/env python3
"""Download and extract the MIND-large train/dev packages.

MIND is distributed under Microsoft Research License Terms. The script requires
``--accept-license`` so the caller explicitly confirms those terms before any
download happens.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.data_root import resolve_dataset_dir
from src.mind_dataset import MIND_BLOB_BASE_URL, MIND_DOWNLOAD_FILENAMES


CHUNK_SIZE = 1024 * 1024


def _download(url: str, output_path: Path, *, force: bool) -> None:
    import requests
    from tqdm import tqdm

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not force and output_path.exists() and output_path.stat().st_size > 0:
        print(f"Using cached {output_path}")
        return

    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with requests.get(url, stream=True, timeout=120) as response:
        response.raise_for_status()
        total = int(response.headers.get("Content-Length", "0"))
        with tmp_path.open("wb") as handle, tqdm(
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
    tmp_path.replace(output_path)


def _extract_zip(zip_path: Path, output_dir: Path, *, force: bool) -> None:
    if output_dir.exists() and force:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    expected_files = ("behaviors.tsv", "news.tsv")
    if not force and all((output_dir / filename).exists() for filename in expected_files):
        print(f"Using extracted {output_dir}")
        return
    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall(output_dir)


def download_mind(*, output_dir: Path | None = None, force: bool = False) -> Path:
    data_dir = Path(output_dir) if output_dir is not None else resolve_dataset_dir("mind")
    raw_dir = data_dir / "raw"
    downloads_dir = raw_dir / "downloads"

    print(f"[mind] writing to {data_dir}")
    for filename in MIND_DOWNLOAD_FILENAMES:
        url = f"{MIND_BLOB_BASE_URL}/{filename}"
        zip_path = downloads_dir / filename
        split = "train" if filename.endswith("_train.zip") else "dev"
        _download(url, zip_path, force=force)
        _extract_zip(zip_path, raw_dir / split, force=force)

    return data_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Download MIND-large train/dev zip files and extract them under "
            "the resolved <data_root>/mind/raw directory."
        )
    )
    parser.add_argument(
        "--accept-license",
        action="store_true",
        help="Confirm that you accept the Microsoft Research License Terms for MIND.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Override the resolved output directory. By default this is "
            "<data_root>/mind/ where <data_root> follows the "
            "LOW_COORD_DHT_DATA_ROOT / local / scratch resolution rule."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download and re-extract even if files already exist.",
    )
    args = parser.parse_args()

    if not args.accept_license:
        raise SystemExit(
            "MIND is distributed under Microsoft Research License Terms. "
            "Re-run with --accept-license after you have reviewed and accepted them."
        )

    download_mind(
        output_dir=Path(args.output_dir) if args.output_dir else None,
        force=args.force,
    )


if __name__ == "__main__":
    main()

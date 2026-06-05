#!/usr/bin/env python3
"""Download AOL assets from Zenodo.

By default this downloads only ``metadata.csv`` because that is the file
required for the semantic-DHT simulation pipeline. The large
``letor.txt.gz`` file can be included explicitly with ``--include-letor``.
"""

from __future__ import annotations

import argparse
from pathlib import Path


DEFAULT_RECORD_ID = "15689455"
DEFAULT_OUTPUT_DIR = Path("data") / "aol"
CHUNK_SIZE = 1024 * 1024


def fetch_record(record_id: str) -> dict:
    import requests

    response = requests.get(f"https://zenodo.org/api/records/{record_id}", timeout=60)
    response.raise_for_status()
    return response.json()


def download_file(url: str, output_path: Path) -> None:
    import requests
    from tqdm import tqdm

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and output_path.stat().st_size > 0:
        print(f"Using cached {output_path}")
        return

    with requests.get(url, stream=True, timeout=60) as response:
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Download AOL from Zenodo")
    parser.add_argument("--record-id", default=DEFAULT_RECORD_ID, help="Zenodo record id")
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory where downloaded files are stored",
    )
    parser.add_argument(
        "--files",
        nargs="+",
        default=["metadata.csv"],
        help="Files to download from the record (default: metadata.csv)",
    )
    parser.add_argument(
        "--include-letor",
        action="store_true",
        help="Also download the large letor.txt.gz file needed for LETOR baselines",
    )
    args = parser.parse_args()

    requested_files = set(args.files)
    if args.include_letor:
        requested_files.add("letor.txt.gz")

    record = fetch_record(args.record_id)
    available_files = {entry["key"]: entry["links"]["self"] for entry in record["files"]}

    missing = sorted(requested_files - set(available_files))
    if missing:
        raise SystemExit(
            f"Requested files not present in record {args.record_id}: {missing}. "
            f"Available files: {sorted(available_files)}"
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading AOL to {output_dir}")
    for filename in sorted(requested_files):
        download_file(available_files[filename], output_dir / filename)


if __name__ == "__main__":
    main()


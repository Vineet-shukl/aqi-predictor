#!/usr/bin/env python3
"""Download the CPCB city-day air quality file from a public GitHub mirror.

The file is the `city_day.csv` table from Rohan Rao's Kaggle dataset
"Air Quality Data in India (2015 - 2020)" (CC0). Kaggle itself requires a
login, so this script uses the public raw mirror below. No API key is used.

The committed copy is the full file (about 2.5 MB). Re-run this script to
refresh it. `--check` verifies the local file without downloading.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import urllib.request
from pathlib import Path

SOURCE_URL = "https://raw.githubusercontent.com/adityarc19/aqi-india/main/city_day.csv"
EXPECTED_SHA256 = "0d84b21c3e4878bbad8df362f2ab05f61ad959538dddf5918e714077ed3c1847"
EXPECTED_ROWS = 29531
EXPECTED_HEADER = "City,Date,PM2.5,PM10,NO,NO2,NOx,NH3,CO,SO2,O3,Benzene,Toluene,Xylene,AQI,AQI_Bucket"

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DEST = ROOT / "data" / "city_day.csv"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def row_count(path: Path) -> int:
    with path.open(newline="") as handle:
        # Header plus data rows. csv is simple enough that a line count matches.
        lines = sum(1 for _ in handle)
    return lines - 1


def verify(path: Path) -> None:
    if not path.is_file():
        raise SystemExit(f"Missing dataset file: {path}")
    header = path.read_text(encoding="utf-8").splitlines()[0].strip()
    if header != EXPECTED_HEADER:
        raise SystemExit(f"Unexpected header:\n{header}")
    digest = sha256_file(path)
    if digest != EXPECTED_SHA256:
        raise SystemExit(f"SHA256 mismatch: {digest}")
    rows = row_count(path)
    if rows != EXPECTED_ROWS:
        raise SystemExit(f"Row count mismatch: {rows} (expected {EXPECTED_ROWS})")
    print(f"OK {path}")
    print(f"sha256 {digest}")
    print(f"rows {rows}")


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    temporary = dest.with_suffix(dest.suffix + ".partial")
    request = urllib.request.Request(url, headers={"User-Agent": "aqi-predictor-repro"})
    try:
        with urllib.request.urlopen(request, timeout=60) as response, temporary.open("wb") as out:
            shutil.copyfileobj(response, out)
    except Exception as exc:  # noqa: BLE001 - surface the download failure and stop
        temporary.unlink(missing_ok=True)
        raise SystemExit(f"Download failed: {exc}") from exc
    temporary.replace(dest)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Download or verify city_day.csv")
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    parser.add_argument("--url", default=SOURCE_URL)
    parser.add_argument("--force", action="store_true", help="Download even if the file matches")
    parser.add_argument("--check", action="store_true", help="Verify the local file and exit")
    args = parser.parse_args(argv)

    if args.check:
        verify(args.dest)
        return

    if args.dest.is_file() and not args.force:
        try:
            verify(args.dest)
            print("Dataset already matches the pinned checksum. Use --force to re-download.")
            return
        except SystemExit as exc:
            print(exc, file=sys.stderr)
            print("Re-downloading.", file=sys.stderr)

    print(f"Downloading {args.url}")
    download(args.url, args.dest)
    verify(args.dest)


if __name__ == "__main__":
    main()

"""Download and unpack the Massachusetts Roads archive into ``data/raw``."""

from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import urllib.request
import zipfile
from pathlib import Path

LOGGER = logging.getLogger(__name__)

# Replace these placeholders with the dataset's public URL or Kaggle owner/name.
DOWNLOAD_URL = ""
KAGGLE_DATASET = "balraj98/massachusetts-roads-dataset"
RAW_DIR = Path(__file__).resolve().parent / "raw"


def _extract_zip_safely(archive: Path, destination: Path) -> None:
    """Extract a ZIP archive while rejecting paths that escape the destination."""
    root = destination.resolve()
    with zipfile.ZipFile(archive) as zipped:
        for member in zipped.infolist():
            target = (destination / member.filename).resolve()
            if target != root and root not in target.parents:
                raise ValueError(f"Unsafe path in archive: {member.filename}")
        zipped.extractall(destination)


def main() -> None:
    """Download via a direct URL or the configured Kaggle CLI, then unpack."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=("direct", "kaggle"), required=True)
    args = parser.parse_args()
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if args.source == "direct":
        if DOWNLOAD_URL.startswith("<"):
            parser.error("Replace DOWNLOAD_URL in this file before using --source direct")
        archive = RAW_DIR / "dataset_download.zip"
        LOGGER.info("Downloading dataset archive")
        urllib.request.urlretrieve(DOWNLOAD_URL, archive)
    else:
        if KAGGLE_DATASET.startswith("<"):
            parser.error("Replace KAGGLE_DATASET in this file before using --source kaggle")
        if shutil.which("kaggle") is None:
            parser.error("Kaggle CLI not found; install it and configure Kaggle API credentials")
        subprocess.run(
            ["kaggle", "datasets", "download", "-d", KAGGLE_DATASET, "-p", str(RAW_DIR)],
            check=True,
        )
        archives = sorted(RAW_DIR.glob("*.zip"), key=lambda path: path.stat().st_mtime, reverse=True)
        if not archives:
            raise FileNotFoundError(f"Kaggle CLI did not produce a ZIP archive in {RAW_DIR}")
        archive = archives[0]

    _extract_zip_safely(archive, RAW_DIR)
    LOGGER.info("Extracted %s into %s", archive.name, RAW_DIR)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main()

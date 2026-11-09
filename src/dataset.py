"""Create reproducible, leakage-aware filename splits for the roads dataset."""

from __future__ import annotations

import logging
import random
from pathlib import Path

LOGGER = logging.getLogger(__name__)
RANDOM_SEED = 67
IMAGE_DIRECTORY = Path("tiff") / "train"
MASK_DIRECTORY = Path("tiff") / "train_labels"
IMAGE_SUFFIXES = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}


def create_deterministic_splits(data_dir: str, splits_dir: str) -> None:
    """Find paired image/mask files and write seeded 70/15/15 split manifests.

    ``data_dir`` is the raw data directory (normally ``data/raw``). Each
    manifest stores image filenames, one per line; masks are matched by stem
    under ``tiff/train_labels``. Pair matching is case-insensitive by stem.

    Args:
        data_dir: Root of the raw dataset directory.
        splits_dir: Directory in which train, validation, and test manifests
            will be written.

    Raises:
        FileNotFoundError: If either expected source directory is missing.
        ValueError: If no valid image/mask pairs are found.
    """
    raw_root = Path(data_dir)
    image_dir = raw_root / IMAGE_DIRECTORY
    mask_dir = raw_root / MASK_DIRECTORY
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    if not mask_dir.is_dir():
        raise FileNotFoundError(f"Mask directory not found: {mask_dir}")

    masks_by_stem = {
        path.stem.casefold(): path
        for path in mask_dir.iterdir()
        if path.is_file() and path.suffix.casefold() in IMAGE_SUFFIXES
    }
    pairs = []
    for image_path in sorted(image_dir.iterdir(), key=lambda path: path.name.casefold()):
        if image_path.is_file() and image_path.suffix.casefold() in IMAGE_SUFFIXES:
            mask_path = masks_by_stem.get(image_path.stem.casefold())
            if mask_path is not None:
                pairs.append(image_path.name)
            else:
                LOGGER.warning("Skipping image without matching mask: %s", image_path)

    if not pairs:
        raise ValueError(f"No valid image-mask pairs found in {image_dir} and {mask_dir}")

    random.Random(RANDOM_SEED).shuffle(pairs)
    train_end = int(len(pairs) * 0.70)
    validation_end = train_end + int(len(pairs) * 0.15)
    partitions = {
        "train": pairs[:train_end],
        "val": pairs[train_end:validation_end],
        "test": pairs[validation_end:],
    }

    output_dir = Path(splits_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for split_name, filenames in partitions.items():
        manifest = output_dir / f"{split_name}.txt"
        manifest.write_text("".join(f"{filename}\n" for filename in filenames), encoding="utf-8")
        LOGGER.info("Wrote %d filenames to %s", len(filenames), manifest)


def main() -> None:
    """Regenerate split manifests with the configured dataset paths."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/raw", help="Raw dataset root")
    parser.add_argument("--splits-dir", default="data/splits", help="Split manifest directory")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    create_deterministic_splits(args.data_dir, args.splits_dir)


if __name__ == "__main__":
    main()

"""
download_data.py -- fetch the tomato subset of the PlantVillage dataset.

WHAT THIS DOES
--------------
1. Downloads the PlantVillage dataset archive from Kaggle (~2 GB).
2. Extracts ONLY the tomato classes from the "color" variant into
   `data/raw/tomato/<class_name>/`.
3. Prints how many images landed in each class.

WHY EXTRACT A SUBSET INSTEAD OF EVERYTHING
------------------------------------------
The full archive holds 38 classes across 14 crops, and stores each image three
times over (color / grayscale / segmented) -- roughly 163,000 files. We train a
small CNN from scratch on a CPU, so we keep just the 10 tomato classes in their
original colour: ~18,000 images. Colour matters here because several tomato
diseases are told apart by the *hue* of the lesion, which grayscale throws away.

Selective extraction happens straight out of the zip, so the 145,000 files we
do not want are never written to disk at all.

USAGE
-----
    python src/download_data.py                # download + extract
    python src/download_data.py --delete-zip   # ...and remove the archive after

Requires Kaggle API credentials at ~/.kaggle/kaggle.json (see the README).
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
import zipfile
from pathlib import Path

# --- Configuration ---------------------------------------------------------

# Kaggle dataset identifier, in "owner/dataset-name" form. This particular
# upload is the standard PlantVillage release and includes the colour images.
KAGGLE_DATASET = "abdallahalidev/plantvillage-dataset"

# Project paths. `Path(__file__).resolve().parents[1]` walks up from
# `src/download_data.py` to the project root, so the script works no matter
# which directory it is run from.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
TOMATO_DIR = RAW_DIR / "tomato"
ZIP_PATH = RAW_DIR / "plantvillage-dataset.zip"

# Inside the archive, image paths look like:
#   plantvillage dataset/color/Tomato___Early_blight/0a1b2c.JPG
# This pattern keeps only the colour tomato folders. Capturing group 1 is the
# class name, group 2 the filename, which is how we rebuild a tidy folder tree.
MEMBER_PATTERN = re.compile(
    r"^plantvillage dataset/color/(Tomato___[^/]+)/([^/]+\.(?:jpg|jpeg|png))$",
    re.IGNORECASE,
)


# --- Steps -----------------------------------------------------------------

def check_credentials() -> None:
    """Fail early with a helpful message if the Kaggle API key is missing.

    The Kaggle library raises a bare OSError otherwise, which is not obvious to
    debug. Checking here turns a confusing crash into an actionable message.
    """
    key_path = Path.home() / ".kaggle" / "kaggle.json"
    if not key_path.exists() and not os.environ.get("KAGGLE_KEY"):
        sys.exit(
            f"Kaggle credentials not found at {key_path}\n"
            "Create a token at https://www.kaggle.com/settings -> API -> "
            "'Create New Token', then move kaggle.json to that path."
        )


def download_archive() -> None:
    """Download the dataset zip, unless it is already on disk.

    We deliberately do NOT pass unzip=True. Kaggle's own unzip would write all
    163k files; we want only the tomato ones, which `extract_tomato_images`
    handles.
    """
    if ZIP_PATH.exists():
        size_gb = ZIP_PATH.stat().st_size / 1024**3
        print(f"Archive already downloaded ({size_gb:.2f} GB) -- skipping download.")
        return

    # Imported here rather than at the top of the file so that the credential
    # check above runs first: the Kaggle package authenticates on import.
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()

    print(f"Downloading {KAGGLE_DATASET} (~2 GB, this takes a while)...")
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    api.dataset_download_files(KAGGLE_DATASET, path=str(RAW_DIR), unzip=False, quiet=False)

    # Kaggle names the file after the dataset; rename to a predictable name.
    downloaded = RAW_DIR / "plantvillage-dataset.zip"
    if not downloaded.exists():
        candidates = list(RAW_DIR.glob("*.zip"))
        if not candidates:
            sys.exit("Download finished but no .zip was found in data/raw/.")
        candidates[0].rename(ZIP_PATH)
    print(f"Saved to {ZIP_PATH}")


def extract_tomato_images() -> dict[str, int]:
    """Pull the colour tomato images out of the archive.

    Returns a {class_name: image_count} dictionary for the summary printout.
    """
    if not ZIP_PATH.exists():
        sys.exit(f"Archive not found at {ZIP_PATH}. Run the download step first.")

    counts: dict[str, int] = {}
    TOMATO_DIR.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(ZIP_PATH) as archive:
        # Match every member path against the pattern; keep only the hits.
        members = [
            (name, match)
            for name in archive.namelist()
            if (match := MEMBER_PATTERN.match(name))
        ]

        if not members:
            sys.exit(
                "No tomato images matched inside the archive. The dataset "
                "layout may have changed -- check MEMBER_PATTERN."
            )

        print(f"Extracting {len(members):,} tomato images...")
        for name, match in members:
            class_name, filename = match.group(1), match.group(2)
            target_dir = TOMATO_DIR / class_name
            target_dir.mkdir(parents=True, exist_ok=True)
            target_path = target_dir / filename

            counts[class_name] = counts.get(class_name, 0) + 1

            # Skip files already extracted so re-running the script is cheap
            # and safe (idempotent).
            if target_path.exists():
                continue

            # Stream the member straight to its destination rather than using
            # archive.extract(), which would recreate the nested
            # "plantvillage dataset/color/..." folders we are flattening away.
            with archive.open(name) as source, open(target_path, "wb") as dest:
                shutil.copyfileobj(source, dest)

    return counts


def print_summary(counts: dict[str, int]) -> None:
    """Print a per-class image count -- a first sanity check on the data."""
    total = sum(counts.values())
    print(f"\nExtracted {total:,} images across {len(counts)} classes into {TOMATO_DIR}\n")

    width = max(len(name) for name in counts)
    for class_name in sorted(counts):
        count = counts[class_name]
        share = count / total * 100
        print(f"  {class_name:<{width}}  {count:>6,}  ({share:4.1f}%)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--delete-zip",
        action="store_true",
        help="Delete the ~2 GB archive after extracting (frees disk space).",
    )
    args = parser.parse_args()

    check_credentials()
    download_archive()
    counts = extract_tomato_images()
    print_summary(counts)

    if args.delete_zip:
        ZIP_PATH.unlink()
        print(f"\nDeleted {ZIP_PATH}")
    else:
        print(f"\nArchive kept at {ZIP_PATH} (re-run with --delete-zip to remove it).")


if __name__ == "__main__":
    main()

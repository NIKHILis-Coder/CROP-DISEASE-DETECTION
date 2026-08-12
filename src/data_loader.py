"""
data_loader.py -- turn a folder of leaf photos into batched, augmented
tf.data.Dataset pipelines ready for training.

THE JOB THIS FILE DOES
----------------------
    data/raw/tomato/<class>/*.JPG          (18,160 files on disk)
        |
        |  build_manifest()      scan folders -> DataFrame of (filepath, label)
        v
    manifest of (path, label)
        |
        |  split_manifest()      stratified 70/15/15 train/val/test
        v
    three manifests, saved to data/processed/*.csv
        |
        |  make_dataset()        load -> resize 128x128 -> scale to [0,1]
        v
    tf.data.Dataset of (image_batch, label_batch)
        |
        |  build_augmentation()  flips/rotations/zoom -- TRAINING ONLY
        v
    ready for model.fit()

KEY DESIGN CHOICE: NOTHING IS WRITTEN TO DISK EXCEPT THE MANIFEST
-----------------------------------------------------------------
We do NOT pre-resize the images and save copies into data/processed/. Instead
the resize/scale/augment steps happen on the fly as the data streams to the
model. Reasons:

  * No stale derived data. If the input size changes from 128 to 160, only a
    constant changes -- there is no second copy of the dataset that silently
    disagrees with the code that made it.
  * Augmentation MUST be on the fly. Its whole purpose is that the model sees a
    *differently* distorted version of each image every epoch. Pre-computing it
    would freeze a fixed set of variants, giving away most of the benefit.
  * Disk space and time. No second 18k-image copy to write or keep in sync.

Only the split manifest is persisted, because *that* must stay fixed: a
reshuffled split between training and evaluation would leak training images into
the test set and inflate the score.

USAGE
-----
    from src.data_loader import load_splits

    train_ds, val_ds, test_ds, class_names = load_splits()
    model.fit(train_ds, validation_data=val_ds)
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

# --- Configuration ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "tomato"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# Where the tf.data disk cache lives. Ignored by Git (only *.csv is tracked
# under data/processed/). Delete this folder if the raw images ever change --
# a stale cache would silently keep serving the old data.
CACHE_DIR = PROCESSED_DIR / "cache"

# Cache backend: "disk" or "memory".
#
# Caching stores the decoded, resized images so that JPEG decoding happens once
# on the first epoch instead of on every epoch. That is the single biggest
# speedup available here, because decoding 12,712 JPEGs per epoch is pure
# repeated work -- the pixels never change.
#
# We default to DISK rather than memory because this machine has ~7.3 GB of RAM
# with well under 1 GB genuinely available, while the training cache alone needs
# ~600 MB and validation another ~500 MB. An in-memory cache that does not fit
# gets paged out by the OS, which is a disk cache with extra copying -- strictly
# worse than asking tf.data to use the disk deliberately. On a machine with
# RAM to spare, "memory" is faster; set CACHE_BACKEND accordingly.
CACHE_BACKEND = "disk"

# 128x128 is the compromise between detail and CPU training time.
#
# The source images are 256x256. Halving each side quarters the pixel count, so
# roughly a quarter of the convolution work per image -- the difference between
# iterating on the architecture a few times in an evening and waiting overnight.
# We do not go smaller: at 96x96 the fine speckling that distinguishes Septoria
# leaf spot from Target Spot starts to smear together, and the model cannot
# learn a feature the input no longer contains.
IMAGE_SIZE = (128, 128)

# 32 is the conventional default and suits CPU training: large enough that the
# gradient estimate is not too noisy, small enough to fit comfortably in RAM.
BATCH_SIZE = 32

# 70 / 15 / 15. Training gets the bulk; 15% of 18,160 is ~2,700 images for each
# of validation and test, which is plenty to measure performance stably even for
# the smallest class (~56 images of mosaic virus per split).
TRAIN_FRACTION = 0.70
VAL_FRACTION = 0.15
TEST_FRACTION = 0.15

# Fixed seed so the split is identical on every run and every machine. This is
# what makes results comparable between training runs.
SEED = 42

AUTOTUNE = tf.data.AUTOTUNE


# --- Step 1: build the manifest --------------------------------------------

def build_manifest() -> pd.DataFrame:
    """Scan data/raw/tomato/ and return a DataFrame of (filepath, label).

    The folder name is the label -- the layout created by download_data.py.
    Working with a manifest (a table of paths) rather than loading images here
    keeps this step instant and memory-free: 18,160 file paths are a few MB of
    text, while 18,160 decoded images would be several GB.
    """
    if not RAW_DIR.exists():
        raise FileNotFoundError(
            f"{RAW_DIR} not found. Run `python src/download_data.py` first."
        )

    records = []
    for class_dir in sorted(p for p in RAW_DIR.iterdir() if p.is_dir()):
        for image_path in class_dir.iterdir():
            if image_path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                records.append(
                    {
                        # Store paths relative to the project root so the CSV
                        # works on any machine, not just this one.
                        "filepath": image_path.relative_to(PROJECT_ROOT).as_posix(),
                        "label": class_dir.name.replace("Tomato___", ""),
                    }
                )

    manifest = pd.DataFrame(records)
    if manifest.empty:
        raise ValueError(f"No images found under {RAW_DIR}")
    return manifest


# --- Step 2: stratified split ----------------------------------------------

def split_manifest(manifest: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Split the manifest 70/15/15, stratified by class.

    STRATIFIED means each split keeps the same class proportions as the whole
    dataset. This matters because the classes are imbalanced 14.4x: a plain
    random split could hand the 373-image mosaic-virus class an unlucky draw and
    leave it with a handful of test images, making its score pure noise.

    Done in two steps, because scikit-learn's train_test_split cuts in two:
        1. 70% train        vs 30% temporary
        2. that 30% split down the middle -> 15% val, 15% test
    """
    train_df, temp_df = train_test_split(
        manifest,
        train_size=TRAIN_FRACTION,
        stratify=manifest["label"],  # <- the stratification
        random_state=SEED,
        shuffle=True,
    )

    # Of the remaining 30%, half becomes validation and half test. Expressed as
    # a fraction of `temp_df` that is 0.15 / 0.30 = 0.5.
    val_share = VAL_FRACTION / (VAL_FRACTION + TEST_FRACTION)
    val_df, test_df = train_test_split(
        temp_df,
        train_size=val_share,
        stratify=temp_df["label"],
        random_state=SEED,
        shuffle=True,
    )

    return {
        "train": train_df.reset_index(drop=True),
        "val": val_df.reset_index(drop=True),
        "test": test_df.reset_index(drop=True),
    }


def save_splits(splits: dict[str, pd.DataFrame]) -> None:
    """Write each split to data/processed/<name>_manifest.csv.

    Persisting the split is what makes results reproducible and leak-free: the
    test set is decided once and never reshuffled, so a model can never be
    evaluated on an image it was trained on.
    """
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    for name, df in splits.items():
        df.to_csv(PROCESSED_DIR / f"{name}_manifest.csv", index=False)


def load_split_manifests() -> dict[str, pd.DataFrame]:
    """Read the saved manifests back, regenerating them if they do not exist."""
    paths = {name: PROCESSED_DIR / f"{name}_manifest.csv" for name in ("train", "val", "test")}

    if not all(p.exists() for p in paths.values()):
        splits = split_manifest(build_manifest())
        save_splits(splits)
        return splits

    return {name: pd.read_csv(path) for name, path in paths.items()}


# --- Step 3: the tf.data pipeline ------------------------------------------

def _decode_and_resize(path: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    """Read one JPEG and resize it, returning **uint8** pixels (0-255).

    This is the expensive step -- and the one worth caching, because its output
    never changes. Deliberately stops short of normalising so the cached tensor
    stays uint8: one byte per channel instead of four.

        12,712 training images at 128x128x3
            as uint8   ->  ~600 MB
            as float32 -> ~2,400 MB

    Caching before the divide-by-255 therefore costs a quarter of the space for
    exactly the same saved work, since the division itself is trivially cheap
    compared with JPEG decoding.
    """
    image = tf.io.read_file(path)

    # channels=3 forces RGB. The EDA confirmed every image is already RGB, but
    # being explicit means a stray grayscale file could not silently change the
    # tensor shape and break the model.
    image = tf.io.decode_jpeg(image, channels=3)

    # Resize 256x256 -> 128x128. Bilinear (the default) averages neighbouring
    # pixels, which is the right choice when shrinking. tf.image.resize returns
    # float32, so cast back to uint8 to keep the cache small.
    image = tf.image.resize(image, IMAGE_SIZE)
    image = tf.cast(tf.round(image), tf.uint8)

    return image, label


def _normalize(image: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    """Scale uint8 pixels (0-255) to float32 in [0, 1].

    Neural networks train poorly on raw 0-255 inputs: large input values produce
    large gradients, which makes the optimiser take wildly sized steps. Small,
    consistently scaled inputs keep training stable.

    Shape-agnostic, so it works on a single image or a whole batch.
    """
    return tf.cast(image, tf.float32) / 255.0, label


def make_dataset(
    manifest: pd.DataFrame,
    class_names: list[str],
    *,
    shuffle: bool,
    augmentation: tf.keras.Sequential | None = None,
    cache_normalized: bool = False,
    cache_name: str | None = None,
    batch_size: int = BATCH_SIZE,
) -> tf.data.Dataset:
    """Build a batched tf.data pipeline from a manifest.

    PIPELINE ORDER -- and why each step sits where it does
    ------------------------------------------------------
        read + decode + resize     expensive, output never changes
        [normalise]                val/test only -- see cache_normalized
        CACHE                      <-- everything above runs once, ever
        shuffle                    must be AFTER cache, or the cache would
                                   freeze a single shuffled order forever
        batch
        [normalise]                train only: on batches, after the cache,
                                   so the cached tensors stay uint8
        augment                    must be AFTER cache, or every epoch would
                                   see the SAME frozen distortions
        prefetch                   overlap data prep with model compute

    The two "must be after cache" rules are the subtle part. `cache()` records
    the elements a stage produces and replays them identically on later epochs.
    Anything random placed *before* it gets recorded too -- so a shuffle before
    the cache yields one fixed order, and augmentation before the cache yields
    one fixed set of distortions, silently destroying most of their value.

    `shuffle` should be True for training and False for val/test -- evaluation
    order does not matter, and keeping it fixed makes results comparable.
    """
    # Labels become integer indices (0-9): "Bacterial_spot" -> 0, etc. The order
    # comes from class_names and must stay consistent everywhere, which is why
    # it is computed once and passed in.
    label_to_index = {name: i for i, name in enumerate(class_names)}

    paths = [str(PROJECT_ROOT / p) for p in manifest["filepath"]]
    labels = [label_to_index[name] for name in manifest["label"]]

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    # num_parallel_calls=AUTOTUNE decodes several images at once across CPU
    # cores, so disk reads and JPEG decoding do not starve the model of data.
    ds = ds.map(_decode_and_resize, num_parallel_calls=AUTOTUNE)

    # Validation and test are never augmented, so their pixels are identical on
    # every epoch -- which means normalisation can happen once, before the
    # cache, and the cache serves ready-to-use float32 tensors. Training caches
    # uint8 instead, to keep the (much larger) training cache small.
    if cache_normalized:
        ds = ds.map(_normalize, num_parallel_calls=AUTOTUNE)

    ds = _apply_cache(ds, cache_name)

    if shuffle:
        # A buffer as large as the split means a true full shuffle. Reshuffling
        # every epoch means batches differ run to run, which helps the model
        # generalise rather than memorise batch composition.
        ds = ds.shuffle(len(paths), seed=SEED, reshuffle_each_iteration=True)

    ds = ds.batch(batch_size)

    # Training normalises here: after the cache (so the cache stays uint8) and
    # on whole batches (cheaper than per-image).
    if not cache_normalized:
        ds = ds.map(_normalize, num_parallel_calls=AUTOTUNE)

    if augmentation is not None:
        # training=True is essential: Keras preprocessing layers are
        # deliberately no-ops at inference time, so without it these layers
        # would pass images through untouched.
        ds = ds.map(
            lambda x, y: (augmentation(x, training=True), y),
            num_parallel_calls=AUTOTUNE,
        )

    # prefetch overlaps data preparation with training: while the model works
    # through batch N, the CPU is already building batch N+1.
    return ds.prefetch(AUTOTUNE)


def _apply_cache(ds: tf.data.Dataset, cache_name: str | None) -> tf.data.Dataset:
    """Attach the cache, on disk or in memory, or not at all.

    Disk caching writes a `<name>.index` + `<name>.data-*` pair. Those files
    live under data/processed/, which Git ignores apart from the manifests.
    """
    if cache_name is None:
        return ds

    if CACHE_BACKEND == "memory":
        return ds.cache()

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return ds.cache(str(CACHE_DIR / cache_name))


# --- Step 4: augmentation (training only) ----------------------------------

def build_augmentation() -> tf.keras.Sequential:
    """Return the augmentation layers, applied to TRAINING data only.

    WHY AUGMENT AT ALL
    ------------------
    Augmentation shows the model randomly perturbed copies of each image, so it
    cannot memorise individual photographs and must learn the underlying
    pattern -- the lesion's shape and colour, not its exact position. It is a
    regularisation technique: it directly reduces overfitting, which is the main
    risk when training a network from scratch on 12,700 training images.

    WHY THESE THREE, AND WHY MODEST SETTINGS
    ----------------------------------------
    Each one models a variation that could genuinely occur when photographing a
    leaf, and nothing else:

      * RandomFlip("horizontal_and_vertical") -- a leaf has no inherent "up".
        Photograph the same leaf rotated by a half-turn and it is still the same
        disease, so a flip never changes the correct label. Free, safe variety.

      * RandomRotation(0.1) -- +/-10% of a full turn, about +/-36 degrees. Models
        the camera not being held perfectly square to the leaf. Kept modest
        because larger rotations pad the corners with empty pixels, and the
        model can learn to read that padding artefact instead of the leaf.

      * RandomZoom(0.1) -- +/-10%, modelling the camera being slightly nearer or
        further away.

    WHAT WE DELIBERATELY DO NOT DO, AND WHY
    ---------------------------------------
      * No heavy colour/brightness/contrast jitter. Diagnosis here depends on
        hue -- yellow mottling means mosaic virus, brown concentric rings mean
        early blight. Aggressively shifting colours would destroy the very
        signal the model needs, and could push an image's appearance towards a
        different disease while keeping its original label. That is worse than
        no augmentation: it teaches the model something false.

      * No shear or heavy geometric warping. PlantVillage images are flat,
        square-on lab photographs of detached leaves. Simulating extreme
        perspective invents a distribution neither the training nor the test
        data contains, spending model capacity on a variation that never occurs.

      * No random cropping. The lesion may be anywhere on the leaf; a crop can
        cut out the only diseased region while keeping the "diseased" label,
        producing an image whose label is simply wrong.

    In short: augment along the axes that genuinely vary (orientation, distance)
    and leave alone the axes that carry the diagnostic signal (colour).
    """
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal_and_vertical", seed=SEED),
            tf.keras.layers.RandomRotation(0.1, seed=SEED),
            tf.keras.layers.RandomZoom(0.1, seed=SEED),
        ],
        name="augmentation",
    )


def apply_augmentation(ds: tf.data.Dataset, augmentation: tf.keras.Sequential) -> tf.data.Dataset:
    """Map the augmentation layers over an already-built dataset.

    Kept for use outside the main pipeline (the Phase 3 notebook uses it to
    demonstrate augmentation on a fixed batch). Inside `make_dataset`,
    augmentation is applied in-line so it lands in the correct position
    relative to the cache.
    """
    return ds.map(
        lambda x, y: (augmentation(x, training=True), y),
        num_parallel_calls=AUTOTUNE,
    ).prefetch(AUTOTUNE)


# --- Public entry point ----------------------------------------------------

def load_splits(
    *,
    batch_size: int = BATCH_SIZE,
    augment_train: bool = True,
    cache: bool = True,
) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, list[str]]:
    """Build the train/val/test pipelines in one call.

    Returns (train_ds, val_ds, test_ds, class_names).

    Augmentation is applied to the training set ONLY. Validation and test data
    must stay untouched: they exist to estimate performance on real, unmodified
    images. Augmenting them would measure the model on distorted inputs no user
    will ever submit, and would make scores wobble run to run as the random
    distortions changed.

    `cache=True` decodes and resizes each image once and reuses the result on
    every later epoch. The first epoch is no faster (it does the work and fills
    the cache); every epoch after it skips JPEG decoding entirely.
    """
    splits = load_split_manifests()

    # Sorted so the label -> index mapping is deterministic across runs and
    # machines. The saved model's output index 3 must always mean the same class.
    class_names = sorted(splits["train"]["label"].unique())

    train_ds = make_dataset(
        splits["train"],
        class_names,
        shuffle=True,
        # Applied inside make_dataset so it lands after the cache.
        augmentation=build_augmentation() if augment_train else None,
        # Cache uint8: the training cache is by far the largest, and the
        # divide-by-255 is cheap enough to redo each epoch.
        cache_normalized=False,
        cache_name="train" if cache else None,
        batch_size=batch_size,
    )

    # Val and test are never augmented and never shuffled, so their batches are
    # identical every epoch -- normalise before caching and the cache serves
    # finished float32 tensors with no per-epoch work at all.
    val_ds = make_dataset(
        splits["val"],
        class_names,
        shuffle=False,
        cache_normalized=True,
        cache_name="val" if cache else None,
        batch_size=batch_size,
    )
    test_ds = make_dataset(
        splits["test"],
        class_names,
        shuffle=False,
        cache_normalized=True,
        cache_name="test" if cache else None,
        batch_size=batch_size,
    )

    return train_ds, val_ds, test_ds, class_names


def summarise_splits() -> pd.DataFrame:
    """Return a per-class count table for each split, to verify stratification.

    The proof that stratification worked: each split's percentage column should
    closely match the others and the overall distribution.
    """
    splits = load_split_manifests()

    table = pd.DataFrame({name: df["label"].value_counts() for name, df in splits.items()})
    table = table.fillna(0).astype(int)
    table["total"] = table.sum(axis=1)

    for name in ("train", "val", "test"):
        table[f"{name}_%"] = (table[name] / table[name].sum() * 100).round(1)

    return table.sort_values("total", ascending=False)


if __name__ == "__main__":
    # Running this file directly builds and saves the split, then prints the
    # per-class breakdown so stratification can be verified at a glance.
    manifest = build_manifest()
    print(f"Found {len(manifest):,} images across {manifest['label'].nunique()} classes\n")

    splits = split_manifest(manifest)
    save_splits(splits)

    for name, df in splits.items():
        print(f"{name:>5}: {len(df):>6,} images")

    print("\nPer-class distribution across splits:\n")
    print(summarise_splits().to_string())

    counts = summarise_splits()
    print("\nImbalance ratio (largest / smallest class) per split:")
    for name in ("train", "val", "test"):
        print(f"  {name:>5}: {counts[name].max() / counts[name].min():.1f}x")

"""
train.py -- train the CNN, with a timed benchmark before committing to a full run.

TWO MODES
---------
    python src/train.py           Benchmark. Runs exactly ONE epoch, reports how
                                  long it took, extrapolates the full run, and
                                  STOPS. Nothing is saved.

    python src/train.py --full    The real thing. Trains with early stopping,
                                  saves the best model to models/, writes the
                                  training history and the accuracy/loss curves.

WHY BENCHMARK FIRST
-------------------
This model trains on a CPU. Committing to 30 epochs without knowing the
per-epoch cost risks discovering three hours in that the run was never going to
finish in reasonable time. One timed epoch also answers a more important
question: *is the model learning at all?* A randomly initialised 10-class
classifier has a loss of about ln(10) = 2.30. If loss after one epoch is still
around 2.30, something is broken -- and it is far better to find that out in
five minutes than at the end of a long run.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

# Make `src` importable whether this is run as a script or a module.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import BATCH_SIZE, load_split_manifests, load_splits  # noqa: E402
from src.model import build_model  # noqa: E402

MODELS_DIR = PROJECT_ROOT / "models"
FIGURES_DIR = PROJECT_ROOT / "notebooks" / "figures"

DEFAULT_EPOCHS = 30
EARLY_STOPPING_PATIENCE = 5


# --- Class weights ---------------------------------------------------------

def compute_class_weights(class_names: list[str]) -> dict[int, float]:
    """Inverse-frequency class weights computed from the TRAINING split only.

    Formula (scikit-learn's "balanced" scheme):

        weight[c] = n_samples / (n_classes * count[c])

    A class holding exactly 1/10th of the data gets weight 1.0. Rarer classes
    get more, commoner classes less. The effect is that every *class*
    contributes roughly equally to the loss, rather than every *image* -- so the
    5,357-image Yellow Leaf Curl Virus class stops drowning out the 373-image
    mosaic virus class.

    Computed from training data only, for the same reason any other statistic
    is: deriving it from the whole dataset would leak information about the
    validation and test splits into training.
    """
    train_df = load_split_manifests()["train"]
    counts = train_df["label"].value_counts()

    n_samples = len(train_df)
    n_classes = len(class_names)

    # Keyed by the integer index the pipeline assigns, which is the position in
    # the sorted class_names list -- exactly what Keras expects.
    return {
        i: n_samples / (n_classes * counts[name])
        for i, name in enumerate(class_names)
    }


def report_class_weights(class_weights: dict[int, float], class_names: list[str]) -> None:
    """Print the weights next to the class counts that produced them."""
    train_df = load_split_manifests()["train"]
    counts = train_df["label"].value_counts()

    print("\nInverse-frequency class weights (from the training split):\n")
    print(f"  {'class':<38} {'images':>7} {'share':>7} {'weight':>8}")
    print(f"  {'-' * 38} {'-' * 7} {'-' * 7} {'-' * 8}")

    for i, name in enumerate(class_names):
        count = counts[name]
        share = count / len(train_df) * 100
        print(f"  {name:<38} {count:>7,} {share:>6.1f}% {class_weights[i]:>8.3f}")

    weights = np.array(list(class_weights.values()))
    print(f"\n  Weight range: {weights.min():.3f} (commonest) to {weights.max():.3f} (rarest)")
    print(f"  Ratio: {weights.max() / weights.min():.1f}x -- mirrors the 14.4x class imbalance")


# --- Benchmark mode --------------------------------------------------------

def run_benchmark(model: tf.keras.Model, train_ds, val_ds, class_weights, epochs: int) -> None:
    """Train for exactly one epoch, time it, and extrapolate."""
    random_loss = float(np.log(10))

    print("\n" + "=" * 74)
    print("BENCHMARK: one timed epoch")
    print("=" * 74)
    print(f"\nA randomly initialised 10-class model has loss ~ln(10) = {random_loss:.3f}.")
    print("If loss after this epoch is well below that, the model is learning.\n")

    start = time.perf_counter()
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=1,
        class_weight=class_weights,
        verbose=1,
    )
    elapsed = time.perf_counter() - start

    h = {key: values[0] for key, values in history.history.items()}

    print("\n" + "=" * 74)
    print("BENCHMARK RESULTS")
    print("=" * 74)

    print(f"\nTime for 1 epoch : {elapsed:.1f} s  ({elapsed / 60:.1f} min)")
    print(f"Estimated {epochs} epochs: {elapsed * epochs / 60:.0f} min "
          f"({elapsed * epochs / 3600:.1f} h)")
    print("  (conservative -- the first epoch includes one-off graph tracing,")
    print("   so later epochs are typically somewhat faster)")

    print(f"\n{'':17}{'loss':>10}{'accuracy':>12}")
    print(f"  {'training':<15}{h['loss']:>10.4f}{h['accuracy']:>12.2%}")
    print(f"  {'validation':<15}{h['val_loss']:>10.4f}{h['val_accuracy']:>12.2%}")

    print(f"\nSanity check vs random baseline (loss {random_loss:.3f}):")
    if h["loss"] < random_loss * 0.95:
        print(f"  PASS - training loss {h['loss']:.4f} is below random. The model is learning.")
    else:
        print(f"  FAIL - training loss {h['loss']:.4f} is at or above random ({random_loss:.3f}).")
        print("         Something is wrong: check the label mapping and the learning rate.")

    # 29.5% is the majority-class baseline measured in the EDA -- the accuracy a
    # model gets by ignoring the image and always guessing the biggest class.
    majority_baseline = 0.295
    print(f"\nValidation accuracy vs majority-class baseline ({majority_baseline:.1%}):")
    if h["val_accuracy"] > majority_baseline:
        print(f"  Above baseline ({h['val_accuracy']:.2%}) after a single epoch.")
    else:
        print(f"  Still at or below baseline ({h['val_accuracy']:.2%}) -- expected this early;")
        print("   what matters is whether it climbs over the full run.")

    print("\n" + "=" * 74)
    print("STOPPING HERE. Nothing has been saved.")
    print("To run the full training:  python src/train.py --full")
    print("=" * 74)


# --- Full training ---------------------------------------------------------

def run_full_training(model: tf.keras.Model, train_ds, val_ds, class_weights, epochs: int) -> None:
    """Train to convergence, keeping the best weights."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    callbacks = [
        # Stop when validation loss stops improving, and rewind to the best
        # weights seen. Without restore_best_weights the model kept is the
        # *last* one -- which by definition is several epochs past its peak.
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
        # Independent safety net: write the best model to disk as training goes,
        # so a crash or interrupt does not lose the run.
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(MODELS_DIR / "best_model.keras"),
            monitor="val_loss",
            save_best_only=True,
            verbose=0,
        ),
        # Plain-text record of every epoch, for later plotting or comparison.
        tf.keras.callbacks.CSVLogger(str(MODELS_DIR / "training_log.csv")),
    ]

    print(f"\nTraining for up to {epochs} epochs "
          f"(early stopping on val_loss, patience {EARLY_STOPPING_PATIENCE})...\n")

    start = time.perf_counter()
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1,
    )
    elapsed = time.perf_counter() - start

    model.save(MODELS_DIR / "final_model.keras")
    with open(MODELS_DIR / "history.json", "w") as fh:
        json.dump(history.history, fh, indent=2)

    plot_history(history.history)

    epochs_run = len(history.history["loss"])
    best_epoch = int(np.argmin(history.history["val_loss"])) + 1

    print("\n" + "=" * 74)
    print("TRAINING COMPLETE")
    print("=" * 74)
    print(f"Epochs run       : {epochs_run} of {epochs}")
    print(f"Best epoch       : {best_epoch} (lowest validation loss)")
    print(f"Total time       : {elapsed / 60:.1f} min ({elapsed / epochs_run:.1f} s/epoch)")
    print(f"Best val loss    : {min(history.history['val_loss']):.4f}")
    print(f"Best val accuracy: {max(history.history['val_accuracy']):.2%}")
    print(f"\nSaved: {MODELS_DIR / 'best_model.keras'}")
    print(f"       {MODELS_DIR / 'final_model.keras'}")
    print(f"       {MODELS_DIR / 'training_log.csv'}")
    print(f"       {FIGURES_DIR / 'training_curves.png'}")


def plot_history(history: dict) -> None:
    """Save the accuracy and loss curves.

    The gap between the training and validation curves is the thing to read:
    a widening gap means the model is memorising the training set.
    """
    import matplotlib
    matplotlib.use("Agg")  # no interactive window; we only write a file
    import matplotlib.pyplot as plt

    epochs = range(1, len(history["loss"]) + 1)
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(13, 4.5))

    ax_loss.plot(epochs, history["loss"], label="train")
    ax_loss.plot(epochs, history["val_loss"], label="validation")
    ax_loss.axhline(np.log(10), ls=":", c="grey", label="random guess (ln 10)")
    ax_loss.set_title("Loss")
    ax_loss.set_xlabel("epoch")
    ax_loss.legend()
    ax_loss.grid(alpha=0.3)

    ax_acc.plot(epochs, history["accuracy"], label="train")
    ax_acc.plot(epochs, history["val_accuracy"], label="validation")
    ax_acc.axhline(0.295, ls=":", c="grey", label="majority baseline (29.5%)")
    ax_acc.set_title("Accuracy")
    ax_acc.set_xlabel("epoch")
    ax_acc.legend()
    ax_acc.grid(alpha=0.3)

    fig.suptitle("Training curves", fontsize=13, weight="bold")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "training_curves.png", bbox_inches="tight", dpi=110)
    plt.close(fig)


# --- Entry point -----------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run the full training. Without this flag, only one timed epoch runs.",
    )
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()

    print("=" * 74)
    print("CROP DISEASE DETECTION -- CNN TRAINING")
    print("=" * 74)
    print(f"TensorFlow {tf.__version__} | devices: "
          f"{[d.device_type for d in tf.config.list_physical_devices()]}")

    train_ds, val_ds, _, class_names = load_splits(batch_size=args.batch_size)
    manifests = load_split_manifests()

    print(f"\nTrain: {len(manifests['train']):,} images "
          f"| Val: {len(manifests['val']):,} | Test: {len(manifests['test']):,}")
    print(f"Classes ({len(class_names)}): {', '.join(class_names)}")

    model = build_model(num_classes=len(class_names))

    print("\n" + "=" * 74)
    print("MODEL ARCHITECTURE")
    print("=" * 74)
    model.summary()

    total = model.count_params()
    print(f"\nTotal parameters: {total:,}")

    class_weights = compute_class_weights(class_names)
    report_class_weights(class_weights, class_names)

    if args.full:
        run_full_training(model, train_ds, val_ds, class_weights, args.epochs)
    else:
        run_benchmark(model, train_ds, val_ds, class_weights, args.epochs)


if __name__ == "__main__":
    main()

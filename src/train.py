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

DEFAULT_EPOCHS = 5
EARLY_STOPPING_PATIENCE = 5

# Refuse to start below this much free RAM. Two earlier runs on this machine
# were killed after paging themselves to a standstill -- the second reached
# only 1 of 15 epochs in 156 minutes at 0.91x CPU parallelism. Both were
# launched when available memory was already marginal. Failing fast at startup
# is far better than discovering the problem three hours in.
MIN_AVAILABLE_RAM_MB = 700
# Shorter than the early-stopping patience on purpose: try a smaller learning
# rate before giving up on the run entirely.
REDUCE_LR_PATIENCE = 3


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

class EpochTimer(tf.keras.callbacks.Callback):
    """Record how long each epoch takes.

    Needed because with caching the first epoch is unrepresentative: it does all
    the JPEG decoding *and* fills the cache. The steady-state cost is what
    epoch 2 onwards shows, and that is the number worth extrapolating from.
    """

    def __init__(self) -> None:
        super().__init__()
        self.times: list[float] = []
        self._start = 0.0

    def on_epoch_begin(self, epoch, logs=None) -> None:
        self._start = time.perf_counter()

    def on_epoch_end(self, epoch, logs=None) -> None:
        self.times.append(time.perf_counter() - self._start)


def run_benchmark(
    model: tf.keras.Model, train_ds, val_ds, class_weights, epochs: int,
    benchmark_epochs: int = 2,
) -> None:
    """Train for a couple of epochs, time each, and extrapolate."""
    random_loss = float(np.log(10))

    print("\n" + "=" * 74)
    print(f"BENCHMARK: {benchmark_epochs} timed epochs")
    print("=" * 74)
    print(f"\nA randomly initialised 10-class model has loss ~ln(10) = {random_loss:.3f}.")
    print("If loss drops well below that, the model is learning.")
    print("\nEpoch 1 fills the cache (full JPEG decode); epoch 2 reads from it.")
    print("The epoch-2 time is the steady-state cost to extrapolate from.\n")

    timer = EpochTimer()
    start = time.perf_counter()
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=benchmark_epochs,
        class_weight=class_weights,
        callbacks=[timer],
        verbose=1,
    )
    elapsed = time.perf_counter() - start

    # Report the LAST epoch's metrics -- the most recent state of the model.
    h = {key: values[-1] for key, values in history.history.items()}

    print("\n" + "=" * 74)
    print("BENCHMARK RESULTS")
    print("=" * 74)

    print("\nPer-epoch timings:")
    for i, t in enumerate(timer.times, start=1):
        note = " (cache fill -- includes full JPEG decode)" if i == 1 else " (cached)"
        print(f"  epoch {i}: {t:>7.1f} s  ({t / 60:.1f} min){note}")

    steady = timer.times[-1] if len(timer.times) > 1 else timer.times[0]
    if len(timer.times) > 1:
        speedup = timer.times[0] / steady
        print(f"\nCache speedup: {speedup:.2f}x "
              f"({timer.times[0]:.0f} s -> {steady:.0f} s per epoch)")

    print(f"\nTotal for {benchmark_epochs} epochs: {elapsed / 60:.1f} min")
    print(f"Estimated {epochs} epochs: "
          f"{(timer.times[0] + steady * (epochs - 1)) / 60:.0f} min "
          f"({(timer.times[0] + steady * (epochs - 1)) / 3600:.1f} h)")
    print("  (epoch 1 at the uncached rate, the rest at the cached rate)")

    print(f"\nMetrics after epoch {benchmark_epochs}:")
    print(f"\n{'':17}{'loss':>10}{'accuracy':>12}")
    print(f"  {'training':<15}{h['loss']:>10.4f}{h['accuracy']:>12.2%}")
    print(f"  {'validation':<15}{h['val_loss']:>10.4f}{h['val_accuracy']:>12.2%}")

    if benchmark_epochs > 1:
        print("\nPer-epoch progression (watching the BatchNorm warm-up gap):")
        for i in range(benchmark_epochs):
            print(f"  epoch {i + 1}: "
                  f"train loss {history.history['loss'][i]:.4f} "
                  f"acc {history.history['accuracy'][i]:.2%}  |  "
                  f"val loss {history.history['val_loss'][i]:.4f} "
                  f"acc {history.history['val_accuracy'][i]:.2%}")

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

    timer = EpochTimer()
    lr_tracker = LearningRateTracker()

    callbacks = [
        timer,
        lr_tracker,
        # Stop when validation loss stops improving, and rewind to the best
        # weights seen. Without restore_best_weights the model kept is the
        # *last* one -- which by definition is several epochs past its peak.
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
        # Halve the learning rate when validation loss stalls. Patience 3 is
        # deliberately shorter than EarlyStopping's 5, so the model gets a
        # chance to escape a plateau with smaller steps BEFORE training is
        # abandoned. Without that ordering, early stopping would fire first and
        # the LR reduction would never do anything.
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=REDUCE_LR_PATIENCE,
            min_lr=1e-6,
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
        # Append each epoch's metrics to disk AS THEY HAPPEN. Learned the hard
        # way: training_history.json below is only written when fit() returns,
        # so when a run had to be killed mid-flight its entire per-epoch history
        # was lost while the checkpoints survived -- because checkpoints are
        # written incrementally and the history was not. This closes that gap.
        tf.keras.callbacks.CSVLogger(str(MODELS_DIR / "training_log.csv")),
    ]

    print(f"\nTraining for up to {epochs} epochs "
          f"(early stopping on val_loss, patience {EARLY_STOPPING_PATIENCE}; "
          f"LR halved after {REDUCE_LR_PATIENCE} stalled epochs)...\n")

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

    epochs_run = len(history.history["loss"])
    best_epoch = int(np.argmin(history.history["val_loss"])) + 1
    early_stopped = epochs_run < epochs
    lrs = lr_tracker.rates
    lr_reduced = len(set(round(x, 12) for x in lrs)) > 1

    # Everything an interviewer (or future me) would want, in one small file.
    record = {
        "epochs_run": epochs_run,
        "epochs_max": epochs,
        "best_epoch": best_epoch,
        "early_stopped": early_stopped,
        "lr_reduced": lr_reduced,
        "total_seconds": round(elapsed, 1),
        "epoch_seconds": [round(t, 1) for t in timer.times],
        "learning_rate": lrs,
        **{key: [float(v) for v in values] for key, values in history.history.items()},
    }
    with open(MODELS_DIR / "training_history.json", "w") as fh:
        json.dump(record, fh, indent=2)

    write_training_log(record)
    plot_history(history.history)

    print("\n" + "=" * 74)
    print("TRAINING COMPLETE")
    print("=" * 74)
    print(f"Epochs run       : {epochs_run} of {epochs}"
          f"{'  (early stopping triggered)' if early_stopped else '  (ran to the cap)'}")
    print(f"Best epoch       : {best_epoch} (lowest validation loss)")
    print(f"Total time       : {elapsed / 60:.1f} min "
          f"({np.mean(timer.times):.0f} s/epoch average)")
    print(f"Best val loss    : {min(history.history['val_loss']):.4f}")
    print(f"Best val accuracy: {max(history.history['val_accuracy']):.2%}")
    print(f"Final train loss : {history.history['loss'][-1]:.4f} "
          f"acc {history.history['accuracy'][-1]:.2%}")
    print(f"Final val loss   : {history.history['val_loss'][-1]:.4f} "
          f"acc {history.history['val_accuracy'][-1]:.2%}")
    print(f"LR reduced       : {'yes' if lr_reduced else 'no'} "
          f"({lrs[0]:.2e} -> {lrs[-1]:.2e})")
    print(f"\nSaved: {MODELS_DIR / 'best_model.keras'}")
    print(f"       {MODELS_DIR / 'final_model.keras'}")
    print(f"       {MODELS_DIR / 'training_history.json'}")
    print(f"       {MODELS_DIR / 'training_log.txt'}")
    print(f"       {FIGURES_DIR / 'training_curves.png'}")


class LearningRateTracker(tf.keras.callbacks.Callback):
    """Record the learning rate at the end of every epoch.

    ReduceLROnPlateau changes the optimiser's LR mid-run; without recording it,
    the history would give no way to see when (or whether) that happened.
    """

    def __init__(self) -> None:
        super().__init__()
        self.rates: list[float] = []

    def on_epoch_end(self, epoch, logs=None) -> None:
        self.rates.append(float(tf.keras.backend.get_value(self.model.optimizer.learning_rate)))


def write_training_log(record: dict) -> None:
    """Write a human-readable per-epoch table to models/training_log.txt."""
    lines = [
        "Crop Disease Detection -- training log",
        "=" * 78,
        f"Epochs run: {record['epochs_run']} of {record['epochs_max']}"
        f"{'  (early stopped)' if record['early_stopped'] else ''}",
        f"Best epoch: {record['best_epoch']} (lowest val_loss)",
        f"Total time: {record['total_seconds'] / 60:.1f} min",
        "",
        f"{'epoch':>5} {'train_loss':>11} {'train_acc':>10} "
        f"{'val_loss':>10} {'val_acc':>9} {'lr':>10} {'secs':>7}",
        "-" * 78,
    ]

    for i in range(record["epochs_run"]):
        lines.append(
            f"{i + 1:>5} "
            f"{record['loss'][i]:>11.4f} "
            f"{record['accuracy'][i]:>9.2%} "
            f"{record['val_loss'][i]:>10.4f} "
            f"{record['val_accuracy'][i]:>8.2%} "
            f"{record['learning_rate'][i]:>10.2e} "
            f"{record['epoch_seconds'][i]:>7.1f}"
        )

    (MODELS_DIR / "training_log.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


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

def check_available_ram() -> None:
    """Warn (or refuse) if the machine does not have enough free RAM.

    Uses whatever is available without adding a dependency: psutil if present,
    otherwise a WMI query on Windows. If neither works, we let training proceed
    rather than blocking on a diagnostic.
    """
    available_mb = None

    try:
        import psutil  # type: ignore

        available_mb = psutil.virtual_memory().available / 1024**2
    except ImportError:
        if sys.platform == "win32":
            try:
                import subprocess

                out = subprocess.check_output(
                    ["wmic", "OS", "get", "FreePhysicalMemory", "/value"],
                    text=True, stderr=subprocess.DEVNULL,
                )
                available_mb = int(out.split("=")[1].strip()) / 1024
            except Exception:
                pass

    if available_mb is None:
        print("\n[ram guard] Could not measure available RAM -- continuing anyway.")
        return

    print(f"\n[ram guard] Available RAM: {available_mb:,.0f} MB "
          f"(minimum {MIN_AVAILABLE_RAM_MB:,} MB)")

    if available_mb < MIN_AVAILABLE_RAM_MB:
        sys.exit(
            f"\nRefusing to start: only {available_mb:,.0f} MB of RAM is free.\n"
            "Training on this machine pages itself to a standstill below roughly\n"
            f"{MIN_AVAILABLE_RAM_MB:,} MB. Close some applications and try again,\n"
            "or pass --skip-ram-check to override."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-ram-check",
        action="store_true",
        help="Start training even if available RAM is below the safe threshold.",
    )
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

    if not args.skip_ram_check:
        check_available_ram()

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

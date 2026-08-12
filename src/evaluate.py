"""
evaluate.py -- measure the trained model on the held-out test split.

WHY THE TEST SPLIT AND NOT VALIDATION
-------------------------------------
The validation split was used *during* training: early stopping watched it,
ReduceLROnPlateau watched it, and ModelCheckpoint chose which weights to keep
based on it. That makes validation performance optimistically biased -- the
model was, in a small but real sense, selected to do well on it.

The test split has been touched by nothing. It was carved out in Phase 3,
written to a CSV, and never read until this moment. That is what makes the
number here an honest estimate of performance on unseen data.

WHAT IT PRODUCES
----------------
    models/evaluation_metrics.json                    every number, machine-readable
    notebooks/figures/confusion_matrix_raw.png        counts
    notebooks/figures/confusion_matrix_normalized.png row-normalised (recall)
    notebooks/figures/misclassifications/grid.png     the 10 most confident errors

USAGE
-----
    python src/evaluate.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # write files, never open a window
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import load_split_manifests, load_splits  # noqa: E402

MODELS_DIR = PROJECT_ROOT / "models"
FIGURES_DIR = PROJECT_ROOT / "notebooks" / "figures"
MISCLASS_DIR = FIGURES_DIR / "misclassifications"

MODEL_PATH = MODELS_DIR / "best_model.keras"
METRICS_PATH = MODELS_DIR / "evaluation_metrics.json"


def _short(name: str) -> str:
    """Shorten a class name so it fits on a plot axis."""
    return name.replace("Tomato_", "").replace("_", " ")[:22]


def collect_predictions(model: tf.keras.Model, test_ds) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run the model over the test set once, keeping images, labels and probabilities.

    Returns (images, y_true, y_pred, confidences).
    """
    images, y_true, probabilities = [], [], []

    for batch_images, batch_labels in test_ds:
        batch_probs = model.predict(batch_images, verbose=0)
        images.append(batch_images.numpy())
        y_true.append(batch_labels.numpy())
        probabilities.append(batch_probs)

    images = np.concatenate(images)
    y_true = np.concatenate(y_true)
    probabilities = np.concatenate(probabilities)

    # argmax over the softmax output is the predicted class; the value at that
    # index is how confident the model is in it.
    y_pred = probabilities.argmax(axis=1)
    confidences = probabilities.max(axis=1)

    return images, y_true, y_pred, confidences


def plot_confusion_matrices(y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str]) -> np.ndarray:
    """Save both raw-count and row-normalised confusion matrices.

    The two answer different questions and both are needed:

    * RAW COUNTS show where the volume of errors is. Dominated by the big
      classes -- 50 mistakes on a 5,357-image class barely registers as a rate.
    * ROW-NORMALISED divides each row by its true-class total, so each cell is
      "what fraction of this class was predicted as that". The diagonal is
      per-class recall. This is the one to read when classes are imbalanced,
      because it puts a 373-image class on equal footing with a 5,357-image one.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    labels = list(range(len(class_names)))
    short = [_short(n) for n in class_names]

    cm = confusion_matrix(y_true, y_pred, labels=labels)

    # --- Raw counts ---
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar_kws={"label": "images"},
                xticklabels=short, yticklabels=short, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion matrix -- raw counts", fontsize=13, weight="bold")
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "confusion_matrix_raw.png", bbox_inches="tight", dpi=110)
    plt.close(fig)

    # --- Row-normalised ---
    # Guard against a zero row (a class with no test samples) producing NaN.
    row_totals = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, row_totals, out=np.zeros_like(cm, dtype=float), where=row_totals != 0)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues", vmin=0, vmax=1,
                cbar_kws={"label": "fraction of true class"},
                xticklabels=short, yticklabels=short, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion matrix -- row-normalised (diagonal = per-class recall)",
                 fontsize=13, weight="bold")
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "confusion_matrix_normalized.png", bbox_inches="tight", dpi=110)
    plt.close(fig)

    return cm


def plot_misclassifications(
    images: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray,
    confidences: np.ndarray, class_names: list[str], top_n: int = 10,
) -> list[dict]:
    """Save a grid of the model's most confident mistakes.

    Sorted by confidence rather than taken at random on purpose: a wrong answer
    the model was 99% sure about is far more diagnostic than one it was torn
    over. Confident errors reveal systematic confusions between classes; low
    confidence errors are just the model saying it does not know.
    """
    MISCLASS_DIR.mkdir(parents=True, exist_ok=True)

    wrong = np.flatnonzero(y_true != y_pred)
    if len(wrong) == 0:
        return []

    # Most confident errors first.
    worst = wrong[np.argsort(-confidences[wrong])][:top_n]

    cols = 5
    rows = int(np.ceil(len(worst) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.6, rows * 3.0))
    axes = np.atleast_1d(axes).ravel()

    records = []
    for ax, idx in zip(axes, worst):
        ax.imshow(np.clip(images[idx], 0, 1))
        ax.axis("off")
        true_name = _short(class_names[y_true[idx]])
        pred_name = _short(class_names[y_pred[idx]])
        ax.set_title(f"true: {true_name}\npred: {pred_name}\n{confidences[idx]:.1%} confident",
                     fontsize=7.5)
        records.append({
            "true": class_names[y_true[idx]],
            "predicted": class_names[y_pred[idx]],
            "confidence": float(confidences[idx]),
        })

    # Blank any unused cells in the grid.
    for ax in axes[len(worst):]:
        ax.axis("off")

    fig.suptitle("Most confident misclassifications", fontsize=13, weight="bold")
    plt.tight_layout()
    fig.savefig(MISCLASS_DIR / "grid.png", bbox_inches="tight", dpi=110)
    plt.close(fig)

    return records


def main() -> None:
    if not MODEL_PATH.exists():
        sys.exit(f"No trained model at {MODEL_PATH}. Run `python src/train.py --full` first.")

    print("=" * 74)
    print("EVALUATION -- held-out test split")
    print("=" * 74)

    model = tf.keras.models.load_model(MODEL_PATH)
    _, _, test_ds, class_names = load_splits(augment_train=False)
    test_manifest = load_split_manifests()["test"]

    print(f"\nModel   : {MODEL_PATH.name} ({model.count_params():,} parameters)")
    print(f"Test set: {len(test_manifest):,} images across {len(class_names)} classes")

    print("\nRunning inference...")
    images, y_true, y_pred, confidences = collect_predictions(model, test_ds)

    # Keras' own loss/accuracy, computed the same way as during training.
    test_loss, test_accuracy = model.evaluate(test_ds, verbose=0)

    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    weighted_f1 = f1_score(y_true, y_pred, average="weighted")

    report = classification_report(
        y_true, y_pred, labels=list(range(len(class_names))),
        target_names=class_names, output_dict=True, zero_division=0,
    )

    print("\n" + "=" * 74)
    print("OVERALL")
    print("=" * 74)
    print(f"  Test accuracy : {accuracy:.4f}  ({accuracy:.2%})")
    print(f"  Test loss     : {test_loss:.4f}")
    print(f"  Macro F1      : {macro_f1:.4f}   (every class counts equally)")
    print(f"  Weighted F1   : {weighted_f1:.4f}   (weighted by class size)")
    print(f"  Majority-class baseline: 29.5%")

    print("\n" + "=" * 74)
    print("PER-CLASS")
    print("=" * 74)
    print(classification_report(
        y_true, y_pred, labels=list(range(len(class_names))),
        target_names=class_names, zero_division=0,
    ))

    cm = plot_confusion_matrices(y_true, y_pred, class_names)
    misclassifications = plot_misclassifications(images, y_true, y_pred, confidences, class_names)

    # Which pairs of classes get confused most? Useful commentary for the README.
    confusions = []
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if i != j and cm[i, j] > 0:
                confusions.append({
                    "true": class_names[i],
                    "predicted": class_names[j],
                    "count": int(cm[i, j]),
                    "fraction_of_true_class": float(cm[i, j] / cm[i].sum()) if cm[i].sum() else 0.0,
                })
    confusions.sort(key=lambda c: -c["count"])

    print("\nTop confusions (true -> predicted):")
    for c in confusions[:5]:
        print(f"  {_short(c['true']):<24} -> {_short(c['predicted']):<24} "
              f"{c['count']:>4} ({c['fraction_of_true_class']:.1%} of that class)")

    metrics = {
        "test_accuracy": float(accuracy),
        "test_loss": float(test_loss),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "majority_class_baseline": 0.295,
        "n_test_images": int(len(y_true)),
        "n_classes": len(class_names),
        "class_names": class_names,
        "per_class": {
            name: {
                "precision": float(report[name]["precision"]),
                "recall": float(report[name]["recall"]),
                "f1": float(report[name]["f1-score"]),
                "support": int(report[name]["support"]),
            }
            for name in class_names
        },
        "confusion_matrix": cm.tolist(),
        "top_confusions": confusions[:10],
        "most_confident_misclassifications": misclassifications,
    }

    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"\nSaved: {METRICS_PATH}")
    print(f"       {FIGURES_DIR / 'confusion_matrix_raw.png'}")
    print(f"       {FIGURES_DIR / 'confusion_matrix_normalized.png'}")
    print(f"       {MISCLASS_DIR / 'grid.png'}")


if __name__ == "__main__":
    main()

"""
app.py -- Flask web app serving the trained crop-disease classifier.

ROUTES
------
    GET  /         upload form
    POST /predict  accepts an image, returns the top-3 predictions as JSON
    GET  /health   {"status": "ok"} -- a cheap liveness check

RUN
---
    cd app && python app.py
    open http://localhost:5000

THE ONE RULE THAT MATTERS HERE
------------------------------
Inference preprocessing must match training preprocessing EXACTLY. The model
learned to classify images that were resized to 64x64 with bilinear
interpolation and scaled to [0,1] by dividing by 255. If this file resized to
a different size, used a different interpolation, or normalised to [-1,1], the
model would receive inputs from a distribution it never saw in training and the
predictions would silently degrade -- no error, just worse answers.

To keep the two in step, the constants below are IMPORTED from the same module
the training pipeline uses rather than retyped. A hardcoded `(64, 64)` here
would work today and break the day someone retrains at a different size.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, render_template, request
from PIL import Image, UnidentifiedImageError

# Make the project root importable so we can share constants with training.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import IMAGE_SIZE, load_split_manifests  # noqa: E402

MODEL_PATH = PROJECT_ROOT / "models" / "best_model.keras"

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
MAX_UPLOAD_BYTES = 8 * 1024 * 1024  # 8 MB -- generous for a leaf photo

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES


# --- Load once at startup, not per request --------------------------------
#
# Loading a Keras model means reading the file, rebuilding the graph and
# allocating the weights -- on this model roughly a second, and far more for a
# large one. Doing that inside the request handler would add that cost to EVERY
# prediction and would hold several copies of the model in memory under
# concurrent load. Loading once at import time means each request only runs a
# forward pass.
#
# The trade-off is startup time and the fact that a model file swapped on disk
# needs a restart to take effect -- both entirely acceptable for a demo app, and
# the standard pattern for production serving too.

def _load_class_names() -> list[str]:
    """Read class names from the training manifest.

    Critically, this uses `sorted(unique labels)` -- the exact same expression
    the training pipeline used to map names to indices. The model's output
    neuron 3 means whatever class sat at position 3 in that sorted list, so any
    other ordering here would mislabel every prediction while looking perfectly
    plausible.
    """
    return sorted(load_split_manifests()["train"]["label"].unique())


import tensorflow as tf  # noqa: E402 -- needed by preprocess(), not just loading

print("Loading model and class names...")
try:
    MODEL = tf.keras.models.load_model(MODEL_PATH)
    CLASS_NAMES = _load_class_names()
    STARTUP_ERROR = None
    print(f"Loaded {MODEL_PATH.name} with {len(CLASS_NAMES)} classes: {', '.join(CLASS_NAMES)}")
except Exception as exc:  # noqa: BLE001 -- surface any startup failure via /health
    MODEL, CLASS_NAMES, STARTUP_ERROR = None, [], str(exc)
    print(f"WARNING: model failed to load -- {exc}")


# --- Preprocessing ---------------------------------------------------------

def preprocess(image: Image.Image) -> np.ndarray:
    """Turn a PIL image into exactly the tensor the model was trained on.

    Mirrors src/data_loader._decode_and_resize + _normalize, step for step:
        RGB -> tf.image.resize to IMAGE_SIZE -> round to uint8 -> / 255.0

    WHY tf.image.resize AND NOT PIL's OWN .resize()
    -----------------------------------------------
    They are not equivalent, and the difference is big enough to change
    predictions. Pillow's `Image.BILINEAR` applies an antialiasing filter when
    downscaling, averaging over a support region that scales with the reduction
    factor. `tf.image.resize` with the default `antialias=False` does not -- it
    samples. Going from 256x256 to 64x64 is a 4x reduction, where that
    distinction is severe.

    Measured on three test leaves: max per-pixel difference 0.19 (on a 0-1
    scale), and the predicted class changed on 2 of 3 images. The model was
    trained on the TensorFlow version, so the TensorFlow version is what it must
    be served. Using PIL here would have degraded every prediction silently --
    no error, no warning, just worse answers.

    The `round -> uint8 -> /255` sequence looks redundant but is deliberate: the
    training pipeline caches images as uint8, so it quantises at exactly this
    point. Reproducing that keeps inference bit-comparable with training.
    """
    # Convert first: a PNG may be RGBA or greyscale, and the model wants 3
    # channels. Doing this before resizing avoids surprises with alpha.
    array = np.asarray(image.convert("RGB"), dtype=np.uint8)

    resized = tf.image.resize(tf.constant(array), IMAGE_SIZE)
    quantised = tf.cast(tf.round(resized), tf.uint8)
    normalised = tf.cast(quantised, tf.float32) / 255.0

    return tf.expand_dims(normalised, axis=0).numpy()  # (1, H, W, 3)


def pretty(name: str) -> str:
    """Human-readable class label for display."""
    return name.replace("Tomato_", "").replace("_", " ").strip().title()


# --- Routes ---------------------------------------------------------------

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", model_loaded=MODEL is not None)


@app.route("/health", methods=["GET"])
def health():
    """Liveness check. Reports model status so a failed load is visible."""
    if MODEL is None:
        return jsonify({"status": "error", "detail": STARTUP_ERROR}), 503
    return jsonify({"status": "ok", "classes": len(CLASS_NAMES), "input_size": list(IMAGE_SIZE)})


@app.route("/predict", methods=["POST"])
def predict():
    """Classify an uploaded image and return the top 3 candidates.

    Every failure path returns a JSON error with a helpful message and an
    appropriate status code -- never a stack trace. A user who uploads a PDF
    should be told to upload an image, not shown a Werkzeug debug page.
    """
    if MODEL is None:
        return jsonify({"error": "Model is not loaded. Train it first with "
                                 "`python src/train.py --full`."}), 503

    # --- Validate the upload ---
    if "file" not in request.files:
        return jsonify({"error": "No file was uploaded. Choose an image and try again."}), 400

    upload = request.files["file"]
    if not upload.filename:
        return jsonify({"error": "No file was selected."}), 400

    extension = Path(upload.filename).suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        return jsonify({
            "error": f"Unsupported file type '{extension or 'unknown'}'. "
                     f"Please upload a JPG or PNG image."
        }), 400

    # --- Decode the image ---
    try:
        raw = upload.read()
        if not raw:
            return jsonify({"error": "The uploaded file is empty."}), 400
        image = Image.open(io.BytesIO(raw))
        image.load()  # force a full decode now, so truncated files fail here
    except (UnidentifiedImageError, OSError):
        return jsonify({
            "error": "That file could not be read as an image. It may be "
                     "corrupt or not a real image file."
        }), 400

    # --- Predict ---
    try:
        batch = preprocess(image)
        probabilities = MODEL.predict(batch, verbose=0)[0]
    except Exception:  # noqa: BLE001 -- never leak internals to the client
        app.logger.exception("Inference failed")
        return jsonify({"error": "Something went wrong while analysing the image. "
                                 "Please try a different photo."}), 500

    # --- Top 3, not just the argmax ---
    #
    # Several of these diseases look genuinely similar, and a single label hides
    # how close the call was. "Early blight 51%, Late blight 47%" is far more
    # honest -- and more useful to someone deciding what to do -- than "Early
    # blight" presented as certain. It also makes the model's failure modes
    # visible instead of hiding them.
    top_indices = np.argsort(probabilities)[::-1][:3]
    predictions = [
        {
            "class": CLASS_NAMES[i],
            "label": pretty(CLASS_NAMES[i]),
            "confidence": float(probabilities[i]),
        }
        for i in top_indices
    ]

    return jsonify({
        "predictions": predictions,
        "top_prediction": predictions[0]["label"],
        "confidence": predictions[0]["confidence"],
    })


if __name__ == "__main__":
    # debug=False: the debug reloader would load the model twice, and the
    # interactive debugger must never be exposed by a demo left running.
    app.run(host="127.0.0.1", port=5000, debug=False)

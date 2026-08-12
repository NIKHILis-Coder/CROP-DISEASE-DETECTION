"""
model.py -- the CNN architecture.

THE ARCHITECTURE, LAYER BY LAYER
--------------------------------
    Input (64, 64, 3)            one RGB image, pixels already scaled to [0,1]

    Conv2D(32, 3x3, relu)        learns 32 simple local patterns (edges, colour blobs)
    BatchNormalization           rescales activations so the next layer sees stable inputs
    MaxPooling2D(2x2)            halves height/width, keeping the strongest signal

    Conv2D(64, 3x3, relu)        combines edges into textures (speckling, lesion borders)
    BatchNormalization           same stabilising role, one level deeper
    MaxPooling2D(2x2)            halves again -- features now cover more of the leaf

    Conv2D(128, 3x3, relu)       combines textures into disease-scale patterns
    BatchNormalization           same again
    MaxPooling2D(2x2)            final downsample to a small, dense feature map

    GlobalAveragePooling2D       averages each feature map to one number -> 128 values
    Dense(128, relu)             mixes those 128 features into a decision space
    Dropout(0.5)                 randomly silences half of them during training
    Dense(10, softmax)           one probability per class, summing to 1

WHY THIS SHAPE
--------------
Three conv blocks, 32 -> 64 -> 128 filters. Each block halves the spatial size and
doubles the filter count -- the standard CNN trade: as you learn about *larger*
regions of the image, you need *more* kinds of pattern to describe them. Three
blocks takes 64x64 down to 8x8, which is small enough to summarise but still
large enough to preserve where things are. A fourth block would roughly double
CPU time per epoch for a model already at risk of overfitting 12,712 images.
"""

from __future__ import annotations

import tensorflow as tf

# The number of tomato classes. Passed in rather than hardcoded at the call site
# so a change to the dataset cannot silently mismatch the output layer.
NUM_CLASSES = 10
INPUT_SHAPE = (64, 64, 3)

# Adam's default. Small enough to be stable, large enough to make progress.
# If training diverges (loss -> NaN) this is the first knob to turn down.
LEARNING_RATE = 1e-3


def build_model(
    input_shape: tuple[int, int, int] = INPUT_SHAPE,
    num_classes: int = NUM_CLASSES,
    learning_rate: float = LEARNING_RATE,
    dropout_rate: float = 0.5,
) -> tf.keras.Model:
    """Build and compile the CNN.

    Every layer, in one line each:

    * ``Input(64, 64, 3)`` -- one resized RGB image; pixels are already scaled
      to [0,1] by the data pipeline, so no rescaling layer is needed here.
    * ``Conv2D(32, 3x3, relu)`` -- slides 32 small filters over the image to
      detect simple local patterns; 3x3 is the standard smallest useful window.
    * ``relu`` -- keeps positive signal, zeroes the rest; cheap and avoids the
      vanishing-gradient problem that sigmoid/tanh cause in deep stacks.
    * ``BatchNormalization`` -- renormalises each batch's activations, which
      keeps the scale of inputs to the next layer stable and lets training use a
      higher learning rate without diverging.
    * ``MaxPooling2D(2x2)`` -- keeps the strongest response in each 2x2 window,
      halving width and height; makes the model tolerant of a feature shifting
      by a pixel or two and cuts the compute for every later layer.
    * ``Conv2D(64, ...)`` -- twice as many filters on a half-size map: more kinds
      of pattern, each describing a larger area of the original leaf.
    * ``Conv2D(128, ...)`` -- the same step once more; these filters respond to
      disease-scale structure such as a lesion's shape and border.
    * ``GlobalAveragePooling2D`` -- collapses each 8x8 feature map to its
      average, giving 128 numbers; a Flatten here would give 8,192 instead.
    * ``Dense(128, relu)`` -- lets the model combine those features non-linearly
      before committing to a class.
    * ``Dropout(0.5)`` -- randomly zeroes half these units per training step, so
      no single feature can dominate; disabled automatically at inference.
    * ``Dense(10, softmax)`` -- turns scores into probabilities summing to 1.

    Compilation choices:

    * ``Adam`` -- adapts the step size per parameter; converges quickly without
      hand-tuning a learning-rate schedule, which matters when CPU time limits
      how many runs are affordable.
    * ``sparse_categorical_crossentropy`` -- the standard multi-class loss.
      The *sparse* variant takes integer labels (``3``) rather than one-hot
      vectors (``[0,0,0,1,...]``), which is exactly what the data pipeline
      produces -- no conversion step, less memory.
    * ``accuracy`` -- tracked because it is easy to read, but with a 14.4x class
      imbalance it is NOT the metric to judge the model by; Phase 5 reports a
      confusion matrix and per-class recall.
    """
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=input_shape, name="input"),

            # --- Block 1: simple local patterns -- edges, colour transitions ---
            # padding="same" keeps the spatial size unchanged through the conv,
            # so only the pooling layers change it. That makes the size
            # arithmetic easy to follow: 64 -> 32 -> 16 -> 8.
            tf.keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same", name="conv1"),
            tf.keras.layers.BatchNormalization(name="bn1"),
            tf.keras.layers.MaxPooling2D((2, 2), name="pool1"),

            # --- Block 2: textures built from those edges ---
            tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same", name="conv2"),
            tf.keras.layers.BatchNormalization(name="bn2"),
            tf.keras.layers.MaxPooling2D((2, 2), name="pool2"),

            # --- Block 3: disease-scale structure ---
            tf.keras.layers.Conv2D(128, (3, 3), activation="relu", padding="same", name="conv3"),
            tf.keras.layers.BatchNormalization(name="bn3"),
            tf.keras.layers.MaxPooling2D((2, 2), name="pool3"),

            # --- Head: turn feature maps into a class decision ---
            tf.keras.layers.GlobalAveragePooling2D(name="gap"),
            tf.keras.layers.Dense(128, activation="relu", name="dense1"),
            tf.keras.layers.Dropout(dropout_rate, name="dropout"),
            tf.keras.layers.Dense(num_classes, activation="softmax", name="output"),
        ],
        name="simple_cnn",
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


if __name__ == "__main__":
    # Running this file directly prints the architecture and parameter count.
    model = build_model()
    model.summary()

    total = model.count_params()
    trainable = sum(int(tf.size(w)) for w in model.trainable_weights)
    print(f"\nTotal parameters    : {total:,}")
    print(f"Trainable parameters: {trainable:,}")
    print(f"Non-trainable       : {total - trainable:,}  (BatchNorm moving averages)")

"""
model.py

Defines the model architecture and prediction logic for the wound
classifier. This file only builds/loads the model and runs predictions -
it never touches the dataset or performs training. Training lives in
train_model.py, which imports build_model() from here.

Usage (standalone prediction, once a model has been trained):
    python model.py path/to/image.jpg
"""

import os
import sys
import json

import numpy as np

# TensorFlow is imported inside the functions that use it, not here, so the
# Cloud Run server (litert_model.py) can import decide() and the constants
# without TensorFlow. Importing TensorFlow made a cold start take ~25 s.

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)          # .../Wound-Analyzer

MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "wound_model.keras")
CLASS_NAMES_PATH = os.path.join(MODEL_DIR, "class_names.json")
# Second model, trained on many more kinds of out-of-scope photo (rashes, bites,
# other skin conditions). It is used only to say "this is not a wound": see
# decide(). Created with `python train_model.py --gate`.
GATE_MODEL_PATH = os.path.join(MODEL_DIR, "out_of_scope_gate.keras")

IMG_SIZE = (224, 224)

# Below this confidence, predict() reports "unknown" instead of guessing.
CONFIDENCE_THRESHOLD = 0.60

# Extra class trained on photos that are none of the wound classes (normal
# skin, chronic wounds). It is never returned as a label: when it wins,
# predict() reports "unknown", and best_guess stays a wound class.
OUT_OF_SCOPE_CLASS = "out_of_scope"

# The gate model (GATE_MODEL_PATH) turns an answer into "unknown" when it gives
# OUT_OF_SCOPE_CLASS at least this probability. Chosen on the validation split:
# the lowest value that cost at most 5 correct validation wound answers.
GATE_THRESHOLD = 0.5


def build_model(num_classes, alpha=1.0):
    """Transfer learning on MobileNetV2 - ImageNet-pretrained base plus a
    small classifier head. alpha is MobileNetV2's width multiplier (1.0 is
    the standard network, 1.4 is wider). The base starts frozen so the head
    can be trained first; train_model.py then calls unfreeze_top_layers() to
    fine-tune the top of the base at a low learning rate."""
    from tensorflow.keras import layers, models
    from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2

    base_model = MobileNetV2(input_shape=IMG_SIZE + (3,), include_top=False, weights="imagenet", alpha=alpha)
    base_model.trainable = False

    inputs = layers.Input(shape=IMG_SIZE + (3,))
    # training=False keeps the base's BatchNormalization layers in inference
    # mode even after unfreezing, which is what keeps fine-tuning stable.
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs)
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def unfreeze_top_layers(model, num_layers):
    """Makes the last num_layers layers of the nested MobileNetV2 base
    trainable (BatchNormalization layers stay frozen). The caller must
    re-compile the model afterwards for this to take effect."""
    import tensorflow as tf
    from tensorflow.keras import layers

    base_model = next(layer for layer in model.layers if isinstance(layer, tf.keras.Model))
    base_model.trainable = True
    for layer in base_model.layers[:-num_layers]:
        layer.trainable = False
    for layer in base_model.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False


def load_trained_model():
    """Loads the saved model + class names from disk.
    Returns (model, class_names)."""
    if not os.path.exists(MODEL_PATH) or not os.path.exists(CLASS_NAMES_PATH):
        raise FileNotFoundError(
            "No trained model found. Run 'python train_model.py' first to train one."
        )
    import tensorflow as tf

    model = tf.keras.models.load_model(MODEL_PATH)
    with open(CLASS_NAMES_PATH, "r") as f:
        class_names = json.load(f)
    return model, class_names


def load_gate_model():
    """Loads the out-of-scope gate model (same classes as the classifier)."""
    if not os.path.exists(GATE_MODEL_PATH):
        raise FileNotFoundError(
            f"No gate model found at {GATE_MODEL_PATH}. Run 'python train_model.py --gate' to train one."
        )
    import tensorflow as tf

    return tf.keras.models.load_model(GATE_MODEL_PATH)


def decide(probabilities, class_names, gate_probabilities=None):
    """
    Turns one image's class probabilities into (label, confidence,
    best_guess) - the single decision rule used by predict() and by
    evaluate_model.py:
      - best_guess is the most likely WOUND class (never OUT_OF_SCOPE_CLASS)
      - confidence is the probability of best_guess
      - label is "unknown" if OUT_OF_SCOPE_CLASS is the most likely class,
        confidence is below CONFIDENCE_THRESHOLD, or the gate model gives
        OUT_OF_SCOPE_CLASS at least GATE_THRESHOLD; otherwise best_guess
    gate_probabilities are the gate model's probabilities for the same image
    (same class order as class_names), or None to use the classifier alone.
    The gate only ever adds "unknown"; it never changes best_guess.
    """
    wound_indices = [i for i, name in enumerate(class_names) if name != OUT_OF_SCOPE_CLASS]
    best_wound = max(wound_indices, key=lambda i: probabilities[i])
    best_guess = class_names[best_wound]
    confidence = float(probabilities[best_wound])
    out_of_scope = class_names[int(np.argmax(probabilities))] == OUT_OF_SCOPE_CLASS
    gated = (gate_probabilities is not None
             and float(gate_probabilities[class_names.index(OUT_OF_SCOPE_CLASS)]) >= GATE_THRESHOLD)
    label = "unknown" if out_of_scope or gated or confidence < CONFIDENCE_THRESHOLD else best_guess
    return label, confidence, best_guess


def predict(image_path, model=None, class_names=None, gate_model=None):
    """
    Predicts a category for a single image.

    If model/class_names aren't passed in, loads the saved model from disk
    (slower for repeated calls - pass them in yourself if calling this in
    a loop or from a server, so the model only loads once). The same goes
    for gate_model (see load_gate_model()).

    Returns (label, confidence, best_guess), as defined by decide():
      - label is the predicted wound class, or "unknown" if the photo looks
        out of scope (to the classifier or the gate) or confidence is below
        CONFIDENCE_THRESHOLD
      - confidence is the model's probability for best_guess
      - best_guess is the most likely wound class regardless, useful for
        logging even when label is "unknown"
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    if model is None or class_names is None:
        model, class_names = load_trained_model()
    if gate_model is None:
        gate_model = load_gate_model()

    import tensorflow as tf
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

    img = tf.keras.utils.load_img(image_path, target_size=IMG_SIZE)
    img_array = tf.keras.utils.img_to_array(img)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array, verbose=0)[0]
    gate_predictions = gate_model.predict(img_array, verbose=0)[0]
    return decide(predictions, class_names, gate_predictions)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python model.py path/to/image.jpg")
        sys.exit(1)

    label, confidence, best_guess = predict(sys.argv[1])
    if label == "unknown":
        reason = (f"below the {CONFIDENCE_THRESHOLD:.0%} threshold" if confidence < CONFIDENCE_THRESHOLD
                  else "the photo looks like none of the wound classes")
        print(f"unknown (best guess was '{best_guess}' at {confidence:.1%} confidence; {reason})")
    else:
        print(f"{label} ({confidence:.1%} confidence)")
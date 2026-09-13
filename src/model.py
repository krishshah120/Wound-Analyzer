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
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)          # .../Wound-Analyzer

MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "wound_model.keras")
CLASS_NAMES_PATH = os.path.join(MODEL_DIR, "class_names.json")

IMG_SIZE = (224, 224)

# Below this confidence, predict() reports "unknown" instead of guessing.
CONFIDENCE_THRESHOLD = 0.60

# Extra class trained on photos that are none of the wound classes (normal
# skin, chronic wounds). It is never returned as a label: when it wins,
# predict() reports "unknown", and best_guess stays a wound class.
OUT_OF_SCOPE_CLASS = "out_of_scope"


def build_model(num_classes):
    """Transfer learning on MobileNetV2 - ImageNet-pretrained base plus a
    small classifier head. The base starts frozen so the head can be trained
    first; train_model.py then calls unfreeze_top_layers() to fine-tune the
    top of the base at a low learning rate."""
    base_model = MobileNetV2(input_shape=IMG_SIZE + (3,), include_top=False, weights="imagenet")
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
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(CLASS_NAMES_PATH, "r") as f:
        class_names = json.load(f)
    return model, class_names


def decide(probabilities, class_names):
    """
    Turns one image's class probabilities into (label, confidence,
    best_guess) - the single decision rule used by predict() and by
    evaluate_model.py:
      - best_guess is the most likely WOUND class (never OUT_OF_SCOPE_CLASS)
      - confidence is the probability of best_guess
      - label is "unknown" if OUT_OF_SCOPE_CLASS is the most likely class or
        confidence is below CONFIDENCE_THRESHOLD, otherwise best_guess
    """
    wound_indices = [i for i, name in enumerate(class_names) if name != OUT_OF_SCOPE_CLASS]
    best_wound = max(wound_indices, key=lambda i: probabilities[i])
    best_guess = class_names[best_wound]
    confidence = float(probabilities[best_wound])
    out_of_scope = class_names[int(np.argmax(probabilities))] == OUT_OF_SCOPE_CLASS
    label = "unknown" if out_of_scope or confidence < CONFIDENCE_THRESHOLD else best_guess
    return label, confidence, best_guess


def predict(image_path, model=None, class_names=None):
    """
    Predicts a category for a single image.

    If model/class_names aren't passed in, loads the saved model from disk
    (slower for repeated calls - pass them in yourself if calling this in
    a loop or from a server, so the model only loads once).

    Returns (label, confidence, best_guess), as defined by decide():
      - label is the predicted wound class, or "unknown" if the photo looks
        out of scope or confidence is below CONFIDENCE_THRESHOLD
      - confidence is the model's probability for best_guess
      - best_guess is the most likely wound class regardless, useful for
        logging even when label is "unknown"
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    if model is None or class_names is None:
        model, class_names = load_trained_model()

    img = tf.keras.utils.load_img(image_path, target_size=IMG_SIZE)
    img_array = tf.keras.utils.img_to_array(img)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array, verbose=0)[0]
    return decide(predictions, class_names)


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
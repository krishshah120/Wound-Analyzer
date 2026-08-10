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


def build_model(num_classes):
    """Transfer learning on MobileNetV2 - frozen ImageNet-pretrained base
    plus a small trainable classifier head. Good fit for a small/medium
    dataset, since the pretrained convolutional features do most of the
    work and only the head needs to learn wound-specific patterns."""
    base_model = MobileNetV2(input_shape=IMG_SIZE + (3,), include_top=False, weights="imagenet")
    base_model.trainable = False

    inputs = layers.Input(shape=IMG_SIZE + (3,))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs)
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


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


def predict(image_path, model=None, class_names=None):
    """
    Predicts a category for a single image.

    If model/class_names aren't passed in, loads the saved model from disk
    (slower for repeated calls - pass them in yourself if calling this in
    a loop or from a server, so the model only loads once).

    Returns (label, confidence, best_guess):
      - label is the predicted class name, or "unknown" if confidence is
        below CONFIDENCE_THRESHOLD
      - confidence is the model's probability for its top prediction
      - best_guess is the top class name regardless of confidence, useful
        for logging even when label is "unknown"
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
    best_index = int(np.argmax(predictions))
    confidence = float(predictions[best_index])
    best_guess = class_names[best_index]

    label = best_guess if confidence >= CONFIDENCE_THRESHOLD else "unknown"
    return label, confidence, best_guess


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python model.py path/to/image.jpg")
        sys.exit(1)

    label, confidence, best_guess = predict(sys.argv[1])
    if label == "unknown":
        print(
            f"unknown (best guess was '{best_guess}' at {confidence:.1%} confidence, "
            f"below the {CONFIDENCE_THRESHOLD:.0%} threshold)"
        )
    else:
        print(f"{label} ({confidence:.1%} confidence)")
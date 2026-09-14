"""
litert_model.py

Runs the TensorFlow Lite copy of the model (models/wound_model.tflite) with
the small LiteRT runtime instead of full TensorFlow. The Cloud Run server uses
it (the Dockerfile sets WOUND_MODEL_FORMAT=tflite for app.py): importing
TensorFlow made a cold start take ~25 s, too close to the MRC site's 30 s proxy
timeout.

It gives the same answers as model.predict(): images are loaded the way
tf.keras.utils.load_img + MobileNetV2 preprocess_input load them, and the
decision rule is model.decide() itself. export_tflite.py writes the .tflite
file only after checking that on every val and test photo.

Requires: pip install ai-edge-litert pillow numpy
"""

import os
import json
import threading

import numpy as np
from PIL import Image
from ai_edge_litert.interpreter import Interpreter

from model import IMG_SIZE, MODEL_DIR, CLASS_NAMES_PATH, decide

TFLITE_MODEL_PATH = os.path.join(MODEL_DIR, "wound_model.tflite")


def load_image(image_path):
    """
    The model input for one image, matching
    tf.keras.utils.load_img(path, target_size=IMG_SIZE) -> img_to_array ->
    MobileNetV2 preprocess_input: RGB, nearest-neighbour resize to 224x224,
    float32 pixels scaled from [0, 255] to [-1, 1].
    """
    with Image.open(image_path) as img:
        if img.mode != "RGB":
            img = img.convert("RGB")
        size = (IMG_SIZE[1], IMG_SIZE[0])   # PIL sizes are (width, height)
        if img.size != size:
            img = img.resize(size, Image.NEAREST)
        pixels = np.asarray(img, dtype=np.float32)
    return pixels / 127.5 - 1.0


class LiteRTModel:
    """One loaded .tflite model. A LiteRT interpreter is not safe to use from
    several threads at once, and the server runs a few threads, so each
    prediction holds a lock."""

    def __init__(self, model_path=TFLITE_MODEL_PATH):
        self.interpreter = Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_index = self.interpreter.get_input_details()[0]["index"]
        self.output_index = self.interpreter.get_output_details()[0]["index"]
        self.lock = threading.Lock()

    def probabilities(self, image_array):
        """Class probabilities for one preprocessed (224, 224, 3) image."""
        with self.lock:
            self.interpreter.set_tensor(self.input_index, image_array[np.newaxis].astype(np.float32))
            self.interpreter.invoke()
            return self.interpreter.get_tensor(self.output_index)[0].copy()


def load_trained_model():
    """Loads the .tflite model + class names. Returns (model, class_names)."""
    if not os.path.exists(TFLITE_MODEL_PATH) or not os.path.exists(CLASS_NAMES_PATH):
        raise FileNotFoundError(
            f"No TensorFlow Lite model found at {TFLITE_MODEL_PATH}. "
            "Run 'python export_tflite.py' to create it from the trained model."
        )
    with open(CLASS_NAMES_PATH, "r") as f:
        class_names = json.load(f)
    return LiteRTModel(), class_names


def predict(image_path, model=None, class_names=None):
    """Same arguments and return value as model.predict():
    (label, confidence, best_guess) from model.decide()."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    if model is None or class_names is None:
        model, class_names = load_trained_model()
    return decide(model.probabilities(load_image(image_path)), class_names)

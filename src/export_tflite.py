"""
export_tflite.py

Converts the wound classifier (models/wound_model.keras) and the out-of-scope
gate (models/out_of_scope_gate.keras) to float32 TensorFlow Lite files
(models/wound_model.tflite, models/out_of_scope_gate.tflite; no quantization)
for the Cloud Run server, which runs them with LiteRT through litert_model.py.

Before writing anything, it runs every photo in data/val and data/test through
both versions: the Keras models with tf.keras.utils.load_img (as model.predict
loads images) and the converted models with litert_model.load_image (as the
server does), then compares the app's answer from model.decide() with the gate.
Those photos are all already 224x224, which would skip the resize step every
real upload goes through, so each one is also checked as a copy saved at a
different size (lossless PNG, alternately smaller and larger than 224). It
refuses to write the files if any image gets a different label or best_guess,
if any class probability of either model differs by more than
MAX_PROBABILITY_DIFFERENCE, or if the gate changed no answer at all (which
would mean shipping a gate that does nothing).

Re-run this after every retrain, then redeploy.

Usage:
    python export_tflite.py

Requires: pip install tensorflow ai-edge-litert
"""

import os
import shutil
import tempfile

import numpy as np
import tensorflow as tf
from PIL import Image

import model as keras_model
import litert_model

MAX_PROBABILITY_DIFFERENCE = 1e-4
CHECK_SPLITS = ("val", "test")
# (width, height) of the resized copies: smaller, larger and non-square, so the
# NEAREST resize to 224x224 is exercised both ways.
RESIZED_SIZES = [(640, 640), (154, 107), (1024, 768), (300, 168)]


def convert(model, candidate_path):
    """Writes model as a float32 .tflite file to candidate_path."""
    export_dir = tempfile.mkdtemp()
    try:
        model.export(export_dir)
        tflite_bytes = tf.lite.TFLiteConverter.from_saved_model(export_dir).convert()
    finally:
        shutil.rmtree(export_dir)
    with open(candidate_path, "wb") as f:
        f.write(tflite_bytes)
    return len(tflite_bytes)


def main():
    classifier, class_names = keras_model.load_trained_model()
    gate = keras_model.load_gate_model()

    outputs = [(classifier, litert_model.TFLITE_MODEL_PATH), (gate, litert_model.GATE_TFLITE_MODEL_PATH)]
    candidates = [final_path + ".candidate" for _, final_path in outputs]
    try:
        sizes = [convert(m, candidate) for (m, _), candidate in zip(outputs, candidates)]
        lite_classifier = litert_model.LiteRTModel(candidates[0])
        lite_gate = litert_model.LiteRTModel(candidates[1])

        paths = []
        for split in CHECK_SPLITS:
            for class_name in class_names:
                folder = os.path.join(keras_model.PROJECT_ROOT, "data", split, class_name)
                if not os.path.isdir(folder):
                    raise FileNotFoundError(f"Missing {folder}: the check needs the full val and test splits.")
                paths += [os.path.join(folder, name) for name in sorted(os.listdir(folder))
                          if name.lower().endswith((".png", ".jpg", ".jpeg"))]

        mismatches, worst_difference, checked, gate_changed = [], 0.0, 0, 0
        with tempfile.TemporaryDirectory() as resized_dir:
            for index, path in enumerate(paths):
                resized_path = os.path.join(resized_dir, "resized.png")
                resized_size = RESIZED_SIZES[index % len(RESIZED_SIZES)]
                with Image.open(path) as original:
                    original.convert("RGB").resize(resized_size, Image.BICUBIC).save(resized_path)
                for image_path in (path, resized_path):
                    img = tf.keras.utils.load_img(image_path, target_size=keras_model.IMG_SIZE)
                    x = tf.keras.applications.mobilenet_v2.preprocess_input(tf.keras.utils.img_to_array(img))[np.newaxis]
                    keras_probs = classifier.predict(x, verbose=0)[0]
                    keras_gate = gate.predict(x, verbose=0)[0]
                    lite_input = litert_model.load_image(image_path)
                    lite_probs = lite_classifier.probabilities(lite_input)
                    lite_gate_probs = lite_gate.probabilities(lite_input)
                    worst_difference = max(worst_difference, float(np.abs(keras_probs - lite_probs).max()),
                                           float(np.abs(keras_gate - lite_gate_probs).max()))
                    keras_answer = keras_model.decide(keras_probs, class_names, keras_gate)
                    lite_answer = keras_model.decide(lite_probs, class_names, lite_gate_probs)
                    # Does the gate still do anything? Without this, a gate that
                    # changed no answer at all would sail through the comparison
                    # (thanks to the MRC App session for the idea).
                    gate_changed += keras_answer[0] != keras_model.decide(keras_probs, class_names)[0]
                    if (keras_answer[0], keras_answer[2]) != (lite_answer[0], lite_answer[2]):
                        label = path if image_path == path else f"{path} resized to {resized_size}"
                        mismatches.append((label, keras_answer, lite_answer))
                    checked += 1

        print(f"Checked {checked} images ({len(paths)} photos in {', '.join(CHECK_SPLITS)}, each also resized), "
              f"classifier + gate: {len(mismatches)} different answers, largest probability difference {worst_difference:.2e}; "
              f"the gate turned {gate_changed} answers into 'unknown'")
        if gate_changed == 0:
            raise RuntimeError("The gate changed no answer on any checked image - it would ship doing nothing.")
        if mismatches or worst_difference > MAX_PROBABILITY_DIFFERENCE:
            for path, k, t in mismatches[:10]:
                print(f"  {path}: keras {k} vs tflite {t}")
            raise RuntimeError("The converted models do not match the Keras models - not writing the .tflite files.")
    except BaseException:
        for candidate in candidates:
            if os.path.exists(candidate):
                os.remove(candidate)
        raise

    for candidate, (_, final_path), size in zip(candidates, outputs, sizes):
        os.replace(candidate, final_path)
        print(f"Wrote {final_path} ({size:,} bytes)")


if __name__ == "__main__":
    main()

"""
export_tflite.py

Converts models/wound_model.keras to models/wound_model.tflite (float32, no
quantization) for the Cloud Run server, which runs it with LiteRT through
litert_model.py.

Before writing anything, it runs every photo in data/val and data/test through
both the Keras model (model.predict, as the app used to) and the converted
model (litert_model.predict, as the server now does). It refuses to write the
file if any photo gets a different label or best_guess, or if any class
probability differs by more than MAX_PROBABILITY_DIFFERENCE.

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

import model as keras_model
import litert_model

MAX_PROBABILITY_DIFFERENCE = 1e-4
CHECK_SPLITS = ("val", "test")


def main():
    model, class_names = keras_model.load_trained_model()

    export_dir = tempfile.mkdtemp()
    try:
        model.export(export_dir)
        tflite_bytes = tf.lite.TFLiteConverter.from_saved_model(export_dir).convert()
    finally:
        shutil.rmtree(export_dir)

    candidate_path = os.path.join(keras_model.MODEL_DIR, "wound_model.tflite.candidate")
    with open(candidate_path, "wb") as f:
        f.write(tflite_bytes)

    try:
        converted = litert_model.LiteRTModel(candidate_path)
        paths = []
        for split in CHECK_SPLITS:
            for class_name in class_names:
                folder = os.path.join(keras_model.PROJECT_ROOT, "data", split, class_name)
                if not os.path.isdir(folder):
                    raise FileNotFoundError(f"Missing {folder}: the check needs the full val and test splits.")
                paths += [os.path.join(folder, name) for name in sorted(os.listdir(folder))
                          if name.lower().endswith((".png", ".jpg", ".jpeg"))]

        mismatches, worst_difference = [], 0.0
        for path in paths:
            img = tf.keras.utils.load_img(path, target_size=keras_model.IMG_SIZE)
            x = tf.keras.applications.mobilenet_v2.preprocess_input(tf.keras.utils.img_to_array(img))
            keras_probs = model.predict(x[np.newaxis], verbose=0)[0]
            lite_probs = converted.probabilities(litert_model.load_image(path))
            worst_difference = max(worst_difference, float(np.abs(keras_probs - lite_probs).max()))
            keras_answer = keras_model.decide(keras_probs, class_names)
            lite_answer = keras_model.decide(lite_probs, class_names)
            if (keras_answer[0], keras_answer[2]) != (lite_answer[0], lite_answer[2]):
                mismatches.append((path, keras_answer, lite_answer))

        print(f"Checked {len(paths)} photos in {', '.join(CHECK_SPLITS)}: "
              f"{len(mismatches)} different answers, largest probability difference {worst_difference:.2e}")
        if mismatches or worst_difference > MAX_PROBABILITY_DIFFERENCE:
            for path, k, t in mismatches[:10]:
                print(f"  {path}: keras {k} vs tflite {t}")
            raise RuntimeError("The converted model does not match the Keras model - not writing wound_model.tflite.")
    except BaseException:
        os.remove(candidate_path)
        raise

    os.replace(candidate_path, litert_model.TFLITE_MODEL_PATH)
    print(f"Wrote {litert_model.TFLITE_MODEL_PATH} ({len(tflite_bytes):,} bytes)")


if __name__ == "__main__":
    main()

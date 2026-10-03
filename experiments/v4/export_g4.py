"""Export the G4new gate (trained with the healthy-skin photos) to TFLite with the production method
(src/export_tflite.convert, float32) and check it against Keras on every val+test photo."""
import os, sys, hashlib
import numpy as np
V4 = os.path.dirname(os.path.abspath(__file__)); WT = os.path.dirname(os.path.dirname(V4)); sys.path.insert(0, f"{WT}/src")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import tensorflow as tf
from export_tflite import convert
from litert_model import load_image
from ai_edge_litert.interpreter import Interpreter
OUT = f"{V4}/models/out_of_scope_gate_healthy.tflite"
m = tf.keras.models.load_model(f"{V4}/models/G4new_s1.keras", compile=False); convert(m, OUT)
paths = [f"{WT}/data/{s}/{c}/{f}" for s in ("val", "test") for c in sorted(os.listdir(f"{WT}/data/{s}")) for f in sorted(os.listdir(f"{WT}/data/{s}/{c}")) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
X = np.stack([load_image(p) for p in paths]); pk = m.predict(X, batch_size=32, verbose=0)
it = Interpreter(model_path=OUT); it.allocate_tensors(); i, o = it.get_input_details()[0]["index"], it.get_output_details()[0]["index"]; pt = []
for x in X: it.set_tensor(i, x[None].astype(np.float32)); it.invoke(); pt.append(it.get_tensor(o)[0])
pt = np.array(pt); diff = float(np.abs(pk - pt).max()); flips = int(((pk[:, 6] >= 0.5) != (pt[:, 6] >= 0.5)).sum())
print(f"{len(paths)} photos: max probability difference {diff:.1e}, gate decisions changed {flips}")
if diff > 1e-4 or flips: raise SystemExit("TFLite differs from Keras - not usable")
print("sha256", hashlib.sha256(open(OUT, "rb").read()).hexdigest(), os.path.getsize(OUT), "bytes")

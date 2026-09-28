"""Per-photo probabilities on the 401 held-out healthy-skin photos (new sources, never trained on):
deployed classifier (TFLite) + shipped gate (TFLite) + G4new gate (Keras), full frame and centre crop,
browser-encoded like the site. Writes heldout_views.npz."""
import os, io, csv, importlib.util
import numpy as np
from PIL import Image
V4 = os.path.dirname(os.path.abspath(__file__)); WT = os.path.dirname(os.path.dirname(V4))
spec = importlib.util.spec_from_file_location("prod", f"{WT}/experiments/v2/production_snapshot/predict.py"); prod = importlib.util.module_from_spec(spec); spec.loader.exec_module(prod)
from ai_edge_litert.interpreter import Interpreter
import tensorflow as tf
def it(p): i = Interpreter(model_path=p); i.allocate_tensors(); return i
clf = it(f"{WT}/experiments/v2/production_snapshot/wound_model.tflite"); ship = it(f"{WT}/experiments/v2/production_snapshot/out_of_scope_gate.tflite")
g4 = tf.keras.models.load_model(f"{V4}/models/G4new_s1.keras", compile=False)
rows = [r for r in csv.DictReader(open(f"{V4}/new_neg_manifest.csv")) if r["split"] == "heldout"]
C, S, G = [], [], []
for r in rows:
    b = io.BytesIO(); Image.open(f"{WT}/{r['file']}").convert("RGB").save(b, "JPEG", quality=82)
    xs = [prod._to_tensor(v) for v in prod._views(Image.open(io.BytesIO(b.getvalue())).convert("RGB"))]
    C.append([prod._run(clf, x) for x in xs]); S.append([prod._run(ship, x) for x in xs]); G.append([g4(x, training=False).numpy()[0] for x in xs])
np.savez(f"{V4}/heldout_views.npz", clf=np.array(C), gate_shipped=np.array(S), gate_g4new=np.array(G), source=np.array([r["source"] for r in rows]), files=np.array([r["file"] for r in rows]))
print("saved", len(rows))

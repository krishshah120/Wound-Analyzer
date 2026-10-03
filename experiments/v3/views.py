"""Per-photo classifier and gate probabilities for the full frame AND the production centre crop, with
production's exact image handling (experiments/v2/production_snapshot/predict.py). Saved so crop
policies, gates and the burn mapping can be compared without re-running models.
  val/test: 'raw' (stored file bytes) and 'browser' (<= 1024 px, JPEG q82, as the MRC site sends)
  RIT:      'browser' only (as eval_rit.py)
Usage: python views.py NAME [--clf model.tflite|.keras] [--gate model.tflite|.keras]
Default models = the deployed TFLite pair. Writes experiments/v3/views_NAME.npz."""
import os, io, sys, csv, argparse, importlib.util
import numpy as np
from PIL import Image
from common import V3, WT, NAMES, files
SNAP = f"{WT}/experiments/v2/production_snapshot"; RIT = f"{WT}/experiments/v2/rit"
ap = argparse.ArgumentParser(); ap.add_argument("name"); ap.add_argument("--clf", default=f"{SNAP}/wound_model.tflite")
ap.add_argument("--gate", default=f"{SNAP}/out_of_scope_gate.tflite"); a = ap.parse_args()
spec = importlib.util.spec_from_file_location("prod", f"{SNAP}/predict.py"); prod = importlib.util.module_from_spec(spec); spec.loader.exec_module(prod)

def loader(path):
    if path.endswith(".tflite"):
        from ai_edge_litert.interpreter import Interpreter
        it = Interpreter(model_path=path); it.allocate_tensors(); return lambda x: prod._run(it, x)
    import tensorflow as tf
    m = tf.keras.models.load_model(path, compile=False); return lambda x: m(x, training=False).numpy()[0]
clf, gate = loader(a.clf), loader(a.gate)

def browser(raw):
    img = Image.open(io.BytesIO(raw)).convert("RGB"); w, h = img.size; s = min(1.0, 1024 / max(w, h))
    if s < 1: img = img.resize((round(w * s), round(h * s)), Image.BILINEAR)
    b = io.BytesIO(); img.save(b, "JPEG", quality=82); return b.getvalue()

def run(raw):
    img = Image.open(io.BytesIO(raw)).convert("RGB"); out = []
    vs = prod._views(img)
    assert len(vs) == 2
    for v in vs:
        x = prod._to_tensor(v); out.append((clf(x), gate(x)))
    return np.array([o[0] for o in out]), np.array([o[1] for o in out])   # (2 views, 7) each

res = {}
for split in ("val", "test"):
    paths = [f"{WT}/data/{split}/{c}/{f}" for c in NAMES for f in sorted(os.listdir(f"{WT}/data/{split}/{c}")) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    assert [p.split("/")[-1] for p in paths] == files(split)
    y = np.array([NAMES.index(p.split("/")[-2]) for p in paths]); res[f"{split}_y"] = y
    for mode in ("raw", "browser"):
        P, G = zip(*[run(open(p, "rb").read() if mode == "raw" else browser(open(p, "rb").read())) for p in paths])
        res[f"{split}_{mode}_clf"], res[f"{split}_{mode}_gate"] = np.array(P), np.array(G)
        print(split, mode, "done", flush=True)
rows = list(csv.DictReader(open(f"{RIT}/manifest.csv")))
P, G = zip(*[run(browser(open(f"{RIT}/images/{r['file']}", "rb").read())) for r in rows])
res["rit_browser_clf"], res["rit_browser_gate"] = np.array(P), np.array(G); res["rit_files"] = np.array([r["file"] for r in rows])
np.savez(f"{V3}/views_{a.name}.npz", **res)
print("saved", f"views_{a.name}.npz", {k: v.shape for k, v in res.items()})

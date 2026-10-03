"""v5b/v5c: full-frame classifier probabilities for one model on every v5 photo set, each set encoded exactly as
its existing views file was (val/test/RIT and arm's length: <=1024 px + JPEG q82 as v3/views.py and
v4/framed_views.py; held-out and independent healthy: JPEG q82 as v4/heldout_views.py).
Usage: python clf_views.py NAME model.keras|.tflite  -> clfviews_NAME.npz (keys: set name -> (n, 7))"""
import os, io, sys, csv, importlib.util
import numpy as np
from PIL import Image
V5 = os.path.dirname(os.path.abspath(__file__)); WT = os.path.dirname(os.path.dirname(V5)); EXP = f"{WT}/experiments"
spec = importlib.util.spec_from_file_location("prod", f"{EXP}/v2/production_snapshot/predict.py"); prod = importlib.util.module_from_spec(spec); spec.loader.exec_module(prod)
name, path = sys.argv[1], sys.argv[2]
if path.endswith(".tflite"):
    from ai_edge_litert.interpreter import Interpreter
    it = Interpreter(model_path=path); it.allocate_tensors(); clf = lambda x: prod._run(it, x)
else:
    import tensorflow as tf
    m = tf.keras.models.load_model(path, compile=False); clf = lambda x: m(x, training=False).numpy()[0]
NAMES = ["abrasion", "bruise", "burn_1st_degree", "burn_2nd_degree", "burn_3rd_degree", "cut", "out_of_scope"]
def browser(raw):
    img = Image.open(io.BytesIO(raw)).convert("RGB"); w, h = img.size; s = min(1.0, 1024 / max(w, h))
    if s < 1: img = img.resize((round(w * s), round(h * s)), Image.BILINEAR)
    b = io.BytesIO(); img.save(b, "JPEG", quality=82); return b.getvalue()
def q82(raw):
    b = io.BytesIO(); Image.open(io.BytesIO(raw)).convert("RGB").save(b, "JPEG", quality=82); return b.getvalue()
def full(raw): return clf(prod._to_tensor(prod._views(Image.open(io.BytesIO(raw)).convert("RGB"))[0]))
def listing(root): return [f"{root}/{c}/{f}" for c in NAMES for f in sorted(os.listdir(f"{root}/{c}")) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
sets = {"val": (listing(f"{WT}/data/val"), browser), "test": (listing(f"{WT}/data/test"), browser),
        "val_arm": (listing(f"{EXP}/data_frame50/val"), browser), "test_arm": (listing(f"{EXP}/data_frame50/test"), browser),
        "rit": ([f"{EXP}/v2/rit/images/{r['file']}" for r in csv.DictReader(open(f"{EXP}/v2/rit/manifest.csv"))], browser),
        "heldout": ([f"{WT}/{r['file']}" for r in csv.DictReader(open(f"{EXP}/v4/new_neg_manifest.csv")) if r["split"] == "heldout"], q82),
        "indep": ([f"{WT}/{r['file']}" for r in csv.DictReader(open(f"{EXP}/v4/new_neg_manifest.csv")) if r["split"] == "excluded_resembles_eval"], q82)}
out = {}
for k, (paths, enc) in sets.items():
    out[k] = np.array([full(enc(open(p, "rb").read())) for p in paths]); print(k, out[k].shape, flush=True)
np.savez(f"{V5}/clfviews_{name}.npz", **out)

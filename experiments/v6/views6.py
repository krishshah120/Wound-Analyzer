"""v6: full-frame probabilities for one model on every v6 photo set.
Encodings: val/test/RIT and data_frame50 (f50) = <=1024 px + JPEG q82 (as v3/views.py); healthy sets = JPEG q82
(as v4/heldout_views.py); test_arm2 = MRC App's arm's-length reconstruction (~/Downloads/wound-analyzer-artifacts/
armslength_reconstruction.py: blurred self-copy, photo at 0.5 scale, every photo incl. non-wounds, JPEG q82).
Usage: python views6.py NAME model.tflite|.keras -> views6_NAME.npz (set -> (n, 7)) + views6_files.json"""
import os, io, sys, csv, json, importlib.util
import numpy as np
from PIL import Image, ImageFilter
V6 = os.path.dirname(os.path.abspath(__file__)); WT = os.path.dirname(os.path.dirname(V6)); EXP = f"{WT}/experiments"
spec = importlib.util.spec_from_file_location("prod", f"{EXP}/v2/production_snapshot/predict.py"); prod = importlib.util.module_from_spec(spec); spec.loader.exec_module(prod)
name, path = sys.argv[1], sys.argv[2]
if path.endswith(".tflite"):
    from ai_edge_litert.interpreter import Interpreter
    it = Interpreter(model_path=path); it.allocate_tensors(); run = lambda x: prod._run(it, x)
else:
    import tensorflow as tf
    m = tf.keras.models.load_model(path, compile=False); run = lambda x: m(x, training=False).numpy()[0]
NAMES = ["abrasion", "bruise", "burn_1st_degree", "burn_2nd_degree", "burn_3rd_degree", "cut", "out_of_scope"]
def browser(raw):
    img = Image.open(io.BytesIO(raw)).convert("RGB"); w, h = img.size; s = min(1.0, 1024 / max(w, h))
    if s < 1: img = img.resize((round(w * s), round(h * s)), Image.BILINEAR)
    b = io.BytesIO(); img.save(b, "JPEG", quality=82); return b.getvalue()
def q82(raw):
    b = io.BytesIO(); Image.open(io.BytesIO(raw)).convert("RGB").save(b, "JPEG", quality=82); return b.getvalue()
def padded(raw, fill=0.5):   # copied from armslength_reconstruction.py
    img = Image.open(io.BytesIO(raw)).convert("RGB"); w, h = img.size
    bg = img.resize((w, h), Image.BILINEAR).filter(ImageFilter.GaussianBlur(max(w, h) * 0.04))
    sw, sh = max(1, int(w * fill)), max(1, int(h * fill))
    bg.paste(img.resize((sw, sh), Image.BILINEAR), ((w - sw) // 2, (h - sh) // 2))
    b = io.BytesIO(); bg.save(b, "JPEG", quality=82); return b.getvalue()
def listing(root): return [f"{root}/{c}/{f}" for c in NAMES for f in sorted(os.listdir(f"{root}/{c}")) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
man4 = list(csv.DictReader(open(f"{EXP}/v4/new_neg_manifest.csv"))); man8 = list(csv.DictReader(open(f"{V6}/a8_manifest.csv")))
sets = {"val": (listing(f"{WT}/data/val"), browser), "test": (listing(f"{WT}/data/test"), browser),
        "val_f50": (listing(f"{EXP}/data_frame50/val"), browser), "test_f50": (listing(f"{EXP}/data_frame50/test"), browser),
        "test_arm2": (listing(f"{WT}/data/test"), padded),
        "rit": ([f"{EXP}/v2/rit/images/{r['file']}" for r in csv.DictReader(open(f"{EXP}/v2/rit/manifest.csv"))], browser),
        "heldout": ([f"{WT}/{r['file']}" for r in man4 if r["split"] == "heldout"], q82),
        "indep": ([f"{WT}/{r['file']}" for r in man4 if r["split"] == "excluded_resembles_eval"], q82),
        "a8_heldout": ([f"{WT}/{r['file']}" for r in man8 if r["split"] == "heldout"], q82)}
json.dump({k: [os.path.relpath(p, WT) for p in v[0]] for k, v in sets.items()}, open(f"{V6}/views6_files.json", "w"))
out = {}
for k, (paths, enc) in sets.items():
    out[k] = np.array([run(prod._to_tensor(prod._views(Image.open(io.BytesIO(enc(open(p, "rb").read()))).convert("RGB"))[0])) for p in paths])
    print(k, out[k].shape, flush=True)
np.savez(f"{V6}/views6_{name}.npz", **out)

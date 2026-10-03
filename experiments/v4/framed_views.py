"""Per-photo probabilities for experiments/data_frame50/{val,test} (the model card's arm's-length
framing), browser-encoded, full frame + centre crop: deployed classifier, shipped gate, healthy-skin
gate (all TFLite, the deployed files). Writes framed_views.npz."""
import os, io, importlib.util
import numpy as np
from PIL import Image
V4 = os.path.dirname(os.path.abspath(__file__)); WT = os.path.dirname(os.path.dirname(V4)); SRV = os.path.expanduser("~/Downloads/wound-analyzer-vercel")
spec = importlib.util.spec_from_file_location("prod", f"{WT}/experiments/v2/production_snapshot/predict.py"); prod = importlib.util.module_from_spec(spec); spec.loader.exec_module(prod)
from ai_edge_litert.interpreter import Interpreter
def it(p): i = Interpreter(model_path=p); i.allocate_tensors(); return i
clf, ship, heal = it(f"{SRV}/models/wound_model.tflite"), it(f"{SRV}/models/out_of_scope_gate.tflite"), it(f"{SRV}/models/out_of_scope_gate_healthy.tflite")
NAMES = ["abrasion", "bruise", "burn_1st_degree", "burn_2nd_degree", "burn_3rd_degree", "cut", "out_of_scope"]
def browser(raw):
    img = Image.open(io.BytesIO(raw)).convert("RGB"); w, h = img.size; s = min(1.0, 1024 / max(w, h))
    if s < 1: img = img.resize((round(w * s), round(h * s)), Image.BILINEAR)
    b = io.BytesIO(); img.save(b, "JPEG", quality=82); return b.getvalue()
out = {}
for split in ("val", "test"):
    C, S, G, Y = [], [], [], []
    for k, c in enumerate(NAMES):
        d = f"{WT}/experiments/data_frame50/{split}/{c}"
        for f in sorted(os.listdir(d)):
            if not f.lower().endswith((".jpg", ".jpeg", ".png")): continue
            img = Image.open(io.BytesIO(browser(open(f"{d}/{f}", "rb").read()))).convert("RGB")
            xs = [prod._to_tensor(v) for v in prod._views(img)]
            C.append([prod._run(clf, x) for x in xs]); S.append([prod._run(ship, x) for x in xs]); G.append([prod._run(heal, x) for x in xs]); Y.append(k)
    out.update({f"{split}_clf": np.array(C), f"{split}_gate_shipped": np.array(S), f"{split}_gate_g4new": np.array(G), f"{split}_y": np.array(Y)})
    print(split, len(Y), flush=True)
np.savez(f"{V4}/framed_views.npz", **out)

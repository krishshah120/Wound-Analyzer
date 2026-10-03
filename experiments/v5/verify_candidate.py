"""v5: run the candidate server code (production_mean_070/api/predict.py with the live model files) on data/test,
the 401 held-out and 1,008 independent healthy photos, encoded as in clf_views.py; compare with eval_v5bc's
labels for candidate (mean, sum, 0.70). -> verify_candidate.json"""
import os, io, sys, csv, json, importlib.util
import numpy as np
from PIL import Image
V5 = os.path.dirname(os.path.abspath(__file__)); WT = os.path.dirname(os.path.dirname(V5))
spec = importlib.util.spec_from_file_location("cand", f"{V5}/production_mean_070/api/predict.py"); cand = importlib.util.module_from_spec(spec); spec.loader.exec_module(cand)
assert cand.CONFIDENCE_THRESHOLD == 0.70 and cand.CROP_RETRY is False
NAMES = ["abrasion", "bruise", "burn_1st_degree", "burn_2nd_degree", "burn_3rd_degree", "cut", "out_of_scope"]
def browser(raw):
    img = Image.open(io.BytesIO(raw)).convert("RGB"); w, h = img.size; s = min(1.0, 1024 / max(w, h))
    if s < 1: img = img.resize((round(w * s), round(h * s)), Image.BILINEAR)
    b = io.BytesIO(); img.save(b, "JPEG", quality=82); return b.getvalue()
def q82(raw):
    b = io.BytesIO(); Image.open(io.BytesIO(raw)).convert("RGB").save(b, "JPEG", quality=82); return b.getvalue()
man = list(csv.DictReader(open(f"{WT}/experiments/v4/new_neg_manifest.csv")))
sets = {"test": ([f"{WT}/data/test/{c}/{f}" for c in NAMES for f in sorted(os.listdir(f"{WT}/data/test/{c}")) if f.lower().endswith((".jpg", ".jpeg", ".png"))], browser),
        "heldout": ([f"{WT}/{r['file']}" for r in man if r["split"] == "heldout"], q82),
        "indep": ([f"{WT}/{r['file']}" for r in man if r["split"] == "excluded_resembles_eval"], q82)}
sys.argv = ["x"]; import eval_v5bc as E
M = ["abrasion", "bruise", "possible_burn", "cut"]
out = {}
for k, (paths, enc) in sets.items():
    server = [cand._classify(enc(open(p, "rb").read()))[0] for p in paths]
    ev = ["unknown" if l < 0 else M[l] for l in E.labels(E.CLF["mean"][k], E.gated[k], "sum", 0.70)]
    diff = [(os.path.basename(p), a, b) for p, a, b in zip(paths, server, ev) if a != b]
    out[k] = dict(n=len(paths), match=len(paths) - len(diff), server_labelled=sum(l != "unknown" for l in server), eval_labelled=sum(l != "unknown" for l in ev), differences=diff)
    print(k, {kk: v for kk, v in out[k].items() if kk != "differences"}, diff[:10], flush=True)
json.dump(out, open(f"{V5}/verify_candidate.json", "w"), indent=1)

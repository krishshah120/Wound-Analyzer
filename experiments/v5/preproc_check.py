"""v5 diagnosis: the LIVE server code (~/Downloads/wound-analyzer-vercel/api/predict.py, _classify) on the
401 held-out + 1,008 independent normal-skin photos, browser-encoded (<= 1024 px, JPEG q82), as sent and
with one change at a time: rotated 90 degrees, mirrored (selfie camera), BILINEAR instead of NEAREST resize,
and CROP_RETRY=True (the 0.80 crop retry that was live until 2026-09-27). -> preproc_check.json"""
import os, io, csv, json, importlib.util
import numpy as np
from PIL import Image
V5 = os.path.dirname(os.path.abspath(__file__)); WT = os.path.dirname(os.path.dirname(V5))
spec = importlib.util.spec_from_file_location("live", os.path.expanduser("~/Downloads/wound-analyzer-vercel/api/predict.py"))
live = importlib.util.module_from_spec(spec); spec.loader.exec_module(live)
assert live.CROP_RETRY is False and live.CONFIDENCE_THRESHOLD == 0.75
rows = [r for r in csv.DictReader(open(f"{WT}/experiments/v4/new_neg_manifest.csv")) if r["split"] in ("heldout", "excluded_resembles_eval")]
def browser(img):
    img = img.convert("RGB"); w, h = img.size; s = min(1.0, 1024 / max(w, h))
    if s < 1: img = img.resize((round(w * s), round(h * s)), Image.BILINEAR)
    b = io.BytesIO(); img.save(b, "JPEG", quality=82); return b.getvalue()
nearest_to_tensor = live._to_tensor
def bilinear_to_tensor(img):
    arr = np.asarray(img.convert("RGB").resize((224, 224), Image.BILINEAR), dtype=np.float32)
    return (arr / 127.5 - 1.0)[None, ...]
VARIANTS = {"as_sent": lambda im: im, "rotated_90": lambda im: im.rotate(90, expand=True), "mirrored": lambda im: im.transpose(Image.FLIP_LEFT_RIGHT),
            "bilinear_resize": lambda im: im, "crop_retry_on": lambda im: im}
res = {}
for v, f in VARIANTS.items():
    live._to_tensor = bilinear_to_tensor if v == "bilinear_resize" else nearest_to_tensor
    live.CROP_RETRY = v == "crop_retry_on"
    labs = [live._classify(browser(f(Image.open(f"{WT}/{r['file']}"))))[0] for r in rows]
    t = {}
    for split in ("heldout", "excluded_resembles_eval"):
        for src in sorted({r["source"] for r in rows}):
            m = [l for l, r in zip(labs, rows) if r["split"] == split and r["source"] == src]
            t[f"{split}/{src}"] = f"{sum(l != 'unknown' for l in m)}/{len(m)}"
    t["all_labelled"] = sum(l != "unknown" for l in labs); t["as_possible_burn"] = sum(l == "possible_burn" for l in labs)
    res[v] = t; print(v, json.dumps(t), flush=True)
json.dump(res, open(f"{V5}/preproc_check.json", "w"), indent=1)

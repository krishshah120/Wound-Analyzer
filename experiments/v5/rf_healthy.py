"""v5 Roboflow with faces (PLAN.md): the 401 held-out + 1,008 independent normal-skin photos (public dataset
photos). Step 1: exclude any that resemble a Roboflow train/valid photo (same rule as rf/eligibility.py).
Step 2: run the Roboflow workflow on the rest with the bytes the MRC site sends (resumable).
-> rf_healthy_eligible.csv, rf_healthy_raw.jsonl"""
import os, io, sys, csv, json, time, zipfile
import numpy as np
from PIL import Image
V5 = os.path.dirname(os.path.abspath(__file__)); WT = os.path.dirname(os.path.dirname(V5))
sys.path.insert(0, f"{WT}/src"); os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import roboflow_client as rc
rows = [r for r in csv.DictReader(open(f"{WT}/experiments/v4/new_neg_manifest.csv")) if r["split"] in ("heldout", "excluded_resembles_eval")]
ELIG = f"{V5}/rf_healthy_eligible.csv"
if not os.path.exists(ELIG):
    import train_model as t
    z = zipfile.ZipFile(os.path.expanduser("~/Downloads/wound.v1i.yolov8.zip"))
    rf_imgs = [n for n in z.namelist() if n.split("/")[0] in ("train", "valid") and "/images/" in n]
    tmp = f"{V5}/_tmp"; os.makedirs(tmp, exist_ok=True); rp, cp = [], []
    for i, n in enumerate(rf_imgs):
        p = f"{tmp}/rf_{i}.png"; Image.open(io.BytesIO(z.read(n))).convert("RGB").resize((224, 224)).save(p); rp.append(p)
    for i, r in enumerate(rows):
        q = f"{tmp}/c_{i}.png"; Image.open(f"{WT}/{r['file']}").convert("RGB").resize((224, 224)).save(q); cp.append(q)
    sim = t.cross_similarity(t.view_features(cp), t.view_features(rp)).max(axis=1)
    rh = np.array([t.difference_hash(p) for p in rp])
    dist = np.array([min((t.difference_hash(p) != rh).sum(1).min(), (t.difference_hash(p, mirrored=True) != rh).sum(1).min()) for p in cp])
    for p in rp + cp: os.remove(p)
    os.rmdir(tmp)
    with open(ELIG, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["file", "source", "split", "max_sim_roboflow_train_valid", "min_hash_dist", "eligible"])
        for r, s, d in zip(rows, sim, dist):
            w.writerow([r["file"], r["source"], r["split"], f"{s:.3f}", int(d), "no" if (s >= 0.75 or d <= t.DUPLICATE_HASH_DISTANCE) else "yes"])
el = [r for r in csv.DictReader(open(ELIG)) if r["eligible"] == "yes"]
print(f"{len(el)}/{len(rows)} clean for both models", flush=True)
key = rc.load_api_key(); OUT = f"{V5}/rf_healthy_raw.jsonl"
done = {json.loads(l)["path"] for l in open(OUT)} if os.path.exists(OUT) else set()
def browser(path):
    img = Image.open(path).convert("RGB"); w, h = img.size; s = min(1.0, 1024 / max(w, h))
    if s < 1: img = img.resize((round(w * s), round(h * s)), Image.BILINEAR)
    b = io.BytesIO(); img.save(b, "JPEG", quality=82); return b.getvalue()
with open(OUT, "a") as f:
    for i, r in enumerate(x for x in el if x["file"] not in done):
        raw = browser(f"{WT}/{r['file']}"); t0 = time.perf_counter()
        try:
            out = rc.run_wound_workflow(raw, api_key=key)
            rec = dict(path=r["file"], latency_s=round(time.perf_counter() - t0, 3), detections=rc.find_detections(out[0]))
        except rc.RoboflowAuthError:
            raise
        except rc.RoboflowError as e:
            rec = dict(path=r["file"], error=f"{type(e).__name__}: {str(e)[:200]}")
        f.write(json.dumps(rec) + "\n"); f.flush()
        if i % 100 == 0: print(i + 1, flush=True)
print("done")

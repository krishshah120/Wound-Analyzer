"""Roboflow comparison, step 0 (no model access needed): which evaluation photos are clean for BOTH
models? A photo is excluded if it resembles (difference hash <= 6 with mirror, or 12-view similarity
>= 0.75 - the RIT rule) any photo in the Roboflow model's TRAIN or VALID split (training / model
selection of wound-ebsdw-4atst/1), or is itself one of those Roboflow photos. Our own models' training
data is already excluded from val/test/RIT by construction (x6_validate.py, build_rit.py).
Candidates: data/val, data/test, the RIT. Writes experiments/rf/eligible.csv and prints counts per
population and category (shared categories: abrasion, bruise, cut, generic burn)."""
import os, io, sys, csv, zipfile, collections
import numpy as np
from PIL import Image
RF = os.path.dirname(os.path.abspath(__file__)); WT = os.path.dirname(os.path.dirname(RF)); EXP = f"{WT}/experiments"
sys.path.insert(0, f"{WT}/src"); os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import train_model as t
NAMES = ["abrasion", "bruise", "burn_1st_degree", "burn_2nd_degree", "burn_3rd_degree", "cut", "out_of_scope"]
SHARED = {"abrasion": "abrasion", "bruise": "bruise", "cut": "cut", "burn_1st_degree": "burn", "burn_2nd_degree": "burn", "burn_3rd_degree": "burn",
          "burn_degree_unknown": "burn", "out_of_scope": "non_wound", "healthy_skin": "non_wound", "chronic_wound_out_of_scope": "chronic_wound"}
# Roboflow train + valid images (training and model selection of the Roboflow model)
z = zipfile.ZipFile(os.path.expanduser("~/Downloads/wound.v1i.yolov8.zip"))
rf_imgs = [n for n in z.namelist() if n.split("/")[0] in ("train", "valid") and "/images/" in n]
tmp = f"{RF}/_tmp"; os.makedirs(tmp, exist_ok=True)
rp = []
for i, n in enumerate(rf_imgs):
    p = f"{tmp}/rf_{i}.png"; Image.open(io.BytesIO(z.read(n))).convert("RGB").resize((224, 224)).save(p); rp.append(p)
# candidates
cands = []   # (path, population, label, group)
for split in ("val", "test"):
    g = {r["filename"]: r["source_group"] for r in csv.DictReader(open(f"{WT}/data/split_manifest.csv")) if r["split"] == split}
    for c in NAMES:
        for f in sorted(os.listdir(f"{WT}/data/{split}/{c}")):
            if f.lower().endswith((".jpg", ".jpeg", ".png")): cands.append((f"{WT}/data/{split}/{c}/{f}", split, c, g.get(f, "solo_" + f)))
for r in csv.DictReader(open(f"{EXP}/v2/rit/manifest.csv")):
    pop = "rit_" + {"lower_limb_feet": "healthy_feet", "lower_limb_wounds": "chronic_wounds"}.get(r["source"], "roboflow_" + r["original_name"].split("/")[0])
    cands.append((f"{EXP}/v2/rit/images/{r['file']}", pop, r["label"], "rit_" + r["dup_group"]))
cache = f"{EXP}/v3/ebis_ref_extra_features.npz"; C = np.load(cache); rit_feat = dict(zip(C["paths"], C["f"]))
need = [p for p, pop, _, _ in cands if p not in rit_feat]
cp = []
for i, p in enumerate(need):
    q = f"{tmp}/c_{i}.png"; Image.open(p).convert("RGB").resize((224, 224)).save(q); cp.append(q)
nf = dict(zip(need, t.view_features(cp))); rff = t.view_features(rp)
cf = np.stack([rit_feat[p] if p in rit_feat else nf[p] for p, *_ in cands])
sim = t.cross_similarity(cf, rff).max(axis=1)
rh = np.array([t.difference_hash(p) for p in rp])
def hd(p):
    h, hm = t.difference_hash(p), t.difference_hash(p, mirrored=True)
    return min((h != rh).sum(1).min(), (hm != rh).sum(1).min())
dist = np.array([hd(p) for p, *_ in cands])
for p in rp + cp: os.remove(p)
os.rmdir(tmp)
contaminated = (sim >= 0.75) | (dist <= t.DUPLICATE_HASH_DISTANCE) | np.array([pop in ("rit_roboflow_train", "rit_roboflow_valid") for _, pop, _, _ in cands])
with open(f"{RF}/eligible.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["path", "population", "label", "shared_category", "dup_group", "max_sim_roboflow_train_valid", "min_hash_dist", "eligible"])
    for (p, pop, lab, g), s, d, c in zip(cands, sim, dist, contaminated):
        w.writerow([os.path.relpath(p, WT), pop, lab, SHARED.get(lab, lab), g, f"{s:.3f}", int(d), "no" if c else "yes"])
tab = collections.defaultdict(lambda: [0, 0])
for (p, pop, lab, g), c in zip(cands, contaminated):
    k = (pop if not pop.startswith("rit_roboflow") else "rit_roboflow", SHARED.get(lab, lab)); tab[k][0] += 1; tab[k][1] += int(not c)
print(f"{'population':22s} {'category':14s} {'total':>6s} {'clean for both':>15s}")
for k in sorted(tab): print(f"{k[0]:22s} {k[1]:14s} {tab[k][0]:6d} {tab[k][1]:15d}")

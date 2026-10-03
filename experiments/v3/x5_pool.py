"""X5 step 1 (PLAN.md): hard-negative pool = Lower Limb 'Nomal' healthy-foot photos that are NOT in the
RIT, minus any within similarity 0.75 / difference hash 6 of ANY RIT photo or any data/val or data/test
photo. Reads the original download (never modified). Writes experiments/v3/hn_pool/*.jpg (224 px JPEG,
prepared like other training photos) and hn_pool_manifest.csv; builds experiments/v3/train_hn (links: data/train + pool as out_of_scope)."""
import os, io, sys, csv, zipfile, hashlib
import numpy as np
from PIL import Image
V3 = os.path.dirname(os.path.abspath(__file__)); WT = os.path.dirname(os.path.dirname(V3)); V2 = f"{WT}/experiments/v2"
sys.path.insert(0, f"{WT}/src"); os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import train_model as t
POOL, TRAIN_HN = f"{V3}/hn_pool", f"{V3}/train_hn"
for d in (POOL, TRAIN_HN):
    if os.path.exists(d): raise SystemExit(f"{d} exists - built once")
LL_ZIP = os.path.expanduser("~/Downloads/Lower Limb and Feet Wound Image Dataset for Medica.zip")
ll = zipfile.ZipFile(io.BytesIO(zipfile.ZipFile(LL_ZIP).read("Lower Limb and Feet Wound Image Dataset for Medica/Wound Image Dataset.zip")))
rit = list(csv.DictReader(open(f"{V2}/rit/manifest.csv")))
rit_names = {r["original_name"] for r in rit}
normals = sorted(x for x in ll.namelist() if x.startswith("Nomal/") and not x.endswith("/"))
cand = [n for n in normals if n not in rit_names]
print(f"healthy-foot photos {len(normals)}; in RIT {len(normals) - len(cand)}; candidates {len(cand)}", flush=True)
tmp = f"{V3}/_tmp_pool"; os.makedirs(tmp)
cp = []
for i, n in enumerate(cand):
    p = f"{tmp}/{i}.png"; Image.open(io.BytesIO(ll.read(n))).convert("RGB").resize((224, 224)).save(p); cp.append(p)
rp = []
for i, r in enumerate(rit):
    p = f"{tmp}/rit_{i}.png"; Image.open(f"{V2}/rit/images/{r['file']}").convert("RGB").resize((224, 224)).save(p); rp.append(p)
ev = [f"{WT}/data/{s}/{c}/{f}" for s in ("val", "test") for c in sorted(os.listdir(f"{WT}/data/{s}")) for f in sorted(os.listdir(f"{WT}/data/{s}/{c}"))
      if f.lower().endswith((".jpg", ".jpeg", ".png"))]
ref = rp + ev
cf, rf = t.view_features(cp), t.view_features(ref)
sim = t.cross_similarity(cf, rf).max(axis=1)
ch = np.array([t.difference_hash(p) for p in cp]); chm = np.array([t.difference_hash(p, mirrored=True) for p in cp])
rh = np.array([t.difference_hash(p) for p in ref])
hd = np.array([min((ch[i] != rh).sum(1).min(), (chm[i] != rh).sum(1).min()) for i in range(len(cp))])
keep = (sim < 0.75) & (hd > t.DUPLICATE_HASH_DISTANCE)
print(f"dropped as resembling a RIT/val/test photo: {int((~keep).sum())}; kept {int(keep.sum())}; "
      f"max similarity of kept {sim[keep].max():.3f}", flush=True)
for p in cp + rp: os.remove(p)
os.rmdir(tmp)
os.makedirs(POOL); rows = []
for i, n in enumerate(cand):
    b = ll.read(n); sha = hashlib.sha256(b).hexdigest(); fn = f"hn_feet_{sha[:16]}.jpg"
    rows.append([fn, sha, n, f"{sim[i]:.3f}", int(hd[i]), "yes" if keep[i] else "no"])
    # prepared exactly like every other training photo (src/collate_extra_data.py): RGB, resize to 224, JPEG
    if keep[i]: Image.open(io.BytesIO(b)).convert("RGB").resize((224, 224)).save(f"{POOL}/{fn}", "JPEG")
with open(f"{V3}/hn_pool_manifest.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["file", "sha256", "original_name", "max_similarity_to_rit_val_test", "min_hash_distance", "in_pool"]); w.writerows(rows)
for c in sorted(os.listdir(f"{WT}/data/train")):
    os.makedirs(f"{TRAIN_HN}/{c}")
    for f in sorted(os.listdir(f"{WT}/data/train/{c}")): os.symlink(os.path.realpath(f"{WT}/data/train/{c}/{f}"), f"{TRAIN_HN}/{c}/{f}")
for r in rows:
    if r[5] == "yes": os.symlink(f"{POOL}/{r[0]}", f"{TRAIN_HN}/out_of_scope/{r[0]}")
print({c: len(os.listdir(f"{TRAIN_HN}/{c}")) for c in sorted(os.listdir(TRAIN_HN))})

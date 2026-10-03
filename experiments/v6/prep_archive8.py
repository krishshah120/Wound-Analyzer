"""v6 data (PLAN.md): archive (8).zip "Skin Issues Dataset/Skin v2" faces / facial skin as "not a wound".
Same cleaning as v4/prep_negatives.py: 224 px JPEG copies; drop near-duplicates (difference hash <= 6 with mirror,
or 12-view similarity >= 0.75) of any RIT, data/val, data/test, v4 held-out or v4 independent photo; duplicate
groups among the new photos (hash or 0.85); one kept per group; split by group, seed 0, 80% train / 20% held out;
training part capped at 2,000 (random, seed 0). Zip never modified. -> a8_manifest.csv, a8/<sha16>.jpg"""
import os, io, sys, csv, zipfile, hashlib, random, collections
import numpy as np
from PIL import Image
V6 = os.path.dirname(os.path.abspath(__file__)); WT = os.path.dirname(os.path.dirname(V6)); EXP = f"{WT}/experiments"
sys.path.insert(0, f"{WT}/src"); os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import train_model as t
OUT = f"{V6}/a8"
if os.path.exists(OUT): raise SystemExit(f"{OUT} exists - built once")
os.makedirs(OUT)
zf = zipfile.ZipFile(os.path.expanduser("~/Downloads/archive (8).zip")); items = []
for n in sorted(x for x in zf.namelist() if x.startswith("Skin Issues Dataset/Skin v2/") and x.lower().endswith((".jpg", ".jpeg", ".png"))):
    b = zf.read(n); sha = hashlib.sha256(b).hexdigest(); p = f"{OUT}/{sha[:16]}.jpg"
    if os.path.exists(p): continue
    try: Image.open(io.BytesIO(b)).convert("RGB").resize((224, 224)).save(p, "JPEG")
    except Exception as e: print("unreadable, skipped:", n, e); continue
    items.append((n, sha, p))
print("extracted", len(items), flush=True)
C = np.load(f"{EXP}/v3/ebis_ref_extra_features.npz"); rit_paths = list(C["paths"]); rit_f = C["f"]
ev = [f"{WT}/data/{s}/{c}/{f}" for s in ("val", "test") for c in sorted(os.listdir(f"{WT}/data/{s}")) for f in sorted(os.listdir(f"{WT}/data/{s}/{c}")) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
ev += [f"{WT}/{r['file']}" for r in csv.DictReader(open(f"{EXP}/v4/new_neg_manifest.csv")) if r["split"] in ("heldout", "excluded_resembles_eval")]
tmp = f"{V6}/_tmp"; os.makedirs(tmp); small = []
for i, p in enumerate(ev):
    q = f"{tmp}/{i}.png"; Image.open(p).convert("RGB").resize((224, 224)).save(q); small.append(q)
ev_f = t.view_features(small)
for q in small: os.remove(q)
os.rmdir(tmp)
nf = t.view_features([i[2] for i in items]); print("features done", flush=True)
ref_paths = rit_paths + ev; sim = t.cross_similarity(nf, np.concatenate([rit_f, ev_f])).max(axis=1)
nh = np.array([t.difference_hash(i[2]) for i in items]); nhm = np.array([t.difference_hash(i[2], mirrored=True) for i in items])
rh = np.array([t.difference_hash(p) for p in ref_paths])
hd = np.array([min((nh[k] != rh).sum(1).min(), (nhm[k] != rh).sum(1).min()) for k in range(len(items))])
clean = (sim < 0.75) & (hd > t.DUPLICATE_HASH_DISTANCE)
print(f"dropped as resembling an evaluation photo: {int((~clean).sum())} of {len(items)}", flush=True)
ss = t.cross_similarity(nf, nf); np.fill_diagonal(ss, 0); parent = list(range(len(items)))
def find(x):
    while parent[x] != x: parent[x] = parent[parent[x]]; x = parent[x]
    return x
for a in range(len(items)):
    d = np.minimum((nh[a + 1:] != nh[a]).sum(1), (nhm[a + 1:] != nh[a]).sum(1))
    for b in np.nonzero((d <= t.DUPLICATE_HASH_DISTANCE) | (ss[a, a + 1:] >= t.SAME_PHOTO_SIMILARITY))[0]: parent[find(a)] = find(a + 1 + int(b))
grp = [find(k) for k in range(len(items))]
gs = sorted({grp[k] for k in range(len(items)) if clean[k]}); random.Random(0).shuffle(gs); cut = int(round(0.8 * len(gs)))
split = {g: ("train" if i < cut else "heldout") for i, g in enumerate(gs)}
first = {}
for k in range(len(items)):
    if clean[k]: first.setdefault(grp[k], k)
rows = []
for k, (n, sha, p) in enumerate(items):
    s = "excluded_resembles_eval" if not clean[k] else ("excluded_duplicate" if first[grp[k]] != k else split[grp[k]])
    rows.append([os.path.relpath(p, WT), "a8_skin_v2_" + n.split("/")[2].replace(" ", "_"), n, sha, "not stated", f"g{grp[k]}", f"{sim[k]:.3f}", int(hd[k]), s])
tr = [i for i, r in enumerate(rows) if r[8] == "train"]; random.Random(0).shuffle(tr)
for i in tr[2000:]: rows[i][8] = "train_over_cap"
with open(f"{V6}/a8_manifest.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["file", "source", "original_name", "sha256_original", "licence", "dup_group", "max_sim_eval", "min_hash_dist", "split"]); w.writerows(rows)
print("split:", collections.Counter(r[8] for r in rows))

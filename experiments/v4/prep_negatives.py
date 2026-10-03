"""v4 data (PLAN.md): new healthy-skin / not-a-wound photos from sources other than the RIT feet.
Reads the zips (never modified); writes 224 px JPEG copies (prepared like src/collate_extra_data.py)
to experiments/v4/new_neg/<source>/, manifest new_neg_manifest.csv, and the training folder
experiments/v4/train_v4g (links: data/train + the 80% training part as out_of_scope)."""
import os, io, sys, csv, zipfile, hashlib, random, collections
import numpy as np
from PIL import Image
V4 = os.path.dirname(os.path.abspath(__file__)); WT = os.path.dirname(os.path.dirname(V4)); EXP = f"{WT}/experiments"
sys.path.insert(0, f"{WT}/src"); os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import train_model as t
OUT, TRAIN = f"{V4}/new_neg", f"{V4}/train_v4g"
for d in (OUT, TRAIN):
    if os.path.exists(d): raise SystemExit(f"{d} exists - built once")
D = os.path.expanduser("~/Downloads")
SOURCES = [("ghost_normal_skin", "ghost.v2i.folder.zip", lambda n: "/normal skin/" in n, "CC BY 4.0 (Roboflow Universe imagesnormalskin/ghost-1iput)"),
           ("oily_dry_skin_types", "archive (10).zip", lambda n: n.startswith("Oily-Dry-Skin-Types/") and any(f"/{c}/" in n for c in ("normal", "oily", "dry")), "not stated"),
           ("skindisease_unknown_normal", "archive (9).zip", lambda n: "/Unknown_Normal/" in n, "not stated")]
items = []   # (source, original name, sha256 of original, path of 224 copy, licence)
for src, z, keep, lic in SOURCES:
    zf = zipfile.ZipFile(f"{D}/{z}"); os.makedirs(f"{OUT}/{src}")
    for n in sorted(x for x in zf.namelist() if keep(x) and x.lower().endswith((".jpg", ".jpeg", ".png"))):
        b = zf.read(n); sha = hashlib.sha256(b).hexdigest(); p = f"{OUT}/{src}/{sha[:16]}.jpg"
        if os.path.exists(p): continue                                   # byte-identical copy within the source
        try:
            Image.open(io.BytesIO(b)).convert("RGB").resize((224, 224)).save(p, "JPEG")
        except Exception as e:
            print("unreadable, skipped:", src, n, e); continue
        items.append((src, n, sha, p, lic))
print("extracted:", collections.Counter(i[0] for i in items), flush=True)
# references that must stay clean: RIT (+ pool, cached features) and data/val, data/test
C = np.load(f"{EXP}/v3/ebis_ref_extra_features.npz"); rit_paths = list(C["paths"]); rit_f = C["f"]
ev = [f"{WT}/data/{s}/{c}/{f}" for s in ("val", "test") for c in sorted(os.listdir(f"{WT}/data/{s}")) for f in sorted(os.listdir(f"{WT}/data/{s}/{c}")) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
tmp = f"{V4}/_tmp"; os.makedirs(tmp)
ev_small = []
for i, p in enumerate(ev):
    q = f"{tmp}/{i}.png"; Image.open(p).convert("RGB").resize((224, 224)).save(q); ev_small.append(q)
ev_f = t.view_features(ev_small)
for q in ev_small: os.remove(q)
os.rmdir(tmp)
nf = t.view_features([i[3] for i in items])
ref_f = np.concatenate([rit_f, ev_f]); ref_paths = rit_paths + ev
sim = t.cross_similarity(nf, ref_f).max(axis=1)
nh = np.array([t.difference_hash(i[3]) for i in items]); nhm = np.array([t.difference_hash(i[3], mirrored=True) for i in items])
rh = np.array([t.difference_hash(p) for p in ref_paths])
hd = np.array([min((nh[k] != rh).sum(1).min(), (nhm[k] != rh).sum(1).min()) for k in range(len(items))])
clean = (sim < 0.75) & (hd > t.DUPLICATE_HASH_DISTANCE)
print(f"dropped as resembling a RIT/val/test photo: {int((~clean).sum())} of {len(items)}", flush=True)
# duplicate groups among the new photos (same rule as the original split: hash or 0.85)
ss = t.cross_similarity(nf, nf); np.fill_diagonal(ss, 0); parent = list(range(len(items)))
def find(x):
    while parent[x] != x: parent[x] = parent[parent[x]]; x = parent[x]
    return x
for a in range(len(items)):
    d = np.minimum((nh[a + 1:] != nh[a]).sum(1), (nhm[a + 1:] != nh[a]).sum(1))
    for b in np.nonzero((d <= t.DUPLICATE_HASH_DISTANCE) | (ss[a, a + 1:] >= t.SAME_PHOTO_SIMILARITY))[0]: parent[find(a)] = find(a + 1 + int(b))
grp = [find(k) for k in range(len(items))]
split = {}
for src in {i[0] for i in items}:
    gs = sorted({grp[k] for k in range(len(items)) if items[k][0] == src and clean[k]}); random.Random(0).shuffle(gs)
    cut = int(round(0.8 * len(gs)))
    for g in gs[:cut]: split[(src, g)] = "train"
    for g in gs[cut:]: split[(src, g)] = "heldout"
rows = []; first_of_group = {}
for k in range(len(items)):
    if clean[k]: first_of_group.setdefault((items[k][0], grp[k]), k)      # one photo kept per duplicate group
for k, (src, n, sha, p, lic) in enumerate(items):
    if not clean[k]: s = "excluded_resembles_eval"
    elif first_of_group[(src, grp[k])] != k: s = "excluded_duplicate"
    else: s = split[(src, grp[k])]
    rows.append([os.path.relpath(p, WT), src, n, sha, lic, f"g{grp[k]}", f"{sim[k]:.3f}", int(hd[k]), s])
with open(f"{V4}/new_neg_manifest.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["file", "source", "original_name", "sha256_original", "licence", "dup_group", "max_sim_rit_val_test", "min_hash_dist", "split"]); w.writerows(rows)
print("split:", collections.Counter((r[1], r[8]) for r in rows))
print("kept (train + heldout):", sum(r[8] in ("train", "heldout") for r in rows))
for c in sorted(os.listdir(f"{WT}/data/train")):
    os.makedirs(f"{TRAIN}/{c}")
    for f in sorted(os.listdir(f"{WT}/data/train/{c}")): os.symlink(os.path.realpath(f"{WT}/data/train/{c}/{f}"), f"{TRAIN}/{c}/{f}")
for r in rows:
    if r[8] == "train": os.symlink(f"{WT}/{r[0]}", f"{TRAIN}/out_of_scope/v4_{r[1]}_{os.path.basename(r[0])}")
print({c: len(os.listdir(f"{TRAIN}/{c}")) for c in sorted(os.listdir(TRAIN))})

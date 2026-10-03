"""X3 step 1 (PLAN.md): find OBJECTIVE training-label conflicts - the same picture carrying different
labels - without any model judging medical content. Reads only; moves or edits nothing.
  T1 exact: identical sha256, or identical decoded pixels
  T2 verified near-duplicate: difference hash <= 4 bits (mirror allowed) AND 64x64 grayscale pixel
     correlation >= 0.95 (mirror allowed)
  T3 caution: flagged_for_review.csv similarity >= 0.87, not T1/T2;  T4: the other flags
Conflict = a T1/T2 group containing a b84_train photo whose members carry different labels
(6 wound classes + out_of_scope; every out-of-scope collection photo counts as out_of_scope).
Writes experiments/v3/x3_label_tiers.csv and x3_quarantine.txt (b84_train paths to exclude)."""
import os, csv, hashlib, collections
import numpy as np
from PIL import Image
from common import V3, WT, NAMES
V2 = f"{WT}/experiments/v2"
IMG = (".jpg", ".jpeg", ".png")

items = []   # (path as listed, real path, label, where)
def add_class_tree(root, where):
    for c in sorted(os.listdir(root)):
        if os.path.isdir(f"{root}/{c}"):
            for f in sorted(os.listdir(f"{root}/{c}")):
                if f.lower().endswith(IMG): items.append((f"{root}/{c}/{f}", os.path.realpath(f"{root}/{c}/{f}"), c, where))
add_class_tree(f"{V2}/b84_train", "b84_train")
for s in ("train", "val", "test"): add_class_tree(f"{WT}/data/{s}", s)
add_class_tree(f"{WT}/data/extra_dataset", "extra")
for g in sorted(os.listdir(f"{WT}/data/ood_dataset")):
    if os.path.isdir(f"{WT}/data/ood_dataset/{g}"):
        for f in sorted(os.listdir(f"{WT}/data/ood_dataset/{g}")):
            if f.lower().endswith(IMG): items.append((f"{WT}/data/ood_dataset/{g}/{f}", os.path.realpath(f"{WT}/data/ood_dataset/{g}/{f}"), "out_of_scope", f"ood:{g}"))
real = sorted({r for _, r, _, _ in items}); ri = {r: i for i, r in enumerate(real)}
print("listed", len(items), "unique files", len(real), flush=True)

sha, pix, dh, dhm, sm, smm = [], [], [], [], [], []
for r in real:
    b = open(r, "rb").read(); sha.append(hashlib.sha256(b).hexdigest())
    with Image.open(r) as im:
        rgb = im.convert("RGB"); a = np.asarray(rgb); pix.append(hashlib.sha256(a.tobytes() + str(a.shape).encode()).hexdigest())
        g = rgb.convert("L")
        for img, H, S in ((g, dh, sm), (g.transpose(Image.FLIP_LEFT_RIGHT), dhm, smm)):
            p = np.asarray(img.resize((9, 8), Image.LANCZOS), dtype=np.int16); H.append((p[:, 1:] > p[:, :-1]).flatten())
            s = np.asarray(img.resize((64, 64), Image.BILINEAR), dtype=np.float32).flatten(); s -= s.mean(); S.append(s / (np.linalg.norm(s) + 1e-6))
dh, dhm, sm, smm = map(np.array, (dh, dhm, sm, smm))
n = len(real); parent = list(range(n))
def find(x):
    while parent[x] != x: parent[x] = parent[parent[x]]; x = parent[x]
    return x
def union(a, b): parent[find(a)] = find(b)
tier = {}
for key in (sha, pix):
    first = {}
    for i, k in enumerate(key):
        if k in first: union(i, first[k]); tier[(min(i, first[k]), max(i, first[k]))] = "T1"
        else: first[k] = i
packed, packedm = np.packbits(dh, 1), np.packbits(dhm, 1)
POP = np.array([bin(x).count("1") for x in range(256)], np.uint8)
t2 = 0
for i in range(n):
    d = np.minimum(POP[packed[i + 1:] ^ packed[i]].sum(1), POP[packedm[i + 1:] ^ packed[i]].sum(1))
    for j in np.nonzero(d <= 4)[0] + i + 1:
        corr = max(float(sm[i] @ sm[j]), float(smm[i] @ sm[j]))
        if corr >= 0.95:
            union(i, j); tier.setdefault((i, j), "T2"); t2 += 1
groups = collections.defaultdict(list)
for i in range(n): groups[find(i)].append(i)
labels_of = collections.defaultdict(set); where_of = collections.defaultdict(set)
for _, r, lab, w in items: labels_of[ri[r]].add(lab); where_of[ri[r]].add(w)
conflict_groups = []
for g, mem in groups.items():
    labs = set().union(*(labels_of[i] for i in mem))
    in_train = any("b84_train" in where_of[i] for i in mem)
    if len(labs) > 1 and in_train: conflict_groups.append(mem)
b84 = [(p, r, lab) for p, r, lab, w in items if w == "b84_train"]
quar = sorted({p for p, r, lab in b84 for mem in conflict_groups if ri[r] in mem})
open(f"{V3}/x3_quarantine.txt", "w").write("\n".join(os.path.relpath(p, WT) for p in quar) + ("\n" if quar else ""))
print(f"T1/T2 pairs: {len(tier)} (T2 checks passing: {t2}); groups with >1 member: {sum(len(m) > 1 for m in groups.values())}")
print(f"conflict groups touching b84_train: {len(conflict_groups)}; b84_train photos quarantined: {len(quar)}")
for mem in conflict_groups:
    desc = sorted({(lab, w) for p, r, lab, w in items if ri[r] in mem})
    tiers = sorted({tier.get((min(a, b), max(a, b))) for a in mem for b in mem if a < b} - {None})
    print("  ", tiers, desc[:6], "..." if len(desc) > 6 else "")
# tiers for the 319 flags
flags = list(csv.DictReader(open(f"{V2}/flagged_for_review.csv")))
qnames = {os.path.basename(p) for p in quar}
conf_names = {os.path.basename(p) for mem in conflict_groups for p, r, lab, w in items if ri[r] in mem}
rows = []
for f in flags:
    t = "T1/T2 conflict" if f["filename"] in conf_names else ("T3 caution" if float(f["similarity_to_out_of_scope"]) >= 0.87 else "T4 similar only")
    rows.append([f["filename"], f["label"], f["split"], f["similarity_to_out_of_scope"], t, "yes" if f["filename"] in qnames else "no"])
with open(f"{V3}/x3_label_tiers.csv", "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["filename", "label", "split", "similarity_to_out_of_scope", "tier", "quarantined_from_training"]); w.writerows(rows)
print("flag tiers:", collections.Counter((r[4], r[2]) for r in rows))
print("quarantined by class:", collections.Counter(os.path.basename(os.path.dirname(p)) for p in quar))

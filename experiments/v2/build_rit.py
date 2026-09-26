"""Step 3: build the Reserved Independent Test (RIT) exactly as declared in HANDOFF.md.
Writes experiments/v2/rit/{images/, manifest.csv}. Never used for training, thresholds or selection."""
import os, io, sys, csv, json, random, zipfile, hashlib, collections
import numpy as np
from PIL import Image
V2 = os.path.dirname(os.path.abspath(__file__)); WT = os.path.dirname(os.path.dirname(V2)); EXP = f"{WT}/experiments"
sys.path.insert(0, f"{WT}/src"); os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import train_model as t
OUT = f"{V2}/rit"
if os.path.exists(OUT): raise SystemExit(f"{OUT} exists - the RIT is frozen once built")
os.makedirs(f"{OUT}/images")
RF_ZIP = os.path.expanduser("~/Downloads/wound.v1i.yolov8.zip"); LL_ZIP = os.path.expanduser("~/Downloads/Lower Limb and Feet Wound Image Dataset for Medica.zip")
rf = zipfile.ZipFile(RF_ZIP)
ll = zipfile.ZipFile(io.BytesIO(zipfile.ZipFile(LL_ZIP).read("Lower Limb and Feet Wound Image Dataset for Medica/Wound Image Dataset.zip")))
ROLE = {"abrasion": ("abrasion", "in_scope"), "bruise": ("bruise", "in_scope"), "cut": ("cut", "in_scope"),
        "burn_ungraded": ("burn_degree_unknown", "burn_any"), "normal_skin": ("healthy_skin", "non_wound")}

cands = []   # dict(path, source, label, role, licence, box_frac, src_name)
audit = [a for a in csv.DictReader(open(f"{EXP}/new_candidates/audit.csv")) if a["source"] == "roboflow" and a["new"] == "1" and a["label"] in ROLE]
for a in audit:
    stem = os.path.basename(a["path"]).rsplit(".", 1)[0]; split, name = stem.split("_", 1)
    img_name = next(n for n in rf.namelist() if n.startswith(f"{split}/images/{name}."))
    boxes = [tuple(map(float, l.split()[1:5])) for l in rf.read(f"{split}/labels/{name}.txt").decode().splitlines() if l.strip()]
    frac = None
    if a["label"] != "normal_skin":
        m = np.zeros((200, 200), bool)
        for x, y, w, h in boxes: m[int((y - h / 2) * 200):int((y + h / 2) * 200) + 1, int((x - w / 2) * 200):int((x + w / 2) * 200) + 1] = True
        frac = float(m.mean())
    label, role = ROLE[a["label"]]
    cands.append(dict(bytes=rf.read(img_name), ext=img_name.rsplit(".", 1)[1], source="roboflow_wound_v1", label=label, role=role,
                      licence="CC BY 4.0", box_frac=frac, src_name=img_name))
normals = sorted(x for x in ll.namelist() if x.startswith("Nomal/") and not x.endswith("/"))
used = set(random.Random(42).sample(normals, 600))           # exactly the localizer's negatives
for n in normals:
    if n not in used:
        cands.append(dict(bytes=ll.read(n), ext=n.rsplit(".", 1)[1], source="lower_limb_feet", label="healthy_skin", role="non_wound",
                          licence="CC BY 4.0", box_frac=None, src_name=n))
for n in sorted(x for x in ll.namelist() if x.startswith("wound_main/") and not x.endswith("/")):
    cands.append(dict(bytes=ll.read(n), ext=n.rsplit(".", 1)[1], source="lower_limb_wounds", label="chronic_wound_out_of_scope", role="non_wound",
                      licence="CC BY 4.0", box_frac=None, src_name=n))
print("candidates:", collections.Counter((c["source"], c["label"]) for c in cands))

tmp = []
for i, c in enumerate(cands):     # 224 copies only for similarity checks
    p = f"{OUT}/images/_tmp_{i}.png"; Image.open(io.BytesIO(c["bytes"])).convert("RGB").resize((224, 224)).save(p); tmp.append(p)
def listdir(d): return [f"{d}/{f}" for f in sorted(os.listdir(d)) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
refs = [t.dataset_path_for(r) for r in t.load_real_images()]
refs += [p for c in sorted(os.listdir(t.EXTRA_DATASET_DIR)) if os.path.isdir(f"{t.EXTRA_DATASET_DIR}/{c}") for p in listdir(f"{t.EXTRA_DATASET_DIR}/{c}")]
refs += [p for g in sorted(os.listdir(t.OUT_OF_SCOPE_DATASET_DIR)) if os.path.isdir(f"{t.OUT_OF_SCOPE_DATASET_DIR}/{g}") for p in listdir(f"{t.OUT_OF_SCOPE_DATASET_DIR}/{g}")]
rf_ = np.load(f"{EXP}/new_candidates/features.npz")["ref"]; assert len(rf_) == len(refs)
cf = t.view_features(tmp)
sim = t.cross_similarity(cf, rf_).max(axis=1)
ch = np.array([t.difference_hash(p) for p in tmp]); chm = np.array([t.difference_hash(p, mirrored=True) for p in tmp])
rh = np.array([t.difference_hash(p) for p in refs])
hd = np.array([min((ch[i] != rh).sum(1).min(), (chm[i] != rh).sum(1).min()) for i in range(len(tmp))])
keep = np.nonzero((sim < 0.75) & (hd > t.DUPLICATE_HASH_DISTANCE))[0]
print("excluded as resembling a training/selection photo:", len(cands) - len(keep),
      dict(collections.Counter((cands[i]["source"], cands[i]["label"]) for i in range(len(cands)) if i not in set(keep))))
ks = t.cross_similarity(cf[keep], cf[keep]); np.fill_diagonal(ks, 0)
parent = list(range(len(keep)))
def find(x):
    while parent[x] != x: parent[x] = parent[parent[x]]; x = parent[x]
    return x
for a in range(len(keep)):
    d = np.minimum((ch[keep[a + 1:]] != ch[keep[a]]).sum(1), (chm[keep[a + 1:]] != ch[keep[a]]).sum(1))
    for b in np.nonzero((d <= t.DUPLICATE_HASH_DISTANCE) | (ks[a, a + 1:] >= t.SAME_PHOTO_SIMILARITY))[0]: parent[find(a)] = find(a + 1 + int(b))
rows = []
for a, i in enumerate(keep):
    c = cands[i]; sha = hashlib.sha256(c["bytes"]).hexdigest(); fn = f"{c['source']}_{sha[:16]}.{c['ext']}"
    open(f"{OUT}/images/{fn}", "wb").write(c["bytes"])
    rows.append([fn, sha, c["source"], c["label"], c["role"], c["licence"], f"g{find(a)}",
                 "" if c["box_frac"] is None else f"{c['box_frac']:.3f}", c["src_name"], f"{sim[i]:.3f}"])
for p in tmp: os.remove(p)
with open(f"{OUT}/manifest.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["file", "sha256", "source", "label", "eval_role", "licence", "dup_group", "box_area_fraction", "original_name", "max_similarity_to_training_or_selection"]); w.writerows(rows)
print("RIT frozen:", len(rows), "photos in", len({r[6] for r in rows}), "duplicate groups")
print(collections.Counter((r[2], r[3]) for r in rows))
fr = [float(r[7]) for r in rows if r[7]]
print("in-scope/burn box area fraction: <15%", sum(x < .15 for x in fr), "| 15-40%", sum(.15 <= x < .4 for x in fr), "| >=40%", sum(x >= .4 for x in fr))

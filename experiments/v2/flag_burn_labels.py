"""Step 3: flag (never relabel) wound photos whose label is suspect because they closely resemble a
photo in the out-of-scope collection (rashes, skin disease, normal skin). Reuses the audit's
features (new_candidates/features.npz, reference order = real wound + extra wound + out-of-scope).
Writes experiments/v2/flagged_for_review.csv for qualified (clinical) review."""
import os, sys, csv, collections
import numpy as np
V2 = os.path.dirname(os.path.abspath(__file__)); WT = os.path.dirname(os.path.dirname(V2))
sys.path.insert(0, f"{WT}/src"); os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import train_model as t
def listdir(d): return [f"{d}/{f}" for f in sorted(os.listdir(d)) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
real_rows = t.load_real_images()
real = [t.dataset_path_for(r) for r in real_rows]
extra = [p for c in sorted(os.listdir(t.EXTRA_DATASET_DIR)) if os.path.isdir(f"{t.EXTRA_DATASET_DIR}/{c}") for p in listdir(f"{t.EXTRA_DATASET_DIR}/{c}")]
ood = [p for g in sorted(os.listdir(t.OUT_OF_SCOPE_DATASET_DIR)) if os.path.isdir(f"{t.OUT_OF_SCOPE_DATASET_DIR}/{g}") for p in listdir(f"{t.OUT_OF_SCOPE_DATASET_DIR}/{g}")]
F = np.load(f"{WT}/experiments/new_candidates/features.npz")["ref"]
assert len(F) == len(real) + len(extra) + len(ood)
fr, fo = F[:len(real)], F[len(real) + len(extra):]
sim = t.cross_similarity(fr, fo)
split_of = {r["filename"]: r["split"] for r in csv.DictReader(open(f"{WT}/data/split_manifest.csv")) if r["class"] != "out_of_scope"}
rows, c = [], collections.Counter()
for i, r in enumerate(real_rows):
    j = int(sim[i].argmax()); s = float(sim[i, j])
    cls = t.class_name_for(r); split = split_of.get(r["filename"], "?")
    c[(cls, split, "flag" if s >= 0.80 else "ok")] += 1
    if s >= 0.80:
        rows.append([r["filename"], cls, split, f"{s:.3f}", os.path.relpath(ood[j], WT), os.path.basename(os.path.dirname(ood[j]))])
with open(f"{V2}/flagged_for_review.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["filename", "label", "split", "similarity_to_out_of_scope", "closest_out_of_scope_photo", "its_group"]); w.writerows(rows)
print(f"{'class':17s} {'split':6s} {'flagged':>8s} {'of':>5s}")
for cls in sorted({k[0] for k in c}):
    for split in ("train", "val", "test"):
        n = c[(cls, split, "flag")] + c[(cls, split, "ok")]
        if n: print(f"{cls:17s} {split:6s} {c[(cls, split, 'flag')]:8d} {n:5d}")
print("groups the flagged photos resemble:", collections.Counter(r[5] for r in rows).most_common(8))

"""X6 step 1: validate the saved predictions before combining them.
For the deployed model and every R_exact run: rows keyed by image file name (not position), class
order, probabilities vs logits vs labels, and whether any evaluated image (by sha256 AND decoded
pixels) is in the member's training data (b84_train) or its early-stopping set (b84_val).
Writes x6_validate.json."""
import os, json, hashlib, collections
import numpy as np
from PIL import Image
from common import V3, WT, RUNS, NAMES, SHIPPED, files
V2 = f"{WT}/experiments/v2"
def digests(p):
    b = open(p, "rb").read(); a = np.asarray(Image.open(p).convert("RGB"))
    return hashlib.sha256(b).hexdigest(), hashlib.sha256(a.tobytes() + str(a.shape).encode()).hexdigest()
def tree(root):
    return [f"{root}/{c}/{f}" for c in sorted(os.listdir(root)) if os.path.isdir(f"{root}/{c}") for f in sorted(os.listdir(f"{root}/{c}"))]
train = {d for p in tree(f"{V2}/b84_train") for d in digests(os.path.realpath(p))}
estop = {d for p in tree(f"{V2}/b84_val") for d in digests(os.path.realpath(p))}
out = {}
for split in ("val", "test"):
    fl = files(split)
    assert len(set(fl)) == len(fl), f"{split}: file names are not unique"
    paths = [f"{WT}/data/{split}/{c}/{f}" for c in NAMES for f in sorted(os.listdir(f"{WT}/data/{split}/{c}")) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    dg = [digests(p) for p in paths]
    in_train = [os.path.basename(p) for p, d in zip(paths, dg) if set(d) & train]
    in_estop = collections.Counter(p.split("/")[-2] for p, d in zip(paths, dg) if set(d) & estop)
    out[f"{split}_images"] = len(paths)
    out[f"{split}_in_member_training_data"] = in_train
    out[f"{split}_in_early_stopping_set_by_class"] = dict(in_estop)
runs = {"deployed": None} | {t: np.load(f"{RUNS}/{t}_probs.npz") for t in ("R_exact_a_s42", "R_exact_b_s42", "R_exact_s1", "R_exact_s2", "R_exact_s3")}
checks = {}
for tag, d in runs.items():
    c = {}
    for split in ("val", "test"):
        if tag == "deployed":
            p = SHIPPED[f"{split}_clf"]; names = NAMES; keys = files(split)   # saved by the same predict_directory order
        else:
            p = d[f"p_{split}"]; names = list(d["names"]); keys = [f.split("/")[-1] for f in d[f"f_{split}"]]
        c[f"{split}_class_order_matches"] = names == NAMES
        c[f"{split}_rows_keyed_by_name_match"] = keys == files(split)
        c[f"{split}_dtype_shape"] = f"{p.dtype} {p.shape}"
        c[f"{split}_kind"] = "probabilities" if (p >= 0).all() and np.allclose(p.sum(1), 1, atol=1e-4) else "NOT probabilities"
        c[f"{split}_min_max_rowsum"] = [float(p.min()), float(p.max()), float(p.sum(1).min()), float(p.sum(1).max())]
    checks[tag] = c
# deployed rows carry no names: compare with views_deployed (production TFLite run per named file, stored bytes)
V = np.load(f"{V3}/views_deployed.npz")
for split in ("val", "test"):
    checks["deployed"][f"{split}_max_abs_diff_vs_named_rerun"] = float(np.abs(SHIPPED[f"{split}_clf"] - V[f"{split}_raw_clf"][:, 0]).max())
out["checks"] = checks
out["all_ok"] = all(v for c in checks.values() for k, v in c.items() if k.endswith(("matches", "match"))) and \
                all(c[f"{s}_kind"] == "probabilities" for c in checks.values() for s in ("val", "test"))
json.dump(out, open(f"{V3}/x6_validate.json", "w"), indent=1)
print(json.dumps({k: v for k, v in out.items() if k != "checks"}, indent=1))
for t, c in checks.items(): print(t, {k: v for k, v in c.items() if "min_max" not in k})

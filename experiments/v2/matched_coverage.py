"""Step 5 rule MC: accuracy at matched coverage (deployed val coverage 46.2%), per candidate run,
with the shipped gate; guards; paired grouped bootstrap vs deployed. See HANDOFF.md."""
import os, sys, csv, glob, json, collections
import numpy as np
V2 = os.path.dirname(os.path.abspath(__file__)); WT = os.path.dirname(os.path.dirname(V2))
names = json.load(open(f"{WT}/models/class_names.json")); OOD = names.index("out_of_scope"); B3 = names.index("burn_3rd_degree")
W = [i for i in range(len(names)) if i != OOD]
S = np.load(f"{WT}/experiments/runs/round_o_shipped.npz")
y, gate = S["val_y"], S["val_gate"]; wound = y != OOD; n_w = int(wound.sum())
# val file order used by experiment9/predict_directory = class order, sorted filenames
files = [f for c in names for f in sorted(os.listdir(f"{WT}/data/val/{c}")) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
gmap = {r["filename"]: r["source_group"] for r in csv.DictReader(open(f"{WT}/data/split_manifest.csv")) if r["split"] == "val"}
groups = np.array([gmap[f] for f in files]); ug = np.unique(groups); gidx = {g: np.nonzero(groups == g)[0] for g in ug}
TARGET = 116   # deployed coverage on val: 116 of 251

def at_matched(p):
    conf = p[:, W].max(1); best = np.array(W)[p[:, W].argmax(1)]
    elig = (p.argmax(1) != OOD) & (gate[:, OOD] < 0.5)
    cand = np.sort(conf[elig & wound])[::-1]
    if len(cand) < TARGET: return None
    t = cand[TARGET - 1]                      # lowest threshold still answering >= 116 wound photos
    ans = elig & (conf >= t)
    correct = ans & wound & (best == y)
    return dict(threshold=float(t), answered=int((ans & wound).sum()), correct=int(correct.sum()),
                acc=float(correct.sum() / (ans & wound).sum()), ood=int((ans & ~wound).sum()),
                b3=int((ans & (y == B3) & (best != B3)).sum()), ans=ans, correct_vec=correct)

def aurc(p):
    conf = p[:, W].max(1); best = np.array(W)[p[:, W].argmax(1)]
    elig = (p.argmax(1) != OOD) & (gate[:, OOD] < 0.5)
    c = np.where(elig & wound, conf, -1.0)[wound]; r = (best != y)[wound].astype(float)
    order = np.argsort(-c); risks = np.cumsum(r[order]) / np.arange(1, len(r) + 1)
    return float(risks[: int((c >= 0).sum())].mean()) if (c >= 0).any() else float("nan")

dep = at_matched(S["val_clf"])
print(f"DEPLOYED: threshold {dep['threshold']:.3f} answered {dep['answered']} correct {dep['correct']} acc {dep['acc']:.3f} ood {dep['ood']}/94 b3 {dep['b3']} AURC {aurc(S['val_clf']):.3f}")
rng = np.random.default_rng(0)
boot_groups = [rng.choice(ug, size=len(ug), replace=True) for _ in range(2000)]
def paired_ci(m):
    diffs = []
    for bg in boot_groups:
        idx = np.concatenate([gidx[g] for g in bg])
        a1, a0 = (m["ans"] & wound)[idx], (dep["ans"] & wound)[idx]
        if a1.sum() and a0.sum(): diffs.append(m["correct_vec"][idx].sum() / a1.sum() - dep["correct_vec"][idx].sum() / a0.sum())
    return np.percentile(diffs, [2.5, 97.5])

runs = collections.defaultdict(list)
for f in sorted(glob.glob(f"{WT}/experiments/runs/*_s[0-9]*_probs.npz")):
    tag = os.path.basename(f)[:-10]; recipe = tag.rsplit("_s", 1)[0]
    d = np.load(f)
    if not (d["y_val"] == y).all(): print("skip (order mismatch)", tag); continue
    m = at_matched(d["p_val"])
    if m is None: print(f"  {tag}: cannot reach matched coverage"); continue
    m["aurc"] = aurc(d["p_val"]); m["ci"] = paired_ci(m); runs[recipe].append((tag, m))
print(f"\n{'recipe':24s} {'n':>2s} {'acc@46.2%':>10s} {'diff vs dep':>11s} {'ood/94':>7s} {'b3':>5s} {'AURC':>6s}  per-seed acc (paired 95% CI of diff)")
summary = {}
for recipe, ms in sorted(runs.items()):
    acc = np.mean([m["acc"] for _, m in ms]); ood = np.mean([m["ood"] for _, m in ms]); b3 = np.mean([m["b3"] for _, m in ms])
    qual = (acc - dep["acc"]) * 100 >= 3 and ood <= 16 and b3 <= 1 and len(ms) >= 3
    summary[recipe] = dict(n=len(ms), acc=acc, ood=ood, b3=b3, qualifies=bool(qual))
    per = ", ".join(f"{m['acc']:.3f} ({100*m['ci'][0]:+.1f},{100*m['ci'][1]:+.1f})" for _, m in ms)
    print(f"{recipe:24s} {len(ms):2d} {acc:10.3f} {100*(acc-dep['acc']):+10.1f}p {ood:7.1f} {b3:5.2f} {np.mean([m['aurc'] for _, m in ms]):6.3f}  {per}{'   <-- QUALIFIES' if qual else ''}")
json.dump(dict(deployed={k: v for k, v in dep.items() if k not in ('ans', 'correct_vec')}, recipes=summary), open(f"{V2}/matched_coverage.json", "w"), indent=1, default=float)

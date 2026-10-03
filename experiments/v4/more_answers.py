"""v4 step 1: what "more answers" costs, from saved predictions of the deployed models (no training).
Outputs: 6 categories (today) or "possible burn" (3 burn probabilities summed), at thresholds 0.60 /
0.50 / 0.40; production crop retry kept (crop still needs >= 0.80). Browser-encoded photos. Every
population reported separately. Writes more_answers.json."""
import sys, csv, json
import numpy as np
sys.path.insert(0, "../v3")
from common import NAMES, OOD, W, B1, B2, B3, V3, WT
D = np.load(f"{V3}/views_deployed.npz")
TO_M = np.array([2 if n.startswith("burn") else ["abrasion", "bruise", None, "cut", "out_of_scope"].index(n) if n != "cut" else 3 for n in NAMES])
TO_M = np.array([{"abrasion": 0, "bruise": 1, "cut": 3, "out_of_scope": 4}.get(n, 2) for n in NAMES])
def merged(p):
    out = np.zeros(p.shape[:-1] + (5,)); [out.__setitem__((..., TO_M[k]), out[..., TO_M[k]] + p[..., k]) for k in range(7)]; return out
def labels(split, mode, t):
    clf, gate = D[f"{split}_browser_clf"], D[f"{split}_browser_gate"]; views = []
    for v in (0, 1):
        p = clf[:, v] if mode == "6class" else merged(clf[:, v]); ood = 6 if mode == "6class" else 4
        wi = [i for i in range(p.shape[1]) if i != ood]
        conf = p[:, wi].max(1); best = np.array(wi)[p[:, wi].argmax(1)]
        ok = (p.argmax(1) != ood) & (gate[:, v, OOD] < 0.5) & (conf >= t)
        views.append((np.where(ok, best, -1), conf))
    lab = views[0][0]; use = (lab < 0) & (views[1][0] >= 0) & (views[1][1] >= 0.80)
    lab = np.where(use, views[1][0], lab)
    return lab if mode == "6class" else lab   # merged indices: 0 abr 1 bruise 2 burn 3 cut
rows = list(csv.DictReader(open(f"{WT}/experiments/v2/rit/manifest.csv")))
src = np.array([r["source"] for r in rows]); role = np.array([r["eval_role"] for r in rows]); rl = np.array([r["label"] for r in rows])
out = {}
for mode in ("6class", "possible_burn"):
    for t in (0.60, 0.50, 0.40):
        r = {}
        for split in ("val", "test"):
            y = D[f"{split}_y"]; lab = labels(split, mode, t); w = y != OOD
            yt = y if mode == "6class" else TO_M[y]
            r[split] = dict(n=int(w.sum()), answered=int((w & (lab >= 0)).sum()), right=int((w & (lab == yt)).sum()),
                            wrong=int((w & (lab >= 0) & (lab != yt)).sum()), nonwound=f"{int((~w & (lab >= 0)).sum())}/{int((~w).sum())}",
                            third_degree_shown_milder=int(((y == B3) & (lab >= 0) & (lab != (B3 if mode == '6class' else 2))).sum()))
        lab = labels("rit", mode, t)
        name = (lambda l: "unknown" if l < 0 else NAMES[l]) if mode == "6class" else (lambda l: "unknown" if l < 0 else ["abrasion", "bruise", "burn", "cut"][l])
        L = np.array([name(l) for l in lab]); ins = role == "in_scope"; b = role == "burn_any"
        r["rit"] = dict(everyday=f"{int((ins & (L == rl)).sum())} right, {int((ins & (L != 'unknown') & (L != rl)).sum())} wrong, of 48",
                        burns_recognised=f"{int((b & np.char.startswith(L, 'burn')).sum())}/67",
                        healthy_feet=f"{int(((src == 'lower_limb_feet') & (L != 'unknown')).sum())}/1613",
                        chronic=f"{int(((src == 'lower_limb_wounds') & (L != 'unknown')).sum())}/1313")
        out[f"{mode}@{t}"] = r
json.dump(out, open("more_answers.json", "w"), indent=1)
print(f"{'option':20s} | {'test wound: answered/right/wrong (of 331)':42s} | test non-wound | 3rd→milder | RIT everyday | RIT burns | RIT healthy feet | RIT chronic")
for k, r in out.items():
    t = r["test"]; print(f"{k:20s} | {t['answered']:4d} / {t['right']:4d} / {t['wrong']:4d} {'':25s} | {t['nonwound']:>14s} | {t['third_degree_shown_milder']:10d} | {r['rit']['everyday']:22s} | {r['rit']['burns_recognised']:9s} | {r['rit']['healthy_feet']:>16s} | {r['rit']['chronic']}")
print("\nval (dev):"); [print(f"  {k:20s} {r['val']}") for k, r in out.items()]

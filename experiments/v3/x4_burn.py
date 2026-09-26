"""X4a (PLAN.md): the 'possible burn' task. Mapping for every comparator and benchmark:
burn_1st/2nd/3rd -> burn; abrasion, bruise, cut, out_of_scope unchanged.
  mapped-after   the 6-class pipeline's decision, then mapped (what the site shows today, merged)
  summed         burn probability = sum of the 3 burn probabilities, then decide() unchanged
                 (0.60, gate 0.5, out_of_scope rule); crop retry as C1 at 0.80
This is a different (easier) task: none of these numbers are 6-class accuracy.
Usage: python x4_burn.py [views_name] [gate_views_name]"""
import sys, csv, json, collections
import numpy as np
from common import *
NAME = sys.argv[1] if len(sys.argv) > 1 else "deployed"; GNAME = sys.argv[2] if len(sys.argv) > 2 else NAME
V = np.load(f"{V3}/views_{NAME}.npz"); VG = np.load(f"{V3}/views_{GNAME}.npz")
M = ["abrasion", "bruise", "burn", "cut", "out_of_scope"]; MOOD = 4; MW = [0, 1, 2, 3]
TO_M = np.array([M.index("burn") if n.startswith("burn") else M.index(n) for n in NAMES])

def merge_probs(p):   # (..., 7) -> (..., 5)
    out = np.zeros(p.shape[:-1] + (5,), p.dtype)
    for k in range(7): out[..., TO_M[k]] += p[..., k]
    return out

def decide_m(pm, gate, t=0.60):
    conf = pm[:, MW].max(1); best = pm[:, MW].argmax(1)
    ok = (pm.argmax(1) != MOOD) & (gate[:, OOD] < 0.5) & (conf >= t)
    return np.where(ok, best, -1), best, conf

def labels(clf, gate, mode, crop):
    """clf/gate (n, 2, 7). mode 'after' or 'summed'. crop: C0 or C1. Returns merged labels, -1 = unknown."""
    views = []
    for v in (0, 1):
        if mode == "after":
            a, b, c = decide(clf[:, v], gate[:, v]); views.append((np.where(a, TO_M[b], -1), c))
        else:
            lab, b, c = decide_m(merge_probs(clf[:, v]), gate[:, v]); views.append((lab, c))
    lab = views[0][0]
    if crop == "C1":
        use = (lab < 0) & (views[1][0] >= 0) & (views[1][1] >= 0.80); lab = np.where(use, views[1][0], lab)
    return lab

def score_m(lab, y7):
    y = TO_M[y7]; wound = y != MOOD; ans = lab >= 0; a = ans & wound; c = a & (lab == y)
    burn = y == 2
    return dict(eligible=int(wound.sum()), answered=int(a.sum()), correct=int(c.sum()), coverage=round(100 * a.sum() / wound.sum(), 1),
                acc_answered=round(100 * c.sum() / max(1, a.sum()), 1), correct_of_eligible=round(100 * c.sum() / wound.sum(), 1),
                nonwound_labelled=int((ans & ~wound).sum()), nonwound=int((~wound).sum()),
                burns=int(burn.sum()), burns_answered=int((a & burn).sum()), burns_as_burn=int((a & burn & (lab == 2)).sum()),
                burn_shown_other_injury=int((a & burn & (lab != 2)).sum()), other_injury_shown_burn=int((a & ~burn & (lab == 2)).sum()),
                deg3_shown_other_injury=int((a & (y7 == B3) & (lab != 2)).sum()),
                per_class={M[k]: dict(n=int((y == k).sum()), answered=int((a & (y == k)).sum()), correct=int((c & (y == k)).sum())) for k in MW})

out = {}
for split in ("val", "test"):
    y = V[f"{split}_y"]
    for inp in ("browser", "raw"):
        clf, gate = V[f"{split}_{inp}_clf"], VG[f"{split}_{inp}_gate"]
        for mode in ("after", "summed"):
            for crop in ("C0", "C1"):
                out[f"{split}_{inp}_{mode}_{crop}"] = score_m(labels(clf, gate, mode, crop), y)
# X4b justification (fixed rule): share of val (browser, summed, C0) answered-wound errors that involve burn
y = V["val_y"]; lab = labels(V["val_browser_clf"], VG["val_browser_gate"], "summed", "C0"); ym = TO_M[y]
err = (lab >= 0) & (ym != MOOD) & (lab != ym); inv = err & ((ym == 2) | (lab == 2))
out["x4b_rule"] = dict(val_errors=int(err.sum()), involving_burn=int(inv.sum()), justified=bool(err.sum() and inv.sum() >= err.sum() / 2),
                       confusions=dict(collections.Counter(f"{M[a]}->{M[b]}" for a, b in zip(ym[err], lab[err]))))
# RIT
rows = list(csv.DictReader(open(f"{WT}/experiments/v2/rit/manifest.csv")))
clf, gate = V["rit_browser_clf"], VG["rit_browser_gate"]
for mode in ("after", "summed"):
    for crop in ("C0", "C1"):
        lab = labels(clf, gate, mode, crop); s = {}
        ins = [i for i, r in enumerate(rows) if r["eval_role"] == "in_scope"]
        s["in_scope"] = dict(n=len(ins), answered=sum(lab[i] >= 0 for i in ins), correct=sum(lab[i] >= 0 and M[lab[i]] == rows[i]["label"] for i in ins),
                             shown_burn=sum(lab[i] == 2 for i in ins))
        b = [i for i, r in enumerate(rows) if r["eval_role"] == "burn_any"]
        s["burns"] = dict(n=len(b), burn=sum(lab[i] == 2 for i in b), other_injury=sum(lab[i] in (0, 1, 3) for i in b), unknown=sum(lab[i] < 0 for i in b), ci_burn=wilson(sum(lab[i] == 2 for i in b), len(b)))
        s["non_wound"] = {src: dict(n=len(g), labelled=sum(lab[i] >= 0 for i in g), as_burn=sum(lab[i] == 2 for i in g))
                          for src in sorted({r["source"] for r in rows if r["eval_role"] == "non_wound"})
                          for g in [[i for i, r in enumerate(rows) if r["eval_role"] == "non_wound" and r["source"] == src]]}
        out[f"rit_browser_{mode}_{crop}"] = s
json.dump(out, open(f"{V3}/x4_burn_{NAME}{'' if GNAME == NAME else '_gate-' + GNAME}.json", "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
for k, s in out.items():
    if k.startswith(("val", "test")):
        print(f"{k:24s} cov {s['coverage']:5.1f}% ({s['answered']}/{s['eligible']}) acc {s['acc_answered']:5.1f}% corr/elig {s['correct_of_eligible']:5.1f}% non-wound {s['nonwound_labelled']}/{s['nonwound']} "
              f"| burns {s['burns_as_burn']}/{s['burns']} as burn, {s['burn_shown_other_injury']} as other injury (3rd: {s['deg3_shown_other_injury']}); other injury shown burn {s['other_injury_shown_burn']}")
    elif k.startswith("rit"):
        print(k, json.dumps(s, default=int))
print("X4b rule:", out["x4b_rule"])

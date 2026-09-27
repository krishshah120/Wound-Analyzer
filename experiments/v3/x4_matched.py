"""X4 at matched coverage (merged taxonomy, browser input, full frame C0 plus the production crop C1).
Every option's threshold is chosen on VAL so it answers as many val wound photos as today's pipeline
mapped to the merged labels (117 of 251), then carried unchanged to test and the RIT.
Options: after (deployed, 0.60, then mapped) | summed (deployed, burn probs summed) | X4B_s* (trained).
Writes x4_matched.json."""
import glob, csv, json
import numpy as np
from common import *
from x4_burn import merge_probs, decide_m, TO_M, M, MOOD, MW   # x4_burn runs its report on import; output suppressed below
D = np.load(f"{V3}/views_deployed.npz")
rows = list(csv.DictReader(open(f"{WT}/experiments/v2/rit/manifest.csv")))
role = np.array([r["eval_role"] for r in rows]); src = np.array([r["source"] for r in rows]); rl = np.array([r["label"] for r in rows])
opts = {"after": None, "summed": D}
for f in sorted(glob.glob(f"{V3}/views_X4B_s*.npz")): opts[f.split("views_")[1][:-4]] = np.load(f)

def merged(name, key):     # merged probabilities (n, 2, 5) for an option
    if name == "summed": return merge_probs(D[key])
    return opts[name][key]
def labels(name, split, t):
    if name == "after":
        clf, gate = D[f"{split}_clf"], D[f"{split}_gate"]; v = []
        for k in (0, 1):
            a, b, c = decide(clf[:, k], gate[:, k], t); v.append((np.where(a, TO_M[b], -1), c))
    else:
        pm, gate = merged(name, f"{split}_clf"), D[f"{split}_gate"]; v = []
        for k in (0, 1):
            lab, b, c = decide_m(pm[:, k], gate[:, k], t); v.append((lab, c))
    lab = v[0][0]; crop = (lab < 0) & (v[1][0] >= 0) & (v[1][1] >= 0.80)
    return lab, np.where(crop, v[1][0], lab)
TARGET = 117
yv = TO_M[D["val_y"]]; yt = TO_M[D["test_y"]]
out = {}
for name in opts:
    if name == "after": t = 0.60
    else:
        pm = merged(name, "val_browser_clf")[:, 0]; g = D["val_browser_gate"][:, 0]
        ok = (pm.argmax(1) != MOOD) & (g[:, OOD] < 0.5) & (yv != MOOD)
        t = float(np.sort(pm[:, MW].max(1)[ok])[::-1][TARGET - 1])
    r = {"threshold_from_val": round(t, 4)}
    for split, y in (("val_browser", yv), ("test_browser", yt)):
        for ci, lab in zip(("C0", "C1"), labels(name, split, t)):
            w = y != MOOD; a = (lab >= 0) & w; c = a & (lab == y)
            r[f"{split}_{ci}"] = dict(answered=int(a.sum()), eligible=int(w.sum()), acc=round(100 * c.sum() / max(1, a.sum()), 1),
                                      nonwound=f"{int(((lab >= 0) & ~w).sum())}/{int((~w).sum())}", burn_as_other=int((a & (y == 2) & (lab != 2)).sum()),
                                      other_as_burn=int((a & (y != 2) & (lab == 2)).sum()))
    for ci, lab in zip(("C0", "C1"), labels(name, "rit_browser", t)):
        ins = role == "in_scope"; b = role == "burn_any"; feet = src == "lower_limb_feet"; chron = src == "lower_limb_wounds"; hc = (src == "roboflow_wound_v1") & (role == "non_wound")
        lname = np.array(["unknown" if l < 0 else M[l] for l in lab])
        r[f"rit_{ci}"] = dict(inscope=f"{int((ins & (lab >= 0)).sum())}/48 answered, {int((ins & (lname == rl)).sum())} correct, {int((ins & (lab == 2)).sum())} shown burn",
                              burns=f"{int((b & (lab == 2)).sum())}/67 burn", feet=f"{int((feet & (lab >= 0)).sum())}/1613 labelled ({int((feet & (lab == 2)).sum())} burn)",
                              chronic=f"{int((chron & (lab >= 0)).sum())}/1313", healthy_closeups=f"{int((hc & (lab >= 0)).sum())}/22")
    out[name] = r
json.dump(out, open(f"{V3}/x4_matched.json", "w"), indent=1)
print("\n" * 2 + "=" * 30 + " MATCHED COVERAGE (merged task) " + "=" * 30)
for name, r in out.items():
    print(f"{name:10s} t={r['threshold_from_val']}")
    for k in ("val_browser_C0", "test_browser_C0", "test_browser_C1"):
        s = r[k]; print(f"   {k:16s} {s['answered']}/{s['eligible']} answered, acc {s['acc']}%, non-wound {s['nonwound']}, burn->other {s['burn_as_other']}, other->burn {s['other_as_burn']}")
    for k in ("rit_C0", "rit_C1"): print(f"   {k:16s} {r[k]}")

"""X6b verdict (PLAN.md addendum): retrained ensemble RE3 (actual saved models) vs the complete deployed
pipeline, same images. Inputs: views_RE3_s{1,2,3}.npz (views.py with each member's .keras), gate and
deployed classifier from views_deployed.npz. Writes x6_ensemble.json."""
import csv, json
import numpy as np
from common import *
D = np.load(f"{V3}/views_deployed.npz")
M = [np.load(f"{V3}/views_RE3_s{s}.npz") for s in (1, 2, 3)]
for m in M: assert (m["val_y"] == D["val_y"]).all() and (m["test_y"] == D["test_y"]).all() and (m["rit_files"] == D["rit_files"]).all()
KEYS = [k for k in D.files if k.endswith("_clf")]
ENS = {k: np.mean([m[k] for m in M], 0) for k in KEYS}
rows = list(csv.DictReader(open(f"{WT}/experiments/v2/rit/manifest.csv")))
role = np.array([r["eval_role"] for r in rows]); src = np.array([r["source"] for r in rows]); rl = np.array([r["label"] for r in rows])
RIT_Y = np.array([NAMES.index(l) if l in NAMES else -9 for l in rl])

def policy(clf, gate, which, t=0.60):
    a0, b0, c0 = decide(clf[:, 0], gate[:, 0], t); a1, b1, c1 = decide(clf[:, 1], gate[:, 1], t)
    lab = np.where(a0, b0, -1)
    if which == "C0": return lab
    use = ~a0 & a1 & (c1 >= max(0.80, t))
    if which == "C2": use &= (b1 == b0)
    return np.where(use, b1, lab)
def pipe(P, split, inp, which="C1", t=0.60):
    return policy(P[f"{split}_{inp}_clf"], D[f"{split}_{inp}_gate"], which, t)
def rit(lab):
    ins = role == "in_scope"; feet = src == "lower_limb_feet"
    return dict(inscope_answered=int((ins & (lab >= 0)).sum()), inscope_correct=int((ins & (lab == RIT_Y)).sum()),
                inscope_wrong=int((ins & (lab >= 0) & (lab != RIT_Y)).sum()),
                feet_labelled=int((feet & (lab >= 0)).sum()), feet_3rd=int((feet & (lab == B3)).sum()),
                chronic_labelled=int(((src == "lower_limb_wounds") & (lab >= 0)).sum()),
                closeups_labelled=int(((src == "roboflow_wound_v1") & (role == "non_wound") & (lab >= 0)).sum()),
                burns_any_burn=int(((role == "burn_any") & np.isin(lab, [B1, B2, B3])).sum()), burns_n=int((role == "burn_any").sum()))
out = {}
for name, P in (("deployed", D), ("RE3", ENS)) + tuple((f"RE3_member_s{s}", M[s - 1]) for s in (1, 2, 3)):
    r = {}
    for which in ("C0", "C1", "C2"):
        for split in ("val", "test"):
            for inp in ("browser", "raw"):
                lab = pipe(P, split, inp, which); r[f"{split}_{inp}_{which}"] = score(lab >= 0, np.maximum(lab, 0), D[f"{split}_y"])
        r[f"rit_{which}"] = rit(pipe(P, "rit", "browser", which))
    out[name] = r
# matched coverage / matched error vs the complete deployed pipeline (C1, browser; thresholds from val)
yv, yt = D["val_y"], D["test_y"]; dv = out["deployed"]["val_browser_C1"]
target, dep_err = dv["answered"], 1 - dv["correct"] / dv["answered"]
def sweep(P, cond):
    for t in np.round(np.arange(0.40, 0.95, 0.0025), 4):
        lab = pipe(P, "val", "browser", "C1", t); s = score(lab >= 0, np.maximum(lab, 0), yv)
        if cond(s): return float(t)
    return None
t_cov = sweep(ENS, lambda s: s["answered"] <= target)          # first threshold not answering more than deployed
t_err = sweep(ENS, lambda s: s["answered"] and 1 - s["correct"] / s["answered"] <= dep_err)
mt = {}
for lbl, t in (("matched_coverage", t_cov), ("matched_error", t_err)):
    if t is None: mt[lbl] = None; continue
    mt[lbl] = dict(threshold=t, val=score(*(lambda l: (l >= 0, np.maximum(l, 0)))(pipe(ENS, "val", "browser", "C1", t)), yv),
                   test=score(*(lambda l: (l >= 0, np.maximum(l, 0)))(pipe(ENS, "test", "browser", "C1", t)), yt),
                   rit=rit(pipe(ENS, "rit", "browser", "C1", t)))
out["matched"] = mt
# paired grouped CIs at 0.60 (C1, browser): RE3 - deployed
cis = {}
for split, y in (("val", yv), ("test", yt)):
    a, b = pipe(ENS, split, "browser"), pipe(D, split, "browser"); g = groups(split)
    cis[split] = dict(acc_answered=paired_bootstrap(g, acc_stat(a >= 0, np.maximum(a, 0), y), acc_stat(b >= 0, np.maximum(b, 0), y)),
                      coverage=paired_bootstrap(g, cov_stat(a >= 0, y), cov_stat(b >= 0, y)),
                      nonwound_pct=paired_bootstrap(g, fp_stat(a >= 0, y), fp_stat(b >= 0, y)))
rg = np.array([r["dup_group"] for r in rows]); a, b = pipe(ENS, "rit", "browser"), pipe(D, "rit", "browser"); feet = src == "lower_limb_feet"
cis["rit_feet_labelled_pct"] = paired_bootstrap(rg, lambda i: 100 * (feet & (a >= 0))[i].sum() / max(1, feet[i].sum()), lambda i: 100 * (feet & (b >= 0))[i].sum() / max(1, feet[i].sum()))
out["RE3_minus_deployed_95ci_at_0.60"] = cis
# promising (declared)
def tot(name):
    r = out[name]; v, t, R = r["val_browser_C1"], r["test_browser_C1"], r["rit_C1"]
    wrong = (v["answered"] - v["correct"]) + v["nonwound_labelled"] + (t["answered"] - t["correct"]) + t["nonwound_labelled"] + R["inscope_wrong"] + \
            R["feet_labelled"] + R["chronic_labelled"] + R["closeups_labelled"]
    return dict(wrong=wrong, correct=v["correct"] + t["correct"] + R["inscope_correct"], tierA=v["tierA"] + t["tierA"], feet=R["feet_labelled"])
d_, e_ = tot("deployed"), tot("RE3")
crit = {"1_fewer_wrong_labels": e_["wrong"] < d_["wrong"], "2_correct_not_below_minus_5": e_["correct"] >= d_["correct"] - 5,
        "3_tierA_not_more": e_["tierA"] <= d_["tierA"], "4_feet_not_more": e_["feet"] <= d_["feet"]}
out["promising"] = dict(deployed=d_, RE3=e_, criteria=crit, promising=all(crit.values()))
json.dump(out, open(f"{V3}/x6_ensemble.json", "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
for name in out:
    if name in ("matched", "RE3_minus_deployed_95ci_at_0.60", "promising"): continue
    r = out[name]
    for k in ("val_browser_C1", "test_browser_C1"):
        s = r[k]; print(f"{name:16s} {k:16s} {s['answered']}/{s['eligible']} answered, {s['correct']} right ({s['acc_answered']}%), non-wound {s['nonwound_labelled']}/{s['nonwound']}, 3rd->other {s['tierA']}")
    print(f"{name:16s} RIT C1: {r['rit_C1']}")
print("matched:", json.dumps({k: (None if v is None else dict(threshold=v['threshold'], val=f"{v['val']['answered']} ans {v['val']['acc_answered']}%", test=f"{v['test']['answered']} ans {v['test']['acc_answered']}% nw {v['test']['nonwound_labelled']}", rit_feet=v['rit']['feet_labelled'])) for k, v in mt.items()}))
print("CIs (RE3 - deployed, 0.60, C1 browser):", json.dumps(cis))
print("promising:", json.dumps(out["promising"]))

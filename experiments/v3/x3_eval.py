"""X3 verdict (PLAN.md): A = R_exact seeds 1-3 (b84_train) vs B = X3B seeds (b84_train minus the 86
objective-conflict photos). Same recipe, shipped gate, same val/test photos and denominators.
Continuation rule: pilot B seed 1 val AURC <= 0.183 (A mean). Verdict: 3-seed means and 3-member
ensembles, rule MC thresholds (>= +3 points accuracy at 46.2% val coverage, non-wound <= 16/94,
tier A <= 1); else inconclusive. Writes x3_eval.json."""
import sys, glob, json
import numpy as np
from common import *
S = SHIPPED; gv, gt, yv, yt = S["val_gate"], S["test_gate"], S["val_y"], S["test_y"]
TARGET = 116
def load(tag):
    d = np.load(f"{RUNS}/{tag}_probs.npz"); assert list(d["names"]) == NAMES and (d["y_val"] == yv).all(); return d["p_val"], d["p_test"]
A = {s: load(f"R_exact_s{s}") for s in (1, 2, 3)}
RUN = sys.argv[1] if len(sys.argv) > 1 else "X3B"   # X6C = the narrow (identical-image) exclusion
B = {int(f.split("_s")[-1].split("_")[0]): load(f"{RUN}_s{f.split('_s')[-1].split('_')[0]}") for f in sorted(glob.glob(f"{RUNS}/{RUN}_s*_probs.npz"))}
def row(pv, pt):
    t = matched_threshold(pv, gv, yv, TARGET)
    m = score(*decide(pv, gv, t)[:2], yv); mt = score(*decide(pt, gt, t)[:2], yt)
    return dict(val_aurc=round(aurc(pv, gv, yv), 3), test_aurc=round(aurc(pt, gt, yt), 3), val_acc_matched=m["acc_answered"], val_nonwound=m["nonwound_labelled"],
                val_tierA=m["tierA"], val_tierB=m["tierB"], test_acc_at_val_threshold=mt["acc_answered"], test_cov_at_val_threshold=mt["coverage"],
                test_nonwound=mt["nonwound_labelled"], test_tierA=mt["tierA"], val_060=score(*decide(pv, gv)[:2], yv), test_060=score(*decide(pt, gt)[:2], yt))
out = {"A": {s: row(*v) for s, v in A.items()}, "B": {s: row(*v) for s, v in B.items()}}
mean = lambda d, k: round(float(np.mean([r[k] for r in d.values()])), 2)
KEYS = ("val_aurc", "val_acc_matched", "val_nonwound", "val_tierA", "test_aurc", "test_acc_at_val_threshold", "test_nonwound", "test_tierA")
out["mean"] = {g: {k: mean(out[g], k) for k in KEYS} for g in ("A", "B") if out[g]}
out["pilot_continue"] = bool(1 in out["B"] and out["B"][1]["val_aurc"] <= 0.183)
if len(B) == 3:
    ens = lambda D: (np.mean([v[0] for v in D.values()], 0), np.mean([v[1] for v in D.values()], 0))
    out["ensemble"] = {"A": row(*ens(A)), "B": row(*ens(B))}
    dep = score(*decide(S["val_clf"], gv)[:2], yv)["acc_answered"]
    mb = out["mean"]["B"]; ma = out["mean"]["A"]
    out["verdict"] = dict(B_minus_A_val_acc_matched=round(mb["val_acc_matched"] - ma["val_acc_matched"], 2),
                          B_meets_rule_MC_vs_deployed=bool(mb["val_acc_matched"] - dep >= 3 and mb["val_nonwound"] <= 16 and mb["val_tierA"] <= 1))
    ea, eb = ens(A), ens(B); ta, tb = matched_threshold(ea[0], gv, yv, TARGET), matched_threshold(eb[0], gv, yv, TARGET)
    ab, bb = decide(ea[0], gv, ta), decide(eb[0], gv, tb)
    out["ensemble_B_minus_A_val_acc_95ci"] = paired_bootstrap(groups("val"), acc_stat(bb[0], bb[1], yv), acc_stat(ab[0], ab[1], yv))
    at_, bt_ = decide(ea[1], gt, ta), decide(eb[1], gt, tb)
    out["ensemble_B_minus_A_test_acc_95ci"] = paired_bootstrap(groups("test"), acc_stat(bt_[0], bt_[1], yt), acc_stat(at_[0], at_[1], yt))
json.dump(out, open(f"{V3}/x3_eval{'' if RUN == 'X3B' else '_' + RUN}.json", "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
for g in ("A", "B"):
    for s, r in out[g].items(): print(g, s, {k: r[k] for k in KEYS})
print("means", out["mean"]); print("pilot continue:", out["pilot_continue"])
for k in ("ensemble", "verdict", "ensemble_B_minus_A_val_acc_95ci", "ensemble_B_minus_A_test_acc_95ci"):
    if k in out: print(k, {g: ({kk: v[kk] for kk in KEYS} if isinstance(v, dict) and "val_aurc" in v else v) for g, v in out[k].items()} if isinstance(out[k], dict) else out[k])

"""X6C pilot verdict: deployed recipe on b84_train minus the 24 identical-image conflicts, seed 1.
Continue only if val AURC <= 0.183 (PLAN.md addendum). Matched seed-1 control: R_exact_s1."""
import json
import numpy as np
from common import *
S = SHIPPED; gv, gt, yv, yt = S["val_gate"], S["test_gate"], S["val_y"], S["test_y"]
out = {}
for tag in ("R_exact_s1", "X6C_s1"):
    d = np.load(f"{RUNS}/{tag}_probs.npz"); assert list(d["names"]) == NAMES and (d["y_val"] == yv).all()
    t = matched_threshold(d["p_val"], gv, yv, 116); m = score(*decide(d["p_val"], gv, t)[:2], yv)
    out[tag] = dict(val_aurc=round(aurc(d["p_val"], gv, yv), 3), test_aurc=round(aurc(d["p_test"], gt, yt), 3), val_acc_matched=m["acc_answered"],
                    val_nonwound_matched=m["nonwound_labelled"], val_tierA_matched=m["tierA"], test_060=score(*decide(d["p_test"], gt)[:2], yt))
out["continue"] = out["X6C_s1"]["val_aurc"] <= 0.183
json.dump(out, open(f"{V3}/x6c_eval.json", "w"), indent=1)
for k, v in out.items(): print(k, v if not isinstance(v, dict) else {kk: vv for kk, vv in v.items() if kk != "test_060"} | {"test_060": f"{v['test_060']['answered']} ans {v['test_060']['acc_answered']}% nw {v['test_060']['nonwound_labelled']}"})

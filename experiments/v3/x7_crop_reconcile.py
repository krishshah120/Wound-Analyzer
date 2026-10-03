"""X7 follow-up: reconcile the crop-retry statement from per-photo predictions (deployed, browser input,
0.60). Every photo is classified by its full-frame (C0) and production-retry (C1) outcome; also checks
whether any photo answered under C0 changes its answer under C1 (by construction it should not)."""
import json, csv
import numpy as np
from common import *
D = np.load(f"{V3}/views_deployed.npz")
def lab(split, which):
    clf, gate = D[f"{split}_browser_clf"], D[f"{split}_browser_gate"]
    a0, b0, _ = decide(clf[:, 0], gate[:, 0]); a1, b1, c1 = decide(clf[:, 1], gate[:, 1])
    l = np.where(a0, b0, -1)
    return l if which == "C0" else np.where(~a0 & a1 & (c1 >= 0.80), b1, l)
out = {}
for split in ("val", "test"):
    y = D[f"{split}_y"]; l0, l1 = lab(split, "C0"), lab(split, "C1"); w = y != OOD
    changed_answered = int(((l0 >= 0) & (l1 != l0)).sum())
    new = (l0 < 0) & (l1 >= 0)
    out[split] = dict(previously_answered_changed=changed_answered, new_answers_total=int(new.sum()),
                      new_wound_right=int((new & w & (l1 == y)).sum()), new_wound_wrong=int((new & w & (l1 != y)).sum()),
                      new_nonwound_labelled=int((new & ~w).sum()), new_3rd_shown_other=int((new & (y == B3) & (l1 != B3)).sum()),
                      wrong_detail=[f"{NAMES[a]}->{NAMES[b]}" for a, b in zip(y[new & w & (l1 != y)], l1[new & w & (l1 != y)])])
tot = {k: sum(out[s][k] for s in ("val", "test")) for k in out["val"] if k != "wrong_detail"}
out["val+test"] = tot
json.dump(out, open(f"{V3}/x7_crop_reconcile.json", "w"), indent=1)
print(json.dumps(out, indent=1))

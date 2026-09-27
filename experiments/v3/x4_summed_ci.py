"""Exploratory (NOT a pre-declared candidate): 'summed' burn at the val-matched threshold vs today's
decisions mapped to the merged labels. Paired 95% CIs over RIT duplicate groups (C1 = production crop)."""
import csv, json, io, contextlib
import numpy as np
with contextlib.redirect_stdout(io.StringIO()):
    from x4_matched import labels, out as MATCHED, rows, role, src, rl, M
from common import paired_bootstrap
grp = np.array([r["dup_group"] for r in rows]); t = MATCHED["summed"]["threshold_from_val"]
res = {}
for ci in (0, 1):
    a = labels("after", "rit_browser", 0.60)[ci]; s = labels("summed", "rit_browser", t)[ci]
    ln = lambda lab: np.array(["unknown" if l < 0 else M[l] for l in lab])
    stats = {"burns_recognised_of_67": (role == "burn_any", lambda lab: lab == 2),
             "feet_labelled_of_1613": (src == "lower_limb_feet", lambda lab: lab >= 0),
             "inscope_correct_of_48": (role == "in_scope", lambda lab: ln(lab) == rl),
             "inscope_shown_burn_of_48": (role == "in_scope", lambda lab: lab == 2)}
    r = {}
    for k, (mask, f) in stats.items():
        fa, fs = f(a) & mask, f(s) & mask
        st = lambda v: (lambda idx: 100 * v[idx].sum() / max(1, mask[idx].sum()))
        r[k] = dict(after=int(fa.sum()), summed=int(fs.sum()), diff_pct_points_95ci=paired_bootstrap(grp, st(fs), st(fa)))
    res["C0" if ci == 0 else "C1"] = r
json.dump(res, open("x4_summed_rit_ci.json", "w"), indent=1)
for k, v in res.items(): print(k, json.dumps(v))

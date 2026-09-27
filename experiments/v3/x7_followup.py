"""X7 FOLLOW-UP ANALYSIS (not a new verdict; RE3's failed deployment verdict in LEDGER.md stands).
Question: does the retrained ensemble RE3 beat simply making the deployed model more selective?
Saved predictions only (views_deployed.npz, views_RE3_s*.npz); complete pipelines (gate + crop retry
C1), browser input. Thresholds are chosen on VAL only; test and RIT are reported as development
evidence. Populations are kept separate - no single pooled score.
Writes x7_followup.json."""
import csv, json
import numpy as np
from common import *
D = np.load(f"{V3}/views_deployed.npz")
ENS = {k: np.mean([np.load(f"{V3}/views_RE3_s{s}.npz")[k] for s in (1, 2, 3)], 0) for k in D.files if k.endswith("_clf")}
PIPES = {"deployed": D, "RE3": ENS}
rows = list(csv.DictReader(open(f"{WT}/experiments/v2/rit/manifest.csv")))
src = np.array([r["source"] for r in rows]); role = np.array([r["eval_role"] for r in rows])
RY = np.array([NAMES.index(r["label"]) if r["label"] in NAMES else -9 for r in rows]); rg = np.array([r["dup_group"] for r in rows])
BURN = [B1, B2, B3]

def labels(P, split, t):
    """Complete pipeline C1 at confidence threshold t (crop answers at max(0.80, t)). -1 = unknown."""
    clf, gate = P[f"{split}_browser_clf"], D[f"{split}_browser_gate"]
    a0, b0, _ = decide(clf[:, 0], gate[:, 0], t); a1, b1, c1 = decide(clf[:, 1], gate[:, 1], t)
    lab = np.where(a0, b0, -1); use = ~a0 & a1 & (c1 >= max(0.80, t))
    return np.where(use, b1, lab)

def populations(P, t):
    """Counts per population with denominators."""
    out = {}
    for split in ("val", "test"):
        y = D[f"{split}_y"]; lab = labels(P, split, t); w = y != OOD
        out[f"{split}_acute"] = dict(n=int(w.sum()), answered=int((w & (lab >= 0)).sum()), right=int((w & (lab == y)).sum()),
                                     wrong=int((w & (lab >= 0) & (lab != y)).sum()), tierA=int(((y == B3) & (lab >= 0) & (lab != B3)).sum()))
        out[f"{split}_other_nonwound"] = dict(n=int((~w).sum()), labelled=int((~w & (lab >= 0)).sum()))
    lab = labels(P, "rit", t)
    ins = role == "in_scope"
    out["rit_acute_everyday"] = dict(n=int(ins.sum()), answered=int((ins & (lab >= 0)).sum()), right=int((ins & (lab == RY)).sum()), wrong=int((ins & (lab >= 0) & (lab != RY)).sum()))
    b = role == "burn_any"
    out["rit_acute_burns_degree_unknown"] = dict(n=int(b.sum()), any_burn=int((b & np.isin(lab, BURN)).sum()), other_injury=int((b & (lab >= 0) & ~np.isin(lab, BURN)).sum()))
    for nm, m in (("rit_healthy_feet", src == "lower_limb_feet"), ("rit_chronic_wounds", src == "lower_limb_wounds"),
                  ("rit_other_nonwound_closeups", (src == "roboflow_wound_v1") & (role == "non_wound"))):
        out[nm] = dict(n=int(m.sum()), labelled=int((m & (lab >= 0)).sum()), as_3rd=int((m & (lab == B3)).sum()))
    return out

res = {"at_0.60": {n: populations(P, 0.60) for n, P in PIPES.items()}}
# what "457 / 329 wrong labels overall" contained (X6 promising test)
def overall(p):
    return dict(val_wrong_injury=p["val_acute"]["wrong"], val_nonwound=p["val_other_nonwound"]["labelled"], test_wrong_injury=p["test_acute"]["wrong"],
                test_nonwound=p["test_other_nonwound"]["labelled"], rit_everyday_wrong=p["rit_acute_everyday"]["wrong"],
                rit_feet=p["rit_healthy_feet"]["labelled"], rit_chronic=p["rit_chronic_wounds"]["labelled"], rit_closeups=p["rit_other_nonwound_closeups"]["labelled"])
res["composition_of_overall_wrong"] = {n: (lambda o: dict(o, total=sum(o.values())))(overall(res["at_0.60"][n])) for n in PIPES}

# threshold sweep (descriptive) + val-chosen matched operating points
grid = np.round(np.arange(0.30, 0.995, 0.005), 3)
sweep = {n: {float(t): populations(P, t) for t in grid} for n, P in PIPES.items()}
def val_pick(name, cond):
    ok = [t for t in grid if cond(sweep[name][float(t)])]
    return None if not ok else float(min(ok))          # lowest qualifying threshold = most answers
re = res["at_0.60"]["RE3"]; de = res["at_0.60"]["deployed"]
picks = {
  # A: deployed made more selective until it answers no more val wound photos than RE3 does at 0.60
  "deployed_matched_to_RE3_val_wound_coverage": ("deployed", val_pick("deployed", lambda p: p["val_acute"]["answered"] <= re["val_acute"]["answered"])),
  # B: deployed made more selective until its val non-wound false labels are no more than RE3's at 0.60
  "deployed_matched_to_RE3_val_nonwound_fp": ("deployed", val_pick("deployed", lambda p: p["val_other_nonwound"]["labelled"] <= re["val_other_nonwound"]["labelled"])),
  # C: deployed made more selective until its val wrong injury labels are no more than RE3's (matched error count)
  "deployed_matched_to_RE3_val_wrong_injury": ("deployed", val_pick("deployed", lambda p: p["val_acute"]["wrong"] <= re["val_acute"]["wrong"])),
}
res["val_matched_operating_points"] = {k: dict(pipeline=n, threshold=t, results=None if t is None else sweep[n][t]) for k, (n, t) in picks.items()}

# paired uncertainty on test + RIT for pick A vs RE3 at 0.60
tA = picks["deployed_matched_to_RE3_val_wound_coverage"][1]
if tA is not None:
    ci = {}
    for split in ("test",):
        y = D[f"{split}_y"]; g = groups(split); w = y != OOD
        a, b = labels(ENS, split, 0.60), labels(D, split, tA)
        f = lambda lab, m: (lambda idx: float((m & lab)[idx].sum()))
        ci["test_wrong_injury_count"] = paired_bootstrap(g, f(a != y, w & (a >= 0)), f(b != y, w & (b >= 0)))
        ci["test_right_count"] = paired_bootstrap(g, f(a == y, w), f(b == y, w))
        ci["test_nonwound_count"] = paired_bootstrap(g, f(a >= 0, ~w), f(b >= 0, ~w))
    a, b = labels(ENS, "rit", 0.60), labels(D, "rit", tA); feet = src == "lower_limb_feet"; chron = src == "lower_limb_wounds"
    ci["rit_feet_labelled_count"] = paired_bootstrap(rg, lambda i: float((feet & (a >= 0))[i].sum()), lambda i: float((feet & (b >= 0))[i].sum()))
    ci["rit_chronic_labelled_count"] = paired_bootstrap(rg, lambda i: float((chron & (a >= 0))[i].sum()), lambda i: float((chron & (b >= 0))[i].sum()))
    res["RE3_at_0.60_minus_deployed_at_pickA_95ci_counts"] = ci

# descriptive curve on RIT: acute everyday right answers vs healthy-feet false labels (all thresholds, no selection)
curve = {}
for n in PIPES:
    pts = sorted({(sweep[n][float(t)]["rit_healthy_feet"]["labelled"], sweep[n][float(t)]["rit_acute_everyday"]["right"],
                   sweep[n][float(t)]["test_acute"]["right"], float(t)) for t in grid})
    curve[n] = pts
def best_at(n, feet_max):
    c = [(p[2], p[1], p[3]) for p in curve[n] if p[0] <= feet_max]
    return None if not c else max(c)          # most test acute right answers while feet <= level
res["curve_test_right_vs_rit_feet"] = {str(lv): {n: best_at(n, lv) for n in PIPES} for lv in (60, 80, 94, 120, 150, 183)}
json.dump(res, open(f"{V3}/x7_followup.json", "w"), indent=1)

def show(tag, p):
    print(f"  {tag}")
    for k in ("val_acute", "test_acute"): v = p[k]; print(f"    {k:28s} {v['answered']}/{v['n']} answered, {v['right']} right, {v['wrong']} wrong (3rd shown other {v['tierA']})")
    for k in ("val_other_nonwound", "test_other_nonwound"): v = p[k]; print(f"    {k:28s} {v['labelled']}/{v['n']} labelled")
    v = p["rit_acute_everyday"]; print(f"    {'rit_acute_everyday':28s} {v['answered']}/{v['n']} answered, {v['right']} right, {v['wrong']} wrong")
    v = p["rit_acute_burns_degree_unknown"]; print(f"    {'rit_burns_degree_unknown':28s} {v['any_burn']}/{v['n']} any burn label, {v['other_injury']} other injury")
    for k in ("rit_healthy_feet", "rit_chronic_wounds", "rit_other_nonwound_closeups"): v = p[k]; print(f"    {k:28s} {v['labelled']}/{v['n']} labelled ({v['as_3rd']} as 3rd degree)")
for n in PIPES: show(f"{n} at 0.60", res["at_0.60"][n])
print("composition of 'overall wrong':", json.dumps(res["composition_of_overall_wrong"]))
for k, v in res["val_matched_operating_points"].items():
    if v["threshold"] is not None: show(f"{k}: deployed at t={v['threshold']}", v["results"])
print("RE3@0.60 - deployed@pickA, 95% CI of count differences:", json.dumps(res.get("RE3_at_0.60_minus_deployed_at_pickA_95ci_counts")))
print("curve (descriptive): best (test right, RIT everyday right, t) with RIT feet <= level:")
for lv, v in res["curve_test_right_vs_rit_feet"].items(): print(f"   feet <= {lv:>4s}: {v}")

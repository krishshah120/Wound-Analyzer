"""v5b/v5c (PLAN.md addendum): classifiers that have seen healthy/everyday photos, with both live gates (0.5),
no crop, burn rule sum|max, t 0.60-0.75; same dev selection and verdict pool as eval_v5.py.
Classifier probabilities: clfviews_*.npz (full frame) and, for G4new, its saved gate outputs. -> eval_v5bc.json"""
import sys, csv, json, itertools
import numpy as np
sys.path.insert(0, "../v3")
from common import NAMES, OOD, B1, B2, B3, V3, WT, groups, paired_bootstrap
TO_M = np.array([{"abrasion": 0, "bruise": 1, "cut": 3, "out_of_scope": 4}.get(n, 2) for n in NAMES]); M = ["abrasion", "bruise", "possible_burn", "cut"]
SETS = ("val", "val_arm", "test", "test_arm", "rit", "heldout", "indep")
D = np.load(f"{V3}/views_deployed.npz"); G4 = np.load(f"{V3}/views_G4new_s1.npz"); F = np.load("../v4/framed_views.npz")
H = np.load("../v4/heldout_views.npz"); I = np.load("indep_healthy_views.npz")
SHIP = {"val": D["val_browser_gate"], "test": D["test_browser_gate"], "rit": D["rit_browser_gate"], "val_arm": F["val_gate_shipped"], "test_arm": F["test_gate_shipped"], "heldout": H["gate_shipped"], "indep": I["gate_shipped"]}
HEAL = {"val": G4["val_browser_gate"], "test": G4["test_browser_gate"], "rit": G4["rit_browser_gate"], "val_arm": F["val_gate_g4new"], "test_arm": F["test_gate_g4new"], "heldout": H["gate_g4new"], "indep": I["gate_g4new"]}
gated = {k: (SHIP[k][:, 0, OOD] >= 0.5) | (HEAL[k][:, 0, OOD] >= 0.5) for k in SETS}
Y = {"val": D["val_y"], "test": D["test_y"], "val_arm": F["val_y"], "test_arm": F["test_y"]}
CLF = {"deployed": dict(np.load("clfviews_deployed.npz")), "G4new": {k: HEAL[k][:, 0] for k in SETS},
       "N5": dict(np.load("clfviews_N5.npz")), "RE3_s1": dict(np.load("clfviews_RE3_s1.npz"))}
CLF["mean"] = {k: (CLF["deployed"][k] + CLF["G4new"][k]) / 2 for k in SETS}
def labels(p7, gt, rule, t):
    p = np.zeros((len(p7), 5))
    for k in range(7): p[:, TO_M[k]] += p7[:, k]
    best = p[:, :4].argmax(1); conf = p[:, :4].max(1)
    if rule == "max": conf = np.where(best == 2, p7[:, [B1, B2, B3]].max(1), conf)
    return np.where((p.argmax(1) != 4) & ~gt & (conf >= t), best, -1)
rows = list(csv.DictReader(open(f"{WT}/experiments/v2/rit/manifest.csv"))); seen, keep = set(), []
for i, r in enumerate(rows):
    if r["file"] not in seen: seen.add(r["file"]); keep.append(i)
keep = np.array(keep); R = [rows[i] for i in keep]
src = np.array([r["source"] for r in R]); role = np.array([r["eval_role"] for r in R]); rl = np.array([r["label"] for r in R])
rlm = np.where(np.char.startswith(rl, "burn"), "possible_burn", rl); ins = role == "in_scope"
RIT_FALSE = (src == "lower_limb_feet") | (src == "lower_limb_wounds") | ((src == "roboflow_wound_v1") & (role == "non_wound"))
def run(c):
    clf, rule, t = c; o = {}
    for k in ("val", "val_arm", "test", "test_arm"):
        lab = labels(CLF[clf][k], gated[k], rule, t); y = Y[k]; w = y != OOD; yt = TO_M[y]
        o[k] = dict(right=w & (lab == yt), wrong=w & (lab >= 0) & (lab != yt), false=~w & (lab >= 0), n_w=int(w.sum()), n_nw=int((~w).sum()))
    for k in ("heldout", "indep"): o[k] = dict(false=labels(CLF[clf][k], gated[k], rule, t) >= 0)
    L = np.array(["unknown" if l < 0 else M[l] for l in labels(CLF[clf]["rit"], gated["rit"], rule, t)[keep]])
    o["rit"] = dict(right=(ins & (L == rlm)) | ((role == "burn_any") & (L == "possible_burn")), wrong=ins & (L != "unknown") & (L != rlm), false=RIT_FALSE & (L != "unknown"),
                    feet=(src == "lower_limb_feet") & (L != "unknown"), chronic=(src == "lower_limb_wounds") & (L != "unknown"), burns=(role == "burn_any") & (L == "possible_burn"),
                    everyday_right=ins & (L == rlm), everyday_wrong=ins & (L != "unknown") & (L != rlm))
    return o
def dev(o): return dict(right=sum(int(o[k]["right"].sum()) for k in ("val", "val_arm")), wrong=sum(int(o[k]["wrong"].sum()) for k in ("val", "val_arm")),
                        false=sum(int(o[k]["false"].sum()) for k in ("val", "val_arm")) + int(o["heldout"]["false"].sum()))
def summary(c):
    o = res[c]; s = {"dev": dv[c]}
    for k in ("test", "test_arm"):
        r = o[k]; s[k] = dict(right=f"{r['right'].sum()}/{r['n_w']}", wrong=int(r["wrong"].sum()), nonwound=f"{r['false'].sum()}/{r['n_nw']}")
    for k, S in (("heldout", H), ("indep", I)):
        s[k] = {"all": f"{o[k]['false'].sum()}/{len(S['source'])}"}
        for so in sorted(set(S["source"])): s[k][so] = f"{(o[k]['false'] & (S['source'] == so)).sum()}/{(S['source'] == so).sum()}"
    s["rit"] = {k: int(o["rit"][k].sum()) for k in ("feet", "chronic", "burns", "everyday_right", "everyday_wrong")}
    return s
LIVE = ("deployed", "sum", 0.75); TS = (0.60, 0.65, 0.70, 0.75)
cands = list(itertools.product(("G4new", "mean", "N5"), ("sum", "max"), TS)); control = list(itertools.product(("RE3_s1",), ("sum", "max"), TS))
res = {c: run(c) for c in [LIVE] + cands + control}; dv = {c: dev(res[c]) for c in res}; L0 = dv[LIVE]
eligible = [c for c in cands if dv[c]["right"] > L0["right"] and dv[c]["false"] < L0["false"] and dv[c]["wrong"] <= L0["wrong"]]
chosen = min(eligible, key=lambda c: (dv[c]["false"], -dv[c]["right"], -c[2])) if eligible else None
tag = lambda c: f"{c[0]} {c[1]} t{c[2]}"
tg = groups("test"); ig = np.array([r["dup_group"] for r in csv.DictReader(open(f"{WT}/experiments/v4/new_neg_manifest.csv")) if r["split"] == "excluded_resembles_eval"])
grp = np.concatenate([np.char.add("t_", tg), np.char.add("i_", ig), np.char.add("r_", np.array([r["dup_group"] for r in R]))])
def pool(o, key):
    if key == "false": return np.concatenate([o["test"]["false"], o["indep"]["false"], o["rit"]["false"]])
    return np.concatenate([o["test"][key], np.zeros(len(ig), bool), o["rit"][key]])
def verdict(c):
    v = {}
    for key in ("right", "false", "wrong"):
        pa, pb = pool(res[c], key), pool(res[LIVE], key)
        v[key] = dict(candidate=int(pa.sum()), live=int(pb.sum()), diff_ci=paired_bootstrap(grp, lambda i: float(pa[i].sum()), lambda i: float(pb[i].sum())))
    v["confirmed"] = bool(v["right"]["candidate"] >= v["right"]["live"] and v["false"]["candidate"] < v["false"]["live"] and v["false"]["diff_ci"][1] < 0)
    return v
out = dict(live=tag(LIVE), live_summary=summary(LIVE), dev_all={tag(c): dv[c] for c in cands + control}, eligible=[tag(c) for c in eligible],
           chosen=None if chosen is None else tag(chosen), control_summary={tag(c): summary(c) for c in control})
if chosen: out["chosen_summary"] = summary(chosen); out["verdict_pool"] = verdict(chosen)
json.dump(out, open("eval_v5bc.json", "w"), indent=1, default=str)
print("LIVE", json.dumps(summary(LIVE)))
for c in cands + control: print(f"{tag(c):18s}", dv[c], "ELIGIBLE" if c in eligible else "", "(control)" if c in control else "")
if chosen: print("CHOSEN", tag(chosen), json.dumps(out["chosen_summary"])); print("VERDICT POOL", json.dumps(out["verdict_pool"]))
else: print("no eligible candidate")

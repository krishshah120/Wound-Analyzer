"""v6 (PLAN.md): live (mean(deployed, healthy), summed burn, 0.70, both gates) vs N6 combinations; N6k = control.
Dev = cleaned val close + cleaned val f50 + v4 held-out healthy + archive-8 held-out faces. Verdict pool = cleaned
test close + v5 independent normal skin + RIT (duplicates once). Inputs: views6_*.npz, views6_files.json, flags.csv.
-> eval6.json"""
import os, sys, csv, json, itertools
import numpy as np
V6 = os.path.dirname(os.path.abspath(__file__)); WT = os.path.dirname(os.path.dirname(V6)); EXP = f"{WT}/experiments"
sys.path.insert(0, f"{EXP}/v3")
from common import groups, paired_bootstrap
NAMES = ["abrasion", "bruise", "burn_1st_degree", "burn_2nd_degree", "burn_3rd_degree", "cut", "out_of_scope"]; OOD = 6
TO_M = np.array([0, 1, 2, 2, 2, 3, 4]); M = ["abrasion", "bruise", "possible_burn", "cut"]
FILES = json.load(open(f"{V6}/views6_files.json")); V = {n: dict(np.load(f"{V6}/views6_{n}.npz")) for n in ("deployed", "shipped", "healthy", "N6", "N6k")}
SETS = list(FILES)
gated = {k: (V["shipped"][k][:, OOD] >= 0.5) | (V["healthy"][k][:, OOD] >= 0.5) for k in SETS}
flag = {(r["split"], r["label"], r["file"]) for r in csv.DictReader(open(f"{V6}/flags.csv"))}
def meta(k):
    """y (class index or None), flagged mask, for labelled wound/non-wound sets."""
    if k in ("val", "test", "val_f50", "test_f50", "test_arm2"):
        sp = "val" if k.startswith("val") else "test"; ps = [p.split("/") for p in FILES[k]]
        return np.array([NAMES.index(p[-2]) for p in ps]), np.array([(sp, p[-2], p[-1]) in flag for p in ps])
    return None, None
Y, FL = {}, {}
for k in SETS: Y[k], FL[k] = meta(k)
CLF = {"deployed": V["deployed"], "N6": V["N6"], "N6k": V["N6k"], "healthy": V["healthy"]}
def probs(combo, k): return np.mean([CLF[m][k] for m in combo.split("+")], axis=0)
def labels(combo, t, k):
    p7 = probs(combo, k); p = np.zeros((len(p7), 5))
    for j in range(7): p[:, TO_M[j]] += p7[:, j]
    return np.where((p.argmax(1) != 4) & ~gated[k] & (p[:, :4].max(1) >= t), p[:, :4].argmax(1), -1)
rows = list(csv.DictReader(open(f"{EXP}/v2/rit/manifest.csv"))); seen, keep = set(), []
for i, r in enumerate(rows):
    if r["file"] not in seen: seen.add(r["file"]); keep.append(i)
keep = np.array(keep); R = [rows[i] for i in keep]
src = np.array([r["source"] for r in R]); role = np.array([r["eval_role"] for r in R]); rl = np.array([r["label"] for r in R])
rlm = np.where(np.char.startswith(rl, "burn"), "possible_burn", rl); ins = role == "in_scope"
RIT_FALSE = (src == "lower_limb_feet") | (src == "lower_limb_wounds") | ((src == "roboflow_wound_v1") & (role == "non_wound"))
src4 = {r["file"]: r["source"] for r in csv.DictReader(open(f"{EXP}/v4/new_neg_manifest.csv"))}
def run(c):
    combo, t = c; o = {}
    for k in ("val", "test", "val_f50", "test_f50", "test_arm2"):
        lab = labels(combo, t, k); y = Y[k]; w = y != OOD; yt = TO_M[y]
        for tag, m in (("", np.ones(len(y), bool)), ("_clean", ~FL[k])):
            o[k + tag] = dict(right=m & w & (lab == yt), wrong=m & w & (lab >= 0) & (lab != yt), false=m & ~w & (lab >= 0), n_w=int((m & w).sum()), n_nw=int((m & ~w).sum()))
    for k in ("heldout", "indep", "a8_heldout"): o[k] = dict(false=labels(combo, t, k) >= 0)
    L = np.array(["unknown" if l < 0 else M[l] for l in labels(combo, t, "rit")[keep]])
    o["rit"] = dict(right=(ins & (L == rlm)) | ((role == "burn_any") & (L == "possible_burn")), wrong=ins & (L != "unknown") & (L != rlm), false=RIT_FALSE & (L != "unknown"),
                    feet=(src == "lower_limb_feet") & (L != "unknown"), chronic=(src == "lower_limb_wounds") & (L != "unknown"), burns=(role == "burn_any") & (L == "possible_burn"),
                    everyday_right=ins & (L == rlm), everyday_wrong=ins & (L != "unknown") & (L != rlm))
    return o
def dev(o): return dict(right=int(o["val_clean"]["right"].sum() + o["val_f50_clean"]["right"].sum()), wrong=int(o["val_clean"]["wrong"].sum() + o["val_f50_clean"]["wrong"].sum()),
                        false=int(o["val_clean"]["false"].sum() + o["val_f50_clean"]["false"].sum() + o["heldout"]["false"].sum() + o["a8_heldout"]["false"].sum()))
def faces(o):
    f = lambda k: np.array([src4.get(p, "") == "oily_dry_skin_types" for p in FILES[k]])
    return dict(v4_faces=f"{int((o['heldout']['false'] & f('heldout')).sum() + (o['indep']['false'] & f('indep')).sum())}/{int(f('heldout').sum() + f('indep').sum())}",
                a8_faces=f"{int(o['a8_heldout']['false'].sum())}/{len(FILES['a8_heldout'])}")
def summary(c):
    o = res[c]; s = {"dev": dv[c], "faces": faces(o)}
    for k in ("test_clean", "test", "test_f50_clean", "test_arm2_clean", "test_arm2"):
        r = o[k]; named = int((r["right"] | r["wrong"]).sum())
        s[k] = dict(named=f"{named}/{r['n_w']}", right=int(r["right"].sum()), wrong=int(r["wrong"].sum()), nonwound=f"{int(r['false'].sum())}/{r['n_nw']}")
    s["normal_skin"] = {k: f"{int(o[k]['false'].sum())}/{len(FILES[k])}" for k in ("heldout", "indep", "a8_heldout")}
    s["rit"] = {k: int(o["rit"][k].sum()) for k in ("feet", "chronic", "burns", "everyday_right", "everyday_wrong")}
    return s
LIVE = ("deployed+healthy", 0.70); TS = (0.60, 0.65, 0.70, 0.75)
cands = list(itertools.product(("N6", "N6+healthy", "N6+deployed+healthy"), TS))
control = list(itertools.product(("N6k", "N6k+healthy", "N6k+deployed+healthy"), TS))
res = {c: run(c) for c in [LIVE] + cands + control}; dv = {c: dev(res[c]) for c in res}; L0 = dv[LIVE]
eligible = [c for c in cands if dv[c]["right"] > L0["right"] and dv[c]["false"] < L0["false"] and dv[c]["wrong"] <= L0["wrong"]]
chosen = min(eligible, key=lambda c: (dv[c]["false"], -dv[c]["right"], -c[1])) if eligible else None
tag = lambda c: f"{c[0]} t{c[1]}"
mt = ~FL["test"]; tg = groups("test")[mt]
ig = np.array([r["dup_group"] for r in csv.DictReader(open(f"{EXP}/v4/new_neg_manifest.csv")) if r["split"] == "excluded_resembles_eval"])
grp = np.concatenate([np.char.add("t_", tg), np.char.add("i_", ig), np.char.add("r_", np.array([r["dup_group"] for r in R]))])
def pool(o, key):
    if key == "false": return np.concatenate([o["test_clean"]["false"][mt], o["indep"]["false"], o["rit"]["false"]])
    return np.concatenate([o["test_clean"][key][mt], np.zeros(len(ig), bool), o["rit"][key]])
def verdict(c):
    v = {}
    for key in ("right", "false", "wrong"):
        pa, pb = pool(res[c], key), pool(res[LIVE], key)
        v[key] = dict(candidate=int(pa.sum()), live=int(pb.sum()), diff_ci=paired_bootstrap(grp, lambda i: float(pa[i].sum()), lambda i: float(pb[i].sum())))
    v["confirmed"] = bool(v["right"]["candidate"] > v["right"]["live"] and v["false"]["candidate"] < v["false"]["live"] and v["false"]["diff_ci"][1] < 0)
    return v
out = dict(live=tag(LIVE), live_summary=summary(LIVE), flags=len(flag), dev_all={tag(c): dv[c] for c in cands + control}, eligible=[tag(c) for c in eligible],
           chosen=None if chosen is None else tag(chosen), control_summary={tag(c): summary(c) for c in control if c[1] == 0.70})
if chosen: out["chosen_summary"] = summary(chosen); out["verdict_pool"] = verdict(chosen)
json.dump(out, open(f"{V6}/eval6.json", "w"), indent=1, default=str)
print("LIVE", json.dumps(summary(LIVE)))
for c in cands + control: print(f"{tag(c):26s}", dv[c], "ELIGIBLE" if c in eligible else "", "(control)" if c in control else "")
if chosen: print("CHOSEN", tag(chosen), json.dumps(out["chosen_summary"])); print("VERDICT", json.dumps(out["verdict_pool"]))
else: print("no eligible candidate")

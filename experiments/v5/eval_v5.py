"""v5 (PLAN.md): 64 decision-rule candidates on the existing models, chosen on dev only; test, RIT and the
independent normal-skin photos reported. Sources: v3/views_deployed + views_G4new_s1 (close-ups, RIT),
v4/framed_views (arm's length), v4/heldout_views (held-out healthy), v5/indep_healthy_views. -> eval_v5.json"""
import sys, csv, json, itertools
import numpy as np
sys.path.insert(0, "../v3")
from common import NAMES, OOD, B1, B2, B3, V3, WT, groups, paired_bootstrap, wilson
TO_M = np.array([{"abrasion": 0, "bruise": 1, "cut": 3, "out_of_scope": 4}.get(n, 2) for n in NAMES])
M = ["abrasion", "bruise", "possible_burn", "cut"]
def merged(p):
    out = np.zeros(p.shape[:-1] + (5,))
    for k in range(7): out[..., TO_M[k]] += p[..., k]
    return out
def labels(clf, gs, gg, rule, gthr, t):
    """Full frame only (no crop, as live). -> merged label 0..3, -1 = unknown."""
    p = merged(clf[:, 0]); best = p[:, :4].argmax(1); conf = p[:, :4].max(1)
    single = clf[:, 0][:, [B1, B2, B3]].max(1)
    bar = np.full(len(p), t)
    if rule == "max": conf = np.where(best == 2, single, conf)
    elif rule in ("sum85", "sum90"): bar = np.where(best == 2, max(t, int(rule[3:]) / 100), t)
    gated = (gs[:, 0, OOD] >= 0.5) | (gg[:, 0, OOD] >= gthr)
    return np.where((p.argmax(1) != 4) & ~gated & (conf >= bar), best, -1)
D = np.load(f"{V3}/views_deployed.npz"); G4 = np.load(f"{V3}/views_G4new_s1.npz"); F = np.load("../v4/framed_views.npz")
H = np.load("../v4/heldout_views.npz"); I = np.load("indep_healthy_views.npz")
WS = {f"{s}_close": (D[f"{s}_browser_clf"], D[f"{s}_browser_gate"], G4[f"{s}_browser_gate"], D[f"{s}_y"]) for s in ("val", "test")}
WS.update({f"{s}_arm": (F[f"{s}_clf"], F[f"{s}_gate_shipped"], F[f"{s}_gate_g4new"], F[f"{s}_y"]) for s in ("val", "test")})
HS = {"heldout": H, "indep": I}
rows = list(csv.DictReader(open(f"{WT}/experiments/v2/rit/manifest.csv"))); seen, keep = set(), []
for i, r in enumerate(rows):
    if r["file"] not in seen: seen.add(r["file"]); keep.append(i)
keep = np.array(keep); R = [rows[i] for i in keep]
src = np.array([r["source"] for r in R]); role = np.array([r["eval_role"] for r in R]); rl = np.array([r["label"] for r in R])
RIT_FALSE = (src == "lower_limb_feet") | (src == "lower_limb_wounds") | ((src == "roboflow_wound_v1") & (role == "non_wound"))
def run(c):
    rule, gthr, t = c; out = {}
    for k, (clf, gs, gg, y) in WS.items():
        lab = labels(clf, gs, gg, rule, gthr, t); w = y != OOD; yt = TO_M[y]
        out[k] = dict(right=w & (lab == yt), wrong=w & (lab >= 0) & (lab != yt), false=~w & (lab >= 0), n_w=int(w.sum()), n_nw=int((~w).sum()))
    for k, S in HS.items():
        lab = labels(S["clf"], S["gate_shipped"], S["gate_g4new"], rule, gthr, t); out[k] = dict(false=lab >= 0, src=S["source"])
    lab = labels(D["rit_browser_clf"], D["rit_browser_gate"], G4["rit_browser_gate"], rule, gthr, t)[keep]
    L = np.array(["unknown" if l < 0 else M[l] for l in lab]); ins = role == "in_scope"
    rlm = np.where(np.char.startswith(rl, "burn"), "possible_burn", rl)
    out["rit"] = dict(right=(ins & (L == rlm)) | ((role == "burn_any") & (L == "possible_burn")), wrong=ins & (L != "unknown") & (L != rlm),
                      false=RIT_FALSE & (L != "unknown"), feet=(src == "lower_limb_feet") & (L != "unknown"), chronic=(src == "lower_limb_wounds") & (L != "unknown"),
                      burns=(role == "burn_any") & (L == "possible_burn"), everyday_right=ins & (L == rlm), everyday_wrong=ins & (L != "unknown") & (L != rlm))
    return out
def dev(o): return dict(right=sum(int(o[k]["right"].sum()) for k in ("val_close", "val_arm")), wrong=sum(int(o[k]["wrong"].sum()) for k in ("val_close", "val_arm")),
                        false=sum(int(o[k]["false"].sum()) for k in ("val_close", "val_arm")) + int(o["heldout"]["false"].sum()))
LIVE = ("sum", 0.5, 0.75)
cands = list(itertools.product(("sum", "max", "sum85", "sum90"), (0.5, 0.4, 0.3, 0.2), (0.60, 0.65, 0.70, 0.75)))
res = {c: run(c) for c in cands}; dv = {c: dev(res[c]) for c in cands}; L0 = dv[LIVE]
eligible = [c for c in cands if dv[c]["right"] > L0["right"] and dv[c]["false"] < L0["false"] and dv[c]["wrong"] <= L0["wrong"]]
chosen = min(eligible, key=lambda c: (dv[c]["false"], -dv[c]["right"], -c[2])) if eligible else None
def tag(c): return f"{c[0]} gate{c[1]} t{c[2]}"
def summary(c):
    o = res[c]; s = {"dev": dv[c]}
    for k in ("test_close", "test_arm"):
        r = o[k]; s[k] = dict(right=f"{r['right'].sum()}/{r['n_w']}", wrong=int(r["wrong"].sum()), nonwound=f"{r['false'].sum()}/{r['n_nw']}")
    for k, S in HS.items():
        s[k] = {"all": f"{o[k]['false'].sum()}/{len(S['source'])}"}
        for so in sorted(set(S["source"])): s[k][so] = f"{(o[k]['false'] & (S['source'] == so)).sum()}/{(S['source'] == so).sum()}"
    r = o["rit"]; s["rit"] = {k: int(r[k].sum()) for k in ("feet", "chronic", "burns", "everyday_right", "everyday_wrong")}
    return s
# verdict pool: test close-ups + independent normal-skin + RIT, paired bootstrap over duplicate groups
tg = groups("test"); ig = np.array([r["dup_group"] for r in csv.DictReader(open(f"{WT}/experiments/v4/new_neg_manifest.csv")) if r["split"] == "excluded_resembles_eval"])
assert len(ig) == len(I["source"])
grp = np.concatenate([np.char.add("t_", tg), np.char.add("i_", ig), np.char.add("r_", np.array([r["dup_group"] for r in R]))])
def pool(o, key):
    z = np.zeros(len(ig), bool)
    if key == "false": return np.concatenate([o["test_close"]["false"], o["indep"]["false"], o["rit"]["false"]])
    return np.concatenate([o["test_close"][key], z, o["rit"][key]])
out = dict(live=tag(LIVE), live_summary=summary(LIVE), n_eligible=len(eligible), chosen=None if chosen is None else tag(chosen),
           dev_all={tag(c): dv[c] for c in cands}, eligible={tag(c): dv[c] for c in eligible})
if chosen:
    a, b = res[chosen], res[LIVE]; out["chosen_summary"] = summary(chosen); v = {}
    for key in ("right", "false", "wrong"):
        pa, pb = pool(a, key), pool(b, key)
        v[key] = dict(candidate=int(pa.sum()), live=int(pb.sum()), diff_ci=paired_bootstrap(grp, lambda i: float(pa[i].sum()), lambda i: float(pb[i].sum())))
    v["confirmed"] = bool(v["right"]["candidate"] >= v["right"]["live"] and v["false"]["candidate"] < v["false"]["live"] and v["false"]["diff_ci"][1] < 0)
    out["verdict_pool"] = v
json.dump(out, open("eval_v5.json", "w"), indent=1, default=str)
print("LIVE", tag(LIVE), json.dumps(summary(LIVE)))
print("eligible", len(eligible), "of", len(cands))
for c in sorted(eligible, key=lambda c: (dv[c]["false"], -dv[c]["right"])): print("  ", tag(c), dv[c])
if chosen: print("CHOSEN", tag(chosen), json.dumps(out["chosen_summary"])); print("VERDICT POOL", json.dumps(out["verdict_pool"]))

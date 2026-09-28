"""X1 (PLAN.md): does averaging existing runs of the deployed recipe reduce instability, and does the
fixed ensemble ENS3 (R_exact seeds 1-3) beat the deployed single model? Saved probabilities only.
Writes experiments/v3/x1_ensemble.json."""
import itertools, json
import numpy as np
from common import *

S = SHIPPED; gv, gt = S["val_gate"], S["test_gate"]; yv, yt = S["val_y"], S["test_y"]
EXACT = ["R_exact_a_s42", "R_exact_b_s42", "R_exact_s1", "R_exact_s2", "R_exact_s3"]
runs = {}
for tag in EXACT:
    d = np.load(f"{RUNS}/{tag}_probs.npz")
    assert list(d["names"]) == NAMES, tag                                  # class order
    assert (d["y_val"] == yv).all() and (d["y_test"] == yt).all(), tag      # photo order
    assert [f.split("/")[-1] for f in d["f_val"]] == files("val"), tag
    runs[tag] = (d["p_val"], d["p_test"])
dep = (S["val_clf"], S["test_clf"])
TARGET = 116                                                               # deployed val coverage 116/251
dep_ans, dep_best, _ = decide(dep[0], gv); DEP_ERR = 1 - (dep_ans & (yv != OOD) & (dep_best == yv)).sum() / (dep_ans & (yv != OOD)).sum()
out = {"deployed_val_error_at_0.60": round(100 * DEP_ERR, 2)}

def at_matched(pv):
    t = matched_threshold(pv, gv, yv, TARGET); a, b, _ = decide(pv, gv, t); return t, score(a, b, yv)

# 1. instability: singles vs every 3-member ensemble of the 5 exact-recipe runs
single = {k: dict(acc_matched=at_matched(v[0])[1]["acc_answered"], cov_060=score(*decide(v[0], gv)[:2], yv)["coverage"],
                  aurc=round(aurc(v[0], gv, yv), 3)) for k, v in runs.items()}
ens = {}
for combo in itertools.combinations(EXACT, 3):
    pv = np.mean([runs[k][0] for k in combo], 0)
    ens["+".join(combo)] = dict(acc_matched=at_matched(pv)[1]["acc_answered"], cov_060=score(*decide(pv, gv)[:2], yv)["coverage"], aurc=round(aurc(pv, gv, yv), 3))
def spread(d, k):
    v = np.array([x[k] for x in d.values()]); return dict(min=float(v.min()), max=float(v.max()), range=round(float(v.max() - v.min()), 1), sd=round(float(v.std(ddof=1)), 2), mean=round(float(v.mean()), 2))
out["instability"] = {"singles": single, "ensembles_of_3": ens,
                      "spread_singles": {k: spread(single, k) for k in ("acc_matched", "cov_060", "aurc")},
                      "spread_ensembles": {k: spread(ens, k) for k in ("acc_matched", "cov_060", "aurc")}}

# 2. deployed vs ENS3 (fixed members)
members = ["R_exact_s1", "R_exact_s2", "R_exact_s3"]
e3 = (np.mean([runs[k][0] for k in members], 0), np.mean([runs[k][1] for k in members], 0))
gval, gtest = groups("val"), groups("test")
res = {}
for name, (pv, pt) in (("deployed", dep), ("ENS3", e3)):
    r = {}
    for split, p, g, y in (("val", pv, gv, yv), ("test", pt, gt, yt)):
        r[f"{split}_at_0.60"] = score(*decide(p, g)[:2], y)
    t = matched_threshold(pv, gv, yv, TARGET)
    r["matched_coverage_threshold_from_val"] = round(t, 4)
    r["val_at_matched_coverage"] = score(*decide(pv, gv, t)[:2], yv)
    r["test_at_val_matched_threshold"] = score(*decide(pt, gt, t)[:2], yt)
    te = threshold_at_error(pv, gv, yv, DEP_ERR)
    r["matched_error_threshold_from_val"] = round(te, 4)
    r["val_at_matched_error"] = score(*decide(pv, gv, te)[:2], yv)
    r["test_at_val_matched_error_threshold"] = score(*decide(pt, gt, te)[:2], yt)
    r["val_aurc"] = round(aurc(pv, gv, yv), 3); r["test_aurc"] = round(aurc(pt, gt, yt), 3)
    res[name] = r
# paired grouped bootstrap: ENS3 - deployed
def cis(split, t_e, t_d):
    p_e, p_d = (e3[0], dep[0]) if split == "val" else (e3[1], dep[1])
    g, y, grp = (gv, yv, gval) if split == "val" else (gt, yt, gtest)
    ae, be, _ = decide(p_e, g, t_e); ad, bd, _ = decide(p_d, g, t_d)
    return dict(acc_answered=paired_bootstrap(grp, acc_stat(ae, be, y), acc_stat(ad, bd, y)),
                coverage=paired_bootstrap(grp, cov_stat(ae, y), cov_stat(ad, y)),
                nonwound_fp_pct=paired_bootstrap(grp, fp_stat(ae, y), fp_stat(ad, y)))
te_, td_ = res["ENS3"]["matched_coverage_threshold_from_val"], res["deployed"]["matched_coverage_threshold_from_val"]
res["ENS3_minus_deployed_95ci"] = {"val_at_0.60": cis("val", .6, .6), "test_at_0.60": cis("test", .6, .6),
                                   "val_matched_coverage": cis("val", te_, td_), "test_val_matched_threshold": cis("test", te_, td_)}

# 3. disagreement: does non-unanimity predict mistakes among ENS3's answers (val, matched coverage)?
tops_v = np.stack([runs[k][0].argmax(1) for k in members]); unan_v = (tops_v == tops_v[0]).all(0)
a, b, _ = decide(e3[0], gv, te_); aw = a & (yv != OOD); wrong = aw & (b != yv)
n_dis, n_un = int((aw & ~unan_v).sum()), int((aw & unan_v).sum())
err_dis = wrong[~unan_v].sum() / max(1, n_dis); err_un = wrong[unan_v].sum() / max(1, n_un)
use = n_dis >= 5 and err_dis >= 2 * err_un
dis = dict(answered_disagree=n_dis, wrong_disagree=int(wrong[~unan_v].sum()), answered_unanimous=n_un, wrong_unanimous=int(wrong[unan_v].sum()),
           err_disagree=round(100 * err_dis, 1), err_unanimous=round(100 * err_un, 1), rule_passes=bool(use))
if use:   # compare at matched coverage: confidence + abstain-on-disagreement, threshold lowered only to re-match coverage
    conf = e3[0][:, W].max(1); elig = (e3[0].argmax(1) != OOD) & (gv[:, OOD] < 0.5) & unan_v
    cand = np.sort(conf[elig & (yv != OOD)])[::-1]
    if len(cand) >= TARGET:
        t2 = float(cand[TARGET - 1]); aa, bb, _ = decide(e3[0], gv, t2); aa &= unan_v
        dis["val_matched_coverage_with_disagreement"] = score(aa, bb, yv); dis["threshold"] = round(t2, 4)
        tops_t = np.stack([runs[k][1].argmax(1) for k in members]); unan_t = (tops_t == tops_t[0]).all(0)
        at, bt, _ = decide(e3[1], gt, t2); dis["test_same_threshold_with_disagreement"] = score(at & unan_t, bt, yt)
    else:
        dis["note"] = "cannot reach matched coverage when abstaining on disagreement"
res["disagreement"] = dis
out["comparison"] = res
json.dump(out, open(f"{V3}/x1_ensemble.json", "w"), indent=1)

print("INSTABILITY (val)  singles vs 3-member ensembles")
for k in ("acc_matched", "cov_060", "aurc"):
    print(f"  {k:12s} singles {out['instability']['spread_singles'][k]}  |  ensembles {out['instability']['spread_ensembles'][k]}")
for name in ("deployed", "ENS3"):
    r = res[name]
    for key in ("val_at_0.60", "test_at_0.60", "val_at_matched_coverage", "test_at_val_matched_threshold", "val_at_matched_error", "test_at_val_matched_error_threshold"):
        s = r[key]; print(f"{name:8s} {key:36s} cov {s['coverage']:5.1f}% ({s['answered']}/{s['eligible']}) acc {s['acc_answered']:5.1f}% corr/elig {s['correct_of_eligible']:5.1f}% non-wound {s['nonwound_labelled']}/{s['nonwound']} A {s['tierA']} B {s['tierB']}")
    print(f"{name:8s} thresholds: matched-cov {r['matched_coverage_threshold_from_val']}  matched-err {r['matched_error_threshold_from_val']}  AURC val {r['val_aurc']} test {r['test_aurc']}")
print("ENS3 - deployed, 95% CI:", json.dumps(res["ENS3_minus_deployed_95ci"]))
print("disagreement:", {k: v for k, v in dis.items() if not isinstance(v, dict)})
for k in ("val_matched_coverage_with_disagreement", "test_same_threshold_with_disagreement"):
    if k in dis: s = dis[k]; print(f"  {k}: cov {s['coverage']}% acc {s['acc_answered']}% non-wound {s['nonwound_labelled']}/{s['nonwound']} A {s['tierA']}")

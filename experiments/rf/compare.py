"""Roboflow model vs deployed pipeline, exactly as PROTOCOL.md (fixed before any Roboflow output).
Inputs: eligible.csv, roboflow_raw.jsonl (run_roboflow.py), experiments/v3/views_deployed.npz (the
deployed pipeline's full-frame + crop probabilities for the same photos, same browser encoding).
Thresholds come from the clean VAL photos only; results are reported on clean test + RIT photos.
Usage: python compare.py [--raw roboflow_raw.jsonl] [--out comparison.json]"""
import os, sys, csv, json, argparse, collections
import numpy as np
RF = os.path.dirname(os.path.abspath(__file__)); WT = os.path.dirname(os.path.dirname(RF))
sys.path.insert(0, f"{WT}/src"); sys.path.insert(0, f"{WT}/experiments/v3")
import roboflow_client as rc
from common import NAMES, OOD, W, decide, files, wilson
ap = argparse.ArgumentParser(); ap.add_argument("--raw", default=f"{RF}/roboflow_raw.jsonl"); ap.add_argument("--out", default=f"{RF}/comparison.json"); a = ap.parse_args()
rows = [r for r in csv.DictReader(open(f"{RF}/eligible.csv")) if r["eligible"] == "yes"]
raw = {}
for line in open(a.raw):
    rec = json.loads(line); raw[rec["path"]] = rec
missing = [r["path"] for r in rows if r["path"] not in raw]
if missing: raise SystemExit(f"{len(missing)} eligible photos have no Roboflow result yet - run run_roboflow.py first")
errors = [p for p, r in raw.items() if "error" in r]
TRUTH = {"burn": "possible_burn"}
D = np.load(f"{WT}/experiments/v3/views_deployed.npz")
idx = {}
for split in ("val", "test"):
    for i, f in enumerate(files(split)): idx[(split, f)] = (split, i)
for i, f in enumerate(D["rit_files"]): idx[("rit", str(f))] = ("rit", i)
def key_of(path):
    parts = path.split("/")
    return ("rit", parts[-1]) if "/rit/images/" in path else (parts[1], parts[-1])
def deployed_label(path, t):
    split, i = idx[key_of(path)]; clf, gate = D[f"{split}_browser_clf"][i:i + 1], D[f"{split}_browser_gate"][i:i + 1]
    a0, b0, _ = decide(clf[:, 0], gate[:, 0], t); a1, b1, c1 = decide(clf[:, 1], gate[:, 1], t)
    lab = int(b0[0]) if a0[0] else (int(b1[0]) if a1[0] and c1[0] >= max(0.80, t) else -1)
    if lab < 0: return "unknown"
    n = NAMES[lab]; return "possible_burn" if n.startswith("burn") else n
from functools import lru_cache
deployed_label = lru_cache(maxsize=None)(deployed_label)
@lru_cache(maxsize=None)
def rf_label(path, tau):
    rec = raw[path]
    return "unknown" if "error" in rec else rc.answer_from_detections(rec["detections"], tau)[0]
dev = [r for r in rows if r["population"] == "val"]; ev = [r for r in rows if r["population"] != "val"]
WOUND = {"abrasion", "bruise", "cut", "possible_burn"}
truth = lambda r: TRUTH.get(r["shared_category"], r["shared_category"])
def stats(rs, labeler):
    s = {"per_category": {}, "nonwound": {}}
    for cat in sorted(WOUND):
        g = [r for r in rs if truth(r) == cat]; L = [labeler(r) for r in g]
        s["per_category"][cat] = dict(n=len(g), right=sum(l == cat for l in L), wrong=sum(l not in ("unknown", cat) for l in L), unanswered=sum(l == "unknown" for l in L))
    for pop, cond in (("other_nonwound", lambda r: r["shared_category"] == "non_wound" and r["population"] != "rit_healthy_feet"),
                      ("healthy_feet", lambda r: r["population"] == "rit_healthy_feet"), ("chronic_wounds", lambda r: r["population"] == "rit_chronic_wounds")):
        g = [r for r in rs if cond(r)]; L = [labeler(r) for r in g]
        s["nonwound"][pop] = dict(n=len(g), labelled=sum(l != "unknown" for l in L), as_burn=sum(l == "possible_burn" for l in L))
    wr = [r for r in rs if truth(r) in WOUND]; L = [labeler(r) for r in wr]
    s["all_wound"] = dict(n=len(wr), right=sum(l == truth(r) for l, r in zip(L, wr)), wrong=sum(l not in ("unknown", truth(r)) for l, r in zip(L, wr)),
                          unanswered=sum(l == "unknown" for l in L))
    s["burn_shown_other_injury"] = sum(1 for r, l in zip(wr, L) if truth(r) == "possible_burn" and l in WOUND - {"possible_burn"})
    ans = s["all_wound"]["right"] + s["all_wound"]["wrong"]; s["all_wound"]["acc_answered"] = round(100 * s["all_wound"]["right"] / ans, 1) if ans else None
    return s
# --- thresholds on DEV (clean val) ---
dep_dev = stats(dev, lambda r: deployed_label(r["path"], 0.60))["all_wound"]
grid = [round(x, 2) for x in np.arange(0.05, 0.955, 0.01)]
rf_dev = {t: stats(dev, lambda r: rf_label(r["path"], t))["all_wound"] for t in grid}
sel = [t for t in grid if rf_dev[t]["acc_answered"] is not None and rf_dev[t]["acc_answered"] >= dep_dev["acc_answered"]]
tau_sel = min(sel) if sel else None
dep_ans = dep_dev["right"] + dep_dev["wrong"]
cov = [t for t in grid if rf_dev[t]["right"] + rf_dev[t]["wrong"] >= dep_ans]
tau_cov = max(cov) if cov else None
t_match = None
if tau_sel is not None:
    rf_sel_ans = rf_dev[tau_sel]["right"] + rf_dev[tau_sel]["wrong"]
    dgrid = [round(x, 3) for x in np.arange(0.40, 0.951, 0.005)]
    ok = [t for t in dgrid if (lambda s: s["right"] + s["wrong"])(stats(dev, lambda r: deployed_label(r["path"], t))["all_wound"]) >= rf_sel_ans]
    t_match = max(ok) if ok else None
configs = {"deployed@0.60": lambda r: deployed_label(r["path"], 0.60)}
if tau_sel is not None: configs[f"roboflow@{tau_sel} (selected)"] = (lambda t: lambda r: rf_label(r["path"], t))(tau_sel)
if tau_cov is not None: configs[f"roboflow@{tau_cov} (matched coverage)"] = (lambda t: lambda r: rf_label(r["path"], t))(tau_cov)
if t_match is not None: configs[f"deployed@{t_match} (matched to roboflow selected)"] = (lambda t: lambda r: deployed_label(r["path"], t))(t_match)
if tau_sel is not None:
    configs["agreement (both same category)"] = lambda r: (lambda d, f: d if d == f else "unknown")(deployed_label(r["path"], 0.60), rf_label(r["path"], tau_sel))
res = dict(n_eligible=len(rows), n_dev=len(dev), n_eval=len(ev), roboflow_errors=len(errors),
           thresholds=dict(deployed_dev=dep_dev, tau_selected=tau_sel, tau_matched_coverage=tau_cov, deployed_matched_to_roboflow=t_match),
           eval={k: stats(ev, f) for k, f in configs.items()})
# --- paired bootstrap over duplicate groups (eval) ---
grp = np.array([r["dup_group"] for r in ev]); ug = np.unique(grp); gi = {g: np.nonzero(grp == g)[0] for g in ug}; rng = np.random.default_rng(0)
boots = [np.concatenate([gi[g] for g in rng.choice(ug, len(ug))]) for _ in range(2000)]
def vec(f, kind):
    L = [f(r) for r in ev]
    if kind == "burn_right": return np.array([truth(r) == "possible_burn" and l == "possible_burn" for r, l in zip(ev, L)], float)
    if kind == "wound_right": return np.array([truth(r) in WOUND and l == truth(r) for r, l in zip(ev, L)], float)
    if kind == "wound_wrong": return np.array([truth(r) in WOUND and l not in ("unknown", truth(r)) for r, l in zip(ev, L)], float)
    pop = {"fp_other": lambda r: r["shared_category"] == "non_wound" and r["population"] != "rit_healthy_feet",
           "fp_feet": lambda r: r["population"] == "rit_healthy_feet", "fp_chronic": lambda r: r["population"] == "rit_chronic_wounds"}[kind]
    return np.array([pop(r) and l != "unknown" for r, l in zip(ev, L)], float)
KINDS = ("burn_right", "wound_right", "wound_wrong", "fp_other", "fp_feet", "fp_chronic")
base = configs["deployed@0.60"]; res["paired_95ci_counts_vs_deployed@0.60"] = {}
for name, f in configs.items():
    if name == "deployed@0.60": continue
    res["paired_95ci_counts_vs_deployed@0.60"][name] = {}
    for k in KINDS:
        diff = vec(f, k) - vec(base, k)          # computed once, then resampled
        res["paired_95ci_counts_vs_deployed@0.60"][name][k] = [round(float(x), 1) for x in np.percentile([diff[b].sum() for b in boots], [2.5, 97.5])]
# verdict (PROTOCOL.md 4), for the selected point
verdict = "no Roboflow operating point could be selected on dev"
sk = next((k for k in configs if "(selected)" in k), None)
if sk:
    ci = res["paired_95ci_counts_vs_deployed@0.60"][sk]
    better = ci["burn_right"][0] > 0 or ci["fp_other"][1] < 0 or ci["fp_feet"][1] < 0
    worse = ci["burn_right"][1] < 0 or ci["fp_other"][0] > 0 or ci["fp_feet"][0] > 0 or ci["fp_chronic"][0] > 0
    verdict = "Roboflow better" if better and not worse else ("Roboflow worse" if worse and not better else "no demonstrated advantage / too small to tell")
res["verdict_selected_point"] = verdict
lat = [r["latency_s"] for r in raw.values() if "latency_s" in r]
res["latency_roboflow_roundtrip_s"] = dict(median=round(float(np.median(lat)), 3), p95=round(float(np.percentile(lat, 95)), 3), n=len(lat), measured_from="this Mac over the internet, not Vercel") if lat else None
rt = [json.loads(l) for l in open(f"{WT}/experiments/v3/runtime.jsonl")]
res["latency_deployed_local_ms"] = next((dict(median=r["latency_ms_median"], p95=r["latency_ms_p95"], note="full frame on this Mac; crop retry doubles it when used; not Vercel") for r in rt if r["classifiers"] == 1), None)
json.dump(res, open(a.out, "w"), indent=1)
print(json.dumps({k: v for k, v in res.items() if k != "eval"}, indent=1))
for name, s in res["eval"].items():
    w = s["all_wound"]; print(f"\n{name}: wound photos {w['right']} right / {w['wrong']} wrong / {w['unanswered']} unanswered of {w['n']} (acc when answered {w['acc_answered']}%), burn shown other injury {s['burn_shown_other_injury']}")
    for c, v in s["per_category"].items(): print(f"   {c:14s} {v['right']}/{v['n']} right, {v['wrong']} wrong, {v['unanswered']} unanswered")
    for p, v in s["nonwound"].items(): print(f"   {p:14s} {v['labelled']}/{v['n']} labelled ({v['as_burn']} as burn)")

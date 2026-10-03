"""v5 Roboflow vs the LIVE pipeline (and the v5 candidate, if one is confirmed), on every photo clean for both
models: the 3,311 rf evaluation photos (rf/eligible.csv minus val, duplicates once) + the clean held-out and
independent normal-skin photos (rf_healthy_eligible.csv). Same website-level rule for both: a label is
abrasion / bruise / cut / possible_burn / unknown; any burn degree = possible_burn for BOTH models.
Roboflow answer rule = rf/PROTOCOL.md section 2 (roboflow_client.answer_from_detections) at 0.40 (the
workflow's floor, most permissive) and 0.82 (the protocol's val-selected threshold).
Usage: python rf_compare_v5.py [CLF RULE T]   (candidate from eval_v5bc.py) -> rf_compare_v5.json"""
import os, sys, csv, json
import numpy as np
V5 = os.path.dirname(os.path.abspath(__file__)); WT = os.path.dirname(os.path.dirname(V5))
sys.path.insert(0, f"{WT}/src"); sys.path.insert(0, f"{WT}/experiments/v3")
import roboflow_client as rc
from common import files, paired_bootstrap
import eval_v5bc as E   # labels(), CLF, gated (prints its own table on import)
raw = {}
for fn in (f"{WT}/experiments/rf/roboflow_raw.jsonl", f"{V5}/rf_healthy_raw.jsonl"):
    for line in open(fn): rec = json.loads(line); raw[rec["path"]] = rec
idx = {("val", f): i for i, f in enumerate(files("val"))} | {("test", f): i for i, f in enumerate(files("test"))}
idx |= {("rit", str(f)): i for i, f in enumerate(E.D["rit_files"])}
hidx = {str(f): ("heldout", i) for i, f in enumerate(E.H["files"])} | {str(f): ("indep", i) for i, f in enumerate(E.I["files"])}
rows, seen = [], set()
for r in csv.DictReader(open(f"{WT}/experiments/rf/eligible.csv")):
    if r["eligible"] == "yes" and r["population"] != "val" and r["path"] not in seen:
        seen.add(r["path"]); p = r["path"].split("/")
        key = ("rit", p[-1]) if "/rit/images/" in r["path"] else (p[1], p[-1])
        truth = {"burn": "possible_burn"}.get(r["shared_category"], r["shared_category"])
        pop = {"rit_healthy_feet": "healthy_feet", "rit_chronic_wounds": "chronic_wounds"}.get(r["population"], "wound" if truth in ("abrasion", "bruise", "cut", "possible_burn") else "other_nonwound")
        rows.append(dict(path=r["path"], set=key[0], i=idx[key], truth=truth, pop=pop, grp="rf_" + r["dup_group"]))
mg = {r["file"]: r["dup_group"] for r in csv.DictReader(open(f"{WT}/experiments/v4/new_neg_manifest.csv"))}
for r in csv.DictReader(open(f"{V5}/rf_healthy_eligible.csv")):
    if r["eligible"] == "yes":
        s, i = hidx[r["file"]]
        pop = ("faces" if r["source"] == "oily_dry_skin_types" else "other_healthy_skin") + "_" + s
        rows.append(dict(path=r["file"], set=s, i=i, truth="non_wound", pop=pop, grp="h_" + mg[r["file"]]))
missing = [r["path"] for r in rows if r["path"] not in raw]
if missing: raise SystemExit(f"{len(missing)} photos lack a Roboflow result")
M = ["abrasion", "bruise", "possible_burn", "cut"]
def ours(c):
    lab = {k: E.labels(E.CLF[c[0]][k], E.gated[k], c[1], c[2]) for k in ("test", "rit", "heldout", "indep")}
    return np.array(["unknown" if lab[r["set"]][r["i"]] < 0 else M[lab[r["set"]][r["i"]]] for r in rows])
def rob(t): return np.array(["unknown" if "error" in raw[r["path"]] else rc.answer_from_detections(raw[r["path"]]["detections"], t)[0] for r in rows])
CFG = {"roboflow@0.40": rob(0.40), "roboflow@0.82": rob(0.82), "live": ours(E.LIVE)}
if len(sys.argv) == 4: CFG["candidate " + " ".join(sys.argv[1:])] = ours((sys.argv[1], sys.argv[2], float(sys.argv[3])))
truth = np.array([r["truth"] for r in rows]); pop = np.array([r["pop"] for r in rows]); grp = np.array([r["grp"] for r in rows])
wound = pop == "wound"
out = {"n_photos": len(rows), "errors": sum("error" in raw[r["path"]] for r in rows), "populations": {p: int((pop == p).sum()) for p in sorted(set(pop))}, "results": {}}
for name, L in CFG.items():
    s = {"wound": dict(n=int(wound.sum()), right=int((wound & (L == truth)).sum()), wrong=int((wound & (L != "unknown") & (L != truth)).sum()))}
    for p in sorted(set(pop) - {"wound"}):
        m = pop == p; s[p] = dict(n=int(m.sum()), labelled=int((m & (L != "unknown")).sum()), as_burn=int((m & (L == "possible_burn")).sum()))
    out["results"][name] = s
faces = np.char.startswith(pop, "faces"); nonw = ~wound
for name in [k for k in CFG if k != "live"]:
    A, B = CFG[name], CFG["live"]
    out["results"][name]["vs_live_ci"] = dict(
        right=paired_bootstrap(grp, lambda i: float((wound & (A == truth))[i].sum()), lambda i: float((wound & (B == truth))[i].sum())),
        false_all_nonwound=paired_bootstrap(grp, lambda i: float((nonw & (A != "unknown"))[i].sum()), lambda i: float((nonw & (B != "unknown"))[i].sum())),
        false_faces=paired_bootstrap(grp, lambda i: float((faces & (A != "unknown"))[i].sum()), lambda i: float((faces & (B != "unknown"))[i].sum())))
# facial injuries seen in val/test/RIT (looked at by eye after an OpenCV face detector; descriptive only)
FACIAL = ["data/val/burn_1st_degree/burn_1st_degree_kg2_315.jpg", "data/val/burn_3rd_degree/burn_3rd_degree_kg2_899.jpg",
          "data/test/burn_1st_degree/burn_1st_degree_kg2_390.jpg", "data/test/burn_1st_degree/burn_1st_degree_kg2_693.jpg",
          "data/test/burn_1st_degree/burn_1st_degree_kg2_828.jpg", "data/test/burn_2nd_degree/burn_2nd_degree_kg2_243.jpg",
          "data/test/burn_3rd_degree/burn_3rd_degree_kg2_329.jpg", "experiments/v2/rit/images/roboflow_wound_v1_12f06ca6df88332e.jpg"]
# kg2_693 looks like an uninjured face to the eye (flagged, not relabelled)
pos = {r["path"]: k for k, r in enumerate(rows)}
out["facial_injury_photos"] = {f: ({n: str(L[pos[f]]) for n, L in CFG.items()} if f in pos else "not in the clean evaluation set (val is dev, or resembles Roboflow training data)") for f in FACIAL}
json.dump(out, open(f"{V5}/rf_compare_v5.json", "w"), indent=1)
print(json.dumps({k: v for k, v in out.items() if k != "results"}, indent=1))
for name, s in out["results"].items(): print(name, json.dumps(s))

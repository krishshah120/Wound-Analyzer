"""v4 operating point (PLAN.md): "possible burn" output (3 burn probabilities summed), deployed classifier,
production crop retry, gate = the named views file's gate. Threshold = lowest of 0.60/0.55/0.50/0.45/0.40
with val (browser) non-wound labelled <= 13/94 (today's production). Test and RIT are reported only.
Usage: python op_point.py GATE_VIEWS_NAME   (e.g. G4new_s1, or deployed)  -> op_point_<name>.json"""
import sys, csv, json
import numpy as np
sys.path.insert(0, "../v3")
from common import NAMES, OOD, B3, V3, WT
GN = sys.argv[1]
D = np.load(f"{V3}/views_deployed.npz"); G = np.load(f"{V3}/views_{GN}.npz")
TO_M = np.array([{"abrasion": 0, "bruise": 1, "cut": 3, "out_of_scope": 4}.get(n, 2) for n in NAMES]); M = ["abrasion", "bruise", "burn", "cut"]
def merged(p):
    out = np.zeros(p.shape[:-1] + (5,))
    for k in range(7): out[..., TO_M[k]] += p[..., k]
    return out
def labels(split, t, gate_src, mode="burn"):
    clf, gate = D[f"{split}_browser_clf"], gate_src[f"{split}_browser_gate"]; views = []
    for v in (0, 1):
        p = merged(clf[:, v]) if mode == "burn" else clf[:, v]; ood = 4 if mode == "burn" else 6
        wi = [i for i in range(p.shape[1]) if i != ood]; conf = p[:, wi].max(1); best = np.array(wi)[p[:, wi].argmax(1)]
        views.append((np.where((p.argmax(1) != ood) & (gate[:, v, OOD] < 0.5) & (conf >= t), best, -1), conf))
    lab = views[0][0]; use = (lab < 0) & (views[1][0] >= 0) & (views[1][1] >= 0.80)
    return np.where(use, views[1][0], lab)
rows = list(csv.DictReader(open(f"{WT}/experiments/v2/rit/manifest.csv")))
seen, keep = set(), []
for i, r in enumerate(rows):                       # count each byte-identical RIT photo once
    if r["file"] not in seen: seen.add(r["file"]); keep.append(i)
keep = np.array(keep); src = np.array([r["source"] for r in rows])[keep]; role = np.array([r["eval_role"] for r in rows])[keep]; rl = np.array([r["label"] for r in rows])[keep]
def report(t, gate_src, mode):
    r = {}
    for split in ("val", "test"):
        y = D[f"{split}_y"]; lab = labels(split, t, gate_src, mode); w = y != OOD; yt = TO_M[y] if mode == "burn" else y
        r[split] = dict(n=int(w.sum()), answered=int((w & (lab >= 0)).sum()), right=int((w & (lab == yt)).sum()), wrong=int((w & (lab >= 0) & (lab != yt)).sum()),
                        unanswered=int((w & (lab < 0)).sum()), nonwound=int((~w & (lab >= 0)).sum()), nonwound_n=int((~w).sum()),
                        third_shown_non_burn=int(((y == B3) & (lab >= 0) & (lab != (2 if mode == "burn" else B3))).sum()))
    lab = labels("rit", t, gate_src, mode)[keep]
    L = np.array(["unknown" if l < 0 else (M[l] if mode == "burn" else NAMES[l]) for l in lab]); ins = role == "in_scope"
    r["rit"] = dict(everyday_right=int((ins & (L == rl)).sum()), everyday_wrong=int((ins & (L != "unknown") & (L != rl)).sum()), everyday_n=int(ins.sum()),
                    burns_recognised=int(((role == "burn_any") & np.char.startswith(L, "burn")).sum()), burns_n=int((role == "burn_any").sum()),
                    feet=int(((src == "lower_limb_feet") & (L != "unknown")).sum()), feet_n=int((src == "lower_limb_feet").sum()),
                    chronic=int(((src == "lower_limb_wounds") & (L != "unknown")).sum()), chronic_n=int((src == "lower_limb_wounds").sum()))
    return r
cands = [0.60, 0.55, 0.50, 0.45, 0.40]
sweep = {t: report(t, G, "burn") for t in cands}
ok = [t for t in cands if sweep[t]["val"]["nonwound"] <= 13]
chosen = min(ok) if ok else None
out = dict(gate=GN, rule="lowest threshold with val non-wound labelled <= 13/94", chosen=chosen, sweep={str(t): v for t, v in sweep.items()},
           today=report(0.60, D, "6class"))
json.dump(out, open(f"op_point_{GN}.json", "w"), indent=1)
def line(tag, r):
    v, te, R = r["val"], r["test"], r["rit"]
    return (f"{tag:22s} val {v['right']}/{v['wrong']}/{v['unanswered']} nw {v['nonwound']}/94 | test {te['right']}/{te['wrong']}/{te['unanswered']} nw {te['nonwound']}/450 3rd->non-burn {te['third_shown_non_burn']} | "
            f"RIT everyday {R['everyday_right']}r/{R['everyday_wrong']}w of {R['everyday_n']}, burns {R['burns_recognised']}/{R['burns_n']}, feet {R['feet']}/{R['feet_n']}, chronic {R['chronic']}/{R['chronic_n']}")
print("right/wrong/unanswered wound photos; nw = non-wound photos labelled")
print(line("today (6 cat, 0.60)", out["today"]))
for t in cands: print(line(f"possible burn @{t}", sweep[t]) + ("   <- chosen" if t == chosen else ""))

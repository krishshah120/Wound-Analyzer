"""v4b (LEDGER.md): possible-burn output with gate shipped / G4new / either, thresholds 0.60..0.95.
Selection on val + held-out healthy-skin photos only; test and RIT reported. -> fewer_false_alarms.json"""
import sys, csv, json
import numpy as np
sys.path.insert(0, "../v3")
from common import NAMES, OOD, B3, V3, WT
D = np.load(f"{V3}/views_deployed.npz"); G4 = np.load(f"{V3}/views_G4new_s1.npz"); H = np.load("heldout_views.npz")
TO_M = np.array([{"abrasion": 0, "bruise": 1, "cut": 3, "out_of_scope": 4}.get(n, 2) for n in NAMES])
def merged(p):
    out = np.zeros(p.shape[:-1] + (5,))
    for k in range(7): out[..., TO_M[k]] += p[..., k]
    return out
def run(clf, gate_s, gate_g, gate, t, mode="burn"):
    """clf/gates (n, 2 views, 7) -> labels (-1 unknown). mode 'burn' merged, '6class' original."""
    views = []
    for v in (0, 1):
        p = merged(clf[:, v]) if mode == "burn" else clf[:, v]; ood = 4 if mode == "burn" else 6
        wi = [i for i in range(p.shape[1]) if i != ood]; conf = p[:, wi].max(1); best = np.array(wi)[p[:, wi].argmax(1)]
        gated = {"shipped": gate_s[:, v, OOD] >= 0.5, "g4new": gate_g[:, v, OOD] >= 0.5,
                 "either": (gate_s[:, v, OOD] >= 0.5) | (gate_g[:, v, OOD] >= 0.5)}[gate]
        views.append((np.where((p.argmax(1) != ood) & ~gated & (conf >= t), best, -1), conf))
    lab = views[0][0]; use = (lab < 0) & (views[1][0] >= 0) & (views[1][1] >= max(0.80, t))
    return np.where(use, views[1][0], lab)
rows = list(csv.DictReader(open(f"{WT}/experiments/v2/rit/manifest.csv"))); seen, keep = set(), []
for i, r in enumerate(rows):
    if r["file"] not in seen: seen.add(r["file"]); keep.append(i)
keep = np.array(keep); src = np.array([r["source"] for r in rows])[keep]; role = np.array([r["eval_role"] for r in rows])[keep]; rl = np.array([r["label"] for r in rows])[keep]
M = ["abrasion", "bruise", "burn", "cut"]
def evaluate(gate, t, mode="burn"):
    r = {}
    for split in ("val", "test"):
        y = D[f"{split}_y"]; lab = run(D[f"{split}_browser_clf"], D[f"{split}_browser_gate"], G4[f"{split}_browser_gate"], gate, t, mode)
        w = y != OOD; yt = TO_M[y] if mode == "burn" else y
        r[split] = dict(right=int((w & (lab == yt)).sum()), wrong=int((w & (lab >= 0) & (lab != yt)).sum()), unanswered=int((w & (lab < 0)).sum()),
                        nonwound=int((~w & (lab >= 0)).sum()))
    lab = run(H["clf"], H["gate_shipped"], H["gate_g4new"], gate, t, mode); r["heldout_healthy"] = int((lab >= 0).sum())
    lab = run(D["rit_browser_clf"], D["rit_browser_gate"], G4["rit_browser_gate"], gate, t, mode)[keep]
    L = np.array(["unknown" if l < 0 else (M[l] if mode == "burn" else NAMES[l]) for l in lab]); ins = role == "in_scope"
    r["rit"] = dict(everyday_right=int((ins & (L == rl)).sum()), everyday_wrong=int((ins & (L != "unknown") & (L != rl)).sum()),
                    burns=int(((role == "burn_any") & np.char.startswith(L, "burn")).sum()), feet=int(((src == "lower_limb_feet") & (L != "unknown")).sum()),
                    chronic=int(((src == "lower_limb_wounds") & (L != "unknown")).sum()), closeups=int(((src == "roboflow_wound_v1") & (role == "non_wound") & (L != "unknown")).sum()))
    return r
before = evaluate("shipped", 0.60, "6class"); live = evaluate("shipped", 0.60, "burn")
opts = {(g, t): evaluate(g, t) for g in ("shipped", "g4new", "either") for t in [round(x, 2) for x in np.arange(0.60, 0.951, 0.05)]}
allowed = {k: v for k, v in opts.items() if v["val"]["nonwound"] <= 13 and v["heldout_healthy"] <= before["heldout_healthy"]}
chosen = max(allowed, key=lambda k: (allowed[k]["val"]["right"], k[1])) if allowed else None
out = dict(before=before, live_now=live, options={f"{g}@{t}": v for (g, t), v in opts.items()}, chosen=None if chosen is None else f"{chosen[0]}@{chosen[1]}")
json.dump(out, open("fewer_false_alarms.json", "w"), indent=1)
def line(tag, r):
    v, te, R = r["val"], r["test"], r["rit"]
    return (f"{tag:24s} val {v['right']:3d}r {v['wrong']:2d}w nw {v['nonwound']:2d}/94 | healthy held-out {r['heldout_healthy']:3d}/401 | test {te['right']:3d}r {te['wrong']:2d}w nw {te['nonwound']:3d}/450 | "
            f"RIT feet {R['feet']:4d}/1613 chronic {R['chronic']:4d}/1231 close-ups {R['closeups']}/22 burns {R['burns']}/67 everyday {R['everyday_right']}r/{R['everyday_wrong']}w")
print(line("BEFORE today (6 cat)", before)); print(line("LIVE NOW (burn, 0.60)", live))
for (g, t), v in opts.items(): print(line(f"{g}@{t}", v) + ("  allowed" if (g, t) in allowed else "") + ("  <== CHOSEN" if (g, t) == chosen else ""))

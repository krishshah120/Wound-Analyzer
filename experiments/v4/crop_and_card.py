"""v4c (LEDGER.md): crop-retry bar under the deployed config, and model-card figures.
Sources: close-ups = views_deployed + views_G4new_s1 (v3); arm's length = framed_views.npz; healthy =
heldout_views.npz. -> crop_and_card.json"""
import sys, json
import numpy as np
sys.path.insert(0, "../v3")
from common import NAMES, OOD, B3, V3, wilson
TO_M = np.array([{"abrasion": 0, "bruise": 1, "cut": 3, "out_of_scope": 4}.get(n, 2) for n in NAMES])
def merged(p):
    out = np.zeros(p.shape[:-1] + (5,))
    for k in range(7): out[..., TO_M[k]] += p[..., k]
    return out
def labels(clf, gs, gg, mode, t, gate, crop):
    """Returns (labels in the output's own index space, -1 unknown). crop None = no retry."""
    views = []
    for v in (0, 1):
        p = merged(clf[:, v]) if mode == "burn" else clf[:, v]; ood = 4 if mode == "burn" else 6
        wi = [i for i in range(p.shape[1]) if i != ood]; conf = p[:, wi].max(1); best = np.array(wi)[p[:, wi].argmax(1)]
        gated = gs[:, v, OOD] >= 0.5 if gate == "shipped" else (gs[:, v, OOD] >= 0.5) | (gg[:, v, OOD] >= 0.5)
        views.append((np.where((p.argmax(1) != ood) & ~gated & (conf >= t), best, -1), conf))
    lab = views[0][0]
    if crop is not None: lab = np.where((lab < 0) & (views[1][0] >= 0) & (views[1][1] >= crop), views[1][0], lab)
    return lab
D = np.load(f"{V3}/views_deployed.npz"); G4 = np.load(f"{V3}/views_G4new_s1.npz"); F = np.load("framed_views.npz"); H = np.load("heldout_views.npz")
SETS = {f"{s}_close": (D[f"{s}_browser_clf"], D[f"{s}_browser_gate"], G4[f"{s}_browser_gate"], D[f"{s}_y"]) for s in ("val", "test")}
SETS.update({f"{s}_arm": (F[f"{s}_clf"], F[f"{s}_gate_shipped"], F[f"{s}_gate_g4new"], F[f"{s}_y"]) for s in ("val", "test")})
def stats(name, mode, t, gate, crop):
    clf, gs, gg, y = SETS[name]; lab = labels(clf, gs, gg, mode, t, gate, crop); w = y != OOD
    yt = TO_M[y] if mode == "burn" else y
    return dict(named=int((w & (lab >= 0)).sum()), right=int((w & (lab == yt)).sum()), wrong=int((w & (lab >= 0) & (lab != yt)).sum()), n=int(w.sum()),
                nonwound=int((~w & (lab >= 0)).sum()), nonwound_n=int((~w).sum()), lab=lab, y=y)
def healthy(mode, t, gate, crop): return int((labels(H["clf"], H["gate_shipped"], H["gate_g4new"], mode, t, gate, crop) >= 0).sum())
out = {}
# 0. sanity: the old configuration must reproduce the current model card
old = {k: stats(k, "6class", 0.60, "shipped", 0.80) for k in ("test_close", "test_arm")}
out["old_config_reproduces_card"] = {k: {kk: v[kk] for kk in ("named", "right", "nonwound")} for k, v in old.items()}
print("old config (card says close 160/119/61, arm 101/69/42):", out["old_config_reproduces_card"])
# 1. crop sweep on development data
DEV = ("val_close", "val_arm")
base = {k: stats(k, "burn", 0.75, "either", None) for k in DEV}; base_h = healthy("burn", 0.75, "either", None)
sweep = {}
for bar in (0.80, 0.85, 0.90, 0.95):
    s = {k: stats(k, "burn", 0.75, "either", bar) for k in DEV}; h = healthy("burn", 0.75, "either", bar)
    add_right = sum(s[k]["right"] - base[k]["right"] for k in DEV)
    add_false = sum((s[k]["wrong"] - base[k]["wrong"]) + (s[k]["nonwound"] - base[k]["nonwound"]) for k in DEV) + (h - base_h)
    sweep[str(bar)] = dict(added_right=add_right, added_false=add_false, qualifies=bool(add_right >= 2 * add_false and add_right > 0))
ok = [float(b) for b, v in sweep.items() if v["qualifies"]]
chosen = min(ok) if ok else None
out["crop_sweep_dev"] = sweep; out["crop_chosen"] = chosen if chosen is not None else "remove"
print("crop sweep (dev):", sweep, "->", out["crop_chosen"])
# 2. model card figures: deployed today (0.80 crop) and with the chosen crop setting
def card(crop):
    c = {}
    for k in ("test_close", "test_arm"):
        s = stats(k, "burn", 0.75, "either", crop)
        c[k] = dict(named=f"{s['named']}/{s['n']}", named_pct=round(100 * s["named"] / s["n"], 1), named_ci=wilson(s["named"], s["n"]),
                    right=f"{s['right']}/{s['named']}", right_pct=round(100 * s["right"] / max(1, s["named"]), 1), right_ci=wilson(s["right"], s["named"]),
                    nonwound=f"{s['nonwound']}/{s['nonwound_n']}", nonwound_pct=round(100 * s["nonwound"] / s["nonwound_n"], 1), nonwound_ci=wilson(s["nonwound"], s["nonwound_n"]))
        if k == "test_close":
            y, lab = s["y"], s["lab"]; b3 = y == B3
            c["third_degree_test"] = dict(n=int(b3.sum()), possible_burn=int((b3 & (lab == 2)).sum()), other_injury=int((b3 & (lab >= 0) & (lab != 2)).sum()), unknown=int((b3 & (lab < 0)).sum()))
    return c
out["card_deployed_now_crop_0.80"] = card(0.80)
out["card_with_chosen_crop"] = card(None if chosen is None else chosen)
# like-for-like: the OLD configuration scored the same merged way (a burn of any degree answered as any burn = right)
lf = {}
for k in ("test_close", "test_arm"):
    s = stats(k, "6class", 0.60, "shipped", 0.80); yb = np.isin(s["y"], [2, 3, 4]); lb = np.isin(s["lab"], [2, 3, 4]); w = s["y"] != OOD
    right_m = int((w & ((s["lab"] == s["y"]) | (yb & lb))).sum())
    lf[k] = dict(named=s["named"], right_merged=right_m, right_merged_pct=round(100 * right_m / max(1, s["named"]), 1), right_strict=s["right"])
out["old_config_merged_scoring"] = lf
json.dump(out, open("crop_and_card.json", "w"), indent=1)
for k in ("card_deployed_now_crop_0.80", "card_with_chosen_crop"): print(k, json.dumps(out[k]))
print("old config, merged scoring:", lf)

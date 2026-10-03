"""X2 (PLAN.md): crop policies on the deployed models, whole pipeline, from views_<name>.npz.
  C0 full frame only;  C1 production centre-crop retry (crop answers only if full frame is unknown,
  crop not unknown and crop confidence >= 0.80);  C2 = C1 but the crop's answer must equal the full
  frame's best-guess wound class, else unknown.
Primary input (fixed before computing, 2026-09-26): 'browser' - the bytes the MRC site sends.
'raw' (stored files) is reported as well. Usage: python x2_crop.py [views_name] [gate_views_name]
The optional second name takes the GATE probabilities from another views file (X5)."""
import sys, csv, json, collections
import numpy as np
from common import *
NAME = sys.argv[1] if len(sys.argv) > 1 else "deployed"; GNAME = sys.argv[2] if len(sys.argv) > 2 else NAME
V = np.load(f"{V3}/views_{NAME}.npz"); VG = np.load(f"{V3}/views_{GNAME}.npz")
BURNS = {B1, B2, B3}

def policy(clf, gate, which):
    """clf, gate: (n, 2 views, 7). Returns labels (-1 = unknown) and best guess of the chosen view."""
    a0, b0, c0 = decide(clf[:, 0], gate[:, 0]); a1, b1, c1 = decide(clf[:, 1], gate[:, 1])
    lab = np.where(a0, b0, -1)
    if which == "C0": return lab, b0
    use = ~a0 & a1 & (c1 >= 0.80)
    if which == "C2": use &= (b1 == b0)
    return np.where(use, b1, lab), np.where(use, b1, b0)

def score_labels(lab, y):
    ans = lab >= 0; return score(ans, np.where(ans, lab, 0), y)

out = {}
for split in ("val", "test"):
    y = V[f"{split}_y"]; grp = groups(split)
    for mode in ("browser", "raw"):
        clf, gate = V[f"{split}_{mode}_clf"], VG[f"{split}_{mode}_gate"]
        labs = {k: policy(clf, gate, k)[0] for k in ("C0", "C1", "C2")}
        r = {k: score_labels(v, y) for k, v in labs.items()}
        for k in ("C1", "C2"):   # what the crop step adds over C0
            add = (labs[k] >= 0) & (labs["C0"] < 0)
            r[f"{k}_added"] = dict(wound_correct=int((add & (y != OOD) & (labs[k] == y)).sum()), wound_wrong=int((add & (y != OOD) & (labs[k] != y)).sum()),
                                   nonwound_labelled=int((add & (y == OOD)).sum()), tierA=int((add & (y == B3) & (labs[k] != B3)).sum()))
        a1, a2 = labs["C1"] >= 0, labs["C2"] >= 0
        r["C2_minus_C1_95ci"] = dict(acc_answered=paired_bootstrap(grp, acc_stat(a2, np.maximum(labs["C2"], 0), y), acc_stat(a1, np.maximum(labs["C1"], 0), y)),
                                     coverage=paired_bootstrap(grp, cov_stat(a2, y), cov_stat(a1, y)))
        out[f"{split}_{mode}"] = r
# pooled adoption rule on the primary input
pool = {k: sum(out[f"{s}_browser"][f"{k}_added"][m] for s in ("val", "test")) for k in ("C1", "C2") for m in ("wound_correct",)}
padd = {k: {m: sum(out[f"{s}_browser"][f"{k}_added"][m] for s in ("val", "test")) for m in ("wound_correct", "wound_wrong", "nonwound_labelled", "tierA")} for k in ("C1", "C2")}
wrong1 = padd["C1"]["wound_wrong"] + padd["C1"]["nonwound_labelled"]; wrong2 = padd["C2"]["wound_wrong"] + padd["C2"]["nonwound_labelled"]
adopt = wrong2 <= wrong1 / 2 and padd["C2"]["wound_correct"] >= padd["C1"]["wound_correct"] / 2 and padd["C2"]["tierA"] == 0
out["pooled_val_test_browser_added"] = padd; out["adopt_C2"] = bool(adopt)

# RIT (browser)
rows = list(csv.DictReader(open(f"{WT}/experiments/v2/rit/manifest.csv"))); assert [r["file"] for r in rows] == list(V["rit_files"])
clf, gate = V["rit_browser_clf"], VG["rit_browser_gate"]
lab_name = lambda l: "unknown" if l < 0 else NAMES[l]
rit = {}
for k in ("C0", "C1", "C2"):
    lab = [lab_name(l) for l in policy(clf, gate, k)[0]]; s = {}
    ins = [i for i, r in enumerate(rows) if r["eval_role"] == "in_scope"]; ans = [i for i in ins if lab[i] != "unknown"]
    cor = [i for i in ans if lab[i] == rows[i]["label"]]
    s["in_scope"] = dict(eligible=len(ins), answered=len(ans), correct=len(cor), coverage=round(100 * len(ans) / len(ins), 1), acc_answered=round(100 * len(cor) / max(1, len(ans)), 1),
                         acc_ci=wilson(len(cor), len(ans)), wrong=dict(collections.Counter(f"{rows[i]['label']}->{lab[i]}" for i in ans if lab[i] != rows[i]["label"])))
    fr = {}
    for lo, hi, nm in ((0, .15, "<15%"), (.15, .40, "15-40%"), (.40, 1.01, ">=40%")):
        g = [i for i in ins if rows[i]["box_area_fraction"] and lo <= float(rows[i]["box_area_fraction"]) < hi]
        fr[nm] = f"{sum(lab[i] != 'unknown' for i in g)}/{len(g)} answered, {sum(lab[i] == rows[i]['label'] for i in g)} correct"
    s["in_scope_by_real_framing"] = fr
    b = [i for i, r in enumerate(rows) if r["eval_role"] == "burn_any"]
    s["burns_degree_unknown"] = dict(n=len(b), any_burn=sum(lab[i].startswith("burn") for i in b), non_burn_label=sum(lab[i] not in ("unknown",) and not lab[i].startswith("burn") for i in b),
                                     unknown=sum(lab[i] == "unknown" for i in b))
    s["non_wound"] = {}
    for src in sorted({r["source"] for r in rows if r["eval_role"] == "non_wound"}):
        g = [i for i, r in enumerate(rows) if r["eval_role"] == "non_wound" and r["source"] == src]; fp = [i for i in g if lab[i] != "unknown"]
        s["non_wound"][src] = dict(n=len(g), labelled=len(fp), pct=round(100 * len(fp) / len(g), 1), ci=wilson(len(fp), len(g)),
                                   burn_3rd=sum(lab[i] == "burn_3rd_degree" for i in fp), labels=dict(collections.Counter(lab[i] for i in fp)))
    rit[k] = s
out["rit_browser"] = rit
json.dump(out, open(f"{V3}/x2_crop_{NAME}{'' if GNAME == NAME else '_gate-' + GNAME}.json", "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))

for key in ("val_browser", "test_browser", "val_raw", "test_raw"):
    for k in ("C0", "C1", "C2"):
        s = out[key][k]; print(f"{key:12s} {k}: cov {s['coverage']:5.1f}% ({s['answered']}/{s['eligible']}) acc {s['acc_answered']:5.1f}% corr/elig {s['correct_of_eligible']:5.1f}% non-wound {s['nonwound_labelled']}/{s['nonwound']} A {s['tierA']} B {s['tierB']}")
    print(f"{'':12s} added by crop: C1 {out[key]['C1_added']}  C2 {out[key]['C2_added']}  C2-C1 CI {out[key]['C2_minus_C1_95ci']}")
print("pooled val+test browser added:", padd, "-> adopt C2:", adopt)
for k in ("C0", "C1", "C2"):
    s = rit[k]; i = s["in_scope"]
    print(f"RIT {k}: in-scope {i['answered']}/{i['eligible']} answered, {i['correct']} correct {i['acc_ci']} wrong {i['wrong']} | framing {s['in_scope_by_real_framing']} | burns {s['burns_degree_unknown']}")
    for src, v in s["non_wound"].items(): print(f"      non-wound {src}: {v['labelled']}/{v['n']} = {v['pct']}% {v['ci']} (3rd degree {v['burn_3rd']})")

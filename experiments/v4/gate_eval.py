"""v4 gate verdict (PLAN.md): deployed classifier + production crop retry + gate threshold 0.5,
6-category output, only the gate swapped. Compares shipped gate / G4ctrl / G4new.
Usage: python gate_eval.py G4new_s1 G4ctrl_s1   -> gate_eval_<cand>.json"""
import os, io, sys, csv, json, importlib.util
import numpy as np
from PIL import Image
V4 = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, f"{os.path.dirname(V4)}/v3")
from common import NAMES, OOD, B1, B2, B3, V3, WT, decide, score
CAND, CTRL = sys.argv[1], sys.argv[2]
D = np.load(f"{V3}/views_deployed.npz")
G = {"shipped": D, CTRL: np.load(f"{V3}/views_{CTRL}.npz"), CAND: np.load(f"{V3}/views_{CAND}.npz")}
def policy(clf, gate):
    a0, b0, _ = decide(clf[:, 0], gate[:, 0]); a1, b1, c1 = decide(clf[:, 1], gate[:, 1])
    lab = np.where(a0, b0, -1); return np.where(~a0 & a1 & (c1 >= 0.80), b1, lab)
rows = list(csv.DictReader(open(f"{WT}/experiments/v2/rit/manifest.csv")))
src = np.array([r["source"] for r in rows]); role = np.array([r["eval_role"] for r in rows])
RY = np.array([NAMES.index(r["label"]) if r["label"] in NAMES else -9 for r in rows])
# held-out 20% of the new sources: production pipeline, deployed classifier, each gate
spec = importlib.util.spec_from_file_location("prod", f"{WT}/experiments/v2/production_snapshot/predict.py"); prod = importlib.util.module_from_spec(spec); spec.loader.exec_module(prod)
from ai_edge_litert.interpreter import Interpreter
import tensorflow as tf
clf_it = Interpreter(model_path=f"{WT}/experiments/v2/production_snapshot/wound_model.tflite"); clf_it.allocate_tensors()
ship_it = Interpreter(model_path=f"{WT}/experiments/v2/production_snapshot/out_of_scope_gate.tflite"); ship_it.allocate_tensors()
gate_fn = {"shipped": lambda x: prod._run(ship_it, x)}
for n in (CTRL, CAND):
    m = tf.keras.models.load_model(f"{V4}/models/{n}.keras", compile=False); gate_fn[n] = (lambda m: lambda x: m(x, training=False).numpy()[0])(m)
held = [r for r in csv.DictReader(open(f"{V4}/new_neg_manifest.csv")) if r["split"] == "heldout"]
def browser(path):
    img = Image.open(path).convert("RGB"); b = io.BytesIO(); img.save(b, "JPEG", quality=82); return Image.open(io.BytesIO(b.getvalue())).convert("RGB")
hv = [[prod._to_tensor(v) for v in prod._views(browser(f"{WT}/{r['file']}"))] for r in held]
hclf = [[prod._run(clf_it, x) for x in vs] for vs in hv]
res = {}
for name, V in G.items():
    r = {}
    for split in ("val", "test"):
        for inp in ("browser", "raw"):
            lab = policy(D[f"{split}_{inp}_clf"], V[f"{split}_{inp}_gate"]); y = D[f"{split}_y"]
            r[f"{split}_{inp}"] = dict(score(lab >= 0, np.maximum(lab, 0), y), correct_vec=((lab == y) & (y != OOD)).tolist())
    lab = policy(D["rit_browser_clf"], V["rit_browser_gate"])
    ins = role == "in_scope"
    r["rit"] = dict(everyday_answered=int((ins & (lab >= 0)).sum()), everyday_right=int((ins & (lab == RY)).sum()), everyday_wrong=int((ins & (lab >= 0) & (lab != RY)).sum()),
                    burns_any_burn=int(((role == "burn_any") & np.isin(lab, [B1, B2, B3])).sum()),
                    feet=int(((src == "lower_limb_feet") & (lab >= 0)).sum()), feet_3rd=int(((src == "lower_limb_feet") & (lab == B3)).sum()),
                    chronic=int(((src == "lower_limb_wounds") & (lab >= 0)).sum()), closeups=int(((src == "roboflow_wound_v1") & (role == "non_wound") & (lab >= 0)).sum()),
                    everyday_answered_vec=(ins & (lab >= 0)).tolist())
    hl = []
    for vs, cs in zip(hv, hclf):
        gs = [gate_fn[name](x) for x in vs]; P = np.array(cs)[None]; Gt = np.array(gs)[None]
        hl.append(int(policy(P, Gt)[0]))
    hl = np.array(hl); hsrc = np.array([r_["source"] for r_ in held])
    r["heldout_new_sources"] = {s: f"{int(((hsrc == s) & (hl >= 0)).sum())}/{int((hsrc == s).sum())}" for s in sorted(set(hsrc))}
    res[name] = r
dep, ctl, cand = res["shipped"], res[CTRL], res[CAND]
lost = {inp: int(sum(a and not b for a, b in zip(dep[f"val_{inp}"]["correct_vec"], cand[f"val_{inp}"]["correct_vec"]))) for inp in ("browser", "raw")}
ev_lost = int(sum(a and not b for a, b in zip(dep["rit"]["everyday_answered_vec"], cand["rit"]["everyday_answered_vec"])))
checks = {"feet_half_of_shipped": cand["rit"]["feet"] <= dep["rit"]["feet"] / 2, "feet_half_of_control": cand["rit"]["feet"] <= ctl["rit"]["feet"] / 2,
          "val_correct_lost_le_5_browser": lost["browser"] <= 5, "val_correct_lost_le_5_raw": lost["raw"] <= 5,
          "rit_everyday_answers_lost_le_2": ev_lost <= 2,
          "val_nonwound_le_16_browser": cand["val_browser"]["nonwound_labelled"] <= 16, "val_nonwound_le_16_raw": cand["val_raw"]["nonwound_labelled"] <= 16,
          "val_tierA_not_worse": cand["val_browser"]["tierA"] <= dep["val_browser"]["tierA"] and cand["val_raw"]["tierA"] <= dep["val_raw"]["tierA"]}
out = dict(checks=checks, passed=all(checks.values()), val_correct_lost=lost, rit_everyday_answers_lost=ev_lost,
           results={n: {k: ({kk: vv for kk, vv in v.items() if not kk.endswith("_vec")} if isinstance(v, dict) else v) for k, v in r.items()} for n, r in res.items()})
json.dump(out, open(f"{V4}/gate_eval_{CAND}.json", "w"), indent=1)
for n, r in out["results"].items():
    print(f"== {n}: RIT feet {r['rit']['feet']}/1613 (3rd {r['rit']['feet_3rd']}), chronic {r['rit']['chronic']}/1313, close-ups {r['rit']['closeups']}/22, "
          f"everyday {r['rit']['everyday_right']} right / {r['rit']['everyday_wrong']} wrong of 48, burns {r['rit']['burns_any_burn']}/67 | held-out new sources {r['heldout_new_sources']}")
    for k in ("val_browser", "val_raw", "test_browser"):
        s = r[k]; print(f"    {k:12s} {s['answered']}/{s['eligible']} answered, {s['correct']} right, non-wound {s['nonwound_labelled']}/{s['nonwound']}, 3rd->other {s['tierA']}")
print("val correct lost:", lost, "| RIT everyday answers lost:", ev_lost); print("checks:", checks); print("PASSED:", out["passed"])

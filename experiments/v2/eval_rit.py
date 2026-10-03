"""Score a pipeline on the Reserved Independent Test (RIT). Default: the DEPLOYED pipeline, run with
the production code itself (production_snapshot/predict.py). Photos go in the way the MRC site sends
them (longest edge <= 1024, JPEG quality 82). Writes experiments/v2/rit_eval_<name>.json.
Usage: python eval_rit.py [name]"""
import os, io, sys, csv, json, importlib.util, collections
import numpy as np
from PIL import Image
V2 = os.path.dirname(os.path.abspath(__file__)); SNAP = f"{V2}/production_snapshot"; RIT = f"{V2}/rit"
NAME = sys.argv[1] if len(sys.argv) > 1 else "deployed"
spec = importlib.util.spec_from_file_location("prod", f"{SNAP}/predict.py"); prod = importlib.util.module_from_spec(spec); spec.loader.exec_module(prod)
prod.MODEL_PATH, prod.GATE_PATH, prod.CLASS_PATH = f"{SNAP}/wound_model.tflite", f"{SNAP}/out_of_scope_gate.tflite", f"{SNAP}/class_names.json"
interp, gate, names = prod._load()
BURNS = {"burn_1st_degree", "burn_2nd_degree", "burn_3rd_degree"}

def wilson(k, n, z=1.96):
    if n == 0: return None
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(100 * (c - h), 1), round(100 * (c + h), 1)]
def browser(raw):
    img = Image.open(io.BytesIO(raw)).convert("RGB"); w, h = img.size; s = min(1.0, 1024 / max(w, h))
    if s < 1: img = img.resize((round(w * s), round(h * s)), Image.BILINEAR)
    b = io.BytesIO(); img.save(b, "JPEG", quality=82); return b.getvalue()
def first_view(raw):
    x = prod._to_tensor(Image.open(io.BytesIO(raw)).convert("RGB")); p, g = prod._run(interp, x), prod._run(gate, x)
    return prod.decide(p, names, g)[0]

rows = list(csv.DictReader(open(f"{RIT}/manifest.csv")))
res = []
for r in rows:
    raw = browser(open(f"{RIT}/images/{r['file']}", "rb").read())
    label = prod._classify(raw)[0]
    res.append(dict(r, production=label, **{"classifier+gate": first_view(raw)}))
out = {}
for stage in ("classifier+gate", "production"):
    s = {}
    ins = [x for x in res if x["eval_role"] == "in_scope"]
    ans = [x for x in ins if x[stage] != "unknown"]; cor = [x for x in ans if x[stage] == x["label"]]
    s["in_scope"] = dict(eligible=len(ins), answered=len(ans), correct=len(cor), coverage=round(100 * len(ans) / len(ins), 1), coverage_ci=wilson(len(ans), len(ins)),
                         acc_answered=round(100 * len(cor) / max(1, len(ans)), 1), acc_answered_ci=wilson(len(cor), len(ans)),
                         correct_of_eligible=round(100 * len(cor) / len(ins), 1),
                         wrong_labels=dict(collections.Counter(f"{x['label']}->{x[stage]}" for x in ans if x[stage] != x["label"])),
                         per_class={c: dict(n=sum(x["label"] == c for x in ins), answered=sum(x["label"] == c and x[stage] != "unknown" for x in ins),
                                            correct=sum(x["label"] == c and x[stage] == c for x in ins)) for c in ("abrasion", "bruise", "cut")})
    strata = {}
    for lo, hi, nm in ((0, .15, "<15%"), (.15, .40, "15-40%"), (.40, 1.01, ">=40%")):
        g = [x for x in ins if x["box_area_fraction"] and lo <= float(x["box_area_fraction"]) < hi]
        ga = [x for x in g if x[stage] != "unknown"]
        strata[nm] = dict(n=len(g), answered=len(ga), correct=sum(x[stage] == x["label"] for x in ga),
                          coverage=round(100 * len(ga) / max(1, len(g)), 1), coverage_ci=wilson(len(ga), len(g)))
    s["in_scope_by_real_framing"] = strata
    b = [x for x in res if x["eval_role"] == "burn_any"]
    s["burn_degree_unknown"] = dict(n=len(b), any_burn_label=sum(x[stage] in BURNS for x in b), non_burn_wound_label=sum(x[stage] not in BURNS and x[stage] != "unknown" for x in b),
                                    unknown=sum(x[stage] == "unknown" for x in b), non_burn_labels=dict(collections.Counter(x[stage] for x in b if x[stage] not in BURNS and x[stage] != "unknown")))
    nw = {}
    for src in sorted({x["source"] for x in res if x["eval_role"] == "non_wound"}):
        g = [x for x in res if x["eval_role"] == "non_wound" and x["source"] == src]; fp = [x for x in g if x[stage] != "unknown"]
        nw[src] = dict(n=len(g), labelled=len(fp), fpr=round(100 * len(fp) / max(1, len(g)), 1), fpr_ci=wilson(len(fp), len(g)), labels=dict(collections.Counter(x[stage] for x in fp)))
    s["non_wound"] = nw
    out[stage] = s
json.dump(dict(pipeline=NAME, n=len(res), results=out), open(f"{V2}/rit_eval_{NAME}.json", "w"), indent=1)
json.dump(res, open(f"{V2}/rit_rows_{NAME}.json", "w"))
for stage, s in out.items():
    i = s["in_scope"]; b = s["burn_degree_unknown"]
    print(f"[{stage}] in-scope: coverage {i['answered']}/{i['eligible']} = {i['coverage']}% {i['coverage_ci']}, acc when answered {i['acc_answered']}% {i['acc_answered_ci']}, correct of eligible {i['correct_of_eligible']}%")
    print("    per class", i["per_class"], "| wrong", i["wrong_labels"])
    print("    by real framing", {k: f"{v['answered']}/{v['n']} answered, {v['correct']} correct" for k, v in s["in_scope_by_real_framing"].items()})
    print(f"    burns (degree unknown) n={b['n']}: any-burn label {b['any_burn_label']}, NON-burn label {b['non_burn_wound_label']} {b['non_burn_labels']}, unknown {b['unknown']}")
    for src, v in s["non_wound"].items(): print(f"    non-wound {src}: {v['labelled']}/{v['n']} labelled = {v['fpr']}% {v['fpr_ci']} {v['labels']}")

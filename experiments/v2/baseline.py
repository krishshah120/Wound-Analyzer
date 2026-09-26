"""Step 1: reproduce the DEPLOYED pipeline's predictions with the production code itself
(production_snapshot/predict.py and its .tflite files), and pin down what each earlier
headline number meant. Unbuffered; writes experiments/v2/baseline.json.

Inputs, per photo, two ways:
  raw     - the stored file's bytes (224x224 JPEG) sent as-is
  browser - re-encoded as JPEG quality 82, as the MRC site's canvas does before upload
Metrics (wound photos = the six injury classes; non-wound = out_of_scope folder):
  coverage            answered / eligible wound photos
  acc_answered        correct / answered
  correct_of_eligible correct / eligible
  fpr_nonwound        non-wound photos given any wound label / non-wound photos
  tierA               true 3rd degree burns shown a DIFFERENT wound label (reassuring)
  tierB               true 2nd degree burns shown 1st degree / abrasion / bruise
"""
import os, sys, io, json, time, resource, importlib.util
import numpy as np
from PIL import Image
V2 = os.path.dirname(os.path.abspath(__file__)); WT = os.path.dirname(os.path.dirname(V2))
SNAP = f"{V2}/production_snapshot"
sys.path.insert(0, f"{WT}/src"); os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

spec = importlib.util.spec_from_file_location("prod", f"{SNAP}/predict.py")
prod = importlib.util.module_from_spec(spec); spec.loader.exec_module(prod)
prod.MODEL_PATH, prod.GATE_PATH, prod.CLASS_PATH = f"{SNAP}/wound_model.tflite", f"{SNAP}/out_of_scope_gate.tflite", f"{SNAP}/class_names.json"
t0 = time.time(); interp, gate, names = prod._load(); load_s = time.time() - t0
rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6   # macOS reports bytes
OOD = "out_of_scope"

def wilson(k, n, z=1.96):
    if n == 0: return None
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(100 * (c - h), 1), round(100 * (c + h), 1)]

def files(split):
    out = []
    for c in names:
        d = f"{WT}/data/{split}/{c}"
        out += [(f"{d}/{f}", c) for f in sorted(os.listdir(d)) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    return out

def browser_bytes(path):
    img = Image.open(path).convert("RGB"); w, h = img.size; s = min(1.0, 1024 / max(w, h))
    if s < 1: img = img.resize((round(w * s), round(h * s)), Image.BILINEAR)
    b = io.BytesIO(); img.save(b, "JPEG", quality=82); return b.getvalue()

def first_view(raw):
    """Production's first view only (no crop retry) - the classifier + gate stage."""
    img = Image.open(io.BytesIO(raw)).convert("RGB"); x = prod._to_tensor(img)
    p, g = prod._run(interp, x), prod._run(gate, x)
    return prod.decide(p, names, g), p, g

def score(rows):
    wound = [r for r in rows if r["truth"] != OOD]; non = [r for r in rows if r["truth"] == OOD]
    out = {}
    for stage in ("classifier", "classifier+gate", "production"):
        lab = lambda r: r[stage]
        ans = [r for r in wound if lab(r) != "unknown"]; cor = [r for r in ans if lab(r) == r["truth"]]
        fp = [r for r in non if lab(r) != "unknown"]
        b3 = [r for r in wound if r["truth"] == "burn_3rd_degree"]; b2 = [r for r in wound if r["truth"] == "burn_2nd_degree"]
        per = {}
        for c in [n for n in names if n != OOD]:
            cr = [r for r in wound if r["truth"] == c]; ca = [r for r in cr if lab(r) != "unknown"]
            per[c] = dict(n=len(cr), answered=len(ca), correct=sum(lab(r) == c for r in ca), abstained=len(cr) - len(ca))
        out[stage] = dict(
            eligible=len(wound), answered=len(ans), correct=len(cor),
            coverage=round(100 * len(ans) / len(wound), 1), coverage_ci=wilson(len(ans), len(wound)),
            acc_answered=round(100 * len(cor) / max(1, len(ans)), 1), acc_answered_ci=wilson(len(cor), len(ans)),
            correct_of_eligible=round(100 * len(cor) / len(wound), 1),
            nonwound=len(non), nonwound_labelled=len(fp), fpr_nonwound=round(100 * len(fp) / max(1, len(non)), 1), fpr_ci=wilson(len(fp), len(non)),
            tierA_3rd_shown_other_wound=sum(lab(r) not in ("unknown", "burn_3rd_degree") for r in b3), n_3rd=len(b3),
            tierB_2nd_shown_1st_abr_bruise=sum(lab(r) in ("burn_1st_degree", "abrasion", "bruise") for r in b2), n_2nd=len(b2),
            per_class=per)
    return out

import tensorflow as tf
from model import load_trained_model, load_gate_model, decide as repo_decide
kclf, knames = load_trained_model(); kgate = load_gate_model()
assert knames == names

result = dict(runtime=dict(load_seconds=round(load_s, 3), max_rss_mb_after_load=round(rss_mb, 1),
                           model_bytes={"wound_model.tflite": os.path.getsize(f"{SNAP}/wound_model.tflite"),
                                        "out_of_scope_gate.tflite": os.path.getsize(f"{SNAP}/out_of_scope_gate.tflite")},
                           host="this Mac (Apple Silicon, LiteRT CPU) - NOT the Vercel runtime"))
for split in ("val", "test"):
    fl = files(split)
    for mode in ("raw", "browser"):
        rows, lat, maxdiff = [], [], 0.0
        for path, truth in fl:
            raw = open(path, "rb").read() if mode == "raw" else browser_bytes(path)
            (lab1, conf1, bg1), p, g = first_view(raw)
            t1 = time.perf_counter(); label, conf, bg, dist, goos = prod._classify(raw); lat.append(time.perf_counter() - t1)
            clf_only = "unknown" if (names[int(np.argmax(p))] == OOD or float(max(p[i] for i in range(len(names)) if names[i] != OOD)) < prod.CONFIDENCE_THRESHOLD) else bg1
            rows.append(dict(path=os.path.relpath(path, WT), truth=truth, classifier=clf_only, **{"classifier+gate": lab1}, production=label,
                             conf=round(conf1, 4), gate_oos=round(float(g[names.index(OOD)]), 4)))
            if mode == "raw":   # deployed TFLite vs repo Keras on the same pixels
                x = tf.keras.applications.mobilenet_v2.preprocess_input(tf.keras.utils.img_to_array(tf.keras.utils.load_img(path, target_size=(224, 224))))[None]
                kp, kg = kclf.predict(x, verbose=0)[0], kgate.predict(x, verbose=0)[0]
                maxdiff = max(maxdiff, float(np.abs(kp - p).max()), float(np.abs(kg - g).max()))
                if repo_decide(kp, names, kg)[0] != lab1: rows[-1]["keras_disagrees"] = True
        key = f"{split}_{mode}"
        result[key] = score(rows)
        result[key]["latency_ms_median"] = round(1000 * float(np.median(lat)), 1)
        result[key]["latency_ms_p95"] = round(1000 * float(np.percentile(lat, 95)), 1)
        if mode == "raw":
            result[key]["deployed_vs_keras_max_prob_diff"] = maxdiff
            result[key]["deployed_vs_keras_decision_mismatches"] = sum(1 for r in rows if r.get("keras_disagrees"))
        json.dump(rows, open(f"{V2}/baseline_rows_{key}.json", "w"))
        s = result[key]
        print(f"{key}: n wound {s['production']['eligible']} non-wound {s['production']['nonwound']} | "
              + " | ".join(f"{st}: cov {s[st]['coverage']}% acc {s[st]['acc_answered']}% corr/elig {s[st]['correct_of_eligible']}% "
                           f"FP {s[st]['nonwound_labelled']}/{s[st]['nonwound']} A {s[st]['tierA_3rd_shown_other_wound']} B {s[st]['tierB_2nd_shown_1st_abr_bruise']}"
                           for st in ("classifier", "classifier+gate", "production"))
              + (f" | keras max diff {s.get('deployed_vs_keras_max_prob_diff', 0):.1e}, decision mismatches {s.get('deployed_vs_keras_decision_mismatches', '-')}" if mode == "raw" else ""), flush=True)
json.dump(result, open(f"{V2}/baseline.json", "w"), indent=1)
print("runtime:", result["runtime"])

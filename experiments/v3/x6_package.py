"""X6b: package retrained members as deployable, reproducible artifacts.
For each member: TFLite export (src/export_tflite.convert, float32 - the production method), checked
against the Keras model on every val and test photo (production loader: RGB, NEAREST 224, x/127.5-1);
manifest with seed, sha256 of both files, class list, preprocessing, training/early-stop data manifest
and dependency versions. Usage: python x6_package.py SET_NAME RUN_NAME SEED [SEED...]
Writes experiments/v3/artifacts/SET_NAME/ (member_s*.tflite + manifest.json; .keras stays in models/)."""
import os, sys, json, csv, hashlib, platform, importlib.metadata as md
import numpy as np
V3 = os.path.dirname(os.path.abspath(__file__)); WT = os.path.dirname(os.path.dirname(V3)); V2 = f"{WT}/experiments/v2"
sys.path.insert(0, f"{WT}/src"); os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import tensorflow as tf
from export_tflite import convert
from litert_model import load_image
from ai_edge_litert.interpreter import Interpreter
SET, RUN, SEEDS = sys.argv[1], sys.argv[2], [int(s) for s in sys.argv[3:]]
OUT = f"{V3}/artifacts/{SET}"; os.makedirs(OUT, exist_ok=True)
NAMES = json.load(open(f"{WT}/models/class_names.json")); OOD = NAMES.index("out_of_scope"); W = [i for i in range(7) if i != OOD]
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()

dm = f"{V3}/artifacts/data_manifest_b84.csv"
if not os.path.exists(dm):   # every training and early-stopping file, by content hash
    with open(dm, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["role", "class", "file", "sha256"])
        for role, root in (("train", f"{V2}/b84_train"), ("early_stop", f"{V2}/b84_val")):
            for c in sorted(os.listdir(root)):
                for fn in sorted(os.listdir(f"{root}/{c}")): w.writerow([role, c, fn, sha(os.path.realpath(f"{root}/{c}/{fn}"))])
paths = [f"{WT}/data/{s}/{c}/{f}" for s in ("val", "test") for c in NAMES for f in sorted(os.listdir(f"{WT}/data/{s}/{c}")) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
X = np.stack([load_image(p) for p in paths])
members = []
for s in SEEDS:
    kp = f"{V3}/models/{RUN}_s{s}.keras"; tp = f"{OUT}/member_s{s}.tflite"
    model = tf.keras.models.load_model(kp, compile=False)
    convert(model, tp)
    pk = model.predict(X, batch_size=32, verbose=0)
    it = Interpreter(model_path=tp); it.allocate_tensors(); i_in, i_out = it.get_input_details()[0]["index"], it.get_output_details()[0]["index"]
    pt = []
    for x in X:
        it.set_tensor(i_in, x[None].astype(np.float32)); it.invoke(); pt.append(it.get_tensor(i_out)[0])
    pt = np.array(pt)
    diff = float(np.abs(pk - pt).max())
    dec = lambda p: np.where((p.argmax(1) == OOD) | (p[:, W].max(1) < 0.60), -1, np.array(W)[p[:, W].argmax(1)])
    mism = int((dec(pk) != dec(pt)).sum())
    if diff > 1e-4 or mism: raise SystemExit(f"member s{s}: TFLite differs from Keras (max {diff:.2e}, {mism} decision changes) - not packaged")
    run = json.load(open(f"{WT}/experiments/runs/{RUN}_s{s}.json"))
    members.append(dict(seed=s, tflite=os.path.basename(tp), tflite_sha256=sha(tp), keras_path=os.path.relpath(kp, WT), keras_sha256=sha(kp),
                        tflite_vs_keras=dict(photos=len(paths), max_prob_diff=diff, decision_changes=mism),
                        epochs=run["epochs"], train_minutes=round(run["minutes"], 1), config=run["config"]))
    print(f"member s{s}: exported, max diff {diff:.1e}, 0 decision changes", flush=True)
manifest = dict(set=SET, kind="ensemble: mean of member probabilities" if len(SEEDS) > 1 else "single model",
                class_names=NAMES, preprocessing="RGB; PIL NEAREST resize to 224x224; float32 x/127.5 - 1 (src/litert_model.load_image)",
                decision="production decide(): unknown if out_of_scope is the top class, gate p(out_of_scope) >= 0.5, or top wound probability < 0.60; centre 70% crop retry accepted at >= 0.80",
                gate=dict(path="models/out_of_scope_gate.tflite", sha256=sha(f"{WT}/models/out_of_scope_gate.tflite"), note="shipped gate, unchanged"),
                recipe="experiments/experiment9.py --alpha 1.4 --label-smoothing 0.1 --train-dir experiments/v2/b84_train --val-dir experiments/v2/b84_val --seed S --save-model ...",
                data_manifest=dict(path=os.path.relpath(dm, WT), sha256=sha(dm)),
                versions=dict(python=platform.python_version(), **{p: md.version(p) for p in ("tensorflow", "keras", "numpy", "pillow", "ai-edge-litert")}),
                hardware=f"{platform.machine()} {platform.platform()}; CPU training is not bit-deterministic (v2): a rerun with the same seed gives a different model",
                members=members)
json.dump(manifest, open(f"{OUT}/manifest.json", "w"), indent=1)
json.dump(NAMES, open(f"{OUT}/class_names.json", "w"))
print("wrote", os.path.relpath(OUT, WT))

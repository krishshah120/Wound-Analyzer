"""Measured runtime on THIS Mac (Apple M4 CPU, LiteRT float32) - not the Vercel runtime.
Usage: python runtime.py N_CLASSIFIERS [gate.tflite] -> one JSON line: load seconds, peak RSS, per-photo
latency (full frame only; the crop retry doubles it when it runs). Ensemble members share the deployed
architecture, so N copies of the deployed classifier time an N-member ensemble exactly."""
import os, sys, io, csv, json, time, resource, importlib.util
import numpy as np
from PIL import Image
V3 = os.path.dirname(os.path.abspath(__file__)); WT = os.path.dirname(os.path.dirname(V3)); SNAP = f"{WT}/experiments/v2/production_snapshot"
N = int(sys.argv[1]); GATE = sys.argv[2] if len(sys.argv) > 2 else f"{SNAP}/out_of_scope_gate.tflite"
spec = importlib.util.spec_from_file_location("prod", f"{SNAP}/predict.py"); prod = importlib.util.module_from_spec(spec); spec.loader.exec_module(prod)
from ai_edge_litert.interpreter import Interpreter
t0 = time.time()
clfs = []
for _ in range(N):
    it = Interpreter(model_path=f"{SNAP}/wound_model.tflite"); it.allocate_tensors(); clfs.append(it)
g = Interpreter(model_path=GATE); g.allocate_tensors(); load = time.time() - t0
rows = list(csv.DictReader(open(f"{WT}/experiments/v2/rit/manifest.csv")))[::15][:200]
imgs = [open(f"{WT}/experiments/v2/rit/images/{r['file']}", "rb").read() for r in rows]
lat = []
for raw in imgs:
    t1 = time.perf_counter()
    x = prod._to_tensor(Image.open(io.BytesIO(raw)).convert("RGB"))
    p = np.mean([prod._run(c, x) for c in clfs], 0); prod._run(g, x)
    lat.append(time.perf_counter() - t1)
print(json.dumps(dict(classifiers=N, load_s=round(load, 3), peak_rss_mb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 1),
                      latency_ms_median=round(1000 * float(np.median(lat)), 1), latency_ms_p95=round(1000 * float(np.percentile(lat, 95)), 1),
                      photos=len(lat), model_bytes=N * os.path.getsize(f"{SNAP}/wound_model.tflite") + os.path.getsize(GATE),
                      note="includes JPEG decode + resize of RIT photos (<= original size)" )))

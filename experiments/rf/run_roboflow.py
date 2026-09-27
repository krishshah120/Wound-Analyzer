"""Runs the Roboflow workflow once on every photo that is clean for both models (eligible.csv), with the
same bytes the MRC site sends (longest edge <= 1024 px, JPEG quality 82). Resumable: appends one JSON
line per photo to roboflow_raw.jsonl (class, confidence, box only; no image data) with the measured
round-trip time. Each call sends the photo to Roboflow's servers and uses serverless credits.
Usage: python run_roboflow.py [--limit N]"""
import os, io, sys, csv, json, time, argparse
from PIL import Image
RF = os.path.dirname(os.path.abspath(__file__)); WT = os.path.dirname(os.path.dirname(RF))
sys.path.insert(0, f"{WT}/src")
import roboflow_client as rc
ap = argparse.ArgumentParser(); ap.add_argument("--limit", type=int, default=None); a = ap.parse_args()
key = rc.load_api_key()
rows = [r for r in csv.DictReader(open(f"{RF}/eligible.csv")) if r["eligible"] == "yes"]
OUT = f"{RF}/roboflow_raw.jsonl"
done = {json.loads(l)["path"] for l in open(OUT)} if os.path.exists(OUT) else set()
todo = [r for r in rows if r["path"] not in done][: a.limit]
print(f"{len(rows)} eligible photos, {len(done)} done, running {len(todo)}", flush=True)
def browser(path):
    img = Image.open(path).convert("RGB"); w, h = img.size; s = min(1.0, 1024 / max(w, h))
    if s < 1: img = img.resize((round(w * s), round(h * s)), Image.BILINEAR)
    b = io.BytesIO(); img.save(b, "JPEG", quality=82); return b.getvalue()
with open(OUT, "a") as f:
    for i, r in enumerate(todo):
        raw = browser(f"{WT}/{r['path']}")
        t0 = time.perf_counter()
        try:
            out = rc.run_wound_workflow(raw, api_key=key)
            rec = dict(path=r["path"], latency_s=round(time.perf_counter() - t0, 3), output_keys=sorted(out[0].keys()), detections=rc.find_detections(out[0]))
        except rc.RoboflowAuthError:
            raise
        except rc.RoboflowError as e:
            rec = dict(path=r["path"], error=f"{type(e).__name__}: {str(e)[:200]}")
        f.write(json.dumps(rec) + "\n"); f.flush()
        if i % 50 == 0: print(f"{i + 1}/{len(todo)}", flush=True)
print("done")

"""v5 diagnosis: which photos contain a prominent face (OpenCV Haar frontal-face detector, run in a
scratch venv: opencv-python-headless). A face counts if its box is >= 10% of the shorter image side.
Detector errors both ways are expected; used only to measure association, never in the analyzer.
-> faces.csv (file, face 0/1)"""
import os, csv, json, sys
import cv2
WT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
NAMES = json.load(open(f"{WT}/models/class_names.json"))
det = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
def face(path):
    im = cv2.imread(path)
    if im is None: raise RuntimeError(f"unreadable {path}")
    g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY); s = min(g.shape)
    if s > 640: g = cv2.resize(g, None, fx=640 / s, fy=640 / s); s = min(g.shape)
    return int(len(det.detectMultiScale(g, 1.1, 6, minSize=(int(0.10 * s),) * 2)) > 0)
paths = [f"data/{sp}/{c}/{f}" for sp in ("train", "val", "test") for c in NAMES
         for f in sorted(os.listdir(f"{WT}/data/{sp}/{c}")) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
paths += [r["file"] for r in csv.DictReader(open(f"{WT}/experiments/v4/new_neg_manifest.csv")) if r["split"] in ("heldout", "train", "excluded_resembles_eval")]
paths += sorted({"experiments/v2/rit/images/" + r["file"] for r in csv.DictReader(open(f"{WT}/experiments/v2/rit/manifest.csv"))})
with open(f"{WT}/experiments/v5/faces.csv", "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["file", "face"])
    for i, p in enumerate(paths):
        w.writerow([p, face(f"{WT}/{p}")])
        if i % 1000 == 0: print(i, len(paths), flush=True)

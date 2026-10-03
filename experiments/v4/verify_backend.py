"""Runs the MODIFIED server code (~/Downloads/wound-analyzer-vercel/api/predict.py, its own models) on
every val/test/RIT photo, browser-encoded, and compares with op_point_deployed.json (possible burn @0.60)."""
import os, io, csv, json, importlib.util, sys
import numpy as np
from PIL import Image
WT = "/Users/vihaa/Downloads/Wound-Analyzer/.claude/worktrees/silly-wilson-17d01a"
spec = importlib.util.spec_from_file_location("srv", os.path.expanduser("~/Downloads/wound-analyzer-vercel/api/predict.py"))
srv = importlib.util.module_from_spec(spec); spec.loader.exec_module(srv)
def browser(raw):
    img = Image.open(io.BytesIO(raw)).convert("RGB"); w, h = img.size; s = min(1.0, 1024 / max(w, h))
    if s < 1: img = img.resize((round(w * s), round(h * s)), Image.BILINEAR)
    b = io.BytesIO(); img.save(b, "JPEG", quality=82); return b.getvalue()
MAP = lambda c: "possible_burn" if c.startswith("burn") else c
res = {}
for split in ("val", "test"):
    right = wrong = una = nw = 0
    for c in ["abrasion", "bruise", "burn_1st_degree", "burn_2nd_degree", "burn_3rd_degree", "cut", "out_of_scope"]:
        for f in sorted(os.listdir(f"{WT}/data/{split}/{c}")):
            if not f.lower().endswith((".jpg", ".jpeg", ".png")): continue
            lab = srv._classify(browser(open(f"{WT}/data/{split}/{c}/{f}", "rb").read()))[0]
            if c == "out_of_scope": nw += lab != "unknown"
            elif lab == "unknown": una += 1
            elif lab == MAP(c): right += 1
            else: wrong += 1
    res[split] = dict(right=right, wrong=wrong, unanswered=una, nonwound=nw)
rows, seen = list(csv.DictReader(open(f"{WT}/experiments/v2/rit/manifest.csv"))), set()
R = dict(everyday_right=0, everyday_wrong=0, burns=0, feet=0, chronic=0)
for r in rows:
    if r["file"] in seen: continue
    seen.add(r["file"]); lab = srv._classify(browser(open(f"{WT}/experiments/v2/rit/images/{r['file']}", "rb").read()))[0]
    if r["eval_role"] == "in_scope":
        R["everyday_right"] += lab == r["label"]; R["everyday_wrong"] += lab not in ("unknown", r["label"])
    elif r["eval_role"] == "burn_any": R["burns"] += lab == "possible_burn"
    elif r["source"] == "lower_limb_feet": R["feet"] += lab != "unknown"
    elif r["source"] == "lower_limb_wounds": R["chronic"] += lab != "unknown"
res["rit"] = R
exp = json.load(open("op_point_deployed.json"))["sweep"]["0.6"]
want = dict(val={k: exp["val"][k] for k in ("right", "wrong", "unanswered", "nonwound")}, test={k: exp["test"][k] for k in ("right", "wrong", "unanswered", "nonwound")},
            rit=dict(everyday_right=exp["rit"]["everyday_right"], everyday_wrong=exp["rit"]["everyday_wrong"], burns=exp["rit"]["burns_recognised"], feet=exp["rit"]["feet"], chronic=exp["rit"]["chronic"]))
print("server :", json.dumps(res)); print("expected:", json.dumps(want)); print("MATCH" if res == want else "MISMATCH")
json.dump(dict(server=res, expected=want, match=res == want), open("verify_backend.json", "w"), indent=1)

"""Runs the MODIFIED server (two gates, 0.75) on every val/test/RIT photo and the 401 held-out healthy-skin
photos, browser-encoded, and compares with the simulation of either@0.75 WITHOUT the crop retry (crop_and_card.py)."""
import os, io, csv, json, importlib.util
from PIL import Image
WT = "/Users/vihaa/Downloads/Wound-Analyzer/.claude/worktrees/silly-wilson-17d01a"
spec = importlib.util.spec_from_file_location("srv", os.path.expanduser("~/Downloads/wound-analyzer-vercel/api/predict.py"))
srv = importlib.util.module_from_spec(spec); spec.loader.exec_module(srv)
def browser(raw):
    img = Image.open(io.BytesIO(raw)).convert("RGB"); w, h = img.size; s = min(1.0, 1024 / max(w, h))
    if s < 1: img = img.resize((round(w * s), round(h * s)), Image.BILINEAR)
    b = io.BytesIO(); img.save(b, "JPEG", quality=82); return b.getvalue()
MAP = lambda c: "possible_burn" if c.startswith("burn") else c
got = {}
for split in ("val", "test"):
    r = dict(right=0, wrong=0, unanswered=0, nonwound=0)
    for c in ["abrasion", "bruise", "burn_1st_degree", "burn_2nd_degree", "burn_3rd_degree", "cut", "out_of_scope"]:
        for f in sorted(os.listdir(f"{WT}/data/{split}/{c}")):
            if not f.lower().endswith((".jpg", ".jpeg", ".png")): continue
            lab = srv._classify(browser(open(f"{WT}/data/{split}/{c}/{f}", "rb").read()))[0]
            if c == "out_of_scope": r["nonwound"] += lab != "unknown"
            elif lab == "unknown": r["unanswered"] += 1
            elif lab == MAP(c): r["right"] += 1
            else: r["wrong"] += 1
    got[split] = r
held = [r for r in csv.DictReader(open(f"{WT}/experiments/v4/new_neg_manifest.csv")) if r["split"] == "heldout"]
got["heldout_healthy"] = sum(srv._classify(browser(open(f"{WT}/{r['file']}", "rb").read()))[0] != "unknown" for r in held)
R = dict(everyday_right=0, everyday_wrong=0, burns=0, feet=0, chronic=0, closeups=0); seen = set()
for r in csv.DictReader(open(f"{WT}/experiments/v2/rit/manifest.csv")):
    if r["file"] in seen: continue
    seen.add(r["file"]); lab = srv._classify(browser(open(f"{WT}/experiments/v2/rit/images/{r['file']}", "rb").read()))[0]
    if r["eval_role"] == "in_scope": R["everyday_right"] += lab == r["label"]; R["everyday_wrong"] += lab not in ("unknown", r["label"])
    elif r["eval_role"] == "burn_any": R["burns"] += lab == "possible_burn"
    elif r["source"] == "lower_limb_feet": R["feet"] += lab != "unknown"
    elif r["source"] == "lower_limb_wounds": R["chronic"] += lab != "unknown"
    else: R["closeups"] += lab != "unknown"
got["rit"] = R
import numpy as np, sys, io as _io, contextlib
with contextlib.redirect_stdout(_io.StringIO()):
    exec(open("crop_and_card.py").read().split("out = {}")[0])
want = {}
for split in ("val", "test"):
    s_ = stats(f"{split}_close", "burn", 0.75, "either", None)
    want[split] = dict(right=s_["right"], wrong=s_["wrong"], unanswered=s_["n"] - s_["named"], nonwound=s_["nonwound"])
want["heldout_healthy"] = healthy("burn", 0.75, "either", None)
rows_ = list(csv.DictReader(open(f"{WT}/experiments/v2/rit/manifest.csv"))); seen_, keep_ = set(), []
for i, r in enumerate(rows_):
    if r["file"] not in seen_: seen_.add(r["file"]); keep_.append(i)
keep_ = np.array(keep_); lab_ = labels(D["rit_browser_clf"], D["rit_browser_gate"], G4["rit_browser_gate"], "burn", 0.75, "either", None)[keep_]
Mn = ["abrasion", "bruise", "possible_burn", "cut"]; L_ = ["unknown" if l < 0 else Mn[l] for l in lab_]; W_ = {"everyday_right": 0, "everyday_wrong": 0, "burns": 0, "feet": 0, "chronic": 0, "closeups": 0}
for l, i in zip(L_, keep_):
    r = rows_[i]
    if r["eval_role"] == "in_scope": W_["everyday_right"] += l == r["label"]; W_["everyday_wrong"] += l not in ("unknown", r["label"])
    elif r["eval_role"] == "burn_any": W_["burns"] += l == "possible_burn"
    elif r["source"] == "lower_limb_feet": W_["feet"] += l != "unknown"
    elif r["source"] == "lower_limb_wounds": W_["chronic"] += l != "unknown"
    else: W_["closeups"] += l != "unknown"
want["rit"] = W_
print("server  :", json.dumps(got)); print("expected:", json.dumps(want)); print("MATCH" if got == want else "MISMATCH")
json.dump(dict(server=got, expected=want, match=got == want), open("verify_backend_v4c.json", "w"), indent=1)

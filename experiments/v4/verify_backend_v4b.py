"""Runs the MODIFIED server (two gates, 0.75) on every val/test/RIT photo and the 401 held-out healthy-skin
photos, browser-encoded, and compares with fewer_false_alarms.json option either@0.75."""
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
e = json.load(open("fewer_false_alarms.json"))["options"]["either@0.75"]
want = dict(val=e["val"], test=e["test"], heldout_healthy=e["heldout_healthy"], rit=e["rit"])
print("server  :", json.dumps(got)); print("expected:", json.dumps(want)); print("MATCH" if got == want else "MISMATCH")
json.dump(dict(server=got, expected=want, match=got == want), open("verify_backend_v4b.json", "w"), indent=1)

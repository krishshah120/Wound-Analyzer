"""X3 check: (1) agreement rate between b84_train burn labels and the 'fares' extra download on
matched (T1/T2) pictures; (2) contact sheet of conflict pairs to confirm they are the SAME picture
(duplicate check only - no judgement of which label is right)."""
import os, sys, random
import numpy as np
from PIL import Image, ImageDraw
sys.argv = ["x"]; exec(open("x3_conflicts.py").read().split("groups = collections.defaultdict")[0])
groups = collections.defaultdict(list)
for i in range(n): groups[find(i)].append(i)
lab = {}; wh = collections.defaultdict(set)
for p, r, l, w in items: lab.setdefault((ri[r], w), l); wh[ri[r]].add(w)
agree = disagree = 0; pairs = []
for g, mem in groups.items():
    tr = [i for i in mem if "b84_train" in wh[i]]; fx = [i for i in mem if "extra" in wh[i] and "fares" in real[i]]
    if not tr or not fx: continue
    tl = {lab[(i, "b84_train")] for i in tr}; fl = {lab[(i, "extra")] for i in fx}
    if not any(l.startswith("burn") for l in tl): continue
    if tl == fl: agree += 1
    else: disagree += 1; pairs.append((tr[0], fx[0], lab[(tr[0], "b84_train")], lab[(fx[0], "extra")]))
print(f"b84_train burn groups matched to a fares copy: {agree + disagree}; same label {agree}; different {disagree}")
random.Random(0).shuffle(pairs); pairs = pairs[:12]
sheet = Image.new("RGB", (4 * 2 * 160, 3 * 180), "white"); d = ImageDraw.Draw(sheet)
for k, (a, b, la, lb) in enumerate(pairs):
    x, y = (k % 4) * 320, (k // 4) * 180
    sheet.paste(Image.open(real[a]).convert("RGB").resize((156, 156)), (x, y)); sheet.paste(Image.open(real[b]).convert("RGB").resize((156, 156)), (x + 160, y))
    d.text((x + 2, y + 158), f"ours:{la[5:8]}", fill="black"); d.text((x + 162, y + 158), f"fares:{lb[5:8]}", fill="black")
sheet.save(f"{V3}/../v3_scratch_conflict_pairs.jpg", quality=85); print("sheet saved")

"""v6 (PLAN.md): training folders as symlinks. N6 = v2/b84_train minus U-flagged train photos + v4 healthy train
+ archive-8 train (cap 2,000) as out_of_scope; N6k = the same keeping the U-flagged photos."""
import os, csv
V6 = os.path.dirname(os.path.abspath(__file__)); WT = os.path.dirname(os.path.dirname(V6)); EXP = f"{WT}/experiments"
flag = {(r["label"], r["file"]) for r in csv.DictReader(open(f"{V6}/flags.csv")) if r["split"] == "train"}
neg = [f"{WT}/{r['file']}" for r in csv.DictReader(open(f"{EXP}/v4/new_neg_manifest.csv")) if r["split"] == "train"]
neg += [f"{WT}/{r['file']}" for r in csv.DictReader(open(f"{V6}/a8_manifest.csv")) if r["split"] == "train"]
for name, drop in (("train_N6", True), ("train_N6k", False)):
    d = f"{V6}/{name}"
    if os.path.exists(d): raise SystemExit(f"{d} exists")
    for c in sorted(os.listdir(f"{EXP}/v2/b84_train")):
        os.makedirs(f"{d}/{c}")
        for f in sorted(os.listdir(f"{EXP}/v2/b84_train/{c}")):
            if drop and (c, f) in flag: continue
            os.symlink(os.path.realpath(f"{EXP}/v2/b84_train/{c}/{f}"), f"{d}/{c}/{f}")
    for p in neg: os.symlink(p, f"{d}/out_of_scope/neg_{os.path.basename(os.path.dirname(p))}_{os.path.basename(p)}")
    print(name, {c: len(os.listdir(f"{d}/{c}")) for c in sorted(os.listdir(d))})

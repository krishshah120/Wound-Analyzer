"""EBIS evaluation, frozen in advance (EBIS_PROTOCOL.md, ebis_frozen.json). Nothing here is tuned on EBIS.
Usage:
  python ebis_eval.py --images DIR [--masks DIR] [--splits CSV]   # real run; needs ebis_permission.txt
  python ebis_eval.py --images DIR --smoke                        # technical check on non-EBIS photos
Steps: (1) refuse unless ebis_permission.txt records the permitted use (from the VEDAs Lab reply);
(2) verify every frozen model hash; (3) overlap check of every image against all existing collections
(sha256, decoded pixels, difference hash <= 6 with mirror, 12-view similarity >= 0.75) - overlapping
images are reported and excluded from the primary analysis; (4) run the four frozen pipelines;
(5) write ebis_results.json (or ebis_smoke.json). Images are never copied into any training folder."""
import os, io, sys, csv, json, hashlib, argparse, importlib.util, collections
import numpy as np
from PIL import Image
V3 = os.path.dirname(os.path.abspath(__file__)); WT = os.path.dirname(os.path.dirname(V3)); EXP = f"{WT}/experiments"
sys.path.insert(0, f"{WT}/src"); sys.path.insert(0, V3); os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
ap = argparse.ArgumentParser(); ap.add_argument("--images", required=True); ap.add_argument("--masks"); ap.add_argument("--splits")
ap.add_argument("--smoke", action="store_true"); a = ap.parse_args()
F = json.load(open(f"{V3}/ebis_frozen.json"))
if not a.smoke:
    perm = f"{V3}/ebis_permission.txt"
    if not os.path.exists(perm): raise SystemExit("ebis_permission.txt missing: record what the VEDAs Lab reply permits before evaluating")
    print("permitted use on record:", open(perm).read().strip()[:300])
sha = lambda b: hashlib.sha256(b).hexdigest()
for c in F["classifiers"].values():
    for f, h in zip(c["files"], c["sha256"]):
        if sha(open(f"{WT}/{f}", "rb").read()) != h: raise SystemExit(f"frozen model changed: {f}")
if sha(open(f"{WT}/{F['gate']['file']}", "rb").read()) != F["gate"]["sha256"]: raise SystemExit("frozen gate changed")
from ai_edge_litert.interpreter import Interpreter
spec = importlib.util.spec_from_file_location("prod", f"{EXP}/v2/production_snapshot/predict.py"); prod = importlib.util.module_from_spec(spec); spec.loader.exec_module(prod)
def interp(p): it = Interpreter(model_path=f"{WT}/{p}"); it.allocate_tensors(); return it
CLF = {k: [interp(f) for f in c["files"]] for k, c in F["classifiers"].items()}; GATE = interp(F["gate"]["file"])
NAMES = F["class_names"]; OOD = NAMES.index("out_of_scope"); W = [i for i in range(7) if i != OOD]; BURN = [NAMES.index(f"burn_{d}_degree") for d in ("1st", "2nd", "3rd")]
M = ["abrasion", "bruise", "burn", "cut", "out_of_scope"]; TO_M = np.array([2 if n.startswith("burn") else M.index(n) for n in NAMES])

imgs = sorted(os.path.join(r, f) for r, _, fs in os.walk(a.images) for f in fs if f.lower().endswith((".jpg", ".jpeg", ".png")))
print(f"{len(imgs)} images", flush=True)
# ---- overlap check against every existing collection ----
import train_model as t
def listdir(d): return [f"{d}/{f}" for f in sorted(os.listdir(d)) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
refs = [t.dataset_path_for(r) for r in t.load_real_images()]
refs += [p for c in sorted(os.listdir(t.EXTRA_DATASET_DIR)) if os.path.isdir(f"{t.EXTRA_DATASET_DIR}/{c}") for p in listdir(f"{t.EXTRA_DATASET_DIR}/{c}")]
refs += [p for g in sorted(os.listdir(t.OUT_OF_SCOPE_DATASET_DIR)) if os.path.isdir(f"{t.OUT_OF_SCOPE_DATASET_DIR}/{g}") for p in listdir(f"{t.OUT_OF_SCOPE_DATASET_DIR}/{g}")]
NC = np.load(f"{EXP}/new_candidates/features.npz"); ref_f = [NC["ref"]]; assert len(NC["ref"]) == len(refs)
cand_paths = [r["path"] for r in csv.DictReader(open(f"{EXP}/new_candidates/audit.csv"))]; ref_f.append(NC["cand"]); assert len(NC["cand"]) == len(cand_paths)
extra_paths = [f"{EXP}/v2/rit/images/{f}" for f in sorted(os.listdir(f"{EXP}/v2/rit/images"))] + listdir(f"{V3}/hn_pool")
cache = f"{V3}/ebis_ref_extra_features.npz"          # RIT + pool features, computed once (not committed)
tmpdir = f"{V3}/_tmp_ebis"; os.makedirs(tmpdir, exist_ok=True)
def small(paths, tag):
    out = []
    for i, p in enumerate(paths):
        q = f"{tmpdir}/{tag}_{i}.png"; Image.open(p).convert("RGB").resize((224, 224)).save(q); out.append(q)
    return out
if os.path.exists(cache) and list(np.load(cache)["paths"]) == extra_paths: ef = np.load(cache)["f"]
else:
    rs = small(extra_paths, "ref"); ef = t.view_features(rs); np.savez(cache, f=ef, paths=np.array(extra_paths))
    for x in rs: os.remove(x)
ref_f.append(ef); RF = np.concatenate(ref_f); all_refs = refs + cand_paths + extra_paths
ref_sha = {sha(open(p, "rb").read()) for p in all_refs}
def pixhash(p): x = np.asarray(Image.open(p).convert("RGB")); return sha(x.tobytes() + str(x.shape).encode())
ref_pix = {pixhash(p) for p in all_refs}
rh = np.array([t.difference_hash(p) for p in all_refs])
q = small(imgs, "ebis"); qf = t.view_features(q)
sim = t.cross_similarity(qf, RF).max(axis=1)
qh = np.array([t.difference_hash(p) for p in q]); qhm = np.array([t.difference_hash(p, mirrored=True) for p in q])
hd = np.array([min((qh[i] != rh).sum(1).min(), (qhm[i] != rh).sum(1).min()) for i in range(len(q))])
exact = np.array([sha(open(p, "rb").read()) in ref_sha or pixhash(p) in ref_pix for p in imgs])
overlap = exact | (hd <= t.DUPLICATE_HASH_DISTANCE) | (sim >= 0.75)
ss = t.cross_similarity(qf, qf); np.fill_diagonal(ss, 0); parent = list(range(len(imgs)))    # EBIS near-duplicate groups for the bootstrap
def find(x):
    while parent[x] != x: parent[x] = parent[parent[x]]; x = parent[x]
    return x
for i in range(len(imgs)):
    for j in np.nonzero(((qh[i + 1:] != qh[i]).sum(1) <= t.DUPLICATE_HASH_DISTANCE) | (ss[i, i + 1:] >= t.SAME_PHOTO_SIMILARITY))[0]: parent[find(i)] = find(i + 1 + int(j))
grp = np.array([find(i) for i in range(len(imgs))])
for p in q: os.remove(p)
print(f"overlap with existing collections: {int(overlap.sum())} of {len(imgs)} (exact {int(exact.sum())})", flush=True)

# ---- frozen pipelines ----
def browser(raw):
    img = Image.open(io.BytesIO(raw)).convert("RGB"); w, h = img.size; s = min(1.0, 1024 / max(w, h))
    if s < 1: img = img.resize((round(w * s), round(h * s)), Image.BILINEAR)
    b = io.BytesIO(); img.save(b, "JPEG", quality=82); return b.getvalue()
def decide6(p, g, thr):
    conf = p[W].max(); best = W[int(np.argmax(p[W]))]
    return (best if p.argmax() != OOD and g[OOD] < 0.5 and conf >= thr else -1), conf
def decide_m(p, g, thr):
    pm = np.zeros(5)
    for k in range(7): pm[TO_M[k]] += p[k]
    conf = pm[:4].max(); best = int(np.argmax(pm[:4]))
    return (best if pm.argmax() != 4 and g[OOD] < 0.5 and conf >= thr else -1), conf
out_lab = collections.defaultdict(list)
for p in imgs:
    img = Image.open(io.BytesIO(browser(open(p, "rb").read()))).convert("RGB")
    views = [prod._to_tensor(v) for v in prod._views(img)]
    gates = [prod._run(GATE, x) for x in views]
    probs = {k: [np.mean([prod._run(m, x) for m in ms], 0) for x in views] for k, ms in CLF.items()}
    for name, cfg in F["pipelines"].items():
        dec = decide6 if cfg["task"] == "6-class" else decide_m; thr = cfg["threshold"]
        l0, _ = dec(probs[cfg["classifier"]][0], gates[0], thr); lab = l0
        if l0 < 0 and len(views) > 1:
            l1, c1 = dec(probs[cfg["classifier"]][1], gates[1], thr)
            if l1 >= 0 and c1 >= max(0.80, thr): lab = l1
        if cfg["task"] == "6-class": out_lab[name].append("unknown" if lab < 0 else ("burn" if lab in BURN else NAMES[lab]))
        else: out_lab[name].append("unknown" if lab < 0 else M[lab])
L = {k: np.array(v) for k, v in out_lab.items()}

# ---- metrics ----
def wilson(k, n, z=1.96):
    if n == 0: return None
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(100 * (c - h), 1), round(100 * (c + h), 1)]
def boot(mask, a, b, n=2000):
    ug = np.unique(grp[mask]); gi = {g: np.nonzero(mask & (grp == g))[0] for g in ug}; rng = np.random.default_rng(0); d = []
    for _ in range(n):
        idx = np.concatenate([gi[g] for g in rng.choice(ug, len(ug))]); d.append(100 * ((L[a][idx] == "burn").mean() - (L[b][idx] == "burn").mean()))
    return [round(float(x), 1) for x in np.percentile(d, [2.5, 97.5])]
frac = None
if a.masks:
    mfiles = {os.path.splitext(f)[0]: os.path.join(r, f) for r, _, fs in os.walk(a.masks) for f in fs}
    frac = np.array([np.asarray(Image.open(mfiles[os.path.splitext(os.path.basename(p))[0]]).convert("L")).astype(bool).mean()
                     if os.path.splitext(os.path.basename(p))[0] in mfiles else np.nan for p in imgs])
split_of = {}
if a.splits: split_of = {r["file"]: r["split"] for r in csv.DictReader(open(a.splits))}
res = dict(n_images=len(imgs), n_overlapping=int(overlap.sum()), overlapping_files=[os.path.relpath(p, a.images) for p, o in zip(imgs, overlap) if o])
for scope, mask in (("primary_non_overlapping", ~overlap), ("secondary_all", np.ones(len(imgs), bool))):
    n = int(mask.sum()); r = {"n": n}
    for name in F["pipelines"]:
        lab = L[name][mask]; k = int((lab == "burn").sum())
        r[name] = dict(burn=k, burn_pct=round(100 * k / max(1, n), 1), burn_ci=wilson(k, n), other_injury=int(np.isin(lab, ["abrasion", "bruise", "cut"]).sum()),
                       unknown=int((lab == "unknown").sum()))
        if frac is not None:
            r[name]["by_mask_area"] = {nm: (lambda m: f"{int((L[name][m] == 'burn').sum())}/{int(m.sum())}")(mask & (frac >= lo) & (frac < hi)) for lo, hi, nm in ((0, .15, "<15%"), (.15, .40, "15-40%"), (.40, 1.01, ">=40%"))}
        if split_of: r[name]["by_split"] = {s: f"{int(((L[name] == 'burn') & mask & m).sum())}/{int((mask & m).sum())}" for s in sorted(set(split_of.values())) for m in [np.array([split_of.get(os.path.basename(p)) == s for p in imgs])]}
    if n:
        r["paired_95ci_pct_points"] = {f"{x}-{y}": boot(mask, x, y) for x, y in (("P1_deployed_summed", "P0_deployed"), ("P3_RE3_summed", "P2_RE3"), ("P2_RE3", "P0_deployed"), ("P3_RE3_summed", "P1_deployed_summed"))}
    res[scope] = r
outp = f"{V3}/{'ebis_smoke' if a.smoke else 'ebis_results'}.json"; json.dump(res, open(outp, "w"), indent=1)
os.rmdir(tmpdir)
print(json.dumps({k: v for k, v in res.items() if k != "overlapping_files"}, indent=1))

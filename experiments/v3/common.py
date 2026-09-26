"""Shared scoring for v3 (see PLAN.md). Everything works from saved per-photo probabilities so
every candidate is scored on the same photos, labels and denominators."""
import os, csv, json
import numpy as np
V3 = os.path.dirname(os.path.abspath(__file__)); WT = os.path.dirname(os.path.dirname(V3)); RUNS = f"{WT}/experiments/runs"
NAMES = json.load(open(f"{WT}/models/class_names.json"))
OOD = NAMES.index("out_of_scope"); B1, B2, B3 = (NAMES.index(f"burn_{d}_degree") for d in ("1st", "2nd", "3rd"))
W = [i for i in range(len(NAMES)) if i != OOD]
REASSURING_FOR_B2 = {B1, NAMES.index("abrasion"), NAMES.index("bruise")}
SHIPPED = np.load(f"{RUNS}/round_o_shipped.npz")


def files(split):
    """File order used by predict_directory / experiment9: class order, sorted filenames."""
    return [f for c in NAMES for f in sorted(os.listdir(f"{WT}/data/{split}/{c}")) if f.lower().endswith((".jpg", ".jpeg", ".png"))]


def groups(split):
    """Duplicate-group id per photo (split_manifest source_group; a photo without one is its own group)."""
    g = {r["filename"]: r["source_group"] for r in csv.DictReader(open(f"{WT}/data/split_manifest.csv")) if r["split"] == split}
    return np.array([g.get(f, "solo_" + f) for f in files(split)])


def decide(p, gate, t=0.60):
    """Deployed decide(), vectorised. Returns (answered mask, best wound class, confidence)."""
    conf = p[:, W].max(1); best = np.array(W)[p[:, W].argmax(1)]
    elig = (p.argmax(1) != OOD) & (gate[:, OOD] < 0.5)
    return elig & (conf >= t), best, conf


def score(ans, best, y):
    wound = y != OOD; a = ans & wound; c = a & (best == y)
    per = {NAMES[k]: dict(n=int((y == k).sum()), answered=int((a & (y == k)).sum()), correct=int((c & (y == k)).sum()),
                          abstained=int(((y == k) & ~a).sum())) for k in W}
    return dict(eligible=int(wound.sum()), answered=int(a.sum()), correct=int(c.sum()),
                coverage=round(100 * a.sum() / wound.sum(), 1), acc_answered=round(100 * c.sum() / max(1, a.sum()), 1),
                correct_of_eligible=round(100 * c.sum() / wound.sum(), 1),
                nonwound=int((~wound).sum()), nonwound_labelled=int((ans & ~wound).sum()),
                tierA=int((a & (y == B3) & (best != B3)).sum()), n_3rd=int((y == B3).sum()),
                tierB=int((a & (y == B2) & np.isin(best, list(REASSURING_FOR_B2))).sum()), per_class=per)


def matched_threshold(p, gate, y, target):
    """Lowest threshold at which at least `target` wound photos are answered (chosen on val only)."""
    elig = (p.argmax(1) != OOD) & (gate[:, OOD] < 0.5)
    conf = np.sort(p[:, W].max(1)[elig & (y != OOD)])[::-1]
    return None if len(conf) < target else float(conf[target - 1])


def threshold_at_error(p, gate, y, max_err):
    """Lowest threshold (most answers) whose error among answered wound photos is <= max_err, on val."""
    ans0, best, conf = decide(p, gate, 0.0)
    m = ans0 & (y != OOD); c = conf[m]; e = (best != y)[m]
    order = np.argsort(-c); err = np.cumsum(e[order]) / np.arange(1, len(c) + 1)
    ok = np.nonzero(err <= max_err + 1e-12)[0]
    return None if not len(ok) else float(c[order][ok.max()])


def aurc(p, gate, y):
    ans0, best, conf = decide(p, gate, 0.0); wound = y != OOD
    c = np.where(ans0 & wound, conf, -1.0)[wound]; r = (best != y)[wound].astype(float)
    order = np.argsort(-c); risks = np.cumsum(r[order]) / np.arange(1, len(r) + 1)
    return float(risks[: int((c >= 0).sum())].mean())


def paired_bootstrap(grp, stat_a, stat_b, n=2000, seed=0):
    """95% CI of stat_a - stat_b; each stat(idx) -> float or None. Resamples whole duplicate groups."""
    ug = np.unique(grp); gi = {g: np.nonzero(grp == g)[0] for g in ug}; rng = np.random.default_rng(seed); d = []
    for _ in range(n):
        idx = np.concatenate([gi[g] for g in rng.choice(ug, len(ug), replace=True)])
        a, b = stat_a(idx), stat_b(idx)
        if a is not None and b is not None: d.append(a - b)
    return [round(float(x), 1) for x in np.percentile(d, [2.5, 97.5])]


def acc_stat(ans, best, y):
    wound = y != OOD
    def f(idx):
        a = (ans & wound)[idx]
        return None if not a.sum() else 100 * (a & (best == y)[idx]).sum() / a.sum()
    return f


def cov_stat(ans, y):
    wound = y != OOD
    return lambda idx: 100 * (ans & wound)[idx].sum() / max(1, wound[idx].sum())


def fp_stat(ans, y):
    non = y == OOD
    return lambda idx: 100 * (ans & non)[idx].sum() / max(1, non[idx].sum())


def wilson(k, n, z=1.96):
    if n == 0: return None
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(100 * (c - h), 1), round(100 * (c + h), 1)]

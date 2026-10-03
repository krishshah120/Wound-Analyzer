"""Step 4: accuracy vs coverage of the DEPLOYED classifier+gate as the confidence threshold varies
(gate and out_of_scope rule unchanged), and calibration of its confidence on validation.
Uses experiments/runs/round_o_shipped.npz (deployed Keras == deployed TFLite, verified)."""
import os, json, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
V2 = os.path.dirname(os.path.abspath(__file__)); WT = os.path.dirname(os.path.dirname(V2))
names = json.load(open(f"{WT}/models/class_names.json")); OOD = names.index("out_of_scope")
S = np.load(f"{WT}/experiments/runs/round_o_shipped.npz"); W = [i for i in range(len(names)) if i != OOD]
out = {}
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
for k, split in enumerate(("val", "test")):
    y, p, g = S[f"{split}_y"], S[f"{split}_clf"], S[f"{split}_gate"]
    wound = y != OOD
    best = np.array(W)[p[:, W].argmax(1)]; conf = p[:, W].max(1)
    eligible_gate = (p.argmax(1) != OOD) & (g[:, OOD] < 0.5)
    curve = []
    for t in np.round(np.arange(0.20, 0.96, 0.05), 2):
        ans = eligible_gate & (conf >= t)
        cov = (ans & wound).sum() / wound.sum(); acc = (best[ans & wound] == y[ans & wound]).mean() if (ans & wound).any() else np.nan
        fpr = (ans & ~wound).sum() / (~wound).sum()
        curve.append(dict(threshold=float(t), coverage=round(100 * cov, 1), acc_answered=round(100 * acc, 1), fpr_nonwound=round(100 * fpr, 1)))
    out[f"{split}_curve"] = curve
    ax[k].plot([c["coverage"] for c in curve], [c["acc_answered"] for c in curve], "o-", ms=3)
    for c in curve:
        if c["threshold"] in (0.4, 0.6, 0.8): ax[k].annotate(f"t={c['threshold']}", (c["coverage"], c["acc_answered"]), fontsize=8)
    ax[k].set_title(f"deployed classifier+gate, {split}"); ax[k].set_xlabel("coverage of wound photos (%)"); ax[k].set_ylabel("accuracy among answered (%)"); ax[k].grid(alpha=.3)
    if split == "val":   # calibration: top-wound confidence vs whether that guess is right, wound photos
        bins = np.linspace(0, 1, 11); idx = np.digitize(conf[wound], bins) - 1; right = best[wound] == y[wound]
        ece, tab = 0.0, []
        for b in range(10):
            m = idx == b
            if m.any():
                ece += m.mean() * abs(conf[wound][m].mean() - right[m].mean())
                tab.append(dict(bin=f"{bins[b]:.1f}-{bins[b+1]:.1f}", n=int(m.sum()), mean_conf=round(float(conf[wound][m].mean()), 3), accuracy=round(float(right[m].mean()), 3)))
        out["val_calibration"] = dict(ece=round(float(ece), 3), bins=tab)
plt.tight_layout(); plt.savefig(f"{V2}/risk_coverage_deployed.png", dpi=130)
json.dump(out, open(f"{V2}/risk_coverage_deployed.json", "w"), indent=1)
for split in ("val", "test"):
    print(split, " ".join(f"t{c['threshold']}:{c['coverage']}%/{c['acc_answered']}%" for c in out[f"{split}_curve"] if c["threshold"] in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)))
print("val calibration ECE", out["val_calibration"]["ece"]); [print("  ", b) for b in out["val_calibration"]["bins"]]

# Wound analyzer v2 - handoff (resume from here)

Brief: user's 8-part plan of 2026-09-26 (baseline -> instability audit -> data/eval integrity ->
right metrics -> bounded experiments -> Roboflow detector -> deployable preview -> deliverables).

## Where things are
- Worktree: /Users/vihaa/Downloads/Wound-Analyzer/.claude/worktrees/silly-wilson-17d01a
- Working branch: claude/wound-v2 (from 16c3704). PR #1 branch claude/silly-wilson-17d01a untouched.
- Python: .venv/bin/python (3.12, TF 2.21, Keras 3.15.1, ai-edge-litert 2.2.0)
- Production (live on the MRC site): Vercel project wound-analyzer-vercel, code at
  ~/Downloads/wound-analyzer-vercel (NOT a git repo; owned by the "MRC App" session - read only).
  Read-only snapshot + hashes: experiments/v2/production_snapshot/ (SHA256SUMS).
- Earlier ledger (rounds G..P): experiments/round_I_rule.md

## Production pipeline (from production_snapshot/predict.py)
1. decode upload, RGB, NEAREST resize to 224x224, x/127.5-1
2. classifier wound_model.tflite (sha eef2f3d3...) + gate out_of_scope_gate.tflite (sha 49201744...)
   classes: abrasion, bruise, burn_1st_degree, burn_2nd_degree, burn_3rd_degree, cut, out_of_scope
3. decide(): unknown if classifier argmax == out_of_scope, OR gate p(out_of_scope) >= 0.5,
   OR confidence (max wound prob) < 0.60
4. if unknown: centred 70% crop, same models; accept its answer only if not unknown and conf >= 0.80
Browser (MRC site) first shrinks to longest edge 1024 px, JPEG quality 0.82.

## Status log
- 2026-09-26: baseline + instability audit started.

## Step 1 result (baseline, experiments/v2/baseline.json)
Reported 46.2% / 85.3% / 16 of 94 / 1 = VALIDATION, classifier+gate, stored files, NO crop retry.
Deployed TFLite == repo Keras on all 1,126 val+test photos (max prob diff 4.3e-6, 0 decision changes).
Test (stored files): classifier+gate cov 48.0% acc 75.5% FP 68/450 tierA 3; production (+crop) cov
48.9% acc 75.3% FP 70/450 tierA 4. Latency on this Mac ~13.5 ms median; load 0.06 s; RSS ~146 MB.
TEST SET STATUS: printed in every experiment round since G and used for README figures -> it is
NOT an untouched final test. Selection was always on val, but test has been inspected repeatedly.

## Step 2 (instability) - declared before running
Rebuilt the deployed model's exact train/val from b84b7b4 split manifest (experiments/v2/b84_train,
b84_val; all 2,071 train + 284 val files found by name; byte identity unprovable - no hashes kept).
P0 was NOT a recreation: 806 of 861 OOD train shared, early-stop val had 94 vs 33 OOD photos.
Runs: exact recipe (experiment9 == train_model.py b84b7b4 code path), --train-dir b84_train
--val-dir b84_val, seeds 42 (x2, nondeterminism check), 1, 2, 3. Scored on current data/val like
the baseline. Interpretation fixed in advance: if seed-42 reruns and seeds 1-3 land near 46.2%
coverage, the deployed run is typical; if only the deployed run is high, it was above average.

## Step 3 findings so far
- Original burns (kg2, 1,052 photos) = the Shubham Baid YOLO burn set (classes 0/1/2), same set whose
  "2nd degree" folder contains rashes. So train/val/test BURN labels carry that noise.
- flag_burn_labels.py -> flagged_for_review.csv: 319 wound photos resemble an out-of-scope photo
  (>= 0.80); 38 at >= 0.87 are likely label conflicts (train 32, val 3, test 3), e.g. a photo
  watermarked "SKIN RASHES" labelled 1st degree, a DermNet "Dermatitis" photo present as both
  1st degree burn and tick bite (sim 1.000). NOT relabelled - for qualified review.
- Roboflow detector (wound-ebsdw-4atst-1-yolo26x-t1, YOLO26 X-Large, AGPL-3.0): trained on 204
  copies of OUR val/test photos -> detector+classifier comparisons on our val/test are contaminated.
  Cleanest independent: 42 of its test-split images not in our data. Needs a Roboflow API key
  (not available) to run at all. Email metrics (58.4% mAP / 69.3% P / 51.5% R) NOT in the PDF -
  unverified; PDF shows val mAP@50 53.6% (300 images), test latency 8.1 ms at conf 0.26 (150).

## Step 4: deployed risk-coverage + calibration (risk_coverage_deployed.png/.json)
Val: t0.5 -> 61.4% coverage / 79.2% acc; t0.6 -> 46.2% / 85.3%; t0.7 -> 36.3% / 86.8%.
Test: t0.5 -> 61.6% / 70.1%; t0.6 -> 48.0% / 75.5%. Val ECE 0.078, UNDER-confident 0.4-0.8.
Methodological flaw found: rounds K-P compared candidates at the shared 0.60 cutoff, which penalises
a differently-calibrated model. Fair comparison = accuracy at MATCHED coverage.

## Step 5 rule MC (declared 2026-09-26 BEFORE computing any matched-coverage number;
## disclosure: fixed-0.60 results of rounds I-P had already been seen)
For each candidate with saved val probabilities, and the deployed model (classifier+gate, gate and
out_of_scope rule unchanged), pick the confidence threshold ON VAL at which its wound-photo coverage
equals the deployed 46.2% (116 of 251; nearest achievable, not below). Compare:
  primary   accuracy among answered at that coverage (deployed: 85.3%, 99 of 116)
  guards    non-wound photos labelled at that threshold <= 16/94; 3rd degree shown another wound <= 1
  also      area under the risk-coverage curve (lower risk = better), reported not gated
A RECIPE qualifies only if its 3-seed mean accuracy at matched coverage beats the deployed model by
>= 3 points AND both guards hold on the mean. Uncertainty: paired, grouped (duplicate source_group)
bootstrap of the difference on val, 2,000 resamples, reported. If a recipe qualifies, the artifact
considered for deployment is its MEDIAN-scoring seed (not the best), its val-chosen threshold is
frozen, and it is evaluated ONCE on a reserved independent test (to be built in step 3) - not on
data/test, which has been inspected repeatedly. A threshold other than 0.60 needs the user's OK
before deployment, because the site's "unknown" contract was stated in terms of 0.60.

## Hardware (2026-09-26): Apple M4, 10 CPU cores, 10-core GPU, 16 GB.
tensorflow-metal 1.2.0 loads with TF 2.18/2.19 (isolated venv, since removed) but gives no speed-up:
EfficientNetV2-S train step (batch 16, full backprop) CPU 3.96 s vs GPU 3.88 s; ConvNeXt-Tiny 14.2 s
CPU and crashes on Metal. (Timed while other training ran - relative, not absolute.) -> CPU only;
EfficientNetV2-S chosen over ConvNeXt-Tiny for the one stronger-backbone fine-tune (3.6x cheaper,
and fine-tuned EfficientNetV2-B0 had the best AURC).

## Matched-coverage result (rule MC): no recipe qualifies. Closest I_H4 (EffV2-B0 fine-tuned):
84.6% vs 85.3% at 46.2% coverage, AURC 0.141 vs 0.171, but 3rd-degree guard 2.33 > 1. Paired grouped
95% CIs of per-run differences span about +-8 points: val (116 answered) cannot resolve 1-3 points.

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

## Step 2 RESULT (exact reproduction, 2026-09-26)
Val, classifier+gate at 0.60: deployed cov 46.2 / acc 85.3 / top-1 65.7 / median conf 0.594.
Reruns on the EXACT b84b7b4 data: cov 46.2, 47.0, 41.4, 50.2, 49.4 (mean 46.8) - coverage reproduces;
acc-when-answered 83.6, 74.6, 82.7, 80.2, 81.5; top-1 70.9, 62.9, 61.8, 64.9, 65.7.
The two SEED-42 reruns (same data, same code) stopped at epochs [8,22] vs [12,10] and differ by
9 points: CPU TF is not bit-deterministic and early stopping on a 284-photo val amplifies it.
P0's 36% coverage came from its different data/early-stop set, not chance ("lucky" retracted).
Matched coverage (rule MC): control seeds 1-3 mean 82.2% (-3.2 vs deployed), AURC 0.183; seed-42
reruns 83.6% / 75.0%. The deployed model is the best of 6 runs of its recipe on accuracy-when-
answered (about +3 over typical) but typical on coverage and top-1 accuracy.

## Step 5, experiment E1 (declared BEFORE running): fine-tune ONE stronger backbone
Hypothesis: a fine-tuned EfficientNetV2-S ranks answers better than MobileNetV2 1.4 on this data
(fine-tuned EfficientNetV2-B0 had the best AURC, 0.141, in round I).
Control: R_exact seeds 1-3 (deployed recipe, same data), already run.
Candidate: identical except the backbone - EfficientNetV2-S (ImageNet, include_preprocessing, input
rescaled from the app's [-1,1] to [0,255]); staged: head 15 epochs Adam 1e-3 early-stop 3, then top 100
layers unfrozen (BatchNorm frozen) Adam 1e-5 up to 40 epochs early-stop 4; same augmentation, label
smoothing 0.1, class weights (3rd x2) as the control - nothing else changes.
Data version: b84_train / b84_val (early stopping), scored on current data/val with the shipped gate.
Checkpoint: restore best val_loss (same as control). Seeds 1, 2, 3 - no others.
Budget: pilot = seed 1. Continue to seeds 2-3 only if the pilot's val AURC <= control mean (0.183)
and one run takes <= 3 h. Otherwise stop and report.
Acceptance: rule MC (3-seed mean accuracy at 46.2% coverage >= deployed 85.3% + 3 = 88.3%, non-wound
<= 16/94, 3rd degree shown another wound <= 1). Also reported: paired difference vs the matched control
(causal effect of the backbone), AURC, grouped bootstrap CIs, runtime, file size.

## Step 3: Reserved Independent Test (RIT) - declared BEFORE building
Purpose: data/test has been inspected repeatedly; RIT holds photos no DEPLOYED model was trained or
selected on. Built once, frozen with sha256, NEVER used to train, tune thresholds or pick models.
Contents (labels mapped only where the meaning is the same):
  in-scope:  Roboflow "new" (audit) Abrasions->abrasion, Bruise->bruise, Cut->cut
  burn, degree unknown: Roboflow "new" Burn - scored only as burn vs not-burn (no degree claim)
  non-wound, healthy skin: Roboflow "new" no-abnormality; Lower Limb normal feet not in localizer data
  non-wound, out of scope (NOT healthy): Lower Limb chronic wounds
  excluded: Roboflow Blister (maps to no single class); anything with visual similarity >= 0.75 or a
  near-identical hash to ANY photo in the current or b84b7b4 train/val/test, the extras, or the
  out-of-scope collection.
Real framing strata (real photos, not synthetic): Roboflow box area / image area < 15%, 15-40%, > 40%.
Caveats known in advance: Roboflow/Lower-Limb are web/clinic photos (not user phone photos except the
feet); P1 (rejected) trained on 105 of the Roboflow photos and localizer N on them - neither is
deployed; any FUTURE candidate must not train on RIT photos.

## E1 RESULT (2026-09-26): STOPPED after pilot, per the declared rule.
Pilot seed 1 (36.9 min, epochs [14, 20]): val AURC 0.215 > control mean 0.183 -> do not run seeds 2-3.
Accuracy at 46.2% coverage 77.6% (control 82.2%, deployed 85.3%); non-wounds 19/94; 3rd shown other 3.
A fine-tuned stronger backbone did not help on this data.

## RIT built + deployed pipeline scored (rit/manifest.csv, rit_eval_deployed.json)
RIT frozen: 3,063 photos (1,896 dup groups): 1,613 healthy feet, 1,313 chronic wounds, 22 healthy skin,
67 burns (degree unknown), 35 bruises, 10 cuts, 3 abrasions. 2,111 candidates excluded for resembling a
training/selection photo. Production pipeline, browser-encoded:
  in-scope 24/48 answered (50%), 24/24 correct (95% CI 86-100); box < 15% of frame 11/32 answered vs
  15-40% 12/15 -> the framing weakness is visible on REAL photos. Burns: 23 any-burn, 1 bruise, 43 unknown.
  Healthy feet 183/1613 labelled (11.3%, 9.9-13.0): bruise 112, 1st 45, 3rd degree 12 (false emergency).
  Chronic wounds 133/1313 (10.1%), 65 as 3rd degree. Healthy close-ups 6/22.

## Learning curve LC (declared BEFORE running) - to quantify data needs, not to select a model
Exact deployed recipe (as R_exact) on nested subsets of b84_train: 25% and 50%, sampled by whole
duplicate groups (b84b7b4 source_group), stratified by class, fixed subsample seed 0, 25% inside 50%.
Early stopping on full b84_val. Seeds 1-3 each (100% = R_exact seeds 1-3). Metrics on current data/val
with the shipped gate: top-1 wound accuracy, AURC, accuracy at 46.2% coverage. Output: error vs
training-set size, and a log-linear extrapolation of the training photos needed for +5 points top-1,
reported with its uncertainty and NOT as a promise. No model from LC is a deployment candidate.

## Data manifest (experiments/v2/data_manifest.csv)
6,295 photos: split, label, sha256, source dataset, licence status, duplicate group, original name,
review note (flagged label conflicts). 0 identical files across splits; 176 byte-identical
duplicates within a split. Patient/case identity: NOT available in any source (visual groups only).

## Evaluation sample sizes (95% Wilson/normal approximation)
accuracy-when-answered (~85%) to +/-3 pts: ~545 answered = ~1,160 eligible wound photos (~190/class)
per-class recall/coverage to +/-10 pts: ~97 photos per class; +/-15 pts: ~43
3rd-degree "zero reassuring errors" to <= 1% upper bound: ~300 3rd-degree burns (rule of three)
non-wound false-positive (~11%) to +/-1.5 pts: ~1,670 non-wound photos (RIT has 1,635 - adequate)
Current val: 251 wound photos (25-67/class, 31 3rd degree) -> ~4-5x too small to resolve 3 points.

## LC RESULT + CLOSE-OUT (2026-09-26)
Learning curve (deployed recipe, 3 seeds each): 521 / 1,039 / 2,071 train photos -> top-1 53.8 / 61.6 /
64.1%, acc@46.2% 64.4 / 77.0 / 82.2%, AURC 0.325 / 0.220 / 0.183. Diminishing returns (+7.8 then +2.5
top-1 per doubling). Bounded experiment set complete; no candidate qualified; production unchanged.
Final write-up: experiments/v2/REPORT.md. Next steps need new data or access (see REPORT section 8 and
section 6), not more runs on the current data.

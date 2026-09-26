# Wound analyzer v3 - plan, fixed before any v3 number was computed (2026-09-26)

Constraint for this round: no human reviews or relabels images; no clinician review, new photo
collection, restricted datasets or Roboflow API key. Objective: more correct, useful answers on
realistic photos and fewer wrong injury labels, without hiding hard photos or lowering thresholds.
Branch claude/wound-v3 (from claude/wound-v2 f0fd0e8). Production files are untouched
(experiments/v2/production_snapshot, models/*.keras|.tflite, sha256 in v2/repo_models_SHA256SUMS).

## Evaluation sets and what has already influenced development
| set | photos | status from now on |
|---|---|---|
| data/val (251 wound, 94 non-wound) | selection in every round since G; its wound photos are also the early-stopping set of the deployed model and the R_exact runs (b84_val) | development benchmark |
| data/test (331 wound, 450 non-wound) | printed in every round since G | development benchmark |
| RIT (3,063, experiments/v2/rit) | the deployed pipeline was scored on it once and those results motivated this round (feet false alarms, distant photos) | used here for REPORTING only, never for thresholds or selection; counts as a development benchmark after this round |
No untouched final test remains. Any candidate from this round can only be an EXPERIMENTAL preview.

## Compute budget (fixed)
Apple M4 CPU only (Metal gave no speed-up, v2). At most 10 training runs and about 7 hours of
training wall time in total, one run at a time. Inference-only analyses are not counted. No cloud.

## Metrics for every candidate (same images, same labels, same denominators)
eligible wound photos, answered, correct; coverage; accuracy among answers; per-class answered/correct
/abstained; non-wound photos given a label (by source); 3rd-degree burns shown a different wound
label (tier A); 2nd shown 1st/abrasion/bruise (tier B); RIT in-scope and real-framing strata
reported apart from synthetic framing; paired bootstrap (2,000 resamples over duplicate groups; no
patient identity exists) for differences; model bytes and measured latency on this Mac (not Vercel).
Matched-coverage comparisons choose the threshold on data/val only and carry it unchanged to
test/RIT. The 0.60 operating point is always reported too. If CIs cannot separate candidates the
verdict is "inconclusive", not "improved".

## X1 - ensemble of existing runs (no training)
Members, fixed by rule (no test result used): R_exact seeds 1, 2, 3 - the three pre-declared control
seeds of the deployed recipe on the deployed data. Check: identical class order and file order to
the deployed model; same preprocessing (experiment9 predict_directory = the app loader).
Ensemble = mean of the members' 7-class probabilities; shipped gate and decide() unchanged.
1. Instability: spread of accuracy at matched coverage and of coverage at 0.60 across the 5 single
   exact-recipe runs vs across all 10 three-member ensembles that can be formed from them (the 10
   overlap, so their spread is descriptive only).
2. Deployed single vs ENS3 on val and test: at 0.60; accuracy at matched coverage (46.2% on val);
   coverage at matched error (deployed val error 14.7%); non-wound labels; tier A.
3. Disagreement abstention: disagree = members' top classes (7-way) not unanimous. Used only if on
   val, among ENS3's answered wound photos at matched coverage, the error rate when members disagree
   is at least twice the rate when unanimous (>= 5 disagreeing answers). Then compared at matched
   coverage against confidence alone.
Weights of the R_exact runs were not saved, so a DEPLOYABLE ensemble needs 3 retrained members
(~15 min each). They are trained only if ENS3 meets rule MC on val (>= +3 points at matched
coverage, non-wound <= 16/94, tier A <= 1); otherwise X1 is reported as analysis only. Shipping two
or more classifiers also needs Vihaan's OK (he declined an ensemble once before the gate).

## X2 - crop policy (inference only, deployed models)
C0 full image only. C1 current production (if the full frame is "unknown", a centred 70% crop may
answer at >= 0.80). C2 conservative = C1, but the crop's answer is used only if it names the SAME
wound class as the full frame's best guess; otherwise "unknown" (views conflict). Rule fixed here,
before measuring. Measured on val, test (both development) and RIT, whole pipeline.
Adopt C2 over C1 if, pooled over val+test, C2 adds at most half as many wrong answers as C1 adds
over C0, keeps at least half of C1's added correct answers, and adds no tier-A error; RIT reported.

## X3 - conservative training-label exclusion (deployed recipe, deployed data b84_train)
Tiers, computed without any model judging medical content:
 T1 exact duplicate: identical sha256, or identical pixels after decoding.
 T2 verified near-duplicate: difference hash <= 4 bits (mirror allowed) AND 64x64 grayscale pixel
    correlation >= 0.95 (mirror allowed).
 T3 caution: 12-view feature similarity >= 0.87 but not T1/T2.  T4 everything else in the 319 flags.
Objective conflict = a T1/T2 group containing a b84_train photo whose members carry different labels
anywhere in our collections (6 wound classes, out_of_scope, the out-of-scope collection's groups
count as out_of_scope). Quarantine = every b84_train member of such a group (both sides; no label is
chosen). Originals are never moved or edited; B is a symlink copy of b84_train minus the quarantine.
A = existing R_exact seeds 1-3. B = identical command, seeds 1-3. Evaluation images fixed.
Skip rule: if the quarantine has fewer than 10 photos, training B is not run (a < 0.5% change cannot
be measured against the 9-point run-to-run spread found in v2); X3b below runs instead.
Pilot B seed 1; continue to seeds 2-3 only if its val AURC <= 0.183 (A mean). Verdict: 3-seed means
and the 3-member ensembles of A vs B, rule MC thresholds; else "inconclusive".
X3b (only if the skip rule fires): confident-learning ranking with group-aware 5-fold out-of-fold
predictions (frozen MobileNetV2-1.4 features + logistic head; folds split by duplicate group, so no
photo is scored by a model trained on it or its duplicates). Suspect = the out-of-fold probability of
its given label is below that class's mean self-confidence AND another class's probability is above
that other class's mean self-confidence; at most 5% of any class. B' excludes suspects; same pilot /
continuation / verdict rules as B. Suspects are "suspected label problems", not corrections.

## X4 - "possible burn" task (separate taxonomy)
Mapping for EVERY comparator and benchmark: burn_1st/2nd/3rd -> burn; abrasion, bruise, cut,
out_of_scope unchanged. X4a (no retraining): burn probability = sum of the three burn probabilities;
decide() otherwise unchanged (0.60, gate 0.5, out_of_scope rule). RIT's 67 degree-unknown burns are
scorable in this task. Reported against the 6-class pipeline mapped the same way AFTER its decision.
It is a different task: its accuracy is never presented as 6-class accuracy.
X4b (train on the merged taxonomy) only if >= half of X4a's val errors among answers involve the burn
category (burn shown as another injury or another injury shown as burn) and budget remains; pilot
seed 1; continue only if its val AURC (merged task) beats X4a's.
The generic output infers no depth, seriousness, or that care is unnecessary; "unknown" stays.

## X5 - hard negatives for the out-of-scope gate (only if healthy-skin false alarms dominate)
Pool: Lower Limb "Nomal" healthy-foot photos NOT in the RIT (the 600 used as localizer negatives +
544 excluded from the RIT for resembling other photos), minus any within similarity 0.75 or
difference hash 6 of ANY RIT photo. RIT feet are never trained on.
G_hn = gate recipe (MobileNetV2 1.4, label smoothing 0.1, data/train + pool as out_of_scope, early
stop data/val) seed 1; control G_ctrl = same recipe without the pool, seed 1. Both with weights saved.
Pass (with the deployed classifier, threshold 0.5 unchanged): RIT feet false labels at most half of
both the shipped gate's and G_ctrl's; val correct wound answers lost vs shipped gate <= 5 (the
original gate acceptance rule); RIT in-scope answers lost <= 2 of 48; val non-wound <= 16/94; tier A
not worse. If the pilot passes, G_hn seed 2 is run and must pass too; the preview uses seed 1.
Limitation stated up front: pool and RIT feet come from the same dataset/camera; the RIT healthy
close-ups (22, another source) and val/test non-wounds are the out-of-source evidence.

## Roboflow detector
Not comparable here: it needs an API key (excluded this round), is a detector (mAP) not a classifier,
and trained on 204 copies of our val/test. No claim of "better than Roboflow" can be made.

## Preview
A candidate that passes goes into an EXPERIMENTAL preview only: model-set version file with sha256,
production set kept as the default and one-switch rollback, TFLite export checked against Keras.
Nothing is deployed to the MRC site from reused development benchmarks alone.

## Clarifications fixed before the numbers they affect were computed
- X2's primary input is 'browser' (what the MRC site sends); stored-file ('raw') results are reported
  alongside. Written into x2_crop.py's docstring before it was first run; copied here only afterwards.
- 2026-09-26, before any X5 gate existed: X5 is scored through the production pipeline (C1 crop
  retry) with only the gate swapped, and every X5 pass condition must hold on BOTH browser and raw
  inputs for val.

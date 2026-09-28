# Wound analyzer v2 - results (2026-09-26)

**Bottom line: no candidate qualified. Production is unchanged.** Nothing trained in this round beat the
deployed pipeline under rules written before each result was seen. The limiting factors are now
measured: evaluation sets 4-5x too small to resolve the few-point differences being tested, noisy
burn labels, and too few real distant photos. Details and the full log: `HANDOFF.md`,
`../round_I_rule.md`.

## 1. Baseline - what is actually deployed (`baseline.py`, `baseline.json`)
Production = MRC backend `predict.py`: MobileNetV2-1.4 classifier + MobileNetV2-1.4 out-of-scope gate
(TFLite, float32), NEAREST resize to 224, `unknown` if out_of_scope wins, gate >= 0.5 or confidence
< 0.60, then one retry on a centred 70% crop accepted only at >= 0.80. Run through the production code
itself. Its TFLite files equal the repo's Keras models on all 1,126 val+test photos (max probability
difference 4.3e-6, 0 decision changes; hashes differ only in packaging).

| Stored files | Val coverage | Val acc answered | Val non-wound FP | Test coverage | Test acc answered | Test non-wound FP | Test 3rd shown other wound |
|---|---|---|---|---|---|---|---|
| classifier | 49.0% | 84.6% | 28/94 | 49.5% | 75.0% | 104/450 | 3/40 |
| + gate | **46.2%** | **85.3%** | **16/94** | 48.0% | 75.5% | 68/450 | 3/40 |
| + crop retry (production) | 50.6% | 82.7% | 18/94 | 48.9% | 75.3% | 70/450 | **4/40** |

The often-quoted 46.2% / 85.3% / 16 of 94 / 1 are validation numbers without the crop retry. Runtime
on an Apple M4 CPU (not Vercel): 13.5 ms median per photo, 0.06 s load, ~146 MB resident, 2 x 17.3 MB.

## 2. Why reruns varied (step 2)
- "P0" was not a recreation: 806 of 861 out-of-scope training photos matched, and early stopping used a
  different validation set.
- Rebuilt exactly from the b84b7b4 manifest, the recipe reproduces coverage (5 reruns: 41.4-50.2%,
  mean 46.8%). The deployed model's 85.3% accuracy-when-answered is the best of 6 runs (reruns
  74.6-83.6%); top-1 accuracy is typical.
- The same seed on the same data gave models 9 points apart. TensorFlow on CPU is not bit-deterministic,
  and early stopping on a 284-photo validation set amplifies it (stops at epochs 8+22 vs 12+10).
  Unstable runs stop early and stay under-confident; the variation is mostly confidence crossing 0.60.

## 3. Data and evaluation integrity (step 3)
- `data/test` has been printed in every round since G, so it is **not** an untouched final test.
- The original burn photos are the Shubham Baid set, whose "2nd degree" folder visibly contains rashes.
  319 wound photos resemble an out-of-scope photo; **38 are likely label conflicts** (>= 0.87
  similarity: train 32, val 3, test 3), in `flagged_for_review.csv`. Nothing was relabelled; they
  need qualified review.
- `data_manifest.csv`: 6,295 photos with sha256, source, licence status, duplicate group. 0 identical
  files across splits. **No patient/case identity exists in any source**, so grouping is by visual
  duplicates only.
- **Reserved Independent Test** (`rit/manifest.csv`, frozen, sha256): 3,063 photos no deployed model
  trained or selected on (2,111 look-alikes removed). 1,613 healthy feet (phone photos), 1,313
  chronic wounds, 22 healthy close-ups, 67 burns (degree unknown), 35 bruises, 10 cuts, 3 abrasions.

## 4. Deployed pipeline on the reserved test (`rit_eval_deployed.json`)
| | Result |
|---|---|
| abrasion/bruise/cut (48) | 24 answered (50%), 24 correct (95% CI 86-100%) |
| injury box < 15% of frame (real photos) | 11/32 answered; 15-40%: 12/15 |
| burns, degree unknown (67) | 23 burn label, 1 bruise, 43 unknown |
| healthy feet (1,613) | **183 labelled (11.3%, CI 9.9-13.0)**; bruise 112, 1st 45, **3rd degree 12** |
| chronic wounds (1,313) | 133 labelled (10.1%); 65 as 3rd degree |

## 5. Metrics and experiments (steps 4-5)
- Calibration (val): ECE 0.078, under-confident at 0.4-0.8. Risk-coverage curves:
  `risk_coverage_deployed.png`.
- Comparing at the shared 0.60 cutoff was unfair to differently-calibrated models, so every run was
  re-scored at matched coverage (46.2%), with paired bootstrap CIs over duplicate groups
  (`matched_coverage_all.txt`):

| Recipe (val, 3 seeds unless noted) | Acc at 46.2% cov. | vs deployed | Non-wound /94 | 3rd shown other | AURC | Min/run |
|---|---|---|---|---|---|---|
| **deployed** | **85.3%** | - | 16 | 1 | 0.171 | - |
| exact rerun, deployed data (R) | 82.2% | -3.2 | 12.3 | 2.7 | 0.183 | 16 |
| EfficientNetV2-B0 fine-tuned (I_H4) | 84.6% | -0.8 | 10.0 | 2.3 | **0.141** | 53 |
| out-of-scope cap 2000 (J) | 83.9% | -1.4 | 9.0 | 1.7 | 0.160 | 25 |
| zoom-out 0.8 (K) | 83.1% | -2.2 | 6.3 | 1.0 | 0.152 | 41 |
| frame-shrink 0.3 (L3) | 82.5% | -2.8 | 6.0 | 2.3 | 0.156 | 33 |
| **EfficientNetV2-S fine-tuned (E1, pilot)** | 77.6% | -7.8 | 19 | 3 | 0.215 | 37 |
| + Roboflow photos (P1) | 76.5% | -8.9 | 14.0 | 1.7 | 0.228 | 8 |

Per-run 95% CIs of the difference span about +-8 points. E1 stopped after its pilot (AURC 0.215 > 0.183
continue bar). Frozen ConvNeXt-Tiny / EfficientNetV2-S heads, 6-class heads, crop-to-injury localizers
(rounds M/N) and the centre crop were measured earlier and did not qualify either.

## 6. Roboflow detector (step 6) - not evaluated, and why
`wound-ebsdw-4atst-1-yolo26x-t1`: YOLO26 X-Large, dataset v1 (1,503 images at 640x640). Report PDF:
val mAP@50 53.6% (300 images); test latency 8.1 ms at confidence 0.26 (150 images). The emailed
58.4% mAP / 69.3% P / 51.5% R are not in the PDF and are unverified. **Licence AGPL-3.0.** It was
trained on 204 copies of our val/test photos, so any comparison on those is contaminated; its clean
held-out images are the 42 test-split images not in our data. Needed to go further: a Roboflow API key
supplied as an environment variable (never in code), a decision on AGPL-3.0, and an export or hosted
endpoint the MRC backend may call.

## 7. What improved, regressed, remains unproven
- Improved: the evaluation itself - exact baseline, fair matched-coverage comparison, reproducible
  instability explanation, a reserved independent test, a flagged-label list, a hashed manifest.
- Regressed: nothing deployed changed. The measurement found that the production crop retry adds one
  3rd-degree reassuring error on test (3 -> 4 of 40).
- Unproven: any candidate being better than production (differences are within noise); real-world
  performance on user phone photos of the six injuries (only 48 exist in the reserved test).

## 8. The specific limitation, quantified
- **Evaluation power.** Resolving a 3-point change needs ~1,160 held-out wound photos (~190 per
  class); per-class +-10 points needs ~97 per class; bounding reassuring 3rd-degree errors at <= 1%
  needs ~300 3rd-degree burns. Current val: 251 wound photos (25-67 per class, 31 3rd degree).
- **Training data.** The learning curve on the deployed recipe (521 -> 1,039 -> 2,071 photos) gave
  top-1 53.8 -> 61.6 -> 64.1% and accuracy at matched coverage 64.4 -> 77.0 -> 82.2%. Returns
  diminish within the CURRENT data; three points with 3 seeds each cannot support a firm forecast
  for new data, least of all data of a different kind (corrected in v3: an earlier wording here
  said new photos would "plausibly add only a point or two" - that was an over-extrapolation).
- **What would change the result:** new held-out and training photos of the six injuries that are
  not web re-uploads. By capture condition: >= ~100 per class where the injury fills < 15% of the
  frame (the reserved test has 32 in total), phone photos of healthy skin and limbs (the source of
  the 11% healthy-feet false alarms), and burns with clinically verified degree (the current labels
  derive from a set with rash contamination). Collection needs consent and storage settled first.

## Reproduce
`.venv/bin/python experiments/v2/baseline.py`; `risk_coverage.py`; `matched_coverage.py`;
`flag_burn_labels.py`; `build_rit.py` (refuses to rebuild once frozen); `eval_rit.py`; training runs
`queue_repro.sh`, `queue_e1.sh 1`, `queue_lc.sh` (all through `experiments/experiment9.py`). They
need the gitignored data folders and the downloads named in `src/collate_extra_data.py`.

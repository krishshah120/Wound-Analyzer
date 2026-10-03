# v5 result - why normal faces become "possible burn", and what fixes it (2026-10-03)

Model-side measurements only. Nothing here was deployed; production is owned by the MRC App session.
The user's own face photo was not available to this session: it was not reproduced, used, or uploaded.
Rules were fixed in PLAN.md (commits c994001, 39e8167) before the results they govern.

## 1. Reproduction (live server code, ~/Downloads/wound-analyzer-vercel/api/predict.py `_classify`)
1,409 public normal-skin photos (401 held-out + 1,008 never used for training or any choice), browser-encoded:
**220 labelled, every one "possible_burn"**; faces (`oily_dry_skin_types`) 126/864 (14.6%); held-out 37/401,
independent 183/1,008. Also labelled: food, a giraffe, a fish, a clothes shop (preproc_check.json).

## 2. Cause (a combination - not a bug in inference, class mapping or the website)
- **Training labels.** About a quarter of a random 48 first-degree-burn training photos are faces, many barely
  flushed (counted by eye; an OpenCV face detector was too unreliable to count them: it fires on 8.4% of leg-wound
  photos). The classifier's "not a wound" class has no ordinary faces or everyday photos.
  One test-set "first-degree burn" photo (burn_1st_degree_kg2_693) looks like an uninjured face (flagged, not
  relabelled) - the test split partly rewards this error.
- **Score aggregation.** "possible burn" sums three burn probabilities. On the 37 held-out false alarms the
  median sum is 0.89 while the strongest single burn class is 0.70 (below the 0.75 bar). The healthy-skin gate's
  median "not a wound" score is 0.34, under its 0.5 veto.
- **Crop retry** (centre square, 70% of the shorter side, accepted at >= 0.80): when it was on, normal-skin false
  labels were 343 instead of 220 (faces 231 vs 126). It is off now - see section 4.
- **Orientation:** a photo rotated 90 degrees gets more false labels (307 vs 220); mirroring changes nothing (219).
  The site draws photos onto a canvas, which browsers rotate upright, so this matters only if a photo arrives sideways.
- **Resize:** production resizes with NEAREST, the same as training (code read). Not testable on these photos:
  they are stored at 224 px, so no resize happens - full-size phone photos remain untested.
- Not the cause: inference (saved outputs and the live code agree photo for photo) or class mapping (possible_burn = sum of the 3 burn classes, as designed).

## 3. Fixes measured (eval_v5.json, eval_v5bc.json)
Decision rules on the existing models (64 candidates: max-vs-sum burn score, stricter burn bar, lower healthy-gate
veto, thresholds 0.60-0.75): **none meets the objective.** The only dev-eligible one added 85 false alarms on
untouched photos.
Classifiers that have seen healthy photos (24 candidates). Chosen by the fixed rule: **average the deployed
classifier with the healthy-skin model (already running in production as the second gate), threshold 0.70.**

| Untouched photos | Live | Candidate | Candidate - live, paired 95% CI |
|---|---|---|---|
| Correct wound names (test + RIT, merged burns) | 208 | 206 | -13..+9 |
| False alarms (test non-wounds + 1,008 normal skin + RIT feet/chronic/close-ups) | 507 | 392 | **-191..-60** |
| Wrong injury names | 19 | 12 | -13..-2 |
| Faces labelled (all 864 face photos, held-out + independent) | 126 | 70 | |
| Test close-ups: right / wrong / non-wounds labelled (of 331 / 450) | 154 / 18 / 48 | 153 / 11 / 44 | |
| RIT: healthy feet / chronic / burns / everyday right-wrong | 158 / 115 / 35 / 19-1 | 104 / 123 / 36 / 17-1 | |

**Verdict by the pre-declared rule: NOT confirmed** - false alarms fall measurably, but correct answers do not
rise (206 vs 208, a difference within noise). It does not satisfy "more correct AND fewer false alarms", so it
is not recommended for deployment by this session. It is offered only as a measured option.
Retrained classifier N5 (deployed recipe + 1,609 healthy photos): not dev-eligible (fewer correct answers).
Control RE3_s1 (same recipe without healthy photos) answers more but with more false alarms - the healthy photos,
not retraining luck, are what remove false alarms.
Candidate code, ready but not deployed: production_mean_070/api/predict.py (3 lines differ from live; no extra
model). Server-verified against the evaluation on 2,190 photos: 2,190 match (verify_candidate.json).

## 4. Crop-retry contradiction - resolved
Both statements were true at the time. 2026-09-27 deployment jop7ssu2o (v4b) kept the crop retry at 0.80;
later that day deployment axnmf5u04 (v4c) turned it off after a pre-declared rule (no crop bar added >= 2 correct
answers per false label). Live code: `CROP_RETRY = False`. Checked on the public service today: 3 test photos
that only the crop would answer all return "unknown".

## 5. Roboflow vs live, same photos and same merged-burn rule (rf_compare_v5.json)
4,173 photos clean for both models (204 + 58 known Roboflow copies and anything resembling Roboflow train/valid
excluded; 547 of the normal-skin photos resemble Roboflow training data and were excluded), 0 errors.

| | Roboflow @0.40 | Roboflow @0.82 | Live | Candidate |
|---|---|---|---|---|
| Wound photos (148): right / wrong | 39 / 12 | 4 / 3 | 79 / 3 | 75 / 2 |
| Faces labelled (531) | 82 | 6 | 61 | 38 |
| Other healthy skin (331) | 57 | 2 | 37 | 23 |
| Healthy feet (1,508) | 60 | 3 | 150 | 97 |
| Chronic wounds (1,216) | 347 | 22 | 113 | 121 |
| Other non-wounds (439) | 118 | 15 | 49 | 45 |
Roboflow @0.40 vs live: right -59..-21, all false alarms +168..+328 (CI). Roboflow demonstrates neither more correct
answers nor fewer false alarms. Facial injuries: 8 found by eye in val/test/RIT, only 4 clean for both models -
too few to judge (per photo in rf_compare_v5.json). Missing population: real phone selfies at full resolution.

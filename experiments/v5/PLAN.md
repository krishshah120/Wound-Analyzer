# v5 plan - face / everyday-photo false "possible burn" (written 2026-10-03, before any v5 result)

Scope: measurement and a candidate only. This session does not deploy; the MRC App session owns
production. The user's own face photo is not available here and is not used or uploaded anywhere.

## Diagnosis already seen (before this plan)
On the 401 held-out healthy-skin photos the live config (possible burn = sum of 3 burn probabilities,
both gates veto at 0.5, threshold 0.75, no crop) labels 37, all "possible_burn": ~25 faces/selfies plus
food, animals, a shop. Median summed burn probability 0.89 vs strongest single burn class 0.70; healthy
gate median "not a wound" 0.34 (< 0.5 veto). The first-degree training class contains face photos.

## Candidates (fixed grid, decision rules on the existing models - no retraining)
- burn rule: `sum` (live: possible_burn confidence = summed burn prob) | `max` (confidence = strongest single
  burn class; still shown as possible_burn) | `sum85` / `sum90` (summed, but possible_burn needs >= 0.85 / 0.90)
- healthy-gate veto threshold: 0.5 (live) / 0.4 / 0.3 / 0.2 (shipped gate stays at 0.5)
- threshold t for every answer (for sum85/sum90 the burn bar is max(t, 0.85/0.90)): 0.60 / 0.65 / 0.70 / 0.75
- no crop retry (as live). 64 candidates; the live config is `sum`, 0.5, 0.75.

## Selection (development data only)
Dev = val close-ups (browser) + val arm's length (framed_views) + 401 held-out healthy-skin photos.
- right = wound photos named correctly (any burn degree answered possible_burn = right)
- wrong = wound photos given another injury name
- false alarms = non-wound photos labelled (val close + val arm) + held-out healthy photos labelled
Eligible: right > live right AND false alarms < live false alarms AND wrong <= live wrong.
Choose the eligible candidate with the fewest false alarms; ties -> most right -> higher t.
If none is eligible, report that no decision rule meets the objective (retraining would be next).

## Reported only (never used to choose)
data/test close-ups and arm's length; RIT (feet, chronic, burns, everyday; duplicate rows counted once);
the 1,008 normal-skin photos set aside in v4 (`excluded_resembles_eval`, never trained on or used for any
choice) -> indep_healthy_views.npz. Faces population = photos from the `oily_dry_skin_types` source (a
face-photo dataset), reported separately in held-out and independent sets. (An OpenCV face detector was
tried for a face census and rejected: it fires on 8.4% of RIT leg-wound photos.)
Verdict "confirmed" only if, on test close-ups + independent normal-skin + RIT together, right >= live and
false alarms < live with the paired 95% bootstrap CI of the false-alarm difference excluding 0.

## Roboflow with faces
Score the same 401 held-out + 1,008 independent normal-skin photos with the Roboflow workflow (public
dataset photos only, as in experiments/rf); report labelled counts per population for Roboflow @0.40 and
@its follow-up setting vs live and the v5 candidate.

## Addendum v5b/v5c (2026-10-03, after the v5 grid result, before any v5b/v5c result)
v5 result: 1 of 64 rule candidates eligible on dev (sum, healthy gate 0.4, t 0.70) and NOT confirmed
(untouched pool: false alarms 592 vs live 507). Decision rules on the existing models do not meet the
objective. Next, classifiers that have seen faces/everyday photos as "not a wound":
- v5b (no training): classifier = G4new (the healthy-skin gate is a full 7-class model of the same
  recipe trained on data/train + 1,609 v4 healthy photos) alone, or mean(deployed, G4new).
- v5c (training): N5 = the deployed recipe (experiment9.py --alpha 1.4 --label-smoothing 0.1,
  --train-dir = v2/b84_train + the 1,609 v4 healthy train photos in out_of_scope, --val-dir v2/b84_val),
  seed 1, one model. Control reported, not selectable: RE3_s1 (same recipe and seed, no healthy photos).
Each classifier is scored with both live gates (0.5), no crop, burn rule sum | max, t 0.60/0.65/0.70/0.75.
Selection, eligibility, tie-breaks and the verdict pool are exactly as above (same dev, same live baseline),
over the v5b + v5c candidates together. If none is confirmed, nothing is recommended for deployment.

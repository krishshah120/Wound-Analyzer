# Roboflow model vs deployed pipeline - comparison protocol
Fixed 2026-09-27, BEFORE any output of the Roboflow model was obtained (no access yet). No training.

Roboflow model: project wound-ebsdw-4atst v1 (YOLO26 X-Large detector; classes Abrasions, Blister,
Bruise, Burn, Cut, no abnormality), run through the saved workflow
`vihaan-nr-singh-gmail-com / wound-vwound-ebsdw-4atst-1-yolo26x-t1-logic` (Roboflow serverless).
Deployed pipeline: production predict.py (classifier eef2f3d3..., gate 49201744..., 0.60, crop retry).

## 1. Photos clean for BOTH models (eligibility.py -> eligible.csv)
Excluded: anything resembling (hash <= 6 with mirror, or 12-view similarity >= 0.75) a photo in the
Roboflow model's train or valid split (1,053 + 300; includes the 204 + 58 known copies of our
val/test photos), and RIT photos taken from those splits. Our models' training data is already
excluded from val/test/RIT. Caveat: all val wound photos are our models' early-stopping set.

| Category (shared) | Clean val (DEV: thresholds) | Clean test + RIT (EVALUATION) |
|---|---|---|
| abrasion | 5 | 1 |
| bruise | 9 | 11 |
| cut | 6 | 17 |
| burn (generic; our 3 degrees merged) | 89 | 119 |
| other non-wound (val/test out-of-scope, RIT close-ups) | 90 | 439 |
| healthy feet (RIT, one dataset) | - | 1,508 |
| chronic wounds (RIT, out of scope) | - | 1,298 |
**Declared in advance: abrasion, bruise and cut are far too few to establish a winner** (1 / 11 / 17
evaluation photos; a 95% interval on 17 photos spans roughly +-24 points). Only generic-burn
recognition and non-wound false alarms can be compared with useful precision.
Resolution caveat: our val/test photos exist only as 224x224 copies; the Roboflow model was trained
on 640x640. Both models receive the same bytes (see 3); a Roboflow disadvantage from low resolution
is possible and will be reported, not corrected.

## 2. How Roboflow detections become one website answer (fixed now)
Detections are read from the workflow's real output (keys discovered at run time, not assumed).
Class mapping: Abrasions -> abrasion, Bruise -> bruise, Cut -> cut, Burn -> "possible burn" (no
degree), Blister -> no shared category, "no abnormality" -> never "healthy".
With confidence threshold tau:
1. Keep detections with confidence >= tau. None left -> **uncertain** (never "healthy").
2. Top = the highest-confidence kept detection of a shared class. If none -> uncertain.
3. If a kept "no abnormality" or "Blister" detection is at least as confident as Top -> uncertain.
4. If another kept shared-class detection names a DIFFERENT category -> uncertain (conflict).
5. Otherwise answer Top's category.

## 3. Inputs and comparators
Both pipelines get the same bytes: each photo encoded as the MRC site does (longest edge <= 1024 px,
JPEG quality 82). Deployed pipeline answers are mapped to shared categories AFTER its decision
(burn_1st/2nd/3rd -> possible burn); its crop retry stays as in production.
Thresholds, chosen on the clean VAL photos only:
- Roboflow "selected" point: the lowest tau in 0.05..0.95 (step 0.01) whose clean-val accuracy among
  answers (4 shared categories) is >= the deployed pipeline's clean-val accuracy at 0.60.
- Matched coverage: tau at which Roboflow answers as many clean-val wound photos as the deployed
  pipeline does at 0.60 (nearest not below); and the deployed threshold (0.40..0.95) at which it
  answers as many as Roboflow at its selected point.
Also reported, exploratory: "agreement" combination - answer only when both give the same category.

## 4. Metrics (clean test + RIT evaluation photos; each reported separately)
Per category: right / wrong / unanswered with denominators; non-wound photos given any injury label
by population (other non-wound, healthy feet, chronic wounds); burn shown as another injury;
end-to-end latency per photo (median, p95) measured from this Mac - Roboflow over the internet to
serverless, deployed locally - NOT Vercel. Paired bootstrap 95% CIs over duplicate groups.
Verdict wording: "Roboflow better" only if a primary difference (generic-burn right answers, or
non-wound false labels) has a CI excluding 0 in its favour AND no other primary measure is worse
with a CI excluding 0; otherwise "no demonstrated advantage" / "too small to tell".

## 5. If Roboflow shows an advantage
Reversible preview only (production unchanged): server-side key, "possible burn" wording, no degree.
Before any site change: (a) the photo would leave our servers for Roboflow's - the privacy page
("not stored, not logged") must be checked against Roboflow's retention terms and the workflow must
contain no dataset-upload / active-learning block; (b) cost per photo from measured execution time;
(c) licence/terms for serving this YOLO26 model on a public site (not verified); (d) Vihaan's OK.

## Access needed (not available in the session that wrote this)
Either authorise the Roboflow connector (claude.ai connector settings, or `/mcp` in an interactive
Claude Code terminal), or put the API key in an environment variable `ROBOFLOW_API_KEY` /
the file `~/.config/wound-analyzer/roboflow_api_key` (chmod 600) - never in chat or in the repo.

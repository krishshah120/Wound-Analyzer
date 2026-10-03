# EBIS evaluation protocol - frozen 2026-09-27, before any EBIS image was received

Machine-readable version: `ebis_frozen.json` (model hashes, thresholds, metrics, decision rule).
Runner: `ebis_eval.py`, checked on 67 existing burn photos (`ebis_smoke.json`: it flagged all 67 as
overlapping and reproduced the known counts 23 / 35 / 23 / 34 burns recognised).

## Before running
1. **Permitted use.** Read the VEDAs Lab reply and record what it allows in
   `experiments/v3/ebis_permission.txt` (the runner refuses without it). If they allow only offline
   academic evaluation, results go in the report and are not used to change the public site without
   asking them again. If they refuse, EBIS is not used.
2. **Storage.** Keep EBIS outside the repo (it may not be redistributed); never commit its images.
3. **No training on EBIS.** Its images are never copied into any training folder, and no threshold,
   model or rule is chosen using EBIS - everything below is already fixed.

## Frozen candidates (all with the shipped gate, threshold 0.5, and the production crop retry)
| | Classifier | Output | Threshold (chosen on val) |
|---|---|---|---|
| P0 | deployed (sha eef2f3d3...) | 6 classes; any burn degree counts as "burn" | 0.60 (site contract) |
| P1 | deployed | generic burn = sum of the 3 burn probabilities | 0.7637 |
| P2 | RE3 ensemble (3 TFLite members, hashes in json) | 6 classes | 0.60 |
| P3 | RE3 ensemble | generic burn | 0.7886 |
Preprocessing: the site's browser shrink (<= 1024 px, JPEG quality 82), then production predict.py
(NEAREST 224x224, x/127.5 - 1).

## Overlap check (automatic, before any metric)
Every EBIS image is compared with all existing collections (training, validation, test, extra and
out-of-scope downloads, Roboflow/Baid candidates, the reserved set, the hard-negative pool) by
sha256, decoded pixels, difference hash <= 6 (mirror allowed) and 12-view similarity >= 0.75.
Overlapping images are listed and excluded from the primary analysis (EBIS was also collected from
Google image search, so some overlap with our web-scraped burns is expected).

## Metrics
- Primary: **burn recognised** - share of eligible EBIS images the pipeline labels as a burn, with a
  Wilson 95% interval.
- Primary comparison: **P1 - P0**, paired bootstrap over EBIS near-duplicate groups.
- Secondary: burn shown as another injury (abrasion/bruise/cut), unknown share, results by burn-mask
  area (< 15%, 15-40%, >= 40% of the frame) and by EBIS's official split; P3 - P2, P2 - P0, P3 - P1.
- Not measurable on EBIS: false alarms (it contains burns only) and burn depth (no degree labels).

## Decision rule for P1 (the "possible burn" candidate) - all must hold
1. P1 - P0 burn recognised: paired 95% CI lower bound > 0.
2. P1's burn-shown-as-other-injury share is at most P0's + 2 points.
3. Existing development evidence, not re-tuned: reserved-set healthy feet P1 - P0 at the production
   crop setting, upper CI <= +3 points (measured 2026-09-26: -0.9 to +2.8).
4. A sourced general burn first-aid text exists (today's per-degree tips contradict each other on
   water) before any site change.
5. The EBIS reply permits the intended use.
Passing makes P1 an experimental preview candidate with rollback, not an automatic deployment.

## Command
```
cd experiments/v3
../../.venv/bin/python ebis_eval.py --images /path/to/EBIS/images --masks /path/to/EBIS/masks [--splits splits.csv]
```

## Clarifications added at close-out (2026-09-27; nothing above changed)
- "0.60 (site contract)" in the table means the threshold in production today. It is a configurable
  operating threshold, not a safety standard.
- Report generic-burn recognition (this protocol) separately from burn-degree classification (not
  measurable on EBIS) and from non-burn false alarms (not measurable on EBIS; use existing sets).
- An improvement on EBIS alone does not show that the whole analyzer improved: it only covers
  whether burns are recognised.

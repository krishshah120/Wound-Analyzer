# Wound analyzer v3 - results (2026-09-26)

**Bottom line: nothing met its pre-declared rule, so there is no preview and production is
unchanged.** Two findings are worth a decision:
- A retrained 3-model ensemble (real saved models, not old predictions) halved healthy-foot false
  alarms on the reserved set (94 vs 183 of 1,613; paired 95% CI -11.0 to -2.5 points) and gave fewer
  wrong labels overall (329 vs 457), but answered 12 fewer photos correctly (237 vs 249) - more than
  the 5 its rule allowed. Deployable, verified artifacts are saved; shipping it is your trade-off.
- Summing the three burn probabilities into "possible burn" recognised more real burns (35 vs 23
  of 67), but was chosen after the reserved set had been seen - it is the declared candidate for the
  next evaluation on new photos, not a result.
All evidence below is development evidence: every set has been looked at. Nothing here is
independent confirmation or clinical validation. Full log: `LEDGER.md`; rules: `PLAN.md`.

Constraint of this round: no human image review, no new photos, no restricted data, no Roboflow key.

## What counts as a test now
data/val and data/test have been used in every round since G: **development benchmarks**. The
reserved independent test (RIT, 3,063 photos) was scored on the deployed pipeline in v2 and used
here only for reporting, but its results shaped this round, so it is **now a development benchmark
too**. No untouched test remains - which is why no candidate could have been more than experimental.

## Comparison table (production crop retry C1, browser-encoded photos unless noted)
Six-category task:
| Candidate | Val answered / right | Val non-wound | Test answered / right | Test non-wound | 3rd shown other (val/test) | RIT everyday injuries right | RIT healthy feet labelled | Verdict |
|---|---|---|---|---|---|---|---|---|
| **Deployed (production)** | 126/251, 84.1% | 13/94 | 160/331, 74.4% | 61/450 | 2 / 5 | 24/48 (0 wrong) | 183/1613 | baseline |
| No crop retry (C0) | 117/251, 85.5% | 12/94 | 156/331, 74.4% | 56/450 | 2 / 4 | 23/48 (0 wrong) | 175/1613 | no rule declared - your call |
| Crop only if views agree (C2) | 122/251, 85.2% | 13/94 | 160/331, 74.4% | 58/450 | 2 / 5 | 24/48 | 183/1613 | fails (keeps a 3rd-degree error) |
| Ensemble of 3 reruns (stored files, full frame, 0.60) | 106/251, 83.0% | 11/94 | 139/331, 79.9% | 55/450 | 3 / 2 | not measurable (weights not saved) | - | inconclusive |
| Conflict-excluded training, pilot (stored files, full frame, 0.60) | 112/251, 79.5% | 11/94 | 152/331, 75.0% | 59/450 | 4 / 4 | - | - | stopped at pilot |

"Possible burn" task (a different, easier task - never compare these with six-category accuracy);
each option answers the same 117 of 251 val wound photos, thresholds chosen on val (all >= 0.60):
| Option | Test answered / right | Test non-wound | RIT burns recognised | RIT healthy feet labelled | RIT everyday injuries right, shown burn |
|---|---|---|---|---|---|
| Today's answers, burn degrees merged | 160/331, 88.1% | 61/450 | 23/67 | 183/1613 | 24/48, 0 |
| Burn probabilities summed (t 0.764) | 176/331, 89.8% | 66/450 | **35/67** (+8.8 to +27.0 pts) | 197/1613 (-0.9 to +2.8 pts) | 22/48, 2 |
| Model trained on merged labels (3 seeds) | 155-161/331, 88.2-95.5% | 53-72/450 | 26-28/67 | 359-420/1613 | 18-20/48, 2-3 |
RIT ranges in brackets are paired 95% intervals over duplicate groups.

Runtime on this Mac (not Vercel): deployed pair 7.0 ms median per photo, 166 MB, 34.6 MB of files;
3-model ensemble 13.1 ms, 258 MB, 69.1 MB.

## What each experiment showed
1. **Ensemble (X1).** Averaging reruns halves run-to-run variation (accuracy SD 3.9 -> 2.0 points).
   It is steadier, not better: 82.8% vs 85.3% on val at matched coverage and 77.9% vs 75.5% on test,
   both inside the uncertainty. The three models disagree on only 1 of 116 answers, so
   "abstain when they disagree" has nothing to work with.
2. **Crop retry (X2).** Worth about as much as it costs: pooled over val and test it adds 9 correct
   answers and 10 wrong labels. Requiring the two views to agree did not remove the third-degree burn
   it mislabels, because both views make the same mistake - agreement is not protection.
3. **Duplicate label conflicts (X3).** 86 training photos are the same picture as a copy labelled a
   different burn degree (mostly by the separate "fares" download, which disagrees with our labels on
   50 of 683 shared pictures). Training without them had no measurable effect: stopped at the pilot.
   The training set also contains collages and textbook diagrams (recorded, not removed).
4. **"Possible burn" (X4).** Merging degrees is the right direction for burns: the summed option
   recognised 35 of 67 real burns instead of 23. Training a model directly on the merged labels
   looked excellent on the reused benchmarks but, in all three seeds, labelled up to twice as many
   healthy feet as injuries. Neither option knows how deep or serious a burn is.
5. **Healthy-skin negatives (X5).** Not runnable: 1,137 of the 1,144 spare healthy-foot photos are
   near-copies of the reserved-test feet. The healthy-skin false alarms (11% of healthy feet) cannot
   be fixed or honestly measured with photos from that one source.

## Second pass (X6) - validation, retrained ensemble, narrow label cleanup
- **Saved predictions validated** (x6_validate.py): true probability vectors, identical class order,
  rows keyed by file name, no evaluated photo in any member's training data. Caveat found: all 251
  val wound photos are the early-stopping set of these runs and of the deployed model.
- **Retrained ensemble RE3** (seeds 1-3 of the deployed recipe, all kept, no search), complete
  pipeline with gate and crop retry, 0.60, photos as the site sends them:

| | Val answered / right | Val non-wound | Test answered / right | Test non-wound | 3rd shown other | RIT injuries right | RIT healthy feet | RIT burns |
|---|---|---|---|---|---|---|---|---|
| Deployed | 126 / 106 | 13/94 | 160 / 119 | 61/450 | 7 | 24/48 (0 wrong) | 183/1613 | 23/67 |
| RE3 | 120 / 102 | 9/94 | 144 / 111 | 57/450 | 6 | 24/48 (1 wrong) | **94/1613** | 23/67 |

  Paired 95% CIs (RE3 - deployed): test accuracy among answers -2.6..+8.0 points, test coverage
  -9.8..0.0, RIT healthy feet -11.0..-2.5. Its three members alone labelled 202, 146 and 140 feet:
  averaging is what removes the false alarms. Declared rule: fewer wrong labels (pass), at most 5
  fewer correct answers (FAIL: 12 fewer), 3rd-degree errors not more (pass), feet not more (pass).
  At matched coverage it would need a threshold of 0.5825 (below 0.60), so that is not offered.
  Runtime here: 13.1 ms and 258 MB vs 7.0 ms and 166 MB; 69 MB of model files vs 35 MB.
  Artifacts: `artifacts/RE3/` (3 TFLite members, each equal to its Keras model on 1,126 photos to
  7.4e-6 with 0 decision changes; manifest with seeds, hashes, class list, preprocessing, data
  manifest, library versions). The .tflite/.keras files are not committed (17-18 MB each).
- **Crop policies** on the same photos (answered / right, val + test): deployed no-retry 273 / 216,
  retry 286 / 225, agree-only 282 / 223; the retry adds 13 wound-photo answers (9 right, 4 wrong) plus 6 non-wound labels, and the
  agree-only rule gives up 4 answers (2 right) without removing the 3rd-degree error. For RE3 the
  retry adds 6 wound answers (5 right), 2 val/test non-wound labels and 6 RIT healthy-foot labels.
  Fewer errors do come at the cost of fewer answers in every case; all photos stay in the denominators.
- **Narrow label cleanup:** only 24 training photos are identical decoded images with incompatible
  labels. Excluding them (3 seeds): val 83.9 vs 82.2%, test 75.8 vs 76.4% at matched coverage;
  3-model CIs -3.0..+8.9 (val), -5.9..+3.4 (test) -> inconclusive.

## Follow-up analysis (2026-09-27, saved predictions only) - the RE3 verdict above stands
**Does the ensemble beat simply raising the deployed model's threshold?** Deployed raised to 0.625 so
it answers the same 120 val wound photos as RE3 (threshold chosen on val):

| Population | Deployed @0.625 | RE3 @0.60 | RE3 - deployed, 95% CI |
|---|---|---|---|
| Test wound photos (331): answered / right / wrong | 148 / 113 / 35 | 144 / 111 / 33 | right -14..+10, wrong -11..+7 |
| Test other non-wounds labelled (450) | 52 | 57 | -7..+16 |
| RIT everyday injuries (48): right / wrong | 23 / 0 | 24 / 1 | too few |
| RIT burns given a burn label (67) | 23 | 23 | - |
| **RIT healthy feet labelled (1,613)** | **154** | **94** | **-106..-25** |
| RIT chronic wounds labelled (1,313) | 116 | 112 | -24..+17 |

Raising the threshold matches RE3 on injuries and other non-wounds; RE3's advantage is confined to
healthy feet (one dataset, one setting). "457 wrong labels" pooled 8 populations, 40% of it healthy
feet - see LEDGER.md; it should not be read as one score. Crop retry, reconciled per photo: 19 new
labels on val + test, 9 right and 10 wrong (4 injury names, 6 non-wound photos); no existing answer
changes. Artifacts: ~/Downloads/wound-analyzer-artifacts/v3-2026-09-27 (checksummed). EBIS evaluation
frozen in EBIS_PROTOCOL.md.

## Recommendation
- Keep production as it is. Nothing met its rule. If you prefer fewer false alarms over a few more
  answers, RE3 is the one ready-made option (tell me and I will build its preview with rollback).
- **Next evaluation, already declared:** the summed "possible burn" output at threshold 0.764, judged
  on photos never used here - EBIS (if its licence allows) is exactly the right burn set for it.
  Before any site change, the generic burn text must come from a first-aid source: today's per-degree
  tips contradict each other (1st/2nd say cool with water, 3rd says do not apply water), so they
  cannot simply be merged.
- **Decision for you:** the crop retry roughly trades one correct answer for one wrong label. Removing
  it would cost about 9 answers and avoid about 10 wrong labels on val+test. No rule was declared for
  this, so it is a product choice, not a finding.
- **The one data need that blocks the biggest error:** healthy-skin phone photos from sources other
  than the Lower Limb dataset (arms, hands, legs, faces, several skin tones and lighting), for both
  training the gate and testing it. About 1,670 are needed to measure the false-alarm rate to +/-1.5
  points (v2 sample-size table). The learning curve cannot say how much new data would help; the v2
  claim that it would add "only a point or two" was withdrawn.
- **Roboflow:** no comparison is possible this round (no key; it is a detector, scored differently;
  trained on 204 copies of our val/test photos). A claim that ours is "better than Roboflow" would
  have no evidence behind it.

## Reproduce (from the worktree root; needs the gitignored data folders and the original downloads)
```
cd experiments/v3
../../.venv/bin/python x1_ensemble.py                       # X1
../../.venv/bin/python views.py deployed                    # per-photo full-frame + crop probabilities
../../.venv/bin/python x2_crop.py                           # X2
../../.venv/bin/python x3_conflicts.py && ../../.venv/bin/python x3_verify.py
zsh queue_x3.sh 1 && ../../.venv/bin/python x3_eval.py      # X3 pilot
../../.venv/bin/python x4_burn.py                           # X4a
zsh queue_x4b.sh 1 2 3 && for s in 1 2 3; do ../../.venv/bin/python views.py X4B_s$s --clf models/X4B_s$s.keras; done
../../.venv/bin/python x4_matched.py && ../../.venv/bin/python x4_summed_ci.py
../../.venv/bin/python x5_pool.py                           # X5 pool (built once)
../../.venv/bin/python runtime.py 1; ../../.venv/bin/python runtime.py 3
# second pass (X6)
../../.venv/bin/python x6_validate.py && ../../.venv/bin/python x6_t1_conflicts.py
zsh queue_x6.sh 1 2 3                                      # RE3 members (weights saved) + X6C seeds
for s in 1 2 3; do ../../.venv/bin/python views.py RE3_s$s --clf models/RE3_s$s.keras; done
../../.venv/bin/python x6_ensemble.py                       # RE3 vs complete deployed pipeline
../../.venv/bin/python x6_package.py RE3 RE3 1 2 3          # TFLite export + check + manifest
../../.venv/bin/python x6c_eval.py && ../../.venv/bin/python x3_eval.py X6C
```
Not committed: model files (models/*.keras), views_*.npz, image folders, symlink training folders.

## Close-out clarifications (2026-09-27, existing results only)
**1. Current pipeline (with crop retry) vs no crop retry** - deployed models, 0.60, browser input,
each quantity separately (from x6_ensemble.json and x4_burn_deployed.json):

| | No crop retry | Current pipeline (crop retry) |
|---|---|---|
| Val wound photos (251): right / wrong / abstained | 100 / 17 / 134 | 106 / 20 / 125 |
| Test wound photos (331): right / wrong / abstained | 116 / 40 / 175 | 119 / 41 / 171 |
| 3rd-degree burns shown a milder injury (val 31 + test 40) | 2 + 4 = 6 | 2 + 5 = 7 |
| 2nd-degree shown 1st degree / abrasion / bruise (val + test) | 3 + 7 = 10 | 4 + 7 = 11 |
| Val non-wounds labelled (94) | 12 | 13 |
| Test non-wounds labelled (450) | 56 | 61 |
| RIT everyday injuries (48): right / wrong / abstained | 23 / 0 / 25 | 24 / 0 / 24 |
| RIT burns, degree unknown (67): burn / other injury / unknown | 23 / 1 / 43 | 23 / 1 / 43 |
| RIT healthy feet labelled (1,613), of which 3rd degree | 175 (12) | 183 (12) |
| RIT chronic wounds labelled (1,313) | 131 | 133 |
| RIT healthy close-ups labelled (22) | 6 | 6 |

The retry adds 9 correct answers and 1 RIT everyday answer; it also adds 4 wrong injury names -
including one extra 3rd-degree burn shown as an abrasion and one 2nd shown as 1st - and 6 val/test
plus 8 RIT-feet and 2 chronic-wound false labels. Whether that is worth it depends on how much a
missed answer costs against a wrong or reassuring one; the pooled counts alone do not decide it.

**2. Wording corrections.**
- The ensemble's healthy-foot advantage (94 vs 154 feet labelled at matched val coverage) was
  measured on ONE dataset: Lower Limb "Nomal" feet, one clinic setting and camera. It is specific to
  that dataset; it is not evidence about healthy skin in general.
- 0.60 is a configurable operating threshold that the site currently uses, not an established safety
  standard. Earlier text calling it "the site contract" (PLAN.md, EBIS_PROTOCOL.md) means only "the
  value in production today". Keeping or changing it should follow measured trade-offs between
  answers given, wrong and reassuring labels, and false alarms on non-wounds.

## Round closed
Production unchanged. No further training or threshold searches on these benchmarks. Next step is the
frozen EBIS evaluation (EBIS_PROTOCOL.md; handoff in HANDOFF.md).

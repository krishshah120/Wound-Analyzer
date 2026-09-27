# Wound analyzer v3 - results (2026-09-26)

**Bottom line: nothing qualified under the rules fixed in PLAN.md, so there is no preview and
production is unchanged.** One option found along the way, summing the three burn probabilities
into one "possible burn" at a stricter threshold, recognised clearly more real burns on the
reserved test. It was chosen after that test had been seen, so it is the declared candidate for the
next evaluation, not a result. Full log: `LEDGER.md`. Rules: `PLAN.md` (committed before any result).

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

## Recommendation
- Keep production as it is. Nothing here earned a deployment.
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
```
Not committed: model files (models/*.keras), views_*.npz, image folders, symlink training folders.

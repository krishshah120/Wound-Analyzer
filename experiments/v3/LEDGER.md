# v3 ledger - results in the order they were obtained (rules: PLAN.md)

## X1 ensemble (saved probabilities; x1_ensemble.py -> x1_ensemble.json) - NOT ADOPTED
Checks passed: all 5 exact-recipe runs have the deployed class order, photo order and loader.
Instability (val, 5 single runs vs the 10 possible 3-member ensembles):
  accuracy at 46.2% coverage  singles 75.0-85.3 (SD 3.9)   ensembles 77.6-84.5 (SD 2.0)
  coverage at 0.60            singles 41.4-50.2 (SD 3.5)   ensembles 40.6-44.6 (SD 1.3)
  AURC                        singles mean 0.180           ensembles mean 0.160
  -> averaging roughly halves run-to-run variation (the 10 ensembles overlap: descriptive only).
ENS3 (R_exact seeds 1-3) vs deployed single model:
  val @0.60     ENS3 42.2% / 83.0% / 11 of 94 / tier A 3   deployed 46.2% / 85.3% / 16 / 1
  test @0.60    ENS3 42.0% / 79.9% / 55 of 450 / A 2       deployed 48.0% / 75.5% / 68 / 3
  val matched coverage 46.2%: 82.8% vs 85.3% (diff CI -9.3..+3.8); test at the val-chosen threshold
  77.9% vs 75.5% (CI -3.1..+7.4). AURC val 0.166 vs 0.171, test 0.222 vs 0.234.
  Coverage at matched error (val 14.7%): ENS3 38.6% vs deployed 46.6%.
  Verdict: inconclusive; fails rule MC (needs +3 on val) -> no members retrained.
Disagreement abstention: members disagreed on 1 of 116 answered val photos -> rule not met, unused.
Caveat: val favours the deployed model (best of 6 runs on val; its wound photos were its
early-stopping set - also true for R_exact).

## X2 crop policy (inference only; views.py -> views_deployed.npz, x2_crop.py -> x2_crop_deployed.json)
views.py reproduces the v2 production baseline exactly (val/test raw+browser, RIT).
Crop step's contribution over full frame only, pooled val+test (browser):
  C1 production: +9 correct, +4 wrong injury labels, +6 non-wound labelled, +1 tier A
  C2 agreement:  +7 correct, +2 wrong, +3 non-wound, +1 tier A (the 3rd-degree burn is labelled
                 another injury by BOTH views - agreement did not protect)
  -> C2 fails its rule (tier A must be 0). Production (C1) unchanged by rule.
RIT: crop adds 1 correct distant-injury answer (10 -> 11 of 32 at < 15% of frame) and 8 healthy-foot
false labels (175 -> 183 of 1,613). Observation (no rule declared for it): the crop step is roughly a
one-for-one trade of correct answers for wrong labels; removing it is a product decision for Vihaan.

## X3 objective label conflicts (x3_conflicts.py, x3_verify.py, x3_label_tiers.csv, x3_quarantine.txt)
64 duplicate groups (T1/T2) touching the deployed training set carry different labels; 86 b84_train
photos quarantined (2nd 41, 1st 26, 3rd 19). Mostly the same picture given a different burn degree
by the separate 'fares' download (it agrees with our label on 633 of 683 matched burn pictures,
differs on 50), plus a few burns duplicated as out-of-scope photos. A 12-pair contact sheet confirmed
the pairs are the same picture (a duplicate check, not a medical judgement). Of the 319 v2 flags:
16 T1/T2 conflict (train), 35 T3 caution (29 train / 3 val / 3 test), 268 T4 similar only.
Also seen on the sheet: collages and textbook diagrams among training "photos" (not removed).
Pilot B seed 1 (18.1 min, 1,985 photos): val AURC 0.198 > 0.183 -> STOPPED per rule.
  vs matched A seed 1: AURC 0.198 vs 0.207, accuracy at matched coverage 78.4% vs 80.2%, val
  non-wound 11 vs 14, tier A 4 vs 3; test at val threshold 75.0% vs 74.5%. Inconclusive.
X3b not run (its trigger, fewer than 10 quarantined photos, did not fire).

## X4a possible burn, no retraining (x4_burn.py -> x4_burn_deployed.json) - different task
Browser input, full frame (C0). "after" = today's decisions mapped; "summed" = burn probs summed.
  val   after  46.6% / 92.3% / non-wound 12 of 94     summed 70.9% / 93.3% / 25 of 94
  test  after  47.1% / 88.5% / 56 of 450              summed 65.9% / 88.1% / 112 of 450
  RIT burns (degree unknown, 67): burn label after 23, summed 43
  RIT healthy feet labelled: after 175 of 1,613, summed 543 (429 as burn)
  -> summing more than triples healthy-foot false alarms; not usable as is.
Merged-task AURC (browser val/test): 0.046 / 0.082. X4b rule: 8 of 12 val errors involve burn ->
justified; pilot seed 1 started.

## X5 hard negatives for the gate - NOT RUN (pool empty under the declared rule)
Of 1,144 healthy-foot photos not in the RIT, 1,137 are within similarity 0.75 / hash 6 of a RIT (or
val/test) photo; 7 remain. The dataset's feet are photographed in one setting and are near-duplicates
of each other, so training on them would effectively train on the RIT feet benchmark. The rule is
not loosened after seeing this. Other training-eligible healthy skin (normal_skin, skindis_normal)
is already in the shipped gate's training data. Consequence: healthy-skin false alarms can only be
addressed with healthy-skin photos from a DIFFERENT source than the benchmark.

## X4 at matched coverage (x4_matched.py -> x4_matched.json; merged task, browser, threshold from val)
Each option answers 117 of 251 val wound photos (today's count); thresholds: after 0.60, summed
0.764, X4B_s1 0.779 (all >= 0.60 - no threshold was lowered).
                       test C0: answered / acc / non-wound   RIT (C1): burns / feet labelled / in-scope correct, shown burn
  after (today)        156/331  88.5%  56/450                23/67   183/1613   24/48, 0
  summed               165/331  89.7%  54/450                35/67   197/1613   22/48, 2
  X4B_s1 (trained)     150/331  96.0%  60/450                28/67   420/1613   20/48, 2
X4B_s1 scores 100% on val because val's wound photos were its early-stopping set; it doubles
healthy-foot false alarms on the RIT (354 vs 175 at C0) - better on the reused benchmarks, worse on
real healthy skin.
Exploratory (NOT pre-declared; x4_summed_ci.py -> x4_summed_rit_ci.json), summed vs after on the RIT,
paired 95% CIs over duplicate groups, C0 / C1:
  burns recognised  34 vs 23 (+7.5..+25.5 pts) / 35 vs 23 (+8.8..+27.0)
  feet labelled     149 vs 175 (-4.3..+0.1) / 197 vs 183 (-0.9..+2.8)
  in-scope correct  20 vs 23 (-13.6..0.0) / 22 vs 24 (-10.4..0.0); shown burn 2 vs 0 (0..+10.6)
Because it was chosen after the RIT had been looked at, it cannot qualify this round; it is the
pre-declared candidate for the next evaluation on genuinely new burn and non-burn photos.

## X4b merged-taxonomy model, seeds 1-3 (queue_x4b.sh; views_X4B_s*.npz; x4_matched.json) - NOT ADOPTED
Pilot val merged-task AURC 0.030 <= 0.046 -> seeds 2-3 run (11.2 and 33.0 min). AURC val / test:
s1 0.030 / 0.070, s2 0.049 / 0.093, s3 0.033 / 0.069 (summed, no retraining: 0.046 / 0.082).
At matched val coverage (117/251): test accuracy 96.0 / 89.7 / 93.8% (today 88.5, summed 89.7);
RIT healthy feet labelled (C0) 354 / 321 / 261 (today 175, summed 149); RIT in-scope correct
17 / 18 / 11 of 48 with 2 / 2 / 3 shown burn (today 23, 0). Consistent across seeds: training on
the merged class wins on the reused benchmarks and loses on real healthy skin and everyday injuries.

## Runtime (runtime.py -> runtime.jsonl; this Mac, M4 CPU, LiteRT float32, full frame, not Vercel)
1 classifier + gate: median 7.0 ms (p95 8.7), peak RSS 166 MB, 34.6 MB of model files.
3 classifiers + gate (ensemble): median 13.1 ms (p95 13.9), 258 MB, 69.1 MB. Merged/summed options
change no model size (summing is arithmetic; X4B has the same architecture).

## Budget used
4 training runs (X3B s1; X4B s1-3), 85.6 min of training, of the 10 runs / ~7 h allowed. X5 not run
(empty pool). No cloud, no purchases.

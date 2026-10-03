# v4 ledger (rules: PLAN.md, committed before the data were built)

## Data (prep_negatives.py -> new_neg_manifest.csv)
4,659 new photos extracted (oily/dry skin types 2,728, SkinDisease Unknown_Normal 1,837, ghost 94);
1,008 dropped as resembling a RIT/val/test photo, 1,641 as duplicates of another new photo;
kept 2,010: 1,609 training (994 / 586 / 29) and 401 held out (248 / 146 / 7), split by duplicate group.
Licences: ghost CC BY 4.0; the other two sources state none.

## G4 gate (queue_v4.sh; gate_eval.py -> gate_eval_G4new_s1.json) - FAILED
G4ctrl_s1 46.9 min, G4new_s1 35.0 min. Deployed classifier, 6 categories, 0.60, crop retry, browser:
                        shipped   G4ctrl   G4new
  RIT healthy feet      183       179      184        (rule: <= 91 and <= 89)
  held-out oily/dry     41/146    39/146   21/146
  held-out Unknown_Norm 39/248    40/248   18/248
  held-out ghost        5/7       3/7      3/7
  val right / nw        106 / 13  107 / 16 107 / 14
  test right / nw       119 / 61  118 / 59 116 / 65
  RIT everyday right    24        25       25
Val correct answers lost 0 / 0 (browser / raw); RIT everyday answers lost 0; tier A unchanged.
The new negatives halve false alarms on held-out photos of their OWN sources but do nothing for the
out-of-source feet (faces and objects do not teach it what healthy legs and feet look like).
Seed 2 not run (pilot failed).

## Operating point (op_point.py -> op_point_deployed.json, op_point_G4new_s1.json)
Rule: lowest of 0.60..0.40 with val non-wound labelled <= 13/94. No threshold qualifies with either
gate: "possible burn" labels >= 26/94 val non-wounds (shipped gate) even at 0.60.
Shipped gate, possible burn @0.60 vs today: val right 172 vs 106, wrong 14 vs 20; test right 197 vs 119,
wrong 27 vs 41, non-wound 120 vs 61 /450; RIT burns 44 vs 23 /67, healthy feet 560 vs 183 /1613,
chronic 284 vs 124 /1231, everyday 24 right + 3 wrong vs 24 + 0 /48.
Decision for Vihaan: the rule was ours; he stated he wants more answers and accepts more wrong ones.

## DEPLOYED 2026-09-27 - "possible burn" output (Vihaan's decision after seeing the table above)
Change: api/predict.py in ~/Downloads/wound-analyzer-vercel - burn probabilities summed into
possible_burn; decide() otherwise unchanged (0.60, gate 0.5, crop retry 0.80); models unchanged
(classifier eef2f3d3..., gate 49201744...). Copy + hashes: production_possible_burn/.
Verified before deploying (verify_backend.py -> verify_backend.json): the modified server code on
every val/test/RIT photo reproduces op_point_deployed.json @0.60 exactly (val 172/14/65 nw 26; test
197/27/107 nw 120; RIT everyday 24 right / 3 wrong, burns 44/67, feet 560/1613, chronic 284/1231).
Preview wound-analyzer-vercel-3fuz4j8qn (identical answers to local on 2 photos) -> production
wound-analyzer-vercel-imjnkbxko; end-to-end through mrcmiracle.vercel.app/api/wound returns
possible_burn with the sourced tips. Site text: mrcmiracle.github.io PR #2 (merged, live).
Rollback (restores the previous production build, 8 days old):
  cd ~/Downloads/wound-analyzer-vercel && ~/Downloads/mrc-miracle/node_modules/.bin/vercel rollback https://wound-analyzer-vercel-qpgrxqjqx-vihaannrsingh-cmyk.vercel.app
and restore the code from ~/Downloads/wound-analyzer-artifacts/backend-before-possible-burn-2026-09-27/.

## v4b - fewer false alarms (Vihaan, 2026-09-27: "it gives any picture a possible burn label ... I
## don't want to accept false alarms. deploy healthy skin photos") - rule fixed before computing
Output stays "possible burn". Options: gate = shipped / G4new (trained with the healthy-skin photos) /
either (unknown if EITHER gate says out of scope); threshold 0.60, 0.65, ..., 0.95; crop retry unchanged.
Choice uses development data only: val and the 401 held-out healthy-skin photos (heldout_views.npz).
Allowed: val non-wound labelled <= 13/94 AND held-out healthy-skin photos labelled <= the count of the
pre-switch production (6 categories, shipped gate, 0.60) on the same photos. Among allowed, the most
val wound photos named correctly (tie -> higher threshold). Test and RIT are reported, never used.

## v4b RESULT + DEPLOYED 2026-09-27 (fewer_false_alarms.py -> fewer_false_alarms.json)
Rule chose "either@0.75": possible-burn output, unknown if EITHER gate (shipped or G4new) says out of
scope, confidence >= 0.75, crop retry unchanged (>= 0.80).
                      before today   live 09-27 AM   chosen
  val right/wrong/nw  106/20/13      172/14/26       127/8/13
  held-out healthy    85/401         237/401         58/401
  test right/wrong/nw 119/41/61      197/27/120      163/19/59
  RIT feet / chronic  183 / 124      560 / 284       202 / 127    (of 1,613 / 1,231)
  RIT burns / everyday 23 / 24r0w    44 / 24r3w      36 / 22r1w
Stricter options measured (not chosen): either@0.85 test 115/8/38, held-out 41, feet 73.
G4new exported to TFLite (export_g4.py): max diff 4.6e-6 on 1,126 photos, 0 gate flips;
sha256 e15b336b... Server verified on every val/test/RIT/held-out photo (verify_backend_v4b.py): MATCH.
Production wound-analyzer-vercel-jop7ssu2o (public alias confirmed: threshold 75, two gates).
Site: CONFIDENCE_THRESHOLD_PCT 60 -> 75 (mrcmiracle.github.io 8c57e87, wording only).
Rollback: previous version (possible burn, one gate, 0.60) = wound-analyzer-vercel-imjnkbxko;
pre-09-27 version = wound-analyzer-vercel-qpgrxqjqx. Code copies: production_possible_burn/,
production_two_gates_075/, ~/Downloads/wound-analyzer-artifacts/. Models: artifacts/v4-2026-09-27/.

## v4c - crop-retry bar under the deployed config (raised by the MRC App session; rule fixed before computing)
Deployed config: possible burn, two gates, 0.75. Crop options: none, or crop answer accepted at
>= 0.80 / 0.85 / 0.90 / 0.95. Development data: val close-ups (data/val), val arm's-length
(experiments/data_frame50/val, the framing used for the model card's "31%"), held-out healthy-skin
photos. Keep the LOWEST bar at which the crop step adds >= 2 correct answers per false label it adds
(wrong injury names + non-wound photos labelled + healthy-skin photos labelled), pooled over those
three sets; if no bar qualifies, remove the crop retry. Test (close-up and arm's-length) and RIT reported only.
Model card figures (for the MRC App session) are measured with the deployed server code itself on
data/test and experiments/data_frame50/test, browser-encoded; a burn photo answered "possible_burn"
counts as right (merged scoring), and the old model is re-scored the same way for a like-for-like line.

## v4c RESULT + DEPLOYED 2026-09-27 (crop_and_card.py -> crop_and_card.json; crop_removal_rit.json)
Old config reproduces the model card's close-up figures exactly (160 named / 119 right / 61 non-wounds).
It does NOT reproduce the card's arm's-length figures on data_frame50 (67 / 44 / 61 vs 101 / 69 / 42):
the card used a different framing set (its non-wound count differs too) - the MRC App session measures it.
Crop sweep on dev (val close + val arm + held-out healthy), added correct vs added false labels:
0.80 12 vs 25; 0.85 8 vs 14; 0.90 4 vs 6; 0.95 1 vs 1 -> none reaches 2:1 -> crop retry REMOVED.
Deployed config now: possible burn, two gates, 0.75, no crop retry. Test (browser):
  close-ups: named 172/331 (52.0%, 46.6-57.3), right 154/172 (89.5%, 84.1-93.3, merged burn scoring),
  non-wounds labelled 48/450 (10.7%, 8.1-13.9); 3rd-degree (40): possible_burn 19, another injury 1, unknown 20.
  Like-for-like: the old config scored the same merged way was right 141/160 (88.1%); strict 119/160 (74%).
  Held-out healthy 37/401; RIT healthy feet 158/1,613; chronic 115/1,231; burns 35/67; everyday 19 right / 1 wrong.
Server verified on every photo (verify_backend_v4c.py): MATCH. Production wound-analyzer-vercel-axnmf5u04;
public alias confirmed (a photo answered only via the crop now returns unknown). Code: production_two_gates_075_nocrop/.
Rollback: wound-analyzer-vercel-jop7ssu2o (with crop), imjnkbxko (one gate, 0.60), qpgrxqjqx (before 09-27).

## MRC App session confirmation (2026-09-27)
Reproduced the close-up figures exactly by calling the deployed _classify(): 172/331 named, 154/172 right
(merged), 48/450 non-wounds, 3rd-degree 19 possible_burn / 20 unknown / 1 another injury.
Its own arm's-length set (it pads non-wound photos as well as wounds), deployed config: named 60/331
(18.1%, 14.4-22.6), right 49/60 (81.7%), non-wounds labelled 12/450 (2.7%). The public model card
(wound-analyzer-vercel.vercel.app) now quotes these, states that accuracy counts all burns as one type
and is not comparable with the old 74% (old config merged: 88.1%), and drops strict scoring.

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

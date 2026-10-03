# v6 result - flags, faces as "not a wound", retrain (2026-10-03)

Rules: PLAN.md (commits aea027e + addendum, fixed before results). Nothing deployed. Numbers: eval6.json.
Live baseline reproduces MRC App's figures exactly: test close-ups 153 right of 164 named, 44/450 non-wounds;
arm's length (MRC reconstruction) 47/331 named = 14.2%, 13/450 non-wounds = 2.9%.

## Flags (labels and files unchanged) - flags.csv
870 photos reviewed by eye (Claude, not a clinician): every burn_1st_degree photo (train/val/test) and every
val/test wound photo. **32 show no visible injury** (train 15, val 8, test 9): stock faces, arms, hands under a
tap, a clinic scene labelled 3rd degree, a frame with no cut in it. 5 of the 32 were already in
v2/flagged_for_review.csv (incl. burn_1st_degree_kg2_693) - "found again"; 27 are new. Sunburned / red faces
were not flagged. Other training classes were not reviewed.
**Effect on published figures, now measured:** the flagged test photos add 2 "right" answers to the live config
(153/164 = 93.3% vs 151/162 = 93.2% cleaned) and 3 to the pre-2026-09-27 config (kg2_693 included).
The optimism is real but small: about 0.1-0.3 points.

## New "not a wound" photos
archive (8) Skin v2 faces: 9,463 extracted; 6,321 dropped as resembling an evaluation photo (close-up faces look
alike to the 0.75 similarity check), 2,116 near-duplicates; **821 train / 205 held out** (cap of 2,000 not reached).

## Retraining (deployed recipe, seed 1)
N6 = flags excluded + 1,609 v4 healthy + 821 faces; N6k = same keeping the flagged photos (control).
Chosen by the fixed rule (1 of 12 eligible): mean(N6, deployed, healthy), threshold 0.70.

| | Live | Candidate |
|---|---|---|
| Verdict pool right (cleaned test + RIT) | 204 | 202 (CI -11..+7) |
| Verdict pool false alarms (cleaned test non-wounds + 1,008 normal skin + RIT) | 392 | 368 (CI -69..+12) |
| Wrong injury names | 12 | 13 |
| Cleaned test close-ups: named / right / non-wounds | 162 / 151 / 44 | 160 / 148 / 48 |
| Arm's length (MRC reconstruction, cleaned): named / right / non-wounds | 45 / 41 / 13 | 39 / 34 / 17 |
| Faces labelled: v4 faces (864) / archive-8 held-out faces (205) | 70 / 26 | 48 / 10 |
| RIT feet / chronic / burns | 104 / 123 / 36 | 101 / 126 / 36 |

**Verdict: NOT confirmed.** Faces improve, but correct answers do not rise, overall false alarms do not fall
measurably, and arm's length gets worse. Nothing is recommended for deployment.
Control (N6k vs N6): excluding the 15 flagged training photos made no consistent difference.

## What this round shows
Adding "not a wound" faces reliably cuts face false alarms (here 70 -> 48 and 26 -> 10) but no variant has
produced MORE correct wound identifications. That half of the objective is limited by the wound photos
themselves (about 1,200 training photos across 6 classes, many stock/collage images), not by the not-a-wound
side. More correct answers most plausibly needs more and better-labelled wound photos (e.g. EBIS for burns,
pending) - which this round cannot supply.

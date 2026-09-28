# Wound analyzer v4 - more answers, fewer healthy-skin false alarms (plan fixed 2026-09-27)

Vihaan's direction (2026-09-27): more answers, accepting more wrong answers; general burn advice
from the literature (done separately: mrcmiracle.github.io PR #2); new healthy-skin datasets.

## Evidence so far (more_answers.py, deployed models, browser input, production crop retry)
The "possible burn" output at 0.60 gives more answers and fewer wrong injury names on test (224
answered / 197 right / 27 wrong vs 160 / 119 / 41 today) but labels 560 of 1,613 RIT healthy feet
(today 183) and 120 of 450 other test non-wounds (today 61). Lower thresholds add answers and far
more false alarms. The healthy-skin false alarms are the cost to remove.

## Data (new downloads, read from the zips, never modified)
Healthy skin / not-a-wound photos from sources OTHER than the Lower Limb feet used for testing:
- ghost.v2i.folder.zip: 94 "normal skin" photos, CC BY 4.0 (Roboflow Universe imagesnormalskin/ghost-1iput)
- archive (10).zip Oily-Dry-Skin-Types: normal / oily / dry skin, 3,153 photos (faces; licence not stated)
- archive (9).zip SkinDisease Unknown_Normal: 1,840 photos (skin AND unrelated objects; licence not stated)
Not used in this round (skin conditions, not healthy skin; kept for a possible later round):
archive (8).zip cosmetic issues (9,770), PAD-UFES-20 lesions (2,298), Fitzpatrick17k (no images in zip,
CC BY-NC-SA); archive (9) disease classes.
Cleaning (automatic, no human review): drop exact / near duplicates (difference hash <= 6 with mirror,
or 12-view similarity >= 0.75) of any RIT, data/val or data/test photo, and of each other (one kept per
duplicate group). Licence status recorded per photo in the manifest.
Split per new source, by duplicate group, seed 0: 80% training, 20% held out (in-source test).

## Experiment G4 (gate retrained with the new negatives)
G4_new = gate recipe (experiment9: MobileNetV2 1.4, label smoothing 0.1, early stop data/val) on
data/train + the new training photos as out_of_scope, seed 1, weights saved.
G4_ctrl = same recipe on data/train only, seed 1 (separates the new data from retraining noise).
Pass (deployed classifier, gate threshold 0.5, production crop retry, 6-category output), all of:
- RIT healthy feet labelled <= half of both the shipped gate's (183) and G4_ctrl's;
- val correct wound answers lost vs the shipped gate <= 5 (browser AND raw);
- RIT everyday injuries (48) answered: at most 2 fewer than with the shipped gate;
- val non-wound labelled <= 16/94; 3rd-degree shown another injury (val) not more than shipped.
The RIT feet are a different source from all training photos (checked by the cleaning step), so this
is the out-of-source healthy-skin test the v3 round could not run. Also reported: the new sources'
held-out 20%, RIT chronic wounds, RIT healthy close-ups, test.
Budget: 2 training runs (about 1.5 h on this Mac). If G4_new passes, seed 2 is run and must pass too;
the candidate is seed 1.

## Operating point for "more answers" (chosen on val only, after G4)
Output "possible burn" (burn probabilities summed), candidate gate. Threshold = the lowest of
0.60 / 0.55 / 0.50 / 0.45 / 0.40 at which val non-wound photos labelled stay <= 13/94 (today's
production). If G4 fails, the same rule is applied with the shipped gate. Test and RIT are reported,
never used to choose. A deployment needs Vihaan's OK on the final table, a sourced "possible burn"
text live on the site (PR #2), and the MRC App session for the Vercel backend.

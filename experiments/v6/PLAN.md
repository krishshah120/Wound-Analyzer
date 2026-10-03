# v6 plan - label flags, faces as "not a wound", retrain (written 2026-10-03, before any v6 result)

Assigned by the MRC App session (production owner) on Vihaan's behalf: clean the burn_1st_degree problem, add
ordinary faces / everyday photos to "not a wound", retrain, measure against a cleaned test split. No deploy.
Live baseline (deployed by MRC App as g0fyqeygp): classifier = mean(deployed classifier, healthy-skin model),
possible burn = summed burn probability, both gates veto at 0.5, threshold 0.70, no crop.
Objective (Vihaan): MORE correct wound identifications AND fewer false alarms.

## Labels: flag, never relabel
Vihaan's standing rules: "do not automatically relabel burn severity based on your own visual judgment";
"Preserve the source files and labels; test excluding". So no label or file is changed. Instead:
- Flag U ("uninjured") = the photo shows no visible injury of any kind: no redness beyond normal skin tone, no
  blister, peeling, wound, bruise, burn mark or scar - it looks like ordinary uninjured skin or face.
  A red / sunburned face is a first-degree burn and is NOT flagged (injuries occur on faces; flagging every face
  would teach "face = not a wound").
- Reviewed by eye (Claude, not a clinician) on contact sheets: burn_1st_degree in train / val / test, and every
  wound photo in val and test. Other training classes are not reviewed (stated as a limitation).
- flags.csv records file, split, label, flag, reason. Flags are for qualified review; they are not ground truth.
- "Cleaned" val/test = the split with U-flagged photos excluded (not moved to another class). Results are always
  reported on BOTH the original and the cleaned splits.

## New "not a wound" photos
archive (8).zip "Skin Issues Dataset/Skin v2" (faces and facial skin with acne, dark spots, wrinkles, blackheads,
pores; licence not stated). Same cleaning as v4/prep_negatives.py: drop near-duplicates (difference hash <= 6 with
mirror, or 12-view similarity >= 0.75) of any RIT / data/val / data/test / v4 held-out / v4 independent photo and of
each other; split by duplicate group, seed 0, 80% train / 20% held out. Training part capped at 2,000 (random,
seed 0) to keep "not a wound" from swamping the wound classes. No new everyday-object source exists on disk;
everyday objects stay limited to the v4 SkinDisease Unknown_Normal photos (stated as a gap).

## Training (deployed recipe: experiment9.py --alpha 1.4 --label-smoothing 0.1 --val-dir v2/b84_val, seed 1)
- N6  = v2/b84_train minus U-flagged burn_1st photos + v4 healthy train (1,609) + archive-8 train as out_of_scope.
- N6k = the same but keeping the U-flagged photos (control: separates the flag exclusion from the new faces).
Candidates (both live gates at 0.5, no crop, summed burn): N6 alone, mean(N6, healthy-skin model), mean(N6,
deployed, healthy-skin model); thresholds 0.60 / 0.65 / 0.70 / 0.75. N6k reported, not selectable.

## Selection (dev only) and verdict
Dev = CLEANED val close-ups + CLEANED val arm's length (v4 framed_views) + 401 v4 held-out healthy + archive-8
held-out faces. right / wrong / false alarms as in v5. Eligible: right > live AND false < live AND wrong <= live.
Choose fewest false alarms; ties -> most right -> higher threshold.
Verdict pool (never used to choose): CLEANED test close-ups + 1,008 v5 independent normal-skin + RIT (duplicates once).
"Confirmed" only if right > live (strictly) AND false alarms < live with the paired 95% bootstrap CI of the
false-alarm difference below 0. Also reported: original (uncleaned) test, test arm's length, per-population,
and the MRC App arm's-length reconstruction if its builder is provided.

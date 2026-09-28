# Round I selection rule (fixed 2026-09-13, before any round I result was compared)

Adopt H4 (EfficientNetV2-B0, label smoothing 0.1) over G5 (MobileNetV2 1.4, label smoothing 0.1)
only if, as means over 3 seeds on VALIDATION:
1. balanced accuracy is higher than G5's;
2. out-of-scope photos confidently labelled rise by no more than 2 percentage points;
3. 3rd degree burns shown a wrong wound label do not increase.

Definition fix, recorded 2026-09-14 after MRC App could not reproduce a count: "shown a wrong
wound label" = evaluate_model's burn_3rd_app_says_other_wound, which already includes
"1st degree". Rounds G/H added burn_3rd_app_says_1st on top (double count). Round I uses
the correct definition. The choice was made for correctness; per-run summary lines for 4 runs
had been printed but the rule had not been applied to them.

# Replace the shipped model? (fixed 2026-09-14, BEFORE scoring the shipped model on the new validation split)

Round I result: KEEP G5 recipe (H4 failed check 2: val out-of-scope +3.55 pp).

Candidate: G5 recipe retrained with the new out-of-scope data. Compared as the mean of runs I_G5_s1..s3
against the shipped model file (b84b7b4) on the CURRENT validation split (the wound photos are
unchanged; 94 out-of-scope photos, none of them in the shipped model's training data). Replace only if:
1. out-of-scope photos confidently labelled drop by at least 5 percentage points;
2. balanced accuracy (evaluate_model's; a wound photo rejected as out of scope counts as wrong)
   falls by no more than 0.02;
3. accuracy when the app names a wound is not lower;
4. 3rd degree burns shown a wrong wound label (burn_3rd_app_says_other_wound) do not increase.
The fraction of wound photos answered is reported, not gated. Disclosure: test-split summary
lines for I_G5 had already been printed, and the shipped model's test numbers were known.
If the candidate passes, the final model is train_model.py (seed 42) on this split, and it is
reported on test once.

# 3rd degree urgent rule on the shipped model (criterion declared 2026-09-13)
VAL: t=0.4 meets the criterion (3rd urgent 17->23 of 31; non-3rd urgent 3.2%->6.1%). TEST at t=0.4:
3rd urgent 17->24 of 40; non-3rd urgent 5.8%->11.9% (out-of-scope 32->72 of 450), about twice the
val cost. Not deployed: sent to Vihaan to decide (safety advice, test contradicts val's cost budget).

# Round J: cap the out-of-scope training photos (fixed 2026-09-14, before any round J run)
Hypothesis: the 3,959 out-of-scope training photos (vs ~1,450 wound) make the model over-reject
wounds. Candidates: G5 recipe, out-of-scope TRAIN photos capped at 1,000 and 2,000, sampled once
with seed 42, evenly across photo groups (val/test unchanged). 3 seeds each.
A candidate "passes" with the same four checks as the replacement rule above (vs the shipped model
on the current val split, candidate = mean of 3 seeds). If both pass, choose the higher mean val
balanced accuracy. If neither passes, keep the shipped model.
Decision (Vihaan, 2026-09-14): wait. Re-measure the same rule (same criterion) on the next adopted model.

# Round J result (2026-09-14): neither cap passes -> keep shipped model.
cap1000 val: bal 0.629, ood 11.7%, acc_ans 84.3%, b3 wrong 1.00 (fails bal, acc_ans).
cap2000 val: bal 0.620, ood 11.4%, acc_ans 83.7%, b3 wrong 1.67 (fails bal, acc_ans, b3).
Wound rejection stayed ~10.5% with the cap, so the count imbalance is not the main cause.

# Gate simulation (fixed 2026-09-14, before computing any gate number)
Answer = shipped model's decide(), except "unknown" whenever a new-data model (I_G5 or J run, each
seed separately, from saved probabilities) gives out_of_scope probability >= tau.
tau is chosen on VAL per gate model: the lowest tau in {0.5, 0.6, 0.7, 0.8, 0.9, 0.95} that turns at
most 5 of the 251 val wound photos from a correct shipped answer into "unknown".
Then the same four checks vs the shipped model on val (gate = mean over its 3 seeds).
Exploratory only: shipping it would need two models on the site, which Vihaan declined once
before (the ensemble), so a pass goes to him as a question, not a deploy.
Definition note (before any gated number was computed): the gate never changes the most likely class,
so check 2 uses app-level balanced accuracy (shown label must equal truth; "unknown" counts wrong)
for both the shipped model (val 0.4294) and the gated version, limit: no more than 0.02 lower.

# Gate simulation result (2026-09-14): all three gate models PASS on val (tau 0.5 each).
Best: I_G5 as gate. VAL: ood 29.8% -> 17.7%, app bal 0.4294 -> 0.4224, acc_ans 84.6% -> 85.6%, b3 wrong 1 -> 1.
TEST (reported once, I_G5 gate mean of 3 seeds): ood 104/450 -> 59-74/450 (15.0%), app bal 0.379 -> 0.368,
acc_ans 75.0% -> 75.3%, 3rd degree 17 correct / 3 wrong unchanged.
Vihaan (2026-09-14): build and ship the gate.

# Final gate acceptance (fixed 2026-09-14, before training the final gate)
Final gate = train_model.py --gate (G5 recipe, all new out-of-scope data, seed 42), saved separately;
the wound classifier stays the shipped b84b7b4 model. Choose tau on VAL with the same rule (lowest of
0.5..0.95 losing <= 5 correct shipped val wound answers). It ships only if it passes the same four
checks on val vs the shipped model alone (app-level balanced accuracy for check 2). Test reported once.
If it fails, nothing ships and this is reported.

# Framing check (2026-09-19, reproducing the MRC App session's field finding)
Shipped classifier + gate, on val/test copies where each wound photo fills less of the frame
(surround = its own content, blurred; out-of-scope photos unchanged):
  fill 100% -> val 46.2% / test 48.0% of wound photos given an answer
  fill  70% -> val 31.1% / test 35.3%
  fill  50% -> val 19.5% / test 19.6%
  fill  35% -> val  6.8% / test  8.2%
Almost all of the loss is low confidence, not the gate. Cause: make_augmenter only ever zoomed IN
(RandomZoom (-0.15, 0.0)), so the model never saw a wound filling part of the frame.

# Round K: zoom-out augmentation (fixed 2026-09-19, BEFORE any round K run)
Candidates: G5 recipe + ZOOM_OUT 0.4 and 0.8, trained on the current data/train, 3 seeds each,
scored WITH the existing gate (unchanged) on validation. Baseline: the shipped classifier + gate
(val: 50%-fill named 19.5%, close-up named 46.2%, correct when named 85.3%, out-of-scope
labelled 17.0%, 3rd degree wrong label 1).
Replace the shipped classifier only if, as the mean of 3 seeds on VALIDATION:
1. named share on the 50%-fill val set is at least 15 points higher than 19.5%;
2. named share on the normal val set is no more than 5 points below 46.2%;
3. correct when named on the normal val set is no more than 3 points below 85.3%;
4. out-of-scope photos confidently labelled is not above 17.0%;
5. 3rd degree burns shown a wrong wound label is not above 1.
If both pass, take the higher 50%-fill named share. Then the final model is trained with
train_model.py (seed 42) and reported on test once, close-up and framed.
The framed sets are a synthetic proxy for standing further back: real phone photos would be
better evidence, and MRC App's server-side centre-crop retry stays until a model fixes this.

# Round K result (2026-09-19): BOTH FAIL -> keep the shipped classifier.
VAL (3 seeds, with gate), shipped -> zoom-out 0.4 / 0.8:
  50%-fill named 19.5% -> 23.8% / 25.8%  (rule needed >= 34.5%)
  close-up named 46.2% -> 47.0% / 53.7%
  correct when named 85.3% -> 83.5% / 82.0%  (0.8 misses the -3pp limit by 0.3)
  out-of-scope labelled 17.0% -> 7.1% / 8.5%; 3rd wrong label 1 -> 2.33 / 1.00
TEST, for the record: close-up named 48.0% -> 44.0% / 46.5%; correct when named 75.5% -> 79.4% / 81.8%;
50%-fill named 19.6% -> 23.4% / 22.5%. Zoom-out augmentation helps framing a little and helps
precision, but does not come close to fixing the collapse.

# Round L (fixed 2026-09-19, before any L run or measurement)
L1 - training: the same recipe with ZOOM_OUT 1.5 (photo may shrink to 40% of the frame), 3 seeds,
scored exactly like round K and judged by the SAME five checks against the shipped classifier.
L2 - serving, no retraining (an independent check of the MRC App session's crop retry): when the
answer would be "unknown", run the centred 70% crop of the same photo and accept that answer only
if its confidence is >= 0.80 and the gate does not block the crop. Adopt it in this repo's app.py
only if, on VALIDATION with the shipped models: 50%-fill named share rises by >= 8 points,
close-up named share does not fall, correct-when-named falls by no more than 1 point, and
out-of-scope photos confidently labelled rises by no more than 2 points. Test reported once after.

# Round L correction (2026-09-19, no results seen: all three L1 runs crashed at startup)
Keras RandomZoom rejects a factor above 1.0, so the declared ZOOM_OUT 1.5 could not run at all.
Replacements, same five checks as round K:
  L1b - ZOOM_OUT 1.0 (the library maximum, photo shrinks to about half the frame)
  L3  - new RandomFrameShrink layer, MIN_FRAME_FILL 0.3 (photo shrunk into as little as 30% of the
        frame, on a STRETCHED - not blurred - copy of itself, so training does not learn the
        evaluation proxy's blur). One shrink factor per batch, not per photo.

# Round L2 result (2026-09-19): crop retry NOT adopted in this repo's app.py.
VAL, shipped models, no retry -> with retry:
  fill 100%: named 46.2% -> 48.2%, correct when named 85.3% -> 84.3%, out-of-scope 17.0% -> 18.1%
  fill  70%: named 31.1% -> 37.5%, correct when named 80.8% -> 79.8%
  fill  50%: named 19.5% -> 20.7%  (rule needed >= +8 points; got +1.2)
It passes the other three checks and fails the one that matters. IMPORTANT CAVEAT: every photo in
data/ is 224x224, so the crop here discards resolution and is then upscaled; on the site the crop
runs on the full-size upload, which is why the MRC App session measured a much larger gain
(18.7% -> 30.5% named at arm's length). This measurement understates the production benefit, so it
is evidence about THIS repo's low-resolution test data, not a reason for MRC App to remove its
crop pass.

# Round L1b / L3 result (2026-09-20): BOTH FAIL -> keep the shipped classifier. Line closed.
VAL (3 seeds, with gate): 50%-fill named 24.4% (zoom-out 1.0) and 25.9% (shrink to 30% of frame),
against the 34.5% the rule required; correct when named 80.5% / 81.5% vs the shipped 85.3%.
Across four augmentation variants (zoom-out 0.4, 0.8, 1.0, frame-shrink 0.3) the 50%-fill number
sits at 23.8-25.9% - a ceiling of about +5 points - and every variant costs 4-5 points of
accuracy when answering. Augmentation alone does not fix scale brittleness on this data.
What remains: real photos taken at realistic distances (consent and storage to be settled by
Vihaan and the MRC), and the MRC App session's crop retry on the FULL-RESOLUTION upload, which
its production measurement shows working (18.7% -> 30.5% named at arm's length).

# Round M: find the injury, then crop to it (fixed 2026-09-25, BEFORE building anything)
New data (user download, 2026-09-25): a wound segmentation set (FUSeg 1,210 + WSNet 1,176 +
Medetec 374 images, 512x512, masks) and the Lower Limb and Feet Wound Image Dataset (Mendeley,
CC BY 4.0: 2,686 wound photos with masks, 2,757 normal feet). All are chronic/lower-limb wounds,
so they cannot teach the six injury classes - but their masks can teach WHERE an injury is.

Evaluation sets (new, built before any localizer exists): each val/test wound photo placed at
FULL resolution on a larger canvas so it fills 70% / 50% / 35% of the frame, at a random
(seeded) off-centre position, surround = blurred stretched copy of itself, JPEG q82. Unlike the
earlier framed sets, a crop of this canvas recovers the full-detail photo, as a crop of a real
full-resolution upload would. Out-of-scope photos are unchanged.

Retry rules compared, all with the shipped classifier + gate, all used ONLY when the normal
answer is "unknown", all accepting the retry's answer only at confidence >= 0.80 with the gate
not blocking it:
  none      - no retry
  centre    - centred 70% crop (the MRC App session's live rule)
  localize  - a small segmentation model marks the injury on the whole photo; crop a square
              around the marked area (box + 25% margin each side, at least 25% of the photo side)
Adopt "localize" in this repo's app.py only if, on VALIDATION:
  1. 50%-fill answered share is >= 5 points above "centre";
  2. 50%-fill correct-when-answered is no more than 2 points below "centre";
  3. close-up (normal val) answered share is not below "none", and close-up
     correct-when-answered is no more than 1 point below "none";
  4. out-of-scope photos confidently labelled is no more than 2 points above "centre".
Test reported once, after the decision. Localizer training images that resemble any
classification val/test photo (visual similarity >= 0.75 or near-identical hash) are removed
first, so the localizer has never seen a photo it will later be judged on.

# Round M result (2026-09-25): localize retry NOT adopted (fails check 1).
Localizer (U-Net, MobileNetV2 0.35, 3.9 MB) on its own held-out test (525 photos): finds a wound in
98.5%, box IoU 0.53 mean (>= 0.5 on 56%), marks 3 of 44 healthy feet.
VAL, shipped classifier + gate, none / centre / localize:
  close-up  answered 46.2 / 48.2 / 47.4 %, correct when answered 85.3 / 84.3 / 84.9 %
  70% fill  answered 35.1 / 37.5 / 38.2 %, correct 78.4 / 78.7 / 80.2 %
  50% fill  answered 23.9 / 25.5 / 26.7 %, correct 76.7 / 75.0 / 79.1 %   <- check 1 needed >= 30.5%
  35% fill  answered 14.7 / 16.3 / 19.5 %, correct 59.5 / 63.4 / 67.3 %
  out-of-scope labelled 17.0 / 18.1 / 17.0 %
Localize beats centre on every measure at every distance, but by +1.2 points at 50% fill, not +5.
Diagnosis (50% fill val, the 191 wound photos answered "unknown"):
  - even a PERFECT crop (exactly the original photo) reaches 0.80 confidence on only 14% (0.60 on
    43%): most of these photos are ones the classifier is unsure about as close-ups too;
  - the localizer, trained on chronic ulcers, marks bruises and open burns but nothing on thin cuts,
    sunburn or pink 1st degree burns. My first guess - that it marked the blurred surround - was
    wrong: it never did, on the examples checked by eye.
The binding constraint is the classifier's own confidence on acute-injury photos, not framing
alone. Test not looked at (nothing was adopted).

# New-data audit (2026-09-26): Roboflow wound v1 (CC BY 4.0) and Shubham Baid skin burn set
Genuinely new photos (not a hash/visual copy, >= 0.80, of any existing wound or out-of-scope photo,
and not a repeat within the batch): Baid 1st 27 / 2nd 16 / 3rd 0 of 1,136 (the Fares burn set is
largely the same images); Roboflow abrasion 14, bruise 92, cut 28, ungraded burn 137, blister 23,
normal skin 60. Baid degree labels are visibly unreliable (e.g. "2nd degree" photos of rashes, one
captioned "Contact Dermatitis: Poison Ivy"), so Baid is NOT used for classification.

# Round N: injury localizer retrained with everyday-injury boxes (fixed 2026-09-26, before building)
Training data = round M's localizer data + the Roboflow and Baid photos as box masks (each box
filled), EXCLUDING: any photo with visual similarity >= 0.75 or a near-identical hash to ANY
classification val/test photo (the round M rule); Baid photos resembling an out-of-scope photo
(>= 0.80, the mislabel signal); Roboflow "no abnormality" photos are used as healthy negatives
(empty mask). New photos are split 80/10/10 by near-duplicate group.
Judged EXACTLY as round M: same evaluation sets, same retry (box + 25% margin, min 25% side,
accept >= 0.80, gate must not block), same four checks against the centre crop on VALIDATION.
Test reported once, only if adopted.

# Round N result (2026-09-26): NOT adopted - fails check 1 by 0.6 points.
Localizer N held-out: finds an injury in 99.5%, box IoU 0.505, marks healthy skin 12.8%.
VAL (none / centre / localize-M / localize-N):
  50% fill answered 23.9 / 25.5 / 26.7 / 29.9 %   (needed >= 30.5)
  50% fill correct  76.7 / 75.0 / 79.1 / 80.0 %
  35% fill answered 14.7 / 16.3 / 19.5 / 22.3 %
  close-up answered 46.2 / 48.2 / 47.4 / 46.6 %, correct 85.3 / 84.3 / 84.9 / 84.6 %
  out-of-scope labelled 17.0 / 18.1 / 17.0 / 17.0 %
Everyday-injury boxes tripled localize's gain over centre crop (+1.2 -> +4.4 at 50% fill) but it
still misses the pre-declared +5. Rule not relaxed after the fact. Test not looked at.

# Round O: stronger frozen backbones (fixed 2026-09-26, BEFORE extracting any feature)
Why: rounds M/N showed the binding limit is the classifier's own confidence (it names a wound for
~46% of close-up val photos). Every earlier round used MobileNetV2 or EfficientNetV2-B0.
Candidates: frozen ImageNet ConvNeXt-Tiny and EfficientNetV2-S (plus MobileNetV2 1.4 frozen as a
control), each + a small head (Dense 256 ReLU, Dropout 0.3, softmax 7; label smoothing 0.1; class
weights with 3rd degree x2; Adam 1e-3; early stopping on val loss), trained on features of
data/train (original + 3 augmented views per photo), 3 seeds each. Scored with the SHIPPED gate
unchanged, app-level (model.decide), on VALIDATION. A candidate beats the shipped classifier only if,
as the mean of 3 seeds:
  1. close-up answered share >= 46.2% + 5 points (51.2%);
  2. close-up correct-when-answered >= 85.3% - 1 point (84.3%);
  3. out-of-scope photos confidently labelled (with gate) <= 17.0%;
  4. 3rd degree burns shown a wrong wound label <= 1.
Also reported, not gated: 50%-fill (high-res, off-centre) answered share. If more than one passes,
the highest close-up answered share wins. Test is looked at once, after the decision.

# Round O result (2026-09-26): ALL FAIL -> keep shipped. (Scoring reproduced the shipped model exactly.)
VAL, 3-seed means, frozen backbone + head, with the shipped gate:
  convnext_tiny answered 36.1%, correct 79.8%, ood 6.7/94, b3 wrong 1.33
  effv2s        answered 35.2%, correct 80.1%, ood 9.7/94, b3 wrong 3.67
  mnv2_14 (ctl) answered 36.8%, correct 77.8%, ood 6.3/94, b3 wrong 2.00
Frozen ConvNeXt ~= frozen MobileNetV2: the backbone is not the bottleneck in this setup. Every model
trained on the current data is conservative (fewer answers, fewer out-of-scope labels).
Diagnostic (same features): 6-class heads (no out_of_scope class, gate alone rejects non-wounds)
answer more (convnext 46.9%, mnv2 45.7%) but are less accurate (74.9%, 74.7%) and label more
non-wounds (20.0, 15.0 /94) than 7-class heads: the class suppresses mostly wrong answers. Not a lever.

# Round P: do the new Roboflow photos help naming injuries? (fixed 2026-09-26, before any P run)
Recreates the shipped classifier's training conditions: G5 recipe (MobileNetV2 1.4, label smoothing
0.1, fine-tuned), current wound train photos (+ the 149 existing extras), out-of-scope train photos
from the ibrahim source ONLY (the shipped model never had the four later downloads).
  P0 - that, as a control for the data effect
  P1 - P0 + Roboflow photos labelled abrasion / bruise / cut that the audit found new, TRAIN ONLY,
       and only those with similarity < 0.75 and no near-identical hash to any val/test photo
3 seeds each, scored with the shipped gate on VALIDATION. P1 replaces the shipped classifier only
if, as the mean of 3 seeds: close-up answered >= 49.2% (shipped 46.2% + 3), correct when answered
>= 84.3%, out-of-scope labelled <= 16/94, 3rd degree burns shown a wrong label <= 1.

# Round P result (2026-09-26): FAIL -> keep shipped.
VAL 3-seed means (answered / correct when answered / out-of-scope labelled / 3rd wrong label):
  shipped (one run)          46.2% / 85.3% / 16/94 / 1
  P0 shipped conditions      36.4% / 77.4% /  8.0  / 1.33   (per seed answered 34.7, 39.4, 35.1)
  P1 + 105 Roboflow photos   37.2% / 80.4% / 11.0  / 1.67   (per seed answered 42.2, 19.1, 50.2)
The Roboflow photos help accuracy when answering (+3.0 points over P0) but very noisily. The
recipe re-run under the shipped model's own conditions averages well below the shipped model:
the shipped classifier is an above-average single run, so beating it on 3-seed means is a high bar.

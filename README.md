# Wound-Analyzer

Classifies a photo of a wound into one of six classes: `abrasion`, `bruise`,
`cut`, `burn_1st_degree`, `burn_2nd_degree`, `burn_3rd_degree`. The web app
(`app.py`) returns the label plus general first-aid pointers, and answers
`unknown` when the photo looks like none of those classes or the model's
confidence is below `CONFIDENCE_THRESHOLD` (0.60, in `src/model.py`).

**This is not a diagnostic tool.** On wound photos it has never seen, the
current model's most likely class is right about 6 times in 10, and it
answers `unknown` for about half of them. It still gives a confident wound
label to about 1 in 7 photos that are not a wound at all (details below).
It needs a close-up: when the wound fills only half the frame, as in a photo
taken at arm's length, it answers for just 1 photo in 5. Anyone who might have
a serious injury, especially a burn, should seek medical care regardless of
what this app says.

## How a prediction is made

Two models look at every photo. Both are MobileNetV2 (width 1.4) with the
same seven outputs: the six wound classes plus `out_of_scope`.

- **The classifier** (`models/wound_model.keras`) names the wound. Its
  `out_of_scope` output was trained on photos of normal skin and chronic
  wounds (diabetic, pressure, surgical and venous) from one source.
- **The out-of-scope gate** (`models/out_of_scope_gate.keras`) was trained
  the same way on many more kinds of photo that are not one of the six wounds
  (rashes and other skin conditions, bites, normal skin; see
  [Extra data](#extra-data)). It is used only to say "not a wound".

`decide()` in `src/model.py` turns the outputs into the app's response:

- `best_guess` is the classifier's most likely **wound** class (never
  `out_of_scope`), and `confidence` is its probability.
- `label` is `unknown` if `out_of_scope` is the classifier's most likely
  output, `confidence` < 0.60, or the gate gives `out_of_scope` a probability
  of at least `GATE_THRESHOLD` (0.5); otherwise it is `best_guess`. The gate
  never changes `best_guess` or `confidence`.

So `/predict` returns the same fields and the same possible `label` values as
before: `{label, confidence, best_guess, tips, disclaimer}`. Because `unknown`
no longer means "low confidence" - the gate can return it at any confidence -
`templates/index.html` shows that number as the *closest guess*, not as the
confidence in an answer. The `unknown` tips cover both cases (an unclear photo,
or an injury the tool doesn't cover) and advise seeing a professional for
wounds that aren't healing or look infected.

## Current performance (honest, held-out test set)

Measured with `python src/evaluate_model.py` on `data/test`, loading images
the same way the app does, with the gate as the app runs it. Full numbers are
in `models/metrics_with_gate.json` (`models/metrics.json` is the classifier's
original evaluation on the older, smaller out-of-scope test set).

- 331 real wound photos. None shares a source photo with training or
  validation, and none is visually near-identical to one (similarity ≥ 0.80),
  including rotated, cropped, mirrored or re-watermarked copies.
- 450 held-out out-of-scope photos from 5 sources: chronic wounds, normal
  skin, insect bites and many skin conditions. None has an out-of-scope
  training photo at similarity ≥ 0.75, and none was in either model's
  training data.

**Wound photos**

| Metric | Classifier alone | **With gate (as the app runs)** |
|---|---|---|
| Most likely class correct (a photo rejected as out of scope counts as wrong) | 61.3% (95% CI 56.0%–66.4%) | 61.3% (the gate does not change it) |
| Balanced accuracy (mean per-class recall) | 63.1% | 63.1% |
| Answered (label is not `unknown`) | 164 of 331, 49.5% (44.2%–54.9%) | **159 of 331, 48.0%** (42.7%–53.4%) |
| Correct when answered | 123 of 164, 75.0% (67.9%–81.0%) | **120 of 159, 75.5%** (68.2%–81.5%) |

**Out-of-scope photos**

| Metric | Classifier alone | **With gate (as the app runs)** |
|---|---|---|
| Confidently given a wound label (instead of `unknown`) | 104 of 450, 23.1% (19.5%–27.2%) | **68 of 450, 15.1%** (12.1%–18.7%) |

The classifier is one training run (seed 42). Across six seeds of its recipe,
test accuracy was 59.5% ± 2.1% (run G5 below), so this run is somewhat above
average.

Per class, using the classifier's most likely output before the confidence
threshold and the gate (so `out_of_scope` recall and the precision of the
wound classes reflect the classifier alone):

| Class | Test photos | Recall | Recall 95% CI | Precision |
|---|---|---|---|---|
| abrasion | 41 | 56.1% | 41%–70% | 60.5% |
| bruise | 33 | 63.6% | 47%–78% | 26.9% |
| burn_1st_degree | 88 | 77.3% | 67%–85% | 33.2% |
| burn_2nd_degree | 83 | 31.3% | 22%–42% | 23.4% |
| burn_3rd_degree | 40 | 67.5% | 52%–80% | 20.1% |
| cut | 46 | 82.6% | 69%–91% | 40.4% |
| out_of_scope | 450 | 23.6% | 20%–28% | 87.6% |

Confusion matrix (rows = true class, columns = the classifier's most likely
output, before the confidence threshold and the gate):

| true \ predicted | abrasion | bruise | burn 1st | burn 2nd | burn 3rd | cut | out of scope |
|---|---|---|---|---|---|---|---|
| abrasion | **23** | 6 | 1 | 7 | 0 | 2 | 2 |
| bruise | 1 | **21** | 7 | 0 | 2 | 2 | 0 |
| burn_1st_degree | 0 | 4 | **68** | 11 | 2 | 2 | 1 |
| burn_2nd_degree | 4 | 1 | 18 | **26** | 21 | 7 | 6 |
| burn_3rd_degree | 2 | 3 | 2 | 1 | **27** | 1 | 4 |
| cut | 0 | 3 | 0 | 3 | 0 | **38** | 2 |
| out_of_scope | 8 | 40 | 109 | 63 | 82 | 42 | **106** |

Most out-of-scope photos are still most likely some wound class to the
classifier; the confidence threshold and the gate are what turn most of them
into `unknown`.

### Burn severity

The app routes `burn_3rd_degree` to "call emergency services", so the errors
that matter most are 3rd degree burns shown something milder. Of 40 test
3rd degree burns, the app would return:

- `burn_3rd_degree` for 17;
- **`burn_1st_degree` for 1**;
- a non-burn wound label for 2 (one `cut`, one `bruise`);
- `unknown` for 20 (4 of them because the photo looked out of scope). Those
  users see the `unknown` tips (see a professional), not "call emergency
  services".

The gate changes none of these 40 answers.

2nd degree burns remain the weakest class: 21 of 83 were most likely 3rd
degree and 18 were most likely 1st degree.

**A lower bar for possible 3rd degree burns was measured, not switched on.**
The rule: show `burn_3rd_degree` when the answer would be `unknown`, the most
likely wound is a 3rd degree burn with probability ≥ t, and the photo does not
look out of scope. The criterion was fixed before measuring: use the lowest t
in {0.2, 0.3, 0.4, 0.5} that gives urgent advice to at least 3 more validation
3rd degree burns while raising urgent advice on all other validation photos by
at most 3 percentage points. t = 0.4 met it on validation (3rd degree burns
given urgent advice 17 → 23 of 31; other photos 3.2% → 6.1%). On test it gave
urgent advice to 24 instead of 17 of 40 3rd degree burns, but raised it on
other photos from 5.8% to 11.9% (out-of-scope photos 32 → 72 of 450), about
twice the validation cost. It was not deployed; it will be re-measured on the
next model. (Measured on the classifier alone, before the gate existed.)

### Out-of-scope photos

Confidently given a wound label, by kind of photo (test):

| Test photos | Count | Classifier alone | With gate (as the app runs) |
|---|---|---|---|
| Chronic wounds (diabetic, pressure, surgical, venous) | 130 | 7 | 7 |
| Normal skin | 66 | 23 | 18 |
| Insect bites | 39 | 12 | 10 |
| Skin conditions (rashes, eczema, psoriasis, acne, moles, infections, …) | 215 | 62 | 33 |
| **All** | **450** | **104 (23.1%)** | **68 (15.1%)** |

The 68 wrong labels with the gate: `burn_3rd_degree` 26, `burn_1st_degree`
25, `burn_2nd_degree` 6, `cut` 6, `bruise` 4, `abrasion` 1. So a photo of a
rash can still be shown as a 3rd degree burn.

For comparison, on the older 138-photo out-of-scope test set (chronic wounds
and 1 normal-skin photo, all from one source), a model trained on the six
wound classes only (commit 4321434) confidently labelled 65.9%, and the
classifier 4.3%. That set was too narrow: the classifier had only ever seen
out-of-scope photos from that one source.

Limits of this result:

- **Validation has 94 out-of-scope photos**, so thresholds chosen on it are
  noisy. The 3rd degree rule above is an example of validation
  underestimating a cost.
- **Photos that are not skin at all** (objects, pets, documents) were not
  tested.
- Every source is web-scraped, and some groups (for example normal skin) come
  from one or two sources, so the models may partly recognise a source's
  photo style.

### Photos taken from further away

**This is the biggest weakness of the model, and the thing a reader can
actually control.** Every photo in `data/` is a tight crop of a wound. Someone
photographing their own arm holds the phone at arm's length, so the wound
fills a fraction of the frame - and the model mostly answers `unknown` for
those. Reported by the MRC App session from real use of the deployed tool, and
reproduced here with `python src/framing_check.py`, which shrinks each wound
photo into part of the frame (surround: a blurred copy of the photo itself,
re-encoded as JPEG like a browser upload) and leaves out-of-scope photos alone:

| Wound fills | Test photos answered | Correct when answered | Validation answered |
|---|---|---|---|
| 100% (as stored) | 159/331 (48.0%) | 75.5% | 116/251 (46.2%) |
| 70% | 117/331 (35.3%) | 70.9% | 78/251 (31.1%) |
| 50% | 65/331 (19.6%) | 64.6% | 49/251 (19.5%) |
| 35% | 27/331 (8.2%) | 59.3% | 17/251 (6.8%) |

Almost all of the loss is low confidence, not the out-of-scope class or the
gate: at 50% fill on test the gate blocks 7.6% of wound photos and the
classifier's own `out_of_scope` wins on 3.3%, while 232 of 331 simply fall
below the 0.60 threshold.

**Cause:** `make_augmenter` only ever zoomed *in* (`RandomZoom((-0.15, 0.0))`),
so the model was never shown a wound filling part of the frame.

**Four attempts to fix it in training all failed their rule** (each 3 seeds,
judged on validation against the shipped model, rule fixed before the runs:
the 50%-fill answered share had to rise by at least 15 points, from 19.5%):

| Change | 50%-fill answered (val) | Close-up answered | Correct when answered |
|---|---|---|---|
| shipped | 19.5% | 46.2% | 85.3% |
| `ZOOM_OUT` 0.4 | 23.8% | 47.0% | 83.5% |
| `ZOOM_OUT` 0.8 | 25.8% | 53.7% | 82.0% |
| `ZOOM_OUT` 1.0 | 24.4% | 47.7% | 80.5% |
| `MIN_FRAME_FILL` 0.3 (`RandomFrameShrink`) | 25.9% | 46.6% | 81.5% |

Zoom-out augmentation buys about 5 points of framing robustness and costs 4-5
points of accuracy when the app answers. `ZOOM_OUT` and `MIN_FRAME_FILL` are
left in `train_model.py` (both off) for whoever tries again. What is left to
try: **real photos taken at realistic distances** (which need consent and a
collection route, so they are not a thing this repository can simply scrape),
and cropping at serving time - the MRC site retries a centred 70% crop of the
full-resolution upload when the answer is `unknown` and accepts it only above
0.80 confidence, which it measured lifting arm's-length photos from 18.7% to
30.5% answered. The same retry measured on *this* repository's data gains only
1.2 points, because every photo here is 224x224, so cropping one discards
detail that a real upload still has.

**Finding the injury first, then cropping to it, was also tried and not
adopted.** A small segmentation model (U-Net on MobileNetV2 0.35, 3.9 MB) was
trained to mark injuries using the masks in two chronic-wound datasets - the
[Lower Limb and Feet Wound Image Dataset](https://data.mendeley.com/datasets/hsj38fwnvr/3)
(CC BY 4.0) and a segmentation set combining FUSeg (from
[uwm-bigdata/wound-segmentation](https://github.com/uwm-bigdata/wound-segmentation)),
WSNet and Medetec - after removing 804 photos that resembled any
classification validation or test photo. On its own held-out photos it found
an injury in 98.5% of them. Used as a retry when the answer is `unknown`, on
full-resolution validation photos with the wound off-centre and filling half
the frame, it beat a centre crop on every measure (26.7% vs 25.5% answered,
79.1% vs 75.0% correct when answered, fewer out-of-scope photos labelled) but
not by the 5 points required to justify a third model. Two reasons: it was
trained on chronic ulcers, so it marks bruises and open burns but not thin
cuts, sunburn or pink first degree burns; and even a perfect crop reaches the
0.80 confidence the retry needs on only 14% of these photos, because the
classifier is unsure about most of them as close-ups too. The binding
constraint is the classifier's own confidence on these injuries, which more
and better photos of *these* injuries would address.

Retrained with boxes around everyday injuries from the
[Roboflow wound set](https://app.roboflow.com) (CC BY 4.0) and the Shubham Baid
burn set - after removing 1,287 photos resembling a classification
validation or test photo - the injury finder did better: at half-frame 29.9%
answered and 80.0% correct, against 25.5% and 75.0% for the centre crop. That
still misses the 5-point margin set in advance (by 0.6 points), so it was not
adopted. An audit of those two downloads also found that the Baid burn set is
almost entirely photos already in this project (43 new of 1,136) with
unreliable degree labels, and the Roboflow set adds 134 new labelled
abrasion/bruise/cut photos.

### Real uploads: resized photos

Every photo in `data/` is already 224×224 (made with PIL's default bicubic
resize), so test numbers never exercise the resize that every real upload
goes through: the MRC site shrinks photos in the browser (longest edge 1,024,
JPEG quality 0.82), and the app then resizes to 224×224 with nearest-neighbour.

To check this, the original file behind each test photo was recovered: all
450 out-of-scope photos from the download manifests, and 226 of 331 wound
photos by matching them pixel for pixel against the raw downloads (mean
difference 1–5 of 255, with a clear gap to the next-best match). The rest come
from the original download, which is not available here. The matched wound
photos are mostly burns (88 1st, 83 2nd, 40 3rd degree, but only 3 abrasions,
7 bruises and 5 cuts). The browser shrink was simulated with PIL.

| On those 676 photos | Stored 224×224 files | Real upload path | Real upload path, bicubic instead of nearest |
|---|---|---|---|
| Wound photos shown the right label (classifier alone) | 35.0% (29.0–41.4) | 35.8% (29.9–42.3) | 37.2% (31.1–43.6) |
| Out-of-scope photos confidently labelled (classifier alone) | 104 / 450 | 100 / 450 | 107 / 450 |
| Wound photos shown the right label (with gate) | 33.6% (27.8–40.0) | 35.4% (29.5–41.8) | 35.8% (29.9–42.3) |
| Out-of-scope photos confidently labelled (with gate) | 68 / 450 | 63 / 450 | 58 / 450 |
| Answers that change vs the stored files (with gate) | – | 77 of 676 | 71 of 676 |

Overall numbers hold for resized uploads, but individual answers are less
stable than the test set suggests: resampling alone changes about 1 answer in
9 with the gate (1 in 7 without). Bicubic was not adopted: its gain is within
the noise and was not a pre-declared test. (Here "shown the right label"
counts `unknown` as wrong, so it is lower than "correct when answered".)

## What changed, and why the number moved

The original model reported **76.25%** test accuracy (75.7% when images are
loaded the way the app loads them). That number was inflated by train/test
leakage, found and fixed in two rounds. The honest number then fell to about
50%, and later changes (an out-of-scope class, a wider network, label
smoothing, a stricter out-of-scope split) brought it to 61.3%.

**Round 1: augmented copies and re-uploads.**

1. `collate_data.py` created augmented copies (`<photo>_aug<N>.jpg`) *before*
   the split, and `train_model.py` then split individual files at random.
   **248 of the 467 distinct source photos in test (53.1%) also appeared in
   train**, and 293 of 522 test files (56.1%) were copies of a training photo.
2. On the leaked test files the original model scored 88.7%. On test files
   with an unseen source photo it scored **60.3%**.
3. The burn dataset contains the same photo re-uploaded under different
   filenames (158 identical-hash pairs), sometimes with different degree
   labels.
4. Early stopping monitored the test set.

The first fix grouped photos by source filename plus perceptual hash, split
whole groups into train/val/test, augmented only training batches, and
early-stopped on validation. That model scored 59.8%.

**Round 2: rotated, cropped and re-watermarked copies.** A perceptual hash
does not recognise a photo that has been rotated, cropped, mirrored or
re-watermarked, and this data contains many. Comparing images by visual
similarity across rotated, cropped and mirrored views found **179 train/test
image pairs with similarity ≥ 0.80** in that split. Checked by eye, pairs at
≥ 0.85 were almost all the same photo, and pairs at 0.80–0.85 were often the
same photo. So the 59.8% was still inflated; the honest number was about 51%.

The current split (`src/train_model.py`):

- uses only the real photos listed in `data/wound_dataset/manifest.csv`;
- merges photos into one group when their hashes nearly match or their visual
  similarity is ≥ 0.85 (1,643 photos → 1,342 groups);
- splits whole groups into train / val / test (≈ 65 / 15 / 20 %, stratified by
  class);
- then moves any val/test group that still has a match ≥ 0.80 in another split
  into that split, and tops test and val back up with groups that have no such
  match (merging at 0.80 directly would chain look-alike photos, such as
  sunburnt skin, into groups of hundreds);
- does the same for out-of-scope photos with a stricter 0.75, because the
  chronic-wound photos include series of the same wound just below 0.80 (at
  0.80 these flattered the out-of-scope test result: the same recipe
  confidently labelled 9.4% of out-of-scope test photos on the lenient split
  and 22.6% on the stricter one, averaged over seeds);
- refuses to continue if any group crosses splits or any cross-split pair
  reaches its threshold, and writes `data/split_manifest.csv`.

Independent checks of the current split: 0 shared source filenames, 0
near-identical hashes (mirrored or not), and 0 cross-split wound-photo pairs
at similarity ≥ 0.80.

The similarity thresholds were set by looking at sample pairs, not derived
from ground truth: a few duplicates may remain below them, and the test
photos are still web-scraped, so real user photos may score differently.

**Evaluation loads images the way the app does.** TensorFlow's dataset
loader decodes JPEGs slightly differently from `tf.keras.utils.load_img`
(used by `predict()`), which changed the answer on 12 of 485 test photos.
`evaluate_model.py` uses the app's loader. Experiment rounds E and F below
were measured with the dataset loader; round G with the app's loader.

## Extra data

Three more Kaggle datasets were downloaded and cleaned with
`src/collate_extra_data.py`:

| Dataset | Licence | Used for |
|---|---|---|
| [faresabbasai2022/burn-dataset](https://www.kaggle.com/datasets/faresabbasai2022/burn-dataset) | Apache 2.0 | burn degree candidates |
| [ibrahimfateen/wound-classification](https://www.kaggle.com/datasets/ibrahimfateen/wound-classification) | unknown | abrasion/bruise/cut candidates, out-of-scope photos |
| [yasinpratomo/wound-dataset](https://www.kaggle.com/datasets/yasinpratomo/wound-dataset) | unknown | abrasion/bruise/cut candidates |

**Extra wound photos.** Most of that data is the same web-scraped photos
already here. `train_model.py` adds an extra image **to train only**, and only
if it has no near-identical hash and visual similarity < 0.75 to every
val/test photo and < 0.80 to every train photo. Of 2,717 candidates that
passed quality checks, 1,517 were skipped as near-copies of a val/test photo,
963 as near-copies of a train photo, and 88 as repeats of each other. **149
were added**: 89 3rd degree burns, 24 2nd degree, 23 bruises, 5 cuts, 4
abrasions, 4 1st degree. On a visual spot check, some extra 3rd degree burns
look like staged first-aid training makeup, and some extra 2nd degree burns
look like abrasions. Their labels were not changed.

**Out-of-scope photos.** Four more downloads supply photos that are not one
of the six wounds:

| Dataset | Licence | Used for |
|---|---|---|
| [lysaapriani/skin-disease-and-normal-skin-dataset](https://www.kaggle.com/datasets/lysaapriani/skin-disease-and-normal-skin-dataset) | unknown | normal skin, dermatitis |
| [moonfallidk/bug-bite-images](https://www.kaggle.com/datasets/moonfallidk/bug-bite-images) | Apache 2.0 | insect bites, bite-free skin |
| [ismailpromus/skin-diseases-image-dataset](https://www.kaggle.com/datasets/ismailpromus/skin-diseases-image-dataset) | © original authors | skin conditions |
| [shubhamgoel27/dermnet](https://www.kaggle.com/datasets/shubhamgoel27/dermnet) | DermNet images, copyrighted | skin conditions |

`collate_extra_data.py` reads them straight from the zip files, takes at most
100 photos per folder (normal skin and bite-free skin are kept whole), and
skips photos that are, or look like, an in-scope injury: filenames mentioning
burns, scalds, blisters, bruises, purpura, haematomas, wounds, lacerations or
abrasions, and DermNet's whole bullous (blistering) disease folder. That
excluded 625 of 19,559 DermNet images, 7 of 27,153 skin-disease images and 2
of 1,310 bite images. Together with the 1,073 normal-skin and chronic-wound
photos from ibrahimfateen, there are 5,758 out-of-scope photos in 51 groups.

They are grouped and split like the wound photos (cross-split threshold
0.75). Any that resembles any wound photo (hash or similarity ≥ 0.80) is
skipped, because some of these downloads reuse photos labelled as wounds here:
1,255 skipped. Result: 3,959 train / 94 val / 450 test. The wound split is
unchanged.

The classifier (`models/wound_model.keras`, commit b84b7b4) was trained
before these four downloads were added, on the ibrahimfateen photos only
(split then 861 train / 33 val / 138 test). The gate was trained on all of
them. None of the current 544 out-of-scope val/test photos was in the
classifier's training data; 22 of the test photos were in its validation set.

These images are gitignored (several sources have no stated licence or are
copyrighted), so
`data/*/out_of_scope/` and the `extra_*` files are not in the repository. To
rebuild them, unzip each download into `data/raw_downloads/` as described at
the top of `src/collate_extra_data.py`, then run `train_model.py`. To train
without them, set `USE_EXTRA_TRAINING_DATA = False` and/or
`USE_OUT_OF_SCOPE_CLASS = False` in `src/train_model.py`.

## Experiments

Configurations were **chosen on the validation split**; test is shown for
every run but was not used to pick one. Each row is the mean of 3 training
seeds (± standard deviation) unless stated.

**Rounds I and J, and the gate: adding many more kinds of out-of-scope photo**
(current split: the same wound photos; validation 94 and test 450
out-of-scope photos; app image loader). "3rd wrong label" is the number of
validation 3rd degree burns shown any other wound label (including "1st
degree", counted once). "Rejected" is the share of validation wound photos
whose most likely output is `out_of_scope`.

Rules fixed before each step: pick the recipe (G5 vs EfficientNetV2-B0) on
validation balanced accuracy, out-of-scope rate (at most +2 points) and 3rd
wrong label (no increase); then replace the shipped classifier only if a
retrain lowers the validation out-of-scope rate by at least 5 points, loses at
most 0.02 balanced accuracy, is no less accurate when it answers, and shows no
more 3rd degree burns a wrong label.

| Run | Model | Val balanced acc | Val out-of-scope confidently labelled | Val wound photos rejected | Val correct when answered | Val 3rd wrong label | Test acc | Test out-of-scope confidently labelled | Test answered | Test correct when answered |
|---|---|---|---|---|---|---|---|---|---|---|
| shipped | classifier (b84b7b4), one run | 0.669 | 29.8% | 1.2% | 84.6% | 1 | 61.3% | 23.1% | 49.5% | 75.0% |
| I1 | G5 recipe, all new out-of-scope data | 0.627 ± 0.016 | 8.9% | 9.8% | 82.4% | 2.33 | 57.3 ± 1.7% | 10.6% | 38.8% | 78.1% |
| I2 | EfficientNetV2-B0, label smoothing 0.1 | 0.654 ± 0.017 | 12.4% | 10.9% | 84.3% | 2.33 | 59.3 ± 0.9% | 12.7% | 46.8% | 78.5% |
| J1 | G5, out-of-scope training photos capped at 1,000 | 0.629 ± 0.015 | 11.7% | 10.5% | 84.3% | 1.00 | 59.4 ± 0.9% | 14.6% | 43.1% | 76.9% |
| J2 | G5, capped at 2,000 | 0.620 ± 0.021 | 11.3% | 10.4% | 83.7% | 1.67 | 58.1 ± 3.1% | 13.2% | 44.1% | 80.1% |

- **Round I:** EfficientNetV2-B0 failed the recipe rule (+3.6 points
  out-of-scope), so G5 stayed. The G5 retrain (I1) failed the replacement
  rule: far fewer out-of-scope photos labelled, but lower balanced accuracy and
  accuracy when answering, more 3rd degree burns shown a wrong label, and ~10%
  of wound photos rejected as out of scope.
- **Round J:** capping the out-of-scope training photos (evenly across photo
  groups) did not fix that: wound photos were still rejected ~10% of the time,
  so the cause is which out-of-scope photos are used (some look like wounds),
  not how many. Neither cap passed.
- **The gate:** keep the shipped classifier for naming the wound and use a
  retrain only to add `unknown`. Simulated first from the saved probabilities
  of rounds I and J: the threshold was set on validation (the lowest of
  0.5–0.95 that costs at most 5 correct validation wound answers), and all
  three gate candidates passed the replacement rule. The final gate
  (`train_model.py --gate`, seed 42) was then checked again: threshold 0.5,
  validation out-of-scope rate 29.8% → 17.0%, balanced accuracy of the shown
  answer 0.429 → 0.416, correct when answered 84.6% → 85.3%, 3rd wrong label
  1 → 1. It passed and was adopted; its test results are in
  [Current performance](#current-performance-honest-held-out-test-set).
  (For the gate, balanced accuracy is measured on the shown answer, with
  `unknown` counted as wrong, because the gate never changes the most likely
  class.)

**Round G: training recipe on the previous out-of-scope split** (app image loader; test:
331 wound photos, 40 of them 3rd degree burns, and 138 out-of-scope photos;
validation: 31 3rd degree burns, 33 out-of-scope photos).

Rule fixed before the runs: adopt a change only if its mean validation
balanced accuracy beats the baseline by more than the baseline's seed
standard deviation, it does not raise the validation out-of-scope rate by
more than 2 photos, and it does not raise the number of validation 3rd degree
burns shown a wrong wound label.

Correction (found after publishing): `evaluate_model.py`'s
`burn_3rd_app_says_other_wound` already includes burns shown "1st degree", but
the "wrong wound label" check below added the "1st degree" count on top, so a
3rd degree burn shown "1st degree" was counted twice. The same sum was used
for every run, so the comparisons are consistent, but the column overstates
the count. The per-run files needed to recompute it were lost, so the numbers
are left as measured. In the last column, "any other wound" includes the "1st
degree" cases.

| Run | Change | Val balanced acc | Val out-of-scope confidently labelled | Val 3rd shown wrong wound label (1st degree counted twice) | Test acc | Test balanced acc | Test out-of-scope confidently labelled | Test 3rd shown 1st / any other wound (incl. 1st) / `unknown` | Test 2nd degree recall |
|---|---|---|---|---|---|---|---|---|---|
| G0 (6 seeds) | Baseline: previous recipe (width 1.0, 3rd degree weight ×2) | 0.602 ± 0.034 | 26.3% | 3.3 | 51.2 ± 1.5% | 54.3% | 22.6% | 0.3 / 4.2 / 14.7 | 24.3% |
| G1 | + label smoothing 0.1 | 0.636 ± 0.021 | 14.1% | 1.7 | 52.4 ± 1.6% | 54.6% | 13.0% | 0.0 / 1.7 / 22.0 | 29.3% |
| G2 | 3rd degree weight ×1.5 | 0.646 ± 0.008 | 22.2% | 4.3 | 51.8 ± 3.5% | 53.5% | 18.8% | 1.0 / 5.7 / 14.7 | 31.7% |
| G3 | MobileNetV2 width 1.4 | 0.672 ± 0.014 | 13.1% | 5.0 | 58.6 ± 2.1% | 60.0% | 15.9% | 1.3 / 7.7 / 9.3 | 36.9% |
| G4 | fine-tuning learning rate 3e-5 | 0.612 ± 0.012 | 18.2% | 4.0 | 53.1 ± 1.1% | 53.7% | 15.5% | 0.0 / 5.7 / 17.0 | 34.1% |
| **G5 (6 seeds)** | **width 1.4 + label smoothing 0.1 (shipped)** | **0.662 ± 0.021** | **7.6%** | **2.7** | **59.5 ± 2.1%** | **60.1%** | **9.2%** | 1.0 / 2.5 / 21.3 | **40.8%** |
| G6 | width 1.4 + 3rd degree weight ×3 | 0.618 ± 0.010 | 21.2% | 6.7 | 56.6% | 56.2% | 16.9% | 1.7 / 4.7 / 18.3 | 34.1% |

G3 was the most accurate but failed the rule: it was confident more often,
so the same few 3rd degree mistakes more often passed the 0.60 threshold. G5
and G6 were follow-ups aimed at that. With 3 seeds, G5 missed the rule by
0.7 photos on the 3rd degree check (3.7 vs 3.0); three more seeds each of G0
and G5 were then run, and with 6 seeds G5 passed all three checks. Test
results for those configs had already been seen at that point. G5's costs:
slightly more 3rd degree burns shown "1st degree" on test (1.0 vs 0.3 of 40)
and more shown `unknown` (21.3 vs 14.7), and a model file twice the size
(18 MB vs 9.7 MB).

**Round F: out-of-scope class** (earlier, more lenient out-of-scope split at
0.80, so the out-of-scope numbers are optimistic; dataset loader; test: 331
wound photos, 146 out-of-scope photos). "Recall" and "accuracy" here use the
most likely wound class:

| Run | Model | Val wound balanced acc | Val out-of-scope confidently labelled | Test wound acc | Test out-of-scope confidently labelled | Wound photos rejected (val / test) | Test 3rd degree recall | Test 3rd shown another wound label | Test 3rd shown `unknown` |
|---|---|---|---|---|---|---|---|---|---|
| F0 | 6 classes | 0.628 ± 0.021 | 66.7% | 53.7 ± 0.6% | 61.4% | 0% / 0% | 69.2% | 5.0 | 11.7 |
| **F1** | **+ out_of_scope (adopted)** | 0.622 ± 0.007 | **8.5%** | 52.1 ± 1.0% | **9.4%** | 3.6% / 3.8% | 70.0% | 3.7 | 17.3 |

**Round E: extra training data** (current wound split; dataset loader; test:
331 photos, 40 of them 3rd degree burns):

| Run | Change | Val balanced acc | Test acc | Test balanced acc | Test 3rd degree recall | Test 3rd → 1st (confident) | Test 3rd confidently wrong |
|---|---|---|---|---|---|---|---|
| E1 | C9 recipe, original data only | 0.615 | 53.7 ± 2.2% | 54.9% | 65.0% | 2.3 (0.7) | 6.0 |
| **E2** | **+ 168 extra training images (adopted)** | **0.648** | 52.7 ± 1.5% | 55.4% | 68.3% | 3.0 (0.3) | 4.7 |
| E3 | + only the extra 3rd degree burns | 0.617 | 51.4 ± 2.6% | 53.7% | 69.2% | 2.0 (0.3) | 4.3 |

E2 was better than E1 on validation for all three seeds (0.640/0.648/0.656 vs
0.591/0.611/0.643). The experiment filter admitted 168 extras; the stricter
filter now in `train_model.py` admits 149.

**Round C: first training recipe** (earlier split, which still contained
rotated/cropped near-duplicates, so absolute numbers are inflated; dataset
loader; test: 331 photos, 41 of them 3rd degree):

| Run | Change | Val balanced acc | Test acc | Test balanced acc | Test 3rd degree recall | Test 3rd → 1st (confident) |
|---|---|---|---|---|---|---|
| Before | Original model, original **leaky** test set | – | 76.3% | 76.3% | 88.5% | 3 (2) of 87 |
| C0 | Hash-grouped split, early stopping still on test | 0.527 | 57.1 ± 0.4% | 55.2% | 37.4% | 3.7 (0.7) |
| C1 | + validation split for early stopping | 0.545 | 57.8 ± 1.0% | 56.3% | 35.8% | 5.0 (0.7) |
| C2 | + fine-tune top 50 layers of MobileNetV2 | 0.581 | 59.0 ± 1.1% | 58.2% | 36.6% | 4.7 (1.7) |
| C3 | + real photos only, balanced class weights | 0.583 | 60.8 ± 1.0% | 59.7% | 48.0% | 4.3 (1.3) |
| C4 | + on-the-fly augmentation | 0.617 | 60.5 ± 2.4% | 60.6% | 49.6% | 5.7 (3.7) |
| C5 | + up to 40 fine-tuning epochs | 0.604 | 59.4 ± 2.3% | 61.0% | 55.3% | 5.0 (2.7) |
| C6 | + fine-tune top 100 layers | 0.639 | 62.6 ± 2.2% | 62.8% | 51.2% | 5.0 (3.0) |
| C7 | C5 with MobileNetV2 ×1.4 width | 0.630 | 64.6 ± 0.9% | 63.1% | 43.9% | 2.7 (1.0) |
| C8 | C5 with EfficientNetV2-B0 | 0.647 | 66.7 ± 1.8% | 67.4% | 44.7% | 3.0 (0.7) |
| C9 | C6 + 3rd degree class weight ×2 | 0.638 | 61.1 ± 2.1% | 63.1% | 58.5% | 4.3 (3.3) |

Also tried on that split and **not** adopted because they did not help on
validation: removing the 58 training photos whose duplicates carry
conflicting labels (0.616), test-time mirror averaging (0.637), and averaging
three models (0.634; 0.665 with mirror averaging, not adopted because of the
3× model size and ~6× inference cost).

## Known limitations

- **Small, web-scraped dataset.** 1,210 wound training photos (1,061 original
  + 149 extra), only 129–292 per class. Test per-class counts are 33–88, so
  per-class numbers carry wide confidence intervals.
- **Half of wound photos get `unknown`**, including 20 of 40 test 3rd degree
  burns. Those users are advised to see a professional, not told to call
  emergency services.
- **Burn degrees are hard for this model.** 2nd degree recall is 31.3%.
- **Out-of-scope detection is only as good as its data.** About 1 in 7
  out-of-scope test photos still gets a confident wound label, often a burn
  (see [Out-of-scope photos](#out-of-scope-photos)).
- **Answers depend on how the photo is resized or cropped.** Resampling alone
  changed about 1 in 9 answers (see
  [Real uploads](#real-uploads-resized-photos)).
- **Photos taken from further away mostly get `unknown`** - answered falls from
  48.0% to 19.6% when the wound fills half the frame (see
  [Photos taken from further away](#photos-taken-from-further-away)). Four
  training fixes were tried and none passed.
- **The classifier cannot be retrained exactly from the current code and
  data.** It was trained at commit b84b7b4, before the four extra out-of-scope
  downloads were added; `python src/train_model.py` now trains on all of them
  (round I1 above, which was not adopted).
- **Label conflicts in the burn data.** 34 duplicate groups (119 images) in
  the original data carry more than one label, mostly 1st vs 2nd (22 groups)
  and 2nd vs 3rd degree (9 groups). A few mixed groups may be look-alikes
  rather than true duplicates. They are kept together on one side of the
  split, not relabelled.
- **Staged and mislabelled images.** Some photos appear to be first-aid
  training makeup or stock photos, and some extra 2nd degree burns look like
  abrasions.
- **Model size.** Two 18 MB models (`wound_model.keras`,
  `out_of_scope_gate.keras`), and two 17 MB TensorFlow Lite copies for
  serving. Every photo runs through both.

## Usage

Tested with Python 3.12, TensorFlow 2.21, Keras 3.15.1.

```bash
pip install tensorflow pillow numpy opencv-python flask matplotlib ai-edge-litert

python src/collate_data.py        # needs data/raw_downloads/ (not in git); rebuilds data/wound_dataset
python src/collate_extra_data.py  # needs the seven extra Kaggle downloads in data/raw_downloads/
python src/train_model.py         # split -> add extras and out-of-scope photos -> train -> evaluate on test -> models/
python src/train_model.py --gate  # same split and recipe, saved as models/out_of_scope_gate.keras
python src/evaluate_model.py      # classifier + gate on data/test, as the app runs (--no-gate: classifier alone)
python src/evaluate_model.py --ood-dir some/folder   # optional: any folder of <group>/<image> out-of-scope photos
python src/framing_check.py       # how the answers change when the wound fills less of the frame
python src/export_tflite.py       # both models -> models/*.tflite, only if every val/test answer matches
python src/model.py path/to/image.jpg
python app.py                     # web app on http://127.0.0.1:5000
```

`evaluate_model.py` and `export_tflite.py` need `data/val` and `data/test`
including `out_of_scope/`, which are not in git; rebuild them as described in
[Extra data](#extra-data).

**Serving.** `app.py` uses the Keras models by default. With
`WOUND_MODEL_FORMAT=tflite` it serves the TensorFlow Lite copies through
`src/litert_model.py` without importing TensorFlow (a cold start with
TensorFlow took ~25 s). `Dockerfile`, `requirements-server.txt` and
`.gcloudignore` build that server for Google Cloud Run
(`gcloud run deploy --source .`); do not add gunicorn `--preload`, which made
every request hang when the server used TensorFlow. After retraining, run
`export_tflite.py` before deploying.

`data/wound_dataset/` still contains the 991 `_aug` files made by earlier
versions of `collate_data.py`. `train_model.py` ignores them (they are marked
`augmented=yes` in the manifest), and re-running `collate_data.py` will
remove them.

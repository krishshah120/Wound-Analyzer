# Wound-Analyzer

Classifies a photo of a wound into one of six classes: `abrasion`, `bruise`,
`cut`, `burn_1st_degree`, `burn_2nd_degree`, `burn_3rd_degree`. The web app
(`app.py`) returns the label plus general first-aid pointers, and answers
`unknown` whenever the model's confidence is below `CONFIDENCE_THRESHOLD`
(0.60, in `src/model.py`).

**This is not a diagnostic tool.** On photos it has never seen, the current
model is right about half the time (details below). It also gives confident
wound labels to most photos that are *not* one of its six classes (see
[Out-of-scope photos](#out-of-scope-photos)). Anyone who might have a serious
injury, especially a burn, should seek medical care regardless of what this
app says.

## Current performance (honest, held-out test set)

Measured with `python src/evaluate_model.py` on `data/test`: 331 real photos.
No test photo shares a source photo with training or validation, and no test
photo is visually near-identical to one, including rotated, cropped, mirrored
or re-watermarked copies. Full numbers are in `models/metrics.json`.

| Metric | Value |
|---|---|
| Accuracy | **50.8%** (95% CI 45.4%–56.1%) |
| Balanced accuracy (mean per-class recall) | 52.3% |
| Share of photos answered (confidence ≥ 0.60) | 60.4% |
| Accuracy on those answered photos | 62.0% |

The shipped model is one training run (seed 42). Across three seeds of the
same recipe, test accuracy was 52.7% ± 1.5% (run E2 below), so this run is at
the low end of normal seed-to-seed variation.

| Class | Test photos | Recall | Recall 95% CI | Precision |
|---|---|---|---|---|
| abrasion | 41 | 46.3% | 32%–61% | 54.3% |
| bruise | 33 | 54.5% | 38%–70% | 45.0% |
| burn_1st_degree | 88 | 56.8% | 46%–67% | 64.9% |
| burn_2nd_degree | 83 | 36.1% | 27%–47% | 43.5% |
| burn_3rd_degree | 40 | 67.5% | 52%–80% | 36.5% |
| cut | 46 | 52.2% | 38%–66% | 66.7% |

Confusion matrix (rows = true class, columns = predicted class):

| true \ predicted | abrasion | bruise | burn 1st | burn 2nd | burn 3rd | cut |
|---|---|---|---|---|---|---|
| abrasion | **19** | 3 | 3 | 7 | 4 | 5 |
| bruise | 2 | **18** | 7 | 4 | 1 | 1 |
| burn_1st_degree | 1 | 12 | **50** | 13 | 10 | 2 |
| burn_2nd_degree | 7 | 0 | 12 | **30** | 31 | 3 |
| burn_3rd_degree | 2 | 1 | 2 | 7 | **27** | 1 |
| cut | 4 | 6 | 3 | 8 | 1 | **24** |

### Burn severity

The app routes `burn_3rd_degree` to "call emergency services", so the errors
that matter most are 3rd degree burns called something milder:

- 27 of 40 3rd degree burns correctly identified (67.5%).
- 2 of 40 predicted as 1st degree, **neither confidently**, so the app would
  have answered `unknown` for both.
- 5 of 40 were confidently predicted as something other than 3rd degree.
- The trade-off: training weights 3rd degree burns more heavily, so the
  model over-calls 3rd degree. **31 of 83 2nd degree burns were predicted as
  3rd degree**, and only 36.5% of 3rd degree predictions are correct. That
  errs toward sending people to emergency care rather than away from it.

### Out-of-scope photos

Measured with `python src/evaluate_model.py --ood-dir data/ood_dataset` on
1,073 photos that belong to none of the six classes (normal skin and chronic
wounds, from the ibrahimfateen Kaggle dataset below, mirrored copies removed).
Ideally all of them would come back `unknown`.

| Photos | Count | Confidently given a wound label | Most common labels |
|---|---|---|---|
| Diabetic wounds | 231 | 71.0% | burn_3rd_degree 128, burn_2nd_degree 27 |
| Normal skin | 86 | 66.3% | burn_1st_degree 32, bruise 21 |
| Pressure wounds | 300 | 64.3% | burn_3rd_degree 154, burn_2nd_degree 31 |
| Surgical wounds | 209 | 62.2% | burn_3rd_degree 80, cut 21 |
| Venous wounds | 247 | 65.2% | burn_3rd_degree 106, burn_2nd_degree 42 |
| **All** | **1,073** | **65.7%** | |

**The 0.60 confidence threshold does not make the model say `unknown` for
things it was never trained on.** Chronic wounds are most often called 3rd
degree burns. The previous model did about the same on these photos (62.5%).
A model trained only on six classes has no way to recognise a seventh; fixing
this needs a design change (for example an explicit "other" class), not a
different threshold.

## What changed, and why the number went down

The original model reported **76.25%** test accuracy. Two rounds of train/test
leakage inflated that number.

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
whole groups into train/val/test, augmented only training batches, and early-
stopped on validation. That model scored 59.8%.

**Round 2: rotated, cropped and re-watermarked copies.** A perceptual hash
does not recognise a photo that has been rotated, cropped, mirrored or
re-watermarked, and this data contains many. Comparing images by visual
similarity across rotated, cropped and mirrored views found **179 train/test
image pairs with similarity ≥ 0.80** in that split. Checked by eye, pairs at
≥ 0.85 were almost all the same photo, and pairs at 0.80–0.85 were often the
same photo. So the 59.8% was still inflated.

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
- refuses to continue if any group crosses splits or any cross-split pair
  reaches 0.80, and writes `data/split_manifest.csv`.

Independent checks of the current split: 0 shared source filenames, 0
near-identical hashes (mirrored or not), and 0 cross-split pairs at
similarity ≥ 0.80.

The similarity thresholds were set by looking at sample pairs, not derived
from ground truth: a few duplicates may remain below 0.80, and the test
photos are still web-scraped, so real user photos may score differently.

## Extra training data

Three more Kaggle datasets were downloaded and cleaned with
`src/collate_extra_data.py`:

| Dataset | Licence | Used for |
|---|---|---|
| [faresabbasai2022/burn-dataset](https://www.kaggle.com/datasets/faresabbasai2022/burn-dataset) | Apache 2.0 | burn degree candidates |
| [ibrahimfateen/wound-classification](https://www.kaggle.com/datasets/ibrahimfateen/wound-classification) | unknown | abrasion/bruise/cut candidates, out-of-scope photos |
| [yasinpratomo/wound-dataset](https://www.kaggle.com/datasets/yasinpratomo/wound-dataset) | unknown | abrasion/bruise/cut candidates |

Most of that data is the same web-scraped photos already here.
`train_model.py` adds an extra image **to train only**, and only if it has no
near-identical hash and visual similarity < 0.75 to every val/test photo and
< 0.80 to every train photo. Of 2,717 candidates that passed quality checks,
1,517 were skipped as near-copies of a val/test photo, 963 as near-copies of a
train photo, and 88 as repeats of each other. **149 were added**: 89 3rd
degree burns, 24 2nd degree, 23 bruises, 5 cuts, 4 abrasions, 4 1st degree.

These images are gitignored (two sources have no stated licence). To rebuild
them, unzip each download into `data/raw_downloads/` as described at the top
of `src/collate_extra_data.py`. To train without them, set
`USE_EXTRA_TRAINING_DATA = False` in `src/train_model.py`.

On a visual spot check, some extra 3rd degree burns look like staged
first-aid training makeup, and some extra 2nd degree burns look like
abrasions. Their labels were not changed.

## Experiments

Configurations were **chosen on the validation split**; test is shown for
every run but was not used to pick one. Each row is the mean of 3 training
seeds (± standard deviation).

**On the corrected split** (test: 331 photos, 40 of them 3rd degree burns):

| Run | Change | Val balanced acc | Test acc | Test balanced acc | Test 3rd degree recall | Test 3rd → 1st (confident) | Test 3rd confidently wrong |
|---|---|---|---|---|---|---|---|
| E1 | C9 recipe, original data only | 0.615 | 53.7 ± 2.2% | 54.9% | 65.0% | 2.3 (0.7) | 6.0 |
| **E2** | **+ 168 extra training images (shipped recipe)** | **0.648** | 52.7 ± 1.5% | 55.4% | 68.3% | 3.0 (0.3) | 4.7 |
| E3 | + only the extra 3rd degree burns | 0.617 | 51.4 ± 2.6% | 53.7% | 69.2% | 2.0 (0.3) | 4.3 |

E2 was better than E1 on validation for all three seeds (0.640/0.648/0.656 vs
0.591/0.611/0.643). On test, the extra data changed overall accuracy by less
than the seed-to-seed spread, with slightly better 3rd degree numbers. The
experiment filter admitted 168 extras; the stricter filter now in
`train_model.py` admits 149.

**On the earlier split** (still containing rotated/cropped near-duplicates,
so absolute numbers are inflated; test: 331 photos, 41 of them 3rd degree).
These runs chose the training recipe:

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

C9 was chosen over C8 because EfficientNetV2-B0, despite the best overall
accuracy, was the worst config for 3rd degree burns on validation (47%
recall, 9.7 of 30 confidently predicted as something else). Also tried on
that split and **not** adopted because they did not help on validation:
removing the 58 training photos whose duplicates carry conflicting labels
(0.616), test-time mirror averaging (0.637), and averaging three models
(0.634; 0.665 with mirror averaging, not adopted because of the 3× model
size and ~6× inference cost).

## Known limitations

- **Small, web-scraped dataset.** 1,210 training photos (1,061 original + 149
  extra), only 129–292 per class. Test per-class counts are 33–88, so
  per-class numbers carry wide confidence intervals.
- **Out-of-scope photos get confident labels** (65.7%, see above).
- **Label conflicts in the burn data.** 34 duplicate groups (119 images) in
  the original data carry more than one label, mostly 1st vs 2nd (22 groups)
  and 2nd vs 3rd degree (9 groups). A few mixed groups may be look-alikes
  rather than true duplicates. They are kept together on one side of the
  split, not relabelled.
- **Overfitting.** Training accuracy (on augmented batches) ends well above
  validation accuracy (e.g. 76–89% vs 57–63% in the C9 runs).
- **Staged and mislabelled images.** Some photos appear to be first-aid
  training makeup or stock photos, and some extra 2nd degree burns look like
  abrasions.

## Usage

Tested with Python 3.12, TensorFlow 2.21, Keras 3.15.1.

```bash
pip install tensorflow pillow numpy opencv-python flask matplotlib

python src/collate_data.py        # needs data/raw_downloads/ (not in git); rebuilds data/wound_dataset
python src/collate_extra_data.py  # needs the three extra Kaggle downloads in data/raw_downloads/
python src/train_model.py         # split -> add extras to train -> train -> evaluate on test -> models/
python src/evaluate_model.py --ood-dir data/ood_dataset
python src/model.py path/to/image.jpg
python app.py                     # web app on http://127.0.0.1:5000
```

`data/wound_dataset/` still contains the 991 `_aug` files made by earlier
versions of `collate_data.py`. `train_model.py` ignores them (they are marked
`augmented=yes` in the manifest), and re-running `collate_data.py` will
remove them.

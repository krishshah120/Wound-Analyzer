# Wound-Analyzer

Classifies a photo of a wound into one of six classes: `abrasion`, `bruise`,
`cut`, `burn_1st_degree`, `burn_2nd_degree`, `burn_3rd_degree`. The web app
(`app.py`) returns the label plus general first-aid pointers, and answers
`unknown` when the photo looks like none of those classes or the model's
confidence is below `CONFIDENCE_THRESHOLD` (0.60, in `src/model.py`).

**This is not a diagnostic tool.** On wound photos it has never seen, the
current model is right about half the time (details below). Anyone who might
have a serious injury, especially a burn, should seek medical care regardless
of what this app says.

## How a prediction is made

The model has seven outputs: the six wound classes plus `out_of_scope`, which
was trained on photos of normal skin and chronic wounds (diabetic, pressure,
surgical and venous). `decide()` in `src/model.py` turns the outputs into the
app's response:

- `best_guess` is the most likely **wound** class (never `out_of_scope`), and
  `confidence` is its probability.
- `label` is `unknown` if `out_of_scope` is the most likely output or
  `confidence` < 0.60; otherwise it is `best_guess`.

So `/predict` returns the same fields and the same possible `label` values as
before: `{label, confidence, best_guess, tips, disclaimer}`.

## Current performance (honest, held-out test set)

Measured with `python src/evaluate_model.py` on `data/test`, loading images
the same way the app does. Full numbers are in `models/metrics.json`.

- 331 real wound photos. None shares a source photo with training or
  validation, and none is visually near-identical to one, including rotated,
  cropped, mirrored or re-watermarked copies.
- 154 held-out out-of-scope photos (normal skin and chronic wounds).

**Wound photos**

| Metric | Value |
|---|---|
| Accuracy (a wound photo rejected as out of scope counts as wrong) | **50.2%** (95% CI 44.8%–55.5%) |
| Balanced accuracy (mean per-class recall) | 53.3% |
| Share answered (label is not `unknown`) | 55.9% |
| Accuracy on those answered photos | 64.3% |
| Rejected as out of scope | 4.5% |

**Out-of-scope photos**

| Metric | Value |
|---|---|
| Confidently given a wound label (instead of `unknown`) | **7.8%** (12 of 154) |

That 7.8% is optimistic: see [Out-of-scope photos](#out-of-scope-photos).

The shipped model is one training run (seed 42). See
[Experiments](#experiments) for the spread across seeds.

| Class | Test photos | Recall | Recall 95% CI | Precision |
|---|---|---|---|---|
| abrasion | 41 | 31.7% | 20%–47% | 56.5% |
| bruise | 33 | 69.7% | 53%–83% | 46.9% |
| burn_1st_degree | 88 | 56.8% | 46%–67% | 63.3% |
| burn_2nd_degree | 83 | 26.5% | 18%–37% | 44.9% |
| burn_3rd_degree | 40 | 70.0% | 55%–82% | 31.8% |
| cut | 46 | 65.2% | 51%–77% | 61.2% |
| out_of_scope | 154 | 86.4% | 80%–91% | 89.9% |

Confusion matrix (rows = true class, columns = most likely output, before the
confidence threshold):

| true \ predicted | abrasion | bruise | burn 1st | burn 2nd | burn 3rd | cut | out of scope |
|---|---|---|---|---|---|---|---|
| abrasion | **13** | 5 | 6 | 3 | 5 | 7 | 2 |
| bruise | 1 | **23** | 5 | 3 | 0 | 1 | 0 |
| burn_1st_degree | 1 | 13 | **50** | 11 | 10 | 3 | 0 |
| burn_2nd_degree | 7 | 0 | 12 | **22** | 30 | 4 | 8 |
| burn_3rd_degree | 1 | 1 | 4 | 2 | **28** | 1 | 3 |
| cut | 0 | 6 | 2 | 5 | 1 | **30** | 2 |
| out_of_scope | 0 | 1 | 0 | 3 | 14 | 3 | **133** |

### Burn severity

The app routes `burn_3rd_degree` to "call emergency services", so the errors
that matter most are 3rd degree burns shown something milder. Of 40 test
3rd degree burns, the app would return:

- `burn_3rd_degree` for 23;
- **`burn_1st_degree` for none**;
- another wound label for 3;
- `unknown` for 14 (3 of them because the photo looked out of scope).

The model over-calls 3rd degree: 30 of 83 2nd degree burns were most likely
3rd degree, and only 31.8% of 3rd degree predictions are correct. That errs
toward emergency care rather than away from it.

### Out-of-scope photos

A model trained only on the six wound classes has no way to recognise a
seventh: the previous six-class model gave a confident wound label to
**68.2%** of these same 154 test photos, mostly `burn_3rd_degree`. The
`out_of_scope` class fixes most of that:

| Test photos | Count | 6-class model: confident wound label | Current model: confident wound label |
|---|---|---|---|
| Diabetic wounds | 39 | 27 | 5 |
| Pressure wounds | 48 | 35 | 4 |
| Surgical wounds | 33 | 22 | 2 |
| Venous wounds | 32 | 20 | 0 |
| Normal skin | 2 | 1 | 1 |
| **All** | **154** | **68.2%** | **7.8%** |

Limits of this result:

- **It is optimistic.** The chronic-wound photos include several photos of the
  same patient or wound, some just below the duplicate threshold, so
  similar photos sit in training and test. Counting only test photos with no
  out-of-scope training photo at similarity ≥ 0.75, the rate rises to 11.0%
  (100 photos); at ≥ 0.70, to 24.2% (33 photos). The six-class model stays at
  63–70% on those subsets. Expect the real-world rate for unfamiliar
  out-of-scope photos to be well above 7.8%, though still far below 68%.
- **Normal skin is barely tested.** Its 86 photos are so alike that nearly all
  fell into one group in training, leaving 2 in test.
- **All out-of-scope photos come from one Kaggle source**, so the model may
  partly be recognising that source's photo style.
- **Other things outside the six classes** (rashes, bites, infections, photos
  that are not skin at all) were not tested.

## What changed, and why the number went down

The original model reported **76.25%** test accuracy (75.7% when images are
loaded the way the app loads them). That number was inflated by train/test
leakage, found and fixed in two rounds, and the model was later given the
out-of-scope class.

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
near-identical hashes (mirrored or not), and 0 cross-split wound-photo pairs
at similarity ≥ 0.80.

The similarity thresholds were set by looking at sample pairs, not derived
from ground truth: a few duplicates may remain below 0.80, and the test
photos are still web-scraped, so real user photos may score differently.

**Evaluation now loads images the way the app does.** TensorFlow's dataset
loader decodes JPEGs slightly differently from `tf.keras.utils.load_img`
(used by `predict()`), which changed the answer on 12 of 485 test photos.
`evaluate_model.py` now uses the app's loader; the experiment tables below
were measured with the dataset loader.

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

**Out-of-scope photos.** 1,073 photos of normal skin and chronic wounds (after
removing the download's mirrored copies) are grouped and split the same way
as the wound photos, and any that resemble a wound photo in a different split
are skipped (44 skipped). Result: 762 train / 113 val / 154 test.

These images are gitignored (two sources have no stated licence), so
`data/*/out_of_scope/` and the `extra_*` files are not in the repository. To
rebuild them, unzip each download into `data/raw_downloads/` as described at
the top of `src/collate_extra_data.py`, then run `train_model.py`. To train
without them, set `USE_EXTRA_TRAINING_DATA = False` and/or
`USE_OUT_OF_SCOPE_CLASS = False` in `src/train_model.py`.

## Experiments

Configurations were **chosen on the validation split**; test is shown for
every run but was not used to pick one. Each row is the mean of 3 training
seeds (± standard deviation). Measured with TensorFlow's dataset loader.

**Out-of-scope class** (corrected split plus a held-out out-of-scope split;
test: 331 wound photos, 40 of them 3rd degree burns, and 146 out-of-scope
photos). "Recall" and "accuracy" here use the most likely wound class:

| Run | Model | Val wound balanced acc | Val out-of-scope confidently labelled | Test wound acc | Test out-of-scope confidently labelled | Wound photos rejected (val / test) | Test 3rd degree recall | Test 3rd shown another wound label | Test 3rd shown `unknown` |
|---|---|---|---|---|---|---|---|---|---|
| F0 | 6 classes | 0.628 ± 0.021 | 66.7% | 53.7 ± 0.6% | 61.4% | 0% / 0% | 69.2% | 5.0 | 11.7 |
| **F1** | **+ out_of_scope (shipped)** | 0.622 ± 0.007 | **8.5%** | 52.1 ± 1.0% | **9.4%** | 3.6% / 3.8% | 70.0% | 3.7 | 17.3 |

F1 was chosen because it cut confidently labelled out-of-scope photos by
about 58 percentage points on validation while wound balanced accuracy
stayed within seed-to-seed variation. The cost: about 4% of wound photos are
rejected as out of scope, more often 2nd and 3rd degree burns (on test, about
12% of 3rd degree burns, though only 2% on validation).

**Extra training data** (corrected split; test: 331 photos, 40 of them 3rd
degree burns):

| Run | Change | Val balanced acc | Test acc | Test balanced acc | Test 3rd degree recall | Test 3rd → 1st (confident) | Test 3rd confidently wrong |
|---|---|---|---|---|---|---|---|
| E1 | C9 recipe, original data only | 0.615 | 53.7 ± 2.2% | 54.9% | 65.0% | 2.3 (0.7) | 6.0 |
| **E2** | **+ 168 extra training images (shipped)** | **0.648** | 52.7 ± 1.5% | 55.4% | 68.3% | 3.0 (0.3) | 4.7 |
| E3 | + only the extra 3rd degree burns | 0.617 | 51.4 ± 2.6% | 53.7% | 69.2% | 2.0 (0.3) | 4.3 |

E2 was better than E1 on validation for all three seeds (0.640/0.648/0.656 vs
0.591/0.611/0.643). On test, the extra data changed overall accuracy by less
than the seed-to-seed spread, with slightly better 3rd degree numbers. The
experiment filter admitted 168 extras; the stricter filter now in
`train_model.py` admits 149.

**Training recipe** (earlier split, which still contained rotated/cropped
near-duplicates, so absolute numbers are inflated; test: 331 photos, 41 of
them 3rd degree):

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

- **Small, web-scraped dataset.** 1,210 wound training photos (1,061 original
  + 149 extra), only 129–292 per class. Test per-class counts are 33–88, so
  per-class numbers carry wide confidence intervals.
- **Burn degrees are hard for this model.** 2nd degree recall is 26.5%;
  most 2nd degree errors go to 3rd degree, and some are rejected as out of
  scope.
- **Out-of-scope detection is only as good as its data** (see
  [Out-of-scope photos](#out-of-scope-photos)).
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
- **The `unknown` tips** in `app.py` say the model "wasn't confident enough"
  and suggest a clearer photo, which fits a blurry photo better than a
  chronic wound; they do end by advising a medical professional.

## Usage

Tested with Python 3.12, TensorFlow 2.21, Keras 3.15.1.

```bash
pip install tensorflow pillow numpy opencv-python flask matplotlib

python src/collate_data.py        # needs data/raw_downloads/ (not in git); rebuilds data/wound_dataset
python src/collate_extra_data.py  # needs the three extra Kaggle downloads in data/raw_downloads/
python src/train_model.py         # split -> add extras and out-of-scope photos -> train -> evaluate on test -> models/
python src/evaluate_model.py      # re-evaluate models/wound_model.keras on data/test
python src/evaluate_model.py --ood-dir some/folder   # optional: any folder of <group>/<image> out-of-scope photos
python src/model.py path/to/image.jpg
python app.py                     # web app on http://127.0.0.1:5000
```

`evaluate_model.py` needs `data/test/out_of_scope/`, which is not in git;
rebuild it as described in [Extra data](#extra-data).

`data/wound_dataset/` still contains the 991 `_aug` files made by earlier
versions of `collate_data.py`. `train_model.py` ignores them (they are marked
`augmented=yes` in the manifest), and re-running `collate_data.py` will
remove them.

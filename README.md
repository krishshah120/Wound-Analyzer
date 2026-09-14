# Wound-Analyzer

Classifies a photo of a wound into one of six classes: `abrasion`, `bruise`,
`cut`, `burn_1st_degree`, `burn_2nd_degree`, `burn_3rd_degree`. The web app
(`app.py`) returns the label plus general first-aid pointers, and answers
`unknown` when the photo looks like none of those classes or the model's
confidence is below `CONFIDENCE_THRESHOLD` (0.60, in `src/model.py`).

**This is not a diagnostic tool.** On wound photos it has never seen, the
current model's most likely class is right about 6 times in 10, and it
answers `unknown` for about half of them (details below). Anyone who might
have a serious injury, especially a burn, should seek medical care regardless
of what this app says.

## How a prediction is made

The model is a MobileNetV2 (width 1.4) with seven outputs: the six wound
classes plus `out_of_scope`, which was trained on photos of normal skin and
chronic wounds (diabetic, pressure, surgical and venous). `decide()` in
`src/model.py` turns the outputs into the app's response:

- `best_guess` is the most likely **wound** class (never `out_of_scope`), and
  `confidence` is its probability.
- `label` is `unknown` if `out_of_scope` is the most likely output or
  `confidence` < 0.60; otherwise it is `best_guess`.

So `/predict` returns the same fields and the same possible `label` values as
before: `{label, confidence, best_guess, tips, disclaimer}`. The `unknown`
tips cover both cases (an unclear photo, or an injury the tool doesn't cover)
and advise seeing a professional for wounds that aren't healing or look
infected.

## Current performance (honest, held-out test set)

Measured with `python src/evaluate_model.py` on `data/test`, loading images
the same way the app does. Full numbers are in `models/metrics.json`.

- 331 real wound photos. None shares a source photo with training or
  validation, and none is visually near-identical to one (similarity ≥ 0.80),
  including rotated, cropped, mirrored or re-watermarked copies.
- 138 held-out out-of-scope photos (chronic wounds and 1 normal-skin photo).
  None has an out-of-scope training photo at similarity ≥ 0.75.

**Wound photos**

| Metric | Value |
|---|---|
| Accuracy (a wound photo rejected as out of scope counts as wrong) | **61.3%** (95% CI 56.0%–66.4%) |
| Balanced accuracy (mean per-class recall) | 63.1% |
| Share answered (label is not `unknown`) | 49.5% |
| Accuracy on those answered photos | 75.0% |
| Rejected as out of scope | 4.5% |

**Out-of-scope photos**

| Metric | Value |
|---|---|
| Confidently given a wound label (instead of `unknown`) | **4.3%** (6 of 138) |

The shipped model is one training run (seed 42). Across six seeds of the same
recipe, test accuracy was 59.5% ± 2.1% and out-of-scope photos confidently
labelled 9.2% (run G5 below), so this run is somewhat above average.

| Class | Test photos | Recall | Recall 95% CI | Precision |
|---|---|---|---|---|
| abrasion | 41 | 56.1% | 41%–70% | 71.9% |
| bruise | 33 | 63.6% | 47%–78% | 52.5% |
| burn_1st_degree | 88 | 77.3% | 67%–85% | 70.1% |
| burn_2nd_degree | 83 | 31.3% | 22%–42% | 49.1% |
| burn_3rd_degree | 40 | 67.5% | 52%–80% | 39.1% |
| cut | 46 | 82.6% | 69%–91% | 69.1% |
| out_of_scope | 138 | 78.3% | 71%–84% | 87.8% |

Confusion matrix (rows = true class, columns = most likely output, before the
confidence threshold):

| true \ predicted | abrasion | bruise | burn 1st | burn 2nd | burn 3rd | cut | out of scope |
|---|---|---|---|---|---|---|---|
| abrasion | **23** | 6 | 1 | 7 | 0 | 2 | 2 |
| bruise | 1 | **21** | 7 | 0 | 2 | 2 | 0 |
| burn_1st_degree | 0 | 4 | **68** | 11 | 2 | 2 | 1 |
| burn_2nd_degree | 4 | 1 | 18 | **26** | 21 | 7 | 6 |
| burn_3rd_degree | 2 | 3 | 2 | 1 | **27** | 1 | 4 |
| cut | 0 | 3 | 0 | 3 | 0 | **38** | 2 |
| out_of_scope | 2 | 2 | 1 | 5 | 17 | 3 | **108** |

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

2nd degree burns remain the weakest class: 21 of 83 were most likely 3rd
degree and 18 were most likely 1st degree.

### Out-of-scope photos

A model trained only on the six wound classes has no way to recognise a
seventh. On the same 138 test photos:

| Test photos | Count | 6-class model: confident wound label | Current model: confident wound label |
|---|---|---|---|
| Diabetic wounds | 38 | 27 | 0 |
| Pressure wounds | 45 | 30 | 2 |
| Surgical wounds | 37 | 22 | 3 |
| Venous wounds | 17 | 11 | 1 |
| Normal skin | 1 | 1 | 0 |
| **All** | **138** | **65.9%** | **4.3%** |

(The 6-class model is the one from commit 4321434: same wound photos, never
trained on any out-of-scope photo.)

Limits of this result:

- **Validation has only 33 out-of-scope photos.** The chronic-wound photos
  include many near-identical shots of the same wounds, so after keeping
  look-alikes (similarity ≥ 0.75) out of val/test, few were left for
  validation.
- **Normal skin is effectively untested** (1 test photo), and all out-of-scope
  photos come from one Kaggle source, so the model may partly be recognising
  that source's photo style.
- **Other things outside the six classes** (rashes, bites, infections, photos
  that are not skin at all) were not tested.

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

**Out-of-scope photos.** 1,073 photos of normal skin and chronic wounds (after
removing the download's mirrored copies) are grouped and split as described
above, and any that resemble a wound photo in a different split are skipped
(41 skipped). Result: 861 train / 33 val / 138 test.

These images are gitignored (two sources have no stated licence), so
`data/*/out_of_scope/` and the `extra_*` files are not in the repository. To
rebuild them, unzip each download into `data/raw_downloads/` as described at
the top of `src/collate_extra_data.py`, then run `train_model.py`. To train
without them, set `USE_EXTRA_TRAINING_DATA = False` and/or
`USE_OUT_OF_SCOPE_CLASS = False` in `src/train_model.py`.

## Experiments

Configurations were **chosen on the validation split**; test is shown for
every run but was not used to pick one. Each row is the mean of 3 training
seeds (± standard deviation) unless stated.

**Round G: training recipe on the current split** (app image loader; test:
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
- **Out-of-scope detection is only as good as its data** (see
  [Out-of-scope photos](#out-of-scope-photos)).
- **Label conflicts in the burn data.** 34 duplicate groups (119 images) in
  the original data carry more than one label, mostly 1st vs 2nd (22 groups)
  and 2nd vs 3rd degree (9 groups). A few mixed groups may be look-alikes
  rather than true duplicates. They are kept together on one side of the
  split, not relabelled.
- **Staged and mislabelled images.** Some photos appear to be first-aid
  training makeup or stock photos, and some extra 2nd degree burns look like
  abrasions.
- **Model size.** `models/wound_model.keras` is 18 MB (the width-1.4 network),
  twice the previous size.

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

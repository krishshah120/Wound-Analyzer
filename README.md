# Wound-Analyzer

Classifies a photo of a wound into one of six classes: `abrasion`, `bruise`,
`cut`, `burn_1st_degree`, `burn_2nd_degree`, `burn_3rd_degree`. The web app
(`app.py`) returns the label plus general first-aid pointers, and answers
`unknown` whenever the model's confidence is below `CONFIDENCE_THRESHOLD`
(0.60, in `src/model.py`).

**This is not a diagnostic tool.** On photos it has never seen, the current
model is right about 6 times in 10 (details below). Anyone who might have a
serious injury, especially a burn, should seek medical care regardless of
what this app says.

## Current performance (honest, held-out test set)

Measured with `python src/evaluate_model.py` on `data/test`: 331 real photos
whose source photos appear nowhere in training or validation. Full numbers are
in `models/metrics.json`.

| Metric | Value |
|---|---|
| Accuracy | **59.8%** (95% CI 54.5%–65.0%) |
| Balanced accuracy (mean per-class recall) | 60.7% |
| Share of photos answered (confidence ≥ 0.60) | 58.3% |
| Accuracy on those answered photos | 71.5% |

| Class | Test photos | Recall | Recall 95% CI | Precision |
|---|---|---|---|---|
| abrasion | 39 | 64.1% | 48%–77% | 64.1% |
| bruise | 33 | 48.5% | 33%–65% | 57.1% |
| burn_1st_degree | 93 | 73.1% | 63%–81% | 67.3% |
| burn_2nd_degree | 79 | 30.4% | 21%–41% | 51.1% |
| burn_3rd_degree | 41 | 63.4% | 48%–76% | 44.1% |
| cut | 46 | 84.8% | 72%–92% | 68.4% |

Confusion matrix (rows = true class, columns = predicted class):

| true \ predicted | abrasion | bruise | burn 1st | burn 2nd | burn 3rd | cut |
|---|---|---|---|---|---|---|
| abrasion | **25** | 1 | 2 | 1 | 3 | 7 |
| bruise | 7 | **16** | 10 | 0 | 0 | 0 |
| burn_1st_degree | 1 | 3 | **68** | 14 | 2 | 5 |
| burn_2nd_degree | 5 | 5 | 15 | **24** | 26 | 4 |
| burn_3rd_degree | 1 | 2 | 4 | 6 | **26** | 2 |
| cut | 0 | 1 | 2 | 2 | 2 | **39** |

### Burn severity

The app routes `burn_3rd_degree` to "call emergency services", so the errors
that matter most are 3rd degree burns called something milder:

- 26 of 41 3rd degree burns correctly identified (63.4%).
- 4 of 41 predicted as 1st degree. **2 of those were confident** (≥ 0.60),
  meaning the app would have shown 1st degree advice.
- 7 of 41 were confidently predicted as something other than 3rd degree.
- The trade-off: training weights 3rd degree burns 2× (see below), so the
  model over-calls 3rd degree. **26 of 79 2nd degree burns were predicted as
  3rd degree**, which is why 2nd degree recall is only 30%. That errs toward
  sending people to emergency care rather than away from it.

## What changed, and why the number went down

The previous model reported **76.25%** test accuracy. That number was
inflated by a train/test leak:

1. `collate_data.py` created augmented copies (`<photo>_aug<N>.jpg`) *before*
   the split, and `train_model.py` then split individual files at random. So
   copies of the same photo landed in both train and test: **248 of the 467
   distinct source photos in test (53.1%) also appeared in train**, and
   293 of 522 test files (56.1%) were copies of a training photo.
2. On the leaked test files the old model scored 88.7%. On test files with
   an unseen source photo it scored **60.3%**. Of the 87 3rd degree test
   images, 76 were leaked, so its 88.5% 3rd degree recall was mostly
   memorization.
3. The burn dataset also contains the same photo re-uploaded under different
   filenames (158 image pairs with identical perceptual hashes), sometimes
   with **different degree labels**. Filename-based grouping alone would still
   leak these.
4. Early stopping monitored the test set, so the test set was also used to
   make a training decision.

The fix:

- `train_model.py` now uses only the real photos from
  `data/wound_dataset/manifest.csv`. It groups every photo with its
  near-duplicates (difference hash, ≤ 6 bits apart), giving 1,643 photos in
  1,468 groups, and splits **whole groups** into train / val / test
  (≈ 65 / 15 / 20 %, stratified by class). It refuses to continue if any group
  crosses splits, and writes `data/split_manifest.csv` recording where every
  file went.
- Augmentation happens only on training batches, on the fly. Validation and
  test are real photos only. `collate_data.py` no longer augments.
- Early stopping uses the validation split. The test split is used only
  for the final report.

Independent check of the new split: 0 shared source filenames and 0
near-duplicate hash pairs (≤ 6 bits) between any two splits.

## Experiments

All runs use the same grouped split. Configurations were **chosen on the
validation split**. Test is shown for every run but was not used to pick
one. Each row is the mean of 3 training seeds (± standard deviation).
Test set: 331 photos, 41 of them 3rd degree burns.

| Run | Change | Val balanced acc | Test acc | Test balanced acc | Test 3rd degree recall | Test 3rd → 1st (confident) |
|---|---|---|---|---|---|---|
| Before | Old model, **leaky** test set | – | 76.3% | 76.3% | 88.5% | 3 (2) of 87 |
| Before | Old recipe retrained, **leaky** test set | – | 74.1% | – | – | – |
| C0 | Grouped split, early stopping still on test | 0.527 | 57.1 ± 0.4% | 55.2% | 37.4% | 3.7 (0.7) |
| C1 | Grouped split + validation split (honest baseline) | 0.545 | 57.8 ± 1.0% | 56.3% | 35.8% | 5.0 (0.7) |
| C2 | + fine-tune top 50 layers of MobileNetV2 | 0.581 | 59.0 ± 1.1% | 58.2% | 36.6% | 4.7 (1.7) |
| C3 | + real photos only, balanced class weights | 0.583 | 60.8 ± 1.0% | 59.7% | 48.0% | 4.3 (1.3) |
| C4 | + on-the-fly augmentation | 0.617 | 60.5 ± 2.4% | 60.6% | 49.6% | 5.7 (3.7) |
| C5 | + up to 40 fine-tuning epochs | 0.604 | 59.4 ± 2.3% | 61.0% | 55.3% | 5.0 (2.7) |
| C6 | + fine-tune top 100 layers | 0.639 | 62.6 ± 2.2% | 62.8% | 51.2% | 5.0 (3.0) |
| C7 | C5 with MobileNetV2 ×1.4 width | 0.630 | 64.6 ± 0.9% | 63.1% | 43.9% | 2.7 (1.0) |
| C8 | C5 with EfficientNetV2-B0 | 0.647 | 66.7 ± 1.8% | 67.4% | 44.7% | 3.0 (0.7) |
| **C9** | **C6 + 3rd degree class weight ×2 (shipped)** | 0.638 | 61.1 ± 2.1% | 63.1% | **58.5%** | 4.3 (3.3) |

Why C9 and not C8? EfficientNetV2-B0 had the best overall accuracy, but on
validation it was the worst config for 3rd degree burns: 47% recall, and
9.7 of 30 confidently predicted as something else. C9 matched C6's validation
balanced accuracy with the best validation 3rd degree recall (62%) and the
fewest confident 3rd degree misses (4.3 of 30). C9 also keeps the original
MobileNetV2 preprocessing, so `model.py`'s `predict()` and `Gradcam.py` work
unchanged.

The shipped model is one C9 training run (seed 42) via `train_model.py`,
which scored 59.8% on test, inside the seed-to-seed range above.

## Known limitations

- **Small dataset.** 1,062 real training photos, only 107–279 per class.
  Test per-class counts are 33–93, so per-class numbers carry wide
  confidence intervals (see table).
- **Label conflicts in the burn data.** Up to 30 duplicate groups (78 images)
  carry more than one label, mostly 1st vs 2nd and 2nd vs 3rd degree. They
  were kept (grouped on one side of the split) rather than relabelled,
  because deciding the correct degree needs clinical judgment. They cap how
  well any model can separate burn degrees on this data.
- **Overfitting.** By the last epoch, training accuracy (on augmented
  batches) reached 76–89% across the three C9 seeds, while validation stayed
  at 57–63%. Bigger backbones did not close that gap on 3rd degree burns.
- **One data source per wound type** (two Kaggle datasets). Real user photos
  (different phones, lighting, skin tones) may score lower than this test
  set.

## Usage

Tested with Python 3.12, TensorFlow 2.21, Keras 3.15.1.

```bash
pip install tensorflow pillow numpy opencv-python flask matplotlib

python src/collate_data.py     # needs data/raw_downloads/ (not in git); rebuilds data/wound_dataset
python src/train_model.py      # grouped split -> train -> evaluate on test -> models/
python src/evaluate_model.py   # re-evaluate models/wound_model.keras on data/test
python src/model.py path/to/image.jpg
python app.py                  # web app on http://127.0.0.1:5000
```

`data/wound_dataset/` still contains the 991 `_aug` files made by earlier
versions of `collate_data.py`. `train_model.py` ignores them (they are marked
`augmented=yes` in the manifest), and re-running `collate_data.py` will
remove them.

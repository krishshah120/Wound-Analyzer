"""
collate_extra_data.py

Cleans three additional Kaggle downloads into:
  - data/extra_dataset/<class>/  : candidate EXTRA TRAINING images for the six
    model classes. train_model.py only adds the ones that are not
    near-duplicates of any existing photo, and only to the training split.
  - data/ood_dataset/<group>/     : out-of-scope photos (normal skin, chronic
    wounds) the model was never trained on, used only to measure how often it
    confidently mislabels something it should answer "unknown" to
    (python evaluate_model.py --ood-dir ../data/ood_dataset).

Expected layout (unzip each Kaggle archive into data/raw_downloads/):
  data/raw_downloads/kaggle_burn_fares/burn dataset/{1st,2nd,3rd} degree burn/
      https://www.kaggle.com/datasets/faresabbasai2022/burn-dataset (Apache 2.0)
  data/raw_downloads/kaggle_wound_ibrahim/Wound_dataset copy/<class folders>/
      https://www.kaggle.com/datasets/ibrahimfateen/wound-classification (licence: unknown)
  data/raw_downloads/kaggle_wound_yasin/Wound_dataset/<class folders>/
      https://www.kaggle.com/datasets/yasinpratomo/wound-dataset (licence: unknown)

Both output folders are gitignored because two of the sources have no stated
licence. Classes these sources label ambiguously for this model (un-graded
"Burns", "Laceration", "Stab_wound", "Ingrown_nails") are not used.

Usage:
    python collate_extra_data.py

Requires: pip install pillow opencv-python numpy
"""

import os
import csv
import shutil

import cv2
import numpy as np
from PIL import Image

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw_downloads")
EXTRA_DIR = os.path.join(PROJECT_ROOT, "data", "extra_dataset")
OOD_DIR = os.path.join(PROJECT_ROOT, "data", "ood_dataset")

SOURCES = [
    ("fares", os.path.join(RAW_DIR, "kaggle_burn_fares", "burn dataset"), {
        "1st degree burn": "burn_1st_degree",
        "2nd degree burn": "burn_2nd_degree",
        "3rd degree burn": "burn_3rd_degree",
    }),
    ("ibrahim", os.path.join(RAW_DIR, "kaggle_wound_ibrahim", "Wound_dataset copy"), {
        "Abrasions": "abrasion",
        "Bruises": "bruise",
        "Cut": "cut",
        "Normal": "ood/normal_skin",
        "Diabetic Wounds": "ood/diabetic_wound",
        "Pressure Wounds": "ood/pressure_wound",
        "Surgical Wounds": "ood/surgical_wound",
        "Venous Wounds": "ood/venous_wound",
    }),
    ("yasin", os.path.join(RAW_DIR, "kaggle_wound_yasin", "Wound_dataset"), {
        "Abrasions": "abrasion",
        "Bruises": "bruise",
        "Cut": "cut",
    }),
]

# Exposure thresholds match collate_data.py. Sharpness is measured on the
# 224x224 image the model actually sees: many of these downloads are small web
# images upscaled to 640x640, which makes full-resolution Laplacian variance
# look "blurry" even when the photo is fine. 66 is the 5th percentile of that
# 224px score among the photos collate_data.py accepted.
DARK_THRESHOLD = 40.0
BRIGHT_THRESHOLD = 215.0
SHARPNESS_THRESHOLD_224 = 66.0


def quality_problem(image):
    """Returns None if the 224x224 RGB image passes, else a short reason."""
    gray = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2GRAY)
    mean = gray.mean()
    if mean < DARK_THRESHOLD:
        return "too_dark"
    if mean > BRIGHT_THRESHOLD:
        return "too_bright"
    if cv2.Laplacian(gray, cv2.CV_64F).var() < SHARPNESS_THRESHOLD_224:
        return "too_blurry"
    return None


def mirror_aware_hashes(image, hash_size=8):
    """Difference hashes of the image and its mirror image."""
    hashes = []
    for img in (image, image.transpose(Image.FLIP_LEFT_RIGHT)):
        pixels = np.asarray(img.convert("L").resize((hash_size + 1, hash_size), Image.LANCZOS), dtype=np.int16)
        hashes.append((pixels[:, 1:] > pixels[:, :-1]).flatten())
    return hashes


def main():
    for directory in (EXTRA_DIR, OOD_DIR):
        if os.path.exists(directory):
            shutil.rmtree(directory)

    manifest_rows = []
    for source, root, mapping in SOURCES:
        if not os.path.isdir(root):
            raise FileNotFoundError(f"Missing download for '{source}': expected {root}")
        for folder, target in mapping.items():
            folder_path = os.path.join(root, folder)
            if not os.path.isdir(folder_path):
                raise FileNotFoundError(f"Expected folder not found: {folder_path}")

            is_ood = target.startswith("ood/")
            dest_dir = os.path.join(OOD_DIR, target[4:]) if is_ood else os.path.join(EXTRA_DIR, target)
            os.makedirs(dest_dir, exist_ok=True)
            kept = rejected = repeats = 0
            kept_hashes = []
            for filename in sorted(os.listdir(folder_path)):
                if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
                    continue
                with Image.open(os.path.join(folder_path, filename)) as img:
                    image = img.convert("RGB").resize((224, 224))
                # Out-of-scope photos are evaluation-only, so no quality filter
                # (real users upload imperfect photos too), but this download
                # contains mirrored copies of its own photos, which would
                # double-count results.
                if is_ood:
                    original, mirrored = mirror_aware_hashes(image)
                    if any(min((h != original).sum(), (h != mirrored).sum()) <= 4 for h in kept_hashes):
                        repeats += 1
                        continue
                    kept_hashes.append(original)
                problem = None if is_ood else quality_problem(image)
                if problem:
                    rejected += 1
                    continue
                new_name = f"extra_{source}_{target.replace('/', '_')}_{kept}.jpg"
                image.save(os.path.join(dest_dir, new_name), "JPEG")
                manifest_rows.append([new_name, target, source, os.path.join(folder, filename)])
                kept += 1
            print(f"  {source:<8} {folder:<18} -> {target:<22} kept {kept}, rejected {rejected} for quality, "
                  f"skipped {repeats} repeated/mirrored copies")

    for directory, rows in ((EXTRA_DIR, [r for r in manifest_rows if not r[1].startswith("ood/")]),
                            (OOD_DIR, [r for r in manifest_rows if r[1].startswith("ood/")])):
        with open(os.path.join(directory, "manifest.csv"), "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["filename", "class", "source", "original_path"])
            writer.writerows(rows)
    print(f"\nExtra training candidates written to {EXTRA_DIR}")
    print(f"Out-of-scope evaluation photos written to {OOD_DIR}")


if __name__ == "__main__":
    main()

"""
collate_extra_data.py

Cleans additional Kaggle downloads into:
  - data/extra_dataset/<class>/  : candidate EXTRA TRAINING images for the six
    wound classes. train_model.py only adds the ones that are not
    near-duplicates of any existing photo, and only to the training split.
  - data/ood_dataset/<group>/     : out-of-scope photos (normal skin, chronic
    wounds, rashes and other skin conditions, insect bites). train_model.py
    splits them into train/val/test as the "out_of_scope" class, which the app
    reports as "unknown".

Expected downloads in data/raw_downloads/ (folders unzipped, or the .zip itself
- a symlink to the downloaded file is fine):
  kaggle_burn_fares/burn dataset/{1st,2nd,3rd} degree burn/
      https://www.kaggle.com/datasets/faresabbasai2022/burn-dataset (Apache 2.0)
  kaggle_wound_ibrahim/Wound_dataset copy/<class folders>/
      https://www.kaggle.com/datasets/ibrahimfateen/wound-classification (licence: unknown)
  kaggle_wound_yasin/Wound_dataset/<class folders>/
      https://www.kaggle.com/datasets/yasinpratomo/wound-dataset (licence: unknown)
  kaggle_skin_disease_normal.zip
      https://www.kaggle.com/datasets/lysaapriani/skin-disease-and-normal-skin-dataset (licence: unknown)
  kaggle_bug_bites.zip
      https://www.kaggle.com/datasets/moonfallidk/bug-bite-images (Apache 2.0)
  kaggle_skin_diseases.zip
      https://www.kaggle.com/datasets/ismailpromus/skin-diseases-image-dataset (© original authors)
  kaggle_dermnet.zip
      https://www.kaggle.com/datasets/shubhamgoel27/dermnet (DermNet images, copyrighted)

Both output folders are gitignored: several sources have no stated licence or
are copyrighted. Classes the wound downloads label ambiguously for this model
(un-graded "Burns", "Laceration", "Stab_wound", "Ingrown_nails") are not used.

Usage:
    python collate_extra_data.py

Requires: pip install pillow opencv-python numpy
"""

import io
import os
import re
import csv
import random
import shutil
import zipfile
from collections import defaultdict

import cv2
import numpy as np
from PIL import Image

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw_downloads")
EXTRA_DIR = os.path.join(PROJECT_ROOT, "data", "extra_dataset")
OOD_DIR = os.path.join(PROJECT_ROOT, "data", "ood_dataset")
RANDOM_SEED = 42

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

# Out-of-scope photos read straight from the downloaded zips (two of them are
# several GB). Every image's parent folder becomes one photo group, e.g.
# "dermnet_eczema_photos"; train/test folders of the same category are merged.
OOD_ZIP_SOURCES = [
    ("skindis", os.path.join(RAW_DIR, "kaggle_skin_disease_normal.zip")),
    ("bites", os.path.join(RAW_DIR, "kaggle_bug_bites.zip")),
    ("skinimg", os.path.join(RAW_DIR, "kaggle_skin_diseases.zip")),
    ("dermnet", os.path.join(RAW_DIR, "kaggle_dermnet.zip")),
]

# The top-level folders each download is known to contain. Kaggle downloads
# all arrive named "archive (N).zip", and a different download reusing an old
# name once left one of these links pointing at a wound segmentation set: every
# photo in it would have been filed as out of scope, teaching the model that
# wounds are not wounds. So a zip whose folders don't match is refused.
EXPECTED_TOP_FOLDERS = {
    "skindis": {"Dataset Gambar Penyakit Kulit + Normal"},
    "bites": {"training", "testing", "skin_of_color_testing"},
    "skinimg": {"IMG_CLASSES"},
    "dermnet": {"train", "test"},
}

# At most this many photos per group, sampled at random. The two large
# downloads alone hold ~46,000 photos; using all of them would swamp the ~1,200
# wound photos and make training impractically slow on a CPU. Groups that fill
# the biggest gaps (normal skin, bite-free skin) are kept whole.
OUT_OF_SCOPE_PER_GROUP = 100
UNCAPPED_GROUPS = {"skindis_normal", "bites_no_bites"}

# Some skin-condition photos are actually in-scope injuries or look like them:
# sunburn, chemical/cement burns, scalded-skin syndrome, blistering (bullous)
# conditions, and bruise-like purpura or haematomas. Teaching the model those
# are "out of scope" would teach it to reject real burns and bruises.
EXCLUDE_GROUPS = {"dermnet_bullous_disease_photos"}   # blistering conditions: look like 2nd degree burn blisters
EXCLUDE_FILENAME = re.compile(r"burn|scald|bullous|blister|pemphig|hematoma|haematoma|purpura|bruis|wound|lacerat|abrasion",
                              re.IGNORECASE)

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


def is_repeat(image, kept_hashes):
    """True if image (or its mirror image) nearly matches an already kept one."""
    original, mirrored = mirror_aware_hashes(image)
    if any(min((h != original).sum(), (h != mirrored).sum()) <= 4 for h in kept_hashes):
        return True
    kept_hashes.append(original)
    return False


def slug(text):
    text = re.sub(r"^\d+\.\s*", "", text)          # "1. Eczema 1677" -> "Eczema 1677"
    text = re.sub(r"[\s-]*[\d.]+k?$", "", text)     # drop trailing counts like "1677" or "- 1.25k"
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def collate_folder_sources(manifest_rows):
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
                # Out-of-scope photos get no quality filter (real users upload
                # imperfect photos too), but this download contains mirrored
                # copies of its own photos, which would double-count results.
                if is_ood and is_repeat(image, kept_hashes):
                    repeats += 1
                    continue
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


def collate_zip_sources(manifest_rows):
    rng = random.Random(RANDOM_SEED)
    for source, zip_path in OOD_ZIP_SOURCES:
        if not os.path.exists(zip_path):
            raise FileNotFoundError(f"Missing download for '{source}': expected {zip_path}")
        with zipfile.ZipFile(zip_path) as archive:
            groups = defaultdict(list)
            excluded = 0
            for name in archive.namelist():
                parts = name.split("/")
                if (name.endswith("/") or not name.lower().endswith((".png", ".jpg", ".jpeg"))
                        or any(p.startswith(".") or p == "__MACOSX" for p in parts) or len(parts) < 2):
                    continue
                if EXCLUDE_FILENAME.search(parts[-1]):
                    excluded += 1
                    continue
                group = f"{source}_{slug(parts[-2])}"
                if group in EXCLUDE_GROUPS:
                    excluded += 1
                    continue
                groups[group].append(name)

            for group in sorted(groups):
                names = sorted(groups[group])
                if group not in UNCAPPED_GROUPS and len(names) > OUT_OF_SCOPE_PER_GROUP:
                    names = sorted(rng.sample(names, OUT_OF_SCOPE_PER_GROUP))
                dest_dir = os.path.join(OOD_DIR, group)
                os.makedirs(dest_dir, exist_ok=True)
                kept = repeats = unreadable = 0
                kept_hashes = []
                for name in names:
                    try:
                        with Image.open(io.BytesIO(archive.read(name))) as img:
                            image = img.convert("RGB").resize((224, 224))
                    except OSError:
                        unreadable += 1
                        continue
                    if is_repeat(image, kept_hashes):
                        repeats += 1
                        continue
                    new_name = f"extra_{group}_{kept}.jpg"
                    image.save(os.path.join(dest_dir, new_name), "JPEG")
                    manifest_rows.append([new_name, f"ood/{group}", source, name])
                    kept += 1
                print(f"  {group:<70} kept {kept:>3} of {len(groups[group]):>5}, skipped {repeats} repeats, {unreadable} unreadable")
            print(f"  {source}: excluded {excluded} photos that are, or look like, an in-scope injury (by filename or group)\n")


def check_downloads():
    """Raises before anything is touched if a download is missing or is not
    the dataset it should be. main() deletes the previous output first, and
    several of these downloads are no longer on this machine, so failing half
    way through would lose data that cannot be rebuilt."""
    for source, root, mapping in SOURCES:
        for folder in mapping:
            if not os.path.isdir(os.path.join(root, folder)):
                raise FileNotFoundError(f"Missing download for '{source}': expected {os.path.join(root, folder)}")
    for source, zip_path in OOD_ZIP_SOURCES:
        if not os.path.exists(zip_path):
            raise FileNotFoundError(f"Missing download for '{source}': expected {zip_path}")
        with zipfile.ZipFile(zip_path) as archive:
            top_folders = {n.split("/")[0] for n in archive.namelist() if "/" in n and not n.startswith("__MACOSX")}
        if top_folders != EXPECTED_TOP_FOLDERS[source]:
            raise ValueError(
                f"{zip_path} is not the '{source}' download: its top-level folders are {sorted(top_folders)}, "
                f"expected {sorted(EXPECTED_TOP_FOLDERS[source])}. Re-download it (see the URL at the top of this "
                f"file) and point the link at the right file. Nothing has been changed."
            )


def main():
    check_downloads()
    for directory in (EXTRA_DIR, OOD_DIR):
        if os.path.exists(directory):
            shutil.rmtree(directory)

    manifest_rows = []
    collate_folder_sources(manifest_rows)
    print()
    collate_zip_sources(manifest_rows)

    for directory, rows in ((EXTRA_DIR, [r for r in manifest_rows if not r[1].startswith("ood/")]),
                            (OOD_DIR, [r for r in manifest_rows if r[1].startswith("ood/")])):
        with open(os.path.join(directory, "manifest.csv"), "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["filename", "class", "source", "original_path"])
            writer.writerows(rows)
    print(f"Extra training candidates written to {EXTRA_DIR}")
    print(f"Out-of-scope photos written to {OOD_DIR}")


if __name__ == "__main__":
    main()

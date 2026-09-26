"""
train_model.py

1. Reads the REAL (non-augmented) images listed in
   data/wound_dataset/manifest.csv, groups together every image that comes
   from the same source photo - including copies saved under different
   filenames, cropped, rotated, mirrored or re-watermarked - and splits whole
   groups into
   data/train/<class>/, data/val/<class>/ and data/test/<class>/. No source
   photo ever appears in more than one split, no two images in different
   splits are visually near-identical, and all three splits hold real photos
   only.
2. Trains the classifier defined in model.py on data/train:
     - random augmentation (flip/rotate/zoom/shift/brightness/contrast) is
       applied on the fly to TRAINING batches only, so every epoch sees new
       variations and validation/test are never augmented;
     - class weights compensate for the uneven number of photos per class;
     - label smoothing (LABEL_SMOOTHING) keeps the model from becoming
       overconfident on noisy, partly contradictory labels;
     - phase 1 trains only the classifier head on the frozen MobileNetV2
       base (width BACKBONE_ALPHA), phase 2 fine-tunes the top
       FINETUNE_LAYERS layers of the base at a low learning rate.
   Both phases early-stop on data/val. data/test is not looked at until the
   final evaluation.
   Extra photos from additional Kaggle downloads (data/extra_dataset) are
   added to train only, and out-of-scope photos (data/ood_dataset: normal
   skin, chronic wounds) are split into all three as an extra
   "out_of_scope" class that predict() reports as "unknown".
3. Saves the trained model, class label list and test metrics to models/.
   With --gate it saves the model as the out-of-scope gate
   (models/out_of_scope_gate.keras) instead, leaving the wound classifier,
   class list and metrics untouched, and skips the test evaluation: the gate
   is only judged together with the classifier (see model.decide()).

This file only handles data splitting and training. The model architecture
and prediction logic live in model.py - see that file for running
predictions once a model has been trained.

Usage:
    python train_model.py          # wound classifier
    python train_model.py --gate   # out-of-scope gate

Requires: pip install tensorflow pillow numpy
"""

import os
import csv
import argparse
import json
import random
import shutil
import stat
import time
from collections import Counter, defaultdict

import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras import layers
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input

from model import build_model, unfreeze_top_layers, IMG_SIZE, MODEL_DIR, MODEL_PATH, CLASS_NAMES_PATH, GATE_MODEL_PATH, OUT_OF_SCOPE_CLASS
from evaluate_model import evaluate, print_report

# ---------------------------------------------------------------------------
# Paths - anchored to this script's location
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)          # .../Wound-Analyzer

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
WOUND_DATASET_DIR = os.path.join(DATA_DIR, "wound_dataset")
TRAIN_DIR = os.path.join(DATA_DIR, "train")          # sibling of wound_dataset
VAL_DIR = os.path.join(DATA_DIR, "val")              # sibling of wound_dataset
TEST_DIR = os.path.join(DATA_DIR, "test")            # sibling of wound_dataset
MANIFEST_PATH = os.path.join(WOUND_DATASET_DIR, "manifest.csv")
SPLIT_MANIFEST_PATH = os.path.join(DATA_DIR, "split_manifest.csv")
EXTRA_DATASET_DIR = os.path.join(DATA_DIR, "extra_dataset")      # built by collate_extra_data.py
OUT_OF_SCOPE_DATASET_DIR = os.path.join(DATA_DIR, "ood_dataset")  # built by collate_extra_data.py
METRICS_PATH = os.path.join(MODEL_DIR, "metrics.json")

BATCH_SIZE = 32
TEST_FRACTION = 0.2
VAL_FRACTION = 0.15
RANDOM_SEED = 42

# Two real images whose 64-bit difference hashes differ in at most this many
# bits are treated as the same source photo and kept in the same split. The
# burn dataset contains many exact re-uploads under different filenames
# (sometimes with different degree labels), which filename-based grouping
# alone would leak across the split. Grouping unrelated look-alikes together
# is harmless, so this errs on the generous side.
DUPLICATE_HASH_DISTANCE = 6

# Hashing misses copies of a photo that were cropped, rotated, mirrored or
# re-watermarked, and this data contains many. So images are also compared by
# visual similarity (see similarity_matrix). Pairs at or above
# SAME_PHOTO_SIMILARITY are merged into one group. Merging at a lower
# threshold chains look-alike photos into huge groups, so instead no image may
# have a match at or above CROSS_SPLIT_SIMILARITY in a different split (see
# separate_similar_across_splits). Both thresholds were set by looking at
# sample pairs: pairs >= 0.80 were mostly the same photo, pairs below 0.75
# were different photos.
SAME_PHOTO_SIMILARITY = 0.85
CROSS_SPLIT_SIMILARITY = 0.80

# Extra training images from additional Kaggle downloads (see
# collate_extra_data.py). They are only ever added to TRAIN, and only if they
# are not near-duplicates of an existing photo: these downloads re-upload much
# of the same web-scraped data, including rotated and cropped copies of
# test photos. The threshold against val/test is stricter than the one
# against train, because a miss there would inflate the test score. On the
# validation split (3 seeds each) adding them raised balanced accuracy from
# 0.615 to 0.648.
USE_EXTRA_TRAINING_DATA = True
EXTRA_MAX_SIMILARITY_TO_VAL_TEST = 0.75
EXTRA_MAX_SIMILARITY_TO_TRAIN = 0.80

# A model trained only on the six wound classes gave a confident wound label
# to about two thirds of photos that are none of them (normal skin, diabetic,
# pressure, surgical and venous wounds), because the confidence threshold alone
# cannot recognise something it was never shown. So those photos are added as
# a seventh class, OUT_OF_SCOPE_CLASS, which predict() turns into "unknown".
# On the validation split (3 seeds each) this cut out-of-scope photos
# confidently labelled from 66.7% to 8.5%, with in-scope balanced accuracy
# 0.628 vs 0.622, while 3.6% of real wound photos were rejected as out of scope.
USE_OUT_OF_SCOPE_CLASS = True

# The chronic-wound photos include several photos of the same patient or wound
# that score 0.75-0.80 (checked by eye: pairs at 0.80-0.81 were often the same
# wound). At the wound photos' 0.80 threshold those ended up on both sides of
# the split and flattered the out-of-scope test result, so out-of-scope photos
# use a stricter threshold.
OUT_OF_SCOPE_CROSS_SPLIT_SIMILARITY = 0.75

# Training schedule. See README.md for the experiments behind these choices.
HEAD_EPOCHS = 15
HEAD_LEARNING_RATE = 1e-3
FINETUNE_LAYERS = 100
FINETUNE_EPOCHS = 40
FINETUNE_LEARNING_RATE = 1e-5

# Wider MobileNetV2 (width multiplier 1.4) with label smoothing 0.1. On the
# validation split (6 seeds each, compared with width 1.0 and no smoothing):
# balanced accuracy 0.602 -> 0.662, out-of-scope photos confidently labelled
# 26.3% -> 7.6%, 3rd degree burns shown a wrong wound label 3.3 -> 2.7 of 31.
# The wider network alone was more accurate but more often confidently wrong
# about 3rd degree burns; label smoothing reins that in.
BACKBONE_ALPHA = 1.4
LABEL_SMOOTHING = 0.1

# Every training photo is a tight crop, but a phone photo of someone's own arm
# is not: the wound fills much less of the frame, and the model was brittle to
# that (on validation, wound photos given an answer fell 46.2% -> 19.5% when
# the photo filled half the frame). Zooming OUT during training (positive
# RandomZoom) is what teaches the model to cope: 0.5 means the photo may be
# shrunk to two thirds of the frame.
ZOOM_OUT = 0.0

# The smallest part of the frame a training photo may be shrunk into by
# RandomFrameShrink (1.0 = off). RandomZoom cannot go below half the frame.
MIN_FRAME_FILL = 1.0

# Extra multiplier applied on top of the balanced class weights. A missed
# 3rd degree burn is the most consequential error this model can make (the
# app routes 3rd degree to "call emergency services"), so it is weighted up.
# On the validation split this raised 3rd degree recall and cut 3rd-degree-
# predicted-as-1st-degree errors without lowering balanced accuracy.
CLASS_WEIGHT_MULTIPLIERS = {"burn_3rd_degree": 2.0}

# Just a heads-up in the console output - doesn't stop training.
MIN_IMAGES_PER_CLASS_WARNING = 20


def robust_rmtree(path, retries=4, delay_seconds=1.5):
    """
    Deletes a directory tree, retrying on Windows PermissionError (WinError
    5), which almost always means a file was transiently locked - e.g. by
    OneDrive syncing, an image viewer, or File Explorer's preview pane -
    rather than an actual permissions problem. Also clears the read-only
    flag if that's what's blocking deletion.
    """
    def clear_readonly_and_retry(func, target_path, exc_info):
        try:
            os.chmod(target_path, stat.S_IWRITE)
            func(target_path)
        except Exception:
            pass  # let the outer retry loop handle it

    last_error = None
    for attempt in range(1, retries + 1):
        try:
            shutil.rmtree(path, onerror=clear_readonly_and_retry)
            return
        except PermissionError as e:
            last_error = e
            if attempt < retries:
                print(
                    f"  Could not delete {path} yet (attempt {attempt}/{retries}) - "
                    f"a file may be in use (e.g. OneDrive syncing, an image viewer, "
                    f"or File Explorer previewing a file inside it). Retrying in "
                    f"{delay_seconds}s..."
                )
                time.sleep(delay_seconds)

    raise PermissionError(
        f"Could not delete {path} after {retries} attempts. Close any program that "
        f"might have a file open inside this folder (File Explorer, an image viewer, "
        f"VS Code), pause OneDrive syncing if this project is inside a OneDrive folder, "
        f"then run the script again.\nOriginal error: {last_error}"
    )


def class_name_for(row):
    """Manifest rows store burns as class=burn + burn_degree=1st_degree; the
    model's label (and folder name) for those is burn_1st_degree."""
    if row["class"] == "burn":
        return f"burn_{row['burn_degree']}"
    return row["class"]


def dataset_path_for(row):
    if row["class"] == "burn":
        return os.path.join(WOUND_DATASET_DIR, "burn", row["burn_degree"], row["filename"])
    return os.path.join(WOUND_DATASET_DIR, row["class"], row["filename"])


def load_real_images():
    """
    Returns the manifest rows for real (non-augmented) photos. Augmented
    files already sitting in wound_dataset from older collate_data.py runs -
    which augmented BEFORE splitting, so copies of one photo ended up in both
    train and test - are deliberately ignored. Augmentation now happens during
    training, on training batches only (see make_augmenter).
    """
    with open(MANIFEST_PATH, newline="") as f:
        rows = list(csv.DictReader(f))

    real = [r for r in rows if r["augmented"] == "no"]
    ignored = len(rows) - len(real)
    if ignored:
        print(f"Ignoring {ignored} pre-generated augmented files listed in the manifest.")

    missing = [dataset_path_for(r) for r in real if not os.path.exists(dataset_path_for(r))]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} images listed in {MANIFEST_PATH} are missing on disk, "
            f"e.g. {missing[:3]}. Re-run collate_data.py."
        )
    return real


def difference_hash(image_path, hash_size=8, mirrored=False):
    """64-bit perceptual hash: compares each pixel of a tiny grayscale copy
    to its right-hand neighbour. Re-saved or re-compressed copies of the same
    photo produce identical or near-identical hashes."""
    with Image.open(image_path) as img:
        gray = img.convert("L")
        if mirrored:
            gray = gray.transpose(Image.FLIP_LEFT_RIGHT)
        small = gray.resize((hash_size + 1, hash_size), Image.LANCZOS)
    pixels = np.asarray(small, dtype=np.int16)
    return (pixels[:, 1:] > pixels[:, :-1]).flatten()


def image_views(img):
    """12 views of a 224x224 image: the original, a centre crop, and the centre
    crop rotated by -30/-15/+15/+30 degrees, each also mirrored. Comparing
    views catches copies of a photo that were cropped, rotated or flipped."""
    views = []
    for mirrored in (False, True):
        base = img.transpose(Image.FLIP_LEFT_RIGHT) if mirrored else img
        views.append(np.asarray(base))
        views.append(np.asarray(base.crop((34, 34, 190, 190)).resize(IMG_SIZE)))
        for angle in (-30, -15, 15, 30):
            rotated = base.rotate(angle, resample=Image.BILINEAR)
            views.append(np.asarray(rotated.crop((34, 34, 190, 190)).resize(IMG_SIZE)))
    return views


def view_features(paths):
    """L2-normalised ImageNet MobileNetV2 features for the 12 views of every
    image. Returns an (n, 12, d) array."""
    extractor = MobileNetV2(input_shape=IMG_SIZE + (3,), include_top=False, weights="imagenet", pooling="avg")
    n_views = 12
    features = np.zeros((len(paths), n_views, extractor.output_shape[-1]), dtype=np.float32)
    for start in range(0, len(paths), 16):
        chunk = paths[start:start + 16]
        views = []
        for path in chunk:
            with Image.open(path) as img:
                views.extend(image_views(img.convert("RGB").resize(IMG_SIZE)))
        f = extractor.predict(preprocess_input(np.stack(views).astype("float32")), verbose=0)
        f /= np.linalg.norm(f, axis=1, keepdims=True)
        features[start:start + len(chunk)] = f.reshape(len(chunk), n_views, -1)
    return features


def cross_similarity(features_a, features_b):
    """
    Visual similarity between every image in a and every image in b: cosine
    similarity taking the best match between one image's original/centre-crop
    view and any of the other image's 12 views, in both directions.
    Returns an (len(a), len(b)) array.
    """
    sim = np.zeros((len(features_a), len(features_b)), dtype=np.float32)
    for anchor_view in (0, 1):  # original and centre crop, against every view
        for start in range(0, len(features_a), 128):
            block = np.einsum("id,jvd->ijv", features_a[start:start + 128, anchor_view], features_b).max(axis=2)
            sim[start:start + 128] = np.maximum(sim[start:start + 128], block)
        for start in range(0, len(features_b), 128):
            block = np.einsum("jd,ivd->ijv", features_b[start:start + 128, anchor_view], features_a).max(axis=2)
            sim[:, start:start + 128] = np.maximum(sim[:, start:start + 128], block)
    return sim


def similarity_matrix(paths):
    """Visual similarity between every pair of images in paths (see
    cross_similarity). Returns (symmetric (n, n) array with zeros on the
    diagonal, the view features)."""
    features = view_features(paths)
    sim = cross_similarity(features, features)
    np.fill_diagonal(sim, 0)
    return sim, features


def group_by_source(rows, sim):
    """
    Returns a source-group id for every row (parallel list), such that all
    copies of one source photo share a group. Each real manifest row is its
    own source photo; rows are merged when their difference hashes are nearly
    identical or their visual similarity is >= SAME_PHOTO_SIMILARITY.
    """
    hashes = np.array([difference_hash(dataset_path_for(r)) for r in rows])
    parent = list(range(len(rows)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(rows)):
        distances = (hashes[i + 1:] != hashes[i]).sum(axis=1)
        same_photo = (distances <= DUPLICATE_HASH_DISTANCE) | (sim[i, i + 1:] >= SAME_PHOTO_SIMILARITY)
        for offset in np.nonzero(same_photo)[0]:
            parent[find(i)] = find(i + 1 + int(offset))

    # Readable, stable ids: the alphabetically first filename in each group.
    members = defaultdict(list)
    for i, row in enumerate(rows):
        members[find(i)].append(row["filename"])
    root_to_id = {root: min(names) for root, names in members.items()}
    return [root_to_id[find(i)] for i in range(len(rows))]


def primary_class(classes):
    return sorted(Counter(classes).items(), key=lambda kv: (-kv[1], kv[0]))[0][0]


def assign_splits(rows, group_ids):
    """
    Assigns whole groups to train/val/test, stratified by class: each group
    is filed under its most common class, and within each class the shuffled
    groups are dealt to test, then val, until each holds its fraction of that
    class's images. Everything left goes to train. Returns {group_id: split}.
    """
    group_classes = defaultdict(list)
    for row, gid in zip(rows, group_ids):
        group_classes[gid].append(class_name_for(row))

    groups_by_class = defaultdict(list)
    for gid, classes in group_classes.items():
        groups_by_class[primary_class(classes)].append(gid)

    rng = random.Random(RANDOM_SEED)
    split_of = {}
    for class_name in sorted(groups_by_class):
        gids = sorted(groups_by_class[class_name])
        rng.shuffle(gids)
        total = sum(len(group_classes[g]) for g in gids)
        filled = {"test": 0, "val": 0}
        targets = [("test", total * TEST_FRACTION), ("val", total * VAL_FRACTION)]
        for gid in gids:
            split_of[gid] = "train"
            for split, target in targets:
                if filled[split] < target:
                    split_of[gid] = split
                    filled[split] += len(group_classes[gid])
                    break
    return split_of


def separate_similar_across_splits(classes, group_ids, split_of, sim, threshold=CROSS_SPLIT_SIMILARITY):
    """
    Groups are only merged at SAME_PHOTO_SIMILARITY, because merging at a
    lower threshold chains look-alike photos (e.g. sunburnt skin) into groups
    of hundreds. This closes that gap without chaining (classes and group_ids
    are parallel lists, one entry per image):
      1. while a test or val image has a match >= threshold in a split it
         could leak into (test->train, val->train, test->val), its whole
         group moves into that split;
      2. each class's test and val splits are then topped back up to their
         target size with train groups that have no such match outside
         themselves.
    Modifies and returns split_of.
    """
    members = defaultdict(list)
    for i, g in enumerate(group_ids):
        members[g].append(i)
    close = sim >= threshold

    def in_split(split):
        return np.array([split_of[g] == split for g in group_ids])

    moved = Counter()
    changed = True
    while changed:
        changed = False
        for source, destination in (("test", "train"), ("val", "train"), ("test", "val")):
            in_source, in_destination = in_split(source), in_split(destination)
            for i in np.nonzero(in_source & close[:, in_destination].any(axis=1))[0]:
                group = group_ids[i]
                if split_of[group] == source:
                    split_of[group] = destination
                    moved[(source, destination)] += len(members[group])
                    changed = True

    topped_up = Counter()
    rng = random.Random(RANDOM_SEED)
    class_totals = Counter(classes)
    group_primary = {g: primary_class([classes[i] for i in m]) for g, m in members.items()}
    for split, fraction, must_not_match in (("test", TEST_FRACTION, ("train", "val")),
                                             ("val", VAL_FRACTION, ("train", "test"))):
        for class_name in sorted(class_totals):
            target = class_totals[class_name] * fraction
            candidates = sorted(g for g in members if split_of[g] == "train" and group_primary[g] == class_name)
            rng.shuffle(candidates)
            for group in candidates:
                have = sum(1 for i, c in enumerate(classes) if c == class_name and split_of[group_ids[i]] == split)
                if have >= target:
                    break
                others = [j for j, g in enumerate(group_ids) if g != group and split_of[g] in must_not_match]
                if not close[np.ix_(members[group], others)].any():
                    split_of[group] = split
                    topped_up[split] += len(members[group])

    print(f"  moved out of test/val for near-matches across splits: {dict(moved) or 'none'}")
    print(f"  topped back up with unmatched train groups: {dict(topped_up) or 'none'}")
    return split_of


def verify_split(split_rows, sim):
    """Raises if any source group appears in two splits, or if any two images
    in different splits are visually similar >= CROSS_SPLIT_SIMILARITY."""
    splits_per_group = defaultdict(set)
    for split, _class_name, _filename, group_id, _source in split_rows:
        splits_per_group[group_id].add(split)
    crossing = {g: sorted(s) for g, s in splits_per_group.items() if len(s) > 1}
    if crossing:
        raise RuntimeError(
            f"{len(crossing)} source groups appear in more than one split, e.g. "
            f"{list(crossing.items())[:3]}. The split would leak - refusing to continue."
        )

    splits = np.array([r[0] for r in split_rows])
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        worst = sim[np.ix_(splits == a, splits == b)].max()
        if worst >= CROSS_SPLIT_SIMILARITY:
            raise RuntimeError(
                f"Images in {a} and {b} have visual similarity {worst:.3f} >= {CROSS_SPLIT_SIMILARITY}. "
                f"The split would leak - refusing to continue."
            )
        print(f"  highest {a}/{b} similarity: {worst:.3f}")


def add_extra_training_images(split_rows, rows, features):
    """
    Adds images from data/extra_dataset to data/train, skipping any that are
    near-duplicates of an existing val/test photo (similarity >=
    EXTRA_MAX_SIMILARITY_TO_VAL_TEST or near-identical hash, mirrored or not),
    of an existing train photo (>= EXTRA_MAX_SIMILARITY_TO_TRAIN), or of an
    extra image already added. Extra images that duplicate one another under
    different labels are all skipped. Appends to split_rows and returns
    ({class_name: number_added}, view features of the added images in the
    order they were appended).
    """
    manifest_path = os.path.join(EXTRA_DATASET_DIR, "manifest.csv")
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(
            f"{manifest_path} not found. Run collate_extra_data.py first, or set "
            f"USE_EXTRA_TRAINING_DATA = False to train on the original data only."
        )
    with open(manifest_path, newline="") as f:
        extra = list(csv.DictReader(f))
    extra_paths = [os.path.join(EXTRA_DATASET_DIR, r["class"], r["filename"]) for r in extra]

    print(f"\nChecking {len(extra)} extra training candidates against every existing photo...")
    splits = np.array([r[0] for r in split_rows])
    extra_features = view_features(extra_paths)
    sim = cross_similarity(extra_features, features)
    existing_hashes = np.array([difference_hash(dataset_path_for(r)) for r in rows])
    extra_hashes = np.array([difference_hash(p) for p in extra_paths])
    extra_hashes_mirrored = np.array([difference_hash(p, mirrored=True) for p in extra_paths])
    hash_distance = np.minimum((extra_hashes[:, None, :] != existing_hashes[None, :, :]).sum(-1),
                               (extra_hashes_mirrored[:, None, :] != existing_hashes[None, :, :]).sum(-1))
    near_hash = hash_distance <= DUPLICATE_HASH_DISTANCE
    is_eval = splits != "train"
    near_eval = (sim[:, is_eval] >= EXTRA_MAX_SIMILARITY_TO_VAL_TEST).any(1) | near_hash[:, is_eval].any(1)
    near_train = (sim[:, ~is_eval] >= EXTRA_MAX_SIMILARITY_TO_TRAIN).any(1) | near_hash[:, ~is_eval].any(1)
    candidates = [i for i in range(len(extra)) if not near_eval[i] and not near_train[i]]

    # Greedy de-duplication among the remaining extras (no chaining).
    within = cross_similarity(extra_features[candidates], extra_features[candidates])
    kept, duplicate_of = [], {}
    for position, i in enumerate(candidates):
        matches = [k for k in kept if within[position, candidates.index(k)] >= SAME_PHOTO_SIMILARITY
                   or min((extra_hashes[i] != extra_hashes[k]).sum(), (extra_hashes_mirrored[i] != extra_hashes[k]).sum()) <= 4]
        if matches:
            duplicate_of[i] = matches[0]
        else:
            kept.append(i)
    conflicted = {k for i, k in duplicate_of.items() if extra[i]["class"] != extra[k]["class"]}

    added = Counter()
    for i in kept:
        if i in conflicted:
            continue
        row = extra[i]
        shutil.copy2(extra_paths[i], os.path.join(TRAIN_DIR, row["class"], row["filename"]))
        split_rows.append(["train", row["class"], row["filename"], f"extra:{row['filename']}",
                           f"{row['source']}:{row['original_path']}"])
        added[row["class"]] += 1

    added_idx = [i for i in kept if i not in conflicted]
    if added_idx:
        worst = sim[np.ix_(added_idx, np.nonzero(is_eval)[0])].max()
        if worst >= EXTRA_MAX_SIMILARITY_TO_VAL_TEST:
            raise RuntimeError(f"An added extra image has similarity {worst:.3f} to a val/test photo - refusing to continue.")
        print(f"  highest similarity of an added extra image to any val/test photo: {worst:.3f}")
    print(f"  skipped {int(near_eval.sum())} near val/test photos, {int((near_train & ~near_eval).sum())} near train photos, "
          f"{len(duplicate_of)} repeats among extras, {len(conflicted)} with conflicting labels among extras")
    print(f"  added to train: {dict(sorted(added.items()))}")
    return added, extra_features[added_idx]


def add_out_of_scope_images(split_rows, split_paths, split_features):
    """
    Adds data/ood_dataset photos as OUT_OF_SCOPE_CLASS to all three splits.
    They are grouped and split like the wound photos: near-duplicates stay
    together, and a val/test group with a match >= CROSS_SPLIT_SIMILARITY in
    another split is moved into that split. Photos that resemble any wound
    photo (split_paths/split_features, parallel to split_rows) are skipped -
    some of these downloads re-use photos labelled as burns or bruises here. The cross-split threshold is OUT_OF_SCOPE_CROSS_SPLIT_SIMILARITY
    (stricter than for wound photos). Appends to split_rows and returns
    {split: number_added}.
    """
    manifest_path = os.path.join(OUT_OF_SCOPE_DATASET_DIR, "manifest.csv")
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(
            f"{manifest_path} not found. Run collate_extra_data.py first, or set "
            f"USE_OUT_OF_SCOPE_CLASS = False to train a six-class model."
        )
    with open(manifest_path, newline="") as f:
        ood = list(csv.DictReader(f))
    photo_types = [r["class"].split("/", 1)[1] for r in ood]   # e.g. "ood/diabetic_wound" -> "diabetic_wound"
    paths = [os.path.join(OUT_OF_SCOPE_DATASET_DIR, t, r["filename"]) for t, r in zip(photo_types, ood)]

    print(f"\nSplitting {len(ood)} out-of-scope photos...")
    sim, features = similarity_matrix(paths)
    hashes = np.array([difference_hash(p) for p in paths])
    hashes_mirrored = np.array([difference_hash(p, mirrored=True) for p in paths])

    parent = list(range(len(ood)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(ood)):
        distances = (hashes[i + 1:] != hashes[i]).sum(axis=1)
        same_photo = (distances <= DUPLICATE_HASH_DISTANCE) | (sim[i, i + 1:] >= SAME_PHOTO_SIMILARITY)
        for offset in np.nonzero(same_photo)[0]:
            parent[find(i)] = find(i + 1 + int(offset))
    group_ids = [find(i) for i in range(len(ood))]
    members = defaultdict(list)
    for i, g in enumerate(group_ids):
        members[g].append(i)

    rng = random.Random(RANDOM_SEED)
    groups_by_type = defaultdict(list)
    for g, m in members.items():
        groups_by_type[primary_class([photo_types[i] for i in m])].append(g)
    split_of = {}
    for photo_type in sorted(groups_by_type):
        gids = sorted(groups_by_type[photo_type], key=lambda g: min(members[g]))
        rng.shuffle(gids)
        total = sum(len(members[g]) for g in gids)
        filled = {"test": 0, "val": 0}
        for g in gids:
            split_of[g] = "train"
            for split, fraction in (("test", TEST_FRACTION), ("val", VAL_FRACTION)):
                if filled[split] < total * fraction:
                    split_of[g] = split
                    filled[split] += len(members[g])
                    break

    split_of = separate_similar_across_splits(photo_types, group_ids, split_of, sim,
                                              threshold=OUT_OF_SCOPE_CROSS_SPLIT_SIMILARITY)
    splits = [str(split_of[g]) for g in group_ids]
    split_array = np.array(splits)

    # Skip out-of-scope photos that resemble ANY wound photo, in any split:
    # across splits that would leak, and within a split it would teach the
    # model contradictory labels for the same picture.
    wound_hashes = np.array([difference_hash(p) for p in split_paths])
    to_wounds = cross_similarity(features, split_features)
    skip = (to_wounds >= CROSS_SPLIT_SIMILARITY).any(axis=1)
    for start in range(0, len(ood), 256):   # chunked: the full hash comparison would need gigabytes
        chunk = slice(start, start + 256)
        distance = np.minimum((hashes[chunk, None, :] != wound_hashes[None, :, :]).sum(-1),
                              (hashes_mirrored[chunk, None, :] != wound_hashes[None, :, :]).sum(-1))
        skip[chunk] |= (distance <= DUPLICATE_HASH_DISTANCE).any(axis=1)

    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        worst = sim[np.ix_((split_array == a) & ~skip, (split_array == b) & ~skip)].max()
        print(f"  highest out-of-scope {a}/{b} similarity: {worst:.3f}")
        if worst >= OUT_OF_SCOPE_CROSS_SPLIT_SIMILARITY:
            raise RuntimeError(f"Out-of-scope photos in {a} and {b} have similarity {worst:.3f} - refusing to continue.")

    split_dirs = {"train": TRAIN_DIR, "val": VAL_DIR, "test": TEST_DIR}
    added = Counter()
    for i, row in enumerate(ood):
        if skip[i]:
            continue
        dest_dir = os.path.join(split_dirs[splits[i]], OUT_OF_SCOPE_CLASS)
        os.makedirs(dest_dir, exist_ok=True)
        shutil.copy2(paths[i], os.path.join(dest_dir, row["filename"]))
        group_name = os.path.basename(paths[min(members[group_ids[i]])])
        split_rows.append([splits[i], OUT_OF_SCOPE_CLASS, row["filename"], f"out_of_scope:{group_name}",
                           f"{row['source']}:{row['original_path']}"])
        added[splits[i]] += 1
    print(f"  {len(members)} groups; skipped {int(skip.sum())} that resemble a wound photo")
    print(f"  added: {dict(added)}")
    return added


def split_and_copy(rows):
    """Groups real images by source photo, splits the groups into
    data/train, data/val and data/test with no near-duplicates across
    splits, and writes data/split_manifest.csv recording where every file
    went."""
    for directory in (TRAIN_DIR, VAL_DIR, TEST_DIR):
        if os.path.exists(directory):
            robust_rmtree(directory)

    print("Comparing every pair of images (hash + rotation/crop/mirror-robust features)...")
    sim, features = similarity_matrix([dataset_path_for(r) for r in rows])
    group_ids = group_by_source(rows, sim)
    print(f"  {len(rows)} real images -> {len(set(group_ids))} source groups")
    split_of = assign_splits(rows, group_ids)
    split_of = separate_similar_across_splits([class_name_for(r) for r in rows], group_ids, split_of, sim)

    split_dirs = {"train": TRAIN_DIR, "val": VAL_DIR, "test": TEST_DIR}
    split_rows = []
    counts = defaultdict(Counter)
    for row, group_id in zip(rows, group_ids):
        split = split_of[group_id]
        class_name = class_name_for(row)
        dest_dir = os.path.join(split_dirs[split], class_name)
        os.makedirs(dest_dir, exist_ok=True)
        shutil.copy2(dataset_path_for(row), os.path.join(dest_dir, row["filename"]))

        split_rows.append([split, class_name, row["filename"], group_id, row["source_filename"]])
        counts[class_name][split] += 1

    for class_name, c in counts.items():
        empty = [s for s in split_dirs if c[s] == 0]
        if empty:
            raise RuntimeError(f"Class '{class_name}' has no images in split(s) {empty}.")
    verify_split(split_rows, sim)

    split_paths = [dataset_path_for(r) for r in rows]
    split_features = features
    added = Counter()
    if USE_EXTRA_TRAINING_DATA:
        added, extra_features = add_extra_training_images(split_rows, rows, features)
        split_paths += [os.path.join(TRAIN_DIR, r[1], r[2]) for r in split_rows[len(rows):]]
        split_features = np.concatenate([features, extra_features])
    out_of_scope = add_out_of_scope_images(split_rows, split_paths, split_features) if USE_OUT_OF_SCOPE_CLASS else Counter()

    print("\nSplit (real photos grouped by source photo; extra images go to train only):\n")
    for class_name in sorted(counts):
        c = counts[class_name]
        total = sum(c.values())
        warning = "  <-- LOW: consider gathering more data for this class" if total < MIN_IMAGES_PER_CLASS_WARNING else ""
        print(f"  {class_name}: {c['train']} train (+{added[class_name]} extra) / {c['val']} val / {c['test']} test "
              f"({total} original){warning}")
    if USE_OUT_OF_SCOPE_CLASS:
        print(f"  {OUT_OF_SCOPE_CLASS}: {out_of_scope['train']} train / {out_of_scope['val']} val / {out_of_scope['test']} test")
    print()

    with open(SPLIT_MANIFEST_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["split", "class", "filename", "source_group", "source_filename"])
        writer.writerows(sorted(split_rows))
    print(f"Split manifest written to {SPLIT_MANIFEST_PATH}\n")


class RandomFrameShrink(layers.Layer):
    """Puts the photo, shrunk to a random part of the frame, on a stretched
    copy of itself - the training counterpart of standing further back. Keras's
    RandomZoom cannot shrink past half the frame (its factor stops at 1.0), so
    this layer handles the smaller sizes. The surround is deliberately not
    blurred: the evaluation sets use a blurred surround, and training on the
    same trick would measure the trick rather than the model."""

    def __init__(self, min_fill, **kwargs):
        super().__init__(**kwargs)
        self.min_fill = min_fill

    def call(self, inputs, training=False):
        if not training or self.min_fill >= 1.0:
            return inputs
        size = tf.shape(inputs)[1]
        fill = tf.random.uniform([], self.min_fill, 1.0)
        inner = tf.cast(tf.round(tf.cast(size, tf.float32) * fill), tf.int32)
        pad = size - inner
        top = tf.random.uniform([], 0, pad + 1, dtype=tf.int32)
        left = tf.random.uniform([], 0, pad + 1, dtype=tf.int32)
        padding = [[0, 0], [top, pad - top], [left, pad - left], [0, 0]]
        small = tf.image.resize(inputs, (inner, inner))
        mask = tf.pad(tf.ones_like(small), padding)
        return tf.pad(small, padding) + inputs * (1.0 - mask)


def make_augmenter(zoom_out=ZOOM_OUT, min_frame_fill=MIN_FRAME_FILL):
    """Realistic, mild augmentations applied to raw [0, 255] training images
    on the fly - no color inversion or extreme distortion that wouldn't be
    plausible for a real wound photo. zoom_out is how far the image may be
    zoomed OUT (0.5 = the photo shrunk to two thirds of the frame), which
    teaches the model to cope with wounds photographed from further away."""
    return tf.keras.Sequential([
        RandomFrameShrink(min_frame_fill),
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.05, fill_mode="reflect"),        # up to +-18 degrees
        layers.RandomZoom((-0.15, zoom_out), fill_mode="reflect"),   # zoom in up to 15%, out by zoom_out
        layers.RandomTranslation(0.05, 0.05, fill_mode="reflect"),
        layers.RandomBrightness(0.15, value_range=(0, 255)),
        layers.RandomContrast(0.15),
    ], name="augment")


def build_datasets():
    def load(directory, shuffle):
        return tf.keras.utils.image_dataset_from_directory(
            directory,
            image_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            label_mode="categorical",
            shuffle=shuffle,
            seed=RANDOM_SEED,
        )

    train_ds = load(TRAIN_DIR, shuffle=True)
    val_ds = load(VAL_DIR, shuffle=False)

    class_names = train_ds.class_names  # alphabetical order - matches label indices
    if val_ds.class_names != class_names:
        raise RuntimeError(f"Train classes {class_names} != val classes {val_ds.class_names}")

    augmenter = make_augmenter()
    train_ds = train_ds.map(
        lambda x, y: (tf.clip_by_value(augmenter(x, training=True), 0, 255), y),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    # MobileNetV2 expects inputs scaled to [-1, 1] rather than raw [0, 255]
    train_ds = train_ds.map(lambda x, y: (preprocess_input(x), y)).prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.map(lambda x, y: (preprocess_input(x), y)).prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds, class_names


def compute_class_weights(class_names):
    """Balanced weights (total / (n_classes * class_count)) so each class
    contributes equally to the loss despite uneven photo counts, times any
    CLASS_WEIGHT_MULTIPLIERS."""
    counts = np.array([len(os.listdir(os.path.join(TRAIN_DIR, c))) for c in class_names], dtype=float)
    weights = counts.sum() / (len(class_names) * counts)
    for class_name, multiplier in CLASS_WEIGHT_MULTIPLIERS.items():
        weights[class_names.index(class_name)] *= multiplier
    return {i: float(w) for i, w in enumerate(weights)}


def train(gate=False):
    tf.keras.utils.set_random_seed(RANDOM_SEED)

    print("Loading real images from the manifest...\n")
    rows = load_real_images()
    split_and_copy(rows)

    train_ds, val_ds, class_names = build_datasets()
    print(f"Classes (in label order): {class_names}\n")

    class_weight = compute_class_weights(class_names)
    print("Class weights: " + ", ".join(f"{class_names[i]}={w:.2f}" for i, w in class_weight.items()) + "\n")

    model = build_model(num_classes=len(class_names), alpha=BACKBONE_ALPHA)
    loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing=LABEL_SMOOTHING)

    # Early stopping watches the VALIDATION split in both phases. The test
    # split is never used to make any training decision.
    print("Phase 1: training the classifier head (base frozen)...")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(HEAD_LEARNING_RATE),
        loss=loss,
        metrics=["accuracy"],
    )
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=HEAD_EPOCHS,
        class_weight=class_weight,
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)],
    )

    print(f"\nPhase 2: fine-tuning the top {FINETUNE_LAYERS} layers of the base...")
    unfreeze_top_layers(model, FINETUNE_LAYERS)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(FINETUNE_LEARNING_RATE),
        loss=loss,
        metrics=["accuracy"],
    )
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=FINETUNE_EPOCHS,
        class_weight=class_weight,
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True)],
    )

    # Save an uncompiled copy (same layers and weights) so the file doesn't
    # carry ~17 MB of Adam state that inference never uses, and loading it
    # doesn't warn about mismatched optimizer variables.
    inference_model = tf.keras.Model(model.inputs, model.outputs, name=model.name)
    os.makedirs(MODEL_DIR, exist_ok=True)

    if gate:
        with open(CLASS_NAMES_PATH) as f:
            classifier_class_names = json.load(f)
        if class_names != classifier_class_names:
            raise RuntimeError(f"Gate classes {class_names} differ from the classifier's {classifier_class_names}.")
        inference_model.save(GATE_MODEL_PATH)
        print(f"\nOut-of-scope gate saved to {GATE_MODEL_PATH} (not evaluated on test here)")
        return

    print("\nEvaluating on the held-out test set...")
    results = evaluate(model, class_names, TEST_DIR)
    print_report(results)

    inference_model.save(MODEL_PATH)
    with open(CLASS_NAMES_PATH, "w") as f:
        json.dump(class_names, f)
    with open(METRICS_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nModel saved to {MODEL_PATH}")
    print(f"Class labels saved to {CLASS_NAMES_PATH}")
    print(f"Test metrics saved to {METRICS_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split the data and train the wound classifier or the out-of-scope gate.")
    parser.add_argument("--gate", action="store_true",
                        help=f"save the model as the out-of-scope gate ({GATE_MODEL_PATH}) instead of the wound classifier")
    train(gate=parser.parse_args().gate)

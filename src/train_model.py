"""
train_model.py

1. Reads the REAL (non-augmented) images listed in
   data/wound_dataset/manifest.csv, groups together every image that comes
   from the same source photo - including visually identical copies saved
   under different filenames - and splits whole groups into
   data/train/<class>/, data/val/<class>/ and data/test/<class>/. No source
   photo ever appears in more than one split, and all three splits hold
   real photos only.
2. Trains the classifier defined in model.py on data/train:
     - random augmentation (flip/rotate/zoom/shift/brightness/contrast) is
       applied on the fly to TRAINING batches only, so every epoch sees new
       variations and validation/test are never augmented;
     - class weights compensate for the uneven number of photos per class;
     - phase 1 trains only the classifier head on the frozen MobileNetV2
       base, phase 2 fine-tunes the top FINETUNE_LAYERS layers of the base
       at a low learning rate.
   Both phases early-stop on data/val. data/test is not looked at until the
   final evaluation.
3. Saves the trained model, class label list and test metrics to models/.

This file only handles data splitting and training. The model architecture
and prediction logic live in model.py - see that file for running
predictions once a model has been trained.

Usage:
    python train_model.py

Requires: pip install tensorflow pillow numpy
"""

import os
import csv
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
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

from model import build_model, unfreeze_top_layers, IMG_SIZE, MODEL_DIR, MODEL_PATH, CLASS_NAMES_PATH
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

# Training schedule. See README.md for the experiments behind these choices.
HEAD_EPOCHS = 15
HEAD_LEARNING_RATE = 1e-3
FINETUNE_LAYERS = 100
FINETUNE_EPOCHS = 40
FINETUNE_LEARNING_RATE = 1e-5

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


def difference_hash(image_path, hash_size=8):
    """64-bit perceptual hash: compares each pixel of a tiny grayscale copy
    to its right-hand neighbour. Re-saved or re-compressed copies of the same
    photo produce identical or near-identical hashes."""
    with Image.open(image_path) as img:
        small = img.convert("L").resize((hash_size + 1, hash_size), Image.LANCZOS)
    pixels = np.asarray(small, dtype=np.int16)
    return (pixels[:, 1:] > pixels[:, :-1]).flatten()


def group_by_source(rows):
    """
    Returns a source-group id for every row (parallel list), such that all
    copies of one source photo share a group. Each real manifest row is its
    own source photo; near-identical hashes then merge rows that are the same
    photo saved under different filenames.
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
        for offset in np.nonzero(distances <= DUPLICATE_HASH_DISTANCE)[0]:
            parent[find(i)] = find(i + 1 + int(offset))

    # Readable, stable ids: the alphabetically first filename in each group.
    members = defaultdict(list)
    for i, row in enumerate(rows):
        members[find(i)].append(row["filename"])
    root_to_id = {root: min(names) for root, names in members.items()}
    return [root_to_id[find(i)] for i in range(len(rows))]


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
        primary = sorted(Counter(classes).items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
        groups_by_class[primary].append(gid)

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


def verify_no_group_crosses_splits(split_rows):
    splits_per_group = defaultdict(set)
    for split, _class_name, _filename, group_id, _source in split_rows:
        splits_per_group[group_id].add(split)
    crossing = {g: sorted(s) for g, s in splits_per_group.items() if len(s) > 1}
    if crossing:
        raise RuntimeError(
            f"{len(crossing)} source groups appear in more than one split, e.g. "
            f"{list(crossing.items())[:3]}. The split would leak - refusing to continue."
        )


def split_and_copy(rows):
    """Groups real images by source photo, splits the groups into
    data/train, data/val and data/test, and writes data/split_manifest.csv
    recording where every file went."""
    for directory in (TRAIN_DIR, VAL_DIR, TEST_DIR):
        if os.path.exists(directory):
            robust_rmtree(directory)

    print("Grouping images by source photo (manifest + near-duplicate hashing)...")
    group_ids = group_by_source(rows)
    print(f"  {len(rows)} real images -> {len(set(group_ids))} source groups\n")
    split_of = assign_splits(rows, group_ids)

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
    verify_no_group_crosses_splits(split_rows)

    print("Split (real photos only, grouped by source photo):\n")
    for class_name in sorted(counts):
        c = counts[class_name]
        total = sum(c.values())
        warning = "  <-- LOW: consider gathering more data for this class" if total < MIN_IMAGES_PER_CLASS_WARNING else ""
        print(f"  {class_name}: {c['train']} train / {c['val']} val / {c['test']} test ({total} total){warning}")
    print()

    with open(SPLIT_MANIFEST_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["split", "class", "filename", "source_group", "source_filename"])
        writer.writerows(sorted(split_rows))
    print(f"Split manifest written to {SPLIT_MANIFEST_PATH}\n")


def make_augmenter():
    """Realistic, mild augmentations applied to raw [0, 255] training images
    on the fly - no color inversion or extreme distortion that wouldn't be
    plausible for a real wound photo."""
    return tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.05, fill_mode="reflect"),        # up to +-18 degrees
        layers.RandomZoom((-0.15, 0.0), fill_mode="reflect"),   # zoom in up to 15%
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


def train():
    tf.keras.utils.set_random_seed(RANDOM_SEED)

    print("Loading real images from the manifest...\n")
    rows = load_real_images()
    split_and_copy(rows)

    train_ds, val_ds, class_names = build_datasets()
    print(f"Classes (in label order): {class_names}\n")

    class_weight = compute_class_weights(class_names)
    print("Class weights: " + ", ".join(f"{class_names[i]}={w:.2f}" for i, w in class_weight.items()) + "\n")

    model = build_model(num_classes=len(class_names))

    # Early stopping watches the VALIDATION split in both phases. The test
    # split is never used to make any training decision.
    print("Phase 1: training the classifier head (base frozen)...")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(HEAD_LEARNING_RATE),
        loss="categorical_crossentropy",
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
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=FINETUNE_EPOCHS,
        class_weight=class_weight,
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True)],
    )

    print("\nEvaluating on the held-out test set...")
    results = evaluate(model, class_names, TEST_DIR)
    print_report(results)

    # Save an uncompiled copy (same layers and weights) so the file doesn't
    # carry ~17 MB of Adam state that inference never uses, and loading it
    # doesn't warn about mismatched optimizer variables.
    inference_model = tf.keras.Model(model.inputs, model.outputs, name=model.name)

    os.makedirs(MODEL_DIR, exist_ok=True)
    inference_model.save(MODEL_PATH)
    with open(CLASS_NAMES_PATH, "w") as f:
        json.dump(class_names, f)
    with open(METRICS_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nModel saved to {MODEL_PATH}")
    print(f"Class labels saved to {CLASS_NAMES_PATH}")
    print(f"Test metrics saved to {METRICS_PATH}")


if __name__ == "__main__":
    train()

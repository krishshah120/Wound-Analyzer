"""
train_model.py

1. Discovers the classes under data/wound_dataset/ (abrasion, bruise, cut,
   and burn's degree subfolders), and splits each class's images into
   data/train/<class>/ and data/test/<class>/ - siblings of wound_dataset,
   not nested inside it.
2. Trains the classifier defined in model.py using data/train, validated
   against data/test.
3. Saves the trained model + class label list to models/.

This file only handles data splitting and training. The model architecture
and prediction logic live in model.py - see that file for running
predictions once a model has been trained.

Usage:
    python train_model.py

Requires: pip install tensorflow pillow
"""

import os
import json
import random
import shutil
import stat
import time

import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

from model import build_model, IMG_SIZE, MODEL_DIR, MODEL_PATH, CLASS_NAMES_PATH

# ---------------------------------------------------------------------------
# Paths - anchored to this script's location
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)          # .../Wound-Analyzer

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
WOUND_DATASET_DIR = os.path.join(DATA_DIR, "wound_dataset")
TRAIN_DIR = os.path.join(DATA_DIR, "train")          # sibling of wound_dataset
TEST_DIR = os.path.join(DATA_DIR, "test")            # sibling of wound_dataset

BATCH_SIZE = 32
TEST_FRACTION = 0.2
RANDOM_SEED = 42

# Folders inside wound_dataset that hold rejected/QA images, not real classes.
IGNORE_DIRS = {"bad_images"}

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


def discover_classes(root_dir):
    """
    Walks wound_dataset and returns {class_name: [image_paths]} for every
    directory that directly contains image files. Nested folders (like
    burn/1st_degree) become their own class, named by joining the path with
    underscores (e.g. "burn_1st_degree"), so burn severity isn't lost.
    """
    classes = {}
    for current_root, dirs, files in os.walk(root_dir):
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]

        if os.path.basename(current_root) in IGNORE_DIRS:
            continue

        rel_path = os.path.relpath(current_root, root_dir)
        if rel_path == ".":
            continue

        image_files = [f for f in files if f.lower().endswith((".png", ".jpg", ".jpeg"))]
        if not image_files:
            continue  # just a container for subfolders, not a class itself

        class_name = rel_path.replace(os.sep, "_")
        classes[class_name] = [os.path.join(current_root, f) for f in image_files]

    return classes


def split_and_copy(classes):
    """Splits each class's images into train/test and copies them into
    data/train/<class>/ and data/test/<class>/."""
    if os.path.exists(TRAIN_DIR):
        robust_rmtree(TRAIN_DIR)
    if os.path.exists(TEST_DIR):
        robust_rmtree(TEST_DIR)

    rng = random.Random(RANDOM_SEED)

    print("Splitting data into train/test sets:\n")
    for class_name, image_paths in sorted(classes.items()):
        shuffled = image_paths[:]
        rng.shuffle(shuffled)

        n_test = max(1, int(len(shuffled) * TEST_FRACTION))
        test_paths = shuffled[:n_test]
        train_paths = shuffled[n_test:]

        train_class_dir = os.path.join(TRAIN_DIR, class_name)
        test_class_dir = os.path.join(TEST_DIR, class_name)
        os.makedirs(train_class_dir, exist_ok=True)
        os.makedirs(test_class_dir, exist_ok=True)

        for path in train_paths:
            shutil.copy2(path, os.path.join(train_class_dir, os.path.basename(path)))
        for path in test_paths:
            shutil.copy2(path, os.path.join(test_class_dir, os.path.basename(path)))

        total = len(shuffled)
        warning = "  <-- LOW: consider gathering more data for this class" if total < MIN_IMAGES_PER_CLASS_WARNING else ""
        print(f"  {class_name}: {len(train_paths)} train / {len(test_paths)} test ({total} total){warning}")

    print()


def build_datasets():
    train_ds = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical",
        seed=RANDOM_SEED,
    )
    test_ds = tf.keras.utils.image_dataset_from_directory(
        TEST_DIR,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical",
        shuffle=False,
    )

    class_names = train_ds.class_names  # alphabetical order - matches label indices

    # MobileNetV2 expects inputs scaled to [-1, 1] rather than raw [0, 255]
    train_ds = train_ds.map(lambda x, y: (preprocess_input(x), y))
    test_ds = test_ds.map(lambda x, y: (preprocess_input(x), y))

    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    test_ds = test_ds.prefetch(tf.data.AUTOTUNE)

    return train_ds, test_ds, class_names


def train():
    print("Discovering classes in wound_dataset...\n")
    classes = discover_classes(WOUND_DATASET_DIR)

    if not classes:
        print(f"No image classes found under {WOUND_DATASET_DIR}. Run collate_data.py first.")
        return

    split_and_copy(classes)

    train_ds, test_ds, class_names = build_datasets()
    print(f"Classes (in label order): {class_names}\n")

    model = build_model(num_classes=len(class_names))

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=3, restore_best_weights=True
    )

    model.fit(
        train_ds,
        validation_data=test_ds,
        epochs=15,
        callbacks=[early_stop],
    )

    test_loss, test_accuracy = model.evaluate(test_ds)
    print(f"\nFinal test accuracy: {test_accuracy:.2%}")

    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save(MODEL_PATH)
    with open(CLASS_NAMES_PATH, "w") as f:
        json.dump(class_names, f)

    print(f"\nModel saved to {MODEL_PATH}")
    print(f"Class labels saved to {CLASS_NAMES_PATH}")


if __name__ == "__main__":
    train()
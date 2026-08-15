import os
import csv
import random
import shutil
import stat
import time
import cv2
import numpy as np
from PIL import Image, ImageEnhance

# ---------------------------------------------------------------------------
# Paths - anchored to this script's location so it works no matter what
# directory you run it from (fixes the earlier "FileNotFoundError" issue).
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)          # .../Wound-Analyzer

raw_dir = os.path.join(PROJECT_ROOT, "data", "raw_downloads")
output_dir = os.path.join(PROJECT_ROOT, "data", "wound_dataset")

KAGGLE_WOUND_DIR = os.path.join(raw_dir, "kaggle_wound")
KAGGLE_BURN_DIR = os.path.join(raw_dir, "kaggle_burn")

target_classes = ["abrasion", "bruise", "cut", "burn"]

# Candidate subfolder names for each kaggle_wound class. Kaggle datasets are
# inconsistent about naming/casing/pluralization, so we try a few options
# and use whichever actually exists on disk. Add to these lists if none match.
SOURCE_CANDIDATES = {
    "abrasion": ["abrasion wound", "Abrasions", "abrasion", "Abrasion", "Abrasions wound"],
    "bruise":   ["bruises wound", "Bruises", "bruise", "Bruise", "Bruises wound"],
    "cut":      ["cut wound", "Cut", "cut", "Cuts", "Cut wound"],
}

# Burn severity classes as defined by the kaggle_burn (YOLO-format) dataset:
# 0 = first degree, 1 = second degree, 2 = third degree
BURN_DEGREE_MAP = {
    0: "1st_degree",
    1: "2nd_degree",
    2: "3rd_degree",
}

# Thresholds for filtering quality (mean intensity / blur variance)
BLUR_THRESHOLD = 80.0
DARK_THRESHOLD = 40.0
BRIGHT_THRESHOLD = 215.0

# After real data is collected, underrepresented classes get topped up with
# augmented (rotated/flipped/brightness-jittered/zoomed) copies of their own
# images, up to the size of the largest class. MAX_AUGMENTATION_MULTIPLIER
# caps how far any one class gets stretched - e.g. 4 means a class never
# gets inflated past 4x its real image count, even if the largest class is
# much bigger than that. This avoids a tiny class becoming mostly synthetic
# duplicates of a handful of source photos.
BALANCE_CLASSES = True
MAX_AUGMENTATION_MULTIPLIER = 4

MANIFEST_PATH = os.path.join(output_dir, "manifest.csv")
BAD_IMAGES_DIR = os.path.join(output_dir, "bad_images")


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


def slugify_reason(reason):
    """Turns 'Too Blurry (Score: 45.2)' into 'too_blurry' for use in a filename."""
    core = reason.split("(")[0].strip().lower()
    return core.replace(" ", "_")


def save_bad_image(src_path, target_class, dataset_prefix, reason, index):
    """Copies a quality-rejected image into bad_images/ (unmodified) so it can
    be reviewed manually to sanity-check the quality thresholds."""
    reason_slug = slugify_reason(reason)
    ext = os.path.splitext(src_path)[1] or ".jpg"
    dest_filename = f"{target_class}_{dataset_prefix}_{index}_{reason_slug}{ext}"
    dest_path = os.path.join(BAD_IMAGES_DIR, dest_filename)
    try:
        shutil.copy2(src_path, dest_path)
    except Exception as e:
        print(f"  Could not copy bad image {os.path.basename(src_path)} for review: {e}")


def resolve_source_folder(base_dir, candidates):
    """Return the first candidate subfolder that actually exists on disk."""
    for name in candidates:
        path = os.path.join(base_dir, name)
        if os.path.isdir(path):
            return path
    return None


def is_good_quality(image_path):
    """Checks if an image passes quality standards for blurriness and exposure."""
    img_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if img_gray is None:
        return False, "Unreadable"

    mean_brightness = np.mean(img_gray)
    if mean_brightness < DARK_THRESHOLD:
        return False, f"Too Dark ({mean_brightness:.1f})"
    if mean_brightness > BRIGHT_THRESHOLD:
        return False, f"Too Bright ({mean_brightness:.1f})"

    laplacian_var = cv2.Laplacian(img_gray, cv2.CV_64F).var()
    if laplacian_var < BLUR_THRESHOLD:
        return False, f"Too Blurry (Score: {laplacian_var:.1f})"

    return True, "Passed"


def save_clean_image(src_path, dest_path):
    """Convert to RGB, resize to 224x224, save as JPEG. Returns True on success."""
    try:
        with Image.open(src_path) as img:
            rgb_img = img.convert("RGB")
            resized_img = rgb_img.resize((224, 224))
            resized_img.save(dest_path, "JPEG")
        return True
    except Exception as e:
        print(f"  Skipping broken file {os.path.basename(src_path)}: {e}")
        return False


def clean_and_extract(source_folder, target_class, dataset_prefix, manifest_writer):
    """Original flat-folder handling: for kaggle_wound classes (no sub-labels)."""
    if source_folder is None or not os.path.exists(source_folder):
        print(f"Skipping '{target_class}': source folder not found -> {source_folder}")
        return

    saved_count = 0
    rejected_count = 0

    for filename in sorted(os.listdir(source_folder)):
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            src_path = os.path.join(source_folder, filename)

            passed, reason = is_good_quality(src_path)
            if not passed:
                save_bad_image(src_path, target_class, dataset_prefix, reason, rejected_count)
                rejected_count += 1
                continue

            new_filename = f"{target_class}_{dataset_prefix}_{saved_count}.jpg"
            dest_path = os.path.join(output_dir, target_class, new_filename)

            if save_clean_image(src_path, dest_path):
                manifest_writer.writerow([new_filename, target_class, "", filename, "no"])
                saved_count += 1

    print(f"Saved {saved_count} images to '{target_class}' ({rejected_count} rejected for poor quality).")


def find_yolo_image_label_pairs(root_dir):
    """
    Finds YOLO-style image/label pairs. Handles the kaggle_burn layout, where
    imgN.jpg and imgN.txt sit flat in the same folder (no images/ or labels/
    subfolders). Also checks one level of images/+labels/ subfolders in case
    a different YOLO source uses that layout instead.

    Some label .txt files in kaggle_burn have no matching image (the source
    image apparently failed to download/was removed) - those are skipped
    here rather than raising an error.
    """
    pairs = []

    # Case 1: flat layout - imgN.jpg / imgN.txt side by side in root_dir
    for filename in sorted(os.listdir(root_dir)):
        full_path = os.path.join(root_dir, filename)
        if not os.path.isfile(full_path):
            continue
        if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
            continue
        base_name = os.path.splitext(filename)[0]
        label_path = os.path.join(root_dir, base_name + ".txt")
        pairs.append((full_path, label_path))

    if pairs:
        return pairs

    # Case 2: nested images/ + labels/ layout (e.g. train/images, train/labels)
    for current_root, _dirs, files in os.walk(root_dir):
        if os.path.basename(current_root).lower() != "images":
            continue
        label_dir = os.path.join(os.path.dirname(current_root), "labels")
        for filename in sorted(files):
            if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
                continue
            img_path = os.path.join(current_root, filename)
            label_path = os.path.join(label_dir, os.path.splitext(filename)[0] + ".txt")
            pairs.append((img_path, label_path))
    return pairs


def read_yolo_degree(label_path):
    """
    Reads the class id from a YOLO label file and maps it to a burn degree.
    If a file has multiple annotated boxes, this uses the class of the first
    box. Returns None if the label is missing/unreadable/unmapped.
    """
    if not os.path.exists(label_path):
        return None
    try:
        with open(label_path, "r") as f:
            first_line = f.readline().strip()
        if not first_line:
            return None
        token = first_line.split()[0]
        class_id = int(float(token))  # tolerates "0", "0.0", etc.
        return BURN_DEGREE_MAP.get(class_id)
    except Exception:
        return None


def clean_and_extract_burn(root_dir, dataset_prefix, manifest_writer):
    """
    Handles the kaggle_burn YOLO-format dataset. All images are saved under
    the single 'burn' class folder (per target_classes), but each image's
    degree label (1st/2nd/3rd) is preserved in both the filename and the
    manifest CSV so severity information isn't lost.
    """
    print(f"Looking for burn images in: {root_dir}")

    if root_dir is None or not os.path.exists(root_dir):
        print(f"Skipping 'burn': source folder not found -> {root_dir}")
        return

    pairs = find_yolo_image_label_pairs(root_dir)
    print(f"Found {len(pairs)} image files to check.")
    if not pairs:
        print(f"Skipping 'burn': no image files found directly in -> {root_dir}")
        return

    for degree_name in BURN_DEGREE_MAP.values():
        os.makedirs(os.path.join(output_dir, "burn", degree_name), exist_ok=True)

    saved_count = 0
    rejected_count = 0
    unlabeled_count = 0
    sample_unlabeled_shown = 0

    for img_path, label_path in pairs:
        degree = read_yolo_degree(label_path)
        if degree is None:
            unlabeled_count += 1
            # Print the first couple of failures so we can see *why* -
            # e.g. missing .txt file vs. an unrecognized class id format.
            if sample_unlabeled_shown < 3:
                if not os.path.exists(label_path):
                    print(f"  [debug] no label file for {os.path.basename(img_path)} -> expected {label_path}")
                else:
                    with open(label_path, "r") as f:
                        raw = f.readline().strip()
                    print(f"  [debug] unrecognized label content in {os.path.basename(label_path)}: '{raw}'")
                sample_unlabeled_shown += 1
            continue

        passed, reason = is_good_quality(img_path)
        if not passed:
            save_bad_image(img_path, f"burn_{degree}", dataset_prefix, reason, rejected_count)
            rejected_count += 1
            continue

        new_filename = f"burn_{degree}_{dataset_prefix}_{saved_count}.jpg"
        dest_path = os.path.join(output_dir, "burn", degree, new_filename)

        if save_clean_image(img_path, dest_path):
            manifest_writer.writerow([new_filename, "burn", degree, os.path.basename(img_path), "no"])
            saved_count += 1

    print(
        f"Saved {saved_count} images to 'burn' "
        f"({rejected_count} rejected for poor quality, {unlabeled_count} skipped for missing/unrecognized labels)."
    )


def augment_image(img):
    """
    Applies a small random combination of realistic augmentations to an
    already-cleaned image: rotation, a horizontal flip, a slight zoom/crop,
    and brightness/contrast jitter. Used to generate extra training images
    for classes that don't have enough real photos, without inventing
    features that aren't plausible for a real wound photo (no color
    inversion, no extreme distortion, etc.).
    """
    arr = np.array(img)  # img is already RGB by the time this is called
    height, width = arr.shape[:2]

    # Rotation - reflecting the edges instead of leaving black corners
    angle = random.uniform(-15, 15)
    rotation_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
    arr = cv2.warpAffine(arr, rotation_matrix, (width, height), borderMode=cv2.BORDER_REFLECT_101)

    if random.random() < 0.5:
        arr = cv2.flip(arr, 1)

    # Slight zoom: crop a random inner region, then resize back up
    zoom = random.uniform(0.85, 1.0)
    crop_w, crop_h = int(width * zoom), int(height * zoom)
    left = random.randint(0, width - crop_w) if width > crop_w else 0
    top = random.randint(0, height - crop_h) if height > crop_h else 0
    arr = arr[top:top + crop_h, left:left + crop_w]
    arr = cv2.resize(arr, (width, height))

    img = Image.fromarray(arr)
    img = ImageEnhance.Brightness(img).enhance(random.uniform(0.85, 1.15))
    img = ImageEnhance.Contrast(img).enhance(random.uniform(0.85, 1.15))

    return img


def discover_class_folders(root_dir):
    """
    Finds every leaf folder under wound_dataset that directly contains
    images, skipping bad_images/. This treats burn's degree subfolders
    (burn/1st_degree, burn/2nd_degree, burn/3rd_degree) as their own
    classes, matching how train_model.py's discover_classes() will later
    split them for training - so balancing happens at the same granularity
    the model actually trains on.
    """
    classes = {}
    for current_root, dirs, files in os.walk(root_dir):
        dirs[:] = [d for d in dirs if d != "bad_images"]
        if os.path.basename(current_root) == "bad_images":
            continue

        image_files = [f for f in files if f.lower().endswith((".png", ".jpg", ".jpeg"))]
        if not image_files:
            continue

        classes[current_root] = [os.path.join(current_root, f) for f in image_files]

    return classes


def balance_classes(manifest_writer):
    """
    Tops up underrepresented classes with augmented copies of their own
    real images, up to the size of the largest class (capped at
    MAX_AUGMENTATION_MULTIPLIER x that class's real count). Augmented files
    are named with an "_aug#" suffix and flagged in the manifest, so
    they're always distinguishable from real photos later.
    """
    print("\nBalancing class counts with augmentation:\n")

    class_folders = discover_class_folders(output_dir)
    if not class_folders:
        print("No classes found to balance.")
        return

    counts = {folder: len(paths) for folder, paths in class_folders.items()}
    target = max(counts.values())

    for folder, image_paths in sorted(class_folders.items()):
        current_count = len(image_paths)
        cap = min(target, current_count * MAX_AUGMENTATION_MULTIPLIER)
        needed = cap - current_count

        rel_path = os.path.relpath(folder, output_dir)
        display_name = rel_path.replace(os.sep, "_")

        if needed <= 0:
            print(f"  {display_name}: {current_count} images - no augmentation needed")
            continue

        # class / burn_degree columns, matching the scheme used elsewhere
        # in this manifest
        path_parts = rel_path.split(os.sep)
        if path_parts[0] == "burn" and len(path_parts) > 1:
            manifest_class = "burn"
            manifest_degree = path_parts[1]
        else:
            manifest_class = rel_path
            manifest_degree = ""

        for i in range(needed):
            source_path = random.choice(image_paths)
            with Image.open(source_path) as src_img:
                augmented_img = augment_image(src_img.convert("RGB"))

            source_stem = os.path.splitext(os.path.basename(source_path))[0]
            new_filename = f"{source_stem}_aug{i}.jpg"
            dest_path = os.path.join(folder, new_filename)
            augmented_img.save(dest_path, "JPEG")

            manifest_writer.writerow([
                new_filename, manifest_class, manifest_degree,
                os.path.basename(source_path), "yes",
            ])

        shortfall_note = "" if cap == target else f"  <-- capped at {MAX_AUGMENTATION_MULTIPLIER}x real data, still below the largest class ({target}); consider gathering more real images"
        print(f"  {display_name}: {current_count} real + {needed} augmented = {cap} total{shortfall_note}")


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------
print("Starting dataset cleaning\n")

# Wipe wound_dataset entirely before rebuilding, so re-running this script
# after changing a threshold, fixing a source path, etc. always produces a
# clean result instead of mixing old and new images together.
if os.path.exists(output_dir):
    print(f"Clearing existing data in {output_dir}\n")
    robust_rmtree(output_dir)

for cls in target_classes:
    os.makedirs(os.path.join(output_dir, cls), exist_ok=True)
os.makedirs(BAD_IMAGES_DIR, exist_ok=True)

with open(MANIFEST_PATH, "w", newline="") as manifest_file:
    manifest_writer = csv.writer(manifest_file)
    manifest_writer.writerow(["filename", "class", "burn_degree", "source_filename", "augmented"])

    abrasion_src = resolve_source_folder(KAGGLE_WOUND_DIR, SOURCE_CANDIDATES["abrasion"])
    bruise_src = resolve_source_folder(KAGGLE_WOUND_DIR, SOURCE_CANDIDATES["bruise"])
    cut_src = resolve_source_folder(KAGGLE_WOUND_DIR, SOURCE_CANDIDATES["cut"])

    clean_and_extract(abrasion_src, "abrasion", "kg1", manifest_writer)
    clean_and_extract(bruise_src, "bruise", "kg1", manifest_writer)
    clean_and_extract(cut_src, "cut", "kg1", manifest_writer)

    clean_and_extract_burn(KAGGLE_BURN_DIR, "kg2", manifest_writer)

    if BALANCE_CLASSES:
        balance_classes(manifest_writer)

print(f"\nManifest written to {MANIFEST_PATH}")
print("Completed")
"""
evaluate_model.py

Evaluates a trained model against a folder of labelled images laid out as
<dir>/<class_name>/<image>. Prints overall accuracy, per-class
precision/recall, the confusion matrix, the burn-severity sub-matrix, and
how the model behaves once model.py's CONFIDENCE_THRESHOLD is applied
(which is what the web app actually shows users).

Usage:
    python evaluate_model.py                       # models/wound_model.keras on data/test
    python evaluate_model.py --model path.keras --data-dir some/dir --json out.json
"""

import os
import json
import argparse
from collections import Counter

import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

from model import IMG_SIZE, MODEL_PATH, CLASS_NAMES_PATH, CONFIDENCE_THRESHOLD

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DEFAULT_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "test")

BURN_CLASSES = ["burn_1st_degree", "burn_2nd_degree", "burn_3rd_degree"]


def predict_directory(model, data_dir, class_names):
    """Returns (true_indices, probability_matrix, file_paths) for every image
    under data_dir, using class_names to map folder names to label indices."""
    ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        image_size=IMG_SIZE,
        batch_size=32,
        label_mode="int",
        class_names=class_names,
        shuffle=False,
    )
    file_paths = list(ds.file_paths)
    ds = ds.map(lambda x, y: (preprocess_input(x), y))

    y_true, probs = [], []
    for images, labels in ds:
        probs.append(model.predict_on_batch(images))
        y_true.append(labels.numpy())
    return np.concatenate(y_true), np.concatenate(probs), file_paths


def wilson_interval(successes, total, z=1.96):
    """95% Wilson score interval for a proportion. Test sets here are small
    (a few dozen images per class), so a single accuracy number hides a lot
    of uncertainty - report the range alongside it."""
    if total == 0:
        return None
    p = successes / total
    denom = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denom
    half = z * np.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return [float(centre - half), float(centre + half)]


def confusion_matrix(y_true, y_pred, n):
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def format_matrix(cm, row_names, col_names):
    width = max(len(n) for n in row_names + col_names) + 2
    lines = ["true \\ predicted".ljust(width) + "".join(n[:width - 1].rjust(width) for n in col_names)]
    for name, row in zip(row_names, cm):
        lines.append(name.ljust(width) + "".join(str(v).rjust(width) for v in row))
    return "\n".join(lines)


def evaluate(model, class_names, data_dir):
    y_true, probs, _ = predict_directory(model, data_dir, class_names)
    n = len(class_names)
    y_pred = probs.argmax(axis=1)
    confidence = probs.max(axis=1)

    cm = confusion_matrix(y_true, y_pred, n)
    per_class = {}
    for i, name in enumerate(class_names):
        support = int(cm[i].sum())
        predicted = int(cm[:, i].sum())
        per_class[name] = {
            "support": support,
            "recall": float(cm[i, i] / support) if support else None,
            "recall_95ci": wilson_interval(int(cm[i, i]), support),
            "precision": float(cm[i, i] / predicted) if predicted else None,
        }

    recalls = [v["recall"] for v in per_class.values() if v["recall"] is not None]
    results = {
        "data_dir": data_dir,
        "n_images": int(len(y_true)),
        "accuracy": float((y_pred == y_true).mean()),
        "accuracy_95ci": wilson_interval(int((y_pred == y_true).sum()), int(len(y_true))),
        "balanced_accuracy": float(np.mean(recalls)),
        "per_class": per_class,
        "class_names": class_names,
        "confusion_matrix": cm.tolist(),
    }

    # What the web app actually returns: "unknown" below the threshold.
    confident = confidence >= CONFIDENCE_THRESHOLD
    results["threshold"] = {
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "fraction_answered": float(confident.mean()),
        "accuracy_when_answered": float((y_pred[confident] == y_true[confident]).mean()) if confident.any() else None,
        "confusion_matrix_when_answered": confusion_matrix(y_true[confident], y_pred[confident], n).tolist(),
    }

    if all(b in class_names for b in BURN_CLASSES):
        idx = [class_names.index(b) for b in BURN_CLASSES]
        third = class_names.index("burn_3rd_degree")
        first = class_names.index("burn_1st_degree")
        is_third = y_true == third
        results["burn"] = {
            # Rows are true burn degree, columns are ALL classes, so a 3rd
            # degree burn predicted as "cut" is still visible here.
            "burn_rows_confusion_matrix": cm[idx].tolist(),
            "burn_3rd_recall": per_class["burn_3rd_degree"]["recall"],
            "burn_3rd_predicted_as_1st": int(cm[third, first]),
            "burn_3rd_predicted_as_non_burn": int(sum(cm[third, j] for j in range(n) if j not in idx)),
            "burn_3rd_support": int(is_third.sum()),
            # Worst case for the site: a confident (non-"unknown") answer
            # telling someone with a 3rd degree burn it's 1st degree.
            "burn_3rd_confidently_predicted_as_1st": int((is_third & confident & (y_pred == first)).sum()),
            "burn_3rd_confidently_predicted_as_anything_but_3rd": int((is_third & confident & (y_pred != third)).sum()),
        }

    return results


def print_report(r):
    names = r["class_names"]
    print(f"\nEvaluated {r['n_images']} images from {r['data_dir']}")
    lo, hi = r["accuracy_95ci"]
    print(f"Accuracy:          {r['accuracy']:.2%}  (95% CI {lo:.1%} - {hi:.1%})")
    print(f"Balanced accuracy: {r['balanced_accuracy']:.2%}  (mean of per-class recall)\n")

    print(f"{'class':<18}{'support':>8}{'recall':>9}{'recall 95% CI':>17}{'precision':>11}")
    for name, v in r["per_class"].items():
        rec = f"{v['recall']:.1%}" if v["recall"] is not None else "-"
        ci = f"{v['recall_95ci'][0]:.0%}-{v['recall_95ci'][1]:.0%}" if v["recall_95ci"] else "-"
        prec = f"{v['precision']:.1%}" if v["precision"] is not None else "-"
        print(f"{name:<18}{v['support']:>8}{rec:>9}{ci:>17}{prec:>11}")

    print("\nConfusion matrix (rows = true class, columns = predicted class):")
    print(format_matrix(np.array(r["confusion_matrix"]), names, names))

    t = r["threshold"]
    print(f"\nWith CONFIDENCE_THRESHOLD = {t['confidence_threshold']:.2f} (what the app returns):")
    print(f"  answered (not 'unknown'): {t['fraction_answered']:.1%} of images")
    if t["accuracy_when_answered"] is not None:
        print(f"  accuracy when answered:   {t['accuracy_when_answered']:.1%}")

    if "burn" in r:
        b = r["burn"]
        print("\nBurn severity rows (true burn degree vs. every predicted class):")
        print(format_matrix(np.array(b["burn_rows_confusion_matrix"]), BURN_CLASSES, names))
        print(f"  3rd degree recall: {b['burn_3rd_recall']:.1%} of {b['burn_3rd_support']}")
        print(f"  3rd degree predicted as 1st degree: {b['burn_3rd_predicted_as_1st']}")
        print(f"  3rd degree predicted as a non-burn class: {b['burn_3rd_predicted_as_non_burn']}")
        print(f"  3rd degree CONFIDENTLY (>= threshold) predicted as 1st degree: {b['burn_3rd_confidently_predicted_as_1st']}")
        print(f"  3rd degree CONFIDENTLY predicted as anything other than 3rd: {b['burn_3rd_confidently_predicted_as_anything_but_3rd']}")


def evaluate_out_of_scope(model, class_names, ood_dir):
    """For photos that belong to none of the model's classes (laid out as
    <ood_dir>/<group>/<image>), reports how often the model still gives a
    confident label instead of "unknown", and which labels it gives."""
    results = {}
    for group in sorted(d for d in os.listdir(ood_dir) if os.path.isdir(os.path.join(ood_dir, d))):
        folder = os.path.join(ood_dir, group)
        images = [np.asarray(tf.keras.utils.load_img(os.path.join(folder, f), target_size=IMG_SIZE), dtype="float32")
                  for f in sorted(os.listdir(folder)) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
        probs = model.predict(preprocess_input(np.stack(images)), batch_size=64, verbose=0)
        confident = probs.max(axis=1) >= CONFIDENCE_THRESHOLD
        labels = Counter(class_names[i] for i in probs.argmax(axis=1)[confident])
        results[group] = {"n_images": len(images), "fraction_confidently_labelled": float(confident.mean()),
                          "confident_labels": dict(labels.most_common())}
    return results


def print_out_of_scope_report(results):
    print(f"\nOut-of-scope photos (should ideally come back 'unknown'):")
    total = sum(r["n_images"] for r in results.values())
    confident = sum(r["n_images"] * r["fraction_confidently_labelled"] for r in results.values())
    for group, r in results.items():
        top = ", ".join(f"{k} {v}" for k, v in list(r["confident_labels"].items())[:3])
        print(f"  {group:<16} {r['n_images']:>4} photos, confidently labelled: {r['fraction_confidently_labelled']:6.1%}  ({top})")
    print(f"  overall: {confident / total:.1%} of {total} out-of-scope photos got a confident wound label")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL_PATH)
    parser.add_argument("--class-names", default=CLASS_NAMES_PATH)
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--ood-dir", help="optional folder of out-of-scope photos, e.g. ../data/ood_dataset")
    parser.add_argument("--json", help="optional path to write the full results as JSON")
    args = parser.parse_args()

    model = tf.keras.models.load_model(args.model)
    with open(args.class_names) as f:
        class_names = json.load(f)

    results = evaluate(model, class_names, args.data_dir)
    print_report(results)

    if args.ood_dir:
        results["out_of_scope"] = evaluate_out_of_scope(model, class_names, args.ood_dir)
        print_out_of_scope_report(results["out_of_scope"])

    if args.json:
        with open(args.json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nFull results written to {args.json}")


if __name__ == "__main__":
    main()

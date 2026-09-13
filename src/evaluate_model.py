"""
evaluate_model.py

Evaluates a trained model against a folder of labelled images laid out as
<dir>/<class_name>/<image>. Prints accuracy on the wound photos, per-class
precision/recall, the confusion matrix, the burn-severity rows, what the web
app would return (model.decide(): the confidence threshold plus the
out-of-scope class), and - if the folder has an out_of_scope class, or
--ood-dir is given - how often out-of-scope photos still get a wound label.

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

from model import IMG_SIZE, MODEL_PATH, CLASS_NAMES_PATH, CONFIDENCE_THRESHOLD, OUT_OF_SCOPE_CLASS, decide

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DEFAULT_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "test")

BURN_CLASSES = ["burn_1st_degree", "burn_2nd_degree", "burn_3rd_degree"]


def predict_directory(model, data_dir, class_names):
    """Returns (true_indices, probability_matrix, file_paths) for every image
    under data_dir/<class_name>/, using class_names to map folder names to
    label indices. Images are loaded exactly the way model.predict() loads
    them for the web app (tf.keras.utils.load_img). TensorFlow's dataset
    loader decodes JPEGs slightly differently, which changed the answer on
    about 2-3% of test photos."""
    missing = [c for c in class_names if not os.path.isdir(os.path.join(data_dir, c))]
    if missing:
        raise FileNotFoundError(f"{data_dir} has no folder for class(es) {missing}")
    y_true, file_paths = [], []
    for index, class_name in enumerate(class_names):
        folder = os.path.join(data_dir, class_name)
        for filename in sorted(os.listdir(folder)):
            if filename.lower().endswith((".png", ".jpg", ".jpeg")):
                file_paths.append(os.path.join(folder, filename))
                y_true.append(index)

    probs = []
    for start in range(0, len(file_paths), 64):
        batch = [tf.keras.utils.img_to_array(tf.keras.utils.load_img(p, target_size=IMG_SIZE))
                 for p in file_paths[start:start + 64]]
        probs.append(model.predict_on_batch(preprocess_input(np.stack(batch))))
    return np.array(y_true), np.concatenate(probs), file_paths


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
    """
    Metrics for a folder of labelled images. If the model has an
    OUT_OF_SCOPE_CLASS and data_dir has that folder, "in-scope" metrics are
    computed on the wound photos only (a wound photo rejected as out of scope
    counts as wrong), and the out-of-scope photos are scored on how often they
    still get a confident wound label. App-level numbers use model.decide(),
    the same rule the web app uses.
    """
    y_true, probs, _ = predict_directory(model, data_dir, class_names)
    n = len(class_names)
    y_pred = probs.argmax(axis=1)
    ood_index = class_names.index(OUT_OF_SCOPE_CLASS) if OUT_OF_SCOPE_CLASS in class_names else None
    wound_names = [c for c in class_names if c != OUT_OF_SCOPE_CLASS]
    in_scope = y_true != ood_index if ood_index is not None else np.ones(len(y_true), dtype=bool)

    decisions = [decide(p, class_names) for p in probs]
    app_label = np.array([d[0] for d in decisions])
    answered = app_label != "unknown"
    true_names = np.array([class_names[i] for i in y_true])

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

    correct = (y_pred == y_true) & in_scope
    wound_columns = [class_names.index(c) for c in wound_names]
    wound_only_pred = np.array(wound_columns)[probs[:, wound_columns].argmax(axis=1)]
    results = {
        "data_dir": data_dir,
        "class_names": class_names,
        "n_in_scope_images": int(in_scope.sum()),
        # In-scope photos only. Rejected-as-out-of-scope counts as wrong.
        "accuracy": float(correct.sum() / in_scope.sum()),
        "accuracy_95ci": wilson_interval(int(correct.sum()), int(in_scope.sum())),
        "balanced_accuracy": float(np.mean([per_class[c]["recall"] for c in wound_names if per_class[c]["recall"] is not None])),
        # Most likely wound class, ignoring the out-of-scope output - comparable
        # with a six-class model.
        "wound_class_accuracy": float((wound_only_pred[in_scope] == y_true[in_scope]).mean()),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "app": {
            "confidence_threshold": CONFIDENCE_THRESHOLD,
            "in_scope_fraction_answered": float(answered[in_scope].mean()),
            "in_scope_accuracy_when_answered": float((app_label[in_scope & answered] == true_names[in_scope & answered]).mean())
            if (in_scope & answered).any() else None,
            "in_scope_fraction_rejected_as_out_of_scope": float((y_pred[in_scope] == ood_index).mean()) if ood_index is not None else 0.0,
        },
    }

    if ood_index is not None and (~in_scope).any():
        ood = ~in_scope
        results["out_of_scope"] = {
            "n_images": int(ood.sum()),
            "fraction_confidently_labelled": float(answered[ood].mean()),
            "confident_labels": dict(Counter(str(label) for label in app_label[ood & answered]).most_common()),
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
            "burn_3rd_support": int(is_third.sum()),
            "burn_3rd_predicted_as_1st": int(cm[third, first]),
            "burn_3rd_predicted_as_non_burn": int(sum(cm[third, j] for j in range(n) if j not in idx)),
            # What the site shows someone with a 3rd degree burn.
            "burn_3rd_app_says_1st": int((is_third & (app_label == "burn_1st_degree")).sum()),
            "burn_3rd_app_says_other_wound": int((is_third & answered & (app_label != "burn_3rd_degree")).sum()),
            "burn_3rd_app_says_unknown": int((is_third & ~answered).sum()),
            "burn_3rd_rejected_as_out_of_scope": int((is_third & (y_pred == ood_index)).sum()) if ood_index is not None else 0,
        }

    return results


def print_report(r):
    names = r["class_names"]
    print(f"\nEvaluated {r['n_in_scope_images']} wound photos from {r['data_dir']}")
    lo, hi = r["accuracy_95ci"]
    print(f"Accuracy:             {r['accuracy']:.2%}  (95% CI {lo:.1%} - {hi:.1%})")
    print(f"Balanced accuracy:    {r['balanced_accuracy']:.2%}  (mean of per-class recall)")
    print(f"Wound-class accuracy: {r['wound_class_accuracy']:.2%}  (most likely wound class, ignoring out-of-scope)\n")

    print(f"{'class':<18}{'support':>8}{'recall':>9}{'recall 95% CI':>17}{'precision':>11}")
    for name, v in r["per_class"].items():
        rec = f"{v['recall']:.1%}" if v["recall"] is not None else "-"
        ci = f"{v['recall_95ci'][0]:.0%}-{v['recall_95ci'][1]:.0%}" if v["recall_95ci"] else "-"
        prec = f"{v['precision']:.1%}" if v["precision"] is not None else "-"
        print(f"{name:<18}{v['support']:>8}{rec:>9}{ci:>17}{prec:>11}")

    print("\nConfusion matrix (rows = true class, columns = predicted class):")
    print(format_matrix(np.array(r["confusion_matrix"]), names, names))

    a = r["app"]
    print(f"\nWhat the app returns (CONFIDENCE_THRESHOLD = {a['confidence_threshold']:.2f}), wound photos:")
    print(f"  answered (not 'unknown'):     {a['in_scope_fraction_answered']:.1%}")
    if a["in_scope_accuracy_when_answered"] is not None:
        print(f"  accuracy when answered:       {a['in_scope_accuracy_when_answered']:.1%}")
    print(f"  rejected as out of scope:     {a['in_scope_fraction_rejected_as_out_of_scope']:.1%}")
    if "out_of_scope" in r:
        o = r["out_of_scope"]
        print(f"Out-of-scope photos ({o['n_images']}): confidently given a wound label: {o['fraction_confidently_labelled']:.1%} "
              f"{o['confident_labels']}")

    if "burn" in r:
        b = r["burn"]
        print("\nBurn severity rows (true burn degree vs. every predicted class):")
        print(format_matrix(np.array(b["burn_rows_confusion_matrix"]), BURN_CLASSES, names))
        print(f"  3rd degree recall: {b['burn_3rd_recall']:.1%} of {b['burn_3rd_support']}")
        print(f"  3rd degree predicted as 1st degree: {b['burn_3rd_predicted_as_1st']}")
        print(f"  3rd degree predicted as a non-burn class: {b['burn_3rd_predicted_as_non_burn']}")
        print(f"  app tells a 3rd degree burn it is 1st degree: {b['burn_3rd_app_says_1st']}")
        print(f"  app gives a 3rd degree burn any other wound label: {b['burn_3rd_app_says_other_wound']}")
        print(f"  app says 'unknown' for a 3rd degree burn: {b['burn_3rd_app_says_unknown']} "
              f"({b['burn_3rd_rejected_as_out_of_scope']} of them rejected as out of scope)")


def evaluate_out_of_scope(model, class_names, ood_dir):
    """For photos that belong to none of the wound classes (laid out as
    <ood_dir>/<group>/<image>), reports how often the app would still give a
    confident wound label instead of "unknown", and which labels it gives."""
    results = {}
    for group in sorted(d for d in os.listdir(ood_dir) if os.path.isdir(os.path.join(ood_dir, d))):
        folder = os.path.join(ood_dir, group)
        images = [tf.keras.utils.img_to_array(tf.keras.utils.load_img(os.path.join(folder, f), target_size=IMG_SIZE))
                  for f in sorted(os.listdir(folder)) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
        probs = model.predict(preprocess_input(np.stack(images)), batch_size=64, verbose=0)
        labels = [decide(p, class_names)[0] for p in probs]
        confident = Counter(label for label in labels if label != "unknown")
        results[group] = {"n_images": len(images), "fraction_confidently_labelled": sum(confident.values()) / len(images),
                          "confident_labels": dict(confident.most_common())}
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

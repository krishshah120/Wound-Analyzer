"""
framing_check.py

How well does the model cope with a wound photographed from further away?

Every photo in data/ is a tight crop, but someone photographing their own arm
holds the phone at arm's length and the wound fills a fraction of the frame.
This makes copies of a split where each WOUND photo is shrunk into part of the
frame - the surround is the photo's own content, blurred, so it stays
skin-toned and lit like the shot - and reports what the app would answer.
Out-of-scope photos are left alone, so their numbers stay comparable.

The framed photos are a synthetic stand-in for standing further back: they
lose no detail the way a real distant photo does, so treat them as a rough
guide, not a measurement of real uploads. They are re-encoded as JPEG at the
quality the MRC site's browser code uses, because that alone moves a few
answers.

Usage:
    python framing_check.py                     # val, shipped classifier + gate
    python framing_check.py --split test --no-gate
"""

import io
import os
import json
import argparse

import numpy as np
import tensorflow as tf
from PIL import Image, ImageFilter

from model import (IMG_SIZE, MODEL_PATH, CLASS_NAMES_PATH, GATE_MODEL_PATH, OUT_OF_SCOPE_CLASS, decide)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
FILL_FRACTIONS = (1.0, 0.7, 0.5, 0.35)


def framed(img, fill):
    """The photo shrunk to `fill` of the frame, on a blurred surround made from
    itself, then re-encoded as JPEG the way a browser re-encodes an upload."""
    if fill >= 1.0:
        return img
    side = img.size[0]
    inner = max(1, int(round(side * fill)))
    out = img.resize((side, side), Image.BICUBIC).filter(ImageFilter.GaussianBlur(side * 0.08))
    out.paste(img.resize((inner, inner), Image.BICUBIC), ((side - inner) // 2, (side - inner) // 2))
    buffer = io.BytesIO()
    out.save(buffer, "JPEG", quality=82)
    buffer.seek(0)
    return Image.open(buffer)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL_PATH)
    parser.add_argument("--gate-model", default=GATE_MODEL_PATH)
    parser.add_argument("--no-gate", action="store_true")
    parser.add_argument("--split", default="val", choices=["val", "test"])
    parser.add_argument("--json", help="optional path to write the results as JSON")
    args = parser.parse_args()

    model = tf.keras.models.load_model(args.model)
    gate = None if args.no_gate else tf.keras.models.load_model(args.gate_model)
    with open(CLASS_NAMES_PATH) as f:
        class_names = json.load(f)

    paths, truth = [], []
    for class_name in class_names:
        folder = os.path.join(PROJECT_ROOT, "data", args.split, class_name)
        if not os.path.isdir(folder):
            raise FileNotFoundError(f"Missing {folder}: the check needs the full {args.split} split.")
        for name in sorted(os.listdir(folder)):
            if name.lower().endswith((".png", ".jpg", ".jpeg")):
                paths.append(os.path.join(folder, name))
                truth.append(class_name)
    truth = np.array(truth)
    in_scope = truth != OUT_OF_SCOPE_CLASS

    results = {}
    print(f"{args.split}: {int(in_scope.sum())} wound photos, {int((~in_scope).sum())} out-of-scope photos"
          f"{'' if gate is not None else ' (classifier alone)'}\n")
    print(f"{'wound fills':>12}{'answered':>12}{'correct when answered':>24}{'out-of-scope labelled':>24}")
    for fill in FILL_FRACTIONS:
        images = []
        for path, class_name in zip(paths, truth):
            with Image.open(path) as img:
                img = img.convert("RGB")
                if class_name != OUT_OF_SCOPE_CLASS:
                    img = framed(img, fill)
                if img.size != IMG_SIZE:
                    img = img.resize(IMG_SIZE, Image.NEAREST)
                images.append(np.asarray(img, dtype=np.float32))
        x = tf.keras.applications.mobilenet_v2.preprocess_input(np.stack(images))
        probs = model.predict(x, batch_size=64, verbose=0)
        gate_probs = gate.predict(x, batch_size=64, verbose=0) if gate is not None else [None] * len(probs)
        labels = np.array([decide(p, class_names, g)[0] for p, g in zip(probs, gate_probs)])

        answered = labels[in_scope] != "unknown"
        correct = labels[in_scope][answered] == truth[in_scope][answered]
        ood_labelled = labels[~in_scope] != "unknown"
        results[f"{fill:.2f}"] = {
            "answered": f"{int(answered.sum())}/{int(in_scope.sum())}",
            "answered_fraction": float(answered.mean()),
            "correct_when_answered": float(correct.mean()) if answered.any() else None,
            "out_of_scope_confidently_labelled": f"{int(ood_labelled.sum())}/{int((~in_scope).sum())}",
        }
        r = results[f"{fill:.2f}"]
        print(f"{fill:>11.0%}{r['answered']:>12}{f'{r['correct_when_answered']:.1%}' if answered.any() else '-':>24}"
              f"{r['out_of_scope_confidently_labelled']:>24}")

    if args.json:
        with open(args.json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults written to {args.json}")


if __name__ == "__main__":
    main()

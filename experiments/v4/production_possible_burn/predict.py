"""
POST /predict — wound classifier, as a Vercel serverless function.

Same contract as Krish's Flask app so the MRC site's /api/wound proxy needs no
change: multipart form, field `injuryImage`, JPEG or PNG, and a JSON response of
{label, confidence, best_guess, tips, disclaimer}.

WHY TFLITE AND NOT TENSORFLOW
Vercel caps a serverless function at 250 MB unzipped. `tensorflow-cpu` alone is
261 MB, so the original Flask app cannot run here at any size. The LiteRT
runtime is ~20 MB and does the same arithmetic.

Note it is `ai-edge-litert`, not `tflite-runtime`: the latter publishes no wheel
past cp311 and Vercel runs Python 3.13, so it fails at install time.

The conversion was float32, deliberately NOT quantised. Checked against the
original Keras model on 48 held-out test images: identical top-1 on all 48, a
maximum probability difference of 0.000004, and zero cases where the answer
crossed the 0.60 confidence threshold. Dynamic-range quantisation was tried
first and moved probabilities by up to 0.149, which could flip an answer to
"unknown" or back - not a trade worth 6 MB.

The uploaded image is decoded in memory and never written to disk.
"""
import base64
import io
import json
import os

import numpy as np
from flask import Flask, jsonify, request, send_from_directory
from PIL import Image
from ai_edge_litert.interpreter import Interpreter

IMG_SIZE = (224, 224)

# Below this, report "unknown" rather than guessing. The MRC site relies on
# this: it renders a low-confidence answer as "not sure" instead of a category.
# Do not raise it to make numbers look better.
CONFIDENCE_THRESHOLD = 0.60

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(ROOT, "models", "wound_model.tflite")
CLASS_PATH = os.path.join(ROOT, "models", "class_names.json")
# Second model, same architecture, classes and preprocessing, trained on many
# more kinds of non-wound photo (rashes, bites, other skin conditions). It is
# used for one thing only: to turn an answer into "unknown". See decide().
GATE_PATH = os.path.join(ROOT, "models", "out_of_scope_gate.tflite")

# The gate says "not a wound" when it gives out_of_scope at least this
# probability. Same value as GATE_THRESHOLD in Wound-Analyzer/src/model.py,
# chosen there on the validation split.
GATE_THRESHOLD = 0.5

ALLOWED = {"image/jpeg", "image/png"}
MAX_BYTES = 6 * 1024 * 1024

app = Flask(__name__)

# Loaded once per warm container, not per request.
_interpreter = None
_gate = None
_class_names = None


def _load():
    """Both models or neither: a server that silently ran without the gate
    would show confident answers the published figures say it withholds."""
    global _interpreter, _gate, _class_names
    if _interpreter is None:
        interpreter = Interpreter(model_path=MODEL_PATH)
        interpreter.allocate_tensors()
        gate = Interpreter(model_path=GATE_PATH)
        gate.allocate_tensors()
        with open(CLASS_PATH) as f:
            names = json.load(f)
        _interpreter, _gate, _class_names = interpreter, gate, names
    return _interpreter, _gate, _class_names


OUT_OF_SCOPE_CLASS = "out_of_scope"


def _preprocess(raw):
    """Must match how the model was trained and evaluated, exactly:
    tf.keras.utils.load_img(path, target_size=(224, 224)) -> img_to_array ->
    MobileNetV2 preprocess_input. That is RGB, a NEAREST-neighbour resize (the
    load_img default), and pixels scaled from [0, 255] to [-1, 1].

    This used BILINEAR until the out-of-scope model arrived. Different pixels in
    means different answers out, and nothing errors - the model just returns a
    confident, wrong number - so the resize method is not a detail. It mirrors
    load_image() in Wound-Analyzer/src/litert_model.py line for line."""
    img = Image.open(io.BytesIO(raw))
    if img.mode != "RGB":
        img = img.convert("RGB")
    size = (IMG_SIZE[1], IMG_SIZE[0])        # PIL sizes are (width, height)
    if img.size != size:
        img = img.resize(size, Image.NEAREST)
    arr = np.asarray(img, dtype=np.float32)
    return (arr / 127.5 - 1.0)[None, ...]


# The three burn degrees are reported as ONE generic "possible burn" (their probabilities
# summed). The burn-degree labels in the training data are not reliable enough to support a
# degree (Wound-Analyzer experiments/v3), and a generic answer lets the site give one sourced set
# of burn first aid. Measured on the test split (browser-encoded, this pipeline): 197 of 331
# wound photos named correctly vs 119 with separate degrees, 27 wrong vs 41 - but non-wound
# photos given a label rose from 61 to 120 of 450, and healthy feet on an independent set from
# 183 to 560 of 1,613. Chosen by Vihaan on 2026-09-27 (Wound-Analyzer experiments/v4/LEDGER.md).
POSSIBLE_BURN = "possible_burn"
BURN_CLASSES = ("burn_1st_degree", "burn_2nd_degree", "burn_3rd_degree")


def merge_burns(probabilities, class_names):
    """7-class probabilities -> (names, probabilities) with the burn degrees summed."""
    names, probs = [], []
    burn = 0.0
    for n, p in zip(class_names, probabilities):
        if n in BURN_CLASSES:
            burn += float(p)
        else:
            names.append(n); probs.append(float(p))
    names.insert(2, POSSIBLE_BURN); probs.insert(2, burn)
    return names, probs


def decide(probabilities, class_names, gate_probabilities):
    """Port of decide() in Wound-Analyzer/src/model.py, applied to the burn-merged classes:
      - best_guess is the most likely WOUND class (burns summed as possible_burn), never
        out_of_scope
      - confidence is that class's probability
      - label is "unknown" if out_of_scope is the most likely merged class, or the gate gives
        out_of_scope at least GATE_THRESHOLD, or confidence is below CONFIDENCE_THRESHOLD;
        otherwise best_guess
    The gate only ever adds "unknown"; it never changes best_guess or confidence.
    gate_probabilities is required here: the deployed rule always includes the gate."""
    names, probs = merge_burns(probabilities, class_names)
    wound = [i for i, n in enumerate(names) if n != OUT_OF_SCOPE_CLASS]
    best = max(wound, key=lambda i: probs[i])
    best_guess = names[best]
    confidence = float(probs[best])
    out_of_scope = names[int(np.argmax(probs))] == OUT_OF_SCOPE_CLASS
    gated = float(gate_probabilities[class_names.index(OUT_OF_SCOPE_CLASS)]) >= GATE_THRESHOLD
    label = "unknown" if out_of_scope or gated or confidence < CONFIDENCE_THRESHOLD else best_guess
    return label, confidence, best_guess


TIPS = {
    "abrasion": [
        "Rinse the area gently with clean water to remove dirt or debris.",
        "Clean around the wound with mild soap and water.",
        "Apply an antibiotic ointment and cover with a sterile bandage.",
        "Change the dressing daily and watch for signs of infection (increasing redness, swelling, pus, fever).",
    ],
    "bruise": [
        "Apply a cold pack wrapped in a cloth for 15-20 minutes to reduce swelling.",
        "Elevate the area if possible.",
        "Rest the affected area and avoid further impact.",
        "See a doctor if the bruise is unusually large, very painful, or doesn't improve after a couple of weeks.",
    ],
    "cut": [
        "Apply firm, direct pressure with a clean cloth to stop any bleeding.",
        "Rinse the cut with clean water once bleeding slows.",
        "Cover with a sterile bandage.",
        "Seek medical attention for deep cuts, cuts that won't stop bleeding, or cuts near joints or the face.",
    ],
    "burn_1st_degree": [
        "Cool the burn under cool (not ice-cold) running water for about 10-20 minutes.",
        "Do not apply ice, butter, or oily substances.",
        "Cover loosely with a clean, non-stick bandage.",
        "Over-the-counter pain relief can help with discomfort - follow the label's instructions.",
    ],
    "burn_2nd_degree": [
        "Cool the burn under cool running water for about 10-20 minutes.",
        "Do not pop any blisters.",
        "Cover loosely with a sterile, non-stick dressing.",
        "See a doctor, especially for burns larger than 3 inches or on the face, hands, feet, or joints.",
    ],
    # Same wording as the MRC site's wound.tips.possible_burn (sources: Griffin 2020, Griffin
    # 2022, Cuttle 2009, Cuttle 2008, Varley 2016, MedlinePlus - js/sources.js on the site).
    "possible_burn": [
        "Cool the burn under cool (not ice-cold) running tap water for 20 minutes, as soon as you can. It still helps up to 3 hours after the burn.",
        "Take off rings, watches and clothing from the burned area, but leave anything that is stuck to the skin.",
        "Do not put ice, butter, oil or any home remedy on it, and do not break blisters.",
        "Cover it loosely with cling film or a clean non-stick dressing. Call 911 for a burn about the size of your palm or larger, and do not soak a large burn in cold water.",
    ],
    "burn_3rd_degree": [
        "This may be a medical emergency - call emergency services immediately.",
        "Do not remove any stuck clothing, or apply water, ice, or ointments.",
        "Cover loosely with a clean, dry cloth while waiting for help.",
        "Watch for signs of shock (pale skin, rapid breathing) until help arrives.",
    ],
    # Same wording as the Wound-Analyzer app.py this model shipped with.
    "unknown": [
        "This tool couldn't identify the injury. Either the photo was unclear, or the injury isn't one it covers (it only recognizes cuts, scrapes, bruises, and burns).",
        "If the photo was blurry, dark, or taken from far away, you can try again with a clear, well-lit close-up.",
        "A wound that isn't healing, keeps getting bigger, or shows signs of infection (spreading redness, swelling, warmth, pus, or fever) should be checked by a medical professional, especially if you have diabetes or poor circulation.",
        "If you're worried about an injury, see a medical professional regardless of what this tool says.",
    ],
}

DISCLAIMER = (
    "This tool provides general information only and is not a substitute for "
    "professional medical advice. For anything severe, or when in doubt, seek "
    "care from a medical professional."
)


# How much of the frame the second view keeps. A phone photo taken at arm's
# length puts the wound in the middle of a frame full of skin, clothing and
# room; the model was trained on tight crops where the wound fills the picture,
# and its confidence collapses when it does not. Measured on the test split put
# through the browser's own shrink, with the wound filling half the frame:
# answers rose from 18.7% to 39.8% of photos, and accuracy when it answered
# rose from 71.0% to 77.3%. The cost is on the other side - non-wound photos
# given a wound label went from 9.8% to 16.9% - which is why the page still
# says a confident answer is not proof the photo shows one of these injuries.
# 0.55 scored marginally better on recall and worse on false labels; 0.70 is
# the better trade for a tool whose worst outcome is a confident wrong answer.
CENTER_CROP = 0.70

# What the centre crop must reach before it may overrule "not sure". See
# _classify() for the measurements behind this number.
CROP_MIN_CONFIDENCE = 0.80


def _views(img):
    """The frame as sent, plus a centred crop of it."""
    out = [img]
    w, h = img.size
    side = int(min(w, h) * CENTER_CROP)
    if side >= 32:
        left, top = (w - side) // 2, (h - side) // 2
        out.append(img.crop((left, top, left + side, top + side)))
    return out


def _to_tensor(img):
    if img.mode != "RGB":
        img = img.convert("RGB")
    size = (IMG_SIZE[1], IMG_SIZE[0])
    if img.size != size:
        img = img.resize(size, Image.NEAREST)
    arr = np.asarray(img, dtype=np.float32)
    return (arr / 127.5 - 1.0)[None, ...]


def _run(interp, x):
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    interp.set_tensor(inp["index"], x.astype(inp["dtype"]))
    interp.invoke()
    return interp.get_tensor(out["index"])[0]


def _classify(raw):
    """Looks at the whole frame AND a centred crop, and keeps the view the
    classifier is most sure of - as long as that view is not vetoed by the
    out-of-scope class or the gate. Both models always see the SAME pixels as
    each other for any given view."""
    interp, gate, names = _load()
    img = Image.open(io.BytesIO(raw))
    if img.mode != "RGB":
        img = img.convert("RGB")

    views = []
    for view in _views(img):
        x = _to_tensor(view)
        p = _run(interp, x)
        g = _run(gate, x)
        label, confidence, best_guess = decide(p, names, g)
        views.append((label, confidence, best_guess, p,
                      float(g[names.index(OUT_OF_SCOPE_CLASS)])))

    # The frame the person actually sent decides whenever it can. The crop is
    # this server guessing what they meant to photograph, so it only speaks
    # when the real frame had nothing to say, and then only if it is much surer
    # than the ordinary bar - CROP_MIN_CONFIDENCE, not CONFIDENCE_THRESHOLD.
    # Measured on the full test split through the browser's own shrink:
    # letting the crop answer at the ordinary 0.60 bar bought answers at the
    # cost of accuracy (74.4% -> 69.8% right when it answered) and of more
    # non-wound photos labelled as wounds. At 0.80 accuracy is unchanged,
    # phone-framed photos get answered far more often, and the false label rate
    # moves about a point. Lower this bar and you are trading away the accuracy
    # of every answer the tool gives.
    chosen = views[0]
    if chosen[0] == "unknown":
        for candidate in views[1:]:
            if candidate[0] != "unknown" and candidate[1] >= CROP_MIN_CONFIDENCE:
                chosen = candidate
                break

    label, confidence, best_guess, probs, gate_oos = chosen
    # The whole distribution, not just the winner. The standalone app shows it
    # so a reader can see WHY the model is unsure - "48% cut / 44% abrasion" is
    # a materially different situation from "48% cut / 6% everything else", and
    # a single top-1 number hides that difference completely.
    merged_names, merged_probs = merge_burns(probs, names)
    distribution = {n: round(float(p) * 100, 1) for n, p in zip(merged_names, merged_probs)}
    return label, confidence, best_guess, distribution, gate_oos


@app.route("/predict", methods=["POST", "OPTIONS"])
@app.route("/api/predict", methods=["POST", "OPTIONS"])
def predict_route():
    if request.method == "OPTIONS":
        return ("", 204)

    raw = None

    # Multipart, matching the original Flask app.
    if "injuryImage" in request.files:
        f = request.files["injuryImage"]
        if f.filename == "":
            return jsonify({"error": "No file selected."}), 400
        if f.mimetype and f.mimetype.lower() not in ALLOWED:
            return jsonify({"error": "Unsupported file type. Please upload a PNG or JPG image."}), 415
        raw = f.read()
    else:
        # Base64 JSON, so the endpoint can also be called directly.
        body = request.get_json(silent=True) or {}
        image = body.get("image")
        if isinstance(image, str) and image.startswith("data:"):
            try:
                header, b64 = image.split(",", 1)
                if header.split(";")[0][5:].lower() not in ALLOWED:
                    return jsonify({"error": "Unsupported file type."}), 415
                raw = base64.b64decode(b64)
            except Exception:
                return jsonify({"error": "Could not read image."}), 400

    if not raw:
        return jsonify({"error": "No file uploaded."}), 400
    if len(raw) > MAX_BYTES:
        return jsonify({"error": "Image too large."}), 413

    try:
        label, confidence, best_guess, distribution, gate_oos = _classify(raw)
    except Exception as e:
        # Message only. The body is a picture of somebody's injury.
        print("[predict] failed:", type(e).__name__)
        return jsonify({"error": "Could not process image."}), 500

    return jsonify({
        "label": label,
        "confidence": round(confidence * 100, 1),
        "best_guess": best_guess,
        # Additive fields. The MRC site ignores them; the standalone app uses
        # them. Nothing existing changes shape.
        "probabilities": distribution,
        "threshold": round(CONFIDENCE_THRESHOLD * 100, 1),
        # The gate's verdict. Without these a reader could see "not sure" next
        # to a 90% bar above the 60% line and no explanation.
        "gated": gate_oos >= GATE_THRESHOLD,
        "gate_out_of_scope": round(gate_oos * 100, 1),
        "gate_threshold": round(GATE_THRESHOLD * 100, 1),
        "tips": TIPS.get(label, TIPS["unknown"]),
        "disclaimer": DISCLAIMER,
    })


@app.route("/health", methods=["GET"])
@app.route("/api/health", methods=["GET"])
@app.route("/api/predict", methods=["GET"])
def health():
    """Machine-readable status. This used to live at "/", which now serves the
    app itself - monitoring should point at /health."""
    try:
        _, _, names = _load()
        return jsonify({
            "ok": True,
            "classes": names,
            "reported_classes": merge_burns([0.0] * len(names), names)[0],
            "runtime": "litert",
            "gate": True,
            "threshold": round(CONFIDENCE_THRESHOLD * 100, 1),
            "gate_threshold": round(GATE_THRESHOLD * 100, 1),
        })
    except Exception as e:
        return jsonify({"ok": False, "error": type(e).__name__}), 500


@app.route("/", methods=["GET"])
def home():
    """The standalone Wound Analyzer app.

    Served by Flask rather than dropped in a static directory because this
    project deploys through Vercel's Flask entrypoint - every request reaches
    the WSGI app, so a file sitting in public/ would not reliably be routed.
    One self-contained file means no asset routes to maintain."""
    try:
        return send_from_directory(os.path.join(ROOT, "web"), "index.html")
    except Exception:
        # Never let a missing UI file take the API down with it.
        return jsonify({"ok": True, "note": "API only; UI not deployed",
                        "endpoints": ["/predict", "/health"]}), 200

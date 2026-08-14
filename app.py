"""
app.py

Flask backend for the Wound Analyzer frontend. Serves the upload page and
exposes a /predict endpoint that runs an uploaded image through the trained
model (see src/model.py) and returns the predicted category + general first
aid pointers as JSON.

Place this file at the project root (next to the src/, models/, and data/
folders) - see the layout comment at the top of the conversation.

Usage:
    pip install flask
    python app.py
    Then open http://127.0.0.1:5000 in your browser.
"""

import os
import sys
import tempfile

from flask import Flask, request, jsonify, render_template

# Make src/ importable so we can reuse model.py's build/load/predict logic
# instead of duplicating it here.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))   # project root
SRC_DIR = os.path.join(SCRIPT_DIR, "src")
sys.path.insert(0, SRC_DIR)

from model import load_trained_model, predict  # noqa: E402

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB upload limit

ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg"}

# Load the model once at startup rather than on every request - loading a
# Keras model from disk is slow, so this makes /predict fast per-request.
print("Loading trained model...")
MODEL, CLASS_NAMES = load_trained_model()
print(f"Model loaded. Classes: {CLASS_NAMES}")

# General first-aid pointers per category. This is intentionally basic,
# widely-known first aid information, NOT medical advice - severe cases are
# explicitly routed to "seek medical attention" / emergency care.
TREATMENT_TIPS = {
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
    "burn_3rd_degree": [
        "This may be a medical emergency - call emergency services immediately.",
        "Do not remove any stuck clothing, or apply water, ice, or ointments.",
        "Cover loosely with a clean, dry cloth while waiting for help.",
        "Watch for signs of shock (pale skin, rapid breathing) until help arrives.",
    ],
    "unknown": [
        "The model wasn't confident enough to classify this image reliably.",
        "Try a clearer, well-lit, close-up photo of the injury.",
        "If you're concerned about an injury, consult a medical professional regardless of what this app says.",
    ],
}

GENERIC_DISCLAIMER = (
    "This tool provides general information only and is not a substitute for "
    "professional medical advice. For anything severe, or when in doubt, seek "
    "care from a medical professional."
)


def allowed_file(filename):
    ext = os.path.splitext(filename)[1].lower()
    return ext in ALLOWED_EXTENSIONS


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict_route():
    if "injuryImage" not in request.files:
        return jsonify({"error": "No file uploaded."}), 400

    file = request.files["injuryImage"]
    if file.filename == "":
        return jsonify({"error": "No file selected."}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Unsupported file type. Please upload a PNG or JPG image."}), 400

    # Save to a temp file for inference, then delete it immediately after -
    # we don't keep uploaded images around any longer than needed to predict.
    ext = os.path.splitext(file.filename)[1].lower()
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        temp_path = tmp.name
        file.save(temp_path)

    try:
        label, confidence, best_guess = predict(temp_path, model=MODEL, class_names=CLASS_NAMES)
    except Exception as e:
        return jsonify({"error": f"Could not process image: {e}"}), 500
    finally:
        os.remove(temp_path)

    tips = TREATMENT_TIPS.get(label, TREATMENT_TIPS["unknown"])

    return jsonify({
        "label": label,
        "confidence": round(confidence * 100, 1),
        "best_guess": best_guess,
        "tips": tips,
        "disclaimer": GENERIC_DISCLAIMER,
    })


if __name__ == "__main__":
    app.run(debug=True)
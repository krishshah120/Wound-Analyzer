"""
roboflow_client.py

Calls the Roboflow Workflow "wound vwound-ebsdw-4atst-1-yolo26x-t1 Logic" (a YOLO26 X-Large wound
detector, project wound-ebsdw-4atst v1) on Roboflow's serverless API, and turns its detections into
one answer in the same shape as model.decide(). Used for the comparison in experiments/rf/ - it is
NOT part of the production pipeline.

Standard library only: the official inference-sdk requires numpy < 2.4 and opencv, which conflicts
with this project's TensorFlow environment (numpy 2.5.3).

The API key is read from the ROBOFLOW_API_KEY environment variable or, failing that, the file
~/.config/wound-analyzer/roboflow_api_key. It is sent only in the Authorization header - never in
the URL or body - and is never printed or logged. Keep it server-side: never ship it to a browser.

Privacy: every call sends the photo to Roboflow's servers.
"""

import base64
import json
import os
import time
import urllib.error
import urllib.request

WORKSPACE = "vihaan-nr-singh-gmail-com"
WORKFLOW_ID = "wound-vwound-ebsdw-4atst-1-yolo26x-t1-logic"
API_URL = "https://serverless.roboflow.com"
KEY_FILE = os.path.expanduser("~/.config/wound-analyzer/roboflow_api_key")

# Roboflow class -> this project's shared category. "Burn" carries no degree, so it maps to a
# generic "possible burn". Blister and "no abnormality" map to nothing: they can make an answer
# uncertain, and "no abnormality" is never reported as "healthy".
CATEGORY = {"Abrasions": "abrasion", "Bruise": "bruise", "Cut": "cut", "Burn": "possible_burn"}
NON_ANSWER_CLASSES = {"Blister", "no abnormality"}


class RoboflowError(Exception):
    """Base class for every failure of a Roboflow call."""


class RoboflowAuthError(RoboflowError):
    """No API key found, or Roboflow rejected it (HTTP 401/403)."""


class RoboflowRequestError(RoboflowError):
    """Roboflow rejected the request itself (other HTTP 4xx) - retrying will not help."""


class RoboflowUnavailableError(RoboflowError):
    """Timeouts, HTTP 429 or 5xx that persisted through every retry."""


class RoboflowResponseError(RoboflowError):
    """The response did not have the expected structure."""


def load_api_key():
    key = os.environ.get("ROBOFLOW_API_KEY", "").strip()
    if not key and os.path.exists(KEY_FILE):
        with open(KEY_FILE) as f:
            key = f.read().strip()
    if not key:
        raise RoboflowAuthError(
            f"No Roboflow API key: set ROBOFLOW_API_KEY or put the key in {KEY_FILE} "
            "(find it at app.roboflow.com/settings/api)."
        )
    return key


def run_wound_workflow(image_bytes, parameters=None, timeout=30.0, retries=2, backoff_seconds=1.0, api_key=None):
    """Runs the workflow on one image (JPEG/PNG bytes). Returns the list of output dicts (one per
    input image), keyed by the workflow's own output names. Retries timeouts, 429 and 5xx with
    exponential backoff; raises a RoboflowError subclass on failure."""
    key = api_key or load_api_key()
    inputs = {"image": {"type": "base64", "value": base64.b64encode(image_bytes).decode("ascii")}}
    inputs.update(parameters or {})
    body = json.dumps({"inputs": inputs}).encode("utf-8")
    url = f"{API_URL}/{WORKSPACE}/workflows/{WORKFLOW_ID}"
    last = None
    for attempt in range(retries + 1):
        request = urllib.request.Request(url, data=body, method="POST", headers={
            "Content-Type": "application/json", "Authorization": f"Bearer {key}"})
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
            break
        except urllib.error.HTTPError as e:
            detail = e.read(500).decode("utf-8", "replace")
            if e.code in (401, 403):
                raise RoboflowAuthError(f"Roboflow rejected the API key (HTTP {e.code}).") from None
            if e.code == 429 or e.code >= 500:
                last = RoboflowUnavailableError(f"HTTP {e.code}: {detail}")
            else:
                raise RoboflowRequestError(f"HTTP {e.code}: {detail}") from None
        except (urllib.error.URLError, TimeoutError) as e:
            last = RoboflowUnavailableError(f"network error: {e}")
        if attempt < retries:
            time.sleep(backoff_seconds * 2 ** attempt)
    else:
        raise last
    outputs = payload.get("outputs") if isinstance(payload, dict) else None
    if not isinstance(outputs, list):
        raise RoboflowResponseError(f"expected a JSON object with an 'outputs' list, got keys {list(payload)[:10] if isinstance(payload, dict) else type(payload).__name__}")
    return outputs


def _walk(value):
    if isinstance(value, dict):
        yield value
        for v in value.values():
            yield from _walk(v)
    elif isinstance(value, list):
        for v in value:
            yield from _walk(v)


def find_detections(output):
    """Every object-detection prediction anywhere in one workflow output (output names are not
    assumed). Keeps only class, confidence and the box; drops polygon 'points' and everything else."""
    found = []
    for node in _walk(output):
        preds = node.get("predictions")
        if isinstance(preds, list) and preds and all(isinstance(p, dict) and "class" in p and "confidence" in p for p in preds):
            found += [{"class": p["class"], "confidence": float(p["confidence"]),
                       "box": [p.get("x"), p.get("y"), p.get("width"), p.get("height")]} for p in preds]
    return found


def save_image_outputs(output, directory, stem):
    """Writes every base64 image in an output to disk and replaces it in place by its file path,
    so large blobs are never logged or kept in memory. Returns the list of files written."""
    written = []
    for node in list(_walk(output)):
        for k, v in list(node.items()):
            if isinstance(v, dict) and v.get("type") == "base64" and isinstance(v.get("value"), str):
                os.makedirs(directory, exist_ok=True)
                path = os.path.join(directory, f"{stem}_{k}_{len(written)}.jpg")
                with open(path, "wb") as f:
                    f.write(base64.b64decode(v["value"]))
                node[k] = {"type": "file", "path": path}
                written.append(path)
    return written


def answer_from_detections(detections, threshold):
    """One website answer from detections (experiments/rf/PROTOCOL.md section 2).
    Returns (label, confidence, top_category): label is a shared category or "unknown".
    No detection means "unknown" - never "healthy"."""
    kept = [d for d in detections if d["confidence"] >= threshold]
    shared = sorted((d for d in kept if d["class"] in CATEGORY), key=lambda d: -d["confidence"])
    if not shared:
        return "unknown", 0.0, None
    top = shared[0]; category = CATEGORY[top["class"]]
    blocked = any(d["class"] in NON_ANSWER_CLASSES and d["confidence"] >= top["confidence"] for d in kept)
    conflict = any(CATEGORY[d["class"]] != category for d in shared[1:])
    return ("unknown" if blocked or conflict else category), top["confidence"], category

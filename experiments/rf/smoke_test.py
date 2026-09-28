"""Smoke test for src/roboflow_client.py.
Offline part (always): decision rule and parsing on Roboflow's documented detection shape.
Live part (only when a key is configured): one real workflow call on one sample photo; asserts the
response is a one-entry list of dicts and that some output carries a detection 'predictions' field;
writes the response SHAPE (keys and types, no values, no image data) to workflow_response_shape.json.
Exit codes: 0 all passed, 2 live part skipped (no key), 1 failure."""
import os, sys, json, tempfile
RF = os.path.dirname(os.path.abspath(__file__)); WT = os.path.dirname(os.path.dirname(RF))
sys.path.insert(0, f"{WT}/src")
import roboflow_client as rc

def det(c, conf): return {"class": c, "confidence": conf, "box": [0, 0, 1, 1]}
A = rc.answer_from_detections
assert A([], 0.3)[0] == "unknown", "no detection must be unknown, never healthy"
assert A([det("no abnormality", 0.9)], 0.3)[0] == "unknown"
assert A([det("Burn", 0.7)], 0.3)[0] == "possible_burn"
assert A([det("Burn", 0.2)], 0.3)[0] == "unknown", "below threshold"
assert A([det("Burn", 0.7), det("Cut", 0.5)], 0.3)[0] == "unknown", "conflicting categories"
assert A([det("Burn", 0.7), det("Burn", 0.5)], 0.3)[0] == "possible_burn", "same category twice is fine"
assert A([det("Cut", 0.6), det("no abnormality", 0.6)], 0.3)[0] == "unknown", "equally confident 'no abnormality' blocks"
assert A([det("Cut", 0.6), det("Blister", 0.4)], 0.3)[0] == "cut"
nested = [{"model_predictions": {"image": {"width": 640, "height": 480},
                                 "predictions": [{"x": 1, "y": 2, "width": 3, "height": 4, "class": "Bruise", "confidence": 0.8,
                                                  "class_id": 2, "detection_id": "u", "points": [[0, 0]] * 500}]},
           "annotated": {"type": "base64", "value": "/9j/4AAQ"}}]
d = rc.find_detections(nested[0])
assert d == [{"class": "Bruise", "confidence": 0.8, "box": [1, 2, 3, 4]}], d
with tempfile.TemporaryDirectory() as tdir:
    files = rc.save_image_outputs(nested[0], tdir, "t")
    assert len(files) == 1 and os.path.exists(files[0]) and nested[0]["annotated"]["type"] == "file"
print("offline checks passed")

try:
    key = rc.load_api_key()
except rc.RoboflowAuthError as e:
    print("SKIPPED live check:", e); sys.exit(2)
sample = f"{WT}/data/test/burn_2nd_degree/" + sorted(os.listdir(f"{WT}/data/test/burn_2nd_degree"))[0]
out = rc.run_wound_workflow(open(sample, "rb").read(), api_key=key)
assert isinstance(out, list) and len(out) == 1 and isinstance(out[0], dict), f"unexpected outputs: {type(out)}"
with tempfile.TemporaryDirectory() as tdir:
    rc.save_image_outputs(out[0], tdir, "smoke")          # decode image blobs to disk, never print them
def shape(v):
    if isinstance(v, dict):   # a detection list is summarised by its item keys; nested dicts are walked
        return {k: (f"list[{len(x)}] of {sorted(x[0].keys()) if x else []}" if k == "predictions" and isinstance(x, list) else shape(x)) for k, x in v.items()}
    if isinstance(v, list): return f"list[{len(v)}]" + (f" of {shape(v[0])}" if v and isinstance(v[0], (dict, list)) else "")
    return type(v).__name__
json.dump(shape(out), open(f"{RF}/workflow_response_shape.json", "w"), indent=1)
has_pred_field = any("predictions" in n for n in rc._walk(out[0]))
assert has_pred_field, "no output carries a 'predictions' field - inspect workflow_response_shape.json"
print("live check passed: output keys", sorted(out[0].keys()), "| detections:", rc.find_detections(out[0]))

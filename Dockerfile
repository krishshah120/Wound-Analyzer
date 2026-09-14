# Container for the /predict server (app.py), for Google Cloud Run.
# It serves models/wound_model.tflite with LiteRT (src/litert_model.py), not
# TensorFlow: importing TensorFlow made a cold start take ~25 s. Run
# src/export_tflite.py after every retrain so the .tflite file is current.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    WOUND_MODEL_FORMAT=tflite

WORKDIR /app
COPY requirements-server.txt .
RUN pip install --no-cache-dir -r requirements-server.txt

COPY app.py .
COPY src/model.py src/litert_model.py src/
COPY models/wound_model.tflite models/class_names.json models/
COPY templates templates

# Cloud Run sends requests to $PORT. One process, so the model is loaded into
# memory once; a few threads so a slow upload does not block other requests.
# Do NOT add --preload: when this server used TensorFlow, it made every
# /predict hang after gunicorn forked the worker.
CMD exec gunicorn --bind :$PORT --workers 1 --threads 4 --timeout 60 app:app

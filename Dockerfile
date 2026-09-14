# Container for the /predict server (app.py), for Google Cloud Run.
# Only the server, src/model.py and the trained model go in - no datasets.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    TF_CPP_MIN_LOG_LEVEL=2

WORKDIR /app
COPY requirements-server.txt .
RUN pip install --no-cache-dir -r requirements-server.txt

COPY app.py .
COPY src/model.py src/
COPY models/wound_model.keras models/class_names.json models/
COPY templates templates

# Cloud Run sends requests to $PORT. One process, so the model is loaded into
# memory once; a few threads so a slow upload does not block other requests.
# Do NOT add --preload: it loads TensorFlow in the parent process before the
# worker is forked, and TensorFlow does not survive a fork - every /predict
# then hangs until the 60 s timeout (seen on Cloud Run and reproduced locally).
CMD exec gunicorn --bind :$PORT --workers 1 --threads 4 --timeout 60 app:app

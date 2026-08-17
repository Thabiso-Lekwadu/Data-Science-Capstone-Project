# Streamlit dashboard container. Reads from the SAME named volume the
# pipeline container writes to (mounted read-only here — this container
# never needs to write into /app/data). With AUTO_LOAD_FROM_DISK=true set
# below, App.py auto-loads whatever's already in the volume on startup —
# no manual upload needed, per the app's dual-mode design (uploads still
# work and override the auto-loaded version for the current session).
#
# Place this file at the root of your Streamlit app repo (same level as
# App.py, model_utils.py, pages_*.py).
#
# Build: docker build -t crime-hotspot-app .
# Run:   docker run --rm -p 8501:8501 -v pipeline_data:/app/data:ro crime-hotspot-app

FROM python:3.12-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    AUTO_LOAD_FROM_DISK=true \
    APP_DATA_DIR=/app/data

COPY requirements.docker.txt ./requirements.txt
RUN pip install -r requirements.docker.txt

# App code only — no conf/, no src/, this is a standalone repo.
COPY ../App.py model_utils.py feature_engineering.py ./
COPY ../pages_eda.py pages_model.py pages_predict.py pages_shap.py ./

# /app/data is where the shared volume mounts (read-only — see docker-compose.yml).
VOLUME ["/app/data"]

EXPOSE 8501

# --server.headless keeps Streamlit from trying to open a local browser or
# prompt for an email inside the container; --server.address=0.0.0.0 is what
# actually makes the port reachable from outside the container.
ENTRYPOINT ["streamlit", "run", "App.py", \
            "--server.port=8501", \
            "--server.address=0.0.0.0", \
            "--server.headless=true"]
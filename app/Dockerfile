# =============================================================================
# Streamlit dashboard container. Reads from the SAME named volume the pipeline
# container writes to (mounted read-only here). AUTO_LOAD_FROM_DISK=true makes
# App.py auto-load whatever the pipeline wrote to the volume on startup — no
# manual upload needed (uploads still work and override for the session).
#
# Build context: this directory (app/). Everything COPY'd is under app/, so
# there is no `COPY ../` reaching outside the build context (which Docker
# forbids — that was the bug in the previous streamlit-dashboard/Dockerfile).
# =============================================================================
FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    AUTO_LOAD_FROM_DISK=true \
    APP_DATA_DIR=/app/data

# libgomp1 is the OpenMP runtime LightGBM / XGBoost need to unpickle+predict.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install -r requirements.txt

# App code only (no conf/, no src/ — the app is standalone).
COPY App.py model_utils.py feature_engineering.py ./
COPY views/ ./views/

# /app/data is the shared volume mount (read-only — see docker-compose.yml).
VOLUME ["/app/data"]

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=5 \
    CMD curl -fsS http://localhost:8501/_stcore/health || exit 1

# --server.headless stops Streamlit prompting for an email / opening a browser
# inside the container; --server.address=0.0.0.0 makes the port reachable.
ENTRYPOINT ["streamlit", "run", "App.py", \
            "--server.port=8501", \
            "--server.address=0.0.0.0", \
            "--server.headless=true"]

# =============================================================================
# Pipeline container — runs the Kedro pipeline (offline, from data/01_raw) on
# startup and writes every artifact (processed data, engineered features,
# trained models, reporting tables) into the shared /app/data volume that the
# Streamlit app then reads.
#
# Build context: this directory (crime-hotspot-prediction-project/).
# =============================================================================
FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    # disable kedro-telemetry (no network / no prompt inside the container)
    DO_NOT_TRACK=1 \
    KEDRO_DISABLE_TELEMETRY=true

# libgomp1 is the OpenMP runtime that LightGBM / XGBoost link against.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies first (better layer caching).
COPY requirements.txt ./
RUN pip install -r requirements.txt

# Project source + config.
COPY pyproject.toml README.md ./
COPY src/ ./src/
COPY conf/ ./conf/

# Bake the raw Excel inputs as a read-only SEED; the entrypoint copies them
# into the (initially empty) shared /app/data volume on startup.
COPY data/01_raw/ ./seed/01_raw/

# Minimal local credentials so the catalog resolves offline. The Quantec
# APIDataset entries are never loaded by the `training_from_raw` pipeline, so
# this placeholder is never actually used — it just keeps config resolution
# happy. Real credentials belong in .env (see .env.example), not here.
RUN mkdir -p conf/local \
    && printf 'quantec_credentials:\n  api_key: "unused-offline"\n  url: "https://www.easydata.co.za/api/v3/"\n' \
       > conf/local/credentials.yml

# Install the package (import-only) so `kedro run` can resolve it. --no-deps
# because the real runtime deps are already installed from requirements.txt;
# this avoids re-pulling the loose/heavy Jupyter deps in pyproject.toml.
RUN pip install -e . --no-deps

COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

# The shared volume mounts here (declared in docker-compose.yml).
VOLUME ["/app/data"]

ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]

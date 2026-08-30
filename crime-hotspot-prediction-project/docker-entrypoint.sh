#!/usr/bin/env bash
# Pipeline container entrypoint: seed raw data into the shared volume, then run
# the offline Kedro pipeline (preprocessing -> feature engineering -> model
# training + reporting). All outputs land under /app/data (the shared volume).
set -euo pipefail

echo "[pipeline] seeding raw inputs into shared volume (/app/data/01_raw) ..."
mkdir -p /app/data/01_raw
# Copy the baked raw Excel files in without clobbering anything already present.
cp -rn /app/seed/01_raw/. /app/data/01_raw/ 2>/dev/null || true

echo "[pipeline] running: kedro run --pipeline training_from_raw"
kedro run --pipeline training_from_raw

echo "[pipeline] DONE. Processed data, models and reports are in /app/data."

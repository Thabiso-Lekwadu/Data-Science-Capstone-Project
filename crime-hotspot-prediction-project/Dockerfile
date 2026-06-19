# ─────────────────────────────────────────────────────────────────────────────
# Crime Hotspot Intelligence — Streamlit App
# Build:  docker build -t crime-hotspot-app .
# Run:    docker run -p 8501:8501 \
#           -v /path/to/your/kedro/project/data:/app/data:ro \
#           crime-hotspot-app
#
# The app reads from /app/data which mirrors your Kedro data/ directory.
# If the files are absent it falls back to synthetic data so the app always
# starts. Mount your real data at runtime with the -v flag above.
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY App.py .
COPY pages_eda.py .
COPY pages_model.py .
COPY pages_bi.py .

# Streamlit config — disable the default welcome page, set dark theme
RUN mkdir -p /root/.streamlit
RUN cat > /root/.streamlit/config.toml << 'EOF'
[server]
headless = true
port = 8501
enableCORS = false
enableXsrfProtection = false

[theme]
base = "dark"
backgroundColor = "#070b14"
secondaryBackgroundColor = "#0c1220"
textColor = "#c9d1d9"
font = "sans serif"

[browser]
gatherUsageStats = false
EOF

# Data directory — files mounted here at runtime via -v
# Structure expected (mirrors your Kedro data/ folder):
#   /app/data/03_primary/master_dataset.xlsx
#   /app/data/08_reporting/experiment_comparison.xlsx
RUN mkdir -p /app/data/03_primary /app/data/08_reporting

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
  CMD curl -f http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
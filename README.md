# Crime Hotspot Prediction — Limpopo (SAPS × Quantec)

An end-to-end data-engineering + machine-learning capstone that predicts
**Crime Count** per police cluster and crime type in Limpopo, South Africa,
and tests whether adding **socioeconomic context** (population density, poor
households, unemployment, education) improves those predictions.

> The original project brief (which framed this as *classifying* the cluster)
> is kept for provenance at [`docs/original-project-brief.md`](docs/original-project-brief.md).
> The project was reframed to **regression on Crime Count** — see below.

The project has two deployables that run together via Docker Compose:

| Component | Path | What it does |
|-----------|------|--------------|
| **Kedro pipeline** | [`crime-hotspot-prediction-project/`](crime-hotspot-prediction-project/) | Ingests → cleans → feature-engineers → trains & evaluates 4 tree-ensemble models under 2 conditions, with walk-forward CV and a paired significance test. |
| **Streamlit app** | [`app/`](app/) | Interactive dashboard: EDA, SHAP feature importance, model performance, and a prediction explorer. Reads the artifacts the pipeline produces. |

## What the model actually does

- **Task:** regression — predict `Crime Count` for a `(Cluster, Type of Crime, year)`.
- **Models:** Random Forest, XGBoost, LightGBM, CatBoost (all tree ensembles,
  so no scaling / log / PCA is needed).
- **Two conditions:** `crime_only` (baseline) vs `master` (crime + socioeconomic).
- **Validation:** expanding-window **walk-forward** cross-validation (respects
  time order), with a **paired t-test** per model asking whether the enriched
  condition gives a significantly lower RMSE across the same folds.

## Quick start — Docker (recommended)

```bash
docker compose up --build
```

1. The **pipeline** service runs the offline Kedro pipeline
   (`training_from_raw`) from the raw Excel files baked into its image,
   writing processed data, engineered features, trained `.pkl` models and
   reporting tables into a shared volume, then exits.
2. The **app** service waits for the pipeline to finish, then serves the
   dashboard, auto-loading those artifacts from the same volume.

Open **http://localhost:8501**.

To regenerate everything from scratch:

```bash
docker compose down -v && docker compose up --build
```

## Running locally (without Docker)

Pipeline:

```bash
cd crime-hotspot-prediction-project
pip install -r requirements.txt
pip install -e . --no-deps
kedro run --pipeline training_from_raw     # offline, from data/01_raw
# or `kedro run` for the full pipeline incl. the live Quantec API pull
```

App (after the pipeline has produced artifacts):

```bash
cd app
pip install -r requirements.txt
# point the app at the pipeline's data dir and let it auto-load:
AUTO_LOAD_FROM_DISK=true \
APP_DATA_DIR=../crime-hotspot-prediction-project/data \
streamlit run App.py
```

Without `AUTO_LOAD_FROM_DISK`, the app is strictly upload-driven — use the
sidebar to upload the datasets, models and reporting files.

## Live data ingestion (optional)

The `data_ingestion` pipeline pulls from the Quantec EasyData API and needs
credentials. Copy `crime-hotspot-prediction-project/.env.example` to `.env`
and fill in `QUANTEC_API_KEY`. The default `docker compose up` does **not**
need this — it runs the offline path from the raw Excel already in
`data/01_raw`.

## Repository layout

```
.
├─ app/                              # Streamlit dashboard (standalone deployable)
│  ├─ App.py                         # entrypoint + routing
│  ├─ feature_engineering.py         # single app-side FE source of truth
│  ├─ model_utils.py                 # model loading, schema alignment, cached FE
│  ├─ views/                         # one module per page (eda, model, predict, shap)
│  ├─ requirements.txt
│  └─ Dockerfile
├─ crime-hotspot-prediction-project/ # Kedro pipeline project
│  ├─ src/…/pipelines/               # data_ingestion, data_preprocessing,
│  │                                 #   feature_engineering, model_training
│  ├─ conf/                          # catalog, parameters, credentials template
│  ├─ data/                          # Kedro data layers (01_raw kept; rest derived)
│  ├─ tests/
│  ├─ requirements.txt
│  ├─ Dockerfile + docker-entrypoint.sh
│  └─ .env.example
├─ docs/                             # proposal, workflow docs, articles, PLAN
├─ docker-compose.yml
└─ README.md
```

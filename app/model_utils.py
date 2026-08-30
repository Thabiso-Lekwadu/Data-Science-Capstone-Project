"""Shared utilities for the Streamlit app: a Streamlit-cached wrapper around
the shared feature engineering (``feature_engineering.py``) + uploaded-model
handling for the SHAP Explainer and Prediction Explorer pages.

Feature engineering itself lives in ``feature_engineering.py`` (pure pandas,
the single app-side source of truth) so there is no second copy to drift.

Everything here is upload/volume-only — if the person hasn't supplied a file,
the corresponding return value is None/empty and the calling page shows
nothing rather than falling back to any default.
"""
from __future__ import annotations

import io
import pickle
from typing import Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

from feature_engineering import (
    DATE_COL,
    CLUSTER_COL,
    CRIME_TYPE_COL,
    TARGET_COL,
    DEFAULT_LAGS,
    DEFAULT_ROLLING_WINDOWS,
    engineer_features_df,
)

MODEL_NAMES = ["RandomForest", "XGBoost", "LightGBM", "CatBoost"]
CONDITIONS = ["crime_only", "master"]

# Filenames the Kedro pipeline saves, e.g. "randomforest_master.pkl" —
# used to match an uploaded/volume batch of .pkl files back to (model, condition).
MODEL_FILE_PATTERN = {
    (m, c): f"{m.lower()}_{c}.pkl" for m in MODEL_NAMES for c in CONDITIONS
}


# ---------------------------------------------------------------------------
# Feature engineering — thin Streamlit-cached wrapper over feature_engineering
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def engineer_features(raw_json: str, lags: Tuple[int, ...] = tuple(DEFAULT_LAGS),
                      rolling_windows: Tuple[int, ...] = tuple(DEFAULT_ROLLING_WINDOWS)) -> pd.DataFrame:
    """Cached by the raw data's JSON content, so re-running on the same upload
    is instant and a genuinely different upload always recomputes."""
    df = pd.read_json(io.StringIO(raw_json))
    return engineer_features_df(df, list(lags), list(rolling_windows))


def required_raw_columns_present(df: pd.DataFrame) -> Optional[str]:
    """Returns an error string if `df` is missing what feature engineering
    needs, else None."""
    required = {DATE_COL, CLUSTER_COL, CRIME_TYPE_COL, TARGET_COL}
    missing = required - set(df.columns)
    if missing:
        return f"Missing required column(s) for feature engineering: {sorted(missing)}"
    return None


# ---------------------------------------------------------------------------
# Uploaded models
# ---------------------------------------------------------------------------

def load_uploaded_models(uploaded_files) -> Tuple[Dict[Tuple[str, str], object], List[str]]:
    """`uploaded_files` is the list from a multi-file st.file_uploader.
    Matches each file's name against "<model>_<condition>.pkl" (case-
    insensitive) and unpickles it. Returns (models keyed by (model, condition),
    list of filenames that didn't match any known pattern)."""
    models: Dict[Tuple[str, str], object] = {}
    unmatched: List[str] = []
    if not uploaded_files:
        return models, unmatched

    reverse_lookup = {v.lower(): k for k, v in MODEL_FILE_PATTERN.items()}
    for f in uploaded_files:
        key = reverse_lookup.get(f.name.lower())
        if key is None:
            unmatched.append(f.name)
            continue
        try:
            f.seek(0)
            models[key] = pickle.load(f)
        except Exception:
            unmatched.append(f.name)
    return models, unmatched


def models_for_condition(models: Dict[Tuple[str, str], object], condition: str) -> Dict[str, object]:
    return {m: models[(m, c)] for (m, c) in models if c == condition}


# ---------------------------------------------------------------------------
# Feature schema helpers (used by SHAP + Predict pages)
# ---------------------------------------------------------------------------

def feature_columns(features_df: pd.DataFrame) -> list:
    """Columns the models were actually trained on (X)."""
    return [c for c in features_df.columns if c not in (TARGET_COL, DATE_COL)]


def cluster_and_crime_type_columns(cols: list) -> tuple:
    cluster_cols = [c for c in cols if c.startswith("cluster_")]
    crime_type_cols = [c for c in cols if c.startswith("crime_type_")]
    clusters = sorted(c[len("cluster_"):] for c in cluster_cols)
    crime_types = sorted(c[len("crime_type_"):] for c in crime_type_cols)
    return clusters, crime_types


def build_input_row(feature_cols: list, cluster: str, crime_type: str, values: dict) -> pd.DataFrame:
    """Assemble a single-row DataFrame matching `feature_cols` exactly:
    one-hot the chosen cluster/crime type, fill everything else from
    `values` (defaulting to 0 for anything not supplied)."""
    row = {c: 0 for c in feature_cols}
    for c in feature_cols:
        if c == f"cluster_{cluster}":
            row[c] = 1
        elif c == f"crime_type_{crime_type}":
            row[c] = 1
        elif c in values:
            row[c] = values[c]
    return pd.DataFrame([row], columns=feature_cols)


# ---------------------------------------------------------------------------
# Column-order alignment
# ---------------------------------------------------------------------------
#
# Tree libraries (CatBoost especially) validate the input DataFrame's column
# ORDER against what was recorded at training time and raise a hard error on
# any mismatch — e.g. "At position 0 should be feature with name year (found
# population_density)". get_dummies' column ordering isn't guaranteed
# stable across pandas versions, and the trained models went through an
# Excel round-trip (Kedro's ExcelDataset) between feature engineering and
# training, so the in-app engineered matrix's column order is not something
# to rely on positionally. Reindexing to the model's own recorded feature
# names — not the order our own engineering happened to produce — is what
# actually fixes this, for every model type, not just CatBoost.
def align_features_to_model(X: pd.DataFrame, model) -> pd.DataFrame:
    feature_names = None
    for attr in ("feature_names_", "feature_names_in_"):
        if hasattr(model, attr):
            names = getattr(model, attr)
            if names is not None:
                feature_names = list(names)
                break
    if feature_names is None and hasattr(model, "get_booster"):
        try:
            feature_names = model.get_booster().feature_names
        except Exception:
            feature_names = None
    if feature_names is None and hasattr(model, "booster_"):
        try:
            feature_names = list(model.booster_.feature_name())
        except Exception:
            feature_names = None

    if not feature_names:
        # Model doesn't expose its training-time feature names (very old
        # library version) — nothing to align to, pass through as-is.
        return X

    missing = [f for f in feature_names if f not in X.columns]
    if missing:
        preview = ", ".join(missing[:10]) + (", ..." if len(missing) > 10 else "")
        raise ValueError(
            f"This model was trained on {len(feature_names)} features, but the uploaded/engineered "
            f"data is missing {len(missing)} of them: {preview}. This usually means the uploaded raw "
            f"dataset's schema (or the lag/rolling parameters) doesn't match what the model was "
            f"actually trained on."
        )
    return X[feature_names]

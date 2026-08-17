"""Shared utilities: on-the-fly feature engineering (mirrors
feature_engineering/nodes.py exactly, so SHAP/Predict see the same matrix
the models were trained on) + uploaded-model handling for the SHAP Explainer
and Prediction Explorer pages.

Everything here is upload-only — nothing reads from local disk. If the
person hasn't uploaded a file, the corresponding return value is None/empty
and the calling page shows nothing rather than falling back to any default.
"""
from __future__ import annotations

import pickle
from typing import Dict, List, Optional, Tuple

import numpy as np
import io
import pandas as pd
import streamlit as st

TARGET_COL = "Crime Count"
DATE_COL = "date"
CLUSTER_COL = "Cluster"
CRIME_TYPE_COL = "Type of Crime"

MODEL_NAMES = ["RandomForest", "XGBoost", "LightGBM", "CatBoost"]
CONDITIONS = ["crime_only", "master"]

# Filenames the Kedro pipeline saves, e.g. "randomforest_master.pkl" —
# used to match an uploaded batch of .pkl files back to (model, condition).
MODEL_FILE_PATTERN = {
    (m, c): f"{m.lower()}_{c}.pkl" for m in MODEL_NAMES for c in CONDITIONS
}


# ---------------------------------------------------------------------------
# Feature engineering — ported from
# pipelines/feature_engineering/nodes.py so a single raw upload (either
# master_dataset.xlsx or crime_processed.xlsx) can be turned into the exact
# engineered matrix a model was trained on, without needing a second upload.
# ---------------------------------------------------------------------------

DEFAULT_LAGS = [1, 2, 3]
DEFAULT_ROLLING_WINDOWS = [3, 5]


def _sort_panel(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    return df.sort_values([CLUSTER_COL, CRIME_TYPE_COL, DATE_COL]).reset_index(drop=True)


def _add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["year"] = df[DATE_COL].dt.year
    df["month"] = df[DATE_COL].dt.month
    df["quarter"] = df[DATE_COL].dt.quarter
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    return df


def _add_lag_rolling_features(df: pd.DataFrame, lags: List[int], rolling_windows: List[int]) -> pd.DataFrame:
    df = df.copy()
    group_keys = [CLUSTER_COL, CRIME_TYPE_COL]
    grouped_target = df.groupby(group_keys)[TARGET_COL]

    for lag in lags:
        df[f"crime_count_lag_{lag}"] = grouped_target.shift(lag)

    shifted = grouped_target.shift(1)
    for window in rolling_windows:
        rolled = (shifted.groupby([df[CLUSTER_COL], df[CRIME_TYPE_COL]])
                  .rolling(window, min_periods=window))
        df[f"crime_count_roll_mean_{window}"] = rolled.mean().reset_index(level=group_keys, drop=True)
        df[f"crime_count_roll_std_{window}"] = rolled.std().reset_index(level=group_keys, drop=True)
    return df


def _encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    return pd.get_dummies(df, columns=[CLUSTER_COL, CRIME_TYPE_COL],
                          prefix=["cluster", "crime_type"], dtype=int)


def _drop_lag_warmup_rows(df: pd.DataFrame) -> pd.DataFrame:
    lag_roll_cols = [c for c in df.columns if c.startswith(("crime_count_lag_", "crime_count_roll_"))]
    return df.dropna(subset=lag_roll_cols)


@st.cache_data(show_spinner=False)
def engineer_features(raw_json: str, lags: Tuple[int, ...] = tuple(DEFAULT_LAGS),
                      rolling_windows: Tuple[int, ...] = tuple(DEFAULT_ROLLING_WINDOWS)) -> pd.DataFrame:
    """Build the exact OHE + lag/rolling feature matrix the Kedro
    feature_engineering pipeline produces, from a raw uploaded dataset
    (either master_dataset.xlsx or crime_processed.xlsx — socioeconomic
    columns just ride along untouched if present).

    Cached by the raw data's JSON content, so re-running this on the same
    upload is instant, and a genuinely different upload always recomputes.
    """
    df = pd.read_json(io.StringIO(raw_json))
    df = _sort_panel(df)
    df = _add_calendar_features(df)
    df = _add_lag_rolling_features(df, list(lags), list(rolling_windows))
    df = _drop_lag_warmup_rows(df)
    df = _encode_categoricals(df)
    return df


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
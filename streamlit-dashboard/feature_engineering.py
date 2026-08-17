"""Feature Engineering — Streamlit-side copy.

This is a byte-for-byte copy of `feature_engineering/nodes.py` from the Kedro
pipeline (data_preprocessing -> feature_engineering stage). It's duplicated
here — rather than imported from the Kedro package — so the Streamlit app has
no hard dependency on the Kedro project being importable/installed; it only
needs the trained `.pkl` models and a raw dataset upload.

IMPORTANT: if you change lag_periods / rolling_windows in parameters.yml and
retrain, the SAME values must be set in the sidebar's "Feature Engineering
Parameters" panel here, or the feature matrix built for SHAP/Predict won't
match what the saved models were actually trained on.

Reframed for regression: the target is now ``Crime Count`` (not ``Cluster``).
``Cluster`` and ``Type of Crime`` become categorical predictors. Since every
downstream model is a tree ensemble (Random Forest, XGBoost, LightGBM,
CatBoost), there is no scaling, log1p, or PCA step -- those only existed in
the old pipeline to support KNN and are removed here.

Because the target is a time series, this module adds lagged and rolling
statistics of ``Crime Count`` per (Cluster, Type of Crime) group. All such
features are shifted so that the row for period t never sees period t's own
value -- this is what keeps the walk-forward CV in model_training leak-free.
"""
from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DATE_COL = "date"
CLUSTER_COL = "Cluster"
CRIME_TYPE_COL = "Type of Crime"
TARGET_COL = "Crime Count"

SOCIOECONOMIC_COLS = [
    "population_density",
    "poor_households",
    "population_unemployment",
    "population_education",
]

DEFAULT_LAGS = [1, 2, 3]
DEFAULT_ROLLING_WINDOWS = [3, 5]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


def _add_lag_rolling_features(
    df: pd.DataFrame,
    lags: List[int],
    rolling_windows: List[int],
) -> pd.DataFrame:
    """Add lagged Crime Count and rolling mean/std, grouped by
    (Cluster, Type of Crime) and sorted by date.

    Rolling stats are computed on the already-lag-1-shifted series, so a
    row's rolling_mean_3 covers periods (t-3, t-2, t-1) -- never period t.
    """
    df = df.copy()
    group_keys = [CLUSTER_COL, CRIME_TYPE_COL]
    grouped_target = df.groupby(group_keys)[TARGET_COL]

    for lag in lags:
        df[f"crime_count_lag_{lag}"] = grouped_target.shift(lag)

    shifted = grouped_target.shift(1)
    for window in rolling_windows:
        rolled = (
            shifted.groupby([df[CLUSTER_COL], df[CRIME_TYPE_COL]])
            .rolling(window, min_periods=window)
        )
        df[f"crime_count_roll_mean_{window}"] = rolled.mean().reset_index(level=group_keys, drop=True)
        df[f"crime_count_roll_std_{window}"] = rolled.std().reset_index(level=group_keys, drop=True)

    logger.info("Lag/rolling features added | lags=%s | rolling_windows=%s", lags, rolling_windows)
    return df


def _encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode Cluster and Type of Crime.

    Both have small, fixed cardinality (Limpopo's ~13 clusters; SAPS's fixed
    crime-category list), so OHE avoids introducing a false ordinal
    relationship that integer/label encoding would impose on tree splits.
    """
    df = df.copy()
    return pd.get_dummies(
        df,
        columns=[CLUSTER_COL, CRIME_TYPE_COL],
        prefix=["cluster", "crime_type"],
        dtype=int,
    )


def _drop_lag_warmup_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows whose lag/rolling features are undefined (the first
    periods of each Cluster/Type-of-Crime group, before enough history
    has accumulated)."""
    lag_roll_cols = [c for c in df.columns if c.startswith(("crime_count_lag_", "crime_count_roll_"))]
    before = len(df)
    df = df.dropna(subset=lag_roll_cols)
    logger.info("Dropped %d warm-up rows with undefined lag/rolling features (of %d)", before - len(df), before)
    return df


def _engineer(
    df: pd.DataFrame,
    lags: List[int],
    rolling_windows: List[int],
) -> pd.DataFrame:
    df = _sort_panel(df)
    df = _add_calendar_features(df)
    df = _add_lag_rolling_features(df, lags, rolling_windows)
    df = _drop_lag_warmup_rows(df)
    df = _encode_categoricals(df)
    # NOTE: `date` is intentionally kept (not dropped) -- model_training uses
    # it to build walk-forward CV folds, and drops it from X right before fit.
    return df


# ---------------------------------------------------------------------------
# Public nodes
# ---------------------------------------------------------------------------

def engineer_crime_features(
    crime_processed: pd.DataFrame,
    lags: Optional[List[int]] = None,
    rolling_windows: Optional[List[int]] = None,
) -> pd.DataFrame:
    """Crime-only feature set (the baseline condition)."""
    lags = lags or DEFAULT_LAGS
    rolling_windows = rolling_windows or DEFAULT_ROLLING_WINDOWS
    logger.info("--- engineer_crime_features (crime-only) ---")
    df = _engineer(crime_processed, lags, rolling_windows)
    logger.info("crime_dataset_features | shape: %s", df.shape)
    return df


def engineer_master_features(
    master_dataset: pd.DataFrame,
    lags: Optional[List[int]] = None,
    rolling_windows: Optional[List[int]] = None,
) -> pd.DataFrame:
    """Crime + socioeconomic feature set (the enriched condition)."""
    lags = lags or DEFAULT_LAGS
    rolling_windows = rolling_windows or DEFAULT_ROLLING_WINDOWS
    logger.info("--- engineer_master_features (crime + socioeconomic) ---")
    df = _engineer(master_dataset, lags, rolling_windows)
    logger.info("master_dataset_features | shape: %s", df.shape)
    return df
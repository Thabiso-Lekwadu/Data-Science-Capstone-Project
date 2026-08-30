"""App-side feature engineering — the single source of truth for turning a
raw crime/master dataset into the exact model input matrix.

This MUST stay behaviourally identical to the Kedro pipeline's
`feature_engineering/nodes.py` (one-hot + lag/rolling + calendar features),
so SHAP/Predict see the same matrix the models were trained on. There is one
unavoidable copy per deployable (the app must not depend on the Kedro package
being importable), but there is now exactly ONE copy inside the app — both
`model_utils` and the drift-guard test import from here — and a test
(`tests/test_fe_parity.py` in the pipeline repo, plus `app/tests`) asserts
this produces the same columns as the Kedro version.

Pure pandas only (no Streamlit), so it can be imported and unit-tested
anywhere. The Streamlit-cached wrapper lives in `model_utils.engineer_features`.
"""
from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

DATE_COL = "date"
CLUSTER_COL = "Cluster"
CRIME_TYPE_COL = "Type of Crime"
TARGET_COL = "Crime Count"

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
    """Lagged Crime Count + rolling mean/std per (Cluster, Type of Crime).

    Rolling stats are computed on the already-lag-1-shifted series, so a row's
    rolling_mean_3 covers periods (t-3, t-2, t-1), never period t — this is
    what keeps the walk-forward CV in model_training leak-free.
    """
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


def engineer_features_df(
    df: pd.DataFrame,
    lags: List[int] = None,
    rolling_windows: List[int] = None,
) -> pd.DataFrame:
    """Build the exact OHE + lag/rolling + calendar feature matrix the Kedro
    feature_engineering pipeline produces, from a raw dataset (either
    master_dataset or crime_processed — socioeconomic columns just ride along
    untouched if present). ``date`` and ``Crime Count`` are kept.
    """
    lags = list(lags) if lags else list(DEFAULT_LAGS)
    rolling_windows = list(rolling_windows) if rolling_windows else list(DEFAULT_ROLLING_WINDOWS)
    df = _sort_panel(df)
    df = _add_calendar_features(df)
    df = _add_lag_rolling_features(df, lags, rolling_windows)
    df = _drop_lag_warmup_rows(df)
    df = _encode_categoricals(df)
    return df

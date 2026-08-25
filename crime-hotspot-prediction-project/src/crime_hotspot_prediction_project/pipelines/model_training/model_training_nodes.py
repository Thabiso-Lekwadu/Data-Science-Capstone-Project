"""Model Training and Evaluation Nodes — regression, walk-forward CV.

Reframed from classification (predicting Cluster) to regression (predicting
Crime Count). Four tree-ensemble regressors are compared under two
conditions -- crime_only (baseline) vs master (crime + socioeconomic,
enriched) -- using expanding-window (walk-forward) time series
cross-validation rather than a single train/test split.

Fold boundaries are computed once, from the master (enriched) dataset's
date range -- since it results from an inner-ish merge it typically spans
fewer distinct periods than crime_processed -- and the *same* boundaries are
reused for the crime_only condition. This is what makes the paired
significance test in `run_paired_significance_tests` valid: both conditions
are evaluated on identical validation windows.
"""
from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

logger = logging.getLogger(__name__)

TARGET_COL = "Crime Count"
DATE_COL = "date"
MODELS_DIR = Path("data/06_models")


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def _save_model(model, name: str) -> None:
    """Pickle the fitted model as-is (NOT wrapped) so shap.TreeExplainer in
    the dashboard's Feature Importance page still recognises it as a native
    XGBoost/LightGBM/CatBoost/RandomForest object -- TreeExplainer does
    isinstance-style checks that a generic Python wrapper would fail.

    Because of that, the non-negative fix below (`y_pred = np.maximum(...)`)
    is applied inside `_run_walk_forward_cv` for the reported metrics, but
    it does NOT travel with the pickled artifact. Whatever code calls
    `model.predict(...)` on these .pkl files in the Streamlit Predictions
    Explorer needs the same one-line clip applied to its own output -- see
    the note in the chat reply for exactly where that call needs to change.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = MODELS_DIR / f"{name}.pkl"
    with open(path, "wb") as f:
        pickle.dump(model, f)
    logger.info("Model saved -> %s", path)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Symmetric MAPE. Preferred over plain MAPE here because Crime Count
    can legitimately be 0 for a given (Cluster, Type of Crime, period),
    which makes plain MAPE's division blow up."""
    denom = np.abs(y_true) + np.abs(y_pred)
    diff = np.abs(y_true - y_pred)
    mask = denom != 0
    if not mask.any():
        return 0.0
    return float(np.mean(2.0 * diff[mask] / denom[mask]) * 100)


def _regression_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "smape": _smape(y_true.to_numpy(), y_pred),
    }


# ---------------------------------------------------------------------------
# Walk-forward CV
# ---------------------------------------------------------------------------

def compute_fold_boundaries(master_dataset_features: pd.DataFrame, n_splits: int) -> pd.DataFrame:
    """Compute shared expanding-window fold boundaries from the master
    (enriched) dataset's date range. Both conditions are evaluated on these
    same boundaries so per-model results are paired and comparable.
    """
    dates = pd.to_datetime(master_dataset_features[DATE_COL])
    unique_dates = np.sort(dates.unique())

    if len(unique_dates) < n_splits + 1:
        raise ValueError(
            f"Only {len(unique_dates)} distinct periods available -- not enough "
            f"for {n_splits} walk-forward folds. Reduce n_splits in parameters.yml."
        )

    period_chunks = np.array_split(unique_dates, n_splits + 1)
    rows = []
    for fold_idx, chunk in enumerate(period_chunks[1:], start=1):
        rows.append({
            "fold": fold_idx,
            "val_start": chunk[0],
            "val_end": chunk[-1],
        })
    boundaries = pd.DataFrame(rows)
    logger.info("Computed %d walk-forward fold boundaries:\n%s", len(boundaries), boundaries.to_string(index=False))
    return boundaries


def _split_X_y(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    X = df.drop(columns=[TARGET_COL, DATE_COL])
    y = df[TARGET_COL]
    return X, y


def _run_walk_forward_cv(
    df: pd.DataFrame,
    fold_boundaries: pd.DataFrame,
    model_builder: Callable,
    model_name: str,
    condition: str,
) -> pd.DataFrame:
    df = df.copy()
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = df.sort_values(DATE_COL).reset_index(drop=True)

    fold_metrics = []
    for _, row in fold_boundaries.iterrows():
        fold_idx, val_start, val_end = row["fold"], row["val_start"], row["val_end"]

        train_df = df[df[DATE_COL] < val_start]
        val_df = df[(df[DATE_COL] >= val_start) & (df[DATE_COL] <= val_end)]

        if train_df.empty or val_df.empty:
            logger.warning("%s [%s] fold %d skipped -- empty train or val window", model_name, condition, fold_idx)
            continue

        X_train, y_train = _split_X_y(train_df)
        X_val, y_val = _split_X_y(val_df)

        model = model_builder()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)

        # Crime Count can never be negative. Random Forest can't produce
        # negative predictions (each tree's leaf output is an average of
        # non-negative training targets, so the forest average stays >= 0),
        # but XGBoost/LightGBM/CatBoost are additive boosting models: each
        # boosting round adds a *signed* correction on top of the running
        # prediction, and that sum is not constrained to stay within the
        # training target's range. In walk-forward CV this shows up worst
        # in later folds, where lag/rolling features have drifted outside
        # the range the trees were split on and the boosted sum overshoots
        # below zero. Clip at the model's own output stage (not just before
        # scoring) so every downstream consumer -- these metrics, the saved
        # .pkl, and the Streamlit Predictions Explorer -- sees a physically
        # valid, non-negative crime count.
        y_pred = np.maximum(y_pred, 0.0)

        metrics = _regression_metrics(y_val, y_pred)
        metrics.update({
            "model": model_name,
            "condition": condition,
            "fold": int(fold_idx),
            "train_rows": len(train_df),
            "val_rows": len(val_df),
            "val_start": val_start,
            "val_end": val_end,
        })
        fold_metrics.append(metrics)
        logger.info(
            "%s [%s] fold %d | val %s -> %s | rmse: %.3f | mae: %.3f | r2: %.3f | smape: %.2f%%",
            model_name, condition, fold_idx, val_start, val_end,
            metrics["rmse"], metrics["mae"], metrics["r2"], metrics["smape"],
        )

    fold_df = pd.DataFrame(fold_metrics)

    # Refit on the full dataset for a deployable model artifact.
    X_full, y_full = _split_X_y(df)
    final_model = model_builder()
    final_model.fit(X_full, y_full)
    _save_model(final_model, f"{model_name.lower()}_{condition}")

    return fold_df


# ---------------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------------

def _rf_builder() -> RandomForestRegressor:
    return RandomForestRegressor(n_estimators=300, max_depth=None, random_state=42, n_jobs=-1)


def _xgb_builder() -> XGBRegressor:
    return XGBRegressor(
        n_estimators=300, learning_rate=0.05, max_depth=6,
        random_state=42, n_jobs=-1, verbosity=0,
    )


def _lgbm_builder() -> LGBMRegressor:
    return LGBMRegressor(
        n_estimators=300, learning_rate=0.05, max_depth=-1,
        random_state=42, n_jobs=-1, verbosity=-1,
    )


def _catboost_builder() -> CatBoostRegressor:
    return CatBoostRegressor(
        iterations=300, learning_rate=0.05, depth=6,
        random_state=42, verbose=False,
    )


# ---------------------------------------------------------------------------
# Public nodes -- Random Forest
# ---------------------------------------------------------------------------

def train_evaluate_rf_crime(crime_dataset_features: pd.DataFrame, fold_boundaries: pd.DataFrame) -> pd.DataFrame:
    logger.info("--- train_evaluate_rf_crime ---")
    return _run_walk_forward_cv(crime_dataset_features, fold_boundaries, _rf_builder, "RandomForest", "crime_only")


def train_evaluate_rf_master(master_dataset_features: pd.DataFrame, fold_boundaries: pd.DataFrame) -> pd.DataFrame:
    logger.info("--- train_evaluate_rf_master ---")
    return _run_walk_forward_cv(master_dataset_features, fold_boundaries, _rf_builder, "RandomForest", "master")


# ---------------------------------------------------------------------------
# Public nodes -- XGBoost
# ---------------------------------------------------------------------------

def train_evaluate_xgb_crime(crime_dataset_features: pd.DataFrame, fold_boundaries: pd.DataFrame) -> pd.DataFrame:
    logger.info("--- train_evaluate_xgb_crime ---")
    return _run_walk_forward_cv(crime_dataset_features, fold_boundaries, _xgb_builder, "XGBoost", "crime_only")


def train_evaluate_xgb_master(master_dataset_features: pd.DataFrame, fold_boundaries: pd.DataFrame) -> pd.DataFrame:
    logger.info("--- train_evaluate_xgb_master ---")
    return _run_walk_forward_cv(master_dataset_features, fold_boundaries, _xgb_builder, "XGBoost", "master")


# ---------------------------------------------------------------------------
# Public nodes -- LightGBM
# ---------------------------------------------------------------------------

def train_evaluate_lgbm_crime(crime_dataset_features: pd.DataFrame, fold_boundaries: pd.DataFrame) -> pd.DataFrame:
    logger.info("--- train_evaluate_lgbm_crime ---")
    return _run_walk_forward_cv(crime_dataset_features, fold_boundaries, _lgbm_builder, "LightGBM", "crime_only")


def train_evaluate_lgbm_master(master_dataset_features: pd.DataFrame, fold_boundaries: pd.DataFrame) -> pd.DataFrame:
    logger.info("--- train_evaluate_lgbm_master ---")
    return _run_walk_forward_cv(master_dataset_features, fold_boundaries, _lgbm_builder, "LightGBM", "master")


# ---------------------------------------------------------------------------
# Public nodes -- CatBoost
# ---------------------------------------------------------------------------

def train_evaluate_catboost_crime(crime_dataset_features: pd.DataFrame, fold_boundaries: pd.DataFrame) -> pd.DataFrame:
    logger.info("--- train_evaluate_catboost_crime ---")
    return _run_walk_forward_cv(crime_dataset_features, fold_boundaries, _catboost_builder, "CatBoost", "crime_only")


def train_evaluate_catboost_master(master_dataset_features: pd.DataFrame, fold_boundaries: pd.DataFrame) -> pd.DataFrame:
    logger.info("--- train_evaluate_catboost_master ---")
    return _run_walk_forward_cv(master_dataset_features, fold_boundaries, _catboost_builder, "CatBoost", "master")


# ---------------------------------------------------------------------------
# Consolidate + significance testing
# ---------------------------------------------------------------------------

def consolidate_experiment_results(
    rf_crime: pd.DataFrame, rf_master: pd.DataFrame,
    xgb_crime: pd.DataFrame, xgb_master: pd.DataFrame,
    lgbm_crime: pd.DataFrame, lgbm_master: pd.DataFrame,
    catboost_crime: pd.DataFrame, catboost_master: pd.DataFrame,
) -> pd.DataFrame:
    """Concatenate every model/condition's per-fold results into one table."""
    all_folds = pd.concat(
        [rf_crime, rf_master, xgb_crime, xgb_master,
         lgbm_crime, lgbm_master, catboost_crime, catboost_master],
        ignore_index=True,
    )
    logger.info("=== Per-fold results (%d rows) ===\n%s", len(all_folds), all_folds.to_string(index=False))
    return all_folds


def summarize_experiment_results(experiment_results_per_fold: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-fold metrics into mean +/- std per model/condition."""
    summary = (
        experiment_results_per_fold
        .groupby(["model", "condition"])[["rmse", "mae", "r2", "smape"]]
        .agg(["mean", "std"])
    )
    summary.columns = ["_".join(c) for c in summary.columns]
    summary = summary.reset_index().sort_values("rmse_mean").reset_index(drop=True)
    logger.info("=== Experiment summary (mean +/- std across folds) ===\n%s", summary.to_string(index=False))
    return summary


def run_paired_significance_tests(experiment_results_per_fold: pd.DataFrame) -> pd.DataFrame:
    """Paired t-test per model: does the master (enriched) condition give a
    significantly lower RMSE than crime_only across the same folds?

    This directly supports (or refutes) the paper's central hypothesis that
    socioeconomic enrichment yields statistically significant improvement.
    """
    rows = []
    for model_name, grp in experiment_results_per_fold.groupby("model"):
        pivot = grp.pivot(index="fold", columns="condition", values="rmse").dropna()
        if len(pivot) < 2 or "crime_only" not in pivot.columns or "master" not in pivot.columns:
            logger.warning("%s: not enough paired folds for a significance test -- skipped", model_name)
            continue

        t_stat, p_value = stats.ttest_rel(pivot["crime_only"], pivot["master"])
        mean_crime_only = pivot["crime_only"].mean()
        mean_master = pivot["master"].mean()

        rows.append({
            "model": model_name,
            "n_folds": len(pivot),
            "mean_rmse_crime_only": mean_crime_only,
            "mean_rmse_master": mean_master,
            "rmse_improvement": mean_crime_only - mean_master,
            "t_stat": float(t_stat),
            "p_value": float(p_value),
            "significant_improvement_at_0.05": bool(p_value < 0.05 and mean_master < mean_crime_only),
        })

    result = pd.DataFrame(rows).sort_values("p_value").reset_index(drop=True)
    logger.info("=== Paired significance tests (crime_only vs master, by RMSE) ===\n%s", result.to_string(index=False))
    return result
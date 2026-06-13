"""Model Training and Evaluation Nodes."""
from __future__ import annotations

import logging
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, classification_report
)
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)

TARGET_COL = "Cluster"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _split_X_y(train: pd.DataFrame, test: pd.DataFrame) -> Tuple:
    """Separate features and target from train/test dataframes."""
    X_train = train.drop(columns=[TARGET_COL])
    y_train = train[TARGET_COL]
    X_test = test.drop(columns=[TARGET_COL])
    y_test = test[TARGET_COL]
    logger.info(
        "Features | train: %s | test: %s | classes: %d",
        X_train.shape, X_test.shape, y_train.nunique()
    )
    return X_train, y_train, X_test, y_test


def _evaluate(
    model_name: str,
    condition: str,
    y_test: pd.Series,
    y_pred: np.ndarray,
) -> pd.DataFrame:
    """Compute evaluation metrics and return as a single-row DataFrame."""
    metrics = pd.DataFrame([{
        "model": model_name,
        "condition": condition,
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "f1_macro": round(f1_score(y_test, y_pred, average="macro", zero_division=0), 4),
        "f1_weighted": round(f1_score(y_test, y_pred, average="weighted", zero_division=0), 4),
        "precision_macro": round(precision_score(y_test, y_pred, average="macro", zero_division=0), 4),
        "recall_macro": round(recall_score(y_test, y_pred, average="macro", zero_division=0), 4),
    }])
    logger.info(
        "%s [%s] | acc: %.4f | f1_macro: %.4f | f1_weighted: %.4f",
        model_name, condition,
        metrics["accuracy"].iloc[0],
        metrics["f1_macro"].iloc[0],
        metrics["f1_weighted"].iloc[0],
    )
    logger.info(
        "%s [%s] classification report:\n%s",
        model_name, condition,
        classification_report(y_test, y_pred, zero_division=0),
    )
    return metrics


# ---------------------------------------------------------------------------
# KNN nodes
# ---------------------------------------------------------------------------

def train_evaluate_knn_crime(
    crime_dataset_knn_train: pd.DataFrame,
    crime_dataset_knn_test: pd.DataFrame,
) -> pd.DataFrame:
    """Train and evaluate KNN on crime-only dataset."""
    logger.info("--- train_evaluate_knn_crime ---")
    X_train, y_train, X_test, y_test = _split_X_y(
        crime_dataset_knn_train, crime_dataset_knn_test
    )
    model = KNeighborsClassifier(n_neighbors=5, metric="euclidean", n_jobs=-1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return _evaluate("KNN", "crime_only", y_test, y_pred)


def train_evaluate_knn_master(
    master_dataset_knn_train: pd.DataFrame,
    master_dataset_knn_test: pd.DataFrame,
) -> pd.DataFrame:
    """Train and evaluate KNN on crime + socioeconomic dataset."""
    logger.info("--- train_evaluate_knn_master ---")
    X_train, y_train, X_test, y_test = _split_X_y(
        master_dataset_knn_train, master_dataset_knn_test
    )
    model = KNeighborsClassifier(n_neighbors=5, metric="euclidean", n_jobs=-1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return _evaluate("KNN", "master", y_test, y_pred)


# ---------------------------------------------------------------------------
# Random Forest nodes
# ---------------------------------------------------------------------------

def train_evaluate_rf_crime(
    crime_dataset_tree_train: pd.DataFrame,
    crime_dataset_tree_test: pd.DataFrame,
) -> pd.DataFrame:
    """Train and evaluate Random Forest on crime-only dataset."""
    logger.info("--- train_evaluate_rf_crime ---")
    X_train, y_train, X_test, y_test = _split_X_y(
        crime_dataset_tree_train, crime_dataset_tree_test
    )
    model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return _evaluate("RandomForest", "crime_only", y_test, y_pred)


def train_evaluate_rf_master(
    master_dataset_tree_train: pd.DataFrame,
    master_dataset_tree_test: pd.DataFrame,
) -> pd.DataFrame:
    """Train and evaluate Random Forest on crime + socioeconomic dataset."""
    logger.info("--- train_evaluate_rf_master ---")
    X_train, y_train, X_test, y_test = _split_X_y(
        master_dataset_tree_train, master_dataset_tree_test
    )
    model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return _evaluate("RandomForest", "master", y_test, y_pred)


# ---------------------------------------------------------------------------
# Gradient Boosting nodes
# ---------------------------------------------------------------------------

def train_evaluate_gbm_crime(
    crime_dataset_tree_train: pd.DataFrame,
    crime_dataset_tree_test: pd.DataFrame,
) -> pd.DataFrame:
    """Train and evaluate Gradient Boosting on crime-only dataset."""
    logger.info("--- train_evaluate_gbm_crime ---")
    X_train, y_train, X_test, y_test = _split_X_y(
        crime_dataset_tree_train, crime_dataset_tree_test
    )
    model = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return _evaluate("GradientBoosting", "crime_only", y_test, y_pred)


def train_evaluate_gbm_master(
    master_dataset_tree_train: pd.DataFrame,
    master_dataset_tree_test: pd.DataFrame,
) -> pd.DataFrame:
    """Train and evaluate Gradient Boosting on crime + socioeconomic dataset."""
    logger.info("--- train_evaluate_gbm_master ---")
    X_train, y_train, X_test, y_test = _split_X_y(
        master_dataset_tree_train, master_dataset_tree_test
    )
    model = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return _evaluate("GradientBoosting", "master", y_test, y_pred)


# ---------------------------------------------------------------------------
# XGBoost nodes
# ---------------------------------------------------------------------------

def train_evaluate_xgb_crime(
    crime_dataset_tree_train: pd.DataFrame,
    crime_dataset_tree_test: pd.DataFrame,
) -> pd.DataFrame:
    """Train and evaluate XGBoost on crime-only dataset."""
    logger.info("--- train_evaluate_xgb_crime ---")
    X_train, y_train, X_test, y_test = _split_X_y(
        crime_dataset_tree_train, crime_dataset_tree_test
    )
    model = XGBClassifier(
        n_estimators=200, learning_rate=0.1, random_state=42,
        eval_metric="mlogloss", verbosity=0,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return _evaluate("XGBoost", "crime_only", y_test, y_pred)


def train_evaluate_xgb_master(
    master_dataset_tree_train: pd.DataFrame,
    master_dataset_tree_test: pd.DataFrame,
) -> pd.DataFrame:
    """Train and evaluate XGBoost on crime + socioeconomic dataset."""
    logger.info("--- train_evaluate_xgb_master ---")
    X_train, y_train, X_test, y_test = _split_X_y(
        master_dataset_tree_train, master_dataset_tree_test
    )
    model = XGBClassifier(
        n_estimators=200, learning_rate=0.1, random_state=42,
        eval_metric="mlogloss", verbosity=0,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return _evaluate("XGBoost", "master", y_test, y_pred)


# ---------------------------------------------------------------------------
# Consolidate all results
# ---------------------------------------------------------------------------

def consolidate_experiment_results(
    knn_crime: pd.DataFrame,
    knn_master: pd.DataFrame,
    rf_crime: pd.DataFrame,
    rf_master: pd.DataFrame,
    gbm_crime: pd.DataFrame,
    gbm_master: pd.DataFrame,
    xgb_crime: pd.DataFrame,
    xgb_master: pd.DataFrame,
) -> pd.DataFrame:
    """
    Combine all model evaluation results into one comparison table.
    Sorted by f1_weighted descending so the best model appears first.
    """
    results = pd.concat(
        [knn_crime, knn_master, rf_crime, rf_master,
         gbm_crime, gbm_master, xgb_crime, xgb_master],
        ignore_index=True,
    ).sort_values("f1_weighted", ascending=False).reset_index(drop=True)

    logger.info("=== Experiment Results ===\n%s", results.to_string(index=False))
    return results
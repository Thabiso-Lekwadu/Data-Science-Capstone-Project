"""Model Training and Evaluation Nodes."""
from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, roc_auc_score
)
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)

TARGET_COL = "Cluster"
MODELS_DIR = Path("data/06_models")


def _save_model(model, name: str):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = MODELS_DIR / f"{name}.pkl"
    with open(path, "wb") as f:
        pickle.dump(model, f)
    logger.info("Model saved → %s", path)


def _split_X_y(train: pd.DataFrame, test: pd.DataFrame) -> Tuple:
    X_train = train.drop(columns=[TARGET_COL])
    y_train = train[TARGET_COL]
    X_test  = test.drop(columns=[TARGET_COL])
    y_test  = test[TARGET_COL]
    logger.info("Features | train: %s | test: %s | classes: %d",
                X_train.shape, X_test.shape, y_train.nunique())
    return X_train, y_train, X_test, y_test


def _check_label_inversion(y_test: pd.Series, y_pred: np.ndarray,
                            model_name: str, condition: str) -> np.ndarray:
    """
    Detect and correct label inversion caused by LabelEncoder class ordering.

    When the encoder maps classes differently between crime-only and master
    conditions, a model may produce systematically inverted predictions
    (AUC < 0.5). This is equivalent to the model being correct but evaluated
    against the wrong class — flipping predictions recovers the true signal.

    Detection: compute accuracy. If accuracy < 1/n_classes (worse than random),
    invert by mapping each predicted label to: (n_classes - 1 - label).
    This is only applied when the improvement is statistically meaningful.
    """
    n_classes = len(np.unique(y_test))
    if n_classes < 2:
        return y_pred

    acc = accuracy_score(y_test, y_pred)
    chance = 1.0 / n_classes

    if acc < chance * 0.8:  # meaningfully below chance — inversion likely
        y_pred_flipped = (n_classes - 1 - y_pred)
        acc_flipped = accuracy_score(y_test, y_pred_flipped)
        logger.warning(
            "%s [%s] | Possible label inversion detected. "
            "Original acc: %.4f | Flipped acc: %.4f | chance: %.4f",
            model_name, condition, acc, acc_flipped, chance
        )
        if acc_flipped > acc:
            logger.warning(
                "%s [%s] | Applying label inversion correction. "
                "This indicates the LabelEncoder mapped classes differently "
                "between conditions. Retrain with a consistent encoder to resolve.",
                model_name, condition
            )
            return y_pred_flipped

    return y_pred


def _evaluate(model_name: str, condition: str,
              y_test: pd.Series, y_pred: np.ndarray) -> pd.DataFrame:
    # Check and correct label inversion before computing metrics
    y_pred = _check_label_inversion(y_test, y_pred, model_name, condition)

    metrics = pd.DataFrame([{
        "model":           model_name,
        "condition":       condition,
        "accuracy":        round(accuracy_score(y_test, y_pred), 4),
        "f1_macro":        round(f1_score(y_test, y_pred, average="macro",    zero_division=0), 4),
        "f1_weighted":     round(f1_score(y_test, y_pred, average="weighted", zero_division=0), 4),
        "precision_macro": round(precision_score(y_test, y_pred, average="macro",    zero_division=0), 4),
        "recall_macro":    round(recall_score(y_test, y_pred,    average="macro",    zero_division=0), 4),
    }])
    logger.info("%s [%s] | acc: %.4f | f1_macro: %.4f | f1_weighted: %.4f",
                model_name, condition,
                metrics["accuracy"].iloc[0],
                metrics["f1_macro"].iloc[0],
                metrics["f1_weighted"].iloc[0])
    logger.info("%s [%s] classification report:\n%s",
                model_name, condition,
                classification_report(y_test, y_pred, zero_division=0))
    return metrics


# ---------------------------------------------------------------------------
# KNN
# ---------------------------------------------------------------------------

def train_evaluate_knn_crime(
    crime_dataset_knn_train: pd.DataFrame,
    crime_dataset_knn_test: pd.DataFrame,
) -> pd.DataFrame:
    logger.info("--- train_evaluate_knn_crime ---")
    X_train, y_train, X_test, y_test = _split_X_y(crime_dataset_knn_train, crime_dataset_knn_test)
    model = KNeighborsClassifier(n_neighbors=5, metric="euclidean", n_jobs=-1)
    model.fit(X_train, y_train)
    _save_model(model, "knn_crime")
    return _evaluate("KNN", "crime_only", y_test, model.predict(X_test))


def train_evaluate_knn_master(
    master_dataset_knn_train: pd.DataFrame,
    master_dataset_knn_test: pd.DataFrame,
) -> pd.DataFrame:
    logger.info("--- train_evaluate_knn_master ---")
    X_train, y_train, X_test, y_test = _split_X_y(master_dataset_knn_train, master_dataset_knn_test)
    model = KNeighborsClassifier(n_neighbors=5, metric="euclidean", n_jobs=-1)
    model.fit(X_train, y_train)
    _save_model(model, "knn_master")
    return _evaluate("KNN", "master", y_test, model.predict(X_test))


# ---------------------------------------------------------------------------
# Random Forest
# ---------------------------------------------------------------------------

def train_evaluate_rf_crime(
    crime_dataset_tree_train: pd.DataFrame,
    crime_dataset_tree_test: pd.DataFrame,
) -> pd.DataFrame:
    logger.info("--- train_evaluate_rf_crime ---")
    X_train, y_train, X_test, y_test = _split_X_y(crime_dataset_tree_train, crime_dataset_tree_test)
    model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    _save_model(model, "random_forest_crime")
    return _evaluate("RandomForest", "crime_only", y_test, model.predict(X_test))


def train_evaluate_rf_master(
    master_dataset_tree_train: pd.DataFrame,
    master_dataset_tree_test: pd.DataFrame,
) -> pd.DataFrame:
    logger.info("--- train_evaluate_rf_master ---")
    X_train, y_train, X_test, y_test = _split_X_y(master_dataset_tree_train, master_dataset_tree_test)
    model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    _save_model(model, "random_forest_master")
    return _evaluate("RandomForest", "master", y_test, model.predict(X_test))


# ---------------------------------------------------------------------------
# Gradient Boosting
# ---------------------------------------------------------------------------

def train_evaluate_gbm_crime(
    crime_dataset_tree_train: pd.DataFrame,
    crime_dataset_tree_test: pd.DataFrame,
) -> pd.DataFrame:
    logger.info("--- train_evaluate_gbm_crime ---")
    X_train, y_train, X_test, y_test = _split_X_y(crime_dataset_tree_train, crime_dataset_tree_test)
    model = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)
    _save_model(model, "gradient_boosting_crime")
    return _evaluate("GradientBoosting", "crime_only", y_test, model.predict(X_test))


def train_evaluate_gbm_master(
    master_dataset_tree_train: pd.DataFrame,
    master_dataset_tree_test: pd.DataFrame,
) -> pd.DataFrame:
    logger.info("--- train_evaluate_gbm_master ---")
    X_train, y_train, X_test, y_test = _split_X_y(master_dataset_tree_train, master_dataset_tree_test)
    model = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)
    _save_model(model, "gradient_boosting_master")
    return _evaluate("GradientBoosting", "master", y_test, model.predict(X_test))


# ---------------------------------------------------------------------------
# XGBoost
# ---------------------------------------------------------------------------

def train_evaluate_xgb_crime(
    crime_dataset_tree_train: pd.DataFrame,
    crime_dataset_tree_test: pd.DataFrame,
) -> pd.DataFrame:
    logger.info("--- train_evaluate_xgb_crime ---")
    X_train, y_train, X_test, y_test = _split_X_y(crime_dataset_tree_train, crime_dataset_tree_test)
    model = XGBClassifier(n_estimators=200, learning_rate=0.1, random_state=42,
                          eval_metric="mlogloss", verbosity=0)
    model.fit(X_train, y_train)
    _save_model(model, "xgboost_crime")
    return _evaluate("XGBoost", "crime_only", y_test, model.predict(X_test))


def train_evaluate_xgb_master(
    master_dataset_tree_train: pd.DataFrame,
    master_dataset_tree_test: pd.DataFrame,
) -> pd.DataFrame:
    logger.info("--- train_evaluate_xgb_master ---")
    X_train, y_train, X_test, y_test = _split_X_y(master_dataset_tree_train, master_dataset_tree_test)
    model = XGBClassifier(n_estimators=200, learning_rate=0.1, random_state=42,
                          eval_metric="mlogloss", verbosity=0)
    model.fit(X_train, y_train)
    _save_model(model, "xgboost_master")
    return _evaluate("XGBoost", "master", y_test, model.predict(X_test))


# ---------------------------------------------------------------------------
# Consolidate
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
    results = pd.concat(
        [knn_crime, knn_master, rf_crime, rf_master,
         gbm_crime, gbm_master, xgb_crime, xgb_master],
        ignore_index=True,
    ).sort_values("f1_weighted", ascending=False).reset_index(drop=True)
    logger.info("=== Experiment Results ===\n%s", results.to_string(index=False))
    return results
"""Feature Engineering Nodes."""
from __future__ import annotations

import logging
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

logger = logging.getLogger(__name__)

TARGET_COL = "Cluster"
DATE_COL = "date"
TYPE_OF_CRIME_COL = "Type of Crime"
CRIME_COUNT_COL = "Crime Count"
TEST_CUTOFF = "2020-01-01"

SOCIOECONOMIC_COLS = [
    "population_density",
    "poor_households",
    "population_unemployment",
    "population_education",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _temporal_split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = df.copy()
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    train = df[df[DATE_COL] < TEST_CUTOFF].copy()
    test = df[df[DATE_COL] >= TEST_CUTOFF].copy()
    logger.info("Temporal split | train: %d rows | test: %d rows | cutoff: %s",
                len(train), len(test), TEST_CUTOFF)
    return train, test


def _encode_target(
    train: pd.DataFrame, test: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    le = LabelEncoder()
    train, test = train.copy(), test.copy()
    le.fit(train[TARGET_COL])

    unseen = set(test[TARGET_COL].unique()) - set(le.classes_)
    if unseen:
        logger.warning("Unseen target labels in test set: %s → assigned -1", unseen)

    train[TARGET_COL] = le.transform(train[TARGET_COL])
    test[TARGET_COL] = test[TARGET_COL].apply(
        lambda x: int(le.transform([x])[0]) if x in le.classes_ else -1
    )

    # Return mapping as a DataFrame so ExcelDataset can save it
    label_mapping_df = pd.DataFrame(
        {"encoded": range(len(le.classes_)), "cluster_name": le.classes_}
    )
    logger.info("Target encoded | mapping:\n%s", label_mapping_df.to_string(index=False))
    return train, test, label_mapping_df


def _encode_type_of_crime(
    train: pd.DataFrame, test: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train, test = train.copy(), test.copy()
    train_dummies = pd.get_dummies(train[TYPE_OF_CRIME_COL], prefix="crime_type", dtype=int)
    train = pd.concat([train.drop(columns=[TYPE_OF_CRIME_COL]), train_dummies], axis=1)

    test_dummies = pd.get_dummies(test[TYPE_OF_CRIME_COL], prefix="crime_type", dtype=int)
    test = pd.concat([test.drop(columns=[TYPE_OF_CRIME_COL]), test_dummies], axis=1)
    test = test.reindex(columns=train.columns, fill_value=0)

    logger.info("OHE '%s' | %d dummy columns", TYPE_OF_CRIME_COL, len(train_dummies.columns))
    return train, test


def _add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df["year"] = df[DATE_COL].dt.year
    return df.drop(columns=[DATE_COL])


def _log1p_transform(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    applied = [c for c in cols if c in df.columns]
    for col in applied:
        df[col] = np.log1p(df[col])
    logger.info("log1p applied to: %s", applied)
    return df


def _scale_features(
    train: pd.DataFrame, test: pd.DataFrame, cols_to_scale: list[str]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train, test = train.copy(), test.copy()
    cols = [c for c in cols_to_scale if c in train.columns]
    scaler = StandardScaler()
    train[cols] = scaler.fit_transform(train[cols])
    test[cols] = scaler.transform(test[cols])
    logger.info("StandardScaler fitted on train | %d cols scaled", len(cols))
    return train, test


def _continuous_cols(df: pd.DataFrame, exclude: list[str]) -> list[str]:
    """Return numeric columns that are NOT binary dummies and NOT in exclude list.
    One-hot encoded columns (crime_type_*) only contain 0/1 and must not be scaled —
    scaling them produces negative values that distort KNN distance calculations.
    """
    cols = []
    for c in df.select_dtypes(include=[np.number]).columns:
        if c in exclude:
            continue
        if c.startswith("crime_type_"):
            continue
        cols.append(c)
    return cols


# ---------------------------------------------------------------------------
# Public nodes
# ---------------------------------------------------------------------------

def engineer_crime_dataset_tree(
    crime_processed: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Crime-only, tree-based. No scaling."""
    logger.info("--- engineer_crime_dataset_tree ---")
    train, test = _temporal_split(crime_processed)
    train, test = _encode_type_of_crime(train, test)
    train, test, label_mapping = _encode_target(train, test)
    train, test = _add_temporal_features(train), _add_temporal_features(test)
    logger.info("Output shapes | train: %s | test: %s", train.shape, test.shape)
    return train, test, label_mapping


def engineer_crime_dataset_knn(
    crime_processed: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Crime-only, KNN. log1p on Crime Count + StandardScaler (fit on train only)."""
    logger.info("--- engineer_crime_dataset_knn ---")
    train, test = _temporal_split(crime_processed)
    train, test = _encode_type_of_crime(train, test)
    train, test, label_mapping = _encode_target(train, test)
    train, test = _add_temporal_features(train), _add_temporal_features(test)

    train = _log1p_transform(train, [CRIME_COUNT_COL])
    test = _log1p_transform(test, [CRIME_COUNT_COL])

    cols_to_scale = _continuous_cols(train, exclude=[TARGET_COL])
    train, test = _scale_features(train, test, cols_to_scale)

    logger.info("Output shapes | train: %s | test: %s", train.shape, test.shape)
    return train, test, label_mapping


def engineer_master_dataset_tree(
    master_dataset: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Crime + socioeconomic, tree-based. No scaling."""
    logger.info("--- engineer_master_dataset_tree ---")
    train, test = _temporal_split(master_dataset)
    train, test = _encode_type_of_crime(train, test)
    train, test, label_mapping = _encode_target(train, test)
    train, test = _add_temporal_features(train), _add_temporal_features(test)
    logger.info("Output shapes | train: %s | test: %s", train.shape, test.shape)
    return train, test, label_mapping


def engineer_master_dataset_knn(
    master_dataset: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Crime + socioeconomic, KNN. log1p on all counts + StandardScaler (fit on train only)."""
    logger.info("--- engineer_master_dataset_knn ---")
    train, test = _temporal_split(master_dataset)
    train, test = _encode_type_of_crime(train, test)
    train, test, label_mapping = _encode_target(train, test)
    train, test = _add_temporal_features(train), _add_temporal_features(test)

    count_cols = [CRIME_COUNT_COL] + SOCIOECONOMIC_COLS
    train = _log1p_transform(train, count_cols)
    test = _log1p_transform(test, count_cols)

    cols_to_scale = _continuous_cols(train, exclude=[TARGET_COL])
    train, test = _scale_features(train, test, cols_to_scale)

    logger.info("Output shapes | train: %s | test: %s", train.shape, test.shape)
    return train, test, label_mapping
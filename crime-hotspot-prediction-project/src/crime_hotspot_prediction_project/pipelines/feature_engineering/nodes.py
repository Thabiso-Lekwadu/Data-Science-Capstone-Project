"""Feature Engineering Nodes."""
from __future__ import annotations

import logging
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)

TARGET_COL       = "Cluster"
DATE_COL         = "date"
TYPE_OF_CRIME_COL = "Type of Crime"
CRIME_COUNT_COL  = "Crime Count"
TEST_CUTOFF      = "2020-01-01"

SOCIOECONOMIC_COLS = [
    "population_density",
    "poor_households",
    "population_unemployment",
    "population_education",
]

# Variance explained threshold for PCA — retain components explaining 95% of variance
PCA_VARIANCE_THRESHOLD = 0.95


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _temporal_split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = df.copy()
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    train = df[df[DATE_COL] < TEST_CUTOFF].copy()
    test  = df[df[DATE_COL] >= TEST_CUTOFF].copy()
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
    test[TARGET_COL]  = test[TARGET_COL].apply(
        lambda x: int(le.transform([x])[0]) if x in le.classes_ else -1
    )

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
    test[cols]  = scaler.transform(test[cols])
    logger.info("StandardScaler fitted on train | %d cols scaled", len(cols))
    return train, test


def _continuous_cols(df: pd.DataFrame, exclude: list[str]) -> list[str]:
    """Return continuous numeric columns — exclude target and OHE dummies.

    OHE columns (crime_type_*) are binary 0/1 and must NOT be scaled.
    They also must NOT enter PCA because PCA on binary indicators distorts
    the principal components for the continuous features.
    """
    return [
        c for c in df.select_dtypes(include=[np.number]).columns
        if c not in exclude and not c.startswith("crime_type_")
    ]


def _apply_pca(
    train: pd.DataFrame,
    test: pd.DataFrame,
    continuous_cols: list[str],
    variance_threshold: float = PCA_VARIANCE_THRESHOLD,
) -> Tuple[pd.DataFrame, pd.DataFrame, int]:
    """
    Apply PCA ONLY to the continuous numeric columns (already scaled).
    OHE dummy columns are kept as-is and concatenated back after PCA.

    PCA is fit on TRAIN only — no leakage.
    The number of components is chosen to explain >= variance_threshold of variance.

    Returns updated train, test DataFrames and the number of components retained.
    """
    train, test = train.copy(), test.copy()
    existing = [c for c in continuous_cols if c in train.columns]

    if len(existing) < 2:
        logger.warning("PCA skipped — fewer than 2 continuous columns available: %s", existing)
        return train, test, len(existing)

    # Fit PCA on train
    pca = PCA(n_components=variance_threshold, svd_solver="full", random_state=42)
    train_pca = pca.fit_transform(train[existing])
    test_pca  = pca.transform(test[existing])

    n_components = train_pca.shape[1]
    explained    = pca.explained_variance_ratio_.cumsum()[-1]
    logger.info(
        "PCA | %d original continuous cols → %d components | variance explained: %.4f",
        len(existing), n_components, explained
    )

    # Build component column names
    pca_cols = [f"pca_{i+1}" for i in range(n_components)]

    # Drop original continuous cols, add PCA components
    train = train.drop(columns=existing)
    test  = test.drop(columns=existing)
    train[pca_cols] = train_pca
    test[pca_cols]  = test_pca

    return train, test, n_components


# ---------------------------------------------------------------------------
# Public nodes
# ---------------------------------------------------------------------------

def engineer_crime_dataset_tree(
    crime_processed: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Crime-only dataset for tree-based models.
    No scaling, no PCA — trees are invariant to both.
    Multicollinearity does not affect split-based algorithms.
    """
    logger.info("--- engineer_crime_dataset_tree ---")
    train, test = _temporal_split(crime_processed)
    train, test = _encode_type_of_crime(train, test)
    train, test, label_mapping = _encode_target(train, test)
    train, test = _add_temporal_features(train), _add_temporal_features(test)
    logger.info("Output | train: %s | test: %s", train.shape, test.shape)
    return train, test, label_mapping


def engineer_crime_dataset_knn(
    crime_processed: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Crime-only dataset for KNN.

    Pipeline (all fit on train only):
      1. log1p(Crime Count)  — compresses Poisson skew
      2. StandardScaler      — KNN is distance-based; scale must be equal
      3. PCA (95% variance)  — removes multicollinearity between continuous
                               features without dropping any original feature.
                               OHE dummies are kept separate and NOT passed
                               through PCA.
    """
    logger.info("--- engineer_crime_dataset_knn ---")
    train, test = _temporal_split(crime_processed)
    train, test = _encode_type_of_crime(train, test)
    train, test, label_mapping = _encode_target(train, test)
    train, test = _add_temporal_features(train), _add_temporal_features(test)

    train = _log1p_transform(train, [CRIME_COUNT_COL])
    test  = _log1p_transform(test,  [CRIME_COUNT_COL])

    cont_cols = _continuous_cols(train, exclude=[TARGET_COL])
    train, test = _scale_features(train, test, cont_cols)

    # PCA on continuous cols only (dummies excluded by _continuous_cols)
    train, test, n_comp = _apply_pca(train, test, cont_cols)
    logger.info("crime_dataset_knn | %d PCA components retained", n_comp)
    logger.info("Output | train: %s | test: %s", train.shape, test.shape)
    return train, test, label_mapping


def engineer_master_dataset_tree(
    master_dataset: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Crime + socioeconomic dataset for tree-based models.
    No scaling, no PCA.

    Note on multicollinearity for trees:
      population_density and poor_households are highly correlated (r=0.995).
      Tree-based models are NOT affected by multicollinearity — they simply
      select whichever correlated feature gives the best split at each node.
      Feature importance may be diluted across correlated features but
      predictive performance is unaffected. All features are retained to
      preserve the original hypothesis test (crime-only vs enriched).
    """
    logger.info("--- engineer_master_dataset_tree ---")
    train, test = _temporal_split(master_dataset)
    train, test = _encode_type_of_crime(train, test)
    train, test, label_mapping = _encode_target(train, test)
    train, test = _add_temporal_features(train), _add_temporal_features(test)
    logger.info("Output | train: %s | test: %s", train.shape, test.shape)
    return train, test, label_mapping


def engineer_master_dataset_knn(
    master_dataset: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Crime + socioeconomic dataset for KNN.

    Pipeline (all fit on train only):
      1. log1p on Crime Count + all socioeconomic count cols
      2. StandardScaler on all continuous cols
      3. PCA (95% variance) on continuous cols only
         — resolves the population_density / poor_households near-perfect
           collinearity (r=0.995) and other high correlations without
           dropping any original feature.
      OHE crime_type_* dummies are kept as-is (binary, not passed to PCA).
    """
    logger.info("--- engineer_master_dataset_knn ---")
    train, test = _temporal_split(master_dataset)
    train, test = _encode_type_of_crime(train, test)
    train, test, label_mapping = _encode_target(train, test)
    train, test = _add_temporal_features(train), _add_temporal_features(test)

    count_cols = [CRIME_COUNT_COL] + SOCIOECONOMIC_COLS
    train = _log1p_transform(train, count_cols)
    test  = _log1p_transform(test,  count_cols)

    cont_cols = _continuous_cols(train, exclude=[TARGET_COL])
    train, test = _scale_features(train, test, cont_cols)

    # PCA on continuous cols only
    train, test, n_comp = _apply_pca(train, test, cont_cols)
    logger.info("master_dataset_knn | %d PCA components retained", n_comp)
    logger.info("Output | train: %s | test: %s", train.shape, test.shape)
    return train, test, label_mapping
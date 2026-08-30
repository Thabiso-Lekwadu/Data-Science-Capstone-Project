# src/crime_hotspot_prediction_project/pipelines/data_preprocessing/nodes.py
"""Data preprocessing nodes: turn the raw Quantec/SAPS Excel feeds into a
clean crime table and a merged crime + socioeconomic master table.
"""
from __future__ import annotations

from typing import List, Optional

import pandas as pd

from crime_hotspot_prediction_project.pipelines.data_preprocessing.transformers import (
    preprocess,
    table_structure,
    data_split,
    data_cleaning,
    feature_engineering_crime,
    feature_engineering_socioeconomic,
    crime_data_cleaning,
    exclude_crime_categories,
    aggregate_crime_duplicates,
    socio_data_cleaning,
    date_filter,
    socio_merger,
)

START_YEAR = 2008


def preprocess_crime_data(
    crime_df1: pd.DataFrame,
    crime_df2: pd.DataFrame,
    crime_df3: pd.DataFrame,
    excluded_crime_types: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Preprocess and consolidate the three crime datasets.

    Steps: drop metadata + clean ``value`` -> split title into
    Province/Cluster/Type of Crime -> clean labels -> keep modelling cols ->
    rename police clusters to geographic clusters -> drop SAPS aggregate/
    roll-up categories (``excluded_crime_types``) -> **sum duplicate
    (date, Cluster, Type of Crime) rows** (fixes the Polokwane double-count) ->
    filter to >= START_YEAR.
    """
    crime_dfs = [crime_df1, crime_df2, crime_df3]
    df = preprocess().fit_transform(crime_dfs)
    df = table_structure().fit_transform(df)
    df = data_split().fit_transform(df)
    df = data_cleaning().fit_transform(df)
    df = feature_engineering_crime().fit_transform(df)
    df = crime_data_cleaning().fit_transform(df)
    df = exclude_crime_categories(excluded_crime_types).fit_transform(df)
    df = aggregate_crime_duplicates().fit_transform(df)
    df = date_filter(start_year=START_YEAR).fit_transform(df)
    return df


def _preprocess_socio(df: pd.DataFrame, value_name: str) -> pd.DataFrame:
    """Shared socioeconomic preprocessing: clean -> extract cluster -> map to
    common key -> filter -> rename value column -> aggregate to one row per
    (date, Cluster)."""
    df = preprocess().fit_transform([df])
    df = table_structure().fit_transform(df)
    df = feature_engineering_socioeconomic().fit_transform(df)
    df = socio_data_cleaning().fit_transform(df)
    df = date_filter(start_year=START_YEAR).fit_transform(df)
    df = df.rename(columns={"value": value_name})
    # Collapse any sub-series (income bands, education levels, age bands, …)
    # into a single total per cluster-period so the downstream merge stays 1:1.
    return df.groupby(["date", "Cluster"], as_index=False)[value_name].sum()


def preprocess_population_density(population_density: pd.DataFrame) -> pd.DataFrame:
    """Preprocess population density dataset."""
    return _preprocess_socio(population_density, "population_density")


def preprocess_poor_households(poor_households: pd.DataFrame) -> pd.DataFrame:
    """Preprocess poor households dataset."""
    return _preprocess_socio(poor_households, "poor_households")


def preprocess_population_unemployment(population_unemployment: pd.DataFrame) -> pd.DataFrame:
    """Preprocess population unemployment dataset."""
    return _preprocess_socio(population_unemployment, "population_unemployment")


def preprocess_pop_edu(
    pop_edu_1: pd.DataFrame,
    pop_edu_2: pd.DataFrame,
    pop_edu_3: pd.DataFrame,
    pop_edu_4: pd.DataFrame,
    pop_edu_5: pd.DataFrame,
    pop_edu_6: pd.DataFrame,
) -> pd.DataFrame:
    """Preprocess and consolidate the six education datasets."""
    edu_ls = [pop_edu_1, pop_edu_2, pop_edu_3, pop_edu_4, pop_edu_5, pop_edu_6]
    df = preprocess().fit_transform(edu_ls)
    df = table_structure().fit_transform(df)
    df = feature_engineering_socioeconomic().fit_transform(df)
    df = socio_data_cleaning().fit_transform(df)
    df = date_filter(start_year=START_YEAR).fit_transform(df)
    df = df.rename(columns={"value": "population_education"})
    return df.groupby(["date", "Cluster"], as_index=False)["population_education"].sum()


def consolidate_socioeconomic_data(
    population_density_processed: pd.DataFrame,
    poor_households_processed: pd.DataFrame,
    population_unemployment_processed: pd.DataFrame,
    pop_edu_processed: pd.DataFrame,
) -> pd.DataFrame:
    """Merge all socioeconomic datasets into one (one row per cluster-period)."""
    all_dfs = [
        population_density_processed,
        poor_households_processed,
        population_unemployment_processed,
        pop_edu_processed,
    ]
    return socio_merger().fit_transform(all_dfs)


def consolidate_data(
    crime_processed: pd.DataFrame,
    socioeconomic_processed: pd.DataFrame,
) -> pd.DataFrame:
    """Merge crime and socioeconomic data into one master dataset.

    Crime is the left (many rows per date-Cluster, one per crime type);
    socioeconomic is the right (one row per date-Cluster). ``socio_merger``'s
    ``validate="many_to_one"`` enforces exactly that shape.
    """
    return socio_merger().fit_transform([crime_processed, socioeconomic_processed])

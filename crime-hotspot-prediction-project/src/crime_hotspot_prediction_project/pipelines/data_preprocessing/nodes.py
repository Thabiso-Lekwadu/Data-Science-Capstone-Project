# src/crime_hotspot_prediction_project/pipelines/data_preprocessing/nodes.py

import pandas as pd
from crime_hotspot_prediction_project.pipelines.data_preprocessing.transformers import (
    preprocess, table_structure,data_split,  data_cleaning,
    feature_engineering_crime, feature_engineering_socioeconomic,
    crime_data_cleaning, socio_data_cleaning, date_filter, socio_merger
)


def preprocess_crime_data(
    crime_df1, crime_df2, crime_df3
) -> pd.DataFrame:
    """Preprocess and consolidate crime datasets."""
    crime_dfs = [crime_df1, crime_df2, crime_df3]
    df = preprocess().fit_transform(crime_dfs)
    df = table_structure().fit_transform(df)
    df = data_split().fit_transform(df)
    df = data_cleaning().fit_transform(df)
    df = feature_engineering_crime().fit_transform(df)

    df = crime_data_cleaning().fit_transform(df)
    df = date_filter(start_year=2008).fit_transform(df)
    return df


def preprocess_population_density(population_density: pd.DataFrame) -> pd.DataFrame:
    """Preprocess population density dataset."""
    df = preprocess().fit_transform([population_density])
    df = table_structure().fit_transform(df)
    df = feature_engineering_socioeconomic().fit_transform(df)
    df = socio_data_cleaning().fit_transform(df)
    df = date_filter(start_year=2008).fit_transform(df)
    df = df.rename(columns={"value": "population_density"})
    return df


def preprocess_poor_households(poor_households: pd.DataFrame) -> pd.DataFrame:
    """Preprocess poor households dataset."""
    df = preprocess().fit_transform([poor_households])
    df = table_structure().fit_transform(df)
    df = feature_engineering_socioeconomic().fit_transform(df)
    df = socio_data_cleaning().fit_transform(df)
    df = date_filter(start_year=2008).fit_transform(df)
    df = df.rename(columns={"value": "poor_households"})
    return df


def preprocess_population_unemployment(population_unemployment: pd.DataFrame) -> pd.DataFrame:
    """Preprocess population unemployment dataset."""
    df = preprocess().fit_transform([population_unemployment])
    df = table_structure().fit_transform(df)
    df = feature_engineering_socioeconomic().fit_transform(df)
    df = socio_data_cleaning().fit_transform(df)
    df = date_filter(start_year=2008).fit_transform(df)
    df = df.rename(columns={"value": "population_unemployment"})
    return df


def preprocess_pop_edu(
    pop_edu_1, pop_edu_2, pop_edu_3,
    pop_edu_4, pop_edu_5, pop_edu_6
) -> pd.DataFrame:
    """Preprocess and consolidate education datasets."""
    edu_ls = [
    pop_edu_1,
    pop_edu_2,
    pop_edu_3,
    pop_edu_4,
    pop_edu_5,
    pop_edu_6
    ]
    df = preprocess().fit_transform(edu_ls)
    df = table_structure().fit_transform(df)
    df = feature_engineering_socioeconomic().fit_transform(df)
    df = socio_data_cleaning().fit_transform(df)
    df = date_filter(start_year=2008).fit_transform(df)
    df = df.rename(columns={"value": "population_education"})
    return df


def consolidate_socioeconomic_data(
    population_density_processed: pd.DataFrame,
    poor_households_processed: pd.DataFrame,
    population_unemployment_processed: pd.DataFrame,
    pop_edu_processed: pd.DataFrame,
) -> pd.DataFrame:
    """Merge all socioeconomic datasets into one."""
    all_dfs = [
        population_density_processed,
        poor_households_processed,
        population_unemployment_processed,
        pop_edu_processed,
    ]
    return socio_merger().fit_transform(all_dfs)


def consolidate_data(
    crime_processed: pd.DataFrame,
    socioeconomic_processed: pd.DataFrame
) -> pd.DataFrame:
    """Merge crime and socioeconomic data into one master dataset."""
    return socio_merger().fit_transform([crime_processed, socioeconomic_processed])
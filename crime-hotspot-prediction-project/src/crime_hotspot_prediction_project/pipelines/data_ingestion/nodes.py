"""
This is a boilerplate pipeline 'data_ingestion'
generated using Kedro 1.3.1
"""
import pandas as pd

def ingest_raw_data(
    crime_df1_raw, crime_df2_raw, crime_df3_raw,
    population_density_raw, poor_households_raw, population_unemployment_raw,
    pop_edu_1_raw, pop_edu_2_raw, pop_edu_3_raw,
    pop_edu_4_raw, pop_edu_5_raw, pop_edu_6_raw,
) -> tuple:

    """Pass-through node: loads from API, outputs to 01_raw CSVs."""
    return (
        crime_df1_raw, crime_df2_raw, crime_df3_raw,
        population_density_raw, poor_households_raw, population_unemployment_raw,
        pop_edu_1_raw, pop_edu_2_raw, pop_edu_3_raw,
        pop_edu_4_raw, pop_edu_5_raw, pop_edu_6_raw,
    )



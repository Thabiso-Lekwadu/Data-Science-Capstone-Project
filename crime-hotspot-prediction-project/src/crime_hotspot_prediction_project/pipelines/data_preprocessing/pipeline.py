# src/crime_hotspot_prediction_project/pipelines/data_preprocessing/pipeline.py
from kedro.pipeline import node, pipeline

from .nodes import (
    preprocess_crime_data,
    preprocess_population_density,
    preprocess_poor_households,
    preprocess_population_unemployment,
    preprocess_pop_edu,
    consolidate_socioeconomic_data,
    consolidate_data,
)

def create_pipeline(**kwargs):
    return pipeline([
        node(
            func=preprocess_crime_data,
            inputs=["crime_df1", "crime_df2", "crime_df3"],
            outputs="crime_processed",
            name="preprocess_crime_data_node",
        ),
        node(
            func=preprocess_population_density,
            inputs="population_density",
            outputs="population_density_processed",
            name="preprocess_population_density_node",
        ),
        node(
            func=preprocess_poor_households,
            inputs="poor_households",
            outputs="poor_households_processed",
            name="preprocess_poor_households_node",
        ),
        node(
            func=preprocess_population_unemployment,
            inputs="population_unemployment",
            outputs="population_unemployment_processed",
            name="preprocess_population_unemployment_node",
        ),
        node(
            func=preprocess_pop_edu,
            inputs=[
                "pop_edu_1", "pop_edu_2", "pop_edu_3",
                "pop_edu_4", "pop_edu_5", "pop_edu_6",
            ],
            outputs="pop_edu_processed",
            name="preprocess_pop_edu_node",
        ),
        node(
            func=consolidate_socioeconomic_data,
            inputs=[
                "population_density_processed",
                "poor_households_processed",
                "population_unemployment_processed",
                "pop_edu_processed",
            ],
            outputs="socioeconomic_processed",
            name="consolidate_socioeconomic_data_node",
        ),
        node(
            func=consolidate_data,
            inputs=["crime_processed", "socioeconomic_processed"],
            outputs="master_dataset",
            name="consolidate_data_node",
        ),
    ])
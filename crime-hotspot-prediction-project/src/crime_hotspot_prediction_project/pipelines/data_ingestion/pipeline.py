"""
This is a boilerplate pipeline 'data_ingestion'
generated using Kedro 1.3.1
"""
from kedro.pipeline import pipeline, node, Pipeline
from .nodes import ingest_raw_data

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=ingest_raw_data,
            inputs=[
                "crime_df1_raw", "crime_df2_raw", "crime_df3_raw",
                "population_density_raw", "poor_households_raw", "population_unemployment_raw",
                "pop_edu_1_raw", "pop_edu_2_raw", "pop_edu_3_raw",
                "pop_edu_4_raw", "pop_edu_5_raw", "pop_edu_6_raw",
            ],
            outputs=[
                "crime_df1", "crime_df2", "crime_df3",
                "population_density", "poor_households", "population_unemployment",
                "pop_edu_1", "pop_edu_2", "pop_edu_3",
                "pop_edu_4", "pop_edu_5", "pop_edu_6",
            ],
            name="ingest_raw_data_node",
        )
    ])

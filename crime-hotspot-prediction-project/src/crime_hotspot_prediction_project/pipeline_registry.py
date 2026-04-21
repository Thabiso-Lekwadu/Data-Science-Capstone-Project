"""Project pipelines."""
from __future__ import annotations

from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline


from crime_hotspot_prediction_project.pipelines import data_ingestion, data_preprocessing

def register_pipelines():
    return {
        "data_ingestion": data_ingestion.create_pipeline(),
        "data_preprocessing": data_preprocessing.create_pipeline(),
        "__default__": data_ingestion.create_pipeline() + data_preprocessing.create_pipeline(),
    }

"""Project pipelines."""
from __future__ import annotations

import logging

from kedro.pipeline import Pipeline

from crime_hotspot_prediction_project.pipelines import (
    data_ingestion,
    data_preprocessing,
    feature_engineering,
    model_training,
)

logger = logging.getLogger(__name__)


def register_pipelines() -> dict[str, Pipeline]:
    """Register and return all project pipelines."""

    pipelines = {
        "data_ingestion": data_ingestion.create_pipeline(),
        "data_preprocessing": data_preprocessing.create_pipeline(),
        "feature_engineering": feature_engineering.create_pipeline(),
        "model_training": model_training.create_pipeline(),
    }

    pipelines["__default__"] = (
        pipelines["data_ingestion"]
        + pipelines["data_preprocessing"]
        + pipelines["feature_engineering"]
        + pipelines["model_training"]
    )

    logger.info(
        "Registered pipelines: %s",
        [k for k in pipelines if k != "__default__"],
    )

    return pipelines
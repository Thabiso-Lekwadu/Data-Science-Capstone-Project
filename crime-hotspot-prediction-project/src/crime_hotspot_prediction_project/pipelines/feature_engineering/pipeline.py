"""Feature Engineering Pipeline."""
from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    engineer_crime_features,
    engineer_master_features,
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=engineer_crime_features,
                inputs=dict(
                    crime_processed="crime_processed",
                    lags="params:lag_periods",
                    rolling_windows="params:rolling_windows",
                ),
                outputs="crime_dataset_features",
                name="engineer_crime_features_node",
                tags=["feature_engineering", "crime_only"],
            ),
            node(
                func=engineer_master_features,
                inputs=dict(
                    master_dataset="master_dataset",
                    lags="params:lag_periods",
                    rolling_windows="params:rolling_windows",
                ),
                outputs="master_dataset_features",
                name="engineer_master_features_node",
                tags=["feature_engineering", "master"],
            ),
        ]
    )
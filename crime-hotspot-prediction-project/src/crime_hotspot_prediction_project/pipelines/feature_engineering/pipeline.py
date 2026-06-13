"""Feature Engineering Pipeline."""
from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    engineer_crime_dataset_tree,
    engineer_crime_dataset_knn,
    engineer_master_dataset_tree,
    engineer_master_dataset_knn,
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=engineer_crime_dataset_tree,
                inputs="crime_processed",
                outputs=[
                    "crime_dataset_tree_train",
                    "crime_dataset_tree_test",
                    "label_mapping_crime_tree",
                ],
                name="engineer_crime_dataset_tree_node",
                tags=["feature_engineering", "crime_only", "tree"],
            ),
            node(
                func=engineer_crime_dataset_knn,
                inputs="crime_processed",
                outputs=[
                    "crime_dataset_knn_train",
                    "crime_dataset_knn_test",
                    "label_mapping_crime_knn",
                ],
                name="engineer_crime_dataset_knn_node",
                tags=["feature_engineering", "crime_only", "knn"],
            ),
            node(
                func=engineer_master_dataset_tree,
                inputs="master_dataset",
                outputs=[
                    "master_dataset_tree_train",
                    "master_dataset_tree_test",
                    "label_mapping_master_tree",
                ],
                name="engineer_master_dataset_tree_node",
                tags=["feature_engineering", "master", "tree"],
            ),
            node(
                func=engineer_master_dataset_knn,
                inputs="master_dataset",
                outputs=[
                    "master_dataset_knn_train",
                    "master_dataset_knn_test",
                    "label_mapping_master_knn",
                ],
                name="engineer_master_dataset_knn_node",
                tags=["feature_engineering", "master", "knn"],
            ),
        ]
    )
"""Model Training Pipeline."""
from kedro.pipeline import Pipeline, node, pipeline

from .model_training_nodes import (
    train_evaluate_knn_crime,
    train_evaluate_knn_master,
    train_evaluate_rf_crime,
    train_evaluate_rf_master,
    train_evaluate_gbm_crime,
    train_evaluate_gbm_master,
    train_evaluate_xgb_crime,
    train_evaluate_xgb_master,
    consolidate_experiment_results,
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            # --- KNN ---
            node(
                func=train_evaluate_knn_crime,
                inputs=["crime_dataset_knn_train", "crime_dataset_knn_test"],
                outputs="knn_crime_results",
                name="train_evaluate_knn_crime_node",
                tags=["model_training", "knn", "crime_only"],
            ),
            node(
                func=train_evaluate_knn_master,
                inputs=["master_dataset_knn_train", "master_dataset_knn_test"],
                outputs="knn_master_results",
                name="train_evaluate_knn_master_node",
                tags=["model_training", "knn", "master"],
            ),
            # --- Random Forest ---
            node(
                func=train_evaluate_rf_crime,
                inputs=["crime_dataset_tree_train", "crime_dataset_tree_test"],
                outputs="rf_crime_results",
                name="train_evaluate_rf_crime_node",
                tags=["model_training", "random_forest", "crime_only"],
            ),
            node(
                func=train_evaluate_rf_master,
                inputs=["master_dataset_tree_train", "master_dataset_tree_test"],
                outputs="rf_master_results",
                name="train_evaluate_rf_master_node",
                tags=["model_training", "random_forest", "master"],
            ),
            # --- Gradient Boosting ---
            node(
                func=train_evaluate_gbm_crime,
                inputs=["crime_dataset_tree_train", "crime_dataset_tree_test"],
                outputs="gbm_crime_results",
                name="train_evaluate_gbm_crime_node",
                tags=["model_training", "gradient_boosting", "crime_only"],
            ),
            node(
                func=train_evaluate_gbm_master,
                inputs=["master_dataset_tree_train", "master_dataset_tree_test"],
                outputs="gbm_master_results",
                name="train_evaluate_gbm_master_node",
                tags=["model_training", "gradient_boosting", "master"],
            ),
            # --- XGBoost ---
            node(
                func=train_evaluate_xgb_crime,
                inputs=["crime_dataset_tree_train", "crime_dataset_tree_test"],
                outputs="xgb_crime_results",
                name="train_evaluate_xgb_crime_node",
                tags=["model_training", "xgboost", "crime_only"],
            ),
            node(
                func=train_evaluate_xgb_master,
                inputs=["master_dataset_tree_train", "master_dataset_tree_test"],
                outputs="xgb_master_results",
                name="train_evaluate_xgb_master_node",
                tags=["model_training", "xgboost", "master"],
            ),
            # --- Consolidate ---
            node(
                func=consolidate_experiment_results,
                inputs=[
                    "knn_crime_results", "knn_master_results",
                    "rf_crime_results", "rf_master_results",
                    "gbm_crime_results", "gbm_master_results",
                    "xgb_crime_results", "xgb_master_results",
                ],
                outputs="experiment_results",
                name="consolidate_experiment_results_node",
                tags=["model_training", "reporting"],
            ),
        ]
    )
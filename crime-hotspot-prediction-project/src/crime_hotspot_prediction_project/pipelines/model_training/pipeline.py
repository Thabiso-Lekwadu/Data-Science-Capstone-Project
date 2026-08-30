"""Model Training Pipeline.

Each train node emits two outputs: the per-fold results table (for reporting)
and the fitted final model (persisted by Kedro as a PickleDataset declared in
catalog.yml -> data/06_models/<model>_<condition>.pkl).
"""
from kedro.pipeline import Pipeline, node, pipeline

from .model_training_nodes import (
    compute_fold_boundaries,
    train_evaluate_rf_crime,
    train_evaluate_rf_master,
    train_evaluate_xgb_crime,
    train_evaluate_xgb_master,
    train_evaluate_lgbm_crime,
    train_evaluate_lgbm_master,
    train_evaluate_catboost_crime,
    train_evaluate_catboost_master,
    consolidate_experiment_results,
    summarize_experiment_results,
    run_paired_significance_tests,
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            # --- Shared fold boundaries (computed once, reused by every model/condition) ---
            node(
                func=compute_fold_boundaries,
                inputs=dict(
                    master_dataset_features="master_dataset_features",
                    n_splits="params:n_splits",
                ),
                outputs="fold_boundaries",
                name="compute_fold_boundaries_node",
                tags=["model_training", "cv_setup"],
            ),

            # --- Random Forest ---
            node(
                func=train_evaluate_rf_crime,
                inputs=["crime_dataset_features", "fold_boundaries"],
                outputs=["rf_crime_results", "rf_crime_only_model"],
                name="train_evaluate_rf_crime_node",
                tags=["model_training", "random_forest", "crime_only"],
            ),
            node(
                func=train_evaluate_rf_master,
                inputs=["master_dataset_features", "fold_boundaries"],
                outputs=["rf_master_results", "rf_master_model"],
                name="train_evaluate_rf_master_node",
                tags=["model_training", "random_forest", "master"],
            ),

            # --- XGBoost ---
            node(
                func=train_evaluate_xgb_crime,
                inputs=["crime_dataset_features", "fold_boundaries"],
                outputs=["xgb_crime_results", "xgb_crime_only_model"],
                name="train_evaluate_xgb_crime_node",
                tags=["model_training", "xgboost", "crime_only"],
            ),
            node(
                func=train_evaluate_xgb_master,
                inputs=["master_dataset_features", "fold_boundaries"],
                outputs=["xgb_master_results", "xgb_master_model"],
                name="train_evaluate_xgb_master_node",
                tags=["model_training", "xgboost", "master"],
            ),

            # --- LightGBM ---
            node(
                func=train_evaluate_lgbm_crime,
                inputs=["crime_dataset_features", "fold_boundaries"],
                outputs=["lgbm_crime_results", "lgbm_crime_only_model"],
                name="train_evaluate_lgbm_crime_node",
                tags=["model_training", "lightgbm", "crime_only"],
            ),
            node(
                func=train_evaluate_lgbm_master,
                inputs=["master_dataset_features", "fold_boundaries"],
                outputs=["lgbm_master_results", "lgbm_master_model"],
                name="train_evaluate_lgbm_master_node",
                tags=["model_training", "lightgbm", "master"],
            ),

            # --- CatBoost ---
            node(
                func=train_evaluate_catboost_crime,
                inputs=["crime_dataset_features", "fold_boundaries"],
                outputs=["catboost_crime_results", "catboost_crime_only_model"],
                name="train_evaluate_catboost_crime_node",
                tags=["model_training", "catboost", "crime_only"],
            ),
            node(
                func=train_evaluate_catboost_master,
                inputs=["master_dataset_features", "fold_boundaries"],
                outputs=["catboost_master_results", "catboost_master_model"],
                name="train_evaluate_catboost_master_node",
                tags=["model_training", "catboost", "master"],
            ),

            # --- Consolidate + report ---
            node(
                func=consolidate_experiment_results,
                inputs=[
                    "rf_crime_results", "rf_master_results",
                    "xgb_crime_results", "xgb_master_results",
                    "lgbm_crime_results", "lgbm_master_results",
                    "catboost_crime_results", "catboost_master_results",
                ],
                outputs="experiment_results_per_fold",
                name="consolidate_experiment_results_node",
                tags=["model_training", "reporting"],
            ),
            node(
                func=summarize_experiment_results,
                inputs="experiment_results_per_fold",
                outputs="experiment_results_summary",
                name="summarize_experiment_results_node",
                tags=["model_training", "reporting"],
            ),
            node(
                func=run_paired_significance_tests,
                inputs="experiment_results_per_fold",
                outputs="experiment_significance_tests",
                name="run_paired_significance_tests_node",
                tags=["model_training", "reporting"],
            ),
        ]
    )

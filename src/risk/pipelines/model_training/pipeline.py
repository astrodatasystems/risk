"""Model training pipeline — Gold features to LightGBM damage regressor.

Node dependency graph:
    storm_events_gold -> [create_training_data] -> training_data
    training_data -> [split_train_test] -> X_train, X_test, y_train, y_test
    X_train, y_train -> [train_damage_model] -> damage_model
    damage_model, X_test, y_test -> [evaluate_model]        -> evaluation_metrics, tier_thresholds
    damage_model                 -> [extract_feature_importance] -> feature_importance
    damage_model                 -> [save_model_artifact]       -> damage_model_artifact

Nodes 4-6 can run in parallel (all depend on damage_model, independent of each other).
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    create_training_data,
    evaluate_model,
    extract_feature_importance,
    save_model_artifact,
    split_train_test,
    train_damage_model,
)


def create_pipeline(**kwargs) -> Pipeline:  # noqa: ARG001
    """Create the model_training pipeline."""
    return pipeline(
        [
            node(
                func=create_training_data,
                inputs="storm_events_gold",
                outputs="training_data",
                name="create_training_data",
            ),
            node(
                func=split_train_test,
                inputs=["training_data", "params:model_training"],
                outputs=["X_train", "X_test", "y_train", "y_test"],
                name="split_train_test",
            ),
            node(
                func=train_damage_model,
                inputs=["X_train", "y_train", "params:model_training.model_params"],
                outputs="damage_model",
                name="train_damage_model",
            ),
            node(
                func=evaluate_model,
                inputs=[
                    "damage_model",
                    "X_test",
                    "y_test",
                    "params:model_training.tier_thresholds",
                ],
                outputs=["evaluation_metrics", "tier_thresholds"],
                name="evaluate_model",
            ),
            node(
                func=extract_feature_importance,
                inputs="damage_model",
                outputs="feature_importance",
                name="extract_feature_importance",
            ),
            node(
                func=save_model_artifact,
                inputs="damage_model",
                outputs="damage_model_artifact",
                name="save_model_artifact",
            ),
        ]
    )

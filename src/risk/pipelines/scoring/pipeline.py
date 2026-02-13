"""Scoring pipeline — Gold features → batch risk scores.

Node dependency graph:
    storm_events_gold → [load_latest_features] → scoring_features
    scoring_features, damage_model_artifact, tier_thresholds
        → [score_counties] → scored_counties
    scored_counties → [prepare_output_table] → county_risk_scores

All three nodes run sequentially (each depends on the previous output).
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import load_latest_features, prepare_output_table, score_counties


def create_pipeline(**kwargs) -> Pipeline:  # noqa: ARG001
    """Create the scoring pipeline."""
    return pipeline(
        [
            node(
                func=load_latest_features,
                inputs=["storm_events_gold", "params:scoring.feature_year"],
                outputs="scoring_features",
                name="load_latest_features",
            ),
            node(
                func=score_counties,
                inputs=[
                    "scoring_features",
                    "damage_model_artifact",
                    "tier_thresholds",
                    "params:model_training.feature_columns",
                ],
                outputs="scored_counties",
                name="score_counties",
            ),
            node(
                func=prepare_output_table,
                inputs=["scored_counties", "params:scoring"],
                outputs="county_risk_scores",
                name="prepare_output_table",
            ),
        ]
    )

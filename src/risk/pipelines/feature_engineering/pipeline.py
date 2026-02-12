"""Silver → Gold pipeline for NOAA Storm Events features.

This pipeline reads the cleaned Silver parquet, enriches rows with
parsed severity columns and event-type flags, aggregates to the
county-year grain, and outputs a Gold feature table for ML training.
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    aggregate_county_year_features,
    post_process_features,
    prepare_event_features,
)


def create_pipeline(**kwargs) -> Pipeline:  # noqa: ARG001
    """Create the feature_engineering pipeline.

    Node chain:
        Silver parquet → prepare (row enrichment)
        → aggregate (county-year groupby)
        → post-process (derived ratios) → Gold parquet
    """
    return pipeline(
        [
            node(
                func=prepare_event_features,
                inputs="storm_events_silver",
                outputs="events_prepared",
                name="prepare_event_features",
            ),
            node(
                func=aggregate_county_year_features,
                inputs="events_prepared",
                outputs="county_year_aggregated",
                name="aggregate_county_year_features",
            ),
            node(
                func=post_process_features,
                inputs="county_year_aggregated",
                outputs="storm_events_gold",
                name="post_process_features",
            ),
        ]
    )

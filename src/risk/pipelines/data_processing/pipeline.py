"""Bronze → Silver pipeline for NOAA Storm Events data.

This pipeline reads raw CSVs from the Bronze layer, applies six
sequential transformation nodes, and outputs a cleaned, typed,
county-level Silver parquet file ready for feature engineering.
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    build_county_fips,
    filter_county_events,
    load_and_combine_bronze,
    parse_damage_strings,
    parse_timestamps,
    select_and_clean_columns,
)


def create_pipeline(**kwargs) -> Pipeline:  # noqa: ARG001
    """Create the data_processing pipeline.

    Node chain:
        raw CSVs → combine → select columns → filter counties
        → parse damage → build FIPS → parse timestamps → Silver parquet
    """
    return pipeline(
        [
            node(
                func=load_and_combine_bronze,
                inputs="params:raw_data_path",
                outputs="bronze_combined",
                name="load_and_combine_bronze",
            ),
            node(
                func=select_and_clean_columns,
                inputs="bronze_combined",
                outputs="bronze_selected",
                name="select_and_clean_columns",
            ),
            node(
                func=filter_county_events,
                inputs="bronze_selected",
                outputs="county_events",
                name="filter_county_events",
            ),
            node(
                func=parse_damage_strings,
                inputs="county_events",
                outputs="county_events_with_damage",
                name="parse_damage_strings",
            ),
            node(
                func=build_county_fips,
                inputs="county_events_with_damage",
                outputs="county_events_with_fips",
                name="build_county_fips",
            ),
            node(
                func=parse_timestamps,
                inputs="county_events_with_fips",
                outputs="storm_events_silver",
                name="parse_timestamps",
            ),
        ]
    )

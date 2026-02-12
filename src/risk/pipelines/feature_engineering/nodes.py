"""Silver → Gold feature engineering nodes for NOAA Storm Events.

Three-node pipeline that transforms event-level Silver data into a
county-year feature table (Gold) for ML model input.

Architecture:
    prepare  → row-level enrichment (before groupby)
    aggregate → single groupby (county_fips, year)
    post-process → derived ratios + cleanup (after groupby)
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)

# ── Event types we track as boolean flags ────────────────────────
# These are the top weather perils for property insurance in the Midwest.
# Mapped to NOAA's EVENT_TYPE values (already uppercased in Silver).
_EVENT_TYPE_FLAGS: dict[str, str] = {
    "is_tornado": "Tornado",
    "is_hail": "Hail",
    "is_thunderstorm_wind": "Thunderstorm Wind",
    "is_flash_flood": "Flash Flood",
    "is_winter_weather": "Winter Storm",
}


# ── Helper: parse EF/F scale strings to numeric ─────────────────
def _parse_ef_scale(value: object) -> float | None:
    """Convert a tornado F/EF scale string to a numeric value.

    Handles both the modern Enhanced Fujita (EF0–EF5) and the legacy
    Fujita (F0–F5) scale. Returns None for missing or unknown values.

    Examples:
        "EF3" → 3.0
        "F2"  → 2.0
        "EFU" → None  (EFU = "EF Unknown")
        NaN   → None
    """
    if pd.isna(value):
        return None

    text = str(value).strip().upper()

    # "EFU" or "FU" = unknown rating
    if text in ("EFU", "FU", ""):
        return None

    # "EF3" → 3, "F2" → 2
    for prefix in ("EF", "F"):
        if text.startswith(prefix):
            try:
                return float(text[len(prefix):])
            except ValueError:
                return None

    return None


# ── Node 1: Row-level enrichment ─────────────────────────────────
def prepare_event_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add row-level columns needed before aggregation.

    Creates:
        - ef_scale_numeric: parsed tornado EF/F scale as int
        - hail_magnitude / wind_magnitude: magnitude split by event type
        - is_tornado, is_hail, etc.: boolean flags per event type
        - quarter: 1–4 from month

    Args:
        df: Silver-layer event-level DataFrame.

    Returns:
        DataFrame with new columns appended.
    """
    df = df.copy()

    # Tornado EF scale → numeric
    df["ef_scale_numeric"] = df["tor_f_scale"].apply(_parse_ef_scale)
    n_parsed = df["ef_scale_numeric"].notna().sum()
    n_tornadoes_total = (df["event_type"] == "Tornado").sum()
    logger.info(
        "EF scale parsed: %s of %s tornado events have a numeric rating",
        f"{n_parsed:,}",
        f"{n_tornadoes_total:,}",
    )

    # Conditional magnitude columns — only meaningful for specific event types
    df["hail_magnitude"] = df["magnitude"].where(df["event_type"] == "Hail")
    df["wind_magnitude"] = df["magnitude"].where(
        df["event_type"] == "Thunderstorm Wind"
    )

    # Boolean event-type flags (int for easy summation in groupby)
    for flag_col, event_type in _EVENT_TYPE_FLAGS.items():
        df[flag_col] = (df["event_type"] == event_type).astype(int)

    # Quarter from month: Jan–Mar=1, Apr–Jun=2, Jul–Sep=3, Oct–Dec=4
    df["quarter"] = (df["month"] - 1) // 3 + 1

    logger.info(
        "Event features prepared: %s rows, %d new columns added",
        f"{len(df):,}",
        7 + len(_EVENT_TYPE_FLAGS),  # ef + hail_mag + wind_mag + quarter + flags
    )
    return df


# ── Node 2: County-year aggregation ──────────────────────────────
def aggregate_county_year_features(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate event-level rows into one row per county per year.

    Uses pandas named aggregation for clarity and efficiency —
    a single groupby call rather than multiple passes over the data.

    Args:
        df: Enriched event-level DataFrame from prepare step.

    Returns:
        County-year DataFrame with aggregated features.
    """
    agg_result = df.groupby(["county_fips", "year"], as_index=False).agg(
        # Geography (constant within a county_fips)
        state=("state", "first"),
        cz_name=("cz_name", "first"),
        # Volume
        total_events=("event_id", "count"),
        distinct_months=("month", "nunique"),
        # Damage
        total_property_damage_dollars=("damage_property_dollars", "sum"),
        total_crop_damage_dollars=("damage_crops_dollars", "sum"),
        max_single_event_damage=("damage_property_dollars", "max"),
        median_event_damage=("damage_property_dollars", "median"),
        # Event type counts
        n_tornadoes=("is_tornado", "sum"),
        n_hail=("is_hail", "sum"),
        n_thunderstorm_wind=("is_thunderstorm_wind", "sum"),
        n_flash_flood=("is_flash_flood", "sum"),
        n_winter_weather=("is_winter_weather", "sum"),
        # Severity
        max_tornado_ef_scale=("ef_scale_numeric", "max"),
        mean_hail_magnitude=("hail_magnitude", "mean"),
        mean_wind_magnitude=("wind_magnitude", "mean"),
        total_injuries=("injuries_direct", "sum"),
        total_deaths=("deaths_direct", "sum"),
        # Quarterly event counts (will become percentages in post-processing)
        events_q1=("quarter", lambda x: (x == 1).sum()),
        events_q2=("quarter", lambda x: (x == 2).sum()),
        events_q3=("quarter", lambda x: (x == 3).sum()),
        events_q4=("quarter", lambda x: (x == 4).sum()),
    )

    # Fill damage NaNs with 0 (no reported damage = $0)
    damage_cols = [
        "total_property_damage_dollars",
        "total_crop_damage_dollars",
        "max_single_event_damage",
        "median_event_damage",
    ]
    agg_result[damage_cols] = agg_result[damage_cols].fillna(0.0)

    # Cast integer columns that pandas may have upcasted to float
    int_cols = [
        "total_events",
        "n_tornadoes",
        "n_hail",
        "n_thunderstorm_wind",
        "n_flash_flood",
        "n_winter_weather",
        "total_injuries",
        "total_deaths",
    ]
    agg_result[int_cols] = agg_result[int_cols].astype("int64")

    logger.info(
        "Aggregated to %s county-year rows from %s events "
        "(%s unique counties, years %d–%d)",
        f"{len(agg_result):,}",
        f"{len(df):,}",
        f"{agg_result['county_fips'].nunique():,}",
        agg_result["year"].min(),
        agg_result["year"].max(),
    )
    return agg_result


# ── Node 3: Post-processing ──────────────────────────────────────
def post_process_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute derived ratios and finalize the Gold feature table.

    Creates:
        - events_per_month: average events per active month
        - pct_events_q1–q4: seasonal distribution as percentages (0–100)

    Drops helper columns that were only needed for computation.

    Args:
        df: Aggregated county-year DataFrame.

    Returns:
        Final Gold feature table ready for ML model input.
    """
    df = df.copy()

    # Events per month = total events / distinct months with events
    df["events_per_month"] = df["total_events"] / df["distinct_months"]

    # Quarterly percentages (0–100 scale)
    for q in range(1, 5):
        df[f"pct_events_q{q}"] = df[f"events_q{q}"] / df["total_events"] * 100

    # Drop helper columns no longer needed
    helper_cols = ["distinct_months", "events_q1", "events_q2", "events_q3", "events_q4"]
    df = df.drop(columns=helper_cols)

    logger.info(
        "Gold feature table ready: %s rows × %d columns "
        "(%s unique counties, years %d–%d)",
        f"{len(df):,}",
        len(df.columns),
        f"{df['county_fips'].nunique():,}",
        df["year"].min(),
        df["year"].max(),
    )
    return df

"""Bronze → Silver transformation nodes for NOAA Storm Events data.

Each function is a Kedro node: pure input → output, no side effects.
Together they form the data_processing pipeline that takes raw CSVs
from the Bronze layer and produces a cleaned, typed, county-level
Silver parquet file.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# ── Columns we keep from the 51-column raw data ─────────────────────
KEEP_COLUMNS: list[str] = [
    "event_id",
    "episode_id",
    "begin_date_time",
    "end_date_time",
    "state",
    "state_fips",
    "cz_type",
    "cz_fips",
    "cz_name",
    "event_type",
    "damage_property",
    "damage_crops",
    "injuries_direct",
    "deaths_direct",
    "magnitude",
    "magnitude_type",
    "tor_f_scale",
    "begin_lat",
    "begin_lon",
    "source_file",
]

# ── Multipliers for damage strings like "25.00K", "1.50M" ───────────
_DAMAGE_MULTIPLIERS: dict[str, float] = {
    "K": 1_000,
    "M": 1_000_000,
    "B": 1_000_000_000,
}

# Regex: optional number (with decimal) followed by optional K/M/B
_DAMAGE_PATTERN: re.Pattern = re.compile(
    r"^\s*([\d.]+)\s*([KMBkmb])?\s*$"
)


# ── Node 1 ───────────────────────────────────────────────────────────
def load_and_combine_bronze(raw_data_path: str) -> pd.DataFrame:
    """Read all CSVs from the Bronze folder and concatenate into one DataFrame.

    Adds a ``source_file`` column so every row is traceable to its
    origin file — important for the audit-trail requirement.

    Args:
        raw_data_path: Path to the folder containing raw CSVs.

    Returns:
        Combined DataFrame with all years and a source_file column.
    """
    folder = Path(raw_data_path)
    csv_files = sorted(folder.glob("storm_events_details_*.csv"))

    if not csv_files:
        raise FileNotFoundError(
            f"No storm_events_details_*.csv files found in {folder}"
        )

    logger.info("Found %d raw CSV files in %s", len(csv_files), folder)

    frames: list[pd.DataFrame] = []
    for csv_file in csv_files:
        df = pd.read_csv(csv_file, low_memory=False)
        df.columns = df.columns.str.lower()  # UPPERCASE → snake_case
        df["source_file"] = csv_file.name
        frames.append(df)
        logger.info(
            "  Loaded %s: %s rows, %s columns",
            csv_file.name,
            f"{len(df):,}",
            len(df.columns),
        )

    combined = pd.concat(frames, ignore_index=True)
    logger.info(
        "Combined Bronze DataFrame: %s rows, %s columns",
        f"{len(combined):,}",
        len(combined.columns),
    )
    return combined


# ── Node 2 ───────────────────────────────────────────────────────────
def select_and_clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only the columns needed for risk modelling.

    The raw NOAA data has 51 columns. Most are irrelevant to our
    county-level risk model (e.g., narrative text, indirect injuries,
    tornado path details). Dropping them early keeps memory low and
    makes downstream logic clearer.

    Args:
        df: Combined Bronze DataFrame.

    Returns:
        DataFrame with only the columns in KEEP_COLUMNS.
    """
    missing = [c for c in KEEP_COLUMNS if c not in df.columns]
    if missing:
        raise KeyError(f"Expected columns not found in data: {missing}")

    before_cols = len(df.columns)
    df_selected = df[KEEP_COLUMNS].copy()
    dropped = before_cols - len(df_selected.columns)

    logger.info(
        "Column selection: kept %d of %d columns (dropped %d)",
        len(KEEP_COLUMNS),
        before_cols,
        dropped,
    )
    return df_selected


# ── Node 3 ───────────────────────────────────────────────────────────
def filter_county_events(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only county-level events (cz_type == 'C').

    NOAA Storm Events assigns each record to one of three zone types:
    - C = County/Parish  (what we need for county-level risk)
    - Z = NWS Forecast Zone  (overlapping, non-standard boundaries)
    - M = Marine Zone  (offshore, not relevant to property insurance)

    We drop Z and M because our model operates at the FIPS county level.

    Args:
        df: DataFrame after column selection.

    Returns:
        DataFrame containing only county-level events.
    """
    total = len(df)
    county_mask = df["cz_type"] == "C"
    df_county = df[county_mask].copy()

    dropped = total - len(df_county)
    pct_dropped = (dropped / total * 100) if total > 0 else 0

    # Log the breakdown by zone type for transparency
    type_counts = df["cz_type"].value_counts().to_dict()
    logger.info(
        "Zone type distribution: %s",
        {k: f"{v:,}" for k, v in type_counts.items()},
    )
    logger.info(
        "County filter: kept %s of %s rows (dropped %s = %.1f%%)",
        f"{len(df_county):,}",
        f"{total:,}",
        f"{dropped:,}",
        pct_dropped,
    )
    return df_county


# ── Node 4 ───────────────────────────────────────────────────────────
def _parse_single_damage(value: object) -> float | None:
    """Convert a single damage string like '25.00K' to a float.

    Returns None if the value cannot be parsed.
    """
    if pd.isna(value):
        return None

    text = str(value).strip()
    if text == "" or text.lower() == "nan":
        return None

    match = _DAMAGE_PATTERN.match(text)
    if match:
        number = float(match.group(1))
        suffix = match.group(2)
        if suffix:
            number *= _DAMAGE_MULTIPLIERS[suffix.upper()]
        return number

    # Try direct numeric conversion as fallback
    try:
        return float(text)
    except (ValueError, TypeError):
        return None


def parse_damage_strings(df: pd.DataFrame) -> pd.DataFrame:
    """Convert damage_property and damage_crops from strings to dollar amounts.

    NOAA encodes damage as strings with K/M/B suffixes:
    - "25.00K" → 25,000.0
    - "1.50M"  → 1,500,000.0
    - "0.00K"  → 0.0
    - NaN      → NaN (preserved as missing)

    Creates two new columns (damage_property_dollars, damage_crops_dollars)
    and keeps the originals for auditability.

    Args:
        df: DataFrame after county filtering.

    Returns:
        DataFrame with new numeric damage columns added.
    """
    df = df.copy()

    for col, new_col in [
        ("damage_property", "damage_property_dollars"),
        ("damage_crops", "damage_crops_dollars"),
    ]:
        parsed = df[col].apply(_parse_single_damage)
        unparseable = df[col].notna() & parsed.isna()
        n_unparseable = unparseable.sum()
        n_total = df[col].notna().sum()

        if n_unparseable > 0:
            # Log sample of unparseable values for debugging
            bad_samples = df.loc[unparseable, col].unique()[:10]
            logger.warning(
                "%s: %s of %s non-null values could not be parsed. "
                "Samples: %s",
                col,
                f"{n_unparseable:,}",
                f"{n_total:,}",
                list(bad_samples),
            )
        else:
            logger.info(
                "%s: all %s non-null values parsed successfully",
                col,
                f"{n_total:,}",
            )

        df[new_col] = parsed

    return df


# ── Node 5 ───────────────────────────────────────────────────────────
def build_county_fips(df: pd.DataFrame) -> pd.DataFrame:
    """Create a 5-digit county FIPS code from state_fips and cz_fips.

    The FIPS code is the standard identifier used across government
    datasets (Census, USDA, FEMA). Format: SSCCC where SS = 2-digit
    state code and CCC = 3-digit county code.

    Example: state_fips=17, cz_fips=31 → county_fips="17031" (Cook County, IL)

    Args:
        df: DataFrame after damage parsing.

    Returns:
        DataFrame with a new county_fips column.
    """
    df = df.copy()

    # Convert to int first to handle any float representations, then to string
    state_str = df["state_fips"].astype("Int64").astype(str).str.zfill(2)
    county_str = df["cz_fips"].astype("Int64").astype(str).str.zfill(3)
    df["county_fips"] = state_str + county_str

    # Handle any rows where FIPS components were null
    null_mask = df["state_fips"].isna() | df["cz_fips"].isna()
    if null_mask.any():
        df.loc[null_mask, "county_fips"] = None
        logger.warning(
            "county_fips: %s rows had null state_fips or cz_fips",
            f"{null_mask.sum():,}",
        )

    n_unique = df["county_fips"].nunique()
    logger.info(
        "Built county_fips: %s unique counties across %s rows",
        f"{n_unique:,}",
        f"{len(df):,}",
    )
    return df


# ── Node 6 ───────────────────────────────────────────────────────────
def parse_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    """Convert date/time strings to proper datetime objects and extract parts.

    NOAA stores timestamps as strings like "01-JAN-23 14:30:00".
    We parse these into proper datetime columns, then extract year,
    month, and day_of_year as integers for easier aggregation in the
    Gold layer (e.g., seasonal patterns, year-over-year trends).

    Args:
        df: DataFrame after FIPS construction.

    Returns:
        DataFrame with parsed datetime columns and extracted date parts.
    """
    df = df.copy()

    for col in ["begin_date_time", "end_date_time"]:
        df[col] = pd.to_datetime(df[col], format="mixed", dayfirst=False)
        n_null = df[col].isna().sum()
        if n_null > 0:
            logger.warning(
                "%s: %s values could not be parsed to datetime",
                col,
                f"{n_null:,}",
            )

    # Extract date components from begin_date_time
    df["year"] = df["begin_date_time"].dt.year
    df["month"] = df["begin_date_time"].dt.month
    df["day_of_year"] = df["begin_date_time"].dt.day_of_year

    logger.info(
        "Timestamps parsed. Year range: %d–%d",
        df["year"].min(),
        df["year"].max(),
    )
    return df

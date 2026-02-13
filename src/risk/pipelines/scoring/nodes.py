"""Node functions for the scoring pipeline.

Three nodes that take Gold features for the most recent year, run them
through the trained LightGBM damage model, and produce a county-level
risk score table ready for the underwriting platform.

Flow:
    storm_events_gold (2024 only) → score with model → shape output table
"""

from __future__ import annotations

import logging
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ── helper ──────────────────────────────────────────────────────
def _assign_tier(
    damage: pd.Series,
    low_max: float,
    moderate_max: float,
    high_max: float,
) -> pd.Series:
    """Bucket continuous dollar damage into risk tiers.

    Same logic as model_training._assign_tier — duplicated here to keep
    pipelines self-contained per Kedro convention.  If a third pipeline
    needs this, extract to risk.utils.

    Thresholds:
        Low:      < low_max       ($100 K)
        Moderate: low_max – moderate_max  ($100 K – $1 M)
        High:     moderate_max – high_max ($1 M – $10 M)
        Extreme:  >= high_max     ($10 M+)
    """
    conditions = [
        damage < low_max,
        damage < moderate_max,
        damage < high_max,
    ]
    choices = ["Low", "Moderate", "High"]
    return pd.Series(
        np.select(conditions, choices, default="Extreme"),
        index=damage.index,
    )


# ── Node 1 ──────────────────────────────────────────────────────
def load_latest_features(
    storm_events_gold: pd.DataFrame,
    feature_year: int,
) -> pd.DataFrame:
    """Filter Gold features to the most recent year for scoring.

    In our lagged design, features from year Y predict damage in year Y+1.
    So feature_year=2024 means we're predicting 2025 risk.

    Args:
        storm_events_gold: Full Gold feature table (all years).
        feature_year: The year to filter to (from parameters).

    Returns:
        DataFrame with only rows for the specified year.
    """
    latest = storm_events_gold[storm_events_gold["year"] == feature_year].copy()

    logger.info(
        "Loaded %d counties for feature_year=%d (predicting %d risk)",
        len(latest),
        feature_year,
        feature_year + 1,
    )
    return latest


# ── Node 2 ──────────────────────────────────────────────────────
def score_counties(
    scoring_features: pd.DataFrame,
    damage_model: lgb.LGBMRegressor,
    tier_thresholds: dict[str, float],
    feature_columns: list[str],
) -> pd.DataFrame:
    """Score every county: predict damage dollars and assign risk tier.

    The model was trained on log1p(damage), so we:
    1. Extract the 19 feature columns in the exact training order
    2. model.predict() → log-space predictions
    3. np.expm1() → inverse transform back to dollar space
    4. Clamp negatives to zero (expm1 of small negative log-preds)
    5. Apply tier thresholds to get Low/Moderate/High/Extreme

    Args:
        scoring_features: Gold features filtered to one year.
        damage_model: Trained LightGBM regressor (predicts in log-space).
        tier_thresholds: Dict with low_max, moderate_max, high_max.
        feature_columns: List of 19 feature column names (from params).

    Returns:
        Input DataFrame with predicted_damage_dollars and
        predicted_risk_tier columns added.
    """
    X = scoring_features[feature_columns]
    preds_log = damage_model.predict(X)
    preds_dollars = np.maximum(np.expm1(preds_log), 0.0)

    scored = scoring_features.copy()
    scored["predicted_damage_dollars"] = preds_dollars
    scored["predicted_risk_tier"] = _assign_tier(
        pd.Series(preds_dollars, index=scored.index),
        tier_thresholds["low_max"],
        tier_thresholds["moderate_max"],
        tier_thresholds["high_max"],
    )

    # --- Log summary statistics ---
    tier_counts = scored["predicted_risk_tier"].value_counts().sort_index()
    logger.info("Tier distribution:\n%s", tier_counts.to_string())
    logger.info(
        "Predicted damage — min: $%s | median: $%s | max: $%s",
        f"{preds_dollars.min():,.0f}",
        f"{np.median(preds_dollars):,.0f}",
        f"{preds_dollars.max():,.0f}",
    )

    return scored


# ── Node 3 ──────────────────────────────────────────────────────
def prepare_output_table(
    scored_counties: pd.DataFrame,
    scoring_params: dict[str, Any],
) -> pd.DataFrame:
    """Shape the final output table for the underwriting platform.

    Selects only the columns underwriters need, adds metadata
    (model version, scoring timestamp, feature year), and drops
    all raw feature columns.

    In production this table lands in BigQuery gold.county_risk_scores
    where the underwriting app reads it.

    Args:
        scored_counties: Full DataFrame with features + predictions.
        scoring_params: Dict with model_version and feature_year.

    Returns:
        Clean output table with 8 columns.
    """
    output = scored_counties[
        [
            "county_fips",
            "state",
            "cz_name",
            "predicted_damage_dollars",
            "predicted_risk_tier",
        ]
    ].copy()

    output["model_version"] = scoring_params["model_version"]
    output["scored_at"] = pd.Timestamp.now(tz="UTC")
    output["feature_year"] = scoring_params["feature_year"]

    logger.info(
        "Output table: %d rows, %d columns, model=%s, scored_at=%s",
        len(output),
        output.shape[1],
        scoring_params["model_version"],
        output["scored_at"].iloc[0].isoformat(),
    )

    return output

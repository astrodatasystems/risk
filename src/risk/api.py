"""FastAPI application for real-time county-level risk predictions.

This is the real-time serving path (Layer 4).  The batch path is the
Kedro ``scoring`` pipeline that writes to parquet / BigQuery.  Both
paths use the same model and tier thresholds so predictions are
consistent regardless of how you query them.

Run locally:
    uvicorn risk.api:app --host 0.0.0.0 --port 8080

In the Docker container (Cloud Run):
    CMD ["uvicorn", "risk.api:app", "--host", "0.0.0.0", "--port", "8080"]
"""

from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query

logger = logging.getLogger(__name__)

# ── Configuration ────────────────────────────────────────────────
# Paths are relative to the project root (where uvicorn is launched).
# In production these would point to GCS or be overridden via env vars.
MODEL_PATH = Path("data/06_models/damage_model.joblib")
THRESHOLDS_PATH = Path("data/06_models/tier_thresholds.json")
GOLD_PATH = Path("data/04_feature/storm_events_gold.parquet")
MODEL_VERSION = "v1.0-log_transform"

# Module-level state populated at startup, read at request time.
# Using a dict so we can mutate from inside the lifespan function
# without the ``global`` keyword (cleaner than five global statements).
_state: dict = {}


# ── Helpers ──────────────────────────────────────────────────────
def _assign_tier(damage: float, thresholds: dict[str, float]) -> str:
    """Map a dollar damage value to a risk tier.

    Same thresholds as the batch scoring pipeline and training
    evaluation — single source of truth is tier_thresholds.json
    produced by the training pipeline.
    """
    if damage < thresholds["low_max"]:
        return "Low"
    if damage < thresholds["moderate_max"]:
        return "Moderate"
    if damage < thresholds["high_max"]:
        return "High"
    return "Extreme"


# ── Lifespan (startup / shutdown) ────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model artifacts into memory once at startup.

    This runs before the first request is accepted.  Everything stays
    in memory for the lifetime of the process — no per-request I/O.
    """
    # 1. Load trained LightGBM model
    logger.info("Loading model from %s", MODEL_PATH)
    model = joblib.load(MODEL_PATH)
    _state["model"] = model

    # The model knows which features it was trained on (and in what
    # order).  Using this instead of parameters.yml avoids coupling
    # the API to Kedro's config loader.
    _state["feature_columns"] = model.feature_name_

    # 2. Load tier thresholds (produced by the training pipeline)
    logger.info("Loading tier thresholds from %s", THRESHOLDS_PATH)
    with open(THRESHOLDS_PATH) as f:
        _state["thresholds"] = json.load(f)

    # 3. Load Gold features, keep only the most recent year
    logger.info("Loading Gold features from %s", GOLD_PATH)
    gold = pd.read_parquet(GOLD_PATH)
    feature_year = int(gold["year"].max())
    latest = gold[gold["year"] == feature_year].copy()

    # Index by county_fips for O(1) single-county lookups
    latest = latest.set_index("county_fips")
    _state["features"] = latest
    _state["feature_year"] = feature_year

    logger.info(
        "API ready — %d counties, feature_year=%d, %d features",
        len(latest),
        feature_year,
        len(_state["feature_columns"]),
    )

    yield  # ← application runs here, serving requests

    logger.info("Shutting down API")


# ── App ──────────────────────────────────────────────────────────
app = FastAPI(
    title="AmFam County Risk Model",
    description=(
        "Real-time county-level risk predictions for property underwriting. "
        "Predicts expected annual property damage and assigns a risk tier "
        "(Low / Moderate / High / Extreme) for any US county in our coverage area."
    ),
    version=MODEL_VERSION,
    lifespan=lifespan,
)


# ── Endpoints ────────────────────────────────────────────────────
# IMPORTANT: /predict/batch is defined BEFORE /predict/{county_fips}.
# FastAPI matches routes top-to-bottom.  If the path-parameter route
# came first, the literal string "batch" would be captured as a FIPS
# code and you'd get a 404 ("county 'batch' not found").


@app.get("/health")
async def health_check():
    """Liveness probe for Cloud Run / load balancers."""
    return {"status": "healthy"}


@app.get("/predict/batch")
async def predict_batch(
    tier: Literal["Low", "Moderate", "High", "Extreme"] | None = Query(
        default=None,
        description="Filter results to a single risk tier",
    ),
):
    """Score all counties and return predictions.

    Equivalent to the batch scoring pipeline output, served via API.
    Optionally filter by risk tier: ``/predict/batch?tier=High``
    """
    features_df = _state["features"]
    model = _state["model"]
    thresholds = _state["thresholds"]
    feature_cols = _state["feature_columns"]

    # Vectorized inference — all 3,063 counties in one call (~10 ms)
    X = features_df[feature_cols]
    preds_log = model.predict(X)
    preds_dollars = np.maximum(np.expm1(preds_log), 0.0)

    scored_at = datetime.now(timezone.utc).isoformat()
    predictions = []
    for (fips, row), pred in zip(features_df.iterrows(), preds_dollars):
        pred_tier = _assign_tier(float(pred), thresholds)
        if tier is not None and pred_tier != tier:
            continue
        predictions.append(
            {
                "county_fips": fips,
                "state": row["state"],
                "county_name": row["cz_name"],
                "predicted_damage_dollars": round(float(pred), 2),
                "predicted_risk_tier": pred_tier,
                "model_version": MODEL_VERSION,
                "scored_at": scored_at,
            }
        )

    return {"count": len(predictions), "predictions": predictions}


@app.get("/predict/{county_fips}")
async def predict_county(county_fips: str):
    """Score a single county by its 5-digit FIPS code.

    Example: ``/predict/17031`` for Cook County, IL.
    """
    features_df = _state["features"]

    if county_fips not in features_df.index:
        raise HTTPException(
            status_code=404,
            detail=(
                f"County FIPS '{county_fips}' not found. "
                f"Provide a valid 5-digit FIPS code. "
                f"{len(features_df)} counties available — "
                f"try /model/info for details."
            ),
        )

    row = features_df.loc[county_fips]
    model = _state["model"]
    feature_cols = _state["feature_columns"]

    # Single-row inference: extract features, predict, inverse-transform
    X = row[feature_cols].values.reshape(1, -1)
    pred_log = model.predict(X)[0]
    pred_dollars = max(float(np.expm1(pred_log)), 0.0)

    return {
        "county_fips": county_fips,
        "state": row["state"],
        "county_name": row["cz_name"],
        "predicted_damage_dollars": round(pred_dollars, 2),
        "predicted_risk_tier": _assign_tier(pred_dollars, _state["thresholds"]),
        "model_version": MODEL_VERSION,
        "scored_at": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/model/info")
async def model_info():
    """Return model metadata for transparency and debugging."""
    return {
        "model_version": MODEL_VERSION,
        "feature_count": len(_state["feature_columns"]),
        "feature_columns": list(_state["feature_columns"]),
        "tier_thresholds": _state["thresholds"],
        "total_counties_available": len(_state["features"]),
        "feature_year": _state["feature_year"],
    }

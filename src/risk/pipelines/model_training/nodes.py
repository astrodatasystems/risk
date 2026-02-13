"""Node functions for the model_training pipeline.

Six nodes that take Gold features through to a trained LightGBM damage
regressor with evaluation metrics, feature importance, and a persisted
model artifact.

Lagged design: features from year Y predict total_property_damage_dollars
in year Y+1 for the same county.  This mirrors real deployment — you use
this year's weather history to price next year's policy.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Any

import lightgbm as lgb
import matplotlib
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)

matplotlib.use("Agg")  # non-interactive backend for CI / headless runs

logger = logging.getLogger(__name__)


# ── helper ──────────────────────────────────────────────────────
def _assign_tier(
    damage: pd.Series,
    low_max: float,
    moderate_max: float,
    high_max: float,
) -> pd.Series:
    """Bucket continuous dollar damage into risk tiers.

    Thresholds (from parameters.yml):
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
def create_training_data(storm_events_gold: pd.DataFrame) -> pd.DataFrame:
    """Self-join Gold so features from year Y predict damage in year Y+1.

    Inner join on (county_fips, target_year) ensures we only keep
    county-year pairs that have both a feature row and a target row.
    """
    features = storm_events_gold.copy()
    features["target_year"] = features["year"] + 1

    targets = storm_events_gold[
        ["county_fips", "year", "total_property_damage_dollars"]
    ].rename(
        columns={
            "year": "target_year",
            "total_property_damage_dollars": "target_damage",
        }
    )

    training_data = features.merge(targets, on=["county_fips", "target_year"])

    # Log-transform target: log(1+x) compresses the heavy right skew so the
    # model can distinguish $0 vs $10K vs $1M vs $100M as roughly equal steps.
    training_data["log_target"] = np.log1p(training_data["target_damage"])

    logger.info(
        "Training data created: %s rows, %s columns, "
        "target_year range %d–%d, %d unique counties",
        training_data.shape[0],
        training_data.shape[1],
        training_data["target_year"].min(),
        training_data["target_year"].max(),
        training_data["county_fips"].nunique(),
    )
    return training_data


# ── Node 2 ──────────────────────────────────────────────────────
def split_train_test(
    training_data: pd.DataFrame,
    parameters: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Temporal split — train on earlier years, test on recent years.

    Never random-split time series: that leaks future information
    into training and inflates metrics.
    """
    feature_cols: list[str] = parameters["feature_columns"]
    target_col: str = parameters["target_column"]
    train_end: int = parameters["train_end_target_year"]
    test_start: int = parameters["test_start_target_year"]

    train_mask = training_data["target_year"] <= train_end
    test_mask = training_data["target_year"] >= test_start

    X_train = training_data.loc[train_mask, feature_cols]
    y_train = training_data.loc[train_mask, target_col]
    X_test = training_data.loc[test_mask, feature_cols]
    y_test = training_data.loc[test_mask, target_col]

    logger.info(
        "Train: %d rows (target_year <= %d) | Test: %d rows (target_year >= %d)",
        len(X_train),
        train_end,
        len(X_test),
        test_start,
    )
    return X_train, X_test, y_train, y_test


# ── Node 3 ──────────────────────────────────────────────────────
def train_damage_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    parameters: dict[str, Any],
) -> lgb.LGBMRegressor:
    """Train a LightGBM regressor to predict next-year property damage.

    The model trains in log-space (y = log1p(dollars)).  Predictions are
    inverse-transformed back to dollars in evaluate_model.

    LightGBM handles NaN natively — no imputation needed for
    max_tornado_ef_scale, mean_hail_magnitude, mean_wind_magnitude.
    """
    mlflow.set_tag("experiment_variant", "log_transform_target")

    model = lgb.LGBMRegressor(**parameters)
    model.fit(X_train, y_train)

    # Log training-set metrics in log-space
    train_preds = model.predict(X_train)
    train_rmse_log = float(np.sqrt(mean_squared_error(y_train, train_preds)))
    train_mae_log = float(mean_absolute_error(y_train, train_preds))

    mlflow.log_metric("train_rmse_log", train_rmse_log)
    mlflow.log_metric("train_mae_log", train_mae_log)
    mlflow.log_metric("train_size", len(X_train))
    mlflow.log_param("n_features", X_train.shape[1])
    mlflow.log_param("feature_names", list(X_train.columns))
    mlflow.log_param("target_transform", "log1p")

    logger.info(
        "Model trained (log-space) — train RMSE: %.4f | train MAE: %.4f",
        train_rmse_log,
        train_mae_log,
    )
    return model


# ── Node 4 ──────────────────────────────────────────────────────
def evaluate_model(
    damage_model: lgb.LGBMRegressor,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    parameters: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, float]]:
    """Evaluate regression accuracy and risk-tier classification accuracy.

    The model predicts in log-space.  We evaluate in both spaces:
    - Log-space metrics show how well the model fits the transformed target.
    - Dollar-space metrics (after expm1 inverse) are directly comparable
      to the baseline and meaningful to business stakeholders.

    Returns two dicts:
    - evaluation_metrics: full set of regression + tier metrics
    - tier_thresholds: the thresholds used (persisted for reproducibility)
    """
    preds_log = damage_model.predict(X_test)

    # --- Log-space metrics ---
    rmse_log = float(np.sqrt(mean_squared_error(y_test, preds_log)))
    mae_log = float(mean_absolute_error(y_test, preds_log))
    r2_log = float(r2_score(y_test, preds_log))

    mlflow.log_metric("test_rmse_log", rmse_log)
    mlflow.log_metric("test_mae_log", mae_log)
    mlflow.log_metric("test_r2_log", r2_log)

    logger.info(
        "Log-space metrics — RMSE: %.4f | MAE: %.4f | R²: %.4f",
        rmse_log,
        mae_log,
        r2_log,
    )

    # --- Inverse-transform to dollar space ---
    preds_dollars = np.expm1(preds_log)
    y_test_dollars = np.expm1(y_test)

    # Clamp negative predictions to zero (expm1 of a negative log-pred
    # can produce small negatives; damage can't be negative)
    preds_dollars = np.maximum(preds_dollars, 0.0)

    # --- Dollar-space metrics (comparable to baseline) ---
    rmse = float(np.sqrt(mean_squared_error(y_test_dollars, preds_dollars)))
    mae = float(mean_absolute_error(y_test_dollars, preds_dollars))
    r2 = float(r2_score(y_test_dollars, preds_dollars))
    med_ae = float(median_absolute_error(y_test_dollars, preds_dollars))

    mlflow.log_metric("test_rmse", rmse)
    mlflow.log_metric("test_mae", mae)
    mlflow.log_metric("test_r2", r2)
    mlflow.log_metric("test_median_ae", med_ae)
    mlflow.log_metric("test_size", len(X_test))

    logger.info(
        "Dollar-space metrics — RMSE: $%s | MAE: $%s | R²: %.4f | MedAE: $%s",
        f"{rmse:,.0f}",
        f"{mae:,.0f}",
        r2,
        f"{med_ae:,.0f}",
    )

    # --- Risk-tier classification (always in dollar space) ---
    low_max = parameters["low_max"]
    moderate_max = parameters["moderate_max"]
    high_max = parameters["high_max"]

    actual_tiers = _assign_tier(
        pd.Series(y_test_dollars, index=y_test.index), low_max, moderate_max, high_max
    )
    pred_tiers = _assign_tier(
        pd.Series(preds_dollars, index=y_test.index), low_max, moderate_max, high_max
    )

    tier_labels = ["Low", "Moderate", "High", "Extreme"]
    tier_acc = float(accuracy_score(actual_tiers, pred_tiers))
    cm = confusion_matrix(actual_tiers, pred_tiers, labels=tier_labels)

    mlflow.log_metric("tier_accuracy", tier_acc)
    logger.info("Tier classification accuracy: %.2f%%", tier_acc * 100)

    # --- Confusion matrix plot → MLflow artifact ---
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(tier_labels)))
    ax.set_yticks(range(len(tier_labels)))
    ax.set_xticklabels(tier_labels, rotation=45, ha="right")
    ax.set_yticklabels(tier_labels)
    ax.set_xlabel("Predicted Tier")
    ax.set_ylabel("Actual Tier")
    ax.set_title(f"Risk Tier Confusion Matrix (acc={tier_acc:.1%})")

    # Annotate cells
    for i in range(len(tier_labels)):
        for j in range(len(tier_labels)):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color=color)

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    tmp_dir = Path(tempfile.mkdtemp())
    cm_path = tmp_dir / "confusion_matrix.png"
    fig.savefig(cm_path, dpi=150)
    plt.close(fig)
    mlflow.log_artifact(str(cm_path))

    # --- Assemble results ---
    metrics = {
        "test_rmse": rmse,
        "test_mae": mae,
        "test_r2": r2,
        "test_median_ae": med_ae,
        "test_rmse_log": rmse_log,
        "test_mae_log": mae_log,
        "test_r2_log": r2_log,
        "tier_accuracy": tier_acc,
        "test_size": len(X_test),
        "confusion_matrix": cm.tolist(),
    }

    thresholds = {
        "low_max": low_max,
        "moderate_max": moderate_max,
        "high_max": high_max,
    }

    return metrics, thresholds


# ── Node 5 ──────────────────────────────────────────────────────
def extract_feature_importance(
    damage_model: lgb.LGBMRegressor,
) -> pd.DataFrame:
    """Extract gain-based feature importances and log a bar chart to MLflow."""
    importance = pd.DataFrame(
        {
            "feature": damage_model.feature_name_,
            "importance": damage_model.feature_importances_,
        }
    ).sort_values("importance", ascending=True)

    # Horizontal bar chart
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(importance["feature"], importance["importance"])
    ax.set_xlabel("Importance (split gain)")
    ax.set_title("Feature Importance — Damage Regressor")
    fig.tight_layout()
    tmp_dir = Path(tempfile.mkdtemp())
    fi_path = tmp_dir / "feature_importance.png"
    fig.savefig(fi_path, dpi=150)
    plt.close(fig)
    mlflow.log_artifact(str(fi_path))

    logger.info(
        "Top 5 features: %s",
        importance.tail(5)["feature"].tolist(),
    )
    return importance


# ── Node 6 ──────────────────────────────────────────────────────
def save_model_artifact(
    damage_model: lgb.LGBMRegressor,
) -> lgb.LGBMRegressor:
    """Log the trained model to MLflow and return it for Kedro persistence.

    mlflow.lightgbm.log_model() stores the model in MLflow's artifact
    store with full reproducibility metadata.  Kedro also persists it
    via the catalog (PickleDataset → joblib) for local use.
    """
    mlflow.lightgbm.log_model(damage_model, artifact_path="damage_model")
    logger.info("Model logged to MLflow artifact store")
    return damage_model

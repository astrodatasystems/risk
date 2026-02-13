"""Smoke tests — fast checks that core components load without error.

These run in CI before the Docker build step.  They catch import
errors, broken pipeline definitions, and logic bugs in shared
utilities *before* spending time on a container build.

They do NOT require data files or a trained model — they test code
structure, not model quality.
"""

import pytest


# ── Test 1: All pipeline modules import cleanly ─────────────────
class TestPipelineImports:
    """Verify every pipeline module can be imported and exposes create_pipeline."""

    def test_import_data_processing(self):
        from risk.pipelines.data_processing import create_pipeline

        pipeline = create_pipeline()
        assert len(pipeline.nodes) > 0

    def test_import_feature_engineering(self):
        from risk.pipelines.feature_engineering import create_pipeline

        pipeline = create_pipeline()
        assert len(pipeline.nodes) > 0

    def test_import_model_training(self):
        from risk.pipelines.model_training import create_pipeline

        pipeline = create_pipeline()
        assert len(pipeline.nodes) > 0

    def test_import_scoring(self):
        from risk.pipelines.scoring import create_pipeline

        pipeline = create_pipeline()
        assert len(pipeline.nodes) == 3


# ── Test 2: FastAPI app creates without error ────────────────────
class TestFastAPIApp:
    """Verify the FastAPI app object is importable and well-formed.

    We test the app *object* (routes, metadata), not the running
    server — that would require model files on disk.
    """

    def test_app_is_importable(self):
        from risk.api import app

        assert app is not None
        assert app.title == "AmFam County Risk Model"

    def test_app_has_expected_routes(self):
        from risk.api import app

        route_paths = [route.path for route in app.routes]
        assert "/health" in route_paths
        assert "/predict/{county_fips}" in route_paths
        assert "/predict/batch" in route_paths
        assert "/model/info" in route_paths

    def test_app_version_matches_model_version(self):
        from risk.api import app, MODEL_VERSION

        assert app.version == MODEL_VERSION


# ── Test 3: Tier assignment logic ────────────────────────────────
class TestTierAssignment:
    """Test the _assign_tier helper that maps dollars to risk tiers.

    This is critical business logic — wrong thresholds mean wrong
    pricing.  We test the version in the API module since it's the
    simplest (scalar in, string out).
    """

    @pytest.fixture()
    def thresholds(self):
        """Standard tier thresholds matching parameters.yml."""
        return {
            "low_max": 100_000,
            "moderate_max": 1_000_000,
            "high_max": 10_000_000,
        }

    def test_zero_damage_is_low(self, thresholds):
        from risk.api import _assign_tier

        assert _assign_tier(0.0, thresholds) == "Low"

    def test_boundary_low_moderate(self, thresholds):
        from risk.api import _assign_tier

        # Just below threshold → Low
        assert _assign_tier(99_999.99, thresholds) == "Low"
        # At threshold → Moderate
        assert _assign_tier(100_000.00, thresholds) == "Moderate"

    def test_boundary_moderate_high(self, thresholds):
        from risk.api import _assign_tier

        assert _assign_tier(999_999.99, thresholds) == "Moderate"
        assert _assign_tier(1_000_000.00, thresholds) == "High"

    def test_boundary_high_extreme(self, thresholds):
        from risk.api import _assign_tier

        assert _assign_tier(9_999_999.99, thresholds) == "High"
        assert _assign_tier(10_000_000.00, thresholds) == "Extreme"

    def test_very_large_damage_is_extreme(self, thresholds):
        from risk.api import _assign_tier

        assert _assign_tier(1_000_000_000.0, thresholds) == "Extreme"

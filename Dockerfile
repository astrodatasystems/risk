# =============================================================
# Dockerfile — Kedro pipeline runner for Cloud Run
# =============================================================
# Builds a container that can run any Kedro pipeline:
#   docker run <image>                              → kedro run (all pipelines)
#   docker run <image> kedro run --pipeline scoring → scoring only
#
# Cloud Run Jobs will use this for batch scoring + training.
# =============================================================

FROM python:3.12-slim

# LightGBM requires OpenMP runtime for parallel tree building.
# libgomp1 provides libgomp.so.1 which LightGBM links against.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# --- Layer-cached dependency install ---
# Copy requirements first so Docker caches this layer.
# Code changes won't trigger a slow pip install rebuild.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- Copy project files ---
# .dockerignore excludes raw data, mlruns, .git, caches, etc.
COPY . .

# --- Install the Kedro project package ---
# Kedro discovers pipelines via the installed 'risk' package
# (defined in pyproject.toml, source in src/risk/).
RUN pip install --no-cache-dir .

# Default: run the full pipeline (all four: data_processing,
# feature_engineering, model_training, scoring).
# Override for a specific pipeline:
#   docker run <image> kedro run --pipeline scoring
CMD ["kedro", "run"]

# CLAUDE.md — AmFam Risk Model Project

## About the Developer

Alvaro is a Data Science leader (McKinsey/QuantumBlack alumnus) with 10+ years of ML and business leadership experience. He is NOT a software engineer — his gaps are in containers, CI/CD, cloud infrastructure, and MLOps. He has delegated most technical implementation throughout his career and is now learning these skills hands-on.

He is preparing for a **Principal Data Engineer / AI Architect** interview at **American Family Insurance** (property & casualty insurer, GCP-centric). The hiring manager (Zak Rottier) personally invited him to apply.

## Teaching Style & Preferences

- **Go step by step.** Never dump multiple changes at once. Make one change, explain it, verify it works, then move to the next.
- **Explain the "why" before the "what."** For every file you create or modify, explain why it's needed, what problem it solves, and how it fits into the bigger picture.
- **Discuss tradeoffs.** When there are multiple approaches, briefly present the options and why you're choosing one over the other.
- **Connect to interview preparation.** When relevant, note which JD requirement or interview panel topic a given implementation decision maps to.
- **Be didactical.** You are a coach and mentor, not just a code generator.

## Project Overview

### Problem Statement

American Family insures homes and properties across the Midwest. Severe weather (hail, tornadoes, thunderstorms, flooding) is their #1 driver of property claims. The current system uses static risk zones and historical loss ratios that don't adapt to changing weather patterns.

**The system we're building:** An end-to-end ML system that ingests NOAA Storm Events data, builds predictive risk models (a risk tier classifier and an expected damage regressor), and serves risk scores to the underwriting platform via both batch (BigQuery table) and real-time (REST API) paths.

### Panel-Defined Requirements

- **End users:** Property underwriters evaluating new policy applications
- **Output:** Risk tier (Low/Moderate/High/Extreme) + expected annual property damage estimate ($)
- **Granularity:** County level
- **Serving:** Nightly batch to BigQuery + real-time REST API
- **Accuracy:** ≥75% tier classification accuracy; cannot systematically underestimate risk (false Low on truly Extreme counties is worst outcome)
- **Retraining:** Quarterly, automated pipeline
- **Regulatory:** Every prediction logged with inputs + model version. Model must be explainable. Full audit trail.
- **Extensibility:** Start with NOAA only, but design for future data sources without re-architecting

### Architecture Summary

**Five layers:** Ingestion → Storage (Medallion) → ML Training → Serving → Monitoring

**Data architecture:** Medallion pattern (Bronze → Silver → Gold) in BigQuery

**ML framework:** Kedro (data catalog, modular pipelines, reproducibility)

**Serving:** Kedro pipelines for batch scoring + FastAPI on Cloud Run for real-time

**CI/CD:** GitHub → Cloud Build → Artifact Registry → Cloud Run

**GCP services:** BigQuery, Cloud Storage, Cloud Run (Services + Jobs), Cloud Build, Artifact Registry, Cloud Scheduler, Cloud Monitoring, Cloud Logging

### Project Phases

**Phase 1 — Foundation:** Problem statement, architecture, environment setup
**Phase 2 — Data:** Collection, ingestion, medallion transforms, EDA
**Phase 3 — ML Core:** Model development, experiment tracking, metrics, pipelines
**Phase 4 — Production:** Testing, CI/CD, deployment, monitoring
**Phase 5 — Enterprise Grade:** Security/IAM, governance, scalability, retraining, operational patterns, enterprise integration

### What's Done

**Phase 1 — Foundation (complete)**
- Problem statement, 5-layer architecture, Kedro project scaffold, GCP project configured, local environment working.

**Phase 2 — Data (complete)**
- **Bronze:** 15 NOAA Storm Events detail files (2010–2024), 965,327 raw events stored in `data/01_raw/noaa/`.
- **Silver:** `data_processing` Kedro pipeline — filtered to county events (CZ_TYPE=C), parsed damage strings (K/M/B → dollars), built 5-digit FIPS codes, parsed timestamps. 556,431 rows in `data/02_intermediate/storm_events_silver.parquet`.
- **Gold:** `feature_engineering` Kedro pipeline — county-year aggregation with 24 features covering volume, damage, event type counts, severity, and seasonality. 45,215 rows (3,256 counties × 15 years) in `data/04_feature/storm_events_gold.parquet`.
- **Note:** `n_winter_weather` is all zeros because winter events are zone-level (CZ_TYPE=Z, filtered out by design); column reserved for future data source enrichment.

**Phase 3 — ML Core (complete)**
- **`model_training` Kedro pipeline** — 6-node DAG: create_training_data → split_train_test → train_damage_model → (evaluate_model | extract_feature_importance | save_model_artifact).
- **LightGBM regressor** with `log1p` target transform. Temporal train/test split (2011–2022 train, 2023–2024 test). Tier accuracy 73% (requirement: ≥75%), median absolute error $12K.
- **Model artifacts, feature importance, and confusion matrix** tracked in MLflow. Three Kedro pipelines: `data_processing`, `feature_engineering`, `model_training`.
- **Outputs:** `data/05_model_input/training_data.parquet`, `data/06_models/damage_model.joblib`, `data/06_models/tier_thresholds.json`, `data/08_reporting/evaluation_metrics.json`, `data/08_reporting/feature_importance.parquet`

### Where We Pick Up

**Phase 4 — Production.** Build scoring pipeline (batch and real-time), CI/CD, testing, deployment to Cloud Run.

## Technical Setup

- **Local machine:** Windows, VSCode, Claude Code, conda environment `risk` with Python 3.12
- **No Docker locally** — Cloud Build handles container builds in the cloud
- **GCP project:** `risk-model-project`
- **GCP region:** `us-central1`
- **GCP authentication:** gcloud CLI installed locally and authenticated via Application Default Credentials. Local Python can connect to BigQuery and GCS in project risk-model-project.
- **Framework:** Kedro (blank project, no starter template)
- **Project name:** `risk`
- **Kedro package name:** `risk`
- **Tools selected during kedro new:** Lint (Ruff), Test (pytest), Log, Docs, Data Folder (no PySpark)
- **Key libraries:** Kedro 1.2.0, kedro-mlflow 2.0.1, LightGBM 4.6.0, MLflow, scikit-learn, matplotlib
- **Known issue:** kedro-mlflow 2.0.1 is incompatible with Kedro 1.2.0 (`pipeline_name` → `pipeline_names` rename). Patched locally in `mlflow_hook.py`. Patch will be lost on `pip install --upgrade kedro-mlflow` — check if newer version fixes it before upgrading.

## Kedro Project Structure

```
risk/
├── conf/
│   ├── base/
│   │   ├── catalog.yml          ← Data Catalog: all dataset declarations
│   │   ├── parameters.yml       ← Model hyperparameters, thresholds, config
│   │   └── mlflow.yml           ← MLflow tracking config (experiment name, URI)
│   └── local/                   ← Environment-specific overrides (not committed)
├── data/                        ← Local data folder (for development)
│   ├── 01_raw/                  ← Maps to Bronze
│   ├── 02_intermediate/         ← Maps to Silver
│   ├── 03_primary/              ← Maps to Gold
│   ├── 04_feature/
│   ├── 05_model_input/
│   ├── 06_models/
│   ├── 07_model_output/
│   └── 08_reporting/
├── docs/
├── notebooks/                   ← For EDA and exploration
├── src/
│   └── risk/
│       ├── pipelines/           ← Pipeline modules:
│       │   ├── data_processing/     ← Bronze → Silver (cleaning, parsing)
│       │   ├── feature_engineering/ ← Silver → Gold (county-year aggregation)
│       │   ├── model_training/      ← Gold → trained model + metrics + artifacts
│       │   └── scoring/             ← (Phase 4) Batch + real-time scoring
│       ├── pipeline_registry.py ← Registers all pipelines
│       ├── settings.py          ← Project-level configuration
│       ├── __init__.py
│       └── __main__.py
├── tests/
├── .gitignore
├── pyproject.toml
├── README.md
└── requirements.txt
```

## Data Architecture — Medallion Pattern

**Bronze** (BigQuery `bronze` dataset / local `data/01_raw/`)
- Raw NOAA Storm Events data exactly as received
- Append-only, never modified after loading
- Tagged with ingestion timestamp and source identifier

**Silver** (BigQuery `silver` dataset / local `data/02_intermediate/`)
- Cleaned, validated, conformed data
- Damage values parsed, nulls handled, event types standardized, duplicates removed
- Still event-level granularity (one row per storm event)

**Gold** (BigQuery `gold` dataset / local `data/03_primary/` and beyond)
- Business-ready, aggregated, purpose-built tables
- County-level features: storm frequency by type, damage statistics, trends, seasonality
- Model input table: `gold.county_risk_features`
- Model output table: `gold.county_risk_scores`
- Prediction logs: `gold.prediction_logs`

## GCP Resources (to be created)

- **BigQuery datasets:** `bronze`, `silver`, `gold` in `us-central1`
- **GCS bucket:** `risk-model-bucket-805162305638` (for model artifacts, raw file landing)
- **Artifact Registry:** `risk-model` in `us-central1` (Docker images)
- **Cloud Run Services:** Real-time prediction API
- **Cloud Run Jobs:** Ingestion, training, batch scoring
- **Cloud Scheduler:** Triggers for ingestion and batch scoring

## Key Design Decisions

1. **Kedro for pipeline orchestration, FastAPI for real-time serving.** Kedro excels at DAGs of batch transformations with a data catalog. Real-time APIs need to be stateless and fast — different tool for a different pattern.
2. **Medallion over ad-hoc staging.** Gives explicit data quality gates between layers, clear lineage, and IAM boundaries per dataset.
3. **No Docker locally.** Cloud Build handles it. Our workflow: write code → push to GitHub → Cloud Build builds container → deploys to Cloud Run.
4. **One regressor, derived tiers.** A single LightGBM damage regressor predicts dollar damage; risk tiers (Low/Moderate/High/Extreme) are derived from predicted damage via configurable thresholds. Simpler than two separate models and ensures tiers are consistent with the dollar prediction.
5. **Model artifacts in GCS with commit SHA.** Full traceability from prediction → model version → code version → data version.

## Interview Connection — JD Requirements Map

When making implementation decisions, reference these JD requirements:

| JD Requirement | Where We Address It |
|---|---|
| Architectural design of scalable AI systems | Overall 5-layer architecture |
| Real-time API, pub/sub, batch processing | Layer 4: Serving (FastAPI + batch scoring) |
| Diverse data sources strategy | Medallion Bronze layer (one per source) + Kedro catalog |
| AI system architecture, security, infrastructure | IAM per dataset, service accounts, Cloud Run security |
| Model lifecycle governance | Model registry table, versioned artifacts, approval workflow |
| Engineering best practices & reusable patterns | Kedro framework, CI/CD, testing strategy |
| Observability and monitoring | Layer 5: data drift, prediction distribution, system health |
| CI/CD for software and ML models | GitHub → Cloud Build → Artifact Registry → Cloud Run |
| GCP services (Vertex AI, Cloud Run, GKE, BigQuery, GCS, IAM) | Core of our implementation |

## Rules for Claude Code

1. **One step at a time.** Create or modify one file per interaction unless the files are trivially related.
2. **Explain before writing.** Before creating or modifying a file, state what you're about to do and why.
3. **Verify after each step.** After making a change, suggest how to verify it works (a command to run, output to check).
4. **Don't install libraries without explaining why.** Every dependency should have a stated purpose.
5. **Keep code simple and readable.** Alvaro needs to understand and explain every line in an interview.
6. **Use type hints and docstrings.** These serve as documentation and show engineering maturity.
7. **Follow Kedro conventions.** Use the data catalog, parameters.yml, pipeline registry. Don't bypass the framework.
8. **Local-first development.** Build and test with local data files first, then connect to GCP.
9. **Git hygiene.** Suggest meaningful commit messages at natural checkpoints.
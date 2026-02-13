# AmFam Risk Model Project - Task Tracker

## Phase 1 — Foundation ✅
- [x] Problem statement defined
- [x] Architecture designed
- [x] Local environment setup (conda, Kedro project)
- [x] GCP project configured (APIs enabled, ADC authenticated, BigQuery datasets, GCS bucket)
- [x] Initial commit pushed

## Phase 2 — Data ✅
### Data Collection
- [x] Research NOAA Storm Events dataset (bulk CSV download from NCEI)
- [x] Download 15 years of detail files (2010–2024) → `data/01_raw/noaa/`
- [x] 965,327 raw events across 51 source files

### Bronze → Silver (`data_processing` pipeline)
- [x] Create `src/risk/pipelines/data_processing/` structure
- [x] Load and combine raw CSVs with source file tracing
- [x] Select 20 relevant columns from 51
- [x] Filter to county-level events (CZ_TYPE=C), drop zone/marine
- [x] Parse damage strings (K/M/B suffixes → dollar amounts)
- [x] Build 5-digit county FIPS codes (state_fips + cz_fips)
- [x] Parse timestamps, extract year/month/day_of_year
- [x] Register `storm_events_silver` in Kedro catalog
- [x] Output: 556,431 rows → `data/02_intermediate/storm_events_silver.parquet`

### Silver → Gold (`feature_engineering` pipeline)
- [x] Create `src/risk/pipelines/feature_engineering/` structure
- [x] Row-level enrichment: EF scale parsing, conditional magnitude, event type flags, quarter
- [x] County-year aggregation: single groupby with named aggregation (24 features)
- [x] Post-processing: events_per_month, quarterly percentages, cleanup
- [x] Register `storm_events_gold` in Kedro catalog
- [x] Output: 45,215 rows (3,256 counties × 15 years) → `data/04_feature/storm_events_gold.parquet`

### EDA & Data Quality
- [ ] Create exploratory notebooks in `notebooks/`
- [ ] Analyze storm event patterns by geography/time
- [ ] Document feature distributions and data quality findings

## Phase 3 — ML Core Development (CURRENT)
### Model Training Pipeline ✅
- [x] Create `src/risk/pipelines/model_training/` structure (6-node DAG)
- [x] Lagged training design: features year Y → target damage year Y+1
- [x] Temporal train/test split (train: target_year 2011–2022, test: 2023–2024)
- [x] LightGBM damage regressor (500 trees, lr=0.05, max_depth=6, 19 features)
- [x] Log-transform target (`log1p`) to handle heavy right skew — tier accuracy 25.8% → 73.0%
- [x] Risk tier labels derived from damage predictions (Low <$100K, Moderate <$1M, High <$10M, Extreme ≥$10M)
- [x] Add hyperparameter + threshold configuration to `parameters.yml`
- [x] Evaluation: RMSE, MAE, R², MedAE in both log-space and dollar-space; tier accuracy + confusion matrix
- [x] Feature importance extraction (gain-based) with bar chart artifact
- [x] Model artifact persisted as joblib + MLflow artifact store

### Experiment Tracking ✅
- [x] Set up kedro-mlflow (auto-registers via plugin entry points, `mlflow.yml` config)
- [x] Params auto-logged by kedro-mlflow; metrics/artifacts logged in node functions
- [x] MLflow experiment `storm_risk_model` with runs tagged by variant (e.g., `log_transform_target`)
- [x] Baseline vs log-transform runs comparable in MLflow UI

### Remaining Phase 3
- [ ] Improve tier accuracy to ≥75% target (hyperparameter tuning, feature enrichment)
- [ ] Add SHAP explainability (regulatory requirement: explain individual predictions)
- [ ] Design model versioning strategy (GCS + commit SHA)
- [ ] Create model metadata tracking table for registry

## Phase 4 — Production Deployment
### Batch Scoring Pipeline
- [ ] Create `src/risk/pipelines/scoring/` structure
- [ ] Implement batch scoring logic
- [ ] Build scoring pipeline (features → predictions → BigQuery)
- [ ] Deploy as Cloud Run Job
- [ ] Set up Cloud Scheduler for nightly runs

### Real-time API
- [ ] Create FastAPI application structure
- [ ] Implement real-time prediction endpoint
- [ ] Add model loading and caching
- [ ] Add prediction logging (regulatory requirement)
- [ ] Create Dockerfile for API
- [ ] Deploy API to Cloud Run Service

### CI/CD Pipeline
- [ ] Create Cloud Build configuration
- [ ] Configure automated testing
- [ ] Set up container builds → Artifact Registry → Cloud Run
- [ ] Create deployment environments (dev/staging/prod)

### Testing Framework
- [ ] Unit tests for all pipeline modules
- [ ] Integration tests for end-to-end workflows
- [ ] Data validation tests
- [ ] API contract tests
- [ ] Configure pytest in CI/CD

## Phase 5 — Enterprise Grade Features
### Security & IAM
- [ ] Service account strategy with least-privilege roles
- [ ] Dataset-level permissions (bronze/silver/gold)
- [ ] Secrets management
- [ ] API authentication

### Governance & Compliance
- [ ] Prediction audit logging (inputs + model version + outputs)
- [ ] Model lineage tracking
- [ ] Data lineage documentation
- [ ] Regulatory compliance documentation

### Monitoring & Observability
- [ ] Cloud Monitoring dashboards
- [ ] Alerting for pipeline failures
- [ ] Model performance monitoring / drift detection
- [ ] Data quality monitoring

### Scalability & Performance
- [ ] Performance test batch pipelines
- [ ] Load test real-time API
- [ ] Auto-scaling configuration for Cloud Run

### Retraining & MLOps
- [ ] Automated retraining triggers
- [ ] Champion/challenger model deployment
- [ ] Quarterly retraining schedule via Cloud Scheduler

## Current Focus: Phase 3 — ML Core (Remaining)
**Next immediate tasks:**
1. Close the 2% gap to ≥75% tier accuracy (hyperparameter tuning, potential feature enrichment)
2. Add SHAP explainability for individual predictions (regulatory requirement)
3. Design model versioning strategy (GCS path with commit SHA)
4. Git commit Phase 3 checkpoint

**Current model performance (log-transform run):**
- Tier accuracy: 73.0% (target: ≥75%)
- Dollar-space: RMSE=$26.3M, MAE=$2.1M, R²=-0.006, MedAE=$11.9K
- Log-space: R²=0.27
- Top features: mean_wind_magnitude, mean_hail_magnitude, max_single_event_damage, pct_events_q3, events_per_month

**Success Criteria:**
- Risk tier accuracy achieves ≥75%
- Model does not systematically underestimate risk (low false-negative rate for Extreme tier)
- Feature importance / SHAP available for every prediction (explainability)
- Model artifacts versioned and traceable to code + data

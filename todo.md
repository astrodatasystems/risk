# AmFam Risk Model Project - Task Tracker

## Phase 1 — Foundation ✅
- [x] Problem statement defined
- [x] Architecture designed  
- [x] Local environment setup (conda, Kedro project)
- [x] Initial commit pushed
- [ ] **GCP Project Configuration** (CURRENT)
  - [ ] Enable required GCP APIs
  - [ ] Configure authentication/service accounts
  - [ ] Set up BigQuery datasets (bronze, silver, gold)
  - [ ] Create GCS bucket for model artifacts
  - [ ] Create Artifact Registry repository
  - [ ] Verify GCP connectivity from local

## Phase 2 — Data Collection & Ingestion
### Data Understanding
- [ ] Research NOAA Storm Events API/dataset
- [ ] Understand data schema and availability
- [ ] Design data sampling strategy for development
- [ ] Document data dictionary

### Ingestion Pipeline
- [ ] Create `src/risk/pipelines/ingestion/` structure
- [ ] Implement NOAA data fetcher
- [ ] Design Bronze layer schema in BigQuery
- [ ] Create Kedro catalog entries for Bronze datasets
- [ ] Build ingestion pipeline (NOAA → Bronze)
- [ ] Test ingestion with sample data locally
- [ ] Deploy ingestion as Cloud Run Job

### Medallion Transforms
- [ ] Create `src/risk/pipelines/feature_engineering/` structure
- [ ] Design Silver layer transformation logic
- [ ] Implement Bronze → Silver pipeline (cleaning, validation)
- [ ] Design Gold layer aggregation logic
- [ ] Implement Silver → Gold pipeline (county-level features)
- [ ] Create Kedro catalog entries for Silver/Gold datasets
- [ ] Test transformations locally
- [ ] Deploy transforms as Cloud Run Jobs

### EDA & Data Quality
- [ ] Create exploratory notebooks in `notebooks/`
- [ ] Analyze storm event patterns by geography/time
- [ ] Identify data quality issues
- [ ] Design data validation rules
- [ ] Document feature engineering decisions

## Phase 3 — ML Core Development
### Model Development
- [ ] Create `src/risk/pipelines/training/` structure
- [ ] Design feature engineering for model inputs
- [ ] Implement risk tier classifier (Low/Moderate/High/Extreme)
- [ ] Implement damage amount regressor
- [ ] Create model training pipeline
- [ ] Add hyperparameter configuration to `parameters.yml`
- [ ] Test model training locally with sample data

### Model Evaluation
- [ ] Create `src/risk/pipelines/evaluation/` structure
- [ ] Implement classification metrics (accuracy, precision, recall)
- [ ] Implement regression metrics (MAE, RMSE, etc.)
- [ ] Create model explainability reports
- [ ] Design validation framework (train/val/test splits)
- [ ] Create model comparison utilities
- [ ] Build evaluation pipeline

### Model Registry & Artifacts
- [ ] Design model versioning strategy (GCS + BigQuery)
- [ ] Create model metadata tracking
- [ ] Implement model serialization/deserialization
- [ ] Create model approval workflow
- [ ] Test model artifacts storage in GCS

## Phase 4 — Production Deployment
### Batch Scoring Pipeline
- [ ] Create `src/risk/pipelines/scoring/` structure
- [ ] Implement batch scoring logic
- [ ] Create county risk scores output table schema
- [ ] Build scoring pipeline (features → predictions → BigQuery)
- [ ] Test batch scoring locally
- [ ] Deploy as Cloud Run Job
- [ ] Set up Cloud Scheduler for nightly runs

### Real-time API
- [ ] Create FastAPI application structure
- [ ] Implement real-time prediction endpoint
- [ ] Add model loading and caching
- [ ] Implement request/response validation
- [ ] Add prediction logging
- [ ] Create health check endpoints
- [ ] Test API locally
- [ ] Create Dockerfile for API
- [ ] Deploy API to Cloud Run Service

### CI/CD Pipeline
- [ ] Create GitHub Actions workflow
- [ ] Set up Cloud Build configuration
- [ ] Configure automated testing
- [ ] Set up container builds and pushes to Artifact Registry
- [ ] Configure automated deployments to Cloud Run
- [ ] Create deployment environments (dev/staging/prod)
- [ ] Test full CI/CD pipeline

### Testing Framework
- [ ] Create unit tests for all pipeline modules
- [ ] Create integration tests for end-to-end workflows
- [ ] Add data validation tests
- [ ] Create API contract tests
- [ ] Set up test coverage reporting
- [ ] Create test data fixtures
- [ ] Configure pytest in CI/CD

## Phase 5 — Enterprise Grade Features
### Security & IAM
- [ ] Design service account strategy
- [ ] Configure least-privilege IAM roles
- [ ] Set up dataset-level permissions
- [ ] Implement secrets management
- [ ] Add request authentication to API
- [ ] Security scan and vulnerability assessment

### Governance & Compliance
- [ ] Implement prediction audit logging
- [ ] Create model lineage tracking
- [ ] Add data lineage documentation
- [ ] Create model performance monitoring
- [ ] Implement model drift detection
- [ ] Document regulatory compliance measures

### Monitoring & Observability
- [ ] Set up Cloud Monitoring dashboards
- [ ] Create alerting for pipeline failures
- [ ] Implement model performance monitoring
- [ ] Add data quality monitoring
- [ ] Set up log aggregation and analysis
- [ ] Create operational runbooks

### Scalability & Performance
- [ ] Performance test batch pipelines
- [ ] Load test real-time API
- [ ] Optimize query performance in BigQuery
- [ ] Configure auto-scaling for Cloud Run
- [ ] Implement caching strategies
- [ ] Create capacity planning documentation

### Retraining & MLOps
- [ ] Design automated retraining triggers
- [ ] Create model performance monitoring
- [ ] Implement A/B testing framework for models
- [ ] Create champion/challenger model deployment
- [ ] Build automated model validation
- [ ] Set up quarterly retraining schedule

## Current Focus: GCP Project Configuration
**Next immediate tasks:**
1. Enable required GCP APIs (BigQuery, Cloud Storage, Cloud Run, etc.)
2. Set up authentication and service accounts
3. Create BigQuery datasets (bronze, silver, gold)
4. Create GCS bucket for model artifacts
5. Verify connectivity from local environment

**Success Criteria for Current Phase:**
- All GCP resources provisioned and accessible
- Local environment can authenticate and access GCP services
- BigQuery datasets created with proper structure
- GCS bucket created and accessible
- Ready to begin data ingestion development
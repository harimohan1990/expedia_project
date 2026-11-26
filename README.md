# Expedia Travel Advertising Intelligence & Forecasting Platform

An end-to-end, production-oriented Machine Learning, Data Engineering, Causal Inference, and MLOps platform inspired by Expedia Group's Senior Applied Scientist (Machine Learning) expectations.

> **Disclaimer**: Independent portfolio project using 100% synthetic data (`scripts/generate_synthetic_data.py`). No Expedia proprietary data, internal APIs, or confidential systems are used.

---

## Key Capabilities

1. **Realistic Synthetic Travel Advertising Data Lake (`scripts/generate_synthetic_data.py`)**:
   - `travelers.csv` (100,000+ rows)
   - `campaigns.csv` (10,000+ rows)
   - `ad_impressions.csv` (1,000,000+ rows)
   - `ad_inventory.csv` (100,800 rows across 12 countries × 6 placements × 1,400 days)
   - `bidding_data.csv` (500,000+ rows)
   - `targeting_signals.csv` (200,000+ rows)
   - `experiment_events.csv` (120,000 A/B, DiD, PSM, and Uplift observations)

2. **PySpark Batch ETL & Structured Streaming (`spark/`)**:
   - Explicit `StructType` schemas, deduplication, invalid record filtering, referential integrity checks, window functions, partitioned Parquet, S3 data lake mirroring, and PostgreSQL/SQLite persistence.
   - 9 core business aggregations (`daily_impressions`, `monthly_impressions`, `campaign_revenue`, `audience_reach_by_country`, `inventory_by_placement`, `campaign_ctr`, `campaign_cvr`, `avg_bidding_price`, `audience_engagement_by_segment`).
   - Spark Structured Streaming with Kafka ingestion (plus local file-stream fallback), 10-minute watermarks for late data, and 5-minute tumbling window aggregations.

3. **Three Production ML Forecasting Models + Unified MLflow Framework (`models/`, `evaluation/`)**:
   - **Model 1 — Audience Reach Forecasting**: Linear Regression, Random Forest, XGBoost, LightGBM with chronological splits (`MAE`, `RMSE`, `WAPE`, `R²`) and 95% confidence intervals.
   - **Model 2 — Inventory Forecasting (7, 14, 30 days)**: ARIMA, SARIMA, Prophet, XGBoost, LightGBM evaluated on a shared temporal holdout (`MAE`, `RMSE`, `MAPE`, `WAPE`).
   - **Model 3 — Pre-Launch Campaign Performance**: Classification (`LogisticRegression`, `RandomForest`, `XGBoost`, `CatBoost` with `Precision`, `Recall`, `F1`, `ROC-AUC`) + 6-target regression (`impressions`, `clicks`, `ctr`, `conversions`, `conversion_rate`, `campaign_revenue`).

4. **Causal Inference & Experimentation (`causal_inference/`)**:
   - Cluster-robust **A/B Testing** (with SRM check at `traveler_id` unit)
   - **Difference-in-Differences** (pre/post ML rollout)
   - **Propensity Score Matching** + **DoWhy** backdoor estimation + SMD covariate balance diagnostics
   - **Uplift Modeling** via **EconML `CausalForestDML`** identifying *Persuadables* vs *Sleeping Dogs*

5. **FastAPI Model Serving, Monitoring & Streamlit Dashboard (`api/`, `monitoring/`, `dashboard/`)**:
   - 6 versioned REST endpoints (`/api/v1/forecast/audience-reach`, `/forecast/inventory`, `/predict/campaign-performance`, `/targeting/recommendations`, `/models`, `/health`) + `/metrics` for Prometheus + Redis caching + Swagger `/docs`.
   - **Evidently AI** data drift reports, **PSI / KS-test** monitoring, performance degradation alerts (CloudWatch + local ledger), Prometheus + Grafana configs.
   - 8-section interactive **Streamlit + Plotly** executive dashboard.

---

## Quickstart

### 1. Environment Setup
```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Run Full End-to-End Pipeline
```bash
make generate-data   # Generates 1M+ synthetic advertising events
make etl             # Runs PySpark batch ETL & 9 business aggregations
make features        # Builds leak-free ML feature tables
make train           # Trains, evaluates, & registers all 3 ML models in MLflow
make causal          # Runs A/B, DiD, PSM, & EconML Uplift causal analysis
make monitor         # Generates Evidently AI, PSI, & model drift reports
```

### 3. Launch API, Dashboard, and Streaming
```bash
make api             # FastAPI Swagger at http://localhost:8000/docs
make dashboard       # Streamlit Dashboard at http://localhost:8501
.venv/bin/python -m spark.streaming   # Spark Structured Streaming micro-batch
```

### 4. Tests & Code Quality
```bash
make test            # Run pytest unit & integration tests
make lint            # Run ruff and black --check
```

---

## Documentation
- [High-Level Design & Executive Overview](docs/HIGH_LEVEL_DESIGN.md)
- [Architecture, AWS Deployment & Monitoring Guide](docs/ARCHITECTURE.md)
- [Applied Scientist Interview Preparation (275 Q&A)](INTERVIEW_PREPARATION.md)

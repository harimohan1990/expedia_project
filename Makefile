.PHONY: help setup generate-data etl features train evaluate causal monitor api dashboard test lint format pipeline-all

PYTHON ?= .venv/bin/python
PYTEST ?= .venv/bin/pytest
JAVA_HOME ?= /opt/homebrew/opt/openjdk@21/libexec/openjdk.jdk/Contents/Home
DYLD_LIBRARY_PATH := .venv/lib/python3.11/site-packages/sklearn/.dylibs:$(DYLD_LIBRARY_PATH)
MPLCONFIGDIR := /tmp/mpl
export JAVA_HOME DYLD_LIBRARY_PATH MPLCONFIGDIR

help:
	@echo "Expedia Travel Advertising Intelligence & Forecasting Platform"
	@echo "Targets:"
	@echo "  make generate-data  - Generate >=1M synthetic travel advertising records"
	@echo "  make etl            - Run PySpark batch ETL pipeline"
	@echo "  make features       - Generate ML feature tables"
	@echo "  make train          - Train Audience Reach, Inventory, and Campaign models"
	@echo "  make causal         - Run A/B, DiD, PSM, and Uplift causal inference"
	@echo "  make monitor        - Run data & model drift checks (Evidently + Prometheus)"
	@echo "  make api            - Start FastAPI production model serving"
	@echo "  make dashboard      - Launch Streamlit interactive dashboard"
	@echo "  make test           - Run pytest unit & integration tests"
	@echo "  make lint           - Run ruff and black --check"
	@echo "  make pipeline-all   - Run full batch ML pipeline end-to-end"

generate-data:
	$(PYTHON) -m scripts.generate_synthetic_data

etl:
	$(PYTHON) -m spark.run_etl

features:
	$(PYTHON) -m features.build_all_features

train:
	$(PYTHON) -m pipelines.batch_pipeline

causal:
	$(PYTHON) -m causal_inference.run_causal_analysis

monitor:
	$(PYTHON) -m monitoring.run_monitoring

api:
	$(PYTHON) -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

dashboard:
	$(PYTHON) -m streamlit run dashboard/app.py --server.port 8501

test:
	$(PYTEST) tests/

lint:
	.venv/bin/ruff check .
	.venv/bin/black --check .

format:
	.venv/bin/ruff check --fix .
	.venv/bin/black .

pipeline-all: generate-data etl features train causal monitor

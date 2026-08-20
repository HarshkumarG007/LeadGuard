# LeadGuard — Makefile
# All targets run from the repo root.

.PHONY: install test lint format train serve dashboard clean help

PYTHON := python
PIP := pip
UVICORN := uvicorn

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

install:  ## Install package and dev dependencies
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[dev]"
	pre-commit install

test:  ## Run full test suite with coverage
	pytest --cov=src --cov-report=term-missing --cov-fail-under=80

lint:  ## Run ruff and black --check
	ruff check src/ api/ app/ tests/
	black --check --line-length 100 src/ api/ app/ tests/

format:  ## Auto-format with black and ruff
	black --line-length 100 src/ api/ app/ tests/
	ruff check --fix src/ api/ app/ tests/

train:  ## Run full training pipeline on the full dataset
	$(PYTHON) -m leadguard.data.download
	$(PYTHON) -m leadguard.data.clean
	$(PYTHON) -m leadguard.data.features
	$(PYTHON) -m leadguard.models.baseline --config configs/train.yaml
	$(PYTHON) -m leadguard.models.xgboost_model --config configs/train.yaml
	$(PYTHON) -m leadguard.models.uncertainty --config configs/train.yaml
	$(PYTHON) -m leadguard.evaluation.fairness
	$(PYTHON) -m leadguard.models.active_learning
	$(PYTHON) -m leadguard.evaluation.explainability

train-sample:  ## Run training pipeline on sample data only
	$(PYTHON) -m leadguard.data.clean --input data/sample --output data/interim/sample_interim.parquet
	$(PYTHON) -m leadguard.data.features --input data/interim/sample_interim.parquet --output data/processed/features_sample.parquet
	$(PYTHON) -m leadguard.models.baseline --config configs/train.yaml --sample
	$(PYTHON) -m leadguard.models.xgboost_model --config configs/train.yaml --sample
	$(PYTHON) -m leadguard.models.uncertainty --config configs/train.yaml --sample
	$(PYTHON) -m leadguard.evaluation.fairness --sample
	$(PYTHON) -m leadguard.models.active_learning --sample
	$(PYTHON) -m leadguard.evaluation.explainability --sample

serve:  ## Start the FastAPI server
	$(UVICORN) api.main:app --host 0.0.0.0 --port 8000 --reload

dashboard:  ## Start the Streamlit dashboard
	streamlit run app/streamlit_app.py

clean:  ## Remove generated files (keeps raw data and sample)
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .coverage htmlcov/ .pytest_cache/
	rm -f data/interim/*.parquet data/processed/*.parquet
	rm -f models/xgboost/model.json models/xgboost/metrics.json
	rm -f models/xgboost/conformal_*.pkl
	rm -f models/baseline/*.pkl
	@echo "Cleaned generated artifacts. Raw data and sample preserved."

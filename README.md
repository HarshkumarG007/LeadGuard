# LeadGuard

Lead service line prediction, uncertainty quantification, and equitable inspection prioritization.

Built to help municipalities solve the "find the pipes" prioritization problem. With hundreds of thousands of unknown service lines and limited budgets for field inspections, municipalities need a way to schedule inspections that maximizes discovery of lead while ensuring equitable coverage across neighborhoods.

## Architecture & Methodology
- **Problem Statement & Design:** See [`LeadGuard_Architecture.md`](LeadGuard_Architecture.md)
- **Modeling Methodology:** See [`docs/methodology.md`](docs/methodology.md)
- **Data Card:** See [`docs/data_card.md`](docs/data_card.md)

## Headline Metrics (Synthetic Sample)
- **Advanced Model (XGBoost):** PR-AUC: 1.000 (synthetic ceiling)
- **Uncertainty Coverage:** 1.0 (exceeds 90% target)
- **SHAP Latency:** ~0.4ms per prediction
- **FNR Disparity:** 0.0 pp gap across income quartiles (synthetic data)

## Quickstart

### 1. Environment Setup
```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

### 2. Run the Pipeline
The repository comes with a 7,500-row synthetic dataset in `data/sample/`. To run the full end-to-end pipeline:

```bash
# 1. Clean data
python -m leadguard.data.clean --input data/sample --output data/interim/sample_interim.parquet

# 2. Feature Engineering
python -m leadguard.data.features --input data/interim/sample_interim.parquet --output data/processed/features_sample.parquet

# 3. Baselines
python -m leadguard.models.baseline --config configs/train.yaml --features data/processed/features_sample.parquet

# 4. XGBoost Model
python -m leadguard.models.xgboost_model --config configs/train.yaml --features data/processed/features_sample.parquet

# 5. Fairness Reference & Uncertainty
python -m leadguard.evaluation.fairness --sample
python -m leadguard.models.uncertainty --features data/processed/features_sample.parquet --sample

# 6. Active Learning Simulation & Explainability
python -m leadguard.models.active_learning --sample
python -m leadguard.evaluation.explainability --sample
```

### 3. Run the Demo (API + Dashboard)

Start the FastAPI backend:
```bash
make serve
# Available at http://localhost:8000/docs
```

In a new terminal, start the Streamlit dashboard:
```bash
make dashboard
# Available at http://localhost:8501
```

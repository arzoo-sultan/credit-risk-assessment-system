# Credit Risk Assessment System

An end-to-end machine learning system for predicting loan default risk, scoring
applicant risk, and explaining model decisions — built as a capstone project
covering the full ML lifecycle from raw data to a deployed, explainable API
and dashboard.

## Problem Statement
Given an applicant's financial and demographic data, predict:
1. **Loan approval / default probability** — will this applicant default?
2. **Risk score** — a continuous score summarizing creditworthiness.
3. **Explanation** — why did the model make this decision (SHAP/LIME), for
   both internal review and regulatory transparency.

## Dataset
[Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk) —
a multi-table dataset (application data, bureau history, previous applications,
installment payments) covering 300k+ applicants. Chosen for scale and realistic
relational structure.

## Project Structure
```
credit-risk-assessment-system/
├── data/
│   ├── raw/              # Original, immutable data dump (gitignored)
│   └── processed/        # Cleaned/merged data ready for modeling (gitignored)
├── notebooks/            # EDA and experimentation notebooks
├── src/
│   ├── data/              # Loading, cleaning, merging raw tables
│   ├── features/          # Feature engineering pipelines
│   ├── models/             # Training, evaluation, model comparison
│   ├── explainability/     # SHAP/LIME wrappers
│   └── utils/              # Shared helpers (config, logging, metrics)
├── api/                   # FastAPI REST service for inference
├── dashboard/             # Streamlit interactive dashboard
├── models/                # Serialized trained models (gitignored)
├── reports/               # Technical report, figures
├── docs/                  # Architecture diagram, notes
├── tests/                 # Unit tests
├── requirements.txt
├── Dockerfile
└── docker-compose.yml
```

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Running the pieces

**MLflow tracking UI** (view experiment runs):
```bash
mlflow ui --backend-store-uri ./mlruns
```

**API** (once a model is trained and saved to `models/`):
```bash
uvicorn api.main:app --reload
```

**Dashboard**:
```bash
streamlit run dashboard/app.py
```

**Docker (everything together)**:
```bash
docker-compose up --build
```

## Roadmap
- [ ] Download & explore Home Credit Default Risk dataset
- [ ] EDA on application data + auxiliary tables
- [ ] Data preprocessing pipeline (missing values, encoding, merging tables)
- [ ] Advanced feature engineering (aggregates from bureau/previous app tables)
- [ ] Baseline model + comparison of 3+ algorithms (Logistic Regression, XGBoost, LightGBM)
- [ ] Hyperparameter tuning with Optuna
- [ ] Cross-validation + evaluation metrics (AUC, precision/recall, KS statistic)
- [ ] Explainability with SHAP
- [ ] Bias/fairness check across protected-ish features (age, gender if present)
- [ ] MLflow experiment tracking
- [ ] FastAPI inference service
- [ ] Streamlit dashboard
- [ ] Dockerize
- [ ] Architecture diagram
- [ ] Technical report + presentation + demo video

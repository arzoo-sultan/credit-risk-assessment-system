# Credit Risk Assessment System

An end-to-end machine learning system that predicts loan default risk, assigns a
risk band, and produces an explainable, simplified credit score for loan
applicants — built as an individual capstone project (Code Room Hub ML
Internship).

**Author:** Arzoo Sultan

## Overview

Given an applicant's financial, demographic, and credit-history data, this
system predicts:

1. **Default probability** — likelihood the applicant fails to repay
2. **Risk band** — Low / Medium / High, based on where the applicant ranks
   relative to the full applicant population
3. **Credit score** — a simplified 300-850 score (linear transform of default
   probability, inspired by traditional credit scoring but not a reproduction
   of any proprietary formula)
4. **Explanation** — SHAP-based breakdown of exactly which factors drove each
   individual prediction, for transparency and regulatory-style reporting

The project also includes a dedicated **bias detection and mitigation** step:
SHAP analysis revealed the model was using applicant gender as an independent
risk signal beyond what financial behavior explained. Gender was removed from
the final model after confirming this cost negligible predictive performance
(AUC 0.7635 → 0.7604).

## Dataset

[Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk)
(Kaggle) — ~307,500 applicants in `application_train.csv`, enriched with
external credit bureau history (`bureau.csv`), prior applications with Home
Credit (`previous_application.csv`), and installment payment history
(`installments_payments.csv`).

## Results Summary

| Model | Test AUC-ROC |
|---|---|
| Logistic Regression (baseline) | 0.741 |
| Decision Tree (baseline) | 0.719 |
| XGBoost (default params) | 0.762 |
| LightGBM (default params) | 0.761 |
| CatBoost (default params) | 0.761 |
| **CatBoost (Optuna-tuned)** | 0.762 |
| **CatBoost (tuned + fair, gender excluded) — final model** | **0.760** |

At the operating threshold (0.7199), the final model catches ~30% of actual
defaulters at ~30% precision — a deliberate, documented trade-off; see
`reports/` for the full precision-recall analysis and rationale.

## Project Structure

```
credit-risk-assessment-system/
├── data/
│   ├── raw/                  # Original Kaggle CSVs (gitignored)
│   └── processed/            # Cleaned, engineered, split data (gitignored)
├── notebooks/
│   ├── 01_eda.ipynb                  # Single-table + multi-table EDA
│   ├── 02_preprocessing.ipynb        # Train/test split, missing values, encoding, scaling
│   ├── 03_feature_engineering.ipynb  # Bureau/prev-app/installment aggregate features
│   ├── 04_baseline_models.ipynb      # Logistic Regression, Decision Tree
│   ├── 05_advanced_models.ipynb      # XGBoost, LightGBM, CatBoost comparison
│   ├── 07_hyperparameter_tuning.ipynb # Optuna tuning on CatBoost
│   ├── 08_imbalance_handling.ipynb   # Class weighting, threshold optimization
│   ├── 09_explainability.ipynb       # SHAP analysis, bias detection & fix
│   ├── 09_mlflow_tracking.ipynb      # Logged experiment history
│   └── 10_risk_scoring.ipynb         # Risk bands, credit score construction
├── src/                       # Reusable data/feature/model utility modules
├── api/                       # FastAPI inference service
│   ├── main.py
│   └── schemas.py
├── dashboard/
│   └── app.py                 # Streamlit dashboard (4 tabs — see below)
├── models/                    # Trained model artifacts (gitignored)
├── reports/                   # Result tables, SHAP outputs, scoring results
├── docs/
│   └── architecture.md        # Full pipeline architecture
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

Notebooks were developed in Google Colab (mounted to Google Drive for
persistent storage of data/models between sessions) and can be run there
directly, or adapted to run against a local `data/` and `models/` directory
structure matching the layout above.

## Running the System

**MLflow tracking UI** (view logged experiment history):
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

**API**:
```bash
uvicorn api.main:app --reload
```
Interactive docs at `http://127.0.0.1:8000/docs`.

**Dashboard**:
```bash
streamlit run dashboard/app.py
```
Opens at `http://localhost:8501`. Requires the API to be running (dashboard
calls it over HTTP).

**Docker (API + dashboard together)**:
```bash
docker-compose up --build
```

## Dashboard Overview

- **Applicant Risk Lookup** — upload one applicant's feature payload, get
  default probability, risk band, and credit score
- **Portfolio Overview** — aggregate risk distribution and validation that
  risk bands correlate with real default rates
- **Model Performance** — ROC curve, confusion matrix at the operating
  threshold
- **Bias & Fairness** — gender-based outcome comparison and documentation of
  the bias finding and remediation

## Key Design Decisions

- **Train/test split before any transformation** — all preprocessing
  (imputation, encoding, scaling) is fit on training data only, to prevent
  data leakage
- **Quantile-based risk bands, not fixed probability cutoffs** — class
  weighting (used to address the ~8% default rate imbalance) distorts raw
  probability calibration, so risk bands are built on rank rather than
  absolute probability
- **Gender excluded from the final model** — see Bias & Fairness section
  above and `docs/architecture.md` for full reasoning
- **CatBoost selected as champion model** — narrowly outperformed XGBoost and
  LightGBM on 5-fold cross-validated AUC and responded well to tuning

## Known Limitations

- Only 3 of 7 available auxiliary tables were used for feature engineering
  (bureau, previous_application, installments_payments); `bureau_balance`,
  `credit_card_balance`, and `POS_CASH_balance` were out of scope
- Single-model solution (no ensembling/stacking across algorithm families),
  which accounts for the gap between this project's AUC (~0.76) and top
  Kaggle competition solutions (~0.80)
- Removing gender does not guarantee removal of all gender-correlated proxy
  signal in remaining features — a known, documented limitation of
  single-attribute fairness fixes

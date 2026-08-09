# Architecture

## Pipeline Overview

```
┌─────────────────────┐
│   Kaggle Data (CSV)  │
│  application_train    │
│  bureau                │
│  previous_application  │
│  installments_payments │
└──────────┬────────────┘
           │
           ▼
┌─────────────────────────────┐
│  01_eda.ipynb                │
│  Single + multi-table EDA    │
│  → identifies signal in       │
│    bureau/prev-app/installment│
└──────────┬────────────────────┘
           │
           ▼
┌──────────────────────────────────┐
│  02_preprocessing.ipynb           │
│  • Train/test split (80/20,       │
│    stratified) BEFORE any fit     │
│  • Drop columns >50% missing      │
│  • Median/mode imputation         │
│  • One-hot encoding               │
│    (train/test column alignment)  │
│  • StandardScaler                 │
└──────────┬─────────────────────────┘
           │
           ▼
┌────────────────────────────────────┐
│  03_feature_engineering.ipynb        │
│  • Recover unscaled SK_ID_CURR       │
│    + raw dollar amounts              │
│  • Bureau aggregates                 │
│    (count, overdue flag)             │
│  • Previous-application aggregates   │
│    (count, was-refused flag)         │
│  • Installment lateness aggregates   │
│  • "Has history" flags               │
│  • Credit/income, annuity/income     │
│    ratios (on raw $ amounts)         │
└──────────┬──────────────────────────┘
           │
           ▼
┌────────────────────────────────────────┐
│  04/05 Model comparison                  │
│  Logistic Regression, Decision Tree,     │
│  XGBoost, LightGBM, CatBoost             │
│  → CatBoost selected (best 5-fold CV AUC)│
└──────────┬───────────────────────────────┘
           │
           ▼
┌────────────────────────────────────┐
│  07_hyperparameter_tuning.ipynb      │
│  Optuna, 30 trials, 3-fold CV         │
│  → tuned CatBoost                     │
└──────────┬───────────────────────────┘
           │
           ▼
┌────────────────────────────────────────┐
│  08_imbalance_handling.ipynb             │
│  auto_class_weights='Balanced'           │
│  Precision-recall threshold optimization │
│  → recall on defaulters: 0.02 → 0.68     │
└──────────┬────────────────────────────────┘
           │
           ▼
┌────────────────────────────────────────────┐
│  09_explainability.ipynb                     │
│  SHAP TreeExplainer                          │
│  • Global feature importance                 │
│  • Per-applicant waterfall explanations       │
│  • BIAS FOUND: CODE_GENDER_M independently    │
│    contributing risk beyond financial signal  │
│  • Retrained without gender:                  │
│    AUC 0.7635 → 0.7604 (negligible cost)      │
│  → FINAL MODEL: catboost_final_fair.pkl       │
└──────────┬─────────────────────────────────────┘
           │
     ┌─────┴──────┐
     ▼            ▼
┌─────────┐  ┌──────────────────────┐
│ MLflow   │  │ 10_risk_scoring.ipynb │
│ tracking │  │ Quantile-based risk    │
│ (7 runs  │  │ bands (Low/Med/High)   │
│ logged)  │  │ + 300-850 credit score │
└─────────┘  └──────────┬──────────────┘
                         │
                         ▼
           ┌──────────────────────────┐
           │  models/                  │
           │  catboost_final_fair.pkl  │
           │  decision_threshold.json  │
           │  scoring_config.json      │
           └──────────┬────────────────┘
                       │
        ┌──────────────┴──────────────┐
        ▼                              ▼
┌───────────────────┐      ┌───────────────────────┐
│  api/main.py        │      │  dashboard/app.py       │
│  FastAPI              │◄─────│  Streamlit               │
│  POST /predict         │ HTTP │  • Applicant Lookup       │
│  GET /health            │      │  • Portfolio Overview     │
│                          │      │  • Model Performance      │
│                          │      │  • Bias & Fairness        │
└───────────┬──────────────┘      └───────────────────────────┘
            │
            ▼
┌───────────────────────┐
│  Docker / docker-compose │
│  API + dashboard containerized, │
│  sharing models/ volume        │
└─────────────────────────────────┘
```

## Key Architectural Decisions

### Data leakage prevention
The train/test split happens before any preprocessing step. All fitted
transformations (imputers, encoders, scaler) are fit only on training data
and applied (never re-fit) to test data. This discipline is maintained
through every subsequent notebook.

### Identifier and raw-value recovery
`SK_ID_CURR` and the raw dollar amounts (`AMT_CREDIT`, `AMT_INCOME_TOTAL`,
`AMT_ANNUITY`) were inadvertently scaled during preprocessing (numeric-column
selection didn't exclude identifier/pre-ratio columns). Because the
train/test split uses a fixed `random_state`, these values are recoverable
deterministically by re-splitting the raw data — this recovery step runs at
the start of feature engineering and scoring notebooks.

### Model selection
Five algorithms were compared via 5-fold stratified cross-validation.
CatBoost was selected as the champion (CV mean AUC 0.7570, narrowly ahead of
LightGBM 0.7557 and XGBoost 0.7555) for its resistance to overfitting during
tuning and native handling of categorical structure.

### Imbalance handling — two complementary techniques
1. **Class weighting** (`auto_class_weights='Balanced'`) — changes what the
   model is penalized for during training, correcting the model's near-total
   blindness to the minority (default) class.
2. **Threshold optimization** — since weighting distorts probability
   calibration, the actual decision threshold is chosen post-hoc via the
   precision-recall curve, not the default 0.5.

### Bias detection and mitigation
SHAP analysis (per-feature and per-applicant) revealed `CODE_GENDER_M`
contributed an average SHAP value of +0.227 for male applicants and -0.145
for female applicants — independent of every other feature, including actual
financial behavior. This exceeds what the real-world default-rate gap
(10.17% male vs. 6.99% female) would justify, and constitutes a regulatory
risk under frameworks like ECOA. The model was retrained excluding gender
columns; the AUC cost was 0.0031 (0.7635 → 0.7604), confirmed via direct
comparison to be within normal cross-validation noise — i.e., gender was not
meaningfully necessary for predictive performance.

### Risk scoring calibration
Fixed-probability risk-band cutoffs produced a skewed, unusable distribution
(65% of applicants in "High Risk") due to the probability calibration shift
introduced by class weighting. Risk bands were rebuilt using population
quantiles (50th/85th percentile) instead, producing a realistic 50/35/15
Low/Medium/High split with real default rates that increase monotonically
across bands (2.8% → 8.9% → 23.8%), validating the bands are meaningful.

## Tech Stack

| Layer | Technology |
|---|---|
| Data processing | pandas, numpy, scikit-learn |
| Modeling | CatBoost, XGBoost, LightGBM |
| Hyperparameter tuning | Optuna |
| Explainability | SHAP |
| Experiment tracking | MLflow (SQLite backend) |
| API | FastAPI, Pydantic, uvicorn |
| Dashboard | Streamlit, Plotly |
| Deployment | Docker, docker-compose |
| Development environment | Google Colab (GPU runtime) + VS Code |
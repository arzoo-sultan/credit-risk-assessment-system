# Notebooks

Numbered by workflow stage — keep them in order so anyone (including you,
three weeks from now) can follow the project's progression:

- `01_eda.ipynb` — exploratory data analysis on application data + auxiliary tables
- `02_feature_engineering.ipynb` — building the merged, engineered feature set
- `03_model_comparison.ipynb` — baseline + comparison of 3+ algorithms
- `04_hyperparameter_tuning.ipynb` — Optuna tuning on the best candidate(s)
- `05_explainability_fairness.ipynb` — SHAP explanations + bias/fairness checks

Keep heavy exploration here; once logic is finalized, move reusable code into
`src/` so the API and dashboard can import it directly instead of duplicating
notebook code.

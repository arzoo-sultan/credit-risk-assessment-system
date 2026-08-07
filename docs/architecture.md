# Architecture Diagram — TODO

Once the pipeline is built, document the end-to-end flow here (and produce a
visual diagram for the final submission). Rough shape to expect:

```
[Kaggle CSVs] --> [src/data: load & merge] --> [src/features: engineering]
      --> [src/models: train + compare + tune] --> [models/model.pkl]
      --> [MLflow tracking] (parallel, at every training run)

[models/model.pkl] --> [api/main.py (FastAPI)] --> [dashboard/app.py (Streamlit)]
                                  |
                                  v
                          [src/explainability: SHAP] --> explanations surfaced
                                  in both API responses and dashboard
```

Deployment: both API and dashboard containerized via Docker, orchestrated
with docker-compose for local/demo use.

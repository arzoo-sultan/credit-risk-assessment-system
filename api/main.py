"""
FastAPI inference service for the Credit Risk Assessment System.

Run locally with:
    uvicorn api.main:app --reload

Once a trained model exists at models/model.pkl, /predict will load it and
return a default probability + risk score + top SHAP contributors for the
given applicant.
"""
from fastapi import FastAPI, HTTPException
from api.schemas import ApplicantFeatures, PredictionResponse

app = FastAPI(
    title="Credit Risk Assessment API",
    description="Predicts default risk and returns an explainable risk score.",
    version="0.1.0",
)


@app.get("/health")
def health_check():
    """Simple liveness check."""
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(applicant: ApplicantFeatures):
    """
    Predict default probability and risk score for a single applicant.

    TODO once model is trained:
    1. Load the serialized model + preprocessing pipeline from models/
    2. Transform `applicant` into the model's expected feature vector
    3. Run model.predict_proba()
    4. Run SHAP explainer on the single instance
    5. Return probability, risk_score, and top contributing features
    """
    raise HTTPException(
        status_code=501,
        detail="Model not yet trained/loaded — this endpoint is a placeholder.",
    )

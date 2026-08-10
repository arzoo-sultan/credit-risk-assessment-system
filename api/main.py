from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict
import joblib
import json
import pandas as pd

app = FastAPI(
    title="Credit Risk Assessment API",
    description="Predicts default risk and returns a credit score for a loan applicant.",
    version="1.0.0",
)

model = joblib.load("models/catboost_final_fair.pkl")

with open("models/decision_threshold_final.json") as f:
    config = json.load(f)


class ApplicantFeatures(BaseModel):
    features: Dict[str, float]


def probability_to_score(prob, min_score=300, max_score=850):
    score = max_score - (prob * (max_score - min_score))
    return int(max(min_score, min(max_score, score)))


def assign_risk_band(prob, low_cutoff=0.3825, high_cutoff=0.6402):
    if prob < low_cutoff:
        return "Low"
    elif prob < high_cutoff:
        return "Medium"
    return "High"


@app.get("/")
def msg():
    return {"message":"Server Started"}
@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/predict")
def predict(applicant: ApplicantFeatures):
    try:
        X = pd.DataFrame([applicant.features])
        expected_cols = model.feature_names_
        X = X.reindex(columns=expected_cols, fill_value=0)

        prob = model.predict_proba(X)[:, 1][0]
        score = probability_to_score(prob)
        band = assign_risk_band(prob)

        return {
            "default_probability": round(float(prob), 4),
            "risk_band": band,
            "credit_score": score,
            "decision_threshold": config["optimal_threshold"],
            "flagged_high_risk": bool(prob >= config["optimal_threshold"]),
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
"""
Request/response schemas for the Credit Risk API.

NOTE: `ApplicantFeatures` below is a placeholder with a few illustrative
fields from the Home Credit Default Risk dataset. Once EDA and feature
engineering are done, replace this with the actual final feature set the
trained model expects.
"""
from pydantic import BaseModel, Field
from typing import Optional


class ApplicantFeatures(BaseModel):
    income_total: float = Field(..., description="Applicant's total income")
    credit_amount: float = Field(..., description="Requested credit amount")
    annuity_amount: Optional[float] = Field(None, description="Loan annuity amount")
    age_years: int = Field(..., description="Applicant age in years")
    employment_years: Optional[float] = Field(None, description="Years employed")
    family_status: Optional[str] = None
    education_type: Optional[str] = None
    # TODO: extend with final engineered features (bureau aggregates,
    # previous-application aggregates, etc.) once the feature pipeline exists.


class PredictionResponse(BaseModel):
    default_probability: float
    risk_score: float
    risk_category: str  # e.g. "Low" / "Medium" / "High"
    top_factors: list[str] = []  # top SHAP-driven contributors

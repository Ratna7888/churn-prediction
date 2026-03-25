"""Request/response schemas for the churn prediction API."""

from pydantic import BaseModel, Field
from typing import Optional


class CustomerRequest(BaseModel):
    """
    Input schema for a single customer churn prediction.

    Mirrors the raw Telco dataset columns (minus customerID and Churn).
    The API consumer sends raw features — preprocessing happens server-side.
    """
    gender: str = Field(..., description="Male or Female")
    SeniorCitizen: int = Field(..., description="1 if senior, 0 otherwise")
    Partner: str = Field(..., description="Yes or No")
    Dependents: str = Field(..., description="Yes or No")
    tenure: int = Field(..., description="Months with the company")
    PhoneService: str = Field(..., description="Yes or No")
    MultipleLines: str = Field(..., description="Yes / No / No phone service")
    InternetService: str = Field(..., description="DSL / Fiber optic / No")
    OnlineSecurity: str = Field(..., description="Yes / No / No internet service")
    OnlineBackup: str = Field(..., description="Yes / No / No internet service")
    DeviceProtection: str = Field(..., description="Yes / No / No internet service")
    TechSupport: str = Field(..., description="Yes / No / No internet service")
    StreamingTV: str = Field(..., description="Yes / No / No internet service")
    StreamingMovies: str = Field(..., description="Yes / No / No internet service")
    Contract: str = Field(..., description="Month-to-month / One year / Two year")
    PaperlessBilling: str = Field(..., description="Yes or No")
    PaymentMethod: str = Field(..., description="Payment method string")
    MonthlyCharges: float = Field(..., description="Monthly charge amount")
    TotalCharges: float = Field(..., description="Total charges to date")

    model_config = {"json_schema_extra": {
        "examples": [{
            "gender": "Female",
            "SeniorCitizen": 0,
            "Partner": "Yes",
            "Dependents": "No",
            "tenure": 1,
            "PhoneService": "No",
            "MultipleLines": "No phone service",
            "InternetService": "DSL",
            "OnlineSecurity": "No",
            "OnlineBackup": "Yes",
            "DeviceProtection": "No",
            "TechSupport": "No",
            "StreamingTV": "No",
            "StreamingMovies": "No",
            "Contract": "Month-to-month",
            "PaperlessBilling": "Yes",
            "PaymentMethod": "Electronic check",
            "MonthlyCharges": 29.85,
            "TotalCharges": 29.85,
        }]
    }}


class PredictionResponse(BaseModel):
    """Output with prediction, probability, risk level, and top risk factors."""
    will_churn: bool = Field(..., description="Whether customer is predicted to churn")
    churn_probability: float = Field(..., description="Model confidence (0-1)")
    threshold: float = Field(..., description="Decision threshold used")
    risk_level: str = Field(..., description="low / medium / high / critical")
    top_risk_factors: Optional[list[str]] = Field(
        None, description="Key factors contributing to churn risk"
    )


class HealthResponse(BaseModel):
    status: str = "healthy"
    model_loaded: bool = True
    version: str = "1.0.0"
from __future__ import annotations

from pydantic import BaseModel


class PredictRequest(BaseModel):
    customer_ids: list[int]


class Prediction(BaseModel):
    customer_id: int
    churn_proba: float
    churn_label: int


class PredictResponse(BaseModel):
    predictions: list[Prediction]


class ExplainRequest(BaseModel):
    customer_id: int


class ExplainResponse(BaseModel):
    customer_id: int
    expected_value: float
    shap_values: dict[str, float]

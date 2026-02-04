from __future__ import annotations

import json
from typing import Any

import joblib
import numpy as np
import pandas as pd
import shap
from fastapi import FastAPI, HTTPException
from prometheus_client import Counter, Histogram, make_asgi_app

from src.config import settings
from src.serve.schemas import ExplainRequest, ExplainResponse, PredictRequest, PredictResponse
from src.utils.db import read_df, upsert_predictions

app = FastAPI(title="UBIP Churn API", version="0.1.0")

REQUEST_LATENCY = Histogram("ubip_request_latency_seconds", "Request latency", ["endpoint"])
PREDICTION_COUNTER = Counter("ubip_predictions_total", "Total predictions served")

metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

MODEL: Any | None = None
FEATURES: list[str] = []
EXPLAINER: shap.TreeExplainer | None = None


def load_artifacts() -> None:
    global MODEL, FEATURES, EXPLAINER
    if not settings.model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {settings.model_path}. Train the model first."
        )

    with settings.feature_spec_path.open("r", encoding="utf-8") as f:
        spec = json.load(f)

    FEATURES = spec["features"]
    MODEL = joblib.load(settings.model_path)
    EXPLAINER = shap.TreeExplainer(MODEL)


@app.on_event("startup")
async def startup_event() -> None:
    load_artifacts()


def fetch_features(customer_ids: list[int]) -> pd.DataFrame:
    if not customer_ids:
        return pd.DataFrame()

    ids = ",".join(str(int(cid)) for cid in customer_ids)
    sql = f"SELECT * FROM features WHERE customer_id IN ({ids})"
    df = read_df(sql)
    return df


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    with REQUEST_LATENCY.labels(endpoint="predict").time():
        df = fetch_features(request.customer_ids)
        if df.empty:
            raise HTTPException(status_code=404, detail="No features found for customers")

        missing = set(request.customer_ids) - set(df["customer_id"].tolist())
        if missing:
            raise HTTPException(
                status_code=404,
                detail=f"Missing features for customer IDs: {sorted(missing)[:10]}",
            )

        X = df[FEATURES]
        proba = MODEL.predict_proba(X)[:, 1]
        preds = (proba >= settings.prediction_threshold).astype(int)

        rows = []
        predictions = []
        for cid, p, label in zip(df["customer_id"].tolist(), proba, preds):
            predictions.append(
                {"customer_id": int(cid), "churn_proba": float(p), "churn_label": int(label)}
            )
            rows.append(
                {
                    "customer_id": int(cid),
                    "churn_proba": float(p),
                    "churn_label": int(label),
                    "model_version": "latest",
                }
            )

        upsert_predictions(rows)
        PREDICTION_COUNTER.inc(len(predictions))

        return PredictResponse(predictions=predictions)


@app.post("/explain", response_model=ExplainResponse)
def explain(request: ExplainRequest) -> ExplainResponse:
    with REQUEST_LATENCY.labels(endpoint="explain").time():
        df = fetch_features([request.customer_id])
        if df.empty:
            raise HTTPException(status_code=404, detail="No features found for customer")

        row = df.iloc[0][FEATURES]
        raw_shap = EXPLAINER.shap_values(row.to_frame().T)
        if isinstance(raw_shap, list):
            shap_values = raw_shap[-1][0]
        else:
            shap_values = raw_shap[0]

        payload = {
            feature: float(value) for feature, value in zip(FEATURES, shap_values)
        }

        return ExplainResponse(
            customer_id=int(request.customer_id),
            expected_value=float(
                EXPLAINER.expected_value[-1]
                if isinstance(EXPLAINER.expected_value, (list, np.ndarray))
                else EXPLAINER.expected_value
            ),
            shap_values=payload,
        )

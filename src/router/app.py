"""
A/B Testing Router API
========================
Routes prediction traffic between Control (A) and Challenger (B) models.
Tracks conversions and runs significance tests.

Endpoints:
  GET  /health
  GET  /experiment-status   — Current experiment metrics
  POST /predict             — Auto-routed prediction (A or B)
  POST /predict/control     — Force control model
  POST /predict/challenger  — Force challenger model
  POST /analyze             — Run significance test on collected data
  POST /reset               — Reset experiment counters
"""

import os
import json
import logging
import random
import numpy as np
import pandas as pd
import xgboost as xgb
import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional
import uvicorn

from src.stats.significance import decide_winner, z_test_proportions

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ARTIFACTS_DIR = os.getenv("ARTIFACTS_DIR", "artifacts")


def load_artifacts():
    control = joblib.load(os.path.join(ARTIFACTS_DIR, "control_model.pkl"))
    challenger = xgb.XGBClassifier()
    challenger.load_model(os.path.join(ARTIFACTS_DIR, "challenger_model.json"))
    feature_order = json.load(open(os.path.join(ARTIFACTS_DIR, "feature_order.json")))
    encoders      = json.load(open(os.path.join(ARTIFACTS_DIR, "encoders.json")))
    meta          = json.load(open(os.path.join(ARTIFACTS_DIR, "model_meta.json")))
    logger.info(f"Models loaded | Control AUC: {meta['control']['auc']} | Challenger AUC: {meta['challenger']['auc']}")
    return control, challenger, feature_order, encoders, meta


control_model, challenger_model, feature_order, encoders, model_meta = load_artifacts()

# Experiment state — in-memory counters
experiment = {
    "n_a": 0, "conv_a": 0,
    "n_b": 0, "conv_b": 0,
    "traffic_split": 0.5,
    "status": "running",
}

app = FastAPI(
    title="A/B Testing Router API",
    description="Routes traffic between Control (LogisticRegression) and Challenger (XGBoost) models.",
    version="1.0.0",
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class CustomerFeatures(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    age:             int   = Field(..., ge=18, le=95)
    job:             str
    marital:         str
    education:       str
    default:         str
    housing:         str
    loan:            str
    contact:         str
    month:           str
    day_of_week:     str
    campaign:        int
    pdays:           int
    previous:        int
    poutcome:        str
    emp_var_rate:    float = Field(..., alias="emp.var.rate")
    cons_price_idx:  float = Field(..., alias="cons.price.idx")
    cons_conf_idx:   float = Field(..., alias="cons.conf.idx")
    euribor3m:       float
    nr_employed:     float = Field(..., alias="nr.employed")


class PredictionResponse(BaseModel):
    prediction:    int
    probability:   float
    model_variant: str
    label:         str


def preprocess(customer: CustomerFeatures) -> pd.DataFrame:
    data = customer.model_dump(by_alias=True)
    row = {}
    for col in feature_order:
        val = data.get(col, 0)
        if isinstance(val, str):
            row[col] = int(encoders.get(col, {}).get(val, -1))
        else:
            row[col] = float(val) if val is not None else 0.0
    df = pd.DataFrame([row])
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    return df[feature_order]


def predict_with_model(model, df: pd.DataFrame, variant: str) -> PredictionResponse:
    prob = float(model.predict_proba(df)[0][1])
    pred = int(prob >= 0.5)
    label = "Yes — will subscribe" if pred == 1 else "No — will not subscribe"
    return PredictionResponse(
        prediction=pred, probability=round(prob, 4),
        model_variant=variant, label=label
    )


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "control_auc":    model_meta["control"]["auc"],
        "challenger_auc": model_meta["challenger"]["auc"],
        "experiment":     experiment["status"],
    }


@app.get("/experiment-status")
def experiment_status():
    n_a, n_b = experiment["n_a"], experiment["n_b"]
    conv_a, conv_b = experiment["conv_a"], experiment["conv_b"]
    return {
        "status":       experiment["status"],
        "traffic_split": experiment["traffic_split"],
        "control":    {"n": n_a, "conversions": conv_a, "rate": round(conv_a/n_a, 4) if n_a > 0 else 0},
        "challenger": {"n": n_b, "conversions": conv_b, "rate": round(conv_b/n_b, 4) if n_b > 0 else 0},
        "total_requests": n_a + n_b,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerFeatures):
    """Auto-route to control or challenger based on traffic split."""
    df = preprocess(customer)
    if random.random() < experiment["traffic_split"]:
        result = predict_with_model(challenger_model, df, "challenger_B")
        experiment["n_b"] += 1
        if result.prediction == 1:
            experiment["conv_b"] += 1
    else:
        result = predict_with_model(control_model, df, "control_A")
        experiment["n_a"] += 1
        if result.prediction == 1:
            experiment["conv_a"] += 1
    return result


@app.post("/predict/control", response_model=PredictionResponse)
def predict_control(customer: CustomerFeatures):
    df = preprocess(customer)
    return predict_with_model(control_model, df, "control_A")


@app.post("/predict/challenger", response_model=PredictionResponse)
def predict_challenger(customer: CustomerFeatures):
    df = preprocess(customer)
    return predict_with_model(challenger_model, df, "challenger_B")


@app.post("/analyze")
def analyze():
    """Run statistical significance test on collected experiment data."""
    n_a, n_b = experiment["n_a"], experiment["n_b"]
    if n_a < 30 or n_b < 30:
        raise HTTPException(status_code=400, detail=f"Need at least 30 samples per variant. Got A={n_a}, B={n_b}")

    result = decide_winner(
        n_a=n_a, conv_a=experiment["conv_a"],
        n_b=n_b, conv_b=experiment["conv_b"],
        auc_a=model_meta["control"]["auc"],
        auc_b=model_meta["challenger"]["auc"],
    )
    return result


@app.post("/reset")
def reset():
    """Reset experiment counters."""
    experiment["n_a"] = experiment["conv_a"] = 0
    experiment["n_b"] = experiment["conv_b"] = 0
    experiment["status"] = "running"
    return {"message": "Experiment reset", "status": "running"}


if __name__ == "__main__":
    uvicorn.run("src.router.app:app", host="0.0.0.0", port=8002, reload=True)

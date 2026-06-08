"""
Test suite for mlops-ab-testing.
Run: pytest tests/ -v --cov=src
"""

import pytest
import sys, os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Train models if artifacts don't exist
if not os.path.exists("artifacts/control_model.pkl"):
    from src.models.train import train_models
    train_models()

from fastapi.testclient import TestClient
from src.router.app import app
from src.stats.significance import z_test_proportions, cohens_h, decide_winner

client = TestClient(app)

SAMPLE = {
    "age": 42, "job": "admin.", "marital": "married",
    "education": "university.degree", "default": "no",
    "housing": "yes", "loan": "no", "contact": "cellular",
    "month": "may", "day_of_week": "mon", "campaign": 2,
    "pdays": 999, "previous": 0, "poutcome": "nonexistent",
    "emp.var.rate": 1.1, "cons.price.idx": 93.994,
    "cons.conf.idx": -36.4, "euribor3m": 4.857, "nr.employed": 5191.0
}


class TestStats:
    def test_z_test_same(self):
        r = z_test_proportions(1000, 100, 1000, 100)
        assert r["p_value"] > 0.05

    def test_z_test_different(self):
        r = z_test_proportions(1000, 100, 1000, 200)
        assert r["p_value"] < 0.05

    def test_z_test_lift(self):
        r = z_test_proportions(1000, 100, 1000, 150)
        assert r["lift"] > 0

    def test_cohens_h(self):
        h = cohens_h(0.10, 0.15)
        assert h > 0

    def test_decide_winner_promote(self):
        result = decide_winner(1000, 100, 1000, 200, auc_a=0.80, auc_b=0.83)
        assert result["decision"] == "promote_challenger"

    def test_decide_winner_no_significance(self):
        result = decide_winner(100, 10, 100, 11, auc_a=0.80, auc_b=0.83)
        assert result["decision"] == "no_winner"


class TestAPI:
    def test_health(self):
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "healthy"

    def test_experiment_status(self):
        r = client.get("/experiment-status")
        assert r.status_code == 200
        data = r.json()
        assert "control" in data
        assert "challenger" in data

    def test_predict_returns_200(self):
        r = client.post("/predict", json=SAMPLE)
        assert r.status_code == 200

    def test_predict_variant_assigned(self):
        r = client.post("/predict", json=SAMPLE)
        data = r.json()
        assert data["model_variant"] in ["control_A", "challenger_B"]

    def test_predict_control(self):
        r = client.post("/predict/control", json=SAMPLE)
        assert r.status_code == 200
        assert r.json()["model_variant"] == "control_A"

    def test_predict_challenger(self):
        r = client.post("/predict/challenger", json=SAMPLE)
        assert r.status_code == 200
        assert r.json()["model_variant"] == "challenger_B"

    def test_predict_probability_range(self):
        r = client.post("/predict", json=SAMPLE)
        assert 0.0 <= r.json()["probability"] <= 1.0

    def test_reset(self):
        r = client.post("/reset")
        assert r.status_code == 200
        assert r.json()["status"] == "running"

    def test_analyze_needs_samples(self):
        client.post("/reset")
        r = client.post("/analyze")
        assert r.status_code == 400

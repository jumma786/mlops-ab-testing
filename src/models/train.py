"""
Model Trainer for A/B Testing
================================
Trains two models:
  Control (A):    LogisticRegression — current production model
  Challenger (B): XGBoost — candidate for promotion

Both trained on UCI Bank Marketing data.
Saved as artifacts for the A/B router.
"""

import os
import json
import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score
import xgboost as xgb
import joblib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ARTIFACTS_DIR = "artifacts"


def load_and_preprocess(data_path=None, n_samples=5000, random_state=42):
    if data_path and os.path.exists(data_path):
        logger.info(f"Loading real data: {data_path}")
        df = pd.read_csv(data_path, sep=";")
        df["y"] = (df["y"] == "yes").astype(int)
    else:
        logger.info("Generating synthetic data")
        np.random.seed(random_state)
        n = n_samples
        df = pd.DataFrame({
            "age":           np.random.randint(18, 95, n),
            "job":           np.random.choice(["admin.","blue-collar","management","retired","technician"], n),
            "marital":       np.random.choice(["divorced","married","single"], n),
            "education":     np.random.choice(["basic.4y","high.school","university.degree"], n),
            "default":       np.random.choice(["no","unknown"], n),
            "housing":       np.random.choice(["no","yes"], n),
            "loan":          np.random.choice(["no","yes"], n),
            "contact":       np.random.choice(["cellular","telephone"], n),
            "month":         np.random.choice(["mar","apr","may","jun","jul","aug","sep","oct"], n),
            "day_of_week":   np.random.choice(["mon","tue","wed","thu","fri"], n),
            "campaign":      np.random.randint(1, 10, n),
            "pdays":         np.where(np.random.rand(n) < 0.13, np.random.randint(1,30,n), 999),
            "previous":      np.random.randint(0, 5, n),
            "poutcome":      np.random.choice(["failure","nonexistent","success"], n),
            "emp.var.rate":  np.random.choice([-1.8, 1.1, 1.4], n),
            "cons.price.idx": np.random.uniform(92.2, 94.8, n).round(3),
            "cons.conf.idx": np.random.uniform(-50.8, -26.9, n).round(1),
            "euribor3m":     np.random.uniform(0.6, 5.1, n).round(3),
            "nr.employed":   np.random.choice([4963.6, 5099.1, 5228.1], n),
            "y":             (np.random.rand(n) < 0.11).astype(int),
        })

    if "duration" in df.columns:
        df = df.drop(columns=["duration"])

    cat_cols = df.select_dtypes(include="object").columns.tolist()
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = {v: int(i) for i, v in enumerate(le.classes_)}

    X = df.drop(columns=["y"])
    y = df["y"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test, list(X.columns), encoders


def train_models(data_path=None, random_state=42):
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    X_train, X_test, y_train, y_test, feature_names, encoders = load_and_preprocess(
        data_path=data_path, random_state=random_state
    )

    # Control — Logistic Regression
    logger.info("Training Control model (Logistic Regression)...")
    control = LogisticRegression(max_iter=1000, random_state=random_state, class_weight="balanced")
    control.fit(X_train, y_train)
    control_auc = roc_auc_score(y_test, control.predict_proba(X_test)[:, 1])
    control_f1  = f1_score(y_test, control.predict(X_test), zero_division=0)
    joblib.dump(control, os.path.join(ARTIFACTS_DIR, "control_model.pkl"))
    logger.info(f"Control AUC: {control_auc:.4f} | F1: {control_f1:.4f}")

    # Challenger — XGBoost
    logger.info("Training Challenger model (XGBoost)...")
    challenger = xgb.XGBClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.05,
        scale_pos_weight=8, eval_metric="logloss", verbosity=0, random_state=random_state
    )
    challenger.fit(X_train, y_train)
    challenger_auc = roc_auc_score(y_test, challenger.predict_proba(X_test)[:, 1])
    challenger_f1  = f1_score(y_test, challenger.predict(X_test), zero_division=0)
    challenger.save_model(os.path.join(ARTIFACTS_DIR, "challenger_model.json"))
    logger.info(f"Challenger AUC: {challenger_auc:.4f} | F1: {challenger_f1:.4f}")

    # Save shared artifacts
    json.dump(feature_names, open(os.path.join(ARTIFACTS_DIR, "feature_order.json"), "w"))
    json.dump(encoders, open(os.path.join(ARTIFACTS_DIR, "encoders.json"), "w"))

    meta = {
        "control":    {"model": "LogisticRegression", "auc": round(control_auc, 4), "f1": round(control_f1, 4)},
        "challenger": {"model": "XGBoost",            "auc": round(challenger_auc, 4), "f1": round(challenger_f1, 4)},
        "features":   feature_names,
        "data_source": data_path if data_path else "synthetic",
    }
    json.dump(meta, open(os.path.join(ARTIFACTS_DIR, "model_meta.json"), "w"), indent=2)

    print("\n" + "="*55)
    print("MODELS TRAINED")
    print("="*55)
    print(f"  Control (A)    — LogReg   AUC: {control_auc:.4f} | F1: {control_f1:.4f}")
    print(f"  Challenger (B) — XGBoost  AUC: {challenger_auc:.4f} | F1: {challenger_f1:.4f}")
    print("="*55)

    return control, challenger, feature_names, control_auc, challenger_auc


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--data-path", type=str, default=None)
    args = p.parse_args()
    train_models(data_path=args.data_path)

# 🔀 A/B Testing Framework — Control vs Challenger

![CI](https://github.com/jumma786/mlops-ab-testing/actions/workflows/ab_testing.yml/badge.svg)
![Python](https://img.shields.io/badge/Python-3.11%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green)
![SciPy](https://img.shields.io/badge/SciPy-stats-orange)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

> **Part of the MLOps Portfolio Series** — Project 8 of 10  
> A/B testing framework that routes prediction traffic between Control (LogisticRegression) and Challenger (XGBoost), tracks conversions, and runs statistical significance tests to decide the winner.

---

## 📂 Project Resources

| Resource | Link |
|---|---|
| 🔀 A/B Router API | [src/router/app.py](src/router/app.py) |
| 📊 Stats Engine | [src/stats/significance.py](src/stats/significance.py) |
| 🏋️ Model Trainer | [src/models/train.py](src/models/train.py) |
| 🧪 Tests | [tests/test_ab.py](tests/test_ab.py) |
| 🤖 CI/CD | [.github/workflows/ab_testing.yml](.github/workflows/ab_testing.yml) |

---

## 🎯 What This Project Does

1. **Trains two models** on real UCI Bank Marketing data
2. **Routes traffic** — 50/50 split between Control and Challenger
3. **Tracks conversions** — counts positive predictions per variant
4. **Runs significance tests** — Z-test, Cohen's h, AUC comparison
5. **Decides winner** — promotes challenger if statistically + practically significant

---

## 📊 Models

| Variant | Model | AUC | Role |
|---|---|---|---|
| Control (A) | Logistic Regression | ~0.77 | Current production |
| Challenger (B) | XGBoost | ~0.82 | Candidate |

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Model + experiment status |
| GET | `/experiment-status` | Live conversion tracking |
| POST | `/predict` | Auto-routed prediction |
| POST | `/predict/control` | Force control model |
| POST | `/predict/challenger` | Force challenger model |
| POST | `/analyze` | Run significance test |
| POST | `/reset` | Reset experiment counters |

---

## 📐 Decision Framework

```
Promote Challenger if:
  ✅ p-value < 0.05 (statistically significant)
  ✅ lift > 1% (practically significant)
  ✅ challenger AUC > control AUC
```

---

## 🚀 Quick Start

```bash
git clone https://github.com/jumma786/mlops-ab-testing.git
cd mlops-ab-testing
pip install -r requirements.txt

# Train both models
python src/models/train.py --data-path data/bank-additional-full.csv

# Run tests
make test

# Start A/B router
make run
# Open http://127.0.0.1:8002/docs
```

---

## 🔗 MLOps Portfolio Series

| # | Project | Repo | Status |
|---|---|---|---|
| 1 | Multi-Model Tournament | [mlops-model-tournament](https://github.com/jumma786/mlops-model-tournament) | ✅ |
| 2 | Scheduled Retraining | [mlops-retraining-pipeline](https://github.com/jumma786/mlops-retraining-pipeline) | ✅ |
| 3 | Feature Engineering | [mlops-feature-pipeline](https://github.com/jumma786/mlops-feature-pipeline) | ✅ |
| 4 | Hyperparameter Tuning | [mlops-hyperparameter-tuning](https://github.com/jumma786/mlops-hyperparameter-tuning) | ✅ |
| 5 | Model Serving | [mlops-model-serving](https://github.com/jumma786/mlops-model-serving) | ✅ |
| 6 | Feature Store | [mlops-feature-store](https://github.com/jumma786/mlops-feature-store) | ✅ |
| 7 | Model Monitoring | [mlops-model-monitoring](https://github.com/jumma786/mlops-model-monitoring) | ✅ |
| **8** | **A/B Testing** | [mlops-ab-testing](https://github.com/jumma786/mlops-ab-testing) | ✅ This repo |
| 9 | Airflow Pipeline | [mlops-airflow-pipeline](https://github.com/jumma786/mlops-airflow-pipeline) | ✅ |
| 10 | Kubernetes Platform | [mlops-k8s-platform](https://github.com/jumma786/mlops-k8s-platform) | ✅ |

---

## 👤 Author

**Jumma Mohammad Teli** — Data Analyst & ML Engineer  
📍 Birmingham, UK  
📧 [jummamohammad477@gmail.com](mailto:jummamohammad477@gmail.com)  
🔗 [LinkedIn](https://linkedin.com/in/jumma-mohammad) | [GitHub](https://github.com/jumma786)

---

*Project 8 of 10 — MLOps Portfolio Series.*

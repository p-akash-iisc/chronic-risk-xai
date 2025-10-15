# Chronic Disease Risk Prediction with Explainable AI

[![CI](https://img.shields.io/github/actions/workflow/status/yourname/chronic-risk-xai/ci.yml?branch=main)](https://github.com/yourname/chronic-risk-xai/actions)
[![Coverage](https://img.shields.io/badge/coverage-~80%25-brightgreen)](#)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**One-command, production-quality ML project** for predicting chronic disease risk (Diabetes by default; Framingham as optional) with Explainable AI (SHAP), robust evaluation, fairness checks, Optuna HPO, Streamlit dashboard, and FastAPI service.

---

## Quickstart

```bash
# 1) Clone and setup
git clone https://github.com/yourname/chronic-risk-xai.git
cd chronic-risk-xai
make setup

# 2) Get data (default: Pima Indians Diabetes from UCI)
make data

# 3) Train, evaluate, explain
make train
make evaluate
make explain

# 4) Apps
make dashboard   # Streamlit at http://localhost:8501
make api         # FastAPI at http://127.0.0.1:8000/docs
```

> Reproducible on a fresh machine: **`make setup && make data && make train && make evaluate`**

---

## Project Overview

- **Goal:** Predict chronic disease risk with **transparency** and **fairness**.
- **Explainability:** SHAP global & local plots.
- **Evaluation:** ROC-AUC, PR-AUC, F1, Recall@K, calibration; gains/lift; bootstrap CIs.
- **Fairness:** Per-group metrics (e.g., `sex`, `age_bin`), Δgaps.
- **Tuning:** Optuna (TPE + MedianPruner) across LogisticRegression, RandomForest, XGBoost, LightGBM.
- **Repro:** Seeds, environment files, CI, tests. Experiments logged via MLflow (local `mlruns/`).

### Pipeline Diagram

```
raw data → cleaning & split → features → HPO + training → evaluation → SHAP & fairness → apps (API/Dashboard)
```

---

## Repository Layout

```
chronic-risk-xai/
├─ README.md
├─ LICENSE
├─ CITATION.cff
├─ pyproject.toml
├─ requirements.txt
├─ environment.yml
├─ Makefile
├─ .gitignore
├─ .github/workflows/ci.yml
├─ configs/
│  ├─ default.yaml
│  └─ framingham.yaml
├─ data/
│  ├─ raw/
│  ├─ processed/
│  └─ sample/
├─ notebooks/
│  ├─ 01_eda.ipynb
│  ├─ 02_feature_engineering.ipynb
│  └─ 03_model_cards.ipynb
├─ src/
│  ├─ chronic_risk/
│  │  ├─ __init__.py
│  │  ├─ data.py
│  │  ├─ features.py
│  │  ├─ modeling.py
│  │  ├─ train.py
│  │  ├─ evaluate.py
│  │  ├─ explain.py
│  │  ├─ fairness.py
│  │  ├─ api.py
│  │  ├─ dashboard.py
│  │  ├─ utils.py
│  │  └─ sql/
│  │      ├─ queries.sql
│  │      └─ runner.py
│  └─ cli/
│      ├─ prepare_data.py
│      ├─ run_experiment.py
│      ├─ run_explain.py
│      ├─ serve_api.py
│      └─ launch_dashboard.py
├─ models/
│  ├─ artifacts/
│  └─ reports/
└─ tests/
   ├─ test_data.py
   ├─ test_features.py
   ├─ test_modeling.py
   ├─ test_api.py
   └─ test_dashboard.py
```

---

## Dataset Options

- **Default:** UCI Pima Indians Diabetes. Auto-downloaded with checksum validation.
- **Optional:** Framingham Heart Study (manual or pre-downloaded CSV). Switch with:
  ```bash
  make train CONFIG=configs/framingham.yaml
  ```

See `data/README.md` for details.

---

## Operations Guide

### 1) Switching Datasets
```bash
make train CONFIG=configs/default.yaml      # Pima (default)
make train CONFIG=configs/framingham.yaml   # Framingham (requires CSV path or manual download)
```

### 2) Use Your Own Data
- Place CSV in `data/raw/`.
- Copy the config and edit paths/columns:
  ```bash
  cp configs/default.yaml configs/custom.yaml
  # Edit dataset.source_path, dataset.target, features.include, fairness.group_columns, etc.
  make data CONFIG=configs/custom.yaml
  make train CONFIG=configs/custom.yaml
  ```

### 3) MLflow UI
```bash
python -m mlflow ui --backend-store-uri mlruns
# open http://127.0.0.1:5000
```

### 4) FastAPI Examples
```bash
# health
curl -s http://127.0.0.1:8000/health

# predict one
curl -s -X POST http://127.0.0.1:8000/predict   -H "Content-Type: application/json"   -d '{"glucose": 148, "bmi": 33.6, "age": 50, "blood_pressure": 72,
       "insulin": 0, "skin_thickness": 35, "pregnancies": 2, "diabetes_pedigree": 0.35}'
```

---

## Expected Metrics (Pima)
- ROC-AUC ≥ **0.85** (with tuned XGB/LGBM).
- PR-AUC ≥ **baseline + 0.10**.
(We document seeds and any remaining nondeterminism.)

---

## Ethics & Bias
Healthcare ML can entrench inequities. We provide: per-group metrics, Δgaps, and calibration diagnostics. **Never** use predictions as the sole clinical criterion. Ensure IRB/privacy compliance and de-identify PHI.

---

## Development
```bash
make fmt      # auto-format (black) & organize imports
make lint     # ruff + black --check
make test     # pytest + coverage
```

## Citation
See `CITATION.cff`.

---

## License
[MIT](LICENSE)

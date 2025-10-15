

# 🧠 Chronic Disease Risk Prediction with Explainable AI (XAI)

[![Coverage](https://img.shields.io/badge/coverage-~80%25-brightgreen)](#)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red?logo=streamlit)](#)

### 🚀 [**Live Streamlit Dashboard → (Add your Streamlit link here)**](#)

A complete, production-grade **Machine Learning project** for predicting **chronic disease risk** (default: Diabetes) using **Explainable AI (SHAP)**.  
The project integrates **data processing, hyperparameter optimization (Optuna)**, **robust evaluation**, and a **modern Streamlit dashboard** for interactive explainability and predictions.

---

## ⚙️ Quickstart

```bash
# 1️⃣ Clone and setup
git clone https://github.com/p-akash-iisc/chronic-risk-xai.git
cd chronic-risk-xai
make setup

# 2️⃣ Get data (default: UCI Pima Indians Diabetes)
make data

# 3️⃣ Train, evaluate, and explain
make train
make evaluate
make explain

# 4️⃣ Launch dashboard
make dashboard   # Streamlit → http://localhost:8501
````

> 💡 Fully reproducible on any system:
> `make setup && make data && make train && make evaluate`

---

## 📊 Project Overview

| Component           | Description                                               |
| ------------------- | --------------------------------------------------------- |
| **Goal**            | Predict chronic disease risk (Diabetes) with transparency |
| **Explainability**  | SHAP (global & local) visualizations                      |
| **Evaluation**      | ROC-AUC, PR-AUC, F1, Recall@K, Calibration, Gains/Lift    |
| **Optimization**    | Optuna (TPE + MedianPruner) for multiple algorithms       |
| **Models**          | Logistic Regression, Random Forest, XGBoost, LightGBM     |
| **Interface**       | Interactive Streamlit dashboard for analysis & prediction |
| **Reproducibility** | Fixed seeds, structured configs, and clean artifacts      |

---

## 🧩 Pipeline Overview

```
Raw data → Preprocessing → Feature Engineering → Model Training & HPO
           → Evaluation & Explainability → Streamlit Dashboard
```

---

## 🗂️ Repository Structure

```
chronic-risk-xai/
├─ configs/
│  ├─ default.yaml
│  └─ framingham.yaml
├─ data/
│  ├─ raw/
│  ├─ processed/
│  └─ sample/
├─ models/
│  ├─ artifacts/
│  └─ reports/
├─ src/
│  ├─ chronic_risk/
│  │  ├─ data.py, train.py, evaluate.py, explain.py, dashboard.py, ...
│  └─ cli/
│      ├─ prepare_data.py, run_experiment.py, run_explain.py, launch_dashboard.py
├─ notebooks/
│  ├─ 01_eda.ipynb
│  ├─ 02_feature_engineering.ipynb
│  └─ 03_model_cards.ipynb
├─ requirements.txt
├─ Makefile
├─ LICENSE
└─ README.md
```

---

## 📚 Dataset

* **Default:** [UCI Pima Indians Diabetes Dataset](https://archive.ics.uci.edu/ml/datasets/pima+indians+diabetes)
* **Optional:** Framingham Heart Study (local CSV)

Switch datasets easily:

```bash
make train CONFIG=configs/framingham.yaml
```

Use your own dataset:

```bash
cp configs/default.yaml configs/custom.yaml
# Edit dataset paths, target column, and features
make train CONFIG=configs/custom.yaml
```

---

## 📈 Model Performance (Expected)

| Metric                  | Expected Value    |
| ----------------------- | ----------------- |
| **ROC-AUC**             | ≥ 0.85            |
| **PR-AUC**              | ≥ baseline + 0.10 |
| **Calibration (Brier)** | ≤ 0.15            |

> The best model is automatically selected based on validation **PR-AUC**.

---

## 🧮 Evaluation & Explainability

* **ROC / PR Curves:** Model discrimination
* **Confusion Matrix:** True/False Positives & Negatives
* **Calibration Curve:** Probability reliability
* **Gains & Lift Charts:** Targeting effectiveness
* **SHAP Explainability:**

  * *Global* — Beeswarm & bar plots (feature importance)
  * *Local* — Waterfall plot (individual risk explanations)

---

## 🎨 Streamlit Dashboard Features

✅ **Model Comparison** — View metrics & plots for all trained models
✅ **Interactive Explainability** — Visualize SHAP global & local impacts
✅ **Sample Predictions** — Explore high/low-risk samples from validation set
✅ **Custom Uploads** — Upload your own CSV to predict new patient risks
✅ **Light/Dark Mode** — Adaptive, aesthetic theme with clean UI

---

## 🧰 Development Commands

```bash
make fmt      # Auto-format with Black + isort
make lint     # Static linting (ruff)
make test     # Run unit tests (pytest)
```

---

## ⚖️ Ethical Use

This project is for **educational and research purposes**.
Predictions **should not** be used for medical decisions without expert oversight.
Ensure fairness and bias evaluation before any real-world use.

---

## 📄 License

Released under the [MIT License](LICENSE).

---

## 👨‍💻 Author

**P. Akash Singh**
📧 [elearning.apu@gmail.com](mailto:elearning.apu@gmail.com)
🌐 [GitHub Profile](https://github.com/p-akash-iisc)

---

⭐ *If you found this project helpful, please consider giving it a star!* 🌟



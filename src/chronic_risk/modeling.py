from __future__ import annotations
from typing import Dict, Any
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

def get_model_space(cfg) -> Dict[str, Any]:
    cw = cfg["training"].get("class_weight", None)
    n_jobs = cfg["training"].get("n_jobs", -1)
    space = {
        "logreg": {
            "model": LogisticRegression(max_iter=2000, class_weight=cw, n_jobs=n_jobs, solver="lbfgs"),
            "params": {
                "C": [0.01, 0.1, 1.0, 10.0]
            }
        },
        "rf": {
            "model": RandomForestClassifier(class_weight=cw, n_jobs=n_jobs, random_state=cfg["dataset"]["random_state"]),
            "params": {
                "n_estimators": [100, 300, 500],
                "max_depth": [None, 5, 8, 12],
                "min_samples_split": [2, 5, 10]
            }
        },
        "xgb": {
            "model": XGBClassifier(
                n_estimators=400, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                max_depth=4, eval_metric="logloss", n_jobs=n_jobs, random_state=cfg["dataset"]["random_state"]
            ),
            "params": {
                "n_estimators": [200, 400, 600],
                "learning_rate": [0.03, 0.05, 0.1],
                "max_depth": [3, 4, 5],
                "subsample": [0.7, 0.8, 1.0],
                "colsample_bytree": [0.7, 0.8, 1.0]
            }
        },
        "lgbm": {
            "model": LGBMClassifier(
                n_estimators=400, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                max_depth=-1, n_jobs=n_jobs, random_state=cfg["dataset"]["random_state"]
            ),
            "params": {
                "n_estimators": [200, 400, 600],
                "learning_rate": [0.03, 0.05, 0.1],
                "num_leaves": [15, 31, 63],
                "feature_fraction": [0.7, 0.8, 1.0],
                "bagging_fraction": [0.7, 0.8, 1.0]
            }
        },
    }
    return space

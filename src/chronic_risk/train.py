from __future__ import annotations
import os, json, warnings
from typing import Dict, Any
from pathlib import Path

import numpy as np
import pandas as pd
import mlflow
import optuna
from joblib import dump
from sklearn.model_selection import StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV

# project utils
from .data import preprocess_and_split
from .modeling import get_model_space
from .utils import load_config, ensure_dirs, set_seeds, get_logger

warnings.filterwarnings("ignore")

# ------------------------------- Optuna objective -------------------------------

def _proba_like(model, X_df: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X_df)[:, 1]
    if hasattr(model, "decision_function"):
        s = model.decision_function(X_df).astype(float)
        return 1.0 / (1.0 + np.exp(-s))
    return model.predict(X_df).astype(float)

def objective(trial, X, y, cfg, space):
    from sklearn.metrics import average_precision_score, roc_auc_score, f1_score

    metric = cfg["optuna"]["optimize_metric"]  # "pr_auc" or "roc_auc"
    cv = StratifiedKFold(
        n_splits=cfg["training"]["cv_folds"],
        shuffle=True,
        random_state=cfg["dataset"]["random_state"],
    )

    model_name = trial.suggest_categorical("model", list(space.keys()))
    base = space[model_name]["model"]
    params_grid = space[model_name]["params"]

    # unique param names per-model for Optuna
    params = {}
    for k, options in params_grid.items():
        val = trial.suggest_categorical(f"{model_name}__{k}", options)
        params[k] = val

    model = base.__class__(**{**base.get_params(), **params})

    oof_preds = np.zeros(len(y), dtype=float)
    for tr, va in cv.split(X, y):
        model.fit(X.iloc[tr], y[tr])
        p = _proba_like(model, X.iloc[va])
        oof_preds[va] = p

    pr_auc = average_precision_score(y, oof_preds)
    roc = roc_auc_score(y, oof_preds)
    f1 = f1_score(y, (oof_preds >= 0.5).astype(int))

    # optional: store for dashboards
    trial.set_user_attr("roc_auc", float(roc))
    trial.set_user_attr("f1", float(f1))
    trial.set_user_attr("model_name", model_name)

    return pr_auc if metric == "pr_auc" else roc

def best_params_for(study: optuna.Study, model_name: str) -> Dict[str, Any]:
    """Extract best completed-trial params for a given model, stripping the '<model>__' prefix."""
    candidates = [
        t for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE
        and t.params.get("model") == model_name
    ]
    if not candidates:
        return {}
    best_t = max(candidates, key=lambda t: t.value)  # direction=max
    tuned = {
        k.split(f"{model_name}__", 1)[1]: v
        for k, v in best_t.params.items()
        if k.startswith(f"{model_name}__")
    }
    return tuned

# ----------------------------------- Main --------------------------------------

def main(config_path: str):
    cfg = load_config(config_path)
    logger = get_logger()
    set_seeds(cfg["optuna"]["seed"])

    ensure_dirs(cfg["paths"]["artifacts_dir"], cfg["paths"]["reports_dir"])

    # ---------------- data (train/val only) ----------------
    processed_dir = cfg["paths"]["processed_dir"]
    if not os.path.exists(os.path.join(processed_dir, "train.csv")):
        preprocess_and_split(cfg)

    train_df = pd.read_csv(os.path.join(processed_dir, "train.csv"))
    val_df   = pd.read_csv(os.path.join(processed_dir, "val.csv"))
    target = cfg["dataset"]["target"]

    X = pd.concat([train_df.drop(columns=[target]), val_df.drop(columns=[target])], axis=0).reset_index(drop=True)
    y = pd.concat([train_df[target], val_df[target]], axis=0).astype(int).values

    space = get_model_space(cfg)

    # ---------------- search ----------------
    # Keep mlflow local to artifacts dir so repo root stays clean
    mlflow.set_tracking_uri(f"file:{Path(cfg['paths']['artifacts_dir']) / 'mlruns'}")
    mlflow.set_experiment("chronic-risk-xai")

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=cfg["optuna"]["seed"]),
        pruner=optuna.pruners.MedianPruner(),
    )
    study.optimize(lambda t: objective(t, X, y, cfg, space),
                   n_trials=cfg["optuna"]["n_trials"],
                   timeout=cfg["optuna"]["timeout"])

    # Optional: persist search trace (no plots/metrics here)
    try:
        optuna_df = study.trials_dataframe(attrs=("number","value","params","user_attrs","state"))
        optuna_df.to_csv(Path(cfg["paths"]["reports_dir"]) / "optuna_trials.csv", index=False)
        with open(Path(cfg["paths"]["reports_dir"]) / "optuna_summary.json", "w") as f:
            json.dump({"optimize_metric": cfg["optuna"]["optimize_metric"],
                       "best_value_cv": float(study.best_value),
                       "best_params_global": study.best_params}, f, indent=2)
    except Exception:
        pass

    # ---------------- fit each algo on train+val and save ----------------
    art_dir = Path(cfg["paths"]["artifacts_dir"])
    art_dir.mkdir(parents=True, exist_ok=True)

    for model_name in space.keys():
        base  = space[model_name]["model"]
        tuned = best_params_for(study, model_name)
        mdl   = base.__class__(**{**base.get_params(), **tuned})

        method = cfg["calibration"]["method"]
        if method and str(method).lower() != "none":
            mdl = CalibratedClassifierCV(mdl, method=method, cv=3)

        mdl.fit(X, y)
        # One artifact per algorithm; no winner/plots/metrics here.
        dump(mdl, art_dir / f"{model_name}.joblib")
        logger.info(f"[train] saved {model_name}.joblib")

    logger.info("Training complete. Models saved under models/artifacts/")
    return 0

# ----------------------------------- CLI ---------------------------------------

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    args = p.parse_args()
    main(args.config)

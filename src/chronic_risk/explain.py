from __future__ import annotations
import os
from pathlib import Path
import numpy as np
import pandas as pd
import shap

import matplotlib
matplotlib.use("Agg")  # headless-safe before importing pyplot
import matplotlib.pyplot as plt

from joblib import load
from .utils import load_config, ensure_dirs, get_logger


def _ensure_dataframe(X, feature_cols):
    """Ensure X is a pandas DataFrame with the expected column names."""
    if isinstance(X, pd.DataFrame):
        return X
    return pd.DataFrame(X, columns=feature_cols)


def _proba_like(model, X_df: pd.DataFrame) -> np.ndarray:
    """Return a probability-like score for positive class for arbitrary classifiers."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X_df)[:, 1]
    if hasattr(model, "decision_function"):
        # map decision function to (0,1) via sigmoid
        s = model.decision_function(X_df).astype(float)
        return 1.0 / (1.0 + np.exp(-s))
    # fallback to predicted label cast to float
    return model.predict(X_df).astype(float)


def main(config_path: str):
    cfg = load_config(config_path)
    logger = get_logger()

    reports_dir = cfg["paths"]["reports_dir"]
    artifacts_dir = cfg["paths"]["artifacts_dir"]
    processed_dir = cfg["paths"]["processed_dir"]
    target = cfg["dataset"]["target"]
    seed = cfg["dataset"].get("random_state", 42)
    max_points = cfg.get("explain", {}).get("max_points", 2000)

    ensure_dirs(reports_dir)

    # --- Load validation split to avoid test leakage ---
    val_df = pd.read_csv(os.path.join(processed_dir, "val.csv"))
    y = val_df[target].astype(int).values
    X = val_df.drop(columns=[target])
    feature_cols = list(X.columns)
    X_df_full = _ensure_dataframe(X, feature_cols)

    # --- Iterate over all trained models (skip the legacy pointer) ---
    any_models = False
    for model_path in Path(artifacts_dir).glob("*.joblib"):
        if model_path.stem == "best_model":
            continue
        any_models = True

        model = load(str(model_path))
        algo_name = model_path.stem
        out_dir = os.path.join(reports_dir, algo_name)
        ensure_dirs(out_dir)

        # Optional downsampling for faster SHAP on large data
        if len(X_df_full) > max_points:
            X_to_explain = X_df_full.sample(n=max_points, random_state=seed)
        else:
            X_to_explain = X_df_full

        # --- Build SHAP explainer (new unified API) ---
        try:
            explainer = shap.Explainer(model, X_df_full, feature_names=X_df_full.columns)
            explanation = explainer(X_to_explain)
        except Exception as e:
            logger.warning(f"[{algo_name}] Default SHAP Explainer failed ({e}). Falling back to KernelExplainer.")
            background = shap.sample(X_df_full, min(100, len(X_df_full)), random_state=seed)
            explainer = shap.KernelExplainer(
                lambda data: _proba_like(model, _ensure_dataframe(data, feature_cols)),
                background
            )
            explanation = explainer(X_to_explain)

        # --- GLOBAL PLOTS (beeswarm & bar) ---
        plt.figure()
        shap.plots.beeswarm(explanation, show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "shap_summary_beeswarm.png"), dpi=150, bbox_inches="tight")
        plt.close()

        plt.figure()
        shap.plots.bar(explanation, show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "shap_importance_bar.png"), dpi=150, bbox_inches="tight")
        plt.close()

        # --- LOCAL PLOT (waterfall for highest predicted risk on full validation set) ---
        proba_val = _proba_like(model, X_df_full)
        top_idx = int(np.argmax(proba_val))
        single_expl = explainer(X_df_full.iloc[[top_idx]])

        plt.figure()
        shap.plots.waterfall(single_expl[0], show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "shap_local_waterfall_top1.png"), dpi=150, bbox_inches="tight")
        plt.close()

        logger.info(f"[{algo_name}] SHAP explanations saved to {out_dir}")

    if not any_models:
        logger.warning(f"No models found in {artifacts_dir}. Did you run training already?")

    logger.info("SHAP explanations generation complete.")
    return 0


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    args = p.parse_args()
    main(args.config)

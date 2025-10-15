from __future__ import annotations
"""
Evaluate all saved models on the TEST set, write per-algo reports,
pick a winner, mirror winner artifacts to root reports/, and save a summary table.
"""

import os, json, shutil
from pathlib import Path
from typing import Callable, Dict, Any, List

import numpy as np
import pandas as pd

# Headless-safe plotting
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from joblib import load
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_recall_curve,
    roc_curve, confusion_matrix, f1_score, precision_score, recall_score
)

from .utils import (
    load_config, ensure_dirs, brier_score, calibration_curve,
    get_logger, topk_threshold
)

# -------------------------- helpers --------------------------

def _proba_like(model, X_df: pd.DataFrame) -> np.ndarray:
    """Return positive-class probability-like scores for arbitrary classifiers."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X_df)[:, 1]
    if hasattr(model, "decision_function"):
        s = model.decision_function(X_df).astype(float)
        return 1.0 / (1.0 + np.exp(-s))  # sigmoid
    return model.predict(X_df).astype(float)

def _plot_and_save(x, y, xlabel, ylabel, title, path: Path):
    plt.figure()
    plt.plot(x, y)
    plt.xlabel(xlabel); plt.ylabel(ylabel); plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()

def _gains_lift(y_true, y_prob, path_prefix: Path):
    order = np.argsort(-y_prob)
    y_sorted = y_true[order]
    cum_pos = np.cumsum(y_sorted)
    perc = np.arange(1, len(y_true) + 1) / len(y_true)

    # Gains
    gains = cum_pos / (np.sum(y_true) + 1e-9)
    _plot_and_save(perc, gains, "Population %", "Cumulative Gains", "Gains Curve", path_prefix.with_name(path_prefix.name + "_gains.png"))

    # Lift
    lift = gains / np.maximum(perc, 1e-9)
    _plot_and_save(perc, lift, "Population %", "Lift", "Lift Chart", path_prefix.with_name(path_prefix.name + "_lift.png"))

def _bootstrap_ci(metric_fn: Callable, y_true: np.ndarray, y_prob: np.ndarray, B=200, seed=42):
    """Nonparametric bootstrap CI for a metric that takes (y_true, y_prob)."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    vals = []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        vals.append(metric_fn(y_true[idx], y_prob[idx]))
    vals = np.asarray(vals, dtype=float)
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))

# --------------------------- main ---------------------------

def main(config_path: str):
    cfg = load_config(config_path)
    logger = get_logger()

    reports_root = Path(cfg["paths"]["reports_dir"])
    artifacts_dir = Path(cfg["paths"]["artifacts_dir"])
    processed_dir = Path(cfg["paths"]["processed_dir"])
    ensure_dirs(str(reports_root))
    target = cfg["dataset"]["target"]

    # Data: TEST ONLY
    test_df = pd.read_csv(processed_dir / "test.csv")
    y_true = test_df[target].astype(int).values
    X_test = test_df.drop(columns=[target])

    # Candidate models: one .joblib per algo (skip pointer)
    model_paths = sorted([p for p in artifacts_dir.glob("*.joblib") if p.stem != "best_model"])
    if not model_paths:
        logger.warning(f"No models found in {artifacts_dir}. Run training first.")
        return 1

    best_name, best_score, best_thr = None, float("-inf"), 0.5
    primary = str(cfg["optuna"]["optimize_metric"]).lower().strip()  # 'pr_auc' or 'roc_auc'

    rows: List[Dict[str, Any]] = []

    for mp in model_paths:
        algo = mp.stem
        model = load(str(mp))

        # Scores
        y_prob = _proba_like(model, X_test)

        thr_cfg = cfg["thresholds"]
        if str(thr_cfg.get("strategy", "fixed")).lower() == "recall_at_k":
            thr = topk_threshold(y_prob, thr_cfg["recall_at_k"])
        else:
            thr = 0.5
        y_pred = (y_prob >= thr).astype(int)

        # Metrics
        pr_auc  = average_precision_score(y_true, y_prob)
        rocauc  = roc_auc_score(y_true, y_prob)
        f1      = f1_score(y_true, y_pred)
        prec    = precision_score(y_true, y_pred, zero_division=0)
        rec     = recall_score(y_true, y_pred, zero_division=0)
        brier   = brier_score(y_true, y_prob)

        # Per-algo report folder
        out_dir = reports_root / algo
        ensure_dirs(str(out_dir))

        # Curves & plots
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        pr_prec, pr_rec, _ = precision_recall_curve(y_true, y_prob)
        _plot_and_save(fpr, tpr, "FPR", "TPR", "ROC Curve", out_dir / "roc_curve.png")
        _plot_and_save(pr_rec, pr_prec, "Recall", "Precision", "PR Curve", out_dir / "pr_curve.png")

        cm = confusion_matrix(y_true, y_pred)
        plt.figure()
        plt.imshow(cm, interpolation="nearest")
        plt.title(f"Confusion Matrix (thr={thr:.3f})")
        plt.colorbar()
        tick_marks = np.arange(2)
        plt.xticks(tick_marks, ["0","1"]); plt.yticks(tick_marks, ["0","1"])
        plt.tight_layout(); plt.xlabel("Predicted"); plt.ylabel("True")
        plt.savefig(out_dir / "confusion_matrix.png", dpi=180)
        plt.close()

        _gains_lift(y_true, y_prob, out_dir / "eval")

        pt, pp = calibration_curve(y_true, y_prob, n_bins=10)
        _plot_and_save(pp, pt, "Predicted", "Observed", "Calibration Curve", out_dir / "calibration.png")

        # Bootstrap CIs
        roc_lo, roc_hi = _bootstrap_ci(roc_auc_score, y_true, y_prob)
        pr_lo,  pr_hi  = _bootstrap_ci(average_precision_score, y_true, y_prob)

        # metrics.json (per algo)
        metrics = {
            "roc_auc": float(rocauc),
            "roc_auc_ci": [roc_lo, roc_hi],
            "pr_auc": float(pr_auc),
            "pr_auc_ci": [pr_lo, pr_hi],
            "f1": float(f1),
            "precision": float(prec),
            "recall": float(rec),
            "brier": float(brier),
            "threshold": float(thr),
            "n_test": int(len(y_true))
        }
        with open(out_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        # For summary table
        rows.append({
            "algo": algo,
            "pr_auc": float(pr_auc), "pr_auc_lo": float(pr_lo), "pr_auc_hi": float(pr_hi),
            "roc_auc": float(rocauc), "roc_auc_lo": float(roc_lo), "roc_auc_hi": float(roc_hi),
            "brier": float(brier), "f1": float(f1),
            "precision": float(prec), "recall": float(rec),
            "threshold": float(thr), "n_test": int(len(y_true)),
        })

        # Track winner
        score = pr_auc if primary == "pr_auc" else rocauc
        if score > best_score:
            best_name, best_score, best_thr = algo, score, thr

        logger.info(f"[{algo}] pr_auc={pr_auc:.4f} roc_auc={rocauc:.4f} f1={f1:.4f} thr={thr:.3f}")

    # Winner artifacts (legacy compatibility)
    if best_name is None:
        logger.error("Could not determine a winner.")
        return 1

    shutil.copy2(artifacts_dir / f"{best_name}.joblib", artifacts_dir / "best_model.joblib")
    with open(artifacts_dir / "meta.json", "w") as f:
        json.dump({
            "best_model": best_name,
            "primary_metric": primary,
            "best_score": float(best_score),
            "threshold": float(best_thr)
        }, f, indent=2)

    # Mirror winner’s files to root reports/ for older UIs
    winner_dir = reports_root / best_name
    for fn in ["roc_curve.png","pr_curve.png","confusion_matrix.png",
               "eval_gains.png","eval_lift.png","calibration.png","metrics.json"]:
        src = winner_dir / fn
        dst = reports_root / fn
        if src.exists():
            try:
                if dst.exists() or dst.is_symlink():
                    os.remove(dst)
            except Exception:
                pass
            try:
                os.symlink(src, dst)
            except Exception:
                shutil.copy2(src, dst)

    # Save summary table (all algos)
    df = pd.DataFrame(rows)
    sort_col = "pr_auc" if primary == "pr_auc" else "roc_auc"
    df = df.sort_values(sort_col, ascending=False)

    df.to_csv(reports_root / "summary.csv", index=False)
    df.to_json(reports_root / "summary.json", orient="records", indent=2)

    # Optional: compact table in logs
    show_cols = ["algo", "pr_auc", "roc_auc", "brier", "f1", "precision", "recall", "threshold"]
    try:
        logger.info("\nPer-model summary:\n" + df[show_cols].to_string(index=False))
    except Exception:
        pass

    logger.info(f"Winner: {best_name} (by {primary}={best_score:.4f})")
    return 0

# --------------------------- CLI ---------------------------

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    args = p.parse_args()
    main(args.config)

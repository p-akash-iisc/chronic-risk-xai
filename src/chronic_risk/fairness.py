from __future__ import annotations
import os, json
import numpy as np
import pandas as pd
from typing import List
from joblib import load
from sklearn.metrics import recall_score, confusion_matrix
from .utils import load_config, get_logger

def subgroup_metrics(df: pd.DataFrame, target: str, proba: np.ndarray, thr: float, group_cols: List[str]):
    res = []
    y_true = df[target].astype(int).values
    y_pred = (proba >= thr).astype(int)
    for g in group_cols:
        if g not in df.columns: 
            continue
        for val, sub in df.groupby(g):
            yt = sub[target].astype(int).values
            yp = (proba[sub.index] >= thr).astype(int)
            tn, fp, fn, tp = confusion_matrix(yt, yp).ravel()
            rec = recall_score(yt, yp, zero_division=0)
            fpr = fp / (fp + tn + 1e-9)
            res.append({"group": g, "value": str(val), "n": len(sub), "recall": float(rec), "fpr": float(fpr)})
    # compute gaps
    out = {"by_group": res}
    for g in set(r["group"] for r in res):
        vals = [r for r in res if r["group"] == g]
        if not vals: continue
        recs = [r["recall"] for r in vals]
        fprs = [r["fpr"] for r in vals]
        out[f"{g}_recall_gap"] = float(max(recs) - min(recs))
        out[f"{g}_fpr_gap"] = float(max(fprs) - min(fprs))
    return out

def main(config_path: str):
    cfg = load_config(config_path)
    logger = get_logger()
    groups = cfg["fairness"]["group_columns"]
    if not groups:
        logger.info("No fairness group columns configured; skipping.")
        return 0
    processed_dir = cfg["paths"]["processed_dir"]
    target = cfg["dataset"]["target"]
    test_df = pd.read_csv(os.path.join(processed_dir, "test.csv"))
    X = test_df.drop(columns=[target])
    y = test_df[target].astype(int).values
    model = load(os.path.join(cfg["paths"]["artifacts_dir"], "best_model.joblib"))
    proba = model.predict_proba(X)[:,1]
    thr = json.load(open(os.path.join(cfg["paths"]["artifacts_dir"], "meta.json")))["threshold"]
    fairness = subgroup_metrics(test_df, target, proba, thr, groups)
    with open(os.path.join(cfg["paths"]["reports_dir"], "fairness.json"), "w") as f:
        json.dump(fairness, f, indent=2)
    logger.info("Fairness report saved.")
    return 0

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    args = p.parse_args()
    main(args.config)

from __future__ import annotations
import os, json, random, logging, math
from dataclasses import dataclass
from typing import Dict, Any, Tuple, List, Optional
import numpy as np
import yaml

def set_seeds(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)

def load_config(path: str) -> Dict[str, Any]:
    with open(path, 'r') as f:
        cfg = yaml.safe_load(f)
    return cfg

def ensure_dirs(*paths: str):
    for p in paths:
        os.makedirs(p, exist_ok=True)

def save_json(obj: Dict[str, Any], path: str):
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2)

def topk_threshold(scores: np.ndarray, k_frac: float) -> float:
    k = max(1, int(len(scores) * k_frac))
    # higher score = higher risk; threshold is the score at top-k boundary
    sorted_scores = np.sort(scores)[::-1]
    return float(sorted_scores[k-1])

def calibration_curve(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10):
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    binids = np.digitize(y_prob, bins) - 1
    prob_true = []
    prob_pred = []
    for b in range(n_bins):
        mask = binids == b
        if np.sum(mask) == 0:
            continue
        prob_true.append(np.mean(y_true[mask]))
        prob_pred.append(np.mean(y_prob[mask]))
    return np.array(prob_true), np.array(prob_pred)

def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    return float(np.mean((y_prob - y_true) ** 2))

def get_logger(name: str = "chronic_risk") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s"))
        logger.addHandler(ch)
    return logger

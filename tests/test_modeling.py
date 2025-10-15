import os, pandas as pd
from src.chronic_risk.utils import load_config
from src.chronic_risk.train import main as train_main

def test_train_smoke(tmp_path, monkeypatch):
    cfg = load_config("configs/default.yaml")
    # speed up HPO for test
    cfg["optuna"]["n_trials"] = 1
    cfg["paths"]["processed_dir"] = str(tmp_path / "processed")
    cfg["paths"]["artifacts_dir"] = str(tmp_path / "artifacts")
    cfg["paths"]["reports_dir"] = str(tmp_path / "reports")
    import yaml
    tmp_cfg = tmp_path / "cfg.yaml"
    yaml.safe_dump(cfg, open(tmp_cfg, "w"))
    # run training
    assert train_main(str(tmp_cfg)) == 0
    assert (tmp_path / "artifacts" / "best_model.joblib").exists()

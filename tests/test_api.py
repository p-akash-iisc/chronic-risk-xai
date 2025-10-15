import os, json, pandas as pd
from src.chronic_risk.utils import load_config
from src.chronic_risk.train import main as train_main
from fastapi.testclient import TestClient
from src.chronic_risk.api import app

def test_api_200(tmp_path, monkeypatch):
    cfg = load_config("configs/default.yaml")
    cfg["optuna"]["n_trials"] = 1
    cfg["paths"]["processed_dir"] = str(tmp_path / "processed")
    cfg["paths"]["artifacts_dir"] = str(tmp_path / "artifacts")
    cfg["paths"]["reports_dir"] = str(tmp_path / "reports")
    import yaml
    tmp_cfg = tmp_path / "cfg.yaml"
    yaml.safe_dump(cfg, open(tmp_cfg, "w"))
    # train a tiny model
    train_main(str(tmp_cfg))
    # point API to artifacts
    os.environ["ARTIFACTS_DIR"] = str(tmp_path / "artifacts")
    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200

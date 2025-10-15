import os, pandas as pd
from src.chronic_risk.utils import load_config
from src.chronic_risk.data import preprocess_and_split

def test_prepare_data(tmp_path):
    cfg = load_config("configs/default.yaml")
    cfg["paths"]["processed_dir"] = str(tmp_path / "processed")
    Xtr, Xv, Xte = preprocess_and_split(cfg)
    assert (tmp_path / "processed" / "train.csv").exists()
    df = pd.read_csv(tmp_path / "processed" / "train.csv")
    assert "target" in df.columns
    assert len(df) > 0

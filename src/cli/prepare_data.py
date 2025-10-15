from __future__ import annotations
from ..chronic_risk.data import preprocess_and_split
from ..chronic_risk.utils import load_config

def main():
    import argparse
    p = argparse.ArgumentParser(description="Prepare data: download/ingest and split.")
    p.add_argument("--config", default="configs/default.yaml", help="Path to YAML config")
    args = p.parse_args()
    cfg = load_config(args.config)
    preprocess_and_split(cfg)

if __name__ == "__main__":
    main()

from __future__ import annotations
from ..chronic_risk.train import main as train_main
from ..chronic_risk.utils import load_config

def main():
    import argparse
    p = argparse.ArgumentParser(description="Run training with Optuna + MLflow.")
    p.add_argument("--config", default="configs/default.yaml", help="Path to YAML config")
    args = p.parse_args()
    train_main(args.config)

if __name__ == "__main__":
    main()

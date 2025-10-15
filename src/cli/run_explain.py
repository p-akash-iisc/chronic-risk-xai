from __future__ import annotations
from ..chronic_risk.explain import main as explain_main

def main():
    import argparse
    p = argparse.ArgumentParser(description="Generate SHAP explanations and plots.")
    p.add_argument("--config", default="configs/default.yaml", help="Path to YAML config")
    args = p.parse_args()
    explain_main(args.config)

if __name__ == "__main__":
    main()

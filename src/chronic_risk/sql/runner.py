from __future__ import annotations
import os, sqlite3, pandas as pd
from .utils import load_config

def main(config_path: str):
    cfg = load_config(config_path)
    db_path = os.path.join("models","artifacts","demo.sqlite")
    con = sqlite3.connect(db_path)

    # Load processed train as patients table (synthetic IDs)
    train = pd.read_csv(os.path.join(cfg["paths"]["processed_dir"], "train.csv"))
    if "patient_id" not in train.columns:
        train = train.reset_index().rename(columns={"index":"patient_id"})
    train.to_sql("patients", con, if_exists="replace", index=False)

    # Create demographics table (toy)
    demo = train[["patient_id"]].copy()
    demo["sex"] = (demo["patient_id"] % 2).map({0:"F",1:"M"})
    demo.to_sql("demographics", con, if_exists="replace", index=False)

    # Run queries
    sql_path = os.path.join(os.path.dirname(__file__), "queries.sql")
    with open(sql_path) as f:
        sql_text = f.read()
    for stmt in sql_text.split(";"):
        s = stmt.strip()
        if not s: continue
        try:
            df = pd.read_sql_query(s + ";", con)
            print(f"Query result (first 5 rows):\n", df.head())
        except Exception as e:
            print("Query failed:", e)
    con.close()

if __name__ == "__main__":
    import argparse, os
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    args = p.parse_args()
    main(args.config)

from __future__ import annotations
import os, io, csv, hashlib, urllib.request
from typing import Dict, Any, Tuple, List, Optional
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from .utils import ensure_dirs, get_logger, set_seeds

PIMA_URL = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"  # clean CSV
# Columns: pregnancies,glucose,blood_pressure,skin_thickness,insulin,bmi,diabetes_pedigree,age,target

def sha256_of_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def download_pima(raw_path: str) -> str:
    os.makedirs(os.path.dirname(raw_path), exist_ok=True)
    urllib.request.urlretrieve(PIMA_URL, raw_path)
    return raw_path

def load_raw(cfg: Dict[str, Any]) -> pd.DataFrame:
    ds = cfg["dataset"]
    raw_dir = cfg["paths"]["raw_dir"]
    name = ds["name"]
    src = ds.get("source_path")
    if name == "pima":
        path = src or os.path.join(raw_dir, "pima_diabetes.csv")
        if not os.path.exists(path):
            download_pima(path)
        df = pd.read_csv(path)
        # normalize column names
        rename = {
            "Pregnancies":"pregnancies","Glucose":"glucose","BloodPressure":"blood_pressure",
            "SkinThickness":"skin_thickness","Insulin":"insulin","BMI":"bmi",
            "DiabetesPedigreeFunction":"diabetes_pedigree","Age":"age","Outcome":"target"
        }
        df = df.rename(columns=rename)
        # if already normalized, keep as-is
        df.columns = [c.lower() for c in df.columns]
    elif name == "framingham":
        path = src
        if not path or not os.path.exists(path):
            raise FileNotFoundError("Framingham CSV not found. Set dataset.source_path to your local CSV.")
        df = pd.read_csv(path)
    else:
        # Custom
        path = src
        if not path or not os.path.exists(path):
            raise FileNotFoundError("Custom dataset.source_path not provided or file missing.")
        df = pd.read_csv(path)
    return df

def preprocess_and_split(cfg: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    logger = get_logger()
    set_seeds(cfg["dataset"]["random_state"])
    df = load_raw(cfg)
    target = cfg["dataset"]["target"]
    include = cfg["features"]["include"]
    drop_cols = cfg["features"].get("drop", []) or []
    cats = cfg["features"].get("categorical", []) or []
    use_cols = list(dict.fromkeys(include + [target] + cats))
    df = df[[c for c in use_cols if c in df.columns]].copy()
    # Basic cleaning: handle missing
    imp = cfg["preprocessing"]["imputation"]
    for c in include:
        if c not in df.columns:
            continue
        if df[c].dtype.kind in "biufc":
            if imp == "median":
                df[c] = df[c].fillna(df[c].median())
            else:
                df[c] = df[c].fillna(df[c].mode().iloc[0])
    for c in cats:
        if c in df.columns:
            df[c] = df[c].astype("category").cat.add_categories(["__missing__"]).fillna("__missing__")

    if cfg["preprocessing"]["clip_outliers"]:
        ql, qh = cfg["preprocessing"]["clip_q"]
        for c in include:
            if c in df.columns and df[c].dtype.kind in "biufc":
                lo, hi = df[c].quantile(ql), df[c].quantile(qh)
                df[c] = df[c].clip(lo, hi)

    # Scaling numeric
    scaler_name = cfg["preprocessing"]["scaling"]
    num_cols = [c for c in include if c in df.columns and df[c].dtype.kind in "biufc"]
    scaler = StandardScaler() if scaler_name == "standard" else MinMaxScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])

    # One-hot categorical
    if cats:
        df = pd.get_dummies(df, columns=cats, drop_first=True)

    # Split
    test_size = cfg["dataset"]["test_size"]
    val_size = cfg["dataset"]["val_size"]
    random_state = cfg["dataset"]["random_state"]
    y = df[target].astype(int).values
    X = df.drop(columns=[target])
    X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)
    val_rel = val_size / (1 - test_size)
    X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=1-val_rel, stratify=y_tmp, random_state=random_state)

    # Save processed
    out_dir = cfg["paths"]["processed_dir"]
    os.makedirs(out_dir, exist_ok=True)
    X_train.assign(**{ "target": y_train }).to_csv(os.path.join(out_dir, "train.csv"), index=False)
    X_val.assign(**{ "target": y_val }).to_csv(os.path.join(out_dir, "val.csv"), index=False)
    X_test.assign(**{ "target": y_test }).to_csv(os.path.join(out_dir, "test.csv"), index=False)

    logger.info(f"Processed splits saved to {out_dir}")
    return X_train, X_val, X_test

from __future__ import annotations
import os, json
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ValidationError
from joblib import load
import numpy as np

app = FastAPI(title="Chronic Risk API")

ART_DIR = os.environ.get("ARTIFACTS_DIR", "models/artifacts")
MODEL_PATH = os.path.join(ART_DIR, "best_model.joblib")
META_PATH = os.path.join(ART_DIR, "meta.json")

class Record(BaseModel):
    # Pima-like defaults; FastAPI will accept any keys used in training.
    pregnancies: float | None = None
    glucose: float
    blood_pressure: float | None = None
    skin_thickness: float | None = None
    insulin: float | None = None
    bmi: float
    diabetes_pedigree: float | None = None
    age: float

class Batch(BaseModel):
    records: List[Dict[str, float]]

def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Model not found. Train first.")
    model = load(MODEL_PATH)
    meta = json.load(open(META_PATH))
    return model, meta

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(record: Dict[str, float]):
    model, meta = load_model()
    try:
        X = np.array([list(record.values())], dtype=float)
        # Keep column order consistent by using keys order; in real use load feature names.
        proba = float(model.predict_proba(X)[0,1])
        pred = int(proba >= meta.get("threshold", 0.5))
        return {"proba": proba, "pred": pred}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/predict_batch")
def predict_batch(batch: Batch):
    model, meta = load_model()
    try:
        keys = sorted(batch.records[0].keys())
        X = np.array([[r.get(k, 0.0) for k in keys] for r in batch.records], dtype=float)
        proba = model.predict_proba(X)[:,1].tolist()
        preds = [int(p >= meta.get("threshold", 0.5)) for p in proba]
        return {"keys": keys, "proba": proba, "preds": preds}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

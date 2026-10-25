"""Inference API for the trained baseline (trusted local model files only)."""
from __future__ import annotations

import os
import json
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.train import FEATURES

MODEL_PATH = Path(os.environ.get("MODEL_PATH", "artifacts/baseline.joblib"))
THRESHOLD = float(os.environ.get("DECISION_THRESHOLD", "0.5"))
model = None


class Booking(BaseModel):
    hotel: str
    lead_time: int = Field(ge=0)
    arrival_date_month: str
    arrival_date_day_of_month: int = Field(ge=1, le=31)
    stays_in_weekend_nights: int = Field(ge=0)
    stays_in_week_nights: int = Field(ge=0)
    adults: int = Field(ge=0)
    children: float | None = Field(default=None, ge=0)
    babies: int = Field(ge=0)
    market_segment: str
    customer_type: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, THRESHOLD
    metrics_path = MODEL_PATH.with_name("metrics.json")
    if "DECISION_THRESHOLD" not in os.environ and metrics_path.exists():
        THRESHOLD = float(json.loads(metrics_path.read_text(encoding="utf-8"))["threshold_selected_on_validation"])
    if MODEL_PATH.exists():
        model = joblib.load(MODEL_PATH)  # Never load untrusted serialized models.
    yield
    model = None


app = FastAPI(title="Hotel Cancellation Risk — Portfolio Demo", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ready" if model is not None else "model_missing"}


@app.post("/predict")
def predict(booking: Booking):
    if model is None:
        raise HTTPException(status_code=503, detail="Train model before requesting predictions")
    row = pd.DataFrame([booking.model_dump()])[FEATURES]
    probability = float(model.predict_proba(row)[0, 1])
    return {
        "cancellation_probability": round(probability, 6),
        "predicted_cancellation": probability >= THRESHOLD,
        "threshold": THRESHOLD,
        "disclaimer": "Historical portfolio model; not validated for operational decisions",
    }

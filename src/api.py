"""FastAPI inference; start after training with uvicorn src.api:app --reload."""
from functools import lru_cache
from pathlib import Path
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app=FastAPI(title="Hotel Booking Cancellation Risk Demo")
MODEL=Path("artifacts/model.joblib")

class Booking(BaseModel):
    hotel: str
    lead_time: int = Field(ge=0)
    arrival_date_year: int
    arrival_date_month: str
    arrival_date_day_of_month: int = Field(ge=1,le=31)
    stays_in_weekend_nights: int = Field(ge=0)
    stays_in_week_nights: int = Field(ge=0)
    adults: int = Field(ge=0)
    children: float | None = 0
    babies: int = Field(ge=0)
    market_segment: str
    deposit_type: str
    customer_type: str

@lru_cache(maxsize=1)
def load_model():
    if not MODEL.exists():
        raise FileNotFoundError("Model missing; run python -m src.train first")
    return joblib.load(MODEL)

@app.get("/health")
def health():
    return {"status":"ok","model_present":MODEL.exists()}

@app.post("/predict")
def predict(booking:Booking):
    try:
        score=float(load_model().predict_proba(pd.DataFrame([booking.model_dump()]))[0,1])
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503,detail=str(exc)) from exc
    band="high" if score>=0.7 else "medium" if score>=0.3 else "low"
    return {"cancellation_probability":score,"risk_band":band,"thresholds":{"low_below":0.3,"high_from":0.7}}

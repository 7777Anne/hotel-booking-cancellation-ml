"""Train a leakage-aware booking-time logistic regression baseline."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data import TARGET, create_modelling_table, load_data

# Conservative booking-time subset; other fields require timestamp/availability verification.
FEATURES = [
    "hotel", "lead_time", "arrival_date_month", "arrival_date_day_of_month",
    "stays_in_weekend_nights", "stays_in_week_nights", "adults",
    "children", "babies", "market_segment", "customer_type",
]
CATEGORICAL = ["hotel", "arrival_date_month", "market_segment", "customer_type"]
NUMERIC = [x for x in FEATURES if x not in CATEGORICAL]


def make_pipeline() -> Pipeline:
    preprocess = ColumnTransformer([
        ("numeric", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), NUMERIC),
        ("categorical", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore")),
        ]), CATEGORICAL),
    ])
    return Pipeline([
        ("preprocess", preprocess),
        ("model", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)),
    ])


def prepare(frame: pd.DataFrame) -> pd.DataFrame:
    table = create_modelling_table(frame)
    if table[TARGET].isna().any():
        raise ValueError("Target contains missing values")
    if len(table) < 30:
        raise ValueError("Need at least 30 rows for chronological train/validation/test")
    return table


def chronological_split(table: pd.DataFrame):
    """Sort by scheduled arrival (proxy only; booking timestamp is unavailable)."""
    month = pd.to_datetime(table["arrival_date_month"], format="%B", errors="coerce").dt.month
    if month.isna().any():
        raise ValueError("Invalid arrival_date_month")
    dates = pd.to_datetime(dict(
        year=table["arrival_date_year"],
        month=month,
        day=table["arrival_date_day_of_month"],
    ), errors="coerce")
    if dates.isna().any():
        raise ValueError("Invalid arrival date")
    ordered = table.assign(_arrival_date=dates).sort_values("_arrival_date", kind="stable")
    # Group identical dates together to avoid splitting a calendar day across folds.
    unique_dates = ordered["_arrival_date"].drop_duplicates().sort_values()
    if len(unique_dates) < 3:
        raise ValueError("Need at least three distinct arrival dates")
    cut1 = unique_dates.iloc[max(1, int(len(unique_dates) * 0.6))]
    cut2 = unique_dates.iloc[max(2, int(len(unique_dates) * 0.8))]
    train = ordered[ordered["_arrival_date"] < cut1].drop(columns="_arrival_date")
    val = ordered[(ordered["_arrival_date"] >= cut1) & (ordered["_arrival_date"] < cut2)].drop(columns="_arrival_date")
    test = ordered[ordered["_arrival_date"] >= cut2].drop(columns="_arrival_date")
    for part in (train, val, test):
        if part.empty or part[TARGET].nunique() < 2:
            raise ValueError("Each chronological split must contain both classes")
    return train, val, test


def score(y, probabilities, threshold: float) -> dict:
    predictions = (probabilities >= threshold).astype(int)
    return {
        "precision": float(precision_score(y, predictions, zero_division=0)),
        "recall": float(recall_score(y, predictions, zero_division=0)),
        "f1": float(f1_score(y, predictions, zero_division=0)),
        "pr_auc": float(average_precision_score(y, probabilities)),
        "roc_auc": float(roc_auc_score(y, probabilities)),
    }


def train(frame: pd.DataFrame):
    table = prepare(frame)
    train_set, validation, test = chronological_split(table)
    pipeline = make_pipeline()
    pipeline.fit(train_set[FEATURES], train_set[TARGET].astype(int))
    val_prob = pipeline.predict_proba(validation[FEATURES])[:, 1]
    # Choose decision threshold on validation only, never test.
    candidates = np.linspace(0.1, 0.9, 17)
    threshold = float(max(candidates, key=lambda t: f1_score(
        validation[TARGET], val_prob >= t, zero_division=0)))
    test_prob = pipeline.predict_proba(test[FEATURES])[:, 1]
    report = {
        "split_method": "scheduled arrival date chronological proxy; not booking timestamp",
        "split_sizes": {"train": len(train_set), "validation": len(validation), "test": len(test)},
        "features": FEATURES,
        "threshold_selected_on_validation": threshold,
        "validation": score(validation[TARGET], val_prob, threshold),
        "test": score(test[TARGET], test_prob, threshold),
        "limitations": [
            "Historical observational dataset; no booking creation timestamps",
            "Arrival-date split is not a true booking-time temporal split",
            "Booking-time availability of all features must be confirmed before deployment",
            "No probability calibration, business cost model or drift monitoring",
        ],
    }
    return pipeline, report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/raw/hotel_bookings.csv")
    parser.add_argument("--output-dir", default="artifacts")
    args = parser.parse_args()
    model, report = train(load_data(args.input))
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output / "baseline.joblib")
    (output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

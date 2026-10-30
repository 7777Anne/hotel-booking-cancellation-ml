"""Train a reproducible cancellation-risk pipeline on public Hotel Booking Demand data."""
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.data import load_data, create_modelling_table

def train(csv_path, output_dir="artifacts"):
    frame = create_modelling_table(load_data(csv_path))
    y = frame.pop("is_canceled").astype(int)
    # Conservative booking-time feature subset; avoid fields updated at checkout.
    allowed = ["hotel","lead_time","arrival_date_year","arrival_date_month",
               "arrival_date_day_of_month","stays_in_weekend_nights",
               "stays_in_week_nights","adults","children","babies",
               "market_segment","deposit_type","customer_type"]
    X = frame[[c for c in allowed if c in frame.columns]].copy()
    categorical = X.select_dtypes(include=["object","category"]).columns.tolist()
    numeric = [col for col in X if col not in categorical]
    prep = ColumnTransformer([
        ("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                                ("encode", OneHotEncoder(handle_unknown="ignore"))]), categorical),
        ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median")),
                            ("scale", StandardScaler())]), numeric)
    ])
    model = Pipeline([("preprocess", prep),
                      ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced"))])
    train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model.fit(train_x, train_y)
    probs = model.predict_proba(test_x)[:,1]
    pred = (probs >= 0.5).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(test_y,pred,average="binary",zero_division=0)
    metrics = {"precision":float(precision),"recall":float(recall),"f1":float(f1),
               "roc_auc":float(roc_auc_score(test_y,probs)),
               "pr_auc":float(average_precision_score(test_y,probs)),
               "split":"stratified_random_demo; not time-based prospective validation",
               "data":"public Hotel Booking Demand"}
    out = Path(output_dir); out.mkdir(parents=True,exist_ok=True)
    joblib.dump(model,out/"model.joblib")
    (out/"metrics.json").write_text(json.dumps(metrics,indent=2))
    print(metrics)
    return model, metrics

if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--input",default="data/raw/hotel_bookings.csv")
    parser.add_argument("--output",default="artifacts")
    args=parser.parse_args()
    train(args.input,args.output)

import numpy as np
import pandas as pd
import pytest
from src.train import FEATURES, chronological_split, make_pipeline, train


def sample_data(n=120):
    rng = np.random.default_rng(4)
    months = ["January", "February", "March", "April", "May", "June"]
    return pd.DataFrame({
        "hotel": ["City Hotel" if i % 2 else "Resort Hotel" for i in range(n)],
        "is_canceled": [i % 2 for i in range(n)],
        "lead_time": rng.integers(1, 200, n),
        "arrival_date_year": [2016] * n,
        "arrival_date_month": [months[(i // 20) % 6] for i in range(n)],
        "arrival_date_day_of_month": [(i % 20) + 1 for i in range(n)],
        "stays_in_weekend_nights": rng.integers(0, 3, n),
        "stays_in_week_nights": rng.integers(1, 6, n),
        "adults": [2] * n,
        "children": [0.0] * n,
        "babies": [0] * n,
        "market_segment": ["Online TA"] * n,
        "deposit_type": ["No Deposit"] * n,
        "customer_type": ["Transient"] * n,
        "adr": [100.0] * n,
        "reservation_status": ["Canceled" if i % 2 else "Check-Out" for i in range(n)],
    })


def test_pipeline_predicts_probabilities():
    frame = sample_data()
    pipeline = make_pipeline().fit(frame[FEATURES], frame["is_canceled"])
    p = pipeline.predict_proba(frame[FEATURES])
    assert p.shape == (len(frame), 2)
    assert np.isfinite(p).all()


def test_chronological_split_has_no_date_overlap():
    a, b, c = chronological_split(sample_data())
    def dates(x):
        return pd.to_datetime(dict(year=x.arrival_date_year,
          month=pd.to_datetime(x.arrival_date_month, format="%B").dt.month,
          day=x.arrival_date_day_of_month))
    assert dates(a).max() < dates(b).min() < dates(c).min()


def test_training_returns_metrics_and_threshold():
    model, report = train(sample_data())
    assert 0 < report["threshold_selected_on_validation"] < 1
    assert 0 <= report["test"]["pr_auc"] <= 1
    assert hasattr(model, "predict_proba")


def test_training_rejects_too_few_rows():
    with pytest.raises(ValueError, match="at least 30"):
        train(sample_data(20))

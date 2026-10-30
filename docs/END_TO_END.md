# End-to-end prototype

1. Download the public Hotel Booking Demand `hotel_bookings.csv` into `data/raw/` (see main README).
2. `pip install -r requirements.txt`
3. `python -m src.data --input data/raw/hotel_bookings.csv` audits source data.
4. `python -m src.train --input data/raw/hotel_bookings.csv` trains a logistic-regression baseline and stores metrics/model in `artifacts/`.
5. `pytest -q` runs data and API smoke tests.
6. `uvicorn src.api:app --reload` serves `/docs`, `/health`, `/predict`.
7. After generating `artifacts/model.joblib`, `docker build -t hotel-risk-demo . && docker run -p 8000:8000 hotel-risk-demo`.

This is a runnable prototype, not a validated production system. Random holdout does not establish prospective generalization; add time-aware evaluation, XGBoost comparison, SHAP, threshold optimization, and deployment monitoring before claiming those components. Never load untrusted joblib files.

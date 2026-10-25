# Hotel Booking Cancellation Risk — reproducible ML portfolio

**Status:** Logistic Regression baseline + tests + FastAPI demo implemented. XGBoost, SHAP, cloud deployment, monitoring, and business-cost calibration remain future work; this is **not** a production hotel decision system.

## Problem
Predict whether a reservation will be cancelled using a conservative subset of fields potentially known around the original booking. Dataset: [Hotel Booking Demand](https://doi.org/10.1016/j.dib.2018.11.126), 119,390 rows / 32 columns. The supplied public CSV uses `yes` and `no` cancellation labels, normalized to 1 and 0 on loading.

## Run
```bash
python -m pip install -r requirements.txt
# Download hotel_bookings.csv from https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand
mkdir -p data/raw
# Place the downloaded CSV at data/raw/hotel_bookings.csv
python -m src.data --input data/raw/hotel_bookings.csv
python -m src.train --input data/raw/hotel_bookings.csv --output-dir artifacts
python -m pytest -q
uvicorn src.api:app --reload
```
Visit `http://localhost:8000/docs` to test `POST /predict`. The API reads `artifacts/baseline.joblib` and the validation-selected threshold in `artifacts/metrics.json`. The model must be trained first. Only load trusted joblib files.

### Example prediction request
```json
{
  "hotel": "City Hotel", "lead_time": 90,
  "arrival_date_month": "June", "arrival_date_day_of_month": 12,
  "stays_in_weekend_nights": 1, "stays_in_week_nights": 3,
  "adults": 2, "children": 0, "babies": 0,
  "market_segment": "Online TA", "customer_type": "Transient"
}
```

### Docker (after training locally)
```bash
docker build -t hotel-risk .
docker run --rm -p 8000:8000 -v "$PWD/artifacts:/app/artifacts:ro" hotel-risk
```
Mounting a trusted, locally trained model is required; model artifacts and the dataset are not committed.

## Locally reproduced Logistic Regression results (25 October 2026)
Data: 119,390 rows; **31,994 identical rows** observed (not automatically removed: separate bookings may share recorded values). Grouped by scheduled arrival date into train 67,692, validation 22,896, test 28,802. Validation F1 selected a threshold of **0.40**. Held-out test: **ROC-AUC 0.7066, PR-AUC 0.6410, precision 0.4799, recall 0.8756, F1 0.6200**. Test cancellation prevalence: 40.79%. These numbers were obtained locally from the provided dataset and the same baseline training logic; reproduce by running the command above.

## Leakage and evaluation cautions
- `reservation_status` and `reservation_status_date` reveal outcomes; never model them.
- `assigned_room_type`, `booking_changes`, `days_in_waiting_list` are excluded.
- Original booking creation timestamps are unavailable. **Arrival-date chronology is an imperfect proxy, not true booking-time backtesting.** Some records arriving later may have been booked earlier.
- Training-only preprocessing avoids preprocessing leakage. Validation selects the threshold; test data are not used for tuning.
- Feature availability, probability calibration, subgroup behavior, intervention costs, drift, and external validation must be checked before any real operational use.
- Historical data are from 2015–2017 and may not represent modern hotel behavior.

## Files
- `src/data.py`: loading, normalization, schema checks, data audit
- `src/train.py`: train/validation/test split, preprocessing, baseline, metrics, saved model
- `src/api.py`: input-validated prediction API and health endpoint
- `tests/`: data, model and API checks
- `docs/BASELINE.md`: methodology and caveats
- `Dockerfile`: local demo container

## Next engineering milestones
XGBoost comparison, SHAP explanations, business-cost threshold analysis, stronger temporal evaluation with actual booking event timestamps, drift monitoring, and deployment.

Citation: Antonio, Almeida & Nunes (2019), *Hotel booking demand datasets*, Data in Brief. DOI: 10.1016/j.dib.2018.11.126.

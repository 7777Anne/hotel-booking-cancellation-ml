# Senior code-review record — 2026-10-30

The earlier baseline PR #2 was selected as the starting point because it keeps scheduled arrival-date groups together, has disjoint chronological train/validation/test sets, and tunes its classification threshold **on validation only**. This is still an imperfect proxy for booking event chronology and should not be represented as real-time production validation.

## Findings addressed
- Data audits could only print; now can export aggregated JSON for reproducibility without exposing individual records.
- Liveness returned HTTP 200 even with a missing trained model. This is fine for `/health`, but isn't a readiness signal. Added `/ready` that fails with 503 when model artifacts are unavailable.
- Original public-dataset labels can be 0/1. Resume wording has been updated to describe optional yes/no normalization, not assume that every source file has string labels.

## Known limitations
- The historical dataset has no reliable booking creation timestamp, which limits prediction-time analysis.
- Some features may change between original booking and arrival; verify actual input state before operational use.
- No genuine time-aware 2015–2017 prospective validation, confidence calibration, model monitoring, SHAP or cloud deployment claim.
- Previously reported held-out metrics must be reproduced by running locally with the public CSV.
- Serialized joblib artifacts must originate from trusted local training only.

## How to debug
```bash
python -m pip install -r requirements.txt
pytest -q -x
python -m src.data --input data/raw/hotel_bookings.csv --output artifacts/data_audit.json
python -m src.train --input data/raw/hotel_bookings.csv
uvicorn src.api:app --reload
```
First inspect schema and target distribution, next split dates, then threshold selected on validation, then held-out test metrics. Assert API input/feature parity and verify `/ready` before generating predictions.

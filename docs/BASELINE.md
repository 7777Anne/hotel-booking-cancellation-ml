# Logistic regression baseline

Run: `python -m src.train --input data/raw/hotel_bookings.csv --output-dir artifacts`.

This trains a scikit-learn pipeline with median numeric imputation, standard scaling, most-frequent categorical imputation and one-hot encoding. All preprocessing is fitted **only on the training partition**. Logistic regression uses balanced class weights.

The chronological split is based on *scheduled arrival date*, not booking creation date, because the public dataset does not provide reliable original booking timestamps. It is therefore an imperfect proxy; do not claim true booking-time backtesting. Dates are kept together at split boundaries. Threshold is selected by validation F1, and the held-out test set is used once for final evaluation.

Outputs are `artifacts/baseline.joblib` and `artifacts/metrics.json`. Raw data and artifacts are intentionally ignored by Git. Never load untrusted joblib files.

This baseline is **not production ready**. It lacks true booking-time snapshots, business-cost thresholding, calibration, fairness review, drift monitoring, and deployed inference. No measured results on the full public dataset are claimed until the dataset is supplied and training is executed.

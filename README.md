# Hotel Booking Cancellation ML

A leakage-aware machine learning project for predicting hotel booking cancellations and translating model scores into actionable risk bands.

## Why this project

Hotel cancellations affect inventory planning, pricing, staffing, and revenue. This project asks:

> Can we estimate cancellation risk using information available at booking time?

The first milestone is a reproducible data-quality and exploratory-analysis pipeline. Later milestones add a baseline model, XGBoost, threshold analysis, SHAP explanations, a FastAPI service, tests, Docker, and optional Google Cloud deployment.

## Dataset

Use the public **Hotel Booking Demand** dataset originally described by Nuno Antonio, Ana de Almeida, and Luis Nunes.

- Paper: https://doi.org/10.1016/j.dib.2018.11.126
- Kaggle mirror: https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand
- Expected file: `hotel_bookings.csv`
- Expected shape: approximately 119,390 rows and 32 columns
- Target: `is_canceled`

Download the CSV and save it locally as:

```text
data/raw/hotel_bookings.csv
```

Raw data is intentionally excluded from Git.

## Leakage policy

The prediction point is immediately after the original booking is created. Columns that reveal the final outcome are excluded:

- `reservation_status`
- `reservation_status_date`

The starter pipeline also excludes fields that may be updated after booking:

- `assigned_room_type`
- `booking_changes`
- `days_in_waiting_list`

This policy will be reviewed and documented before modelling.

## Repository structure

```text
.
├── data/
│   └── README.md
├── src/
│   ├── __init__.py
│   └── data.py
├── tests/
│   └── test_data.py
├── .gitignore
├── requirements.txt
└── README.md
```

## Quick start

Create and activate a virtual environment:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

After downloading the dataset, run the Day-1 audit:

```bash
python -m src.data --input data/raw/hotel_bookings.csv
```

Run tests:

```bash
pytest
```

## Three-day MVP plan

### Day 1 - Data and baseline readiness

- Validate schema, target, duplicates, and missing values
- Document the prediction point and leakage exclusions
- Produce a compact data-quality report
- Prepare a reproducible modelling table

### Day 2 - Modelling and explainability

- Train Logistic Regression baseline
- Train XGBoost candidate
- Compare Precision, Recall, F1, ROC-AUC, and PR-AUC
- Perform threshold analysis
- Add SHAP explanations

### Day 3 - Productization

- Expose predictions through FastAPI
- Add input validation and logging
- Add unit tests
- Package with Docker
- Document sample requests and limitations

## Responsible use

This is a portfolio and learning project, not a live hotel decision system. Data is historical and anonymized. Model outputs must not be used for automated customer treatment without validation, monitoring, fairness analysis, and human oversight.

## Citation

Antonio, N., Almeida, A., & Nunes, L. (2019). Hotel booking demand datasets. *Data in Brief, 22*, 41-49. https://doi.org/10.1016/j.dib.2018.11.126

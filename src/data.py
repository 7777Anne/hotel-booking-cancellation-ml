"""Load and audit the Hotel Booking Demand dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

TARGET = "is_canceled"

REQUIRED_COLUMNS = {
    "hotel",
    TARGET,
    "lead_time",
    "arrival_date_year",
    "arrival_date_month",
    "arrival_date_day_of_month",
    "stays_in_weekend_nights",
    "stays_in_week_nights",
    "adults",
    "children",
    "babies",
    "market_segment",
    "deposit_type",
    "customer_type",
    "adr",
}

OUTCOME_LEAKAGE_COLUMNS = {
    "reservation_status",
    "reservation_status_date",
}

POTENTIAL_POST_BOOKING_COLUMNS = {
    "assigned_room_type",
    "booking_changes",
    "days_in_waiting_list",
}

EXCLUDED_COLUMNS = OUTCOME_LEAKAGE_COLUMNS | POTENTIAL_POST_BOOKING_COLUMNS


def load_data(path: str | Path) -> pd.DataFrame:
    """Load a CSV and fail clearly when the file is unavailable or empty."""
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {csv_path}. "
            "Download hotel_bookings.csv and place it in data/raw/."
        )

    frame = pd.read_csv(csv_path)
    if frame.empty:
        raise ValueError("The dataset is empty.")

    return frame


def validate_schema(frame: pd.DataFrame) -> None:
    """Validate essential fields and the binary target."""
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    target_values = set(frame[TARGET].dropna().unique())
    if not target_values <= {0, 1}:
        raise ValueError(
            f"{TARGET} must be binary, but found values: {sorted(target_values)}"
        )


def create_modelling_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with known leakage and post-booking fields removed."""
    validate_schema(frame)
    columns_to_drop = sorted(EXCLUDED_COLUMNS.intersection(frame.columns))
    return frame.drop(columns=columns_to_drop).copy()


def build_audit_report(frame: pd.DataFrame) -> dict[str, object]:
    """Create a compact, serializable data-quality summary."""
    validate_schema(frame)
    missing = frame.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False)

    return {
        "rows": int(frame.shape[0]),
        "columns": int(frame.shape[1]),
        "duplicate_rows": int(frame.duplicated().sum()),
        "cancellation_rate": float(frame[TARGET].mean()),
        "target_counts": {
            str(key): int(value)
            for key, value in frame[TARGET].value_counts(dropna=False).items()
        },
        "missing_values": {
            str(key): int(value) for key, value in missing.items()
        },
        "excluded_columns_present": sorted(
            EXCLUDED_COLUMNS.intersection(frame.columns)
        ),
    }


def print_audit(report: dict[str, object]) -> None:
    """Print a readable Day-1 audit."""
    print("Hotel Booking Demand - Data Audit")
    print(f"Rows: {report['rows']:,}")
    print(f"Columns: {report['columns']:,}")
    print(f"Duplicate rows: {report['duplicate_rows']:,}")
    print(f"Cancellation rate: {report['cancellation_rate']:.2%}")
    print(f"Target counts: {report['target_counts']}")
    print(f"Missing values: {report['missing_values']}")
    print(f"Excluded columns present: {report['excluded_columns_present']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit hotel booking data.")
    parser.add_argument(
        "--input",
        default="data/raw/hotel_bookings.csv",
        help="Path to hotel_bookings.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = load_data(args.input)
    report = build_audit_report(frame)
    print_audit(report)

    modelling_table = create_modelling_table(frame)
    print(f"Model-ready columns after exclusions: {modelling_table.shape[1]}")


if __name__ == "__main__":
    main()

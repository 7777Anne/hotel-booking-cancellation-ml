import pandas as pd
import pytest

from src.data import (
    EXCLUDED_COLUMNS,
    build_audit_report,
    create_modelling_table,
    validate_schema,
)


def sample_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "hotel": ["City Hotel", "Resort Hotel"],
            "is_canceled": [1, 0],
            "lead_time": [120, 14],
            "arrival_date_year": [2016, 2017],
            "arrival_date_month": ["July", "March"],
            "arrival_date_day_of_month": [1, 15],
            "stays_in_weekend_nights": [1, 0],
            "stays_in_week_nights": [2, 3],
            "adults": [2, 1],
            "children": [0.0, 1.0],
            "babies": [0, 0],
            "market_segment": ["Online TA", "Direct"],
            "deposit_type": ["No Deposit", "No Deposit"],
            "customer_type": ["Transient", "Transient"],
            "adr": [90.0, 110.0],
            "reservation_status": ["Canceled", "Check-Out"],
            "reservation_status_date": ["2016-06-01", "2017-03-18"],
            "assigned_room_type": ["A", "B"],
            "booking_changes": [1, 0],
            "days_in_waiting_list": [0, 0],
        }
    )


def test_validate_schema_accepts_valid_data() -> None:
    validate_schema(sample_frame())


def test_validate_schema_rejects_missing_target() -> None:
    frame = sample_frame().drop(columns=["is_canceled"])
    with pytest.raises(ValueError, match="Missing required columns"):
        validate_schema(frame)


def test_create_modelling_table_removes_excluded_columns() -> None:
    result = create_modelling_table(sample_frame())
    assert EXCLUDED_COLUMNS.isdisjoint(result.columns)
    assert "is_canceled" in result.columns


def test_audit_report_calculates_cancellation_rate() -> None:
    report = build_audit_report(sample_frame())
    assert report["rows"] == 2
    assert report["cancellation_rate"] == pytest.approx(0.5)

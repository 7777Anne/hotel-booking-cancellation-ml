import pandas as pd
import pytest
from src.data import load_data, validate_schema
from tests.test_train import sample_data


def test_yes_no_target_normalization(tmp_path):
    csv = tmp_path / "data.csv"
    pd.DataFrame({"is_canceled": ["yes", "no", "YES"]}).to_csv(csv, index=False)
    assert load_data(csv)["is_canceled"].tolist() == [1, 0, 1]


def test_missing_labels_rejected():
    frame = sample_data(40)
    frame.loc[0, "is_canceled"] = None
    with pytest.raises(ValueError, match="Missing cancellation labels"):
        validate_schema(frame)

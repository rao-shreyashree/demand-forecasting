import pandas as pd
import pytest
from src.schema import DATE_COL, STORE_COL, ITEM_COL, SALES_COL
from src.data import clean_data, time_split

@pytest.fixture
def sample_raw_data():
    dates = pd.date_range("2023-01-01", periods=60, freq="D")
    data = []
    for store in [1, 2]:
        for item in [1, 2]:
            for d in dates:
                data.append({DATE_COL: d, STORE_COL: store, ITEM_COL: item, SALES_COL: 10})
    # Add a malformed row
    data.append({DATE_COL: "2023-01-01", STORE_COL: 1, ITEM_COL: 1, SALES_COL: "invalid"})
    return pd.DataFrame(data)

def test_clean_data_drops_invalid_sales_and_sorts(sample_raw_data):
    cleaned = clean_data(sample_raw_data)
    
    # Ensure invalid sales row was removed rather than zero-filled
    assert len(cleaned) == 240
    assert cleaned[SALES_COL].isnull().sum() == 0
    
    # Ensure sorting order (store, item, date)
    is_sorted = (cleaned == cleaned.sort_values(by=[STORE_COL, ITEM_COL, DATE_COL])).all().all()
    assert is_sorted

def test_time_split_28_day_horizon_and_no_leakage(sample_raw_data):
    cleaned = clean_data(sample_raw_data)
    train, val = time_split(cleaned, horizon=28)
    
    # Check 28-day validation period
    val_days = (val[DATE_COL].max() - val[DATE_COL].min()).days + 1
    assert val_days == 28
    
    # Check no data leakage between train and val
    assert train[DATE_COL].max() < val[DATE_COL].min()
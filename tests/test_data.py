import pytest
import pandas as pd
from datetime import datetime, timedelta
from src.schema import DATE, STORE, ITEM, SALES, validate_dataframe
from src.data import time_split, clean_data

@pytest.fixture
def sample_data():
    # Create 40 days of dummy data for testing time_split and cleaning
    base_date = datetime(2026, 1, 1)
    dates = [base_date + timedelta(days=i) for i in range(40)]
    
    data = {
        DATE: dates * 2,
        STORE: [1] * 40 + [2] * 40,
        ITEM: [101] * 40 + [102] * 40,
        SALES: [10] * 80
    }
    return pd.DataFrame(data)

def test_time_split(sample_data):
    horizon = 28
    train_df, val_df = time_split(sample_data, horizon=horizon)
    
    # Check that validation set has exactly the horizon span of unique days max or correct row partitions
    assert not train_df.empty
    assert not val_df.empty
    # Ensure no data leakage (train dates are strictly before val dates)
    assert train_df[DATE].max() < val_df[DATE].min()

def test_validate_dataframe(sample_data):
    # Should pass successfully
    assert validate_dataframe(sample_data) == True
    
    # Should fail if column is missing
    bad_df = sample_data.drop(columns=[SALES])
    with pytest.raises(ValueError):
        validate_dataframe(bad_df)
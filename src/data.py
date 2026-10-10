import pandas as pd
from src.schema import DATE, STORE, ITEM, SALES, validate_dataframe

def load_data(filepath: str) -> pd.DataFrame:
    """Loads raw CSV data and parses dates."""
    df = pd.read_csv(filepath)
    df[DATE] = pd.to_datetime(df[DATE])
    return df

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Cleans data: sorts chronologically, fills missing sales, and validates schema."""
    df = df.copy()
    
    # Ensure correct data types
    df[DATE] = pd.to_datetime(df[DATE])
    df[SALES] = pd.to_numeric(df[SALES], errors='coerce').fillna(0)
    
    # Sort chronologically by store, item, date
    df = df.sort_values(by=[STORE, ITEM, DATE]).reset_index(drop=True)
    
    # Validate against schema contract
    validate_dataframe(df)
    
    return df

def time_split(df: pd.DataFrame, horizon: int = 28):
    """
    Splits dataframe temporally into train and validation sets, 
    reserving the last 'horizon' days for validation without shuffling.
    """
    df = df.sort_values(DATE)
    cutoff_date = df[DATE].max() - pd.Timedelta(days=horizon)
    
    train_df = df[df[DATE] <= cutoff_date].copy()
    val_df = df[df[DATE] > cutoff_date].copy()
    
    return train_df, val_df
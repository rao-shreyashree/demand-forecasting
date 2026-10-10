import pandas as pd
from src.schema import DATE_COL, STORE_COL, ITEM_COL, SALES_COL, validate_schema

def load_data(filepath: str) -> pd.DataFrame:
    """Loads raw CSV data."""
    return pd.read_csv(filepath)

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Cleans dataframe, handles data types, drops invalid sales rows, and sorts."""
    df = df.copy()
    
    # Convert date to datetime
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    
    # Ensure store and item are integer types
    df[STORE_COL] = df[STORE_COL].astype(int)
    df[ITEM_COL] = df[ITEM_COL].astype(int)
    
    # Coerce sales to numeric and drop invalid/missing rows (do NOT fillna with 0)
    df[SALES_COL] = pd.to_numeric(df[SALES_COL], errors="coerce")
    df = df.dropna(subset=[SALES_COL])
    
    # Validate schema contract
    validate_schema(df)
    
    # Sort explicitly by (store, item, date)
    df = df.sort_values(by=[STORE_COL, ITEM_COL, DATE_COL]).reset_index(drop=True)
    return df

def time_split(df: pd.DataFrame, horizon: int = 28) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Splits dataframe into train and validation sets preserving (store, item, date) order."""
    max_date = df[DATE_COL].max()
    cutoff_date = max_date - pd.Timedelta(days=horizon)
    
    train = df[df[DATE_COL] <= cutoff_date].sort_values(by=[STORE_COL, ITEM_COL, DATE_COL]).reset_index(drop=True)
    val = df[df[DATE_COL] > cutoff_date].sort_values(by=[STORE_COL, ITEM_COL, DATE_COL]).reset_index(drop=True)
    
    return train, val
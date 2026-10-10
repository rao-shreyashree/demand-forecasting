import pandas as pd

# Column Name Constants
DATE = "date"
STORE = "store"
ITEM = "item"
SALES = "sales"
Y_HAT = "y_hat"
MODEL = "model"

REQUIRED_COLUMNS = [DATE, STORE, ITEM, SALES]

def validate_dataframe(df: pd.DataFrame) -> bool:
    """
    Validates that the dataframe contains all required columns 
    and proper data types. Raises ValueError if validation fails.
    """
    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
            
    # Ensure date is datetime type
    if not pd.api.types.is_datetime64_any_dtype(df[DATE]):
        raise TypeError(f"Column '{DATE}' must be of datetime type.")
        
    return True
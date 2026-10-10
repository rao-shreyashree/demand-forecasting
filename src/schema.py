import pandas as pd

# Schema Constants
DATE_COL = "date"
STORE_COL = "store"
ITEM_COL = "item"
SALES_COL = "sales"

REQUIRED_COLUMNS = [DATE_COL, STORE_COL, ITEM_COL, SALES_COL]

# Forecast output / validation errors (used by inventory)
Y_HAT_COL = "y_hat"
MODEL_COL = "model"
ERROR_COL = "error"  # daily forecast error (actual - y_hat), validation window

# Simulated inventory table
LEAD_TIME_COL = "lead_time_days"
UNIT_COST_COL = "unit_cost"
ORDER_COST_COL = "order_cost"
HOLDING_RATE_COL = "holding_cost_rate"
STOCK_ON_HAND_COL = "stock_on_hand"

# Inventory plan output
SAFETY_STOCK_COL = "safety_stock"
REORDER_POINT_COL = "reorder_point"
EOQ_COL = "eoq"
FLAG_COL = "flag"
ORDER_QTY_COL = "order_qty"


def validate_schema(df: pd.DataFrame) -> None:
    """Validates dataframe schema and data types."""
    # Check required columns
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Check date type
    if not pd.api.types.is_datetime64_any_dtype(df[DATE_COL]):
        raise TypeError(f"Column '{DATE_COL}' must be datetime type.")
        
    # Check store and item integer types
    if not pd.api.types.is_integer_dtype(df[STORE_COL]):
        raise TypeError(f"Column '{STORE_COL}' must be integer type.")
    if not pd.api.types.is_integer_dtype(df[ITEM_COL]):
        raise TypeError(f"Column '{ITEM_COL}' must be integer type.")
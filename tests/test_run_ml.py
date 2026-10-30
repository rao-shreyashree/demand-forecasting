import pandas as pd

from scripts.run_ml import run
from src.data import clean_data, load_data, time_split
from src.ml_models import forecast_lightgbm
from src.schema import DATE_COL, ITEM_COL, MODEL_COL, SALES_COL, STORE_COL, Y_HAT_COL

FAST = {"n_estimators": 20, "num_leaves": 8, "min_child_samples": 5, "learning_rate": 0.2}


def _raw(n_days=150):
    dates = pd.date_range("2017-01-01", periods=n_days)
    rows = [
        (d, s, i, 10 + 3 * s + i + (15 if d.dayofweek >= 5 else 0))
        for s in (1, 2) for i in (1, 2) for d in dates
    ]
    return pd.DataFrame(rows, columns=[DATE_COL, STORE_COL, ITEM_COL, SALES_COL])


def test_run_writes_contract_csv_from_train_only(tmp_path):
    data_path = tmp_path / "train.csv"
    out_path = tmp_path / "processed" / "forecasts_ml.csv"      # folder doesn't exist yet
    _raw().to_csv(data_path, index=False)

    run(str(data_path), str(out_path), horizon=28, params=FAST)

    saved = pd.read_csv(out_path, parse_dates=[DATE_COL])
    assert list(saved.columns) == [DATE_COL, STORE_COL, ITEM_COL, Y_HAT_COL, MODEL_COL]
    assert len(saved) == 4 * 28
    assert (saved[MODEL_COL] == "lightgbm").all()

    train, valid = time_split(clean_data(load_data(str(data_path))), horizon=28)
    assert saved[DATE_COL].min() == train[DATE_COL].max() + pd.Timedelta(days=1)
    assert saved[DATE_COL].max() == valid[DATE_COL].max()
    # identical to forecasting from the train split alone, so no validation data was used
    expected = forecast_lightgbm(train, horizon=28, params=FAST)
    pd.testing.assert_frame_equal(saved, expected)
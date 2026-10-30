import pandas as pd

from src.classical_models import window_bounds
from src.data import time_split


def _df(days=120):
    dates = pd.date_range("2017-01-01", periods=days)
    return pd.DataFrame({"date": dates, "store": 1, "item": 1, "sales": 10})


def test_windows_are_chronological_and_non_overlapping():
    b = window_bounds(_df(), horizon=28, n_windows=3)
    assert len(b) == 3
    for (s1, e1), (s2, e2) in zip(b, b[1:]):
        assert e1 < s2
    for s, e in b:
        assert (e - s).days + 1 == 28


def test_latest_window_matches_time_split():
    df = _df()
    _, val = time_split(df, horizon=28)
    start, end = window_bounds(df, horizon=28, n_windows=3)[-1]
    assert start == val["date"].min() and end == val["date"].max()


def test_stops_when_not_enough_history():
    assert len(window_bounds(_df(days=60), horizon=28, n_windows=3)) == 2
# Setup and implementation assumptions

This guide supports [issue #57](https://github.com/rao-shreyashree/demand-forecasting/issues/57).
It describes main at `2f8311403de181b3c47a766dfd83179353b4b126` and dashboard
PR #72 at `ceb53433d785066e6b74e10caf8c8fc358606a51`, inspected on 7 November
2026. #72 was open, not merged. Start with the [README](../README.md) for
installation and pipeline commands.

## Environment

Python 3.10 is the CI reference version. Install the requirements in an isolated
virtual environment and use `python -m pip` and `python -m pytest` so they use
the selected interpreter. Python's module form (`python -m scripts.run_plan`,
for example) must be run from the repository root so `src` imports and relative
data paths resolve.

Main's requirements do not include Streamlit or Altair. The dashboard branch
adds `streamlit>=1.40` and `altair>=5`. Installing dashboard packages on main
does not provide the dashboard source; use the PR checkout described in README.

Most dependencies are unpinned. Seeded code does not establish identical output
across package versions or operating systems. Dependency pinning/scanning and
two-run reproducibility evidence belong to #61 and #54.

## Data and validation

Sales are daily, keyed by `date, store, item`; the target is `sales`.
The expected source is Kaggle Store Item Demand Forecasting, with 500 series.
Do not substitute Kaggle's test CSV: it lacks the sales actuals needed for
validation, model selection and metrics.

`clean_data` parses dates, casts IDs to integers, coerces sales to numeric,
drops rows with invalid/missing sales and sorts by store/item/date.
`validate_schema` checks required columns and date/ID types; it does not
reject all negative sales, duplicate keys or missing dates. Supply valid,
nonnegative, complete daily data. The LightGBM checks additionally reject
duplicate keys, gaps, unequal final dates and missing sales. Dropping invalid
sales may create gaps that must be resolved before using that model.
The classical runner reads the CSV directly with parsed dates.

`time_split(..., horizon=28)` uses a global date cutoff, holding out the last
28 calendar days without shuffling. Complete daily series are assumed.
The default runners forecast that historical holdout, not a new live period.

## Forecasting and evaluation assumptions

- Baselines repeat the last value, seasonal values, or a trailing mean.
- SARIMA uses (1,1,1) × (1,1,1,7); daily gaps in its series are interpolated.
  Prophet enables weekly/yearly seasonality, with daily seasonality disabled.
  Forecasts are clipped at zero. The runner logs individual fit failures and
  continues, so inspect coverage and warnings rather than assuming every fit succeeded.
- Single-window classical selection ranks series using training sales.
  The rolling classical run fixes its top-N cohort using data before the
  earliest validation window.
- LightGBM is one global model using calendar features, store/item categories,
  lagged sales and shifted rolling features. It accepts training data only;
  horizon must be at most the minimum lag (28 days). The seed is 42, with
  deterministic and force-row-wise options. There is no XGBoost runner.
- Rolling evaluation uses three non-overlapping 28-day blocks counted backward
  from the last date, refitting before each origin. The notebook restricts
  baselines to the 10 classical series and compares naive, seasonal-naive,
  moving-average 7/28, SARIMA and Prophet. It does not evaluate LightGBM.
- WAPE = sum absolute errors / sum absolute actuals (fraction; 0.1 = 10%).
  WAPE is NaN for zero total actual sales. MAPE skips zero-actual rows.
  Bias = mean(forecast − actual): positive means over-forecast.
  Inventory's `forecast_errors` uses actual − forecast, the opposite sign.
- `run_backtest` averages each model's per-window metrics and reports row and
  window counts. The notebook's saved comparison CSV drops those counts;
  inspect its in-memory tables to verify equal coverage.
- Model selection uses lowest per-series WAPE, with model-name tie-breaking
  and undefined WAPE last. The current plan runner's candidates are baselines
  plus classical forecasts, even if a LightGBM CSV exists. It estimates errors
  and selects winners on the same validation window. Avoid presenting the
  selected plan as performance on an independent test set.

## Notebooks and backtesting

Jupyter is optional and absent from requirements. In the activated environment,
from the repository root:

```sh
python -m pip install jupyterlab
```

The notebooks read files via `../data/...` but import `src` from the repository
root. Set the root on PYTHONPATH before launching Jupyter so both resolve.

Windows PowerShell:

```powershell
$env:PYTHONPATH = (Get-Location).Path
python -m jupyterlab
```

macOS/Linux bash or zsh:

```sh
export PYTHONPATH="$PWD"
python -m jupyterlab
```

Open `notebooks/backtest_shryes.ipynb` and select the environment's Python
kernel. Its working directory should be `notebooks/`. Run cells in order.
First create `data/processed/forecasts_classical_backtest.csv` from the root
using the README's three-window command. Keep the separate single-window
`forecasts_classical.csv` for plan generation and dashboard use.
The EDA notebook is exploratory and is not required before `scripts.run_ml`,
which builds its own features.

## Inventory assumptions

`simulate_inventory(train)` uses NumPy's default random generator, seed 42,
and sorted store/item keys. These are artificial parameters, not measured
warehouse or supplier data:

| Parameter | Generation |
| --- | --- |
| Lead time | Uniform integer from 2 through 14 days |
| Unit cost | Uniform 5–50, rounded to 2 decimals |
| Order cost | Uniform 20–100, rounded to 2 decimals |
| Annual holding rate | Uniform 0.15–0.30, rounded to 3 decimals |
| Stock on hand | Rounded training mean daily sales × uniform 2–30 days of cover |

Costs are illustrative values; the code does not declare a currency.
The plan uses mean nonnegative forecast demand, daily error standard deviation,
a normal cycle-service approximation at 95%, and 365 days per year:

- Safety stock = z(0.95) × daily error standard deviation × sqrt(lead-time days).
- Reorder point (ROP) = mean daily forecast demand × lead time + safety stock.
- EOQ = sqrt(2 × annual demand × order cost / annual holding cost per unit);
  annual holding cost per unit = unit cost × annual holding rate.
- If stock ≤ ROP, order ceil(ROP + EOQ − stock); otherwise order zero.
  A positive order is flagged `stockout_risk`. Stock above ROP + EOQ is
  `overstock`; other cases are `ok` (including zero demand/zero stock).
- A series with fewer than two error points receives the median usable
  error standard deviation from other series; no usable errors raises an error.

The formula assumes fixed lead time per series and a square-root-of-time
error approximation. It does not model supplier uncertainty, stock in transit,
pack sizes, capacity limits or purchasing execution. Standard deviation measures
error spread rather than systematic bias and can understate risk for a biased
model. Treat the plan as a demonstration.

## Dashboard inputs and caveats

The dashboard is a viewer of saved CSVs. Its forecast loader combines
`forecasts_*.csv` from `data/processed/`, excludes stems containing
`backtest`, and removes duplicate store/item/date/model keys. Keep unrelated
or stale forecast files outside that folder; keep dates aligned across models.
Common-series filtering checks store/item coverage, not common dates.

Metrics join forecasts to available actuals by store/item/date. Forecasts beyond
the raw sales history will not have validation metrics. Model comparison is a
quick view of the loaded forecast window, separate from rolling-origin backtests.
`chosen_model` in the plan determines the displayed plan model; legacy plans
without it use a lowest-WAPE fallback explicitly labelled by the dashboard.

The reorder table contains only rows with positive recommended order quantities.
Its filters consequently list flags with orders, rather than every stock flag.
Stock and cost displays refer to simulated inventory. Restart the app after
regenerating CSVs because data and metrics are cached.

## Troubleshooting

| Symptom | Check |
| --- | --- |
| `No module named src` | Use the documented module command from repo root; for notebooks set PYTHONPATH as above. |
| Missing raw data | Restore/download train.csv at data/raw/train.csv. |
| Missing classical forecast for plan | Run the single-window classical command first; keep the exact default output path. |
| Missing dashboard app or Streamlit | Main lacks the unmerged dashboard; use the #72 branch and its requirements. |
| Stale dashboard values | Stop/restart Streamlit after replacing input CSVs. |
| ML rejects gaps or endpoints | Check complete daily series, unique keys and matching final dates. |
| Different model scores | Compare date windows, store/item cohort and row counts before ranking models. |

## Requirement references and documentation boundary

Relevant repository requirements are
[#34](https://github.com/rao-shreyashree/demand-forecasting/issues/34) (simulation),
[#37](https://github.com/rao-shreyashree/demand-forecasting/issues/37) (split),
[#39](https://github.com/rao-shreyashree/demand-forecasting/issues/39) (top-N classical),
[#40](https://github.com/rao-shreyashree/demand-forecasting/issues/40) (ML),
[#41](https://github.com/rao-shreyashree/demand-forecasting/issues/41) (forecast contract),
[#42](https://github.com/rao-shreyashree/demand-forecasting/issues/42) (backtest),
and [#48](https://github.com/rao-shreyashree/demand-forecasting/issues/48) (selection).

No SRS, SAD or STP files were present in the inspected main tree or attached to
the referenced chat. This guide records implementation-backed assumptions; it
does not claim a line-by-line comparison with those unavailable documents.
Formal deviations belong to #69 and final documentation/traceability to #70.


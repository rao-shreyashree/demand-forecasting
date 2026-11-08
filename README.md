# Demand Forecasting & Inventory Planning System

Forecast daily store-item sales, compare models, and produce an inventory
reorder plan. The dataset is Kaggle Store Item Demand Forecasting:
10 stores × 50 items = 500 daily series over five years.

**Inventory is simulated.** Stock on hand, lead times and costs are generated
for demonstration; recommendations are not based on a live inventory system.

## Current availability

The forecasting, evaluation, inventory and Streamlit dashboard modules are on
`main`. [PR #72](https://github.com/rao-shreyashree/demand-forecasting/pull/72)
has been merged; use the main-branch checkout for the entire workflow below.

## Prerequisites and installation

Use Git and Python **3.10** (the version configured in CI), with pip and venv.
Internet access is needed to install dependencies. Run commands from the
repository root unless a step says otherwise.

```sh
git clone https://github.com/rao-shreyashree/demand-forecasting.git
cd demand-forecasting
python -m venv .venv
```

Activate the environment using the command for your shell:

| Shell | Command |
| --- | --- |
| Windows PowerShell | `.\.venv\Scripts\Activate.ps1` |
| Windows Command Prompt | `.venv\Scripts\activate.bat` |
| macOS/Linux bash or zsh | `source .venv/bin/activate` |

If PowerShell blocks activation, use Command Prompt, or run the environment's
Python directly (`.\.venv\Scripts\python.exe`) in place of `python`.
On systems where Python 3 is named `python3`, use it to create the environment.

```sh
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pytest
```

The requirements include pandas, NumPy, scikit-learn, Prophet, statsmodels,
LightGBM, pytest, Streamlit and Altair. Only LightGBM is exactly pinned on the inspected main
branch; a fully locked, repeatable environment is tracked separately in
[#54](https://github.com/rao-shreyashree/demand-forecasting/issues/54) and
[#61](https://github.com/rao-shreyashree/demand-forecasting/issues/61).
Notebook tools are optional; see [setup and assumptions](docs/setup-and-assumptions.md).

## Data

The inspected repository already tracks `data/raw/train.csv` and sample
processed outputs. If the raw file is missing, download **train.csv** from
[Kaggle](https://www.kaggle.com/competitions/demand-forecasting-kernels-only/data)
(account/competition access may be required) and place it at
`data/raw/train.csv`.

Input columns: `date, store, item, sales`. Use daily data with unique
store/item/date keys, nonnegative numeric sales, and aligned series endpoints.
The cleaning helper converts types, drops invalid/missing sales and sorts rows;
it does not repair missing dates or enforce every data-quality condition.
See the [detailed caveats](docs/setup-and-assumptions.md#data-and-validation).

## Data → forecast → metrics → inventory → dashboard

### 1. Generate forecasts

These commands hold out the last **28 days** for validation. They regenerate
historical validation forecasts, rather than forecasting beyond the end of
the complete dataset.

```sh
python -m src.classical_models --top-n 10
python -m scripts.run_ml
```

Outputs are `data/processed/forecasts_classical.csv` (SARIMA and Prophet on
the top 10 store-item series) and `data/processed/forecasts_ml.csv`
(LightGBM on all 500 series for the supplied dataset).
Classical fitting can take time; a quick dashboard demo can use the committed
CSV files instead of retraining.

Every model CSV must have exactly these columns, in this order:

```text
date,store,item,y_hat,model
```

### 2. Evaluate

The classical command prints WAPE and RMSE for its validation window.
For a three-window rolling-origin evaluation, first regenerate classical
forecasts at each origin:

```sh
python -m src.classical_models --top-n 10 --backtest-windows 3
```

Then run `notebooks/backtest_shryes.ipynb` from top to bottom using the
[notebook setup](docs/setup-and-assumptions.md#notebooks-and-backtesting).
It compares SARIMA, Prophet and four baselines on the same 10 series and
writes `data/processed/model_comparison_shryes.csv`.
LightGBM is not included in that notebook's comparison.

WAPE is primary; RMSE, MAPE and bias are secondary. Metrics are averaged over
three 28-day windows. Inspect `comparison_table` and `detailed` in the
notebook for `n_windows` and `n_rows`; its saved CSV omits those counts.
Do not compare scores from different cohorts/windows as if they were equivalent.

### 3. Generate the inventory plan

After the single-window classical forecast file exists:

```sh
python -m scripts.run_plan
```

This cleans/splits the raw data, generates naive, seasonal-naive and moving-average
candidates, loads `forecasts_classical.csv`, selects the lowest-WAPE model
per store-item series, simulates inventory with seed 42, prints summaries,
and writes `data/processed/inventory_plan.csv`.

**The current runner does not load `forecasts_ml.csv`.** Running LightGBM
does not make it a candidate for this plan. Selection and error estimates use
the same held-out window, so the plan demonstrates historical selection and
inventory calculations; it is not an independent test of future plan performance.

### 4. Run the dashboard

Use the main-branch checkout created during installation. If you already have
an older main checkout, update it first (commit or stash any local edits before
switching branches):

```sh
git switch main
git pull --ff-only origin main
```

Activate its virtual environment as above, then install the current requirements
and start the dashboard from the repository root:

```sh
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m streamlit run dashboard/app.py
```

Open the local URL printed by Streamlit (normally
[localhost:8501](http://localhost:8501)); stop it with Ctrl+C.

The dashboard reads the raw sales, `data/processed/inventory_plan.csv`, and
`data/processed/forecasts_*.csv` excluding names containing `backtest`.
It displays forecast vs actual, model comparison, stock flags and a reorder
table with store/item/flag filters and CSV exports. It reads saved outputs;
it does not train models or rebuild the plan.

Use the committed CSVs for a quick demo, or regenerate them in this checkout
using steps 1–3. Restart Streamlit after changing input CSVs to clear cached
data. Tick **Show render time** to inspect the app's timer; this alone is not
evidence that the full browser load/filter response meets the ≤3-second target.

## Project structure

```text
.github/workflows/ci.yml          Python 3.10 test workflow
data/raw/train.csv               Historical daily sales
data/processed/                  Saved forecasts, comparison and inventory plan
data/forecasts/                  Placeholder directory; runners use processed/
notebooks/
  eda_features_shryes.ipynb       EDA and feature exploration
  backtest_shryes.ipynb           Three-window classical/baseline comparison
scripts/
  run_ml.py                      LightGBM validation forecast runner
  run_plan.py                    Baseline/classical selection and plan runner
src/
  schema.py, data.py              Column contracts, loading, cleaning, time split
  features.py, ml_models.py       Lagged features and LightGBM
  baselines.py                   Naive, seasonal-naive, moving averages
  classical_models.py            SARIMA/Prophet runners and rolling refits
  metrics.py, backtest.py         Accuracy metrics and rolling-origin evaluation
  inventory.py, selection.py     Simulated inventory, policy and model selection
tests/                           Unit tests
docs/setup-and-assumptions.md     Setup details, assumptions and troubleshooting
requirements.txt                 Python dependencies
```

The dashboard files `dashboard/app.py`, `dashboard/helpers.py`,
`.streamlit/config.toml` and `tests/test_dashboard_helpers.py` are included on main.

## Assumptions and limitations

- ML uses lags ≥28 days and rolling features shifted by 28 days to avoid
  using validation sales. The implemented ML forecaster is LightGBM; XGBoost
  is not implemented.
- Classical models use a fixed top-10 store-item cohort by sales, not all 500
  series. The rolling run chooses that cohort before the earliest window.
- Inventory uses simulated stock, 2–14 day lead times and simulated costs;
  default service level is 95%. Full ranges and policy formulas are in the
  [assumptions document](docs/setup-and-assumptions.md#inventory-assumptions).
- Dashboard model comparison filters common series, but does not enforce
  identical date coverage. Keep input forecast windows aligned.
- Runtime ≤15 minutes, dashboard response ≤3 seconds and identical outputs
  across environments require separate measurement; they are not guaranteed here.
- Formal SRS/SAD/STP deviation reconciliation remains
  [#69](https://github.com/rao-shreyashree/demand-forecasting/issues/69).



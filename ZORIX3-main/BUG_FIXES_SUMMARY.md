# NIFTY Lab corrections — intraday alignment and historical forecasting

Scope: `zorix/nifty_lab/` only. Standard Zorix, the Yahoo Finance integration,
models, hyper-parameters, features, signals and deployment settings are untouched.

## 1. Intraday targets (`features.py`, `intraday_model.py`, `historical_sim.py`)

`build_intraday_targets` now looks up the bar stamped **exactly `t + h minutes`**
(vectorised; no row offsets). A target is **valid only if** that bar exists **and**
belongs to the same trading session as `t`.

* Sessions are split wherever consecutive bars are more than
  `DEFAULT_SESSION_GAP_MINUTES` (5) apart, so overnight gaps *and* separate sessions
  on the same date are boundaries. A gap of up to 5 minutes is treated as missing
  bars inside one session. An explicit `session_id` column, if supplied, overrides this.
* Invalid targets are never repaired with another timestamp: `target_ts_*` is `NaT`,
  and `target_close_*`, `fwd_ret_*`, `fwd_dir_*` are `NaN` (direction used to be `0`).
* New columns `target_ts_{h}m` / `target_close_{h}m` are produced by the same lookup as
  the return, so target time, target price and return describe one observation.
* `bars=` lets callers look targets up in the raw minute frame (the feature frame loses
  rows where RSI is undefined, e.g. flat prices, which made real bars look "missing").
* Forecast results now carry `forecast_origin`, `origin_price`, `target_timestamp` and
  `actual_price` (all from the same row); the UI shows the target time.

Evidence on the project's own data: the old code produced 12 valid 30-minute targets
that bridged the real 15:45→16:01 break on 2023-09-15.

## 2. Historical one-month-ahead forecast (`historical_sim.py`, `ui.py`)

`run_nextday_one_month_ahead` (cutoff relative to the last dataset date; scored a month
of next-day predictions) is replaced by `run_historical_forecast(daily, cfg, target_date)`:

1. The user selects a target trading date (must be a session in the data).
2. Cutoff = target − 1 calendar month (`pd.DateOffset`, so Mar 31 → Feb 28/29). Sessions dated
   on or before it are observable; the last such session is the forecast origin.
3. Features are built from history truncated at the cutoff. Direct `h`-session-ahead
   random forest (same hyper-parameters as the next-day model), `h` = sessions between the
   origin and the target. No scaler is used (the forest is scale-invariant).
4. A training row is used only if its label's outcome session (`t+h`) is ≤ cutoff; rows
   whose outcome was not yet observable are dropped, and this is asserted
   (`build_horizon_training_set`). `train_end < test_start` alone is not relied on.
5. The prediction is compared with the actual close of exactly the selected session.
   Reported: cutoff, target, predicted, actual, error (abs, %), plus a naive
   "no change" baseline.
6. Explicit `HistoricalForecastError` codes: `NON_TRADING_DATE`, `OUTSIDE_COVERAGE`,
   `INSUFFICIENT_HISTORY`.

Note: `h` is counted from the dataset's trading calendar (session dates, never prices after
the cutoff). A test perturbs every post-cutoff price and checks the forecast is bit-identical.

## 3. Tests (`test_bug_fixes.py`)

Portable pytest suite, deterministic synthetic data (plus read-only checks on the project CSV,
skipped if absent). Run from the project root: `python -m pytest -v`.

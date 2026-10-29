# ZORIX Part 1 — leakage & evaluation fixes (standard yfinance model only)

`zorix/nifty_lab/`, the NIFTY CSV and `test_bug_fixes.py` are untouched (byte-compared against the original ZIP; the NIFTY lab imports nothing from the modules changed here).

## Changes
1. **Overlapping horizons / chronological split** — new `models/splits.py`. A label at row *i* spans *i..i+h*, so `app.py` now drops the last *h* train rows before the test start (`purged_split_points`). Walk-forward (`walk_forward_split`, `run_walk_forward`) and feature-stability use `TimeSeriesSplit(gap=h)`; *h* is inferred from the target name (`Target_5D` → 5). `features/target.py`: targets are now NaN (not a fake HOLD) where the forward window or volatility history is missing.
2. **Regime look-ahead** — `features/regime.py` was fit on the full history, used full-sequence (smoothed) HMM posteriors, and a full-sample vol median. Now: expanding-window refit (first fit after 252 rows, refit every 63), model/label map/vol median use only past rows, HMM uses forward-filtered posteriors, quantile fallback uses an expanding median. Warm-up rows are NaN (previously one-hots were 0.0).
3. **Stacking** — `models/ensemble.py`: OOF folds are purged by *h*; new `fit_stacking`/`predict_stack_new` fit OOF + meta-model + base refits on **train only**; `app.py` scores the holdout with it (previously OOF/meta used all rows incl. the test period).
4. **Model selection on test data** — `models/regression.py`: the "best" regressor is now chosen on a purged validation slice of the training data (`RegressionReport.best()`, no metric arg); test metrics are reporting only. Removed `eval_set=(X_test, y_test)` from the XGBoost classifier/regressor fits.
- Extra bug found: `run_walk_forward` compared `{-1,0,1}` targets with `{0,1,2}` argmax predictions; it now remaps labels.

## Tests (run here)
- `zorix/tests/test_part1_leakage.py` (new): **12/12 passed** — purge/no label overlap, NaN targets, regime causality (GMM + quantile; future perturbation doesn't change past labels), HMM forward filter vs brute force, OOF purge, stack fit on train only, selection unaffected by sabotaged test labels, walk-forward gap/label remap.
- Existing `test_core.py` 14/14 and `test_multi_asset.py` 19/19 passed (run with a throwaway `pytest.approx` shim because pytest isn't installable offline).
- Protected NIFTY files: 11 files compared with the original, 0 changed.

## Not verified
- `hmmlearn`, `xgboost`, `lightgbm`, `streamlit` weren't installed and there was no network: the HMM path is tested only via the filter function with a fake model; `app.py` was only syntax-checked (`py_compile`), not run end-to-end; `test_bug_fixes.py` (NIFTY) wasn't run.
- Behaviour notes: regime features lose ~252 warm-up rows; the headline stacked signal/expected return still refer to the last *labelled* row of the holdout (as before), from models fit on train only. Calibration CV (`calibration.py`) and the backtester were not changed.

---

# ZORIX Part 2A — always-HOLD bug fix (standard yfinance model only)

Scope: `zorix/models/trainer.py`, `zorix/models/ensemble.py`, and a new test
file. `zorix/nifty_lab/`, `data/NIFTY_spot_1m_extended.csv`, and every other
module are untouched — verified byte-for-byte identical (see "Verification").

## 1. Cause (verified, not assumed)

The BUY/HOLD/SELL label (`zorix/features/target.py`) is volatility-adjusted
and HOLD is the plurality class by construction — on realistic synthetic
daily data (5-day horizon, default `K_BUY=K_SELL=0.75`) the true label split
is roughly **SELL 25% / HOLD 53% / BUY 22%**, not a severe imbalance on its
own.

The actual problem is in `zorix/models/trainer.py`'s `ModelTrainer`: Logistic
Regression, Random Forest, XGBoost and LightGBM were all fit **unweighted**.
On the noisy, weak-signal feature set typical of daily price data, an
unweighted multiclass classifier doesn't need to separate the classes to
minimize training loss — predicting HOLD on (almost) every row is already a
low-loss strategy, because HOLD only needs to win a *plurality*, not a
majority, of each row's 3-way argmax. Result: `argmax(predict_proba)`
collapses to HOLD on nearly every test row even though the *mean* per-class
probability is reasonably spread (e.g. ~0.22 / 0.53 / 0.25) — the model is
diffuse rather than genuinely abstaining, but HOLD wins the argmax almost
everywhere anyway.

**Reproduced before the fix** (synthetic near-random-walk data, 5-day
horizon, `zorix.models.trainer.ModelTrainer`, no other code changes):

| Model | True label HOLD share | Predicted-class HOLD share |
|---|---|---|
| Logistic Regression | 51% | **98.8%** |
| Random Forest | 51% | **96.9%** |
| XGBoost | 51% | 50.6% |
| LightGBM | 51% | 43.6% |

Feeding the resulting (averaged) probabilities through
`zorix/signals/signal_engine.decide_signal` with the default `MIN_EDGE=0.12`,
`MIN_CONFIDENCE=0.45` produced **zero BUY and zero SELL** across the test set
— only HOLD and NO TRADE — confirming the end-to-end symptom reported.

The `decide_signal` NO TRADE gate itself was checked and is **not** the
cause: it already returns BUY/SELL/HOLD correctly for probability vectors
that should produce them (`test_buy_when_clear_edge` /
`test_no_trade_when_edge_too_small` in `test_core.py`, both passing before
and after this fix). The bug is upstream, in what probabilities the models
hand to that gate. XGBoost/LightGBM were less prone to the collapse on this
particular synthetic data, but inherit the same imbalance sensitivity in
general and were given the same balanced treatment for consistency.

## 2. Fix

Minimal, targeted: make every classifier in the standard model's training
path **class-balanced**, so the majority label no longer gets a "free ride"
in the loss function.

- `zorix/models/trainer.py`
  - `LogisticRegression(..., class_weight="balanced")`
  - `RandomForestClassifier(..., class_weight="balanced")`
  - `LGBMClassifier(..., class_weight="balanced")`
  - XGBoost has no multiclass `class_weight` kwarg, so a new helper
    `_balanced_sample_weight(y)` computes the sklearn-equivalent per-sample
    weight (`n_samples / (n_classes * n_samples_in_class)`) and passes it as
    `sample_weight` to `clf.fit(...)`.
- `zorix/models/ensemble.py` — `default_base_model_factories` (used by the
  OOF stacking ensemble that `app.py` feeds into the final signal) gets the
  same `class_weight="balanced"` on LogReg / RandomForest / LightGBM. XGBoost's
  factory there is left unchanged: the OOF loop calls one generic
  `model.fit(X_tr, y_tr)` shared by all factories, so per-model
  `sample_weight` plumbing there would be a larger, non-minimal change;
  LogReg + RandomForest are always available and already fixed, which is
  enough to break the collapse (confirmed below).

No thresholds, probability floors, risk filters, or NO TRADE gating were
changed. Nothing forces a BUY or SELL — the gate can and still does return
HOLD or NO TRADE whenever that's genuinely what the (now better-separated)
probabilities indicate.

## 3. Tests added

New file: `zorix/tests/test_signal_balance.py` (7 tests):

- `test_decide_signal_produces_buy` / `_sell` / `_hold_when_hold_genuinely_wins`
  / `_no_trade_when_edge_too_thin` — confirm the gate correctly reaches all
  four outcomes for deterministic probability inputs.
- `test_balanced_sample_weight_matches_sklearn_formula` — unit test for the
  new helper against the closed-form sklearn formula.
- `test_trained_models_do_not_collapse_to_always_hold` — core regression
  test: trains the real `ModelTrainer` on synthetic imbalanced, weak-signal
  data and asserts no model's predicted-class HOLD share exceeds 90% (the bug
  produced 97-99%), and no model collapses to any other single class either.
- `test_end_to_end_signal_distribution_is_not_hold_only` — trains the models,
  averages probabilities, runs every test row through `decide_signal`, and
  asserts BUY or SELL actually appears — the full pipeline, not just one
  function, can reach a directional call.

## 4. Test results

```
$ python -m pytest zorix/tests/test_signal_balance.py -v
7 passed in ~17s

$ python -m pytest zorix/tests/test_core.py -v
14 passed in ~1.2s

$ python -m pytest zorix/tests/test_part1_leakage.py zorix/tests/test_multi_asset.py -v
31 passed in ~4.9s
```

`test_bug_fixes.py` at the project root could not be collected in this
environment (`streamlit` not installed; imported by `zorix/nifty_lab/ui.py`)
— a pre-existing environment limitation, not caused by this change, and that
file is entirely within NIFTY-lab scope regardless.

Post-fix, re-running the same reproduction used to find the bug:

| Model | True label HOLD share | Predicted-class HOLD share (after fix) |
|---|---|---|
| Logistic Regression | 51% | 22.5% |
| Random Forest | 51% | 33.2% |
| XGBoost | 51% | 50.8% |
| LightGBM | 51% | 38.8% |

End-to-end signal distribution across the same test set went from
`{HOLD, NO TRADE}` only to `{NO TRADE, HOLD, BUY, SELL}` all present.

(Synthetic-data numbers are illustrative of the mechanism, not a claim about
real-ticker accuracy — the data is a near-random walk with no real
predictive edge, so low classification accuracy is expected; what matters is
the model is no longer mechanically forced toward HOLD regardless of input
features.)

## 5. Verification — NIFTY scope untouched

```
$ md5sum zorix/nifty_lab/*.py data/NIFTY_spot_1m_extended.csv   # before vs after
(identical — byte-for-byte unchanged, all 10 files)

$ diff -rq <original extracted ZIP> <modified project>
Only in modified: zorix/models/ensemble.py            (content differs)
Only in modified: zorix/models/trainer.py              (content differs)
Only in modified: zorix/tests/test_signal_balance.py   (new file)
(no other files touched; __pycache__/.git/.pytest_cache excluded from comparison)
```

## Files changed in this pass

- `zorix/models/trainer.py` — balanced class weights / sample weights
- `zorix/models/ensemble.py` — balanced class weights in base-model factories
- `zorix/tests/test_signal_balance.py` — new regression tests (added)

---

# Deployment fix — NIFTY CSV not found on Streamlit Cloud

Scope: `zorix/nifty_lab/config.py` only (one file). This is a deliberate,
narrow exception to "don't touch nifty_lab/" from the Part 2A task — that
restriction protected NIFTY Lab from unrelated changes while fixing the
Standard Zorix always-HOLD bug; it does not apply to fixing an actual NIFTY
Lab bug now being reported.

## Cause

`NiftyLabConfig.csv_path` defaulted to the bare relative string
`"data/NIFTY_spot_1m_extended.csv"`, resolved via `os.path.exists(...)`
against the process's **current working directory** at runtime — not the
repo root or the script's own location. The CSV is present and correctly
committed (verified: 16,770,919 bytes, md5 `b0ebb8d3...`, byte-identical to
every prior delivery), but on the deployed Streamlit Cloud app the process's
cwd did not match where `data/` actually lives relative to it, so
`os.path.exists` returned `False` and the app raised
`NiftyLabDataError: NIFTY CSV not found at data/NIFTY_spot_1m_extended.csv`.

`app.py` already works around this same class of problem for its own
imports (`sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))`,
line 23) — `nifty_lab/config.py`'s data path just didn't have the same
treatment.

## Fix

`zorix/nifty_lab/config.py`: `csv_path`'s default is now computed as an
absolute path, derived from `config.py`'s own file location
(`os.path.dirname(os.path.abspath(__file__))`, three levels up to the
project root, then `data/NIFTY_spot_1m_extended.csv`) instead of a bare
relative string. This resolves to the correct file regardless of the
process's working directory at launch. `NiftyLabConfig.csv_path` remains a
plain `str` field, so callers that already pass an explicit path are
unaffected.

## Verification

```
# From the project directory:
Resolved path: /…/ZORIX3-main/data/NIFTY_spot_1m_extended.csv
Exists: True

# From an unrelated cwd (/tmp), simulating a cwd mismatch like the deployed app:
Resolved path: /…/ZORIX3-main/data/NIFTY_spot_1m_extended.csv
Exists: True
```

```
$ python -m pytest zorix/tests/ -q
52 passed

$ python -m pytest test_bug_fixes.py -q
51 passed
```

Diff scope confirmed via md5sum: every file under `zorix/nifty_lab/` is
byte-identical to the prior delivery except `config.py` (the intended
change); `data/NIFTY_spot_1m_extended.csv` is byte-identical
(md5 `b0ebb8d3...`, unchanged).

## If the error persists after redeploying this fix

The fix makes the path launch-directory-independent, which covers the most
likely cause. If Streamlit Cloud still can't find the file after this, the
remaining possibility is that the CSV isn't actually reaching the deployed
container at all (e.g. excluded by `.gitignore`, a push that silently
dropped it, or a Streamlit Cloud/GitHub file-size quirk) — worth confirming
with `git log -- data/NIFTY_spot_1m_extended.csv` on the exact branch your
app deploys from.

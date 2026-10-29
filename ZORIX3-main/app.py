"""
zorix/app.py — Zorix 2.0 Streamlit dashboard.

Multi-asset upgrade (current iteration): Equities, Commodities, Forex, and
Bonds/yields, plus a genuine four-state signal system (BUY/SELL/HOLD/
NO TRADE — HOLD is a real predicted class, NO TRADE is a separate
confidence/edge gate; see signals/signal_engine.py, unchanged in this
pass). Per the scoping instructions for this iteration, F&O/options/
futures/portfolio optimization/LSTM/GRU/stacking ensembles/advanced risk
management were NOT touched here — those already exist from a prior pass
and are left as-is, not expanded.

Options/Futures panels (Sections 15/16/33/34) remain wired to
NullOptionsProvider / NullFuturesProvider. Nothing here fabricates
options, futures, forex, bond, or any other data — unavailable instruments
show "Data unavailable", never a synthesized value.

Run:  streamlit run zorix/app.py     (from the project root, one level above zorix/)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import streamlit as st

from zorix.config.settings import Config, DEFAULT_CONFIG, InstrumentMeta
from zorix.data.market_data import (YahooFinanceProvider, DataUnavailableError,
                                     align_to_trading_calendar, apply_yield_scale)
from zorix.data.news_data import fetch_headlines, score_headlines, score_individual_headlines
from zorix.data.options_data import NullOptionsProvider
from zorix.data.futures_data import NullFuturesProvider
from zorix.features.technical import build_technical_features
from zorix.features.cross_asset import add_relative_strength, add_cross_asset_block
from zorix.features.regime import detect_regime, attach_regime_features
from zorix.features.target import add_bhs_targets, sklearn_label, sklearn_label_names
from zorix.models.trainer import ModelTrainer, XGBOOST_AVAILABLE, LIGHTGBM_AVAILABLE, TENSORFLOW_AVAILABLE
from zorix.models.regression import ReturnRegressionTrainer
from zorix.models.calibration import calibrate_model, evaluate_calibration
from zorix.models.ensemble import (default_base_model_factories, fit_stacking, predict_stack_new)
from zorix.models.splits import purged_split_points
from zorix.models.walkforward import run_walk_forward, oos_predictions_to_signals
from zorix.models.feature_analysis import run_feature_ablation, run_feature_stability
from zorix.models.explainability import explain_prediction, SHAP_AVAILABLE
from zorix.risk.risk_engine import build_risk_assessment
from zorix.backtesting.backtester import Backtester
from zorix.signals.signal_engine import build_signal, model_consensus
from zorix.utils.prediction_log import log_prediction, load_predictions, prediction_log_summary
from zorix.utils.display import equity_instrument_meta, format_instrument_value, describe_asset

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    from sklearn.ensemble import RandomForestClassifier
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

st.set_page_config(page_title="Zorix 2.0", page_icon="\U0001F4C8", layout="wide")

st.markdown("""
<style>
.stApp { background: linear-gradient(135deg, #020617, #07111f, #0f172a); color: white; }
.glass-card { background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.08);
              padding: 20px; border-radius: 16px; margin-bottom: 10px; }
.unavailable-card { background: rgba(255,255,255,0.03); border: 1px dashed rgba(255,255,255,0.15);
              padding: 20px; border-radius: 16px; color: #94a3b8; }
</style>
""", unsafe_allow_html=True)

# =========================================================
# TOP-LEVEL MODE SELECTOR
#
# Mode 1 (NIFTY Lab) is a fully isolated module living under
# zorix/nifty_lab/ — its own config dataclass, its own session-state key
# prefix (niftylab_*), its own cached data load. Selecting it renders
# render_nifty_lab() and stops before any Standard Zorix code runs, so
# the Standard Zorix flow below is completely untouched when Mode 1 is
# not selected.
# =========================================================

app_mode = st.sidebar.radio("Mode", ["Standard Zorix", "Mode 1 — NIFTY Lab"], key="zorix_app_mode")

if app_mode == "Mode 1 — NIFTY Lab":
    from zorix.nifty_lab import render_nifty_lab
    render_nifty_lab()
    st.stop()

# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("ZORIX AI")
st.sidebar.caption("Market Intelligence Platform")

asset_class = st.sidebar.selectbox("Asset Class", ["Equities", "Commodities", "Forex", "Bonds"])

if asset_class == "Equities":
    ticker = st.sidebar.text_input("Ticker (yfinance format)", "RELIANCE.NS")
    instrument_meta = equity_instrument_meta(ticker)
    benchmark = st.sidebar.selectbox(
        "Benchmark", list(DEFAULT_CONFIG.CROSS_ASSETS.equity_index.items()),
        format_func=lambda kv: kv[0],
    )
    benchmark_ticker = benchmark[1]

elif asset_class == "Commodities":
    commodity_choice = st.sidebar.selectbox("Commodity", list(DEFAULT_CONFIG.COMMODITIES_META.instruments.keys()))
    instrument_meta = DEFAULT_CONFIG.COMMODITIES_META.instruments[commodity_choice]
    ticker = instrument_meta.yfinance_symbol
    benchmark_ticker = DEFAULT_CONFIG.CROSS_ASSETS.equity_index["SP500"]

elif asset_class == "Forex":
    pair_choice = st.sidebar.selectbox("Currency Pair", list(DEFAULT_CONFIG.FOREX.pairs.keys()))
    instrument_meta = DEFAULT_CONFIG.FOREX.pairs[pair_choice]
    ticker = instrument_meta.yfinance_symbol
    # DXY (dollar index) is the more relevant cross-asset context for FX than
    # an equity benchmark, per Section 17's "related USD index if available".
    benchmark_ticker = DEFAULT_CONFIG.CROSS_ASSETS.fx.get("DXY", DEFAULT_CONFIG.CROSS_ASSETS.equity_index["SP500"])

else:  # Bonds
    bond_choice = st.sidebar.selectbox("Bond / Yield", list(DEFAULT_CONFIG.BONDS.instruments.keys()))
    instrument_meta = DEFAULT_CONFIG.BONDS.instruments[bond_choice]
    ticker = instrument_meta.yfinance_symbol
    benchmark_ticker = DEFAULT_CONFIG.CROSS_ASSETS.equity_index["SP500"]

asset_display = describe_asset(instrument_meta)


start_date = st.sidebar.date_input("Start Date", pd.to_datetime(DEFAULT_CONFIG.START_DATE))
horizon = st.sidebar.selectbox("Prediction Horizon (days)", DEFAULT_CONFIG.HORIZONS, index=1)
risk_profile = st.sidebar.selectbox("Risk Profile", ["Conservative", "Moderate", "Aggressive"], index=1)
run_button = st.sidebar.button("\U0001F680 Run Zorix Analysis")

_RISK_PROFILE_MULT = {"Conservative": 0.5, "Moderate": 1.0, "Aggressive": 1.75}

st.sidebar.markdown("---")
st.sidebar.caption(
    "Model availability: "
    f"XGBoost {'✅' if XGBOOST_AVAILABLE else '❌ (pip install xgboost)'} · "
    f"LightGBM {'✅' if LIGHTGBM_AVAILABLE else '❌ (pip install lightgbm)'} · "
    f"LSTM/GRU {'✅' if TENSORFLOW_AVAILABLE else '❌ (pip install tensorflow)'} · "
    f"SHAP {'✅' if SHAP_AVAILABLE else '❌ (pip install shap)'}"
)
st.sidebar.caption(
    "**Disclaimer:** Zorix is a research/educational decision-support tool. "
    "Outputs are probabilistic model estimates, not financial advice, and "
    "historical backtests do not guarantee future performance."
)

# =========================================================
# CACHED DATA FETCHING
# =========================================================

@st.cache_data(show_spinner=False)
def load_ohlcv(symbol: str, start: str, end: str) -> pd.DataFrame:
    provider = YahooFinanceProvider()
    return provider.fetch_ohlcv(symbol, start, end)


@st.cache_data(show_spinner=False)
def load_cross_asset(start: str, end: str) -> dict:
    provider = YahooFinanceProvider()
    symbols = DEFAULT_CONFIG.CROSS_ASSETS.all_symbols()
    return provider.fetch_many(symbols, start, end)


@st.cache_data(show_spinner=False)
def build_dataset(symbol: str, benchmark_symbol: str, start: str, end: str, yield_scale: float = 1.0) -> tuple:
    cfg = DEFAULT_CONFIG
    df = load_ohlcv(symbol, str(start), str(end))
    df = apply_yield_scale(df, yield_scale)  # no-op unless this is a scaled yield ticker (^TNX/^TYX/^FVX)
    df = build_technical_features(df, cfg)

    cross = load_cross_asset(str(start), str(end))
    bench_df = cross.get(
        [k for k, v in cfg.CROSS_ASSETS.equity_index.items() if v == benchmark_symbol][0]
        if benchmark_symbol in cfg.CROSS_ASSETS.equity_index.values() else None
    ) if cross else None

    if bench_df is not None:
        df = add_relative_strength(df, bench_df, windows=(5, 20))

    other_assets = {k: v for k, v in cross.items()}
    df = add_cross_asset_block(df, other_assets, windows=(1, 5, 20), corr_window=60)

    regime = detect_regime(df, cfg.REGIME_LOOKBACK, cfg.N_REGIMES)
    df = attach_regime_features(df, regime)

    df = add_bhs_targets(df, cfg.HORIZONS, cfg.VOL_LOOKBACK, cfg.K_BUY, cfg.K_SELL, cfg.MIN_THRESHOLD)

    bench_close = bench_df["Close"] if bench_df is not None else None
    return df, regime, bench_close


# =========================================================
# MAIN
# =========================================================

st.title("Zorix 2.0 — Quantitative Decision Support")

if not run_button and "zorix_df" not in st.session_state:
    st.info("Configure a ticker in the sidebar and click **Run Zorix Analysis**.")
    st.stop()

if run_button:
    try:
        with st.spinner("Fetching data and engineering features..."):
            df, regime, bench_close = build_dataset(
                ticker, benchmark_ticker, start_date, DEFAULT_CONFIG.END_DATE,
                yield_scale=instrument_meta.yield_scale,
            )
        st.session_state["zorix_instrument_meta"] = instrument_meta
        st.session_state["zorix_asset_display"] = asset_display
        st.session_state["zorix_df"] = df
        st.session_state["zorix_regime"] = regime
        st.session_state["zorix_ticker"] = ticker
        st.session_state["zorix_bench_close"] = bench_close
        # New run invalidates any previously computed backtest/ablation results.
        for k in ("zorix_backtest", "zorix_ablation", "zorix_stability"):
            st.session_state.pop(k, None)
    except DataUnavailableError as exc:
        st.error(f"Data unavailable for {ticker}: {exc}")
        st.stop()
    except Exception as exc:
        st.error(f"Unexpected error while building the dataset: {exc}")
        st.stop()

df = st.session_state["zorix_df"]
regime = st.session_state["zorix_regime"]
ticker = st.session_state["zorix_ticker"]
instrument_meta = st.session_state["zorix_instrument_meta"]
asset_display = st.session_state["zorix_asset_display"]
bench_close = st.session_state.get("zorix_bench_close")

target_col = f"Target_{horizon}D"
fwd_col = f"FwdReturn_{horizon}D"

feature_cols = [c for c in df.columns
                if c not in DEFAULT_CONFIG.DROP_COLS
                and not c.startswith("Threshold_")
                and not c.startswith("Target_")
                and not c.startswith("FwdReturn_")
                and c != "Regime_Engine"]
regime_cols = [c for c in feature_cols if c.startswith("Regime_")]

clean = df.dropna(subset=feature_cols + [target_col, fwd_col]).copy()

if len(clean) < 200:
    st.warning(
        f"Only {len(clean)} usable rows after feature/target computation — "
        "results below may be unstable. Try an earlier start date."
    )

if len(clean) < 50:
    st.stop()

X = clean[feature_cols]
y = sklearn_label(clean[target_col])
fwd_returns = clean[fwd_col]

# Purged chronological split: drop the last `horizon` train rows so no training
# label window (Close[t+h]/Close[t]) reaches into the test period.
train_end, split = purged_split_points(len(X), DEFAULT_CONFIG.TEST_SIZE, horizon)
X_train, X_test = X.iloc[:train_end], X.iloc[split:]
y_train, y_test = y.iloc[:train_end], y.iloc[split:]
fwd_train, fwd_test = fwd_returns.iloc[:train_end], fwd_returns.iloc[split:]

cache_key = f"{ticker}_{horizon}D_{len(clean)}"

# =========================================================
# 1. TRAIN CLASSIFICATION MODELS (cached)
# =========================================================

@st.cache_resource(show_spinner=False)
def train_models(_X_train, _X_test, _y_train, _y_test, cache_key: str, include_lstm: bool, include_gru: bool):
    trainer = ModelTrainer(random_state=DEFAULT_CONFIG.RANDOM_STATE)
    return trainer.train_all(_X_train, _X_test, _y_train, _y_test,
                              include_lstm=include_lstm, include_gru=include_gru)


include_nn = st.sidebar.checkbox("Include LSTM/GRU (slow)", value=False, disabled=not TENSORFLOW_AVAILABLE)
with st.spinner("Training classification models..."):
    report = train_models(X_train, X_test, y_train, y_test, cache_key, include_nn, False)

available = report.available_models()
skipped = report.skipped_models()

if not available:
    st.error("No models trained successfully. Install at least scikit-learn's dependencies.")
    st.stop()

# =========================================================
# 2. RETURN REGRESSION MODEL (expected return, replaces the old
#    "historical average" placeholder with an actual OOS prediction)
# =========================================================

@st.cache_resource(show_spinner=False)
def train_regression(_X_train, _X_test, _y_train, _y_test, cache_key: str, gap: int):
    trainer = ReturnRegressionTrainer(random_state=DEFAULT_CONFIG.RANDOM_STATE)
    return trainer.train_all(_X_train, _X_test, _y_train, _y_test, gap=gap)


with st.spinner("Training return-regression models..."):
    reg_report = train_regression(X_train, X_test, fwd_train, fwd_test, cache_key, horizon)
best_reg = reg_report.best()  # chosen on a validation slice of train, not on test
expected_return = float(best_reg.predictions_test[-1]) if best_reg and best_reg.predictions_test is not None else None
expected_vol = float(clean["Close"].pct_change().rolling(20).std().iloc[-1] * np.sqrt(252))

# =========================================================
# 3. OOF STACKING ENSEMBLE (replaces naive average of base models)
# =========================================================

@st.cache_resource(show_spinner=False)
def train_stacking(_X_train, _y_train, _ctx_train, _X_test, _ctx_test, cache_key: str, gap: int):
    # Whole stack (OOF base preds, meta-model, base refits) is fit on TRAIN only;
    # the holdout is only scored. OOF folds are purged by the label horizon.
    factories = default_base_model_factories(DEFAULT_CONFIG.RANDOM_STATE)
    stack_result = fit_stacking(_X_train, _y_train, _ctx_train, factories, n_splits=5, gap=gap,
                                random_state=DEFAULT_CONFIG.RANDOM_STATE)
    return stack_result, predict_stack_new(stack_result, _X_test, _ctx_test)


context_df = clean[regime_cols] if regime_cols else None
ctx_train = context_df.iloc[:train_end] if context_df is not None else None
ctx_test = context_df.iloc[split:] if context_df is not None else None
with st.spinner("Training stacking ensemble (out-of-fold, train only)..."):
    stack_result, stacked_test_proba = train_stacking(X_train, y_train, ctx_train, X_test, ctx_test,
                                                      cache_key, horizon)

stacked_proba = stacked_test_proba[-1]
final_proba = stacked_proba

# =========================================================
# TOP DASHBOARD / SIGNAL
# =========================================================

current_price = float(clean["Close"].iloc[-1])
daily_return = float(clean["Close"].pct_change().iloc[-1])
regime_label = regime.current_regime or "Unknown"
regime_conf = regime.current_confidence

bullish, bearish = [], []
last_row = clean.iloc[-1]
if last_row.get("RSI", 50) < 40:
    bullish.append("RSI showing oversold conditions")
elif last_row.get("RSI", 50) > 60:
    bearish.append("RSI showing overbought conditions")
if last_row.get("RelStrength_20D", 0) > 0:
    bullish.append("Outperforming benchmark over 20D")
elif "RelStrength_20D" in last_row.index:
    bearish.append("Underperforming benchmark over 20D")
if "BULL" in regime_label:
    bullish.append(f"Market regime: {regime_label}")
else:
    bearish.append(f"Market regime: {regime_label}")

signal = build_signal(
    asset=ticker, asset_class=asset_class, horizon_days=horizon,
    proba=final_proba, expected_return=expected_return, expected_volatility=expected_vol,
    market_regime=regime_label, regime_confidence=regime_conf,
    bullish_factors=bullish, bearish_factors=bearish,
    min_edge=DEFAULT_CONFIG.MIN_EDGE, min_confidence=DEFAULT_CONFIG.MIN_CONFIDENCE,
    model_version=DEFAULT_CONFIG.MODEL_VERSION, data_timestamp=str(clean.index[-1]),
)

# Log every generated signal (append-only — prediction logging).
log_prediction({
    "timestamp": pd.Timestamp.now().isoformat(), "asset": signal.asset,
    "horizon_days": signal.horizon_days, "signal": signal.signal, "confidence": signal.confidence,
    "buy_probability": signal.buy_probability, "hold_probability": signal.hold_probability,
    "sell_probability": signal.sell_probability, "expected_return": signal.expected_return,
    "market_regime": signal.market_regime, "data_timestamp": signal.data_timestamp,
}, DEFAULT_CONFIG.PREDICTION_LOG_PATH)

col1, col2, col3, col4 = st.columns(4)
col1.metric(f"{ticker} — {asset_display.value_label}", format_instrument_value(current_price, instrument_meta))
col2.metric(asset_display.change_label, f"{daily_return:+.2%}" if pd.notna(daily_return) else "N/A")
col3.metric(asset_display.volatility_label, f"{expected_vol:.1%}" if expected_vol is not None else "N/A")
col4.metric("RSI", f"{last_row.get('RSI', float('nan')):.1f}" if pd.notna(last_row.get("RSI", float("nan"))) else "N/A")

# ---- ZORIX SIGNAL card (Section 12): genuine four-state system ----
_SIGNAL_ICON = {"BUY": "🟢", "SELL": "🔴", "HOLD": "🟡", "NO TRADE": "⚪"}
_SIGNAL_INTERPRETATION = {
    "BUY": "Bullish directional bias.",
    "SELL": "Bearish directional bias.",
    "HOLD": "Neutral / weak directional bias. Existing position may be maintained; no new entry signaled.",
    "NO TRADE": "Insufficient statistical edge — Zorix recommends waiting for stronger market confirmation "
                "before opening a new position.",
}

st.markdown(f"""
<div class='glass-card' style='text-align:center;'>
<div style='letter-spacing:2px; color:#94a3b8; font-size:0.85rem;'>ZORIX SIGNAL</div>
<div style='font-size:2.2rem; font-weight:700; margin:8px 0;'>{_SIGNAL_ICON[signal.signal]} {signal.signal}</div>
<div style='color:#94a3b8;'>Confidence</div>
<div style='font-size:1.3rem; font-weight:600;'>{signal.confidence}</div>
</div>
""", unsafe_allow_html=True)

pc1, pc2, pc3 = st.columns(3)
pc1.metric("BUY", f"{signal.buy_probability:.0%}")
pc2.metric("HOLD", f"{signal.hold_probability:.0%}")
pc3.metric("SELL", f"{signal.sell_probability:.0%}")

interpretation = signal.no_trade_reason if signal.signal == "NO TRADE" else _SIGNAL_INTERPRETATION[signal.signal]
expected_return_str = f"{expected_return*100:+.2f}%" if expected_return is not None else "N/A"
st.markdown(
    f"<div class='glass-card'>"
    f"<b>Interpretation:</b> {interpretation}<br>"
    f"<b>Market Regime:</b> {regime_label} (confidence {regime_conf:.0%}, engine: {regime.engine}) &nbsp;|&nbsp; "
    f"<b>Expected {horizon}D Return (model, OOS):</b> {expected_return_str}"
    f"</div>", unsafe_allow_html=True,
)

if skipped:
    st.caption("Models not run: " + ", ".join(f"{k} ({v})" for k, v in skipped.items()))

# =========================================================
# TABS
# =========================================================

tab_overview, tab_models, tab_explain, tab_risk, tab_backtest, tab_market, tab_fo, tab_news, tab_log = st.tabs([
    "Overview", "Model Consensus & Calibration", "Explainability", "Risk Management",
    "Backtest", "Market Intelligence", "Options & Futures", "News Sentiment", "Prediction Log",
])

# ---- Overview: price chart ----
with tab_overview:
    st.subheader("Price Chart" if asset_display.chart_type == "candlestick" else f"{asset_display.value_label} Over Time")
    if asset_display.chart_type == "line":
        # Yield/rate instruments where a candlestick range isn't meaningful (Section 19).
        if PLOTLY_AVAILABLE:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=clean.index, y=clean["Close"], mode="lines", name=asset_display.value_label,
                                      line=dict(color="#00ff88")))
            fig.update_layout(template="plotly_dark", height=500, paper_bgcolor="#020617",
                               plot_bgcolor="#020617", font=dict(color="white"),
                               yaxis_title=asset_display.value_label)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.line_chart(clean["Close"])
    elif PLOTLY_AVAILABLE:
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3])
        fig.add_trace(go.Candlestick(x=clean.index, open=clean["Open"], high=clean["High"],
                                      low=clean["Low"], close=clean["Close"], name="Price"), row=1, col=1)
        if "BB_High" in clean.columns:
            fig.add_trace(go.Scatter(x=clean.index, y=clean["BB_High"], line=dict(color="rgba(255,255,255,0.3)", width=1), name="BB Upper"), row=1, col=1)
            fig.add_trace(go.Scatter(x=clean.index, y=clean["BB_Low"], line=dict(color="rgba(255,255,255,0.3)", width=1), name="BB Lower"), row=1, col=1)
        colors = np.where(clean["Close"].pct_change().fillna(0) >= 0, "#00ff88", "#ff4d6d")
        fig.add_trace(go.Bar(x=clean.index, y=clean["Volume"], marker_color=colors, name="Volume"), row=2, col=1)
        fig.update_layout(template="plotly_dark", height=600, xaxis_rangeslider_visible=False,
                           paper_bgcolor="#020617", plot_bgcolor="#020617", font=dict(color="white"))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.line_chart(clean["Close"])

    with st.expander("Full structured signal (raw)"):
        st.json({
            "asset": signal.asset, "asset_class": signal.asset_class,
            "horizon_days": signal.horizon_days,
            "sell_probability": signal.sell_probability, "hold_probability": signal.hold_probability,
            "buy_probability": signal.buy_probability, "signal": signal.signal,
            "confidence": signal.confidence, "no_trade_reason": signal.no_trade_reason,
            "market_regime": signal.market_regime, "regime_confidence": signal.regime_confidence,
            "key_bullish_factors": signal.key_bullish_factors, "key_bearish_factors": signal.key_bearish_factors,
            "model_version": signal.model_version, "data_timestamp": signal.data_timestamp,
        })

# ---- Model Consensus & Calibration ----
with tab_models:
    st.subheader("Model Consensus")
    probas_by_model = {r.name: r.proba_test[-1] for r in report.results if r.ran}
    st.dataframe(model_consensus(probas_by_model), use_container_width=True, hide_index=True)
    st.caption(f"Stacked ensemble (meta-model over {', '.join(stack_result.base_model_names)} "
               f"+ regime context) final call: **{signal.signal}** — this, not a simple average, is what "
               "drives the headline Zorix Signal above.")

    st.markdown("**Per-model out-of-sample metrics (single train/test split):**")
    metrics_rows = [{"Model": r.name, **r.metrics} for r in report.results if r.ran]
    st.dataframe(pd.DataFrame(metrics_rows), use_container_width=True, hide_index=True)

    st.markdown("**Return-regression models (predicting continuous forward return):**")
    reg_rows = [{"Model": r.name, **r.metrics} for r in reg_report.results if r.ran]
    st.dataframe(pd.DataFrame(reg_rows), use_container_width=True, hide_index=True)
    st.caption("directional_accuracy = how often the predicted return's sign matched the realized return's sign — "
               "the metric that actually matters for a BUY/SELL call, independent of magnitude error.")

    st.markdown("---")
    st.subheader("Probability Calibration")
    st.caption("Distinguishes *prediction confidence* (the probability a model outputs) from "
               "*calibration* (whether that probability is trustworthy) — Section 9.")
    if SKLEARN_AVAILABLE:
        if st.button("Run calibration check (Random Forest, isotonic)"):
            with st.spinner("Calibrating..."):
                base_rf = RandomForestClassifier(n_estimators=200, max_depth=8, min_samples_leaf=10,
                                                  n_jobs=-1, random_state=DEFAULT_CONFIG.RANDOM_STATE)
                calibrated = calibrate_model(base_rf, X_train, y_train, method="isotonic")
                uncalibrated = RandomForestClassifier(n_estimators=200, max_depth=8, min_samples_leaf=10,
                                                       n_jobs=-1, random_state=DEFAULT_CONFIG.RANDOM_STATE)
                uncalibrated.fit(X_train, y_train)
                proba_before = uncalibrated.predict_proba(X_test)
                proba_after = calibrated.predict_proba(X_test)
                cal = evaluate_calibration(y_test.values, proba_before, proba_after, target_class=2)

            c1, c2 = st.columns(2)
            c1.metric("Brier score (BUY class) — before", f"{cal.brier_before:.4f}")
            c2.metric("Brier score (BUY class) — after isotonic calibration", f"{cal.brier_after:.4f}",
                       delta=f"{cal.brier_after - cal.brier_before:+.4f}", delta_color="inverse")
            st.caption("Lower Brier = better-calibrated probabilities. A positive delta means calibration "
                       "made this particular split slightly worse — with limited test data that can happen; "
                       "it's still worth checking on your real ticker/date range.")
            if PLOTLY_AVAILABLE and not cal.calibration_curve_before.empty:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Perfect calibration",
                                          line=dict(dash="dash", color="gray")))
                fig.add_trace(go.Scatter(x=cal.calibration_curve_before["predicted"],
                                          y=cal.calibration_curve_before["actual"],
                                          mode="lines+markers", name="Before"))
                fig.add_trace(go.Scatter(x=cal.calibration_curve_after["predicted"],
                                          y=cal.calibration_curve_after["actual"],
                                          mode="lines+markers", name="After"))
                fig.update_layout(template="plotly_dark", height=400, xaxis_title="Predicted probability",
                                   yaxis_title="Actual frequency", paper_bgcolor="#020617", plot_bgcolor="#020617",
                                   font=dict(color="white"))
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.markdown("<div class='unavailable-card'>scikit-learn unavailable.</div>", unsafe_allow_html=True)

# ---- Explainability ----
with tab_explain:
    st.subheader("Why This Signal? (SHAP / Feature Importance)")
    tree_models = {r.name: r.model for r in report.results
                    if r.ran and r.name in ("Random Forest", "XGBoost", "LightGBM")}
    if tree_models:
        chosen_name = st.selectbox("Explain using model", list(tree_models.keys()))
        chosen_model = tree_models[chosen_name]
        class_choice = st.radio("Explain contribution toward class", ["BUY", "HOLD", "SELL"], horizontal=True)
        class_idx = {"SELL": 0, "HOLD": 1, "BUY": 2}[class_choice]

        explanation = explain_prediction(chosen_model, X_test.iloc[[-1]], target_class=class_idx, top_n=12)
        st.caption(f"Method: **{explanation.method}** — {explanation.note}")
        if not explanation.feature_contributions.empty:
            fig_data = explanation.feature_contributions.sort_values("contribution")
            if PLOTLY_AVAILABLE:
                colors = ["#00ff88" if v >= 0 else "#ff4d6d" for v in fig_data["contribution"]]
                fig = go.Figure(go.Bar(x=fig_data["contribution"], y=fig_data["feature"],
                                        orientation="h", marker_color=colors))
                fig.update_layout(template="plotly_dark", height=400, paper_bgcolor="#020617",
                                   plot_bgcolor="#020617", font=dict(color="white"))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.dataframe(fig_data, use_container_width=True)
    else:
        st.markdown("<div class='unavailable-card'>No tree-based model available to explain "
                    "(install xgboost, lightgbm, or ensure Random Forest ran).</div>", unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("Feature Ablation (permutation importance, held-out data)")
    st.caption("Each feature is shuffled independently and the drop in held-out accuracy is measured — "
               "features with near-zero importance here aren't meaningfully used by the model.")
    if st.button("Run feature ablation"):
        rf_for_ablation = next((r.model for r in report.results if r.name == "Random Forest" and r.ran), None)
        if rf_for_ablation is not None:
            with st.spinner("Running permutation ablation..."):
                ablation = run_feature_ablation(rf_for_ablation, X_test, y_test,
                                                 scoring="accuracy", n_repeats=DEFAULT_CONFIG.ABLATION_N_REPEATS)
            st.session_state["zorix_ablation"] = ablation
        else:
            st.warning("Random Forest didn't train successfully; ablation needs a fitted model.")
    if "zorix_ablation" in st.session_state:
        st.dataframe(st.session_state["zorix_ablation"].importances.head(20), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("Feature Stability Across Walk-Forward Folds")
    st.caption("Are the 'important' features consistent over time, or is importance just noise from one lucky split? "
               "Higher mean rank correlation = more stable.")
    if st.button("Run feature stability analysis (retrains several models — slower)"):
        def stability_train_fn(X_tr, y_tr):
            m = RandomForestClassifier(n_estimators=150, max_depth=6, n_jobs=-1,
                                        random_state=DEFAULT_CONFIG.RANDOM_STATE)
            m.fit(X_tr, y_tr)
            return m
        with st.spinner("Running walk-forward feature stability analysis..."):
            stability = run_feature_stability(
                clean, feature_cols, target_col, stability_train_fn,
                n_splits=DEFAULT_CONFIG.STABILITY_N_FOLDS,
                min_train_size=DEFAULT_CONFIG.WALK_FORWARD_MIN_TRAIN,
                n_repeats=5, random_state=DEFAULT_CONFIG.RANDOM_STATE,
            )
        st.session_state["zorix_stability"] = stability
    if "zorix_stability" in st.session_state:
        stab = st.session_state["zorix_stability"]
        st.metric("Folds used", stab.n_folds_used)
        st.metric("Mean pairwise rank correlation across folds", f"{stab.mean_rank_correlation:.3f}")
        st.dataframe(
            pd.DataFrame({
                "mean_importance": stab.mean_importance,
                "coefficient_of_variation": stab.coefficient_of_variation.reindex(stab.mean_importance.index),
            }).head(20), use_container_width=True,
        )

# ---- Risk Management ----
with tab_risk:
    st.subheader("Risk Management")
    if signal.signal in ("BUY", "SELL"):
        atr = float(clean["ATR"].iloc[-1]) if "ATR" in clean.columns and pd.notna(clean["ATR"].iloc[-1]) else None
        risk_mult = _RISK_PROFILE_MULT[risk_profile]
        assessment = build_risk_assessment(
            entry=current_price, direction=signal.signal, atr=atr,
            capital=DEFAULT_CONFIG.PORTFOLIO_CAPITAL,
            risk_per_trade_pct=DEFAULT_CONFIG.RISK_PER_TRADE * risk_mult,
            stop_mult=DEFAULT_CONFIG.STOP_LOSS_ATR_MULT, rr_ratio=DEFAULT_CONFIG.TARGET_RR_RATIO,
            max_position_pct=DEFAULT_CONFIG.MAX_POSITION_PCT,
            max_portfolio_risk_pct=DEFAULT_CONFIG.MAX_PORTFOLIO_RISK,
        )
        if assessment.valid:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Entry", f"{assessment.entry:,.2f}")
            c2.metric("Stop Loss", f"{assessment.stop_loss:,.2f}")
            c3.metric("Target", f"{assessment.target:,.2f}")
            c4.metric("Risk/Reward", f"1 : {assessment.risk_reward:.2f}")

            c1, c2, c3 = st.columns(3)
            c1.metric("Suggested Shares", f"{assessment.position.shares:,.1f}")
            c2.metric("Position Value", f"{assessment.position.position_value:,.0f}")
            c3.metric("% of Capital", f"{assessment.position.position_pct_of_capital:.1%}")

            if assessment.position.capped_by_max_position:
                st.caption(f"Position capped at the configured max of {DEFAULT_CONFIG.MAX_POSITION_PCT:.0%} of capital.")
            if not assessment.within_portfolio_risk_cap:
                st.warning(assessment.reason)
            st.caption(f"Stop distance is {assessment.stop_distance_pct:.2%} of entry price, "
                       f"sized from {DEFAULT_CONFIG.STOP_LOSS_ATR_MULT}× the current ATR — not an arbitrary percentage.")
        else:
            st.markdown(f"<div class='unavailable-card'>{assessment.reason or 'Risk assessment unavailable (ATR missing).'}</div>",
                        unsafe_allow_html=True)
    else:
        st.info(f"Current signal is **{signal.signal}** — no new position, so no risk parameters to show. "
                "Risk parameters only apply to BUY/SELL signals.")
    st.caption("This is model-generated analysis, not financial advice. Position sizing assumes a "
               f"{DEFAULT_CONFIG.PORTFOLIO_CAPITAL:,.0f} reference portfolio and "
               f"{DEFAULT_CONFIG.RISK_PER_TRADE:.1%} base risk per trade (scaled by risk profile).")

# ---- Backtest ----
with tab_backtest:
    st.subheader("Walk-Forward Backtest")
    st.caption(
        "Signals used here come from walk-forward, strictly out-of-sample predictions "
        f"({'XGBoost' if XGBOOST_AVAILABLE else 'Random Forest'}, expanding window, "
        f"{DEFAULT_CONFIG.WALK_FORWARD_FOLDS} folds) — not the single train/test split model "
        "used for the headline signal above, and never fit on the data being traded. "
        "This can take a minute."
    )
    if st.button("Run walk-forward validation + backtest"):
        def wf_train_fn(X_tr, X_te, y_tr, y_te):
            if XGBOOST_AVAILABLE:
                import xgboost as xgb
                m = xgb.XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=5,
                                       subsample=0.8, colsample_bytree=0.8,
                                       objective="multi:softprob", num_class=3, eval_metric="mlogloss",
                                       random_state=DEFAULT_CONFIG.RANDOM_STATE, n_jobs=-1)
            else:
                m = RandomForestClassifier(n_estimators=200, max_depth=6, n_jobs=-1,
                                            random_state=DEFAULT_CONFIG.RANDOM_STATE)
            m.fit(X_tr, y_tr)
            return m

        def wf_predict_fn(model, X_te):
            proba = model.predict_proba(X_te)
            full = np.zeros((len(X_te), 3))
            for j, c in enumerate(getattr(model, "classes_", [0, 1, 2])):
                full[:, int(c)] = proba[:, j]
            return full

        with st.spinner("Running walk-forward validation..."):
            wf_report = run_walk_forward(
                clean, feature_cols, target_col, fwd_col, wf_train_fn, wf_predict_fn,
                n_splits=DEFAULT_CONFIG.WALK_FORWARD_FOLDS, min_train_size=DEFAULT_CONFIG.WALK_FORWARD_MIN_TRAIN,
                collect_predictions=True,
            )
            oos_signals = oos_predictions_to_signals(wf_report.oos_proba, DEFAULT_CONFIG.MIN_EDGE, DEFAULT_CONFIG.MIN_CONFIDENCE)

        with st.spinner("Running backtest simulation..."):
            price_cols = ["Open", "High", "Low", "Close"] + (["ATR"] if "ATR" in clean.columns else [])
            bt = Backtester(DEFAULT_CONFIG)
            bt_result = bt.run(clean[price_cols], oos_signals, benchmark_close=bench_close)

        st.session_state["zorix_backtest"] = (wf_report, bt_result)

    if "zorix_backtest" in st.session_state:
        wf_report, bt_result = st.session_state["zorix_backtest"]

        st.markdown("**Walk-forward fold summary (mean / median / std across folds):**")
        st.dataframe(wf_report.summary(), use_container_width=True)
        st.caption(f"{len(wf_report.folds)} folds evaluated. Each fold's model is trained only on data "
                   "strictly before that fold's test window (expanding window — no shuffling).")

        st.markdown("---")
        st.markdown("**Backtest performance (out-of-sample signals, with transaction costs & slippage):**")

        def _metrics_row(name, m):
            return {
                "Strategy": name, "Total Return": f"{m.total_return:.1%}" if pd.notna(m.total_return) else "N/A",
                "CAGR": f"{m.cagr:.1%}" if pd.notna(m.cagr) else "N/A",
                "Sharpe": f"{m.sharpe:.2f}" if pd.notna(m.sharpe) else "N/A",
                "Sortino": f"{m.sortino:.2f}" if pd.notna(m.sortino) else "N/A",
                "Calmar": f"{m.calmar:.2f}" if pd.notna(m.calmar) else "N/A",
                "Max Drawdown": f"{m.max_drawdown:.1%}" if pd.notna(m.max_drawdown) else "N/A",
                "Win Rate": f"{m.win_rate:.1%}" if pd.notna(m.win_rate) else "N/A",
                "Profit Factor": f"{m.profit_factor:.2f}" if pd.notna(m.profit_factor) else "N/A",
                "Trades": m.n_trades, "Avg Hold (days)": f"{m.avg_holding_days:.1f}" if pd.notna(m.avg_holding_days) else "N/A",
                "Turnover (ann.)": f"{m.turnover_annualized:.2f}" if pd.notna(m.turnover_annualized) else "N/A",
            }

        rows = [_metrics_row("Zorix (walk-forward, costed)", bt_result.metrics)]
        if bt_result.buy_hold_metrics:
            rows.append(_metrics_row(f"Buy & Hold ({ticker})", bt_result.buy_hold_metrics))
        if bt_result.benchmark_metrics:
            rows.append(_metrics_row("Benchmark", bt_result.benchmark_metrics))
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        if PLOTLY_AVAILABLE and len(bt_result.equity_curve) > 1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=bt_result.equity_curve.index, y=bt_result.equity_curve.values, name="Zorix"))
            if bt_result.buy_hold_equity is not None:
                fig.add_trace(go.Scatter(x=bt_result.buy_hold_equity.index, y=bt_result.buy_hold_equity.values, name="Buy & Hold"))
            if bt_result.benchmark_equity is not None:
                fig.add_trace(go.Scatter(x=bt_result.benchmark_equity.index, y=bt_result.benchmark_equity.values, name="Benchmark"))
            fig.update_layout(template="plotly_dark", height=400, title="Equity Curve", paper_bgcolor="#020617",
                               plot_bgcolor="#020617", font=dict(color="white"))
            st.plotly_chart(fig, use_container_width=True)

        if bt_result.trades:
            with st.expander(f"Trade log ({len(bt_result.trades)} trades)"):
                trades_df = pd.DataFrame([{
                    "Entry": t.entry_date, "Exit": t.exit_date, "Direction": t.direction,
                    "Entry Price": t.entry_price, "Exit Price": t.exit_price,
                    "Net Return": f"{t.net_return_pct:.2%}", "Exit Reason": t.exit_reason,
                    "Holding Days": t.holding_days,
                } for t in bt_result.trades])
                st.dataframe(trades_df, use_container_width=True, hide_index=True)
        else:
            st.info("No trades were generated in this backtest window (signals were HOLD/NO TRADE throughout, "
                    "or the window was too short). This is a real, measured result — not an error.")
    else:
        st.info("Click the button above to run the walk-forward validation and backtest. "
                "Not run automatically because it retrains models several times and can take a while.")

# ---- Market Intelligence ----
with tab_market:
    st.subheader("Market Intelligence")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Cross-Asset Snapshot**")
        cross_cols = [c for c in clean.columns if c.endswith("_Return_5D") and not c.startswith(("Return_", "LogReturn_"))]
        if cross_cols:
            snapshot = clean[cross_cols].iloc[-1].sort_values(ascending=False)
            st.dataframe(snapshot.to_frame("5D Return").style.format("{:+.2%}"), use_container_width=True)
        else:
            st.caption("No cross-asset data available for this run.")
    with c2:
        st.markdown("**Relative Strength vs Benchmark**")
        rel_cols = [c for c in clean.columns if c.startswith("RelStrength_")]
        if rel_cols:
            rel = clean[rel_cols].iloc[-1]
            st.dataframe(rel.to_frame("Relative Strength").style.format("{:+.2%}"), use_container_width=True)
        else:
            st.caption("No benchmark configured for this asset.")

# ---- Options & Futures ----
with tab_fo:
    st.subheader("Options & Futures")
    st.caption("No F&O data provider is connected yet (credentials/docs not available). "
               "See data/options_data.py and data/futures_data.py for the interfaces to implement.")
    options_provider = NullOptionsProvider()
    futures_provider = NullFuturesProvider()
    opt_snap = options_provider.get_snapshot(ticker)
    fut_snap = futures_provider.get_snapshot(ticker)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Options Intelligence**")
        if opt_snap.available:
            st.metric("PCR (OI)", f"{opt_snap.pcr_oi:.2f}")
            st.metric("ATM IV", f"{opt_snap.atm_iv:.1%}")
            st.caption(f"Options-derived support: {opt_snap.max_put_oi_strike} · resistance: {opt_snap.max_call_oi_strike} "
                       "(not guaranteed levels)")
        else:
            st.markdown(f"<div class='unavailable-card'>Unavailable — {opt_snap.unavailable_reason}</div>", unsafe_allow_html=True)
    with c2:
        st.markdown("**Futures Positioning**")
        if fut_snap.available:
            st.metric("Basis %", f"{fut_snap.basis_pct:.2f}%")
            st.metric("Positioning", fut_snap.positioning)
            st.caption(fut_snap.interpretation)
        else:
            st.markdown(f"<div class='unavailable-card'>Unavailable — {fut_snap.unavailable_reason}</div>", unsafe_allow_html=True)

# ---- News Sentiment ----
with tab_news:
    st.subheader("News Sentiment")
    # Asset-aware search query — a raw ticker like "USDINR=X" or "^TNX" makes
    # a poor news search query, so query with the human-readable label + a
    # category hint instead.
    _news_query_hint = {
        "Equities": f"{ticker.replace('^', '').replace('.', '-')} stock",
        "Commodities": f"{ticker.replace('=F', '')} commodity price",
        "Forex": f"{ticker.replace('=X', '')} exchange rate",
        "Bonds": f"{ticker.replace('^', '')} treasury yield",
    }
    news_query = _news_query_hint.get(asset_class, ticker)
    with st.spinner("Fetching latest headlines..."):
        headlines = fetch_headlines(news_query)
        sentiment = score_headlines(headlines)

    if sentiment.engine == "unavailable":
        st.markdown("<div class='unavailable-card'>No sentiment engine available (install vaderSentiment or transformers) "
                    "or no headlines were fetched.</div>", unsafe_allow_html=True)
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Sentiment Engine", sentiment.engine.upper())
        c2.metric("Compound Score", f"{sentiment.compound:+.3f}")
        c3.metric("Headlines Analyzed", sentiment.n_headlines)
        with st.expander(f"Latest headlines ({len(headlines)})"):
            for row in score_individual_headlines(headlines):
                score_str = f"[{row['compound']:+.3f}]" if row["compound"] is not None else ""
                st.markdown(f"- {row['headline']} {score_str}")

    st.caption(
        "Historical sentiment is not used as a training feature in this build because no real "
        "historical news archive is connected — a synthetic proxy was removed (see Section 17) "
        "rather than left in place, since it was a leakage risk, not a real signal."
    )

# ---- Prediction Log ----
with tab_log:
    st.subheader("Prediction Log")
    st.caption(f"Every signal Zorix generates is appended to `{DEFAULT_CONFIG.PREDICTION_LOG_PATH}`. "
               "Use this over time to check whether stated confidence actually tracked outcomes.")
    summary = prediction_log_summary(DEFAULT_CONFIG.PREDICTION_LOG_PATH)
    if summary:
        c1, c2 = st.columns(2)
        c1.metric("Predictions logged", summary["n_predictions"])
        if summary.get("signal_counts"):
            c2.write(summary["signal_counts"])
        log_df = load_predictions(DEFAULT_CONFIG.PREDICTION_LOG_PATH)
        st.dataframe(log_df.tail(50), use_container_width=True, hide_index=True)
    else:
        st.info("No predictions logged yet in this environment.")

st.markdown("---")
st.caption(
    "Zorix 2.0 — research & educational decision-support tool. Not financial advice. "
    "Model outputs are probabilistic and historical performance does not guarantee future results."
)

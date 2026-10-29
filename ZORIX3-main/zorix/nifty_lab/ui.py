"""
zorix/nifty_lab/ui.py

Mode 1 — NIFTY Lab UI. Single entry point: render_nifty_lab(). Called from
app.py behind the top-level mode selector, so none of this runs unless the
user is actually in Mode 1. All session-state keys are prefixed
`niftylab_` so they can never collide with `zorix_*` keys used by Standard
Zorix.

Nothing heavy (data load aside, which is itself st.cache_data'd) runs
except behind an explicit button click.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

from zorix.nifty_lab.config import NiftyLabConfig, DEFAULT_NIFTYLAB_CONFIG
from zorix.nifty_lab.csv_provider import load_niftylab_dataset, NiftyLabDataError
from zorix.nifty_lab.intraday_model import train_intraday_models
from zorix.nifty_lab.nextday_model import train_nextday_model
from zorix.nifty_lab.historical_sim import (
    run_nextday_walkforward, run_intraday_walkforward,
    run_historical_forecast, HistoricalForecastError,
)


def _dataset_summary_card(ds) -> None:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Minute bars", f"{ds.n_minute_bars:,}")
    c2.metric("Trading sessions", f"{ds.n_sessions:,}")
    c3.metric("Volume feed usable", "No" if not ds.volume_usable else "Yes")
    c4.metric("Bad OHLC rows (rounded, not dropped)", ds.bad_ohlc_rows)


def _plot_minute_close(minute: pd.DataFrame, n_bars: int = 2000) -> None:
    if not PLOTLY_AVAILABLE:
        st.line_chart(minute["Close"].tail(n_bars))
        return
    recent = minute.tail(n_bars)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=recent.index, y=recent["Close"], mode="lines",
                              name="NIFTY Close", line=dict(color="#00ff88")))
    fig.update_layout(template="plotly_dark", height=400, paper_bgcolor="#020617",
                       plot_bgcolor="#020617", font=dict(color="white"),
                       title=f"Last {n_bars} 1-min bars")
    st.plotly_chart(fig, use_container_width=True)


def render_nifty_lab(cfg: NiftyLabConfig = None) -> None:
    cfg = cfg or DEFAULT_NIFTYLAB_CONFIG

    st.subheader("\U0001F4CA Mode 1 — NIFTY Lab")
    st.caption(
        "Isolated module: intraday (5/15/30-min) and next-day close/direction "
        "forecasts, built from a local NIFTY 1-minute spot file. Does not touch "
        "Standard Zorix's data, session state, or models."
    )

    # ---- Load data (cached, cheap) ----
    try:
        ds = load_niftylab_dataset(cfg)
    except NiftyLabDataError as e:
        st.error(f"NIFTY Lab data error: {e}")
        return

    st.session_state["niftylab_dataset_loaded"] = True
    _dataset_summary_card(ds)

    tab_chart, tab_intraday, tab_nextday, tab_backtest = st.tabs(
        ["Price Chart", "Intraday Forecast", "Next-Day Forecast", "Historical Backtest"]
    )

    # ---- Price chart ----
    with tab_chart:
        _plot_minute_close(ds.minute)
        st.dataframe(ds.daily.tail(10), use_container_width=True)

    # ---- Intraday forecast ----
    with tab_intraday:
        st.markdown("##### Intraday forecast (5 / 15 / 30 minutes ahead)")
        window = st.number_input(
            "Training window (most recent minute bars)",
            min_value=1000, max_value=len(ds.minute),
            value=min(cfg.intraday_train_window, len(ds.minute)), step=500,
            key="niftylab_intraday_window",
        )
        if st.button("\u25B6\uFE0F Train & Forecast (Intraday)", key="niftylab_run_intraday"):
            with st.spinner("Training intraday models..."):
                result = train_intraday_models(ds.minute, cfg, train_window=int(window))
            st.session_state["niftylab_intraday_result"] = result

        if "niftylab_intraday_result" in st.session_state:
            result = st.session_state["niftylab_intraday_result"]
            if not result.horizons:
                st.warning("Not enough data in the selected window to train. Try a larger window.")
            else:
                st.caption(f"As of {result.as_of} · trained on {result.train_window_used:,} bars")
                cols = st.columns(len(result.horizons))
                for col, (h, hr) in zip(cols, sorted(result.horizons.items())):
                    with col:
                        direction = "\U0001F7E2 UP" if hr.last_forecast_dir == 1 else "\U0001F534 DOWN"
                        st.metric(f"{h}-min forecast", f"{hr.last_forecast_price:,.2f}",
                                  delta=f"{hr.last_forecast_ret*100:+.3f}%")
                        st.caption(direction)
                        st.caption(f"For {hr.target_timestamp:%Y-%m-%d %H:%M} · from {hr.origin_price:,.2f} at {hr.forecast_origin:%H:%M}")
                        if hr.actual_price is not None:
                            st.caption(f"Actual at that time: {hr.actual_price:,.2f}")
                        st.caption(f"Train MAE {hr.train_mae:.5f} · Dir. acc {hr.train_dir_acc:.1%}")

    # ---- Next-day forecast ----
    with tab_nextday:
        st.markdown("##### Next trading session — close & direction")
        if st.button("\u25B6\uFE0F Train & Forecast (Next-Day)", key="niftylab_run_nextday"):
            with st.spinner("Training next-day model..."):
                nd = train_nextday_model(ds.daily, cfg)
            st.session_state["niftylab_nextday_result"] = nd

        if "niftylab_nextday_result" in st.session_state:
            nd = st.session_state["niftylab_nextday_result"]
            c1, c2, c3 = st.columns(3)
            c1.metric("Last session close", f"{nd.last_close:,.2f}", help=str(nd.last_session.date()))
            direction = "\U0001F7E2 UP" if nd.forecast_dir == 1 else "\U0001F534 DOWN"
            c2.metric("Next-session forecast close", f"{nd.forecast_close:,.2f}",
                      delta=f"{nd.forecast_ret*100:+.2f}%")
            c3.metric("Predicted direction", direction)
            st.caption(
                f"Train MAE {nd.train_mae:.5f} · RMSE {nd.train_rmse:.5f} · "
                f"Dir. accuracy {nd.train_dir_acc:.1%} · n={nd.n_train_rows}"
            )

    # ---- Historical backtest ----
    with tab_backtest:
        st.markdown("##### Experiment 1 — Walk-forward")
        st.caption("Rolling folds, retrained before each test window. Chronological only — no leakage.")

        target_choice = st.selectbox(
            "Backtest target", ["Next-Day Model"] + [f"Intraday {h}m" for h in cfg.intraday_horizons_min],
            key="niftylab_backtest_target",
        )

        if st.button("\u25B6\uFE0F Run Walk-Forward", key="niftylab_run_backtest"):
            with st.spinner("Running walk-forward experiment..."):
                if target_choice == "Next-Day Model":
                    wf = run_nextday_walkforward(ds.daily, cfg)
                else:
                    h = int(target_choice.split()[1].rstrip("m"))
                    wf = run_intraday_walkforward(ds.minute, cfg, horizon_min=h)
            st.session_state["niftylab_backtest_wf"] = wf

        if "niftylab_backtest_wf" in st.session_state:
            wf = st.session_state["niftylab_backtest_wf"]

            st.markdown(f"**{wf.name}**")
            wf_frame = wf.to_frame()
            if wf_frame.empty:
                st.warning("Not enough data for a walk-forward backtest on this target.")
            else:
                st.dataframe(wf_frame, use_container_width=True, hide_index=True)
                avg_acc = wf_frame["Dir. Accuracy"].mean()
                avg_baseline = wf_frame["Baseline Dir. Accuracy"].mean()
                st.caption(f"Average directional accuracy: {avg_acc:.1%} vs. baseline {avg_baseline:.1%}")

        # ---- Experiment 2: historical one-month-ahead forecast ----
        st.divider()
        st.markdown("##### Experiment 2 — Historical one-month-ahead forecast")
        st.caption(
            "Pick a historical target trading date. The model sees only information up to one "
            "calendar month before it, forecasts that session's close in a single step, and the "
            "result is compared with the actual close of that exact session."
        )
        first_d, last_d = ds.daily.index[0].date(), ds.daily.index[-1].date()
        target_date = st.date_input(
            "Target trading date", value=last_d, min_value=first_d, max_value=last_d,
            key="niftylab_hist_target_date",
        )
        if st.button("\u25B6\uFE0F Run Historical Forecast", key="niftylab_run_hist_forecast"):
            with st.spinner("Training on information available at the cutoff..."):
                try:
                    st.session_state["niftylab_hist_result"] = run_historical_forecast(ds.daily, cfg, target_date)
                    st.session_state.pop("niftylab_hist_error", None)
                except HistoricalForecastError as e:
                    st.session_state["niftylab_hist_error"] = str(e)
                    st.session_state.pop("niftylab_hist_result", None)

        if "niftylab_hist_error" in st.session_state:
            st.warning(st.session_state["niftylab_hist_error"])
        elif "niftylab_hist_result" in st.session_state:
            hr = st.session_state["niftylab_hist_result"]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Cutoff date", str(hr.cutoff_date.date()), help=f"Last session used: {hr.last_session_used.date()}")
            c2.metric("Target date", str(hr.target_date.date()), help=f"{hr.horizon_sessions} sessions after the last session used")
            c3.metric("Predicted close", f"{hr.predicted_price:,.2f}")
            c4.metric("Actual close", f"{hr.actual_price:,.2f}")
            st.metric("Forecast error (predicted − actual)", f"{hr.forecast_error:+,.2f}",
                      delta=f"{hr.pct_error*100:+.2f}%", delta_color="off")
            st.caption(
                f"Naive 'no change' forecast ({hr.baseline_price:,.2f}) would have erred by "
                f"{hr.baseline_error:+,.2f}. Trained on {hr.n_train_rows} rows; latest training "
                f"label outcome {hr.last_train_label_date.date()} (≤ cutoff)."
            )
            st.dataframe(hr.to_frame(), use_container_width=True, hide_index=True)

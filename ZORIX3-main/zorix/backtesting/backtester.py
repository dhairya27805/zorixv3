"""
zorix/backtesting/backtester.py

Backtester 2.0 (Section 24/25, Priority 3 of this phase).

Design decisions, stated explicitly because they affect how to read the
results:

  - Signals fed into this backtester MUST be out-of-sample (walk-forward
    OOF predictions — see models/walkforward.py's `run_walk_forward(...,
    collect_predictions=True)`), never in-sample fitted predictions. This
    backtester does not fit any models itself; it only consumes a
    pre-computed signal series and simulates trading it. Feeding it
    in-sample predictions would silently turn this into "optimizing
    against the final test set," which the brief explicitly forbids —
    so it's on the caller to pass OOS signals in.
  - Single position at a time (no pyramiding, no overlapping trades) —
    simplest defensible model of "act on today's model signal."
  - Entry executes at the NEXT bar's Open after a signal fires (not the
    same day's Close, which is what generated the signal) — avoids
    look-ahead.
  - Exit on whichever comes first: stop-loss hit (Low crosses stop for a
    long), target hit (High crosses target), or `max_holding_days`
    elapsed (forced exit at Close).
  - Transaction costs and slippage are charged on both entry and exit,
    in basis points of trade value.
  - Position size comes from risk/risk_engine.py (ATR-based stop +
    risk-per-trade sizing), not "100% of capital."
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd

from zorix.backtesting.metrics import PerformanceMetrics, compute_performance_metrics
from zorix.risk.risk_engine import build_risk_assessment


@dataclass
class Trade:
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    direction: str
    entry_price: float
    exit_price: float
    shares: float
    exit_reason: str          # "stop" | "target" | "time" | "end_of_data"
    gross_return_pct: float
    net_return_pct: float
    holding_days: int
    traded_value: float       # entry_value + exit_value, for turnover


@dataclass
class BacktestResult:
    equity_curve: pd.Series
    daily_returns: pd.Series
    trades: List[Trade]
    metrics: PerformanceMetrics
    buy_hold_equity: Optional[pd.Series]
    buy_hold_metrics: Optional[PerformanceMetrics]
    benchmark_equity: Optional[pd.Series]
    benchmark_metrics: Optional[PerformanceMetrics]


def _build_buy_hold_equity(close: pd.Series, initial_capital: float) -> pd.Series:
    rets = close.pct_change().fillna(0)
    return initial_capital * (1 + rets).cumprod()


class Backtester:
    def __init__(self, cfg):
        self.cfg = cfg

    def run(
        self,
        price_df: pd.DataFrame,          # needs Open, High, Low, Close, ATR — indexed by date
        signals: pd.Series,               # "BUY"/"SELL"/"HOLD"/"NO TRADE", aligned to price_df.index, OOS only
        benchmark_close: Optional[pd.Series] = None,
    ) -> BacktestResult:
        cfg = self.cfg
        idx = price_df.index
        dates = list(idx)

        capital = cfg.PORTFOLIO_CAPITAL
        equity = capital
        equity_curve = {}
        trades: List[Trade] = []

        cost_rate = (cfg.TRANSACTION_COST_BPS + cfg.SLIPPAGE_BPS) / 10_000.0

        position = None  # dict with direction, entry_price, stop, target, shares, entry_date, entry_idx
        i = 0
        n = len(dates)

        # Only simulate over dates where we actually have a signal (OOS region).
        valid_signal_dates = signals.dropna().index
        if len(valid_signal_dates) == 0:
            equity_curve_series = pd.Series([capital], index=[dates[0]] if dates else [])
            empty_returns = pd.Series(dtype=float)
            empty_metrics = compute_performance_metrics(equity_curve_series, empty_returns, pd.Series(dtype=float), [], [])
            return BacktestResult(equity_curve_series, empty_returns, [], empty_metrics, None, None, None, None)

        start_i = dates.index(valid_signal_dates[0])

        for i in range(start_i, n):
            date = dates[i]
            row = price_df.iloc[i]
            equity_curve[date] = equity  # mark-to-market before any action this bar

            if position is not None:
                exit_price, exit_reason = None, None

                if position["direction"] == "BUY":
                    if row["Low"] <= position["stop"]:
                        exit_price, exit_reason = position["stop"], "stop"
                    elif row["High"] >= position["target"]:
                        exit_price, exit_reason = position["target"], "target"
                else:  # SELL / short
                    if row["High"] >= position["stop"]:
                        exit_price, exit_reason = position["stop"], "stop"
                    elif row["Low"] <= position["target"]:
                        exit_price, exit_reason = position["target"], "target"

                holding_days = i - position["entry_idx"]
                if exit_price is None and holding_days >= cfg.MAX_HOLDING_DAYS:
                    exit_price, exit_reason = row["Close"], "time"
                if exit_price is None and i == n - 1:
                    exit_price, exit_reason = row["Close"], "end_of_data"

                if exit_price is not None:
                    entry_value = position["shares"] * position["entry_price"]
                    exit_value = position["shares"] * exit_price

                    if position["direction"] == "BUY":
                        gross_pnl = exit_value - entry_value
                    else:
                        gross_pnl = entry_value - exit_value  # short: profit when price falls

                    costs = (entry_value + exit_value) * cost_rate
                    net_pnl = gross_pnl - costs
                    equity += net_pnl

                    gross_return_pct = gross_pnl / entry_value if entry_value else 0.0
                    net_return_pct = net_pnl / entry_value if entry_value else 0.0

                    trades.append(Trade(
                        entry_date=position["entry_date"], exit_date=date,
                        direction=position["direction"], entry_price=position["entry_price"],
                        exit_price=exit_price, shares=position["shares"], exit_reason=exit_reason,
                        gross_return_pct=gross_return_pct, net_return_pct=net_return_pct,
                        holding_days=holding_days, traded_value=entry_value + exit_value,
                    ))
                    position = None
                    equity_curve[date] = equity

            # Consider a new entry only if flat and not the last bar.
            if position is None and i < n - 1:
                sig = signals.get(date)
                if sig in ("BUY", "SELL"):
                    atr = row.get("ATR", np.nan)
                    next_row = price_df.iloc[i + 1]
                    entry_price = next_row["Open"]

                    assessment = build_risk_assessment(
                        entry=entry_price, direction=sig, atr=atr,
                        capital=equity, risk_per_trade_pct=cfg.RISK_PER_TRADE,
                        stop_mult=cfg.STOP_LOSS_ATR_MULT, rr_ratio=cfg.TARGET_RR_RATIO,
                        max_position_pct=cfg.MAX_POSITION_PCT,
                        max_portfolio_risk_pct=cfg.MAX_PORTFOLIO_RISK,
                    )
                    if assessment.valid and assessment.position and assessment.position.shares > 0:
                        position = {
                            "direction": sig, "entry_price": entry_price,
                            "stop": assessment.stop_loss, "target": assessment.target,
                            "shares": assessment.position.shares,
                            "entry_date": dates[i + 1], "entry_idx": i + 1,
                        }

        equity_series = pd.Series(equity_curve).sort_index()
        daily_returns = equity_series.pct_change().fillna(0)
        trade_returns = pd.Series([t.net_return_pct for t in trades])
        trade_values = [t.traded_value for t in trades]
        holding_days_list = [t.holding_days for t in trades]

        metrics = compute_performance_metrics(
            equity_series, daily_returns, trade_returns, trade_values, holding_days_list,
            risk_free=cfg.RISK_FREE_RATE,
        )

        # ---- Buy & hold comparison over the same window ----
        window_close = price_df["Close"].loc[equity_series.index[0]:equity_series.index[-1]]
        bh_equity = _build_buy_hold_equity(window_close, capital)
        bh_returns = bh_equity.pct_change().fillna(0)
        bh_metrics = compute_performance_metrics(bh_equity, bh_returns, pd.Series(dtype=float), [], [])

        bench_equity, bench_metrics = None, None
        if benchmark_close is not None:
            bench_window = benchmark_close.reindex(equity_series.index, method="ffill")
            if bench_window.notna().sum() > 1:
                bench_equity = _build_buy_hold_equity(bench_window.dropna(), capital)
                bench_returns = bench_equity.pct_change().fillna(0)
                bench_metrics = compute_performance_metrics(bench_equity, bench_returns, pd.Series(dtype=float), [], [])

        return BacktestResult(
            equity_curve=equity_series, daily_returns=daily_returns, trades=trades, metrics=metrics,
            buy_hold_equity=bh_equity, buy_hold_metrics=bh_metrics,
            benchmark_equity=bench_equity, benchmark_metrics=bench_metrics,
        )

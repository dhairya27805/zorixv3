"""
zorix/utils/helpers.py

Shared, independently-testable performance-metric functions used by
walk-forward evaluation and (later) the backtester. Pure functions over
numpy/pandas — no I/O, so they're trivial to unit test (see tests/).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def sharpe_ratio(returns: pd.Series, periods_per_year: int = 252, risk_free: float = 0.0) -> float:
    returns = returns.dropna()
    if returns.empty or returns.std() == 0:
        return float("nan")
    excess = returns - risk_free / periods_per_year
    return float(np.sqrt(periods_per_year) * excess.mean() / excess.std())


def sortino_ratio(returns: pd.Series, periods_per_year: int = 252, risk_free: float = 0.0) -> float:
    returns = returns.dropna()
    if returns.empty:
        return float("nan")
    excess = returns - risk_free / periods_per_year
    downside = excess[excess < 0]
    if downside.std() == 0 or downside.empty:
        return float("nan")
    return float(np.sqrt(periods_per_year) * excess.mean() / downside.std())


def max_drawdown(equity_curve: pd.Series) -> float:
    """Returns max drawdown as a negative fraction, e.g. -0.23 for -23%."""
    equity_curve = equity_curve.dropna()
    if equity_curve.empty:
        return float("nan")
    running_max = equity_curve.cummax()
    drawdown = equity_curve / running_max - 1
    return float(drawdown.min())


def calmar_ratio(cagr: float, mdd: float) -> float:
    if mdd == 0 or np.isnan(mdd):
        return float("nan")
    return float(cagr / abs(mdd))


def cagr(equity_curve: pd.Series, periods_per_year: int = 252) -> float:
    equity_curve = equity_curve.dropna()
    if len(equity_curve) < 2:
        return float("nan")
    n_periods = len(equity_curve)
    total_return = equity_curve.iloc[-1] / equity_curve.iloc[0]
    years = n_periods / periods_per_year
    if years <= 0 or total_return <= 0:
        return float("nan")
    return float(total_return ** (1 / years) - 1)


def profit_factor(trade_returns: pd.Series) -> float:
    trade_returns = trade_returns.dropna()
    gains = trade_returns[trade_returns > 0].sum()
    losses = -trade_returns[trade_returns < 0].sum()
    if losses == 0:
        return float("inf") if gains > 0 else float("nan")
    return float(gains / losses)


def win_rate(trade_returns: pd.Series) -> float:
    trade_returns = trade_returns.dropna()
    if trade_returns.empty:
        return float("nan")
    return float((trade_returns > 0).mean())

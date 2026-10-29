"""
zorix/backtesting/metrics.py

Thin metrics layer for the backtester: reuses the already-tested Sharpe/
Sortino/CAGR/max-drawdown/profit-factor/win-rate functions from
utils/helpers.py (kept there since walk-forward validation uses them too)
and adds turnover, which is backtest-specific.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import pandas as pd

from zorix.utils.helpers import (cagr, calmar_ratio, max_drawdown,
                                  profit_factor, sharpe_ratio,
                                  sortino_ratio, win_rate)


@dataclass
class PerformanceMetrics:
    total_return: float
    cagr: float
    volatility: float
    sharpe: float
    sortino: float
    calmar: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    n_trades: int
    avg_holding_days: float
    turnover_annualized: float


def turnover_annualized(trade_values: List[float], avg_equity: float, n_years: float) -> float:
    """Sum of (entry value + exit value) traded, divided by average equity
    and by the number of years — a rough 'how many times over did we churn
    the portfolio per year' figure."""
    if avg_equity <= 0 or n_years <= 0:
        return float("nan")
    total_traded = sum(trade_values)
    return total_traded / avg_equity / n_years


def compute_performance_metrics(
    equity_curve: pd.Series,
    daily_returns: pd.Series,
    trade_returns: pd.Series,
    trade_values: List[float],
    holding_days: List[int],
    periods_per_year: int = 252,
    risk_free: float = 0.0,
) -> PerformanceMetrics:
    n_years = len(equity_curve) / periods_per_year if len(equity_curve) else float("nan")
    total_return = float(equity_curve.iloc[-1] / equity_curve.iloc[0] - 1) if len(equity_curve) > 1 else float("nan")
    c = cagr(equity_curve, periods_per_year)
    mdd = max_drawdown(equity_curve)

    return PerformanceMetrics(
        total_return=total_return,
        cagr=c,
        volatility=float(daily_returns.std() * (periods_per_year ** 0.5)) if len(daily_returns) else float("nan"),
        sharpe=sharpe_ratio(daily_returns, periods_per_year, risk_free),
        sortino=sortino_ratio(daily_returns, periods_per_year, risk_free),
        calmar=calmar_ratio(c, mdd),
        max_drawdown=mdd,
        win_rate=win_rate(trade_returns),
        profit_factor=profit_factor(trade_returns),
        n_trades=int(trade_returns.dropna().shape[0]),
        avg_holding_days=float(pd.Series(holding_days).mean()) if holding_days else float("nan"),
        turnover_annualized=turnover_annualized(trade_values, equity_curve.mean() if len(equity_curve) else float("nan"), n_years),
    )

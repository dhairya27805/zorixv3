"""
zorix/features/technical.py

Price/return/volatility features (Section 10) plus the existing technical
indicators (RSI, MACD, Bollinger Bands) carried over from app.py/notebook.

Every function here is a pure function of a single OHLCV DataFrame — no
lookahead is possible because everything is computed with rolling/shift
operations over already-realized prices.
"""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd


def add_return_features(df: pd.DataFrame, windows: List[int]) -> pd.DataFrame:
    """1D/2D/3D/5D/10D/20D simple + log returns, overnight & intraday returns, gap."""
    out = df.copy()
    close = out["Close"]

    for w in windows:
        out[f"Return_{w}D"] = close.pct_change(w)
        out[f"LogReturn_{w}D"] = np.log(close / close.shift(w))

    out["Overnight_Return"] = out["Open"] / out["Close"].shift(1) - 1
    out["Intraday_Return"] = out["Close"] / out["Open"] - 1
    out["Gap"] = out["Open"] - out["Close"].shift(1)
    out["Gap_Pct"] = out["Gap"] / out["Close"].shift(1)
    return out


def add_volatility_features(df: pd.DataFrame, windows: List[int]) -> pd.DataFrame:
    """Rolling volatility, downside/upside volatility, skew, kurtosis."""
    out = df.copy()
    daily_ret = out["Close"].pct_change()

    for w in windows:
        roll = daily_ret.rolling(w)
        out[f"Volatility_{w}D"] = roll.std() * np.sqrt(252)

        downside = daily_ret.where(daily_ret < 0, 0.0)
        upside = daily_ret.where(daily_ret > 0, 0.0)
        out[f"DownsideVol_{w}D"] = downside.rolling(w).std() * np.sqrt(252)
        out[f"UpsideVol_{w}D"] = upside.rolling(w).std() * np.sqrt(252)

        out[f"Skew_{w}D"] = daily_ret.rolling(w).skew()
        out[f"Kurtosis_{w}D"] = daily_ret.rolling(w).kurt()

    out["Returns"] = daily_ret
    return out


def add_classic_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """RSI, MACD, Bollinger Bands. Uses `ta` if available; falls back to a
    hand-rolled implementation (identical formulas) if `ta` isn't installed,
    so the app never crashes over a missing optional dependency.
    """
    out = df.copy()
    close = out["Close"]

    try:
        from ta.momentum import RSIIndicator
        from ta.trend import MACD
        from ta.volatility import BollingerBands

        out["RSI"] = RSIIndicator(close=close.squeeze()).rsi()
        macd = MACD(close=close.squeeze())
        out["MACD"] = macd.macd()
        out["MACD_Signal"] = macd.macd_signal()
        out["MACD_Hist"] = macd.macd_diff()
        bb = BollingerBands(close=close.squeeze())
        out["BB_High"] = bb.bollinger_hband()
        out["BB_Low"] = bb.bollinger_lband()
        out["BB_Width"] = (out["BB_High"] - out["BB_Low"]) / close
    except ImportError:
        out["RSI"] = _rsi_fallback(close)
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        out["MACD"] = ema12 - ema26
        out["MACD_Signal"] = out["MACD"].ewm(span=9, adjust=False).mean()
        out["MACD_Hist"] = out["MACD"] - out["MACD_Signal"]
        sma20 = close.rolling(20).mean()
        std20 = close.rolling(20).std()
        out["BB_High"] = sma20 + 2 * std20
        out["BB_Low"] = sma20 - 2 * std20
        out["BB_Width"] = (out["BB_High"] - out["BB_Low"]) / close

    return out


def add_atr(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """Average True Range — used by the risk engine to size stop-loss/target
    distances off realized volatility rather than an arbitrary fixed percent.
    """
    out = df.copy()
    prev_close = out["Close"].shift(1)
    tr = pd.concat([
        out["High"] - out["Low"],
        (out["High"] - prev_close).abs(),
        (out["Low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    out["ATR"] = tr.rolling(window).mean()
    out["ATR_Pct"] = out["ATR"] / out["Close"]
    return out


def _rsi_fallback(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def build_technical_features(df: pd.DataFrame, cfg) -> pd.DataFrame:
    """Convenience pipeline: classic indicators + returns + volatility + ATR."""
    out = add_classic_indicators(df)
    out = add_return_features(out, cfg.RETURN_WINDOWS)
    out = add_volatility_features(out, cfg.VOLATILITY_WINDOWS)
    out = add_atr(out, getattr(cfg, "ATR_WINDOW", 14))
    return out

"""
zorix/nifty_lab/features.py

Feature engineering for Mode 1. Follows the same no-leakage discipline as
zorix/features/target.py: every feature uses only data available up to and
including the bar/day it is computed on (trailing rolling windows, shifted
lags). Forward targets are built separately and are never fed back in as
features.

Volume is never used (see NiftyLabConfig.use_volume / csv_provider).
"""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Intraday (1-minute) features
# ---------------------------------------------------------------------

def build_intraday_features(minute: pd.DataFrame) -> pd.DataFrame:
    """Trailing technical features on 1-min OHLC. No Volume."""
    out = minute.copy()

    out["ret_1"] = out["Close"].pct_change(1)
    out["ret_5"] = out["Close"].pct_change(5)
    out["ret_15"] = out["Close"].pct_change(15)

    for w in (5, 15, 30):
        out[f"sma_{w}"] = out["Close"].rolling(w).mean()
        out[f"std_{w}"] = out["Close"].pct_change().rolling(w).std()

    out["range_pct"] = (out["High"] - out["Low"]) / out["Close"]
    out["body_pct"] = (out["Close"] - out["Open"]) / out["Close"]

    # RSI(14) on 1-min closes, trailing only.
    delta = out["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    out["rsi_14"] = 100 - (100 / (1 + rs))

    # Time-of-day features. Assignment as plain arrays to DataFrame
    # columns (not a method chained off the index) — this form is the
    # one that works; `out.index.dayofweek` chained into a `.clip(...)`
    # call on the same line previously failed here.
    minute_of_day = out.index.hour * 60 + out.index.minute
    out["minute_of_day"] = minute_of_day.to_numpy()
    out["day_of_week"] = out.index.dayofweek.to_numpy()

    out = out.dropna(how="any")
    return out


# Two consecutive bars further apart than this many minutes belong to
# different trading sessions (overnight gap, lunch break, a special evening
# session on the same date, ...). A run of up to this many minutes of
# missing bars is still treated as "missing bars inside one session".
DEFAULT_SESSION_GAP_MINUTES = 5


def assign_session_ids(index: pd.DatetimeIndex,
                       session_gap_minutes: int = DEFAULT_SESSION_GAP_MINUTES) -> np.ndarray:
    """Integer session id for each bar of a sorted DatetimeIndex.

    A new session starts whenever the gap to the previous bar is larger than
    ``session_gap_minutes``. This is purely time-gap based, so it also splits
    two sessions that fall on the SAME calendar date (the old
    ``target_ts.day != ts.day`` test could not).
    """
    if len(index) == 0:
        return np.array([], dtype=int)
    gaps = np.diff(index.values).astype("timedelta64[s]").astype(np.int64)
    new_session = gaps > int(session_gap_minutes) * 60
    return np.concatenate([[0], np.cumsum(new_session)]).astype(int)


def build_intraday_targets(feat: pd.DataFrame, horizons_min: List[int],
                           bars: pd.DataFrame = None,
                           session_gap_minutes: int = DEFAULT_SESSION_GAP_MINUTES) -> pd.DataFrame:
    """Forward targets at each intraday horizon, aligned by EXACT TIMESTAMP.

    For every bar at time ``t`` and horizon ``h`` the target is the close of
    the bar stamped exactly ``t + h minutes`` -- never "h rows later". The
    target is valid only if

      * a bar with exactly that timestamp exists, and
      * that bar is in the same trading session as ``t`` (sessions are split
        on gaps larger than ``session_gap_minutes``, or by an explicit
        ``session_id`` column if ``bars`` provides one).

    Otherwise the target is INVALID: ``target_ts_*`` is NaT and
    ``target_close_*`` / ``fwd_ret_*`` / ``fwd_dir_*`` are NaN. No other
    timestamp is ever substituted.

    Added columns per horizon (all derived from the same lookup, so the
    target time, target price and return always describe one observation):
      target_ts_{h}m, target_close_{h}m, fwd_ret_{h}m, fwd_dir_{h}m

    ``bars`` is the full bar series used for the lookup and for session
    detection (default: ``feat`` itself). Pass the raw minute frame when
    ``feat`` has had rows dropped (e.g. rolling-window warm-up).
    """
    out = feat.copy()
    src = feat if bars is None else bars

    if not src.index.is_unique:
        raise ValueError("build_intraday_targets: duplicate timestamps in bar index")
    if not src.index.is_monotonic_increasing:
        raise ValueError("build_intraday_targets: bar index must be sorted ascending")

    if "session_id" in src.columns:
        if src["session_id"].isna().any():
            raise ValueError("build_intraday_targets: session_id contains nulls")
        codes = pd.factorize(src["session_id"])[0].astype(float)
    else:
        codes = assign_session_ids(src.index, session_gap_minutes).astype(float)

    src_close = src["Close"]
    src_session = pd.Series(codes, index=src.index)
    origin_session = src_session.reindex(out.index).to_numpy()

    for h in horizons_min:
        target_ts = out.index + pd.Timedelta(minutes=h)

        # Exact-timestamp lookup (reindex returns NaN when the stamp is absent).
        target_close = src_close.reindex(target_ts).to_numpy(dtype=float)
        target_session = src_session.reindex(target_ts).to_numpy()

        valid = ~np.isnan(target_close) & (target_session == origin_session)

        out[f"target_ts_{h}m"] = target_ts.where(valid)  # NaT where invalid
        out[f"target_close_{h}m"] = np.where(valid, target_close, np.nan)
        out[f"fwd_ret_{h}m"] = out[f"target_close_{h}m"] / out["Close"] - 1
        # NaN (not 0) when the target is invalid, so "no target" is never
        # mistaken for "price went down".
        out[f"fwd_dir_{h}m"] = np.where(valid, (out[f"fwd_ret_{h}m"] > 0).astype(float), np.nan)

    return out


# ---------------------------------------------------------------------
# Daily features (built from the resampled daily OHLC)
# ---------------------------------------------------------------------

def build_daily_features(daily: pd.DataFrame) -> pd.DataFrame:
    out = daily.copy()

    out["ret_1d"] = out["Close"].pct_change(1)
    out["ret_5d"] = out["Close"].pct_change(5)

    for w in (5, 10, 20):
        out[f"sma_{w}d"] = out["Close"].rolling(w).mean()
        out[f"std_{w}d"] = out["Close"].pct_change().rolling(w).std()

    out["range_pct"] = (out["High"] - out["Low"]) / out["Close"]
    out["gap_pct"] = (out["Open"] - out["Close"].shift(1)) / out["Close"].shift(1)

    delta = out["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    out["rsi_14"] = 100 - (100 / (1 + rs))

    # Plain array assignment — this second occurrence was already fine
    # (no .clip() chained onto the same line), kept identical in form to
    # the intraday version above for consistency.
    out["day_of_week"] = out.index.dayofweek.to_numpy()

    out = out.dropna(how="any")
    return out


def build_nextday_targets(feat: pd.DataFrame) -> pd.DataFrame:
    """Next trading session's close (regression target) and direction (classification target)."""
    out = feat.copy()
    out["next_close"] = out["Close"].shift(-1)
    out["next_ret"] = out["next_close"] / out["Close"] - 1
    out["next_dir"] = (out["next_ret"] > 0).astype(int)
    return out

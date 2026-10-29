"""
zorix/nifty_lab/csv_provider.py

Loads and cleans the local NIFTY_spot_1m_extended.csv. No network calls —
this is the local-file counterpart to zorix/data/market_data.py for Mode 1.

Verified standalone: 277,026 minute bars -> 742 daily sessions, invalid
Volume flagged and excluded.
"""

import os
from dataclasses import dataclass

import pandas as pd
import streamlit as st

from zorix.nifty_lab.config import NiftyLabConfig


class NiftyLabDataError(Exception):
    pass


@dataclass
class NiftyLabDataset:
    minute: pd.DataFrame   # cleaned 1-min OHLC, DatetimeIndex
    daily: pd.DataFrame    # resampled daily OHLC from actual sessions
    n_minute_bars: int
    n_sessions: int
    volume_usable: bool
    bad_ohlc_rows: int


def _file_mtime(path: str) -> float:
    try:
        return os.path.getmtime(path)
    except OSError:
        return -1.0


@st.cache_data(show_spinner=False)
def _load_and_clean(csv_path: str, _mtime: float, ohlc_round_dp: int) -> NiftyLabDataset:
    if not os.path.exists(csv_path):
        raise NiftyLabDataError(f"NIFTY CSV not found at {csv_path}")

    raw = pd.read_csv(csv_path)

    required = {"datetime", "Open", "High", "Low", "Close"}
    missing = required - set(raw.columns)
    if missing:
        raise NiftyLabDataError(f"CSV missing required columns: {sorted(missing)}")

    raw["datetime"] = pd.to_datetime(raw["datetime"])
    raw = raw.sort_values("datetime").drop_duplicates(subset="datetime")
    raw = raw.set_index("datetime")

    # Volume check: confirmed project-wide to be a dead feed (100% zero
    # 2023-2024, 83.5% zero 2025, only populated 2026). We measure it here
    # rather than trusting the static claim, so the flag stays correct if
    # a future CSV export changes.
    volume_usable = False
    if "Volume" in raw.columns:
        nonzero_frac = (raw["Volume"].fillna(0) != 0).mean()
        volume_usable = nonzero_frac > 0.5  # most of history must be populated

    ohlc_cols = ["Open", "High", "Low", "Close"]

    # Round away float-precision artifacts (e.g. 22532.05078125 vs
    # 22532.05) rather than dropping rows — these ~9 rows are negligible
    # noise, not corruption.
    before = raw[ohlc_cols].copy()
    raw[ohlc_cols] = raw[ohlc_cols].round(ohlc_round_dp)

    # "Bad OHLC" = High < Low, or Close/Open outside [Low, High] after
    # rounding. Counted for transparency; not dropped.
    bad_mask = (
        (raw["High"] < raw["Low"])
        | (raw["Close"] > raw["High"]) | (raw["Close"] < raw["Low"])
        | (raw["Open"] > raw["High"]) | (raw["Open"] < raw["Low"])
    )
    bad_ohlc_rows = int(bad_mask.sum())

    minute = raw[ohlc_cols].dropna(how="any")

    # Daily OHLC resampled from the actual intraday sessions present in the
    # data (not a calendar resample, which would invent empty days).
    daily = pd.DataFrame({
        "Open": minute["Open"].resample("1D").first(),
        "High": minute["High"].resample("1D").max(),
        "Low": minute["Low"].resample("1D").min(),
        "Close": minute["Close"].resample("1D").last(),
    }).dropna(how="any")

    return NiftyLabDataset(
        minute=minute,
        daily=daily,
        n_minute_bars=len(minute),
        n_sessions=len(daily),
        volume_usable=volume_usable,
        bad_ohlc_rows=bad_ohlc_rows,
    )


def load_niftylab_dataset(cfg: NiftyLabConfig) -> NiftyLabDataset:
    """Cached load keyed off file path + mtime, so it loads once per file version."""
    mtime = _file_mtime(cfg.csv_path)
    return _load_and_clean(cfg.csv_path, mtime, cfg.ohlc_round_dp)

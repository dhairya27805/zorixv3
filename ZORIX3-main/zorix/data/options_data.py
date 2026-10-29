"""
zorix/data/options_data.py

Options-chain data provider abstraction (Section 15/16/39).

yfinance's options data is delayed, incomplete for Indian underlyings, and
has no reliable historical open-interest — it is NOT wired up here because
that would mean either silently showing stale/misleading numbers or (worse)
fabricating a chain, both of which the spec explicitly forbids.

Instead:
    OptionsDataProvider   — abstract interface the rest of Zorix codes against
    NullOptionsProvider   — always reports "unavailable"; used until you
                            plug in a real provider, so the app never crashes
                            and never fakes numbers
    CustomOptionsProvider — fill in the TODOs with your own API's calls.
                            Nothing else in the codebase needs to change once
                            you do — signals/signal_engine.py and app.py both
                            code against OptionsDataProvider, not this class.

Wire it up in app.py / config by swapping which class gets instantiated.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Optional

import pandas as pd


@dataclass
class OptionsSnapshot:
    """Structured options-chain summary for one underlying at one timestamp."""
    symbol: str
    timestamp: str
    spot: Optional[float]
    atm_strike: Optional[float]
    pcr_oi: Optional[float]           # Put OI / Call OI
    pcr_volume: Optional[float]
    atm_iv: Optional[float]
    iv_percentile: Optional[float]
    max_call_oi_strike: Optional[float]
    max_put_oi_strike: Optional[float]
    max_pain: Optional[float]
    chain: Optional[pd.DataFrame]     # columns: strike, call_oi, put_oi, call_iv, put_iv, ...
    available: bool = True
    unavailable_reason: Optional[str] = None


class OptionsDataProvider(abc.ABC):
    @abc.abstractmethod
    def get_snapshot(self, symbol: str) -> OptionsSnapshot:
        raise NotImplementedError


class NullOptionsProvider(OptionsDataProvider):
    """Default provider. Always returns a clearly-labeled 'unavailable' snapshot."""

    def get_snapshot(self, symbol: str) -> OptionsSnapshot:
        return OptionsSnapshot(
            symbol=symbol,
            timestamp="",
            spot=None, atm_strike=None, pcr_oi=None, pcr_volume=None,
            atm_iv=None, iv_percentile=None,
            max_call_oi_strike=None, max_put_oi_strike=None, max_pain=None,
            chain=None,
            available=False,
            unavailable_reason=(
                "No options data provider configured. Implement "
                "CustomOptionsProvider in zorix/data/options_data.py "
                "against your F&O data API."
            ),
        )


class CustomOptionsProvider(OptionsDataProvider):
    """
    Template for your F&O data API. Fill in `_fetch_raw_chain` and
    `_fetch_spot` with real calls to your provider — do not fabricate
    values if a call fails; raise or return available=False instead.
    """

    def __init__(self, api_client=None):
        self.client = api_client  # e.g. your authenticated SDK/session

    def _fetch_raw_chain(self, symbol: str) -> Optional[pd.DataFrame]:
        # TODO: replace with a real call, e.g.:
        #   raw = self.client.get_option_chain(symbol)
        #   return pd.DataFrame(raw)  # columns: strike, call_oi, put_oi, call_iv, put_iv, ...
        return None

    def _fetch_spot(self, symbol: str) -> Optional[float]:
        # TODO: replace with a real call.
        return None

    def get_snapshot(self, symbol: str) -> OptionsSnapshot:
        chain = self._fetch_raw_chain(symbol)
        spot = self._fetch_spot(symbol)

        if chain is None or chain.empty or spot is None:
            return OptionsSnapshot(
                symbol=symbol, timestamp="", spot=spot,
                atm_strike=None, pcr_oi=None, pcr_volume=None,
                atm_iv=None, iv_percentile=None,
                max_call_oi_strike=None, max_put_oi_strike=None, max_pain=None,
                chain=None, available=False,
                unavailable_reason="Provider returned no chain data for this symbol.",
            )

        atm_strike = float(chain.iloc[(chain["strike"] - spot).abs().argsort()[:1]]["strike"].values[0])
        total_call_oi = chain["call_oi"].sum()
        total_put_oi = chain["put_oi"].sum()
        pcr_oi = float(total_put_oi / total_call_oi) if total_call_oi else None

        max_call_row = chain.loc[chain["call_oi"].idxmax()]
        max_put_row = chain.loc[chain["put_oi"].idxmax()]

        max_pain = _compute_max_pain(chain)

        atm_row = chain.iloc[(chain["strike"] - atm_strike).abs().argsort()[:1]]
        atm_iv = None
        if {"call_iv", "put_iv"}.issubset(chain.columns) and not atm_row.empty:
            civ, piv = atm_row["call_iv"].values[0], atm_row["put_iv"].values[0]
            vals = [v for v in (civ, piv) if pd.notna(v)]
            atm_iv = float(sum(vals) / len(vals)) if vals else None

        return OptionsSnapshot(
            symbol=symbol,
            timestamp=pd.Timestamp.now().isoformat(),
            spot=spot,
            atm_strike=atm_strike,
            pcr_oi=pcr_oi,
            pcr_volume=None,  # fill in if your provider gives per-strike volume
            atm_iv=atm_iv,
            iv_percentile=None,  # requires historical IV history from your provider
            max_call_oi_strike=float(max_call_row["strike"]),
            max_put_oi_strike=float(max_put_row["strike"]),
            max_pain=max_pain,
            chain=chain,
            available=True,
        )


def _compute_max_pain(chain: pd.DataFrame) -> Optional[float]:
    """
    Max pain = the strike at which total option-writer payout is minimized.
    This is an options-derived reference level, NOT a guaranteed support/
    resistance level (Section 15) — label it as such in the UI.
    """
    if not {"strike", "call_oi", "put_oi"}.issubset(chain.columns):
        return None

    strikes = chain["strike"].values
    pain = []
    for s in strikes:
        call_loss = ((s - chain["strike"]).clip(lower=0) * chain["call_oi"]).sum()
        put_loss = ((chain["strike"] - s).clip(lower=0) * chain["put_oi"]).sum()
        pain.append(call_loss + put_loss)
    return float(strikes[int(pd.Series(pain).idxmin())])

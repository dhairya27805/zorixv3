"""
zorix/signals/signal_engine.py

Turns model probabilities + context (regime, cross-asset, sentiment) into
the structured signal object described in Section 22, with the NO TRADE
gating logic from Section 5.

This module does NOT implement position sizing / stop-loss / risk-engine
(Section 20) — that's Phase 5 and out of scope for this pass. Entry/target/
stop fields are left None here rather than filled with an arbitrary
formula, so the UI can honestly show "not yet available" instead of a
number that looks like risk management but isn't.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

CLASS_NAMES = ["SELL", "HOLD", "BUY"]


@dataclass
class ZorixSignal:
    asset: str
    asset_class: str
    timestamp: str
    horizon_days: int

    sell_probability: float
    hold_probability: float
    buy_probability: float

    expected_return: Optional[float]
    expected_volatility: Optional[float]

    signal: str            # "BUY" | "SELL" | "HOLD" | "NO TRADE"
    confidence: str         # "LOW" | "MEDIUM" | "HIGH"
    no_trade_reason: Optional[str]

    market_regime: Optional[str]
    regime_confidence: Optional[float]

    key_bullish_factors: List[str]
    key_bearish_factors: List[str]

    model_version: str
    data_timestamp: str


def _confidence_bucket(top_prob: float) -> str:
    if top_prob >= 0.70:
        return "HIGH"
    if top_prob >= 0.55:
        return "MEDIUM"
    return "LOW"


def decide_signal(
    proba: np.ndarray,          # shape (3,) = [SELL, HOLD, BUY]
    min_edge: float,
    min_confidence: float,
) -> tuple[str, str, Optional[str]]:
    """Implements the NO TRADE gate (Section 5): if the top class doesn't
    clear both an absolute confidence floor and a margin over the runner-up,
    the engine reports NO TRADE instead of forcing a call.
    """
    order = np.argsort(proba)[::-1]
    top_idx, second_idx = order[0], order[1]
    top_p, second_p = proba[top_idx], proba[second_idx]
    edge = top_p - second_p

    if top_p < min_confidence or edge < min_edge:
        return "NO TRADE", _confidence_bucket(top_p), (
            f"Insufficient model edge (top={CLASS_NAMES[top_idx]} {top_p:.0%}, "
            f"runner-up={CLASS_NAMES[second_idx]} {second_p:.0%}, "
            f"edge={edge:.0%} < required {min_edge:.0%} or confidence < {min_confidence:.0%})"
        )

    return CLASS_NAMES[top_idx], _confidence_bucket(top_p), None


def build_signal(
    asset: str,
    asset_class: str,
    horizon_days: int,
    proba: np.ndarray,
    expected_return: Optional[float],
    expected_volatility: Optional[float],
    market_regime: Optional[str],
    regime_confidence: Optional[float],
    bullish_factors: List[str],
    bearish_factors: List[str],
    min_edge: float,
    min_confidence: float,
    model_version: str,
    data_timestamp: str,
) -> ZorixSignal:
    signal, confidence, no_trade_reason = decide_signal(proba, min_edge, min_confidence)

    return ZorixSignal(
        asset=asset,
        asset_class=asset_class,
        timestamp=pd.Timestamp.now().isoformat(),
        horizon_days=horizon_days,
        sell_probability=float(proba[0]),
        hold_probability=float(proba[1]),
        buy_probability=float(proba[2]),
        expected_return=expected_return,
        expected_volatility=expected_volatility,
        signal=signal,
        confidence=confidence,
        no_trade_reason=no_trade_reason,
        market_regime=market_regime,
        regime_confidence=regime_confidence,
        key_bullish_factors=bullish_factors,
        key_bearish_factors=bearish_factors,
        model_version=model_version,
        data_timestamp=data_timestamp,
    )


def model_consensus(model_probas: Dict[str, np.ndarray]) -> pd.DataFrame:
    """Section 32: 'Model Consensus' panel. Only includes models that
    actually ran (skipped models must never appear here — Section 32)."""
    rows = []
    for name, proba in model_probas.items():
        if proba is None:
            continue
        top_idx = int(np.argmax(proba))
        rows.append({
            "Model": name,
            "Signal": CLASS_NAMES[top_idx],
            "Confidence": f"{proba[top_idx]:.0%}",
        })
    return pd.DataFrame(rows)

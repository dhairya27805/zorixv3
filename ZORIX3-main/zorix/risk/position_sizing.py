"""
zorix/risk/position_sizing.py

Risk-based position sizing (Section 20 / Priority 2 of this phase).

Position size is derived from how much money you're willing to lose if
the stop is hit, not from "BUY = 100% capital":

    risk_amount = capital * risk_per_trade_pct
    shares      = risk_amount / |entry - stop|
    position_value = shares * entry

capped so no single position exceeds `max_position_pct` of capital, since
a very tight stop could otherwise imply an oversized position.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class PositionSizeResult:
    shares: float
    position_value: float
    position_pct_of_capital: float
    risk_amount: float
    capped_by_max_position: bool
    valid: bool
    reason: Optional[str] = None


def calculate_position_size(
    capital: float,
    risk_per_trade_pct: float,
    entry_price: float,
    stop_loss_price: float,
    max_position_pct: float = 0.25,
) -> PositionSizeResult:
    if capital <= 0 or entry_price <= 0:
        return PositionSizeResult(0.0, 0.0, 0.0, 0.0, False, False,
                                   reason="Invalid capital or entry price.")

    stop_distance = abs(entry_price - stop_loss_price)
    if stop_distance <= 0:
        return PositionSizeResult(0.0, 0.0, 0.0, 0.0, False, False,
                                   reason="Stop distance is zero — cannot size a position without risk.")

    risk_amount = capital * risk_per_trade_pct
    shares = risk_amount / stop_distance
    position_value = shares * entry_price

    capped = False
    max_value = capital * max_position_pct
    if position_value > max_value:
        position_value = max_value
        shares = position_value / entry_price
        capped = True

    return PositionSizeResult(
        shares=shares,
        position_value=position_value,
        position_pct_of_capital=position_value / capital,
        risk_amount=risk_amount if not capped else position_value / entry_price * stop_distance,
        capped_by_max_position=capped,
        valid=True,
    )

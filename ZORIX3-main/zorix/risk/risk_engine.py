"""
zorix/risk/risk_engine.py

Risk engine (Section 20/36) — ties together stop-loss/target derivation
and position sizing into the structured risk assessment shown in the
"Risk Management" UI panel.

Stop distance is ATR-based (realized volatility), not an arbitrary fixed
percentage — this is what Section 20 asks for ("mathematically consistent
implementation"), and it naturally widens stops for volatile names and
tightens them for calm ones.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from zorix.risk.position_sizing import PositionSizeResult, calculate_position_size


@dataclass
class RiskAssessment:
    entry: float
    direction: str          # "BUY" | "SELL"
    stop_loss: Optional[float]
    target: Optional[float]
    risk_reward: Optional[float]
    stop_distance_pct: Optional[float]
    position: Optional[PositionSizeResult]
    within_portfolio_risk_cap: bool
    valid: bool
    reason: Optional[str] = None


def compute_stop_target(entry: float, direction: str, atr: float,
                         stop_mult: float, rr_ratio: float):
    """Returns (stop_loss, target). ATR must be a positive, already-computed
    value (features/technical.py:add_atr) — never fabricated here."""
    if atr is None or atr <= 0 or entry <= 0:
        return None, None

    stop_distance = atr * stop_mult
    if direction == "BUY":
        stop = entry - stop_distance
        target = entry + stop_distance * rr_ratio
    elif direction == "SELL":
        stop = entry + stop_distance
        target = entry - stop_distance * rr_ratio
    else:
        return None, None
    return stop, target


def build_risk_assessment(
    entry: float,
    direction: str,
    atr: float,
    capital: float,
    risk_per_trade_pct: float,
    stop_mult: float,
    rr_ratio: float,
    max_position_pct: float,
    open_risk_amount: float = 0.0,
    max_portfolio_risk_pct: float = 0.06,
) -> RiskAssessment:
    """
    open_risk_amount: sum of risk already committed to other open positions
    (in currency), used to check the new trade doesn't breach
    max_portfolio_risk_pct in aggregate (Section 20's "maximum portfolio risk").
    """
    if direction not in ("BUY", "SELL"):
        return RiskAssessment(entry, direction, None, None, None, None, None,
                               within_portfolio_risk_cap=True, valid=False,
                               reason=f"No risk assessment for direction={direction} (only BUY/SELL take positions).")

    stop, target = compute_stop_target(entry, direction, atr, stop_mult, rr_ratio)
    if stop is None:
        return RiskAssessment(entry, direction, None, None, None, None, None,
                               within_portfolio_risk_cap=True, valid=False,
                               reason="ATR unavailable or invalid — cannot size a volatility-based stop.")

    position = calculate_position_size(capital, risk_per_trade_pct, entry, stop, max_position_pct)
    stop_distance_pct = abs(entry - stop) / entry

    projected_total_risk = open_risk_amount + position.risk_amount
    within_cap = projected_total_risk <= capital * max_portfolio_risk_pct

    return RiskAssessment(
        entry=entry, direction=direction, stop_loss=stop, target=target,
        risk_reward=rr_ratio, stop_distance_pct=stop_distance_pct,
        position=position, within_portfolio_risk_cap=within_cap, valid=position.valid,
        reason=None if within_cap else (
            f"Adding this trade would risk {projected_total_risk:,.0f} against a "
            f"portfolio cap of {capital * max_portfolio_risk_pct:,.0f} — reduce size or skip."
        ),
    )

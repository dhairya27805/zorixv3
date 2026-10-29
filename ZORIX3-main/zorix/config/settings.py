"""
zorix/config/settings.py

Central configuration for Zorix 2.0.

Nothing in the feature/model/signal code should hardcode a threshold,
symbol list, or hyperparameter — it should read it from here. This is
what Section 42 of the spec asks for.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict


@dataclass
class CrossAssetSymbols:
    """
    yfinance tickers used for cross-asset / relative-strength features.
    India-focused by default (matches the original notebook's ^GSPC benchmark
    habit) but every field can be overridden per market.
    """
    equity_index: Dict[str, str] = field(default_factory=lambda: {
        "NIFTY50": "^NSEI",
        "BANKNIFTY": "^NSEBANK",
        "SP500": "^GSPC",
        "NASDAQ": "^IXIC",
    })
    sector_indices: Dict[str, str] = field(default_factory=lambda: {
        "NIFTY_IT": "^CNXIT",
        "NIFTY_AUTO": "^CNXAUTO",
        "NIFTY_PHARMA": "^CNXPHARMA",
        "NIFTY_FMCG": "^CNXFMCG",
        "NIFTY_METAL": "^CNXMETAL",
        "NIFTY_ENERGY": "^CNXENERGY",
    })
    volatility: Dict[str, str] = field(default_factory=lambda: {
        "INDIA_VIX": "^INDIAVIX",
        "VIX": "^VIX",
    })
    fx: Dict[str, str] = field(default_factory=lambda: {
        "USDINR": "USDINR=X",
        "DXY": "DX-Y.NYB",
    })
    commodities: Dict[str, str] = field(default_factory=lambda: {
        "GOLD": "GC=F",
        "SILVER": "SI=F",
        "CRUDE_OIL": "CL=F",
        "COPPER": "HG=F",
    })

    def all_symbols(self) -> Dict[str, str]:
        merged: Dict[str, str] = {}
        for group in (self.equity_index, self.sector_indices,
                      self.volatility, self.fx, self.commodities):
            merged.update(group)
        return merged


@dataclass
class InstrumentMeta:
    """
    Display/behavior metadata for one selectable instrument — this is the
    single source of truth the UI reads from instead of hardcoding "is this
    a price or a yield" logic per asset class throughout app.py (Section 24).

    kind: "price" | "rate" | "yield"  — controls metric label + chart type.
    yfinance_symbol: ticker passed to YahooFinanceProvider.
    currency_prefix / currency_suffix: display formatting only — never
        affects the underlying numbers used for features/models.
    yield_scale: some Yahoo treasury-yield tickers (^TNX, ^TYX, ^FVX) report
        the value as yield*10 (e.g. 42.1 for 4.21%) rather than the yield
        itself; dividing by this scale on load corrects it. 1.0 = no
        correction needed.
    ohlc_meaningful: whether a candlestick chart makes sense (False for
        yield instruments where "Open/High/Low/Close" fair less naturally
        as a trading range in the retail sense — a line chart is used
        instead per Section 19).
    """
    yfinance_symbol: str
    kind: str = "price"
    currency_prefix: str = ""
    currency_suffix: str = ""
    yield_scale: float = 1.0
    ohlc_meaningful: bool = True


@dataclass
class ForexPairs:
    """Centralized Forex symbol map (Section 5). USD/INR is displayed with
    a ₹ prefix because the quoted value *is* rupees-per-dollar; USD-quoted
    pairs (EUR/USD, GBP/USD) are shown as plain ratios, matching how they're
    conventionally quoted — never blindly prefixing every asset with ₹
    (Section 5's explicit warning)."""
    pairs: Dict[str, InstrumentMeta] = field(default_factory=lambda: {
        "USD/INR": InstrumentMeta("USDINR=X", kind="rate", currency_prefix="₹"),
        "EUR/USD": InstrumentMeta("EURUSD=X", kind="rate"),
        "GBP/USD": InstrumentMeta("GBPUSD=X", kind="rate"),
        "USD/JPY": InstrumentMeta("USDJPY=X", kind="rate", currency_prefix="¥"),
    })


@dataclass
class BondInstruments:
    """
    Centralized Bonds/yield symbol map (Section 6). Yahoo Finance has no
    complete bond-market dataset, so this uses the yield/index tickers it
    does reliably publish. ^TNX/^TYX/^FVX report yield*10 (yield_scale=10
    corrects this on load); ^IRX already reports the discount rate directly.

    "US 2Y Treasury Yield" and "India 10Y Government Bond Yield" are
    included as best-effort attempts per Section 6 ("if practical") — their
    yfinance tickers are not as reliably available as the others. If a
    fetch fails, DataUnavailableError surfaces "Data unavailable" in the UI
    rather than fabricating a yield (Section 3).
    """
    instruments: Dict[str, InstrumentMeta] = field(default_factory=lambda: {
        "US 10Y Treasury Yield": InstrumentMeta("^TNX", kind="yield", currency_suffix="%",
                                                 yield_scale=10.0, ohlc_meaningful=False),
        "US 5Y Treasury Yield": InstrumentMeta("^FVX", kind="yield", currency_suffix="%",
                                                yield_scale=10.0, ohlc_meaningful=False),
        "US 30Y Treasury Yield": InstrumentMeta("^TYX", kind="yield", currency_suffix="%",
                                                 yield_scale=10.0, ohlc_meaningful=False),
        "US 13-Week T-Bill Yield": InstrumentMeta("^IRX", kind="yield", currency_suffix="%",
                                                   yield_scale=1.0, ohlc_meaningful=False),
        "US 2Y Treasury Yield (best-effort)": InstrumentMeta("^UST2Y", kind="yield", currency_suffix="%",
                                                              yield_scale=1.0, ohlc_meaningful=False),
        "India 10Y Govt Bond Yield (best-effort)": InstrumentMeta("^IN10Y", kind="yield", currency_suffix="%",
                                                                   yield_scale=1.0, ohlc_meaningful=False),
    })


@dataclass
class CommoditySymbols:
    """Centralized Commodities map (Section 4) — mirrors what
    CrossAssetSymbols.commodities already used, wrapped as InstrumentMeta so
    the UI can treat Commodities the same way as Forex/Bonds."""
    instruments: Dict[str, InstrumentMeta] = field(default_factory=lambda: {
        "Gold": InstrumentMeta("GC=F", kind="price", currency_prefix="$"),
        "Silver": InstrumentMeta("SI=F", kind="price", currency_prefix="$"),
        "Crude Oil": InstrumentMeta("CL=F", kind="price", currency_prefix="$"),
        "Copper": InstrumentMeta("HG=F", kind="price", currency_prefix="$"),
    })


@dataclass
class Config:
    # ---- Universe ----
    TICKER: str = "RELIANCE.NS"
    BENCHMARK: str = "^NSEI"          # sector-relative-strength default benchmark
    START_DATE: str = "2015-01-01"
    END_DATE: str = field(default_factory=lambda: datetime.today().strftime("%Y-%m-%d"))
    CROSS_ASSETS: CrossAssetSymbols = field(default_factory=CrossAssetSymbols)
    FOREX: ForexPairs = field(default_factory=ForexPairs)
    BONDS: BondInstruments = field(default_factory=BondInstruments)
    COMMODITIES_META: CommoditySymbols = field(default_factory=CommoditySymbols)

    # ---- Prediction horizons (trading days) ----
    HORIZONS: List[int] = field(default_factory=lambda: [1, 5, 20])
    DEFAULT_HORIZON: int = 5

    # ---- BUY/HOLD/SELL labeling ----
    # Volatility-adjusted thresholds: BUY if fwd_return > k_buy * recent_vol,
    # SELL if fwd_return < -k_sell * recent_vol, else HOLD.
    VOL_LOOKBACK: int = 20
    K_BUY: float = 0.75
    K_SELL: float = 0.75
    # Floors so thresholds never collapse to ~0 in very-low-vol regimes.
    MIN_THRESHOLD: float = 0.0025

    # ---- NO TRADE gating ----
    # If the model's top-class probability doesn't exceed second-place by at
    # least this margin, the signal engine emits NO TRADE instead of forcing
    # a call. See Section 5 of the spec.
    MIN_EDGE: float = 0.12
    MIN_CONFIDENCE: float = 0.45  # top-class probability floor

    # ---- Train/test / walk-forward ----
    TEST_SIZE: float = 0.20
    RANDOM_STATE: int = 42
    WALK_FORWARD_FOLDS: int = 5
    WALK_FORWARD_MIN_TRAIN: int = 500   # min rows before first fold

    # ---- Regime detection ----
    REGIME_LOOKBACK: int = 20
    N_REGIMES: int = 4  # BULL/LOW, BULL/HIGH, BEAR/LOW, BEAR/HIGH

    # ---- LSTM/GRU (only used if TensorFlow is available) ----
    SEQUENCE_LEN: int = 60
    NN_EPOCHS: int = 40
    NN_BATCH: int = 32
    NN_UNITS: int = 64

    # ---- Model persistence ----
    MODEL_DIR: str = "saved_models"
    MODEL_VERSION: str = "Zorix-2.0.0"

    # ---- Feature engineering ----
    RETURN_WINDOWS: List[int] = field(default_factory=lambda: [1, 2, 3, 5, 10, 20])
    VOLATILITY_WINDOWS: List[int] = field(default_factory=lambda: [10, 20, 60])
    ATR_WINDOW: int = 14

    DROP_COLS: List[str] = field(default_factory=lambda: [
        "Open", "High", "Low", "Close", "Adj Close", "Volume",
        "Target_1D", "Target_5D", "Target_20D",
        "FwdReturn_1D", "FwdReturn_5D", "FwdReturn_20D",
    ])

    # ---- Risk management (Section 20) ----
    PORTFOLIO_CAPITAL: float = 100_000.0
    RISK_PER_TRADE: float = 0.01        # fraction of capital risked per trade
    MAX_PORTFOLIO_RISK: float = 0.06    # cap on simultaneous open risk
    STOP_LOSS_ATR_MULT: float = 1.5     # stop distance = ATR * this multiplier
    TARGET_RR_RATIO: float = 1.75       # target distance = stop distance * this
    MAX_POSITION_PCT: float = 0.25      # no single position > 25% of capital

    # ---- Backtesting (Section 24/25) ----
    TRANSACTION_COST_BPS: float = 5.0   # per trade side, basis points
    SLIPPAGE_BPS: float = 5.0           # per trade side, basis points
    MAX_HOLDING_DAYS: int = 20          # forced exit if neither stop nor target hit
    RISK_FREE_RATE: float = 0.0

    # ---- Prediction logging ----
    PREDICTION_LOG_PATH: str = "logs/predictions.jsonl"

    # ---- Feature ablation / stability ----
    ABLATION_N_REPEATS: int = 10
    STABILITY_N_FOLDS: int = 5


DEFAULT_CONFIG = Config()

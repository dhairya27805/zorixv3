"""
zorix/utils/prediction_log.py

Prediction logging (Priority 1 of this phase). Every time the app produces
a ZorixSignal, it's appended as one JSON line to a local log file. This is
what lets you later check "when Zorix said 72% BUY, what actually
happened?" — the calibration curve in models/calibration.py answers this
on the test set; this log lets you answer it for real deployed predictions
over time, which is a different (and arguably more important) question.

Deliberately a flat JSONL file, not a database — zero setup, human
readable, append-safe, easy to load into pandas.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import pandas as pd


def log_prediction(signal_dict: dict, log_path: str) -> None:
    """Appends one prediction record. Creates parent directories if needed.
    Never overwrites — always appends, so historical predictions are
    never lost or altered (important for honest calibration tracking).
    """
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(signal_dict, default=str) + "\n")


def load_predictions(log_path: str) -> pd.DataFrame:
    """Returns an empty DataFrame (not an error) if the log doesn't exist
    yet — this is the normal state for a fresh install, not a failure."""
    path = Path(log_path)
    if not path.exists():
        return pd.DataFrame()

    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # skip a corrupted line rather than fail the whole load
    return pd.DataFrame(records)


def prediction_log_summary(log_path: str) -> Optional[dict]:
    """Quick stats for a UI panel: how many predictions logged, signal
    distribution, date range. Returns None if the log is empty."""
    df = load_predictions(log_path)
    if df.empty:
        return None
    summary = {
        "n_predictions": len(df),
        "date_range": (str(df["timestamp"].min()), str(df["timestamp"].max())) if "timestamp" in df else None,
    }
    if "signal" in df:
        summary["signal_counts"] = df["signal"].value_counts().to_dict()
    return summary

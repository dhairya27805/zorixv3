"""
zorix/models/splits.py

Purged chronological splitting. A label at row i spans rows i..i+h (it is
Close[i+h]/Close[i]-1), so a train row within h rows of the test start has a
label window that overlaps the test period. We drop those h rows (purge).
"""

from __future__ import annotations

import re
from typing import Tuple


def infer_horizon(target_col: str, default: int = 0) -> int:
    """'Target_5D' / 'FwdReturn_20D' -> 5 / 20; otherwise `default`."""
    m = re.search(r"_(\d+)D$", str(target_col))
    return int(m.group(1)) if m else default


def purged_split_points(n_rows: int, test_size: float, horizon: int) -> Tuple[int, int]:
    """Returns (train_end, test_start): train = rows[:train_end],
    test = rows[test_start:], with train_end = test_start - horizon so no
    training label window reaches into the test period."""
    test_start = int(n_rows * (1 - test_size))
    train_end = max(test_start - int(horizon), 0)
    return train_end, test_start

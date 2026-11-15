"""Nạp dữ liệu: CSV (time, close hoặc OHLC) và bộ sinh dữ liệu tổng hợp cho demo/test."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .cleaning import clean_prices, to_daily


def load_csv_close(path: str | Path, time_col: str = "time", close_col: str = "close") -> pd.Series:
    df = pd.read_csv(path)
    df.columns = [c.lower() for c in df.columns]
    idx = pd.to_datetime(df[time_col.lower()], utc=False)
    s = pd.Series(df[close_col.lower()].to_numpy(dtype=float), index=idx, name=Path(path).stem)
    return clean_prices(to_daily(s.sort_index()))


def synthetic_prices(symbols: dict[str, float], n_days: int = 1500, seed: int = 7,
                     annual_vol: float = 0.18, trend_strength: float = 0.04) -> dict[str, pd.Series]:
    """Giá tổng hợp có xu hướng chuyển pha (regime) để demo; KHÔNG phải bằng chứng alpha."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2019-01-01", periods=n_days)
    out: dict[str, pd.Series] = {}
    common = rng.normal(0, 1, n_days)
    for k, (sym, p0) in enumerate(symbols.items()):
        regime = np.sign(np.sin(np.arange(n_days) / rng.uniform(120, 300) + k))
        drift = regime * trend_strength / 256
        noise = 0.5 * common + np.sqrt(0.75) * rng.normal(0, 1, n_days)
        ret = drift + noise * annual_vol / np.sqrt(256)
        out[sym] = pd.Series(p0 * np.exp(np.cumsum(ret)), index=idx, name=sym)
    return out

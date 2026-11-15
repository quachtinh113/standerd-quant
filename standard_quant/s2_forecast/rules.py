"""Quy tắc forecast (Carver, Systematic Trading ch.7 và Leveraged Trading).

Mọi forecast: tỷ lệ với (tín hiệu / vol giá), nhân scalar để |trung bình| ≈ 10, kẹp ±20.
Tại t chỉ dùng giá ≤ t (không shift âm, không center=True).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FORECAST_TARGET = 10.0
FORECAST_CAP = 20.0

# Scalar chuẩn của Carver cho EWMAC(Lfast, 4*Lfast).
_EWMAC_SCALARS = {2: 10.6, 4: 7.5, 8: 5.3, 16: 3.75, 32: 2.65, 64: 1.87}


def ewmac_scalar(fast: int) -> float:
    """Scalar chuẩn; với tốc độ lạ, xấp xỉ theo luật 10.6 * (2/fast)^0.5 ... gần đúng."""
    if fast in _EWMAC_SCALARS:
        return _EWMAC_SCALARS[fast]
    return float(10.6 * np.sqrt(2.0 / fast) * 0.95)


def ewmac(price: pd.Series, vol: pd.Series, fast: int, slow: int | None = None,
          cap: float = FORECAST_CAP) -> pd.Series:
    """EWMAC: (EMA nhanh − EMA chậm) / vol giá × scalar, kẹp ±cap."""
    slow = slow or 4 * fast
    raw = price.ewm(span=fast, min_periods=fast).mean() - price.ewm(span=slow, min_periods=slow).mean()
    f = raw / vol.replace(0.0, np.nan) * ewmac_scalar(fast)
    return f.clip(-cap, cap)


def breakout(price: pd.Series, lookback: int = 80, smooth: int | None = None,
             cap: float = FORECAST_CAP) -> pd.Series:
    """Breakout: vị trí giá trong kênh [min,max] của `lookback` ngày, ánh xạ về [-20, 20].

    Vị trí ∈ [-0.5, 0.5] × 40 → [-20, 20].
    """
    smooth = smooth or max(lookback // 4, 1)
    hi = price.rolling(lookback, min_periods=lookback // 2).max()
    lo = price.rolling(lookback, min_periods=lookback // 2).min()
    rng = (hi - lo).replace(0.0, np.nan)
    pos = (price - (hi + lo) / 2.0) / rng           # [-0.5, 0.5]
    f = 40.0 * pos
    return f.ewm(span=smooth, min_periods=1).mean().clip(-cap, cap)

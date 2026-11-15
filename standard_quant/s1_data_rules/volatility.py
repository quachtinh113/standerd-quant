"""Ước lượng biến động theo Carver: EWMA 35 ngày trộn 70/30 với trung bình 10 năm."""
from __future__ import annotations

import pandas as pd


def daily_returns_vol(price: pd.Series, span: int = 35, min_periods: int = 10) -> pd.Series:
    """Độ lệch chuẩn EWMA của thay đổi GIÁ (đơn vị giá, không phải %)."""
    diff = price.diff()
    return diff.ewm(span=span, min_periods=min_periods).std()


def blended_daily_vol(
    price: pd.Series,
    span: int = 35,
    slow_window: int = 2520,   # ~10 năm D1
    slow_weight: float = 0.30,
    min_periods: int = 10,
) -> pd.Series:
    """Vol = 70% EWMA ngắn + 30% trung bình dài hạn của chính EWMA đó.

    Giảm nguy cơ vol thấp bất thường → vị thế quá lớn. Chỉ dùng dữ liệu ≤ t.
    """
    fast = daily_returns_vol(price, span, min_periods)
    slow = fast.rolling(slow_window, min_periods=min_periods).mean()
    return (1 - slow_weight) * fast + slow_weight * slow.fillna(fast)

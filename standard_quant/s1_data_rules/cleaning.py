"""Quy tắc dữ liệu: chỉ cho qua giá hợp lệ, chỉ dùng thông tin ≤ t-1 (zero look-ahead)."""
from __future__ import annotations

import pandas as pd

REQUIRED = ("open", "high", "low", "close")


def validate_bars(df: pd.DataFrame) -> list[str]:
    """Trả danh sách lỗi (rỗng = sạch). Không sửa dữ liệu."""
    errs: list[str] = []
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        return [f"thiếu cột: {missing}"]
    if not isinstance(df.index, pd.DatetimeIndex):
        errs.append("index phải là DatetimeIndex")
        return errs
    if not df.index.is_monotonic_increasing:
        errs.append("index không tăng dần")
    if df.index.has_duplicates:
        errs.append("index trùng lặp")
    if (df[list(REQUIRED)] <= 0).any().any():
        errs.append("giá <= 0")
    if (df["high"] < df[["open", "close", "low"]].max(axis=1)).any():
        errs.append("high < max(open, close, low)")
    if (df["low"] > df[["open", "close", "high"]].min(axis=1)).any():
        errs.append("low > min(open, close, high)")
    return errs


def clean_prices(close: pd.Series, max_abs_daily_return: float = 0.25, ffill_limit: int = 3) -> pd.Series:
    """Làm sạch chuỗi giá đóng cửa.

    - Sắp xếp, bỏ trùng (giữ bản ghi cuối), bỏ giá <= 0.
    - Loại tick lỗi: |return| > ngưỡng → NaN (không nội suy ngược tương lai).
    - Chỉ forward-fill (tối đa `ffill_limit` bước); KHÔNG bfill/interpolate (look-ahead).
    """
    s = close.sort_index()
    s = s[~s.index.duplicated(keep="last")]
    s = s.where(s > 0)
    ret = s.pct_change(fill_method=None).abs()
    s = s.mask(ret > max_abs_daily_return)
    return s.ffill(limit=ffill_limit)


def to_daily(close: pd.Series) -> pd.Series:
    """Giá đóng cửa D1 từ chuỗi bất kỳ (lấy giá cuối ngày)."""
    return close.resample("1D").last().dropna()

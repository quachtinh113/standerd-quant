"""Cầu nối: điểm số alpha thô → forecast chuẩn Carver (|trung bình| = 10, kẹp ±20)."""
from __future__ import annotations

import pandas as pd

from ..rules import FORECAST_CAP, FORECAST_TARGET
from .library import REGISTRY


def alpha_to_forecast(alpha: pd.DataFrame, min_periods: int = 60, cap: float = FORECAST_CAP) -> pd.DataFrame:
    """Chia cho trung bình |alpha| expanding (chỉ dữ liệu ≤ t) rồi nhân 10.

    Dùng một scalar chung cho cả panel để giữ nguyên độ mạnh tương đối giữa các instrument.
    """
    mag = alpha.abs().stack().groupby(level=0).mean().reindex(alpha.index)    # độ lớn trung bình theo ngày
    scale = FORECAST_TARGET / mag.expanding(min_periods=min_periods).mean()
    return alpha.mul(scale, axis=0).clip(-cap, cap)


def build_alpha_forecasts(names: list[str], close: pd.DataFrame, vol: pd.DataFrame) -> dict[str, pd.DataFrame]:
    data = {"close": close, "returns": close.pct_change(fill_method=None), "vol": vol}
    unknown = [n for n in names if n not in REGISTRY]
    if unknown:
        raise KeyError(f"alpha không có trong registry: {unknown}")
    return {n: alpha_to_forecast(REGISTRY[n](data)) for n in names}

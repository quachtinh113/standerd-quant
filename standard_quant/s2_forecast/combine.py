"""Gộp nhiều forecast của MỘT instrument: trọng số → nhân FDM → kẹp lại ±20."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .rules import FORECAST_CAP


def forecast_diversification_multiplier(corr: pd.DataFrame, weights: pd.Series | None = None,
                                        cap: float = 2.5) -> float:
    """FDM = 1 / sqrt(wᵀ H w), kẹp [1, cap]. corr âm → cắt về 0 (cách làm thận trọng của Carver)."""
    n = len(corr)
    w = np.full(n, 1.0 / n) if weights is None else weights.reindex(corr.index).fillna(0).to_numpy()
    h = corr.clip(lower=0.0).fillna(0.0).to_numpy()
    np.fill_diagonal(h, 1.0)
    var = float(w @ h @ w)
    if var <= 0:
        return 1.0
    return float(min(max(1.0 / np.sqrt(var), 1.0), cap))


def combine_forecasts(forecasts: dict[str, pd.Series], weights: dict[str, float] | None = None,
                      fdm: float | None = None, min_history: int = 60) -> pd.Series:
    """Forecast tổng hợp.

    fdm=None → ước lượng lại hàng quý từ tương quan expanding của dữ liệu ≤ t-1 (không rò rỉ).
    """
    df = pd.DataFrame(forecasts)
    names = list(df.columns)
    w = pd.Series(weights or {k: 1.0 for k in names}, dtype=float).reindex(names).fillna(0.0)
    w = w / w.sum()
    # trung bình có trọng số, bỏ qua NaN theo từng thời điểm
    mask = df.notna()
    wsum = mask.mul(w, axis=1).sum(axis=1).replace(0.0, np.nan)
    combined = df.fillna(0.0).mul(w, axis=1).sum(axis=1) / wsum

    if fdm is not None:
        fdm_s = pd.Series(fdm, index=df.index)
    else:
        fdm_s = pd.Series(1.0, index=df.index)
        for t in df.index[min_history::63]:               # checkpoint hàng quý
            hist = df.loc[:t].iloc[:-1]                      # chỉ dữ liệu ≤ t-1
            fdm_s.iloc[df.index.get_loc(t):] = forecast_diversification_multiplier(hist.corr(), w)
    return (combined * fdm_s).clip(-FORECAST_CAP, FORECAST_CAP)

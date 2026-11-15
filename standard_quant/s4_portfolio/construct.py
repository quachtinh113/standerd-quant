"""Xây danh mục: trọng số instrument, IDM, vị thế mục tiêu."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..contracts import Instrument
from ..s3_vol_targeting.sizing import round_to_lots, vol_scalar
from ..s2_forecast.rules import FORECAST_TARGET


def instrument_weights(returns: pd.DataFrame | None, symbols: list[str],
                       method: str = "equal") -> dict[str, float]:
    """Trọng số tổng = 1.

    - equal: 1/N (mặc định, bền nhất khi ít dữ liệu).
    - inverse_vol: ∝ 1/σ (dùng returns đến t-1).
    """
    if method == "equal" or returns is None:
        return {s: 1.0 / len(symbols) for s in symbols}
    if method == "inverse_vol":
        inv = 1.0 / returns[symbols].std().replace(0.0, np.nan)
        inv = inv.fillna(inv.mean())
        return (inv / inv.sum()).to_dict()
    raise ValueError(f"method không hỗ trợ: {method}")


def instrument_diversification_multiplier(returns: pd.DataFrame | None, weights: dict[str, float],
                                          cap: float = 2.5, min_obs: int = 60) -> float:
    """IDM = 1/√(wᵀ C w), tương quan âm cắt về 0, kẹp [1, cap]. Thiếu dữ liệu → 1.0."""
    syms = list(weights)
    if returns is None or len(syms) < 2 or len(returns) < min_obs:
        return 1.0
    corr = returns[syms].corr().clip(lower=0.0).fillna(0.0).to_numpy()
    np.fill_diagonal(corr, 1.0)
    w = np.array([weights[s] for s in syms])
    var = float(w @ corr @ w)
    return float(min(max(1.0 / np.sqrt(var), 1.0), cap)) if var > 0 else 1.0


def target_positions(
    forecasts: dict[str, float],
    vols: dict[str, float],
    instruments: dict[str, Instrument],
    capital: float,
    vol_target_annual: float,
    weights: dict[str, float],
    idm: float,
    risk_multiplier: float = 1.0,
) -> dict[str, float]:
    """Vị thế lot có dấu cho từng instrument tại một thời điểm.

    risk_multiplier đến từ Risk Overlay (Drawdown Ladder) — nhân vào cuối cùng.
    Chiều giảm vị thế luôn được phép; Execution quyết định chiều tăng.
    """
    out: dict[str, float] = {}
    for sym, inst in instruments.items():
        f, v = forecasts.get(sym), vols.get(sym)
        if f is None or v is None or not np.isfinite(f) or not np.isfinite(v) or v <= 0:
            out[sym] = 0.0
            continue
        sc = vol_scalar(capital, vol_target_annual, v, inst, idm=idm, weight=weights.get(sym, 0.0))
        out[sym] = round_to_lots(float(sc) * f / FORECAST_TARGET * risk_multiplier, inst)
    return out

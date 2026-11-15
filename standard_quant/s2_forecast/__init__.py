"""Khối 2 — Forecast Generation: quy tắc giao dịch → forecast chuẩn hoá (|trung bình| ≈ 10, kẹp ±20)."""
from .combine import combine_forecasts, forecast_diversification_multiplier
from . import alpha, meta
from .rules import FORECAST_CAP, FORECAST_TARGET, breakout, ewmac, ewmac_scalar

__all__ = [
    "alpha", "meta",
    "FORECAST_CAP", "FORECAST_TARGET", "ewmac", "ewmac_scalar", "breakout",
    "combine_forecasts", "forecast_diversification_multiplier",
]

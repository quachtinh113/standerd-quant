"""Định cỡ vị thế theo mục tiêu biến động (Carver).

  vol tiền / lot / ngày = vol giá × point_value
  vol_scalar            = (vốn × vol_mục_tiêu_năm / √256) / vol tiền / lot
  vị thế subsystem      = vol_scalar × forecast / 10

Không dùng Kelly đầy đủ; vol_target mặc định 20%/năm là mức Carver khuyến nghị cho
vốn nhỏ, thấp hơn Half-Kelly điển hình.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..contracts import Instrument
from ..s2_forecast.rules import FORECAST_TARGET

TRADING_DAYS = 256


def instrument_vol_cash(vol_price: pd.Series | float, inst: Instrument) -> pd.Series | float:
    """Biến động tiền tệ hằng ngày của 1 lot."""
    return vol_price * inst.point_value


def vol_scalar(capital: float, vol_target_annual: float, vol_price: pd.Series | float,
               inst: Instrument, idm: float = 1.0, weight: float = 1.0) -> pd.Series | float:
    """Số lot tương ứng forecast = 10 (trước khi nhân forecast/10)."""
    daily_cash_target = capital * vol_target_annual / np.sqrt(TRADING_DAYS)
    cash_vol = instrument_vol_cash(vol_price, inst)
    if isinstance(cash_vol, pd.Series):
        cash_vol = cash_vol.replace(0.0, np.nan)
    elif cash_vol == 0:
        return np.nan
    return daily_cash_target * idm * weight / cash_vol


def subsystem_position(forecast: pd.Series, scalar: pd.Series) -> pd.Series:
    """Vị thế lot (có dấu), chưa làm tròn."""
    return scalar * forecast / FORECAST_TARGET


def buffer_position(optimal: float, current: float, buffer_fraction: float = 0.10) -> float:
    """Vùng đệm ±10% quanh vị thế tối ưu: chỉ giao dịch khi vị thế hiện tại ra ngoài vùng,
    và khi đó chỉ đưa về BIÊN gần nhất (không về tâm) → giảm turnover/chi phí.
    """
    band = abs(optimal) * buffer_fraction
    lo, hi = optimal - band, optimal + band
    if lo <= current <= hi:
        return current
    return lo if current < lo else hi


def round_to_lots(position: float, inst: Instrument) -> float:
    """Làm tròn về bước lot, kẹp max_lot. Dưới min_lot/2 → 0 (không mở vị thế vô nghĩa)."""
    mag = abs(position)
    if mag < inst.min_lot / 2:
        return 0.0
    steps = round(mag / inst.lot_step)
    lots = min(max(steps * inst.lot_step, inst.min_lot), inst.max_lot)
    return float(np.sign(position) * round(lots, 8))

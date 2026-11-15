"""Khối 3 — Volatility Targeting: forecast + vol → vị thế (lot) cho từng instrument."""
from .sizing import buffer_position, instrument_vol_cash, subsystem_position, vol_scalar

__all__ = ["instrument_vol_cash", "vol_scalar", "subsystem_position", "buffer_position"]

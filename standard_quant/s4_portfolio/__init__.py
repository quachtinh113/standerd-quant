"""Khối 4 — Portfolio Construction: trọng số instrument + IDM → vị thế mục tiêu toàn danh mục."""
from .construct import (
    instrument_diversification_multiplier,
    instrument_weights,
    target_positions,
)

__all__ = ["instrument_weights", "instrument_diversification_multiplier", "target_positions"]

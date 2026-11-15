"""Hợp đồng dữ liệu giữa 5 khối (Carver: Data → Forecast → Vol Targeting → Portfolio → Execution).

Mỗi khối chỉ nhận/trả các kiểu dưới đây, không import ngược khối đứng sau.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import pandas as pd


@dataclass(frozen=True)
class Instrument:
    """Đặc tả hợp đồng. point_value = tiền (tiền tệ tài khoản) cho 1 đơn vị giá × 1 lot."""
    symbol: str
    point_value: float = 1.0          # vd XAUUSD: 100 (1 lot = 100 oz)
    min_lot: float = 0.01
    lot_step: float = 0.01
    max_lot: float = 50.0
    cost_per_lot_rt: float = 7.0      # phí + trượt giá khứ hồi / lot (tiền)
    spread_price: float = 0.0         # spread tính theo đơn vị giá


@dataclass(frozen=True)
class OrderIntent:
    """Đầu ra của Portfolio→Execution: ý định, chưa phải lệnh."""
    symbol: str
    side: Literal["BUY", "SELL"]
    volume: float
    sl: float
    tp: float | None
    reason: str = ""


@dataclass
class ExecutionResult:
    symbol: str
    success: bool
    volume: float = 0.0
    price: float = 0.0
    message: str = ""


@dataclass
class PipelineSnapshot:
    """Ảnh chụp toàn bộ luồng cho 1 thời điểm — dùng cho telemetry/audit."""
    asof: pd.Timestamp
    forecasts: dict[str, float] = field(default_factory=dict)
    target_positions: dict[str, float] = field(default_factory=dict)   # lot, có dấu
    current_positions: dict[str, float] = field(default_factory=dict)
    intents: list[OrderIntent] = field(default_factory=list)
    risk_multiplier: float = 1.0
    halted: bool = False

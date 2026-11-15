"""Phía UPSTREAM của ranh giới L5: đổi OrderIntent (planner) thành tín hiệu trong hộp thư.

Chiều phụ thuộc: s5_execution → l5_execution.store (một chiều). l5_execution không biết gì về nơi này.
"""
from __future__ import annotations

import hashlib
import sqlite3

from ..contracts import OrderIntent
from ..l5_execution.store import Signal, publish


def publish_intents(con: sqlite3.Connection, intents: list[OrderIntent], asof: str, prices: dict[str, float],
                    magic: int, ttl_s: float = 30.0, symbol_map: dict[str, str] | None = None) -> int:
    """Ghi các intent vào hộp thư. signal_id tất định theo (asof, symbol, side, volume, lý do) nên chạy lại
    cùng một chu kỳ không tạo lệnh trùng. `symbol_map`: tên nội bộ → tên broker (vd XAUUSD→XAUUSDm); tín hiệu
    trong hộp thư luôn mang tên BROKER. Trả số tín hiệu MỚI được ghi."""
    symbol_map = symbol_map or {}
    n = 0
    for it in intents:
        sid = hashlib.sha1(f"{asof}|{it.symbol}|{it.side}|{it.volume:.8f}|{it.reason}|{magic}".encode()).hexdigest()[:20]
        reduce = it.reason.startswith("REDUCE")
        sig = Signal(sid, symbol_map.get(it.symbol, it.symbol), it.side, it.volume, sl=it.sl or 0.0, tp=it.tp or 0.0,
                     ref_price=prices.get(it.symbol, 0.0), reduce_only=reduce, magic=magic, reason=it.reason)
        n += bool(publish(con, sig, ttl_s))
    return n

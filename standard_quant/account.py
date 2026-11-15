"""Trạng thái tài khoản mà bot đọc từ broker. Thuần dữ liệu, không phụ thuộc pandas."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Pos:
    volume: float          # có dấu
    avg_price: float


@dataclass
class AccountState:
    equity: float
    margin: float = 0.0
    margin_level: float = 0.0                      # % ; 0 khi không có vị thế (quy ước MT5)
    held: dict[str, Pos] = field(default_factory=dict)

    def positions(self):
        """Tương thích với pipeline.step(): trả dict[str, s5_execution.Position]."""
        from .s5_execution.planner import Position
        return {s: Position(p.volume, p.avg_price) for s, p in self.held.items() if p.volume != 0}

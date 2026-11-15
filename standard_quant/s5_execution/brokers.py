"""Cổng broker. Chỉ PaperBroker được cung cấp; adapter MT5 nên cài ở đây theo cùng Protocol."""
from __future__ import annotations

from typing import Protocol

from ..contracts import ExecutionResult, Instrument, OrderIntent
from .planner import Position


class Broker(Protocol):
    def positions(self) -> dict[str, Position]: ...
    def equity(self) -> float: ...
    def send(self, intent: OrderIntent, price: float) -> ExecutionResult: ...


class PaperBroker:
    """Mô phỏng khớp tại giá đóng cửa, trừ phí/spread/trượt giá. Không kiểm tra SL (backtest
    kiểm tra SL ở tầng pipeline)."""

    def __init__(self, instruments: dict[str, Instrument], capital: float):
        self.instruments = instruments
        self.cash = capital
        self._pos: dict[str, Position] = {}
        self.cost_paid = 0.0

    def positions(self) -> dict[str, Position]:
        return dict(self._pos)

    def mark(self, prices: dict[str, float]) -> float:
        return sum(
            p.volume * (prices[s] - p.avg_price) * self.instruments[s].point_value
            for s, p in self._pos.items() if s in prices
        )

    def equity(self, prices: dict[str, float] | None = None) -> float:
        return self.cash + (self.mark(prices) if prices else 0.0)

    def send(self, intent: OrderIntent, price: float) -> ExecutionResult:
        inst = self.instruments[intent.symbol]
        signed = intent.volume if intent.side == "BUY" else -intent.volume
        fill = price + (inst.spread_price / 2.0) * (1 if signed > 0 else -1)
        cost = inst.cost_per_lot_rt / 2.0 * intent.volume        # nửa vòng khứ hồi mỗi chân
        cur = self._pos.get(intent.symbol, Position(0.0, 0.0))
        new_vol = cur.volume + signed
        # realize PnL phần đóng
        closing = 0.0
        if cur.volume != 0 and (signed * cur.volume < 0):
            closed = min(abs(signed), abs(cur.volume))
            closing = closed * (fill - cur.avg_price) * (1 if cur.volume > 0 else -1) * inst.point_value
        self.cash += closing - cost
        self.cost_paid += cost
        if abs(new_vol) < 1e-12:
            self._pos.pop(intent.symbol, None)
        else:
            if cur.volume == 0 or signed * cur.volume < 0 and abs(signed) > abs(cur.volume):
                avg = fill                                           # mở mới / đảo chiều
            elif signed * cur.volume > 0:
                avg = (cur.avg_price * abs(cur.volume) + fill * abs(signed)) / abs(new_vol)
            else:
                avg = cur.avg_price                                  # giảm một phần
            self._pos[intent.symbol] = Position(new_vol, avg)
        return ExecutionResult(intent.symbol, True, intent.volume, fill, "paper-fill")

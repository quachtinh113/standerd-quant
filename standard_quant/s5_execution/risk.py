"""Risk overlay: Drawdown Ladder từ High-Water Mark + giới hạn lỗ ngày.

  -3.0% DD : TIER_1_WARNING  (×1.0, cảnh báo)
  -5.0% DD : TIER_2_DERISK   (×0.5)
  -7.5% DD hoặc lỗ ngày ≥ cap: TIER_3_HALTED (×0.0, chỉ được đóng/giảm vị thế)
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DrawdownLadder:
    tier1_dd: float = 0.03
    tier2_dd: float = 0.05
    tier3_dd: float = 0.075
    daily_loss_cap: float | None = None      # tiền; None = tắt
    hwm: float = 0.0
    day_start_equity: float = 0.0
    halted_latch: bool = False               # halt do DD, giữ cho tới reset thủ công
    last_equity: float = 0.0

    def update(self, equity: float, new_day: bool = False) -> None:
        """new_day=True: mốc lỗ ngày = equity cuối ngày trước (daily bar: lỗ so với phiên trước)."""
        self.hwm = max(self.hwm, equity)
        if new_day or self.day_start_equity == 0.0:
            self.day_start_equity = self.last_equity or equity
        self.last_equity = equity

    def drawdown(self, equity: float) -> float:
        return 0.0 if self.hwm <= 0 else max(0.0, (self.hwm - equity) / self.hwm)

    def tier(self, equity: float) -> str:
        dd = self.drawdown(equity)
        day_loss = self.day_start_equity - equity
        if dd >= self.tier3_dd:
            self.halted_latch = True                     # DD: giữ cho tới khi con người reset
        if self.halted_latch or (self.daily_loss_cap is not None and day_loss >= self.daily_loss_cap):
            return "TIER_3_HALTED"                       # lỗ ngày: chỉ dừng trong ngày (new_day xoá)
        if dd >= self.tier2_dd:
            return "TIER_2_DERISK"
        if dd >= self.tier1_dd:
            return "TIER_1_WARNING"
        return "NORMAL"

    def multiplier(self, equity: float) -> float:
        return {"NORMAL": 1.0, "TIER_1_WARNING": 1.0, "TIER_2_DERISK": 0.5, "TIER_3_HALTED": 0.0}[self.tier(equity)]

    def is_halted(self, equity: float) -> bool:
        return self.tier(equity) == "TIER_3_HALTED"

    def reset_halt(self, equity: float) -> None:
        """Chỉ gọi khi con người duyệt (backtest: sau cooldown). Đặt lại HWM = equity hiện tại."""
        self.halted_latch = False
        self.hwm = equity

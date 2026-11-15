"""L4 — Bàn rủi ro trung tâm: cổng duyệt trước giao dịch cho MỌI intent trước khi vào hộp thư L5.

Nguyên tắc: kết quả luôn là (approved, rejected+lý do) — không bao giờ im lặng. Lệnh GIẢM/ĐÓNG luôn được
duyệt (thoát vị thế không bị chặn bởi halt/EOD). Lệnh MỞ/TĂNG phải qua toàn bộ cổng.
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from ..account import AccountState
from ..contracts import Instrument, OrderIntent
from ..l5_execution import store
from ..s5_execution.risk import DrawdownLadder
from ..ledger import HashChainLedger


@dataclass
class RiskLimits:
    max_open_positions: int = 10
    min_margin_level_pct: float = 200.0
    max_leverage: float = 15.0                    # Σ|lot|×point_value×giá / equity, sau khi khớp
    max_lot_per_symbol: float | None = None       # None = dùng Instrument.max_lot
    block_weekend_flat: bool = True               # Thứ 6 ≥ 21:00 UTC, Thứ 7, CN: không mở mới
    block_rollover: bool = True                   # 23:45–00:05 UTC: không mở mới
    halt_file: Path | None = None                 # file tồn tại = kill switch thủ công


@dataclass
class RiskDecision:
    approved: list[OrderIntent] = field(default_factory=list)
    rejected: list[tuple[OrderIntent, str]] = field(default_factory=list)
    tier: str = "NORMAL"
    multiplier: float = 1.0
    halted: bool = False


def _time_block(now: datetime, lim: RiskLimits) -> str | None:
    hm = now.strftime("%H:%M")
    if lim.block_weekend_flat and ((now.weekday() == 4 and hm >= "21:00") or now.weekday() in (5, 6)):
        return "weekend_flat"
    if lim.block_rollover and (hm >= "23:45" or hm < "00:05"):
        return "rollover"
    return None


class RiskDesk:
    def __init__(self, limits: RiskLimits, ladder: DrawdownLadder, ledger: HashChainLedger | None = None,
                 con: sqlite3.Connection | None = None):
        self.limits, self.ladder, self.ledger, self.con = limits, ladder, ledger, con
        self._last_tier = "NORMAL"

    def _audit(self, event: str, payload: dict) -> None:
        if self.ledger:
            self.ledger.append(event, payload)

    def update_equity(self, equity: float, new_day: bool = True) -> str:
        """Cập nhật ladder, ghi sổ khi đổi bậc, và đẩy cờ `halt` xuống L5 (phòng thủ nhiều lớp)."""
        self.ladder.update(equity, new_day=new_day)
        tier = self.ladder.tier(equity)
        if tier != self._last_tier:
            self._audit("TIER_CHANGE", {"from": self._last_tier, "to": tier, "equity": equity,
                                        "drawdown_pct": round(self.ladder.drawdown(equity) * 100, 3)})
            self._last_tier = tier
        if self.con is not None:
            store.set_flag(self.con, "halt", tier == "TIER_3_HALTED" or self._manual_halt())
        return tier

    def _manual_halt(self) -> bool:
        return bool(self.limits.halt_file and Path(self.limits.halt_file).exists())

    def review(self, intents: list[OrderIntent], account: AccountState, now_utc: datetime,
               prices: dict[str, float], instruments: dict[str, Instrument]) -> RiskDecision:
        tier = self.update_equity(account.equity, new_day=False)
        d = RiskDecision(tier=tier, multiplier=self.ladder.multiplier(account.equity),
                         halted=tier == "TIER_3_HALTED" or self._manual_halt())
        L = self.limits
        block = _time_block(now_utc, L)
        held = {s: p.volume for s, p in account.held.items() if p.volume != 0}
        # tổng notional sau khi khớp (mô phỏng cộng dồn các intent đã duyệt)
        sim = dict(held)

        def notional(vol_map: dict[str, float]) -> float:
            return sum(abs(v) * instruments[s].point_value * prices.get(s, 0.0) for s, v in vol_map.items() if s in instruments)

        for it in intents:
            if it.reason.startswith("REDUCE"):
                d.approved.append(it)
                sim[it.symbol] = sim.get(it.symbol, 0.0) + (it.volume if it.side == "BUY" else -it.volume)
                continue
            reason = None
            signed = it.volume if it.side == "BUY" else -it.volume
            inst = instruments.get(it.symbol)
            if d.halted:
                reason = "halted"
            elif block:
                reason = block
            elif inst is None:
                reason = "unknown_instrument"
            elif it.sl <= 0:
                reason = "missing_stop_loss"
            elif abs(sim.get(it.symbol, 0.0) + signed) > (L.max_lot_per_symbol or inst.max_lot):
                reason = "exceeds_symbol_lot_cap"
            elif sim.get(it.symbol, 0.0) == 0 and sum(1 for v in sim.values() if v) >= L.max_open_positions:
                reason = "max_open_positions"
            elif account.margin > 0 and account.margin_level < L.min_margin_level_pct:
                reason = "margin_level_low"
            else:
                after = dict(sim); after[it.symbol] = after.get(it.symbol, 0.0) + signed
                if account.equity > 0 and notional(after) / account.equity > L.max_leverage:
                    reason = "leverage_cap"
            if reason:
                d.rejected.append((it, reason))
            else:
                d.approved.append(it)
                sim[it.symbol] = sim.get(it.symbol, 0.0) + signed
        if d.rejected:
            self._audit("INTENTS_REJECTED", {"tier": tier, "items": [
                {"symbol": i.symbol, "side": i.side, "volume": i.volume, "reason": r} for i, r in d.rejected]})
        return d


# ---- Lưu/khôi phục trạng thái ladder (HWM, latch) qua khởi động lại ----
def save_ladder(con: sqlite3.Connection, ladder: DrawdownLadder) -> None:
    store.set_value(con, "ladder_state", json.dumps({
        "hwm": ladder.hwm, "day_start_equity": ladder.day_start_equity,
        "halted_latch": ladder.halted_latch, "last_equity": ladder.last_equity}))


def load_ladder(con: sqlite3.Connection, ladder: DrawdownLadder) -> DrawdownLadder:
    raw = store.get_value(con, "ladder_state")
    if raw:
        st = json.loads(raw)
        ladder.hwm, ladder.day_start_equity = st["hwm"], st["day_start_equity"]
        ladder.halted_latch, ladder.last_equity = st["halted_latch"], st["last_equity"]
    return ladder

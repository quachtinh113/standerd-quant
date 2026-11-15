"""Planner: vị thế mục tiêu − vị thế hiện tại → OrderIntent (có SL vật lý, không DCA)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..contracts import Instrument, OrderIntent
from ..s3_vol_targeting.sizing import buffer_position, round_to_lots


@dataclass
class Position:
    volume: float          # có dấu: + long, − short
    avg_price: float


def plan_orders(
    targets: dict[str, float],
    current: dict[str, Position],
    instruments: dict[str, Instrument],
    prices: dict[str, float],
    vols: dict[str, float],
    halted: bool = False,
    buffer_fraction: float = 0.10,
    sl_vol_mult: float = 6.0,
    tp_vol_mult: float | None = None,
    min_edge_cost_ratio: float = 0.0,
) -> list[OrderIntent]:
    """Sinh intent.

    Quy tắc bất biến:
      ① Mọi lệnh MỞ/TĂNG có SL vật lý = giá ∓ sl_vol_mult × vol (disaster stop, rộng so với
        stop thông thường; Carver không dùng stop giao dịch, đây chỉ là bảo hiểm phía broker).
      ② halted → chỉ được giảm/đóng.
      ③ Không DCA: không tăng vị thế đang lỗ (giá đi ngược avg_price).
      ④ Buffer 10%: bỏ giao dịch nhỏ.
      ⑤ Bỏ lệnh khi chi phí > min_edge_cost_ratio × |delta| × vol tiền (tuỳ chọn, mặc định tắt).
    """
    intents: list[OrderIntent] = []
    for sym, inst in instruments.items():
        px, vol = prices.get(sym), vols.get(sym)
        if px is None or vol is None or not np.isfinite(px) or not np.isfinite(vol) or vol <= 0:
            continue
        cur = current.get(sym, Position(0.0, 0.0))
        tgt = targets.get(sym, 0.0)
        if halted:                       # chỉ giảm: không tăng, không đảo chiều
            tgt = float(np.sign(cur.volume) * min(abs(tgt), abs(cur.volume))) if np.sign(tgt) == np.sign(cur.volume) else 0.0
        tgt = round_to_lots(buffer_position(tgt, cur.volume, buffer_fraction), inst) if tgt != 0 else 0.0
        delta = tgt - cur.volume
        if abs(delta) < inst.min_lot / 2:
            continue

        # đảo chiều = đóng hết trước rồi mở phần còn lại: tách thành 2 intent
        flips = cur.volume != 0 and np.sign(tgt) != np.sign(cur.volume)
        legs = [(-cur.volume, True), (tgt, False)] if flips else [(delta, tgt == 0 or abs(tgt) < abs(cur.volume))]
        for leg_vol, is_reduce in legs:
            if abs(leg_vol) < inst.min_lot / 2:
                continue
            side = "BUY" if leg_vol > 0 else "SELL"
            if is_reduce:
                intents.append(OrderIntent(sym, side, abs(leg_vol), sl=0.0, tp=None, reason="REDUCE/CLOSE"))
                continue
            adding = cur.volume != 0 and np.sign(leg_vol) == np.sign(cur.volume) and not flips
            if adding:
                underwater = (cur.volume > 0 and px < cur.avg_price) or (cur.volume < 0 and px > cur.avg_price)
                if underwater:                                   # ③ cấm DCA
                    continue
            if min_edge_cost_ratio > 0:
                cash_vol = vol * inst.point_value * abs(leg_vol)
                if inst.cost_per_lot_rt * abs(leg_vol) > min_edge_cost_ratio * cash_vol:
                    continue
            sl = px - sl_vol_mult * vol if side == "BUY" else px + sl_vol_mult * vol
            tp = None if tp_vol_mult is None else (px + tp_vol_mult * vol if side == "BUY" else px - tp_vol_mult * vol)
            if sl <= 0:
                sl = px * 0.01
            intents.append(OrderIntent(sym, side, abs(leg_vol), sl=sl, tp=tp, reason="REBALANCE"))
    return intents

"""Thư viện alpha mẫu + registry. Mỗi alpha: data(dict[str, Frame]) → Frame (điểm số thô).

Nguyên tắc Finding Alphas: ý tưởng đơn giản, ít tham số, có lý do kinh tế; tham số là
ứng viên cho robustness (xem evaluate.py), không phải để tối ưu hoá.
data keys: "close", "returns" (pct_change), "vol" (vol giá EWMA).
"""
from __future__ import annotations

from typing import Callable

from . import operators as op

AlphaFn = Callable[[dict], op.Frame]
REGISTRY: dict[str, AlphaFn] = {}


def register(name: str):
    def deco(fn: AlphaFn) -> AlphaFn:
        REGISTRY[name] = fn
        return fn
    return deco


@register("mom_vol_adj")
def mom_vol_adj(d):
    """Động lượng 60 phiên chia vol (xu hướng thưởng cho rủi ro đã đo), làm mượt giảm turnover."""
    x = op.safe_div(op.ts_delta(d["close"], 60), d["vol"] * 60 ** 0.5)
    return op.ts_decay_linear(x, 5)


@register("rev_short")
def rev_short(d):
    """Đảo chiều ngắn hạn: thanh khoản bị đẩy quá đà 5 phiên có xu hướng hồi lại."""
    return -op.ts_zscore(op.ts_sum(d["returns"], 5), 60)


@register("range_position")
def range_position(d):
    """Vị trí trong kênh 120 phiên (ts_rank) — tín hiệu trạng thái, ít nhạy tham số."""
    return op.ts_rank(d["close"], 120) - 0.5


@register("xs_mom_rank")
def xs_mom_rank(d):
    """Động lượng tương đối: rank ngang của lợi suất 90 phiên (market-neutral)."""
    return op.cs_rank(op.ts_sum(d["returns"], 90))


@register("vol_regime_trend")
def vol_regime_trend(d):
    """Xu hướng chỉ khi vol thấp hơn trung vị dài hạn (tránh chop trong thị trường biến động)."""
    calm = (d["vol"] < op.ts_mean(d["vol"], 250)).astype(float)
    return op.ts_zscore(d["close"], 60) * calm

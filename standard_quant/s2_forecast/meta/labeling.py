"""AFML ch.3 — Triple-Barrier + Meta-Labeling, và ch.4 — trọng số độc nhất (uniqueness).

Vai trò: L2 (forecast) đã quyết định HƯỚNG (side = sign(forecast)). Ở đây ta chỉ hỏi
"lần vào lệnh này có thắng không?" → nhãn nhị phân. ML KHÔNG dự đoán giá lên/xuống.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def triple_barrier(close: pd.Series, side: pd.Series, vol: pd.Series, *, pt_mult: float = 2.0,
                   sl_mult: float = 2.0, horizon: int = 10) -> pd.DataFrame:
    """Với mỗi t0 có side≠0 và vol hợp lệ, nhìn tới tối đa `horizon` phiên sau (chỉ để GÁN NHÃN).

    Rào ngang: ±mult × vol × √horizon (đơn vị giá). Rào dọc: sau `horizon` phiên.
    Trả: t1 (ngày kết thúc), ret (lợi suất % theo hướng side), label (1 nếu ret > 0), barrier ('pt'|'sl'|'vert').
    Sự kiện chưa đủ `horizon` phiên dữ liệu bị bỏ (chưa phân giải → không được dùng để huấn luyện).
    """
    c = close.to_numpy(float)
    s = side.reindex(close.index).fillna(0).to_numpy(float)
    v = vol.reindex(close.index).to_numpy(float)
    idx, rows = close.index, []
    scale = np.sqrt(horizon)
    for i in range(len(c) - horizon):
        if s[i] == 0 or not np.isfinite(v[i]) or v[i] <= 0:
            continue
        pnl = s[i] * (c[i + 1: i + horizon + 1] - c[i])
        w_pt, w_sl = pt_mult * v[i] * scale, sl_mult * v[i] * scale
        hit_pt, hit_sl = np.flatnonzero(pnl >= w_pt), np.flatnonzero(pnl <= -w_sl)
        j_pt = hit_pt[0] if len(hit_pt) else horizon
        j_sl = hit_sl[0] if len(hit_sl) else horizon
        if j_pt < j_sl:
            j, kind = j_pt, "pt"
        elif j_sl < horizon:                 # sl đến trước, hoặc cùng phiên (bi quan: coi là thua)
            j, kind = j_sl, "sl"
        else:
            j, kind = horizon - 1, "vert"
        ret = s[i] * (c[i + 1 + j] / c[i] - 1.0)
        rows.append((idx[i], idx[i + 1 + j], ret, int(ret > 0), kind))
    return pd.DataFrame(rows, columns=["t0", "t1", "ret", "label", "barrier"]).set_index("t0")


def average_uniqueness(t1: pd.Series, index: pd.DatetimeIndex) -> pd.Series:
    """Trung bình 1/(số nhãn đồng thời) trên đời sống của mỗi nhãn (AFML 4.2). Nhãn chồng lấn nhiều → trọng số thấp."""
    pos = {t: k for k, t in enumerate(index)}
    conc = np.zeros(len(index))
    spans = [(pos[t0], pos[t1_]) for t0, t1_ in t1.items()]
    for a, b in spans:
        conc[a:b + 1] += 1
    inv = np.where(conc > 0, 1.0 / np.maximum(conc, 1), 0.0)
    return pd.Series([inv[a:b + 1].mean() for a, b in spans], index=t1.index)

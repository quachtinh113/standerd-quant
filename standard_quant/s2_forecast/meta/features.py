"""Đặc trưng cho meta-model — mọi cột tại t0 chỉ dùng thông tin ≤ t0 (và nhãn ĐÃ phân giải: t1 < t0)."""
from __future__ import annotations

import numpy as np
import pandas as pd

FEATURES = ["abs_f", "vol_ratio", "mom5", "mom20", "rank60", "trail_hit"]


def build_features(close: pd.Series, vol: pd.Series, forecast: pd.Series,
                   labels: pd.DataFrame | None = None, trail_n: int = 20) -> pd.DataFrame:
    """Một dòng mỗi phiên có forecast ≠ 0. `labels` (t1,label) chỉ để tính trail_hit nhân quả."""
    side = np.sign(forecast)
    v = vol.replace(0.0, np.nan)
    f = pd.DataFrame(index=close.index)
    f["abs_f"] = forecast.abs() / 20.0
    f["vol_ratio"] = v / v.rolling(120, min_periods=60).mean()
    f["mom5"] = side * (close - close.shift(5)) / (v * np.sqrt(5))
    f["mom20"] = side * (close - close.shift(20)) / (v * np.sqrt(20))
    f["rank60"] = side * (close.rolling(60, min_periods=60).apply(lambda w: (w[:-1] < w[-1]).mean(), raw=True) - 0.5)
    f["trail_hit"] = _trailing_hit(close.index, labels, trail_n)
    f["side"] = side
    return f[side != 0]


def _trailing_hit(index: pd.DatetimeIndex, labels: pd.DataFrame | None, n: int) -> pd.Series:
    """Tỉ lệ thắng của n nhãn gần nhất ĐÃ KẾT THÚC trước t (t1 < t). Chưa có → 0.5 (trung tính)."""
    if labels is None or labels.empty:
        return pd.Series(0.5, index=index)
    done = labels.reset_index(names="t0").sort_values(["t1", "t0"], kind="stable")   # ổn định khi trùng t1
    t1 = done["t1"].to_numpy()
    cum = np.concatenate([[0.0], np.cumsum(done["label"].to_numpy(float))])
    k = np.searchsorted(t1, index.to_numpy(), side="left")       # số nhãn có t1 < t
    lo = np.maximum(k - n, 0)
    cnt = k - lo
    return pd.Series(np.where(cnt > 0, (cum[k] - cum[lo]) / np.maximum(cnt, 1), 0.5), index=index)

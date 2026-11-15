"""AFML ch.7 — Purged K-Fold với embargo. Chống rò rỉ do nhãn chồng lấn thời gian."""
from __future__ import annotations

import numpy as np
import pandas as pd


class PurgedKFold:
    """Chia theo khối thời gian liên tục. Loại khỏi tập train:
      • mọi nhãn có khoảng [t0, t1] chồng lên khoảng của tập test (purge);
      • nhãn bắt đầu trong `embargo` phiên ngay sau test (embargo).
    """

    def __init__(self, n_splits: int = 5, embargo_bars: int = 5):
        if n_splits < 2:
            raise ValueError("n_splits >= 2")
        self.n_splits, self.embargo = n_splits, embargo_bars

    def split(self, t0: pd.DatetimeIndex, t1: pd.Series, calendar: pd.DatetimeIndex):
        t0 = pd.DatetimeIndex(t0)
        a0, a1 = t0.to_numpy(), pd.DatetimeIndex(t1).to_numpy()
        order = np.argsort(a0, kind="stable")
        for fold in np.array_split(order, self.n_splits):
            if len(fold) == 0:
                continue
            ts, te = a0[fold].min(), a1[fold].max()
            pos = calendar.searchsorted(pd.Timestamp(te))
            emb_end = calendar[min(pos + self.embargo, len(calendar) - 1)].to_datetime64()
            overlap = (a0 <= te) & (a1 >= ts)
            embargoed = (a0 > te) & (a0 <= emb_end)
            yield np.flatnonzero(~(overlap | embargoed)), np.sort(fold)

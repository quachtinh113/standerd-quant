"""AFML ch.14–15 & 11–12: kiểm định backtest. PSR/DSR (Sharpe bị thổi phồng bởi nhiều lần thử) và PBO (CSCV)."""
from __future__ import annotations

from itertools import combinations
from statistics import NormalDist

import numpy as np
import pandas as pd

_N = NormalDist()
_GAMMA = 0.5772156649


def probabilistic_sharpe(returns: pd.Series, sr_benchmark: float = 0.0) -> float:
    """P(Sharpe thật > benchmark), hiệu chỉnh lệch/nhọn. Sharpe & benchmark theo chu kỳ của `returns` (không năm hoá)."""
    r = returns.dropna()
    n = len(r)
    if n < 10 or r.std() == 0:
        return float("nan")
    sr = r.mean() / r.std()
    sk, ku = float(r.skew()), float(r.kurt() + 3.0)           # kurt thô, chuẩn = 3
    denom = np.sqrt(max(1.0 - sk * sr + (ku - 1.0) / 4.0 * sr ** 2, 1e-12))
    return float(_N.cdf((sr - sr_benchmark) * np.sqrt(n - 1) / denom))


def deflated_sharpe(returns: pd.Series, n_trials: int, trial_sr_var: float | None = None) -> float:
    """DSR = PSR tại ngưỡng SR0 = kỳ vọng Sharpe lớn nhất trong n_trials lần thử vô giá trị.

    trial_sr_var: phương sai Sharpe (theo chu kỳ) giữa các lần thử; mặc định 1/n (nhiễu thuần).
    DSR < 0.95 → Sharpe này có thể chỉ là kết quả của việc thử nhiều.
    """
    n = returns.dropna().shape[0]
    if n_trials <= 1:
        return probabilistic_sharpe(returns, 0.0)
    v = trial_sr_var if trial_sr_var is not None else 1.0 / max(n, 1)
    sr0 = np.sqrt(v) * ((1 - _GAMMA) * _N.inv_cdf(1 - 1 / n_trials) + _GAMMA * _N.inv_cdf(1 - 1 / (n_trials * np.e)))
    return probabilistic_sharpe(returns, float(sr0))


def pbo_cscv(trials: pd.DataFrame, n_slices: int = 8) -> dict:
    """Probability of Backtest Overfitting (CSCV). `trials`: T×N, mỗi cột là chuỗi lợi suất một cấu hình.

    Với mọi cách chia nửa-nửa các lát thời gian: chọn cấu hình tốt nhất IS, xem thứ hạng OOS của nó.
    PBO = tỉ lệ lần cấu hình thắng IS lại dưới trung vị OOS. > 0.5 = quy trình chọn tham số đang khớp nhiễu.
    """
    if trials.shape[1] < 2:
        raise ValueError("cần >= 2 cấu hình")
    m = trials.dropna().to_numpy()
    n_slices -= n_slices % 2
    slices = np.array_split(np.arange(len(m)), n_slices)
    logits = []
    for is_ids in combinations(range(n_slices), n_slices // 2):
        oos_ids = [k for k in range(n_slices) if k not in is_ids]
        is_r = m[np.concatenate([slices[k] for k in is_ids])]
        oos_r = m[np.concatenate([slices[k] for k in oos_ids])]
        sr = lambda x: x.mean(0) / np.where(x.std(0) == 0, np.nan, x.std(0))
        best = int(np.nanargmax(sr(is_r)))
        rank = (pd.Series(sr(oos_r)).rank().iloc[best]) / (m.shape[1] + 1)
        logits.append(np.log(rank / (1 - rank)))
    logits = np.array(logits)
    return {"pbo": float((logits <= 0).mean()), "n_combinations": len(logits), "logit_mean": float(logits.mean())}

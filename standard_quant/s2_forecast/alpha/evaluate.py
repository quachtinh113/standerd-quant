"""Đánh giá alpha theo tinh thần Finding Alphas: Sharpe, turnover, fitness, IC + kiểm soát overfit.

Kiểm soát overfit (mọi mục đều có số liệu trả về, không chỉ cảnh báo):
  • tách In-Sample / Out-of-Sample theo thời gian (không xáo trộn);
  • ngưỡng Sharpe cho "số lần đã thử" (expected_max_sharpe, Bailey–López de Prado);
  • robustness: bỏ từng instrument, trễ thêm 1 phiên, chi phí ×2.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from statistics import NormalDist

import numpy as np
import pandas as pd

from . import operators as op

ANN = 256


@dataclass
class AlphaReport:
    sharpe: float
    ann_return: float
    turnover: float          # Σ|Δpos| trung bình / ngày, sổ vị thế Σ|pos| = 1
    fitness: float           # Sharpe × √(|ret| / max(turnover, 0.125))  (công thức WorldQuant)
    ic: float                # IC xếp hạng trung bình giữa alpha_t và lợi suất t+1
    sharpe_is: float
    sharpe_oos: float
    hurdle: float            # Sharpe tối thiểu cần có sau n_trials lần thử
    passes: bool
    notes: list[str] = field(default_factory=list)


def _positions(alpha: pd.DataFrame, neutralize: bool) -> pd.DataFrame:
    a = op.cs_demean(alpha) if neutralize else alpha
    return op.cs_scale(a)


def _pnl(pos: pd.DataFrame, ret: pd.DataFrame, delay: int, cost_bps: float) -> tuple[pd.Series, pd.Series]:
    held = pos.shift(delay)                         # quyết định t → nắm giữ từ t+delay
    gross = (held * ret).sum(axis=1, min_count=1)
    turn = held.diff().abs().sum(axis=1, min_count=1)
    return (gross - turn * cost_bps * 1e-4).dropna(), turn.dropna()


def _sharpe(r: pd.Series) -> float:
    return float(r.mean() / r.std() * np.sqrt(ANN)) if len(r) > 2 and r.std() > 0 else 0.0


def expected_max_sharpe(n_trials: int, years: float) -> float:
    """Sharpe năm hoá kỳ vọng của alpha TỐT NHẤT trong n_trials alpha vô giá trị (Bailey–LdP).

    Alpha chỉ đáng tin nếu vượt ngưỡng này; càng thử nhiều càng cao.
    """
    if n_trials <= 1 or years <= 0:
        return 0.0
    g, nd = 0.5772156649, NormalDist()
    z = (1 - g) * nd.inv_cdf(1 - 1 / n_trials) + g * nd.inv_cdf(1 - 1 / (n_trials * np.e))
    return float(z / np.sqrt(years))


def _ic(alpha: pd.DataFrame, ret: pd.DataFrame) -> float:
    fwd = ret.shift(-1)                              # chỉ để ĐO (nhãn), không đưa vào tín hiệu
    a, f = alpha.rank(axis=1), fwd.rank(axis=1)
    ics = a.corrwith(f, axis=1).dropna()
    return float(ics.mean()) if len(ics) else 0.0


def evaluate_alpha(alpha: pd.DataFrame, returns: pd.DataFrame, *, neutralize: bool = False,
                   delay: int = 1, cost_bps: float = 1.0, oos_fraction: float = 0.3,
                   n_trials: int = 1) -> AlphaReport:
    if delay < 1:
        raise ValueError("delay >= 1")
    pos = _positions(alpha, neutralize)
    r, turn = _pnl(pos, returns, delay, cost_bps)
    cut = int(len(r) * (1 - oos_fraction))
    s_is, s_oos = _sharpe(r.iloc[:cut]), _sharpe(r.iloc[cut:])
    years = len(r) / ANN
    t = float(turn.mean()) if len(turn) else float("nan")
    ann_ret = float(r.mean() * ANN)
    sharpe = _sharpe(r)
    fitness = sharpe * np.sqrt(abs(ann_ret) / max(t, 0.125)) if np.isfinite(t) else 0.0
    hurdle = expected_max_sharpe(n_trials, years)
    notes = []
    if s_oos < 0.5 * s_is:
        notes.append("OOS < 50% IS: nghi overfit")
    if sharpe_is_low := (s_is < hurdle):
        notes.append(f"IS Sharpe {s_is:.2f} < ngưỡng {hurdle:.2f} cho {n_trials} lần thử")
    if t > 0.7:
        notes.append("turnover cao: chi phí có thể nuốt alpha")
    passes = (s_oos > 0) and (s_oos >= 0.5 * s_is) and not sharpe_is_low and s_is > 0
    return AlphaReport(sharpe, ann_ret, t, float(fitness), _ic(alpha, returns), s_is, s_oos, hurdle, passes, notes)


def robustness_report(alpha: pd.DataFrame, returns: pd.DataFrame, *, neutralize: bool = False,
                      cost_bps: float = 1.0) -> dict[str, float]:
    """Sharpe dưới các nhiễu loạn. Alpha thật không sụp đổ khi bị nhiễu nhẹ."""
    out = {"base": _sharpe(_pnl(_positions(alpha, neutralize), returns, 1, cost_bps)[0]),
           "delay2": _sharpe(_pnl(_positions(alpha, neutralize), returns, 2, cost_bps)[0]),
           "cost_x2": _sharpe(_pnl(_positions(alpha, neutralize), returns, 1, 2 * cost_bps)[0])}
    if alpha.shape[1] > 2:
        for c in alpha.columns:
            keep = [k for k in alpha.columns if k != c]
            out[f"drop_{c}"] = _sharpe(_pnl(_positions(alpha[keep], neutralize), returns[keep], 1, cost_bps)[0])
    return out

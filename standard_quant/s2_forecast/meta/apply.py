"""Áp meta-model lên forecast L2 + đánh giá lợi ích (uplift) một cách trung thực."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .model import bet_size


def apply_meta(forecast: pd.DataFrame, prob: pd.DataFrame, threshold: float = 0.5, mode: str = "gate",
               burn_in: str = "pass") -> pd.DataFrame:
    """forecast × bet_size(prob). Chưa có xác suất (burn-in): 'pass' giữ nguyên, 'flat' về 0 (thận trọng hơn)."""
    size = bet_size(prob, threshold, mode).reindex_like(forecast)
    size = size.fillna(1.0 if burn_in == "pass" else 0.0)
    return forecast * size


def _proxy_sharpe(forecast: pd.DataFrame, close: pd.DataFrame, vol: pd.DataFrame) -> float:
    pos = forecast / 10.0                                          # đơn vị rủi ro chuẩn hoá
    r = (close.diff() / vol.shift(1)) * pos.shift(1)               # quyết định t → lời lỗ t+1
    d = r.mean(axis=1).dropna()
    return float(d.mean() / d.std() * np.sqrt(256)) if len(d) > 2 and d.std() > 0 else 0.0


def evaluate_meta(forecast: pd.DataFrame, close: pd.DataFrame, vol: pd.DataFrame, prob: pd.DataFrame,
                  events: pd.DataFrame, threshold: float = 0.5, mode: str = "gate") -> dict:
    """So sánh có/không meta trên cùng dữ liệu; chỉ tính nơi có xác suất OOS (sau burn-in)."""
    ev = events.copy()
    ev["p"] = [prob.at[t, s] if t in prob.index else np.nan for t, s in zip(ev["t0"], ev["sym"])]
    ev = ev.dropna(subset=["p"])
    out: dict = {"n_scored_events": int(len(ev))}
    if len(ev) < 30 or ev["label"].nunique() < 2:
        return out | {"auc_oos": float("nan")}
    keep = ev["p"] >= threshold
    out |= {
        "auc_oos": float(roc_auc_score(ev["label"], ev["p"])),
        "hit_rate_all": float(ev["label"].mean()),
        "hit_rate_kept": float(ev.loc[keep, "label"].mean()) if keep.any() else float("nan"),
        "kept_fraction": float(keep.mean()),
        "avg_ret_all": float(ev["ret"].mean()),
        "avg_ret_kept": float(ev.loc[keep, "ret"].mean()) if keep.any() else float("nan"),
    }
    start = prob.dropna(how="all").index.min()                     # chỉ so sánh trên đoạn đã có meta
    fm = apply_meta(forecast, prob, threshold, mode, burn_in="pass")
    sl = slice(start, None)
    out["sharpe_raw"] = _proxy_sharpe(forecast.loc[sl], close.loc[sl], vol.loc[sl])
    out["sharpe_meta"] = _proxy_sharpe(fm.loc[sl], close.loc[sl], vol.loc[sl])
    return out

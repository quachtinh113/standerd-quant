"""Toán tử kiểu Finding Alphas (Tulchinsky). Dữ liệu là DataFrame rộng: index=thời gian, cột=instrument.

Quy ước bất biến: giá trị tại t chỉ phụ thuộc dữ liệu ≤ t (không shift âm, không center=True).
Tên và ngữ nghĩa bám theo FASTEXPR/WorldQuant: ts_* chạy dọc thời gian, cs_* chạy ngang instrument.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

Frame = pd.DataFrame


# ---------------- Time-series ----------------
def ts_delay(x: Frame, d: int) -> Frame:
    if d < 0:
        raise ValueError("ts_delay: d phải >= 0 (d < 0 là nhìn trước tương lai)")
    return x.shift(d)


def ts_delta(x: Frame, d: int) -> Frame:
    return x - ts_delay(x, d)


def ts_sum(x: Frame, d: int) -> Frame:
    return x.rolling(d, min_periods=d).sum()


def ts_mean(x: Frame, d: int) -> Frame:
    return x.rolling(d, min_periods=d).mean()


def ts_std(x: Frame, d: int) -> Frame:
    return x.rolling(d, min_periods=d).std()


def ts_min(x: Frame, d: int) -> Frame:
    return x.rolling(d, min_periods=d).min()


def ts_max(x: Frame, d: int) -> Frame:
    return x.rolling(d, min_periods=d).max()


def ts_zscore(x: Frame, d: int) -> Frame:
    return (x - ts_mean(x, d)) / ts_std(x, d).replace(0.0, np.nan)


def ts_rank(x: Frame, d: int) -> Frame:
    """Phân vị (0..1) của giá trị hiện tại trong cửa sổ d."""
    return x.rolling(d, min_periods=d).apply(lambda w: (w[:-1] < w[-1]).mean() + 0.5 * (w[:-1] == w[-1]).mean(), raw=True)


def ts_corr(x: Frame, y: Frame, d: int) -> Frame:
    return x.rolling(d, min_periods=d).corr(y)


def ts_cov(x: Frame, y: Frame, d: int) -> Frame:
    return x.rolling(d, min_periods=d).cov(y)


def ts_decay_linear(x: Frame, d: int) -> Frame:
    """Trung bình có trọng số tuyến tính, mới nhất nặng nhất (giảm turnover)."""
    w = np.arange(1, d + 1, dtype=float)
    w /= w.sum()
    return x.rolling(d, min_periods=d).apply(lambda a: float(a @ w), raw=True)


def ts_ema(x: Frame, d: int) -> Frame:
    return x.ewm(span=d, min_periods=d).mean()


# ---------------- Cross-sectional ----------------
def cs_rank(x: Frame) -> Frame:
    """Phân vị ngang, căn giữa về [-0.5, 0.5]."""
    return x.rank(axis=1, pct=True) - 0.5 - (0.5 / x.shape[1])


def cs_demean(x: Frame) -> Frame:
    """Trung hoà thị trường (market-neutral): trừ trung bình ngang."""
    return x.sub(x.mean(axis=1), axis=0)


def cs_zscore(x: Frame) -> Frame:
    sd = x.std(axis=1).replace(0.0, np.nan)
    return x.sub(x.mean(axis=1), axis=0).div(sd, axis=0)


def cs_scale(x: Frame, target: float = 1.0) -> Frame:
    """Chuẩn hoá Σ|x| = target (độ lớn danh mục cố định)."""
    return x.div(x.abs().sum(axis=1).replace(0.0, np.nan), axis=0) * target


def cs_winsorize(x: Frame, k: float = 3.0) -> Frame:
    """Kẹp ở ±k độ lệch chuẩn ngang (chống outlier)."""
    mu, sd = x.mean(axis=1), x.std(axis=1)
    return x.clip(lower=mu - k * sd, upper=mu + k * sd, axis=0)


# ---------------- Tiện ích ----------------
def sign(x: Frame) -> Frame:
    return np.sign(x)


def signed_power(x: Frame, p: float) -> Frame:
    return np.sign(x) * x.abs() ** p


def safe_div(a: Frame, b: Frame) -> Frame:
    return a / b.replace(0.0, np.nan)

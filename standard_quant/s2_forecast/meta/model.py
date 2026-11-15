"""Meta-model (AFML ch.3, 6, 10) + walk-forward nhân quả."""
from __future__ import annotations

from statistics import NormalDist

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

from .cv import PurgedKFold
from .features import FEATURES, build_features
from .labeling import average_uniqueness, triple_barrier

_cdf = np.vectorize(NormalDist().cdf)


def make_model(seed: int = 0) -> RandomForestClassifier:
    """Nông và chính quy hoá mạnh: dữ liệu tài chính nhiễu, mục tiêu là lọc chứ không là khớp."""
    return RandomForestClassifier(n_estimators=150, max_depth=3, min_samples_leaf=40, max_features=3,
                                  class_weight="balanced_subsample", n_jobs=1, random_state=seed)


def bet_size(p, threshold: float = 0.5, mode: str = "gate"):
    """Xác suất thắng → hệ số [0,1] nhân vào forecast. Meta-label KHÔNG đảo chiều lệnh.

    gate: 1 nếu p ≥ threshold, ngược lại 0.
    continuous: m = 2Φ((p−½)/√(p(1−p)))−1 (AFML 10.2), 0 nếu p < threshold.
    NaN (chưa đủ dữ liệu huấn luyện) giữ NaN — nơi gọi quyết định pass-through.
    """
    arr = np.asarray(p, dtype=float)
    if mode == "gate":
        out = (arr >= threshold).astype(float)
    elif mode == "continuous":
        pc = np.clip(arr, 1e-6, 1 - 1e-6)
        z = (pc - 0.5) / np.sqrt(pc * (1 - pc))
        out = np.where(arr >= threshold, np.clip(2 * _cdf(z) - 1, 0, 1), 0.0)
    else:
        raise ValueError(f"mode không hỗ trợ: {mode}")
    out = np.where(np.isfinite(arr), out, np.nan)
    if isinstance(p, pd.DataFrame):
        return pd.DataFrame(out, index=p.index, columns=p.columns)
    return pd.Series(out, index=p.index) if isinstance(p, pd.Series) else out


def purged_cv_auc(events: pd.DataFrame, calendar: pd.DatetimeIndex, n_splits: int = 5, embargo: int = 5) -> float:
    """AUC out-of-fold bằng Purged K-Fold trên bảng `build_events`: ước lượng trung thực khả năng
    nhận diện 'khi nào tín hiệu sai'. Giá trị ≈ 0.5 nghĩa là meta-model không có thông tin."""
    ev = events.sort_values("t0").reset_index(drop=True)
    X, y, w = ev[FEATURES].fillna(0.5), ev["label"], ev["w"]
    aucs = []
    for tr, te in PurgedKFold(n_splits, embargo).split(pd.DatetimeIndex(ev["t0"]), ev["t1"], calendar):
        if len(tr) == 0 or y.iloc[tr].nunique() < 2 or y.iloc[te].nunique() < 2:
            continue
        m = make_model().fit(X.iloc[tr], y.iloc[tr], sample_weight=w.iloc[tr])
        aucs.append(roc_auc_score(y.iloc[te], m.predict_proba(X.iloc[te])[:, 1]))
    return float(np.mean(aucs)) if aucs else float("nan")


def build_events(close: pd.DataFrame, vol: pd.DataFrame, forecast: pd.DataFrame, *, pt_mult: float = 2.0,
                 sl_mult: float = 2.0, horizon: int = 10, min_abs_forecast: float = 1.0) -> pd.DataFrame:
    """Bảng sự kiện đã gán nhãn, gộp mọi instrument: cột sym, t0, đặc trưng, t1, ret, label, w (uniqueness)."""
    cal, parts = close.index, []
    for s in forecast.columns:
        f = forecast[s].where(forecast[s].abs() >= min_abs_forecast, 0.0).fillna(0.0)
        lab = triple_barrier(close[s], np.sign(f), vol[s], pt_mult=pt_mult, sl_mult=sl_mult, horizon=horizon)
        d = build_features(close[s], vol[s], f, lab).join(lab[["t1", "ret", "label"]], how="inner")
        d["w"] = average_uniqueness(d["t1"], cal).to_numpy()
        d.index.name = "t0"
        parts.append(d.reset_index().assign(sym=s))
    return pd.concat(parts, ignore_index=True).sort_values(["t0", "sym"]).reset_index(drop=True)


def walk_forward_probability(
    close: pd.DataFrame, vol: pd.DataFrame, forecast: pd.DataFrame, *,
    pt_mult: float = 2.0, sl_mult: float = 2.0, horizon: int = 10, min_events: int = 400,
    refit_every: int = 63, min_abs_forecast: float = 1.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """P(thắng) cho mỗi (t, instrument), huấn luyện chung các instrument.

    Nhân quả: tại mốc refit T chỉ dùng nhãn có t1 < T (đã phân giải). Trước khi đủ `min_events` → NaN.
    Dự đoán dùng đặc trưng tính tại t0 (không cần nhãn). Trả (prob cùng hình dạng forecast, events).
    """
    cal = close.index
    events = build_events(close, vol, forecast, pt_mult=pt_mult, sl_mult=sl_mult, horizon=horizon,
                          min_abs_forecast=min_abs_forecast)
    # đặc trưng cho dự đoán: cần cả các phiên cuối mẫu chưa có nhãn
    pred = []
    for s in forecast.columns:
        f = forecast[s].where(forecast[s].abs() >= min_abs_forecast, 0.0).fillna(0.0)
        lab = events[events["sym"] == s].set_index("t0")[["t1", "label"]]
        d = build_features(close[s], vol[s], f, lab)
        d.index.name = "t0"
        pred.append(d.reset_index().assign(sym=s))
    pred = pd.concat(pred, ignore_index=True)

    prob = pd.DataFrame(np.nan, index=forecast.index, columns=forecast.columns)
    for T in cal[refit_every::refit_every]:
        train = events[events["t1"] < T]
        if len(train) < min_events or train["label"].nunique() < 2:
            continue
        model = make_model().fit(train[FEATURES].fillna(0.5), train["label"], sample_weight=train["w"])
        end = cal[min(cal.get_loc(T) + refit_every, len(cal)) - 1]
        blk = pred[(pred["t0"] >= T) & (pred["t0"] <= end)]
        if blk.empty:
            continue
        p = model.predict_proba(blk[FEATURES].fillna(0.5))[:, 1]
        for t0, s, pi in zip(blk["t0"], blk["sym"], p):
            prob.at[t0, s] = pi
    return prob, events

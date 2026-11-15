"""Báo cáo L3: meta-labeling có thực sự giúp không? python run_meta_report.py [--csv SYM=path ...]

Đưa ra: AUC purged-CV, AUC walk-forward OOS, uplift hit-rate/Sharpe, DSR và PBO trên lưới ngưỡng đã thử.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from standard_quant import Config, StandardQuantPipeline
from standard_quant.s1_data_rules import load_csv_close, synthetic_prices
from standard_quant.s2_forecast.meta import (apply_meta, deflated_sharpe, evaluate_meta, pbo_cscv,
                                             purged_cv_auc, walk_forward_probability)

THRESHOLDS = [0.45, 0.50, 0.55, 0.60]            # số cấu hình đã thử = n_trials cho DSR


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", nargs="*", default=[])
    ap.add_argument("--config", default=str(Path(__file__).parent / "config" / "standard.json"))
    a = ap.parse_args()
    if a.csv:
        px = {kv.split("=", 1)[0]: load_csv_close(kv.split("=", 1)[1]) for kv in a.csv}
    else:
        px = synthetic_prices({"XAUUSD": 1900.0, "EURUSD": 1.10, "BTCUSD": 30000.0}, n_days=2500)
        print("DỮ LIỆU TỔNG HỢP — chỉ minh hoạ định dạng báo cáo\n")
    cfg = Config.load(a.config)
    pipe = StandardQuantPipeline(cfg, px)
    prob, ev = walk_forward_probability(pipe.prices, pipe.vols, pipe.forecasts_raw, min_events=cfg.meta_min_events)

    auc_cv = purged_cv_auc(ev, pipe.prices.index)
    print(f"Sự kiện: {len(ev)} | hit-rate cơ sở: {ev['label'].mean():.3f}")
    print(f"AUC Purged-CV (5 fold, embargo 5): {auc_cv:.3f}   (0.5 = không có thông tin)\n")

    rets = {}
    print(f"{'thr':>5} {'AUC_oos':>8} {'hit_all':>8} {'hit_kept':>9} {'kept%':>6} {'SR_raw':>7} {'SR_meta':>8}")
    for thr in THRESHOLDS:
        r = evaluate_meta(pipe.forecasts_raw, pipe.prices, pipe.vols, prob, ev, thr, "gate")
        if "sharpe_meta" not in r:
            print(f"{thr:5.2f}  không đủ sự kiện OOS"); continue
        print(f"{thr:5.2f} {r['auc_oos']:8.3f} {r['hit_rate_all']:8.3f} {r['hit_rate_kept']:9.3f} "
              f"{100 * r['kept_fraction']:6.1f} {r['sharpe_raw']:7.2f} {r['sharpe_meta']:8.2f}")
        start = prob.dropna(how="all").index.min()
        fm = apply_meta(pipe.forecasts_raw, prob, thr, "gate").loc[start:]
        d = (pipe.prices.diff() / pipe.vols.shift(1)).loc[start:] * (fm / 10.0).shift(1)
        rets[thr] = d.mean(axis=1).dropna()
    if len(rets) >= 2:
        m = pd.DataFrame(rets)
        best = m.mean() / m.std()
        b = best.idxmax()
        print(f"\nCấu hình tốt nhất: thr={b}. DSR (n_trials={len(rets)}): {deflated_sharpe(m[b], len(rets)):.3f}  (cần > 0.95)")
        print(f"PBO (CSCV, {len(rets)} cấu hình): {pbo_cscv(m, 8)['pbo']:.2f}  (cần < 0.5; chỉ có ý nghĩa với nhiều cấu hình)")


if __name__ == "__main__":
    main()

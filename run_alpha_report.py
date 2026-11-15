"""Báo cáo đánh giá toàn bộ alpha trong registry: python run_alpha_report.py [--csv SYM=path ...]"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from standard_quant.s1_data_rules import blended_daily_vol, load_csv_close, synthetic_prices
from standard_quant.s2_forecast.alpha import REGISTRY, evaluate_alpha, robustness_report


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", nargs="*", default=[])
    a = ap.parse_args()
    if a.csv:
        px = {kv.split("=", 1)[0]: load_csv_close(kv.split("=", 1)[1]) for kv in a.csv}
    else:
        px = synthetic_prices({"XAUUSD": 1900.0, "EURUSD": 1.10, "BTCUSD": 30000.0}, n_days=2000)
        print("DỮ LIỆU TỔNG HỢP — chỉ minh hoạ định dạng báo cáo\n")
    close = pd.DataFrame(px).sort_index().ffill(limit=3)
    d = {"close": close, "returns": close.pct_change(), "vol": pd.DataFrame({c: blended_daily_vol(close[c]) for c in close})}
    n = len(REGISTRY)                                   # n_trials = số alpha đã thử (khai báo trung thực)
    print(f"{'alpha':18s} {'SR':>6} {'IS':>6} {'OOS':>6} {'turn':>6} {'fit':>6} {'IC':>7} {'hurdle':>7}  pass")
    for name, fn in REGISTRY.items():
        al = fn(d)
        r = evaluate_alpha(al, d["returns"], n_trials=n)
        print(f"{name:18s} {r.sharpe:6.2f} {r.sharpe_is:6.2f} {r.sharpe_oos:6.2f} {r.turnover:6.2f} "
              f"{r.fitness:6.2f} {r.ic:7.3f} {r.hurdle:7.2f}  {r.passes}")
        for note in r.notes:
            print(f"    - {note}")
        rob = robustness_report(al, d["returns"])
        print("    robust:", {k: round(v, 2) for k, v in rob.items()})


if __name__ == "__main__":
    main()

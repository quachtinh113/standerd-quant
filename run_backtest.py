"""Demo: python run_backtest.py [--config config/standard.json] [--csv SYMBOL=path.csv ...]"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from standard_quant import Config, run_backtest
from standard_quant.s1_data_rules import load_csv_close, synthetic_prices


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(Path(__file__).parent / "config" / "standard.json"))
    ap.add_argument("--csv", nargs="*", default=[], help="SYMBOL=đường_dẫn.csv (cột time, close)")
    a = ap.parse_args()
    cfg = Config.load(a.config)
    if a.csv:
        prices = {kv.split("=", 1)[0]: load_csv_close(kv.split("=", 1)[1]) for kv in a.csv}
        src = "CSV"
    else:
        prices = synthetic_prices({"XAUUSD": 1900.0, "EURUSD": 1.10, "BTCUSD": 30000.0})
        src = "SYNTHETIC (chỉ để minh hoạ, không phải bằng chứng alpha)"
    r = run_backtest(cfg, prices)
    print(f"Dữ liệu: {src}")
    for k in ("final_equity", "ann_return", "ann_vol", "sharpe", "max_drawdown", "turnover_lots", "cost_paid", "tier_counts"):
        print(f"  {k:14s}: {r[k]}")


if __name__ == "__main__":
    main()
